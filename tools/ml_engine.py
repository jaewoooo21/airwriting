import os
import time
import json
import threading
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# LSTM Model Definition
class AirWritingLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, num_classes=5):
        super(AirWritingLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out

class MLEngine:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        self.csv_file = os.path.join(self.data_dir, 'airwriting_sequences.csv')
        self.model_file = os.path.join(self.data_dir, 'lstm_model.pth')
        self.labels_file = os.path.join(self.data_dir, 'labels.json')
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        self.max_seq_length = 60 # Pad/truncate all sequences to this length
        self.input_features = 3  # (x, y, z)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[MLEngine] 🔧 PyTorch Engine Initialized using device: {self.device}")
        
        self.model = None
        self.label_map = {} # string to index
        self.idx_map = {}   # index to string
        
        self.training_lock = threading.Lock()
        self.is_training = False
        self.stats = {
            "accuracy": 0.0,
            "sample_counts": {},
            "last_trained": "Never"
        }
        
        self.load_model()

    def _normalize_sequence(self, sequence):
        """Center the sequence to zero mean and scale correctly."""
        sequence = np.array(sequence)
        if len(sequence) == 0:
            return sequence
            
        min_vals = np.min(sequence, axis=0)
        max_vals = np.max(sequence, axis=0)
        center = (max_vals + min_vals) / 2.0
        scale = np.max(max_vals - min_vals)
        if scale == 0:
            scale = 1.0
            
        return (sequence - center) / scale

    def extract_feature(self, stroke_data):
        """
        Takes N frames of [x, y, z] and converts it to a fixed length (max_seq_length, 3).
        Truncates if longer, zeroes if shorter.
        Returns flattened 1D array for DataFrame storage (max_seq_length * 3).
        """
        if len(stroke_data) < 5:
            return None
            
        # 1. Normalize XYZ position
        norm_seq = self._normalize_sequence(stroke_data)
        
        # 2. Pad or Truncate
        n_frames = len(norm_seq)
        fixed_seq = np.zeros((self.max_seq_length, self.input_features), dtype=np.float32)
        
        if n_frames > self.max_seq_length:
            fixed_seq = norm_seq[:self.max_seq_length]
        else:
            fixed_seq[:n_frames] = norm_seq
            
        return fixed_seq.flatten().tolist()

    def encode_labels(self, labels):
        unique_labels = sorted(list(set(labels)))
        self.label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
        self.idx_map = {i: lbl for i, lbl in enumerate(unique_labels)}
        
        # Save mapping
        with open(self.labels_file, 'w') as f:
            json.dump(self.idx_map, f)
            
        return [self.label_map[lbl] for lbl in labels], len(unique_labels)

    def load_model(self):
        """Load the PyTorch LSTM model and label mappings."""
        if os.path.exists(self.model_file) and os.path.exists(self.labels_file):
            try:
                with open(self.labels_file, 'r') as f:
                    str_idx_map = json.load(f)
                    self.idx_map = {int(k): v for k, v in str_idx_map.items()}
                    self.label_map = {v: k for k, v in self.idx_map.items()}
                    
                num_classes = len(self.idx_map)
                
                self.model = AirWritingLSTM(input_size=self.input_features, num_classes=num_classes).to(self.device)
                self.model.load_state_dict(torch.load(self.model_file, map_location=self.device))
                self.model.eval()
                print(f"[MLEngine]  Loaded LSTM Model. Classes: {num_classes}")
                
            except Exception as e:
                print(f"[MLEngine]  Error loading model: {e}")
                self.model = None
        else:
            print("[MLEngine]  No existing LSTM model found. Needs training.")
            self.model = None

    def train_background(self):
        if not os.path.exists(self.csv_file):
            print("[MLEngine]  No data to train on.")
            return False
            
        if not self.training_lock.acquire(blocking=False):
            print("[MLEngine]  Training already in progress, skipping request.")
            return False
            
        self.is_training = True
        thread = threading.Thread(target=self._run_training_routine)
        thread.start()
        return True

    def _run_training_routine(self):
        try:
            print("[MLEngine] 🚀 Starting PyTorch LSTM Training...")
            df = pd.read_csv(self.csv_file)
            
            if len(df) < 5:
                print("[MLEngine] ❌ Not enough data for LSTM training.")
                return

            labels = df.iloc[:, 0].values
            features = df.iloc[:, 1:].values
            
            y_encoded, num_classes = self.encode_labels(labels)
            
            # Update stats
            temp_counts = {}
            for lbl in labels:
                temp_counts[lbl] = temp_counts.get(lbl, 0) + 1
            self.stats["sample_counts"] = temp_counts
            
            # Reshape features to (Batch, Sequence, Features)
            # Flattened size is (max_seq_length * input_features)
            X = features.reshape(-1, self.max_seq_length, self.input_features).astype(np.float32)
            Y = np.array(y_encoded, dtype=np.int64)
            
            X_tensor = torch.tensor(X)
            Y_tensor = torch.tensor(Y)
            
            dataset = TensorDataset(X_tensor, Y_tensor)
            loader = DataLoader(dataset, batch_size=16, shuffle=True)
            
            # Instantiate model
            model = AirWritingLSTM(input_size=self.input_features, num_classes=num_classes).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.005)
            
            model.train()
            num_epochs = 30
            
            for epoch in range(num_epochs):
                total_loss = 0
                correct = 0
                total = 0
                
                for batch_x, batch_y in loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                    
                acc = 100 * correct / (total + 1e-6)
                if (epoch+1) % 10 == 0:
                    print(f"[MLEngine] Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(loader):.4f}, Acc: {acc:.2f}%")
            
            # Evaluate final accuracy on train set (for stats)
            self.stats["accuracy"] = acc
            self.stats["last_trained"] = time.strftime("%H:%M:%S")
            
            # Save model
            torch.save(model.state_dict(), self.model_file)
            self.model = model
            self.model.eval()
            
            print("[MLEngine] ✅ PyTorch LSTM Training Complete.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[MLEngine] ❌ PyTorch Training failed: {e}")
        finally:
            self.is_training = False
            self.training_lock.release()

    def get_status(self):
        return {
            "status": "TRAINING" if self.is_training else "READY",
            "model_loaded": self.model is not None,
            "accuracy": round(self.stats["accuracy"], 2),
            "samples": self.stats["sample_counts"],
            "last_trained": self.stats["last_trained"]
        }

    def predict(self, raw_stroke):
        if self.model is None or len(self.idx_map) == 0:
            return "NO_MODEL", 0.0
            
        feat = self.extract_feature(raw_stroke)
        if feat is None:
            return "TOO_SHORT", 0.0
            
        try:
            # Reconstruct tensor (1, seq_length, features)
            x_arr = np.array(feat, dtype=np.float32).reshape(1, self.max_seq_length, self.input_features)
            x_tensor = torch.tensor(x_arr).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(x_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                max_prob, predicted_class = torch.max(probs, 1)
                
                score = max_prob.item()
                label_idx = predicted_class.item()
                predicted_label = self.idx_map.get(label_idx, "UNKNOWN")
                
                return predicted_label, round(score, 3)
                
        except Exception as e:
            print(f"[MLEngine.predict] Error: {e}")
            return "ERROR", 0.0

    def add_training_data(self, label, raw_stroke):
        feat = self.extract_feature(raw_stroke)
        if feat is None:
            return False
            
        row = [label] + feat
        df_new = pd.DataFrame([row])
        
        if os.path.exists(self.csv_file):
            df_new.to_csv(self.csv_file, mode='a', header=False, index=False)
        else:
            # Generate headers [label, f1, f2, ...]
            headers = ['label'] + [f'f{i}' for i in range(len(feat))]
            df_new.columns = headers
            df_new.to_csv(self.csv_file, index=False)
            
        return True
