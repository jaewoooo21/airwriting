import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

class MLEngine:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        self.csv_file = os.path.join(self.data_dir, 'airwriting_dataset.csv')
        self.model_file = os.path.join(self.data_dir, 'rf_model.pkl')
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        self.num_points = 20
        self.model = None
        
        self.load_model()

    def load_model(self):
        """Load the model if it exists."""
        if os.path.exists(self.model_file):
            try:
                self.model = joblib.load(self.model_file)
                print("[MLEngine] 🧠 Loaded existing model.")
            except Exception as e:
                print(f"[MLEngine] ⚠️ Error loading model: {e}")
                self.model = None
        else:
            print("[MLEngine] ⚠️ No existing model found. Needs training.")
            self.model = None

    def resample_stroke(self, stroke_data):
        n = len(stroke_data)
        if n == 0:
            return np.zeros((self.num_points, 3))
        if n == 1:
            return np.repeat(stroke_data, self.num_points, axis=0)
            
        orig_idx = np.linspace(0, 1, n)
        target_idx = np.linspace(0, 1, self.num_points)
        
        resampled = np.zeros((self.num_points, stroke_data.shape[1]))
        for i in range(stroke_data.shape[1]):
            resampled[:, i] = np.interp(target_idx, orig_idx, stroke_data[:, i])
        return resampled

    def normalize_stroke(self, stroke_data):
        min_vals = np.min(stroke_data, axis=0)
        max_vals = np.max(stroke_data, axis=0)
        
        center = (max_vals + min_vals) / 2.0
        scale = np.max(max_vals - min_vals)
        if scale == 0:
            scale = 1.0
            
        normalized = (stroke_data - center) / scale
        return normalized

    def extract_feature(self, stroke_data):
        """Converts raw (N, 3) stroke data into a flattened (60,) feature vector."""
        if len(stroke_data) < 5:
            return None
        norm_coords = self.normalize_stroke(stroke_data)
        resampled = self.resample_stroke(norm_coords)
        return resampled.flatten()

    def train_background(self):
        """Reads CSV and trains the RandomForest model. Should be called in a separate thread."""
        if not os.path.exists(self.csv_file):
            print("[MLEngine] ❌ No data to train on.")
            return False
            
        print("[MLEngine] 🧠 Re-training model in background...")
        try:
            df = pd.read_csv(self.csv_file)
            
            features = []
            labels = []
            grouped = df.groupby(['session_id', 'stroke_idx'])
            
            for _, group in grouped:
                label = group['label'].iloc[0]
                coords = group[['fk_x', 'fk_y', 'fk_z']].values
                feat = self.extract_feature(coords)
                if feat is not None:
                    features.append(feat)
                    labels.append(label)
                    
            X = np.array(features)
            y = np.array(labels)
            
            if len(np.unique(y)) < 2 or len(X) < 5:
                print("[MLEngine] ⚠️ Not enough diverse data to train (needs >=2 classes).")
                return False
                
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X, y)
            
            # Atomic save
            joblib.dump(clf, self.model_file)
            self.model = clf
            
            print(f"[MLEngine] ✅ Training complete! Model knows: {np.unique(y)}")
            return True
            
        except Exception as e:
            print(f"[MLEngine] ❌ Training failed: {e}")
            return False

    def predict(self, stroke_data):
        """Predicts the label for a given (N, 3) stroke."""
        if self.model is None:
            return None, 0.0
            
        feat = self.extract_feature(stroke_data)
        if feat is None:
            return None, 0.0
            
        feat = feat.reshape(1, -1)
        
        try:
            probs = self.model.predict_proba(feat)[0]
            max_idx = np.argmax(probs)
            label = self.model.classes_[max_idx]
            confidence = probs[max_idx]
            return label, confidence
        except Exception as e:
            print(f"[MLEngine] Prediction error: {e}")
            return None, 0.0

    def save_stroke(self, label, stroke_data_full):
        """Appends a new stroke (with quaternions) to the CSV."""
        import csv
        import time
        
        write_header = not os.path.exists(self.csv_file)
        session_id = int(time.time())
        stroke_idx = np.random.randint(10000, 99999) # Pseudo-unique
        
        try:
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        'session_id', 'label', 'stroke_idx', 'frame_idx', 'timestamp',
                        'fk_x', 'fk_y', 'fk_z',
                        'q_w', 'q_x', 'q_y', 'q_z'
                    ])
                
                for i, row in enumerate(stroke_data_full):
                    writer.writerow([
                        session_id, label, stroke_idx, i, time.time(),
                        row[0], row[1], row[2], # FK pos
                        row[3], row[4], row[5], row[6]  # Quat
                    ])
            print(f"[MLEngine] 💾 Saved new sample for '{label}' (len={len(stroke_data_full)})")
            return True
        except Exception as e:
            print(f"[MLEngine] ❌ Save error: {e}")
            return False
