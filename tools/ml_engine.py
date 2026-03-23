import csv
import json
import os
import threading
import time
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

# Feature constants
NUM_POINTS = 20
MIN_CLASSES = 2
MIN_TOTAL_SAMPLES = 4
MIN_SAMPLES_PER_CLASS = 2
RECOMMENDED_SAMPLES_PER_CLASS = 15

class AirWritingCNN(nn.Module):
    """
    1D-CNN for 3D gesture recognition.
    Input: (batch, 3, 20)
    """
    def __init__(self, num_classes):
        super(AirWritingCNN, self).__init__()
        self.conv1 = nn.Conv1d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)  # Output: (batch, 32, 10)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.mean(x, dim=2)
        x = self.fc(x)
        return x

def resample_stroke(stroke_data, num_points=NUM_POINTS):
    n = len(stroke_data)
    if n == 0: return np.zeros((num_points, 3), dtype=np.float32)
    if n == 1: return np.repeat(stroke_data.astype(np.float32), num_points, axis=0)
    orig_idx = np.linspace(0.0, 1.0, n)
    target_idx = np.linspace(0.0, 1.0, num_points)
    resampled = np.zeros((num_points, stroke_data.shape[1]), dtype=np.float32)
    for i in range(stroke_data.shape[1]):
        resampled[:, i] = np.interp(target_idx, orig_idx, stroke_data[:, i])
    return resampled

def normalize_stroke(stroke_data):
    stroke_data = np.asarray(stroke_data, dtype=np.float32)
    if len(stroke_data) == 0: return stroke_data
    min_vals = np.min(stroke_data, axis=0)
    max_vals = np.max(stroke_data, axis=0)
    center = (max_vals + min_vals) / 2.0
    scale = np.max(max_vals - min_vals)
    if scale <= 1e-8: scale = 1.0
    return (stroke_data - center) / scale

def compute_direction_features(resampled_stroke):
    deltas = np.diff(resampled_stroke, axis=0)
    norms = np.linalg.norm(deltas, axis=1, keepdims=True)
    norms[norms <= 1e-8] = 1.0
    return (deltas / norms).astype(np.float32)

class MLEngine:
    def __init__(self):
        root = os.path.join(os.path.dirname(__file__), "..")
        self.data_dir = os.path.join(root, "data")
        self.dataset_file = os.path.join(self.data_dir, "airwriting_dataset.csv")
        self.model_file = os.path.join(self.data_dir, "cnn_model.pth")
        self.labels_file = os.path.join(self.data_dir, "labels.json")

        os.makedirs(self.data_dir, exist_ok=True)
        self.model = None
        self.classes = []
        self.training_lock = threading.Lock()
        self.is_training = False
        self._stroke_counter = int(time.time())
        
        self.stats = {
            "status": "READY",
            "model_loaded": False,
            "predict_ready": False,
            "model_type": "cnn",
            "model_classes": [],
            "model_feature_dim": 60,
            "accuracy": 0.0,
            "total_samples": 0,
            "class_counts": {},
            "class_count": 0,
            "min_samples_per_class": 0,
            "is_trainable": False,
            "trainability_reason": "No dataset yet.",
            "requirements": {
                "min_classes": MIN_CLASSES,
                "min_total_samples": MIN_TOTAL_SAMPLES,
                "min_samples_per_class": MIN_SAMPLES_PER_CLASS,
                "recommended_samples_per_class": RECOMMENDED_SAMPLES_PER_CLASS,
            },
            "last_trained": "Never",
            "last_error": "",
        }
        self._load_model()
        self._refresh_dataset_stats()

    def _load_model(self):
        if os.path.exists(self.model_file) and os.path.exists(self.labels_file):
            try:
                with open(self.labels_file, "r") as f:
                    self.classes = json.load(f)
                self.model = AirWritingCNN(len(self.classes))
                self.model.load_state_dict(torch.load(self.model_file, map_location=torch.device('cpu')))
                self.model.eval()
                self.stats["model_loaded"] = True
                self.stats["predict_ready"] = True
                self.stats["model_classes"] = self.classes
            except Exception as e:
                print(f"[MLEngine] CNN load failed: {e}")
                self.model = None

    def _update_trainability(self, class_counts):
        total_samples = int(sum(class_counts.values()))
        class_count = len(class_counts)
        min_samples = int(min(class_counts.values())) if class_counts else 0
        reasons = []
        if class_count < MIN_CLASSES:
            reasons.append(f"Need at least {MIN_CLASSES} labels.")
        if total_samples < MIN_TOTAL_SAMPLES:
            reasons.append(f"Need at least {MIN_TOTAL_SAMPLES} strokes total.")
        if class_counts and min_samples < MIN_SAMPLES_PER_CLASS:
            reasons.append(f"Each label needs at least {MIN_SAMPLES_PER_CLASS} strokes.")
        if not class_counts:
            reasons = ["No recorded strokes yet."]
        self.stats["class_count"] = class_count
        self.stats["min_samples_per_class"] = min_samples
        self.stats["is_trainable"] = len(reasons) == 0
        self.stats["trainability_reason"] = "Ready to train." if self.stats["is_trainable"] else " ".join(reasons)
        self.stats["status"] = "TRAINING" if self.is_training else ("READY" if self.stats["is_trainable"] else "NOT_READY")

    def _refresh_dataset_stats(self):
        if not os.path.exists(self.dataset_file):
            self._update_trainability({})
            return
        try:
            df = pd.read_csv(self.dataset_file)
            if df.empty or "label" not in df.columns:
                self._update_trainability({})
                return
            grouped = df.groupby(["session_id", "stroke_idx"])["label"].first()
            counts = Counter(grouped.values.tolist())
            self.stats["class_counts"] = dict(sorted(counts.items()))
            self.stats["total_samples"] = int(sum(counts.values()))
            self._update_trainability(self.stats["class_counts"])
        except Exception as exc:
            self.stats["last_error"] = str(exc)
            self._update_trainability({})

    def _stroke_to_positions(self, stroke):
        arr = np.asarray(stroke, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 5 or arr.shape[1] < 3: return None
        return arr[:, :3]

    def _extract_feature_vector(self, stroke):
        coords = self._stroke_to_positions(stroke)
        if coords is None: return None
        norm_coords = normalize_stroke(coords)
        return resample_stroke(norm_coords, NUM_POINTS).flatten().astype(np.float32)

    def _load_training_data(self):
        if not os.path.exists(self.dataset_file): return None, None
        df = pd.read_csv(self.dataset_file)
        if df.empty: return None, None
        features = []; labels = []
        grouped = df.groupby(["session_id", "stroke_idx"], sort=False)
        for (_, _), group in grouped:
            label = str(group["label"].iloc[0]).strip().upper()
            coords = group[["fk_x", "fk_y", "fk_z"]].values.astype(np.float32)
            feat = self._extract_feature_vector(coords)
            if feat is not None:
                features.append(feat)
                labels.append(label)
        if not features: return None, None
        return np.vstack(features), np.array(labels)

    def save_stroke(self, label, stroke_full):
        coords = self._stroke_to_positions(stroke_full)
        if coords is None: return False
        self._stroke_counter += 1
        session_id = int(time.time())
        stroke_idx = self._stroke_counter
        write_header = not os.path.exists(self.dataset_file)
        with open(self.dataset_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["session_id", "label", "stroke_idx", "frame_idx", "timestamp", "fk_x", "fk_y", "fk_z", "q_w", "q_x", "q_y", "q_z"])
            for frame_idx, frame in enumerate(np.asarray(stroke_full, dtype=np.float32)):
                x, y, z = float(frame[0]), float(frame[1]), float(frame[2])
                qw, qx, qy, qz = (map(float, frame[3:7]) if frame.shape[0] >= 7 else (1.0, 0.0, 0.0, 0.0))
                writer.writerow([session_id, str(label).strip().upper(), stroke_idx, frame_idx, time.time(), x, y, z, qw, qx, qy, qz])
        self._refresh_dataset_stats()
        return True

    def train_background(self):
        if self.is_training or not self.stats["is_trainable"] or not self.training_lock.acquire(blocking=False): return False
        self.is_training = True
        self.stats["status"] = "TRAINING"
        threading.Thread(target=self._run_training_routine, daemon=True).start()
        return True

    def _run_training_routine(self):
        try:
            X, y = self._load_training_data()
            if X is None or y is None or len(np.unique(y)) < MIN_CLASSES: return
            unique_labels = sorted(list(np.unique(y)))
            label_to_idx = {l: i for i, l in enumerate(unique_labels)}
            y_idx = np.array([label_to_idx[l] for l in y])
            self.classes = unique_labels
            with open(self.labels_file, "w") as f: json.dump(self.classes, f)
            X_tensor = torch.from_numpy(X.reshape(-1, 3, 20)).float()
            y_tensor = torch.from_numpy(y_idx).long()
            model = AirWritingCNN(len(unique_labels))
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            model.train()
            for epoch in range(100):
                optimizer.zero_grad()
                loss = criterion(model(X_tensor), y_tensor)
                loss.backward(); optimizer.step()
            torch.save(model.state_dict(), self.model_file)
            self.model = model; self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_tensor)
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == y_tensor).sum().item() / len(y_tensor)
            self.stats.update({"accuracy": float(accuracy), "last_trained": time.strftime("%Y-%m-%d %H:%M:%S"), "model_loaded": True, "predict_ready": True, "model_classes": self.classes})
        except Exception as exc: self.stats["last_error"] = str(exc)
        finally:
            self.is_training = False
            self._refresh_dataset_stats()
            self.training_lock.release()

    def predict(self, raw_stroke, top_n=None):
        coords = self._stroke_to_positions(raw_stroke)
        if coords is None: return [] if top_n is not None else ("TOO_SHORT", 0.0)
        norm_coords = normalize_stroke(coords)
        resampled = resample_stroke(norm_coords, NUM_POINTS)
        if self.model is not None:
            try:
                input_tensor = torch.from_numpy(resampled.T).float().unsqueeze(0)
                with torch.no_grad():
                    probs = torch.softmax(self.model(input_tensor), dim=1)[0]
                order = torch.argsort(probs, descending=True)
                predictions = [{"label": self.classes[i.item()], "confidence": float(probs[i.item()])} for i in order]
                if top_n is None: return predictions[0]["label"], round(predictions[0]["confidence"], 3)
                return predictions[:max(1, int(top_n))]
            except Exception as e: print(f"[MLEngine] CNN Predict Error: {e}")
        return [] if top_n is not None else ("NO_MODEL", 0.0)

    def get_stats(self):
        self._refresh_dataset_stats()
        return dict(self.stats)

    def get_status(self): return self.get_stats()
    def add_training_data(self, label, raw_stroke): return self.save_stroke(label, raw_stroke)
