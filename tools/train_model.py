import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CSV_FILE = os.path.join(DATA_DIR, 'airwriting_dataset.csv')
MODEL_FILE = os.path.join(DATA_DIR, 'rf_model.pkl')

NUM_POINTS = 20  # Uniformly resample every stroke to 20 points 

def resample_stroke(stroke_data, num_points):
    """
    Resamples a stroke (N points) into exactly `num_points` points
    using linear interpolation, to ensure fixed feature size.
    """
    n = len(stroke_data)
    if n == 0:
        return np.zeros((num_points, 3))
    if n == 1:
        return np.repeat(stroke_data, num_points, axis=0)
        
    # Original indices [0, 1, ..., n-1]
    orig_idx = np.linspace(0, 1, n)
    # Target indices [0, ..., 1]
    target_idx = np.linspace(0, 1, num_points)
    
    resampled = np.zeros((num_points, stroke_data.shape[1]))
    for i in range(stroke_data.shape[1]):
        resampled[:, i] = np.interp(target_idx, orig_idx, stroke_data[:, i])
        
    return resampled

def normalize_stroke(stroke_data):
    """
    Center the stroke at (0,0,0) and scale its max bounding box dimension to 1.
    """
    min_vals = np.min(stroke_data, axis=0)
    max_vals = np.max(stroke_data, axis=0)
    
    center = (max_vals + min_vals) / 2.0
    scale = np.max(max_vals - min_vals)
    if scale == 0:
        scale = 1.0
        
    normalized = (stroke_data - center) / scale
    return normalized

def extract_features(df):
    """
    Group the dataframe by stroke, and extract flattened ML features.
    """
    features = []
    labels = []
    
    # Group by session and stroke_idx to get unique strokes
    grouped = df.groupby(['session_id', 'stroke_idx'])
    
    for (session_id, stroke_idx), group in grouped:
        label = group['label'].iloc[0]
        
        # Extract FK coordinates
        coords = group[['fk_x', 'fk_y', 'fk_z']].values
        
        # Skip strokes that are too short (noise)
        if len(coords) < 5:
            continue
            
        # 1. Normalize
        norm_coords = normalize_stroke(coords)
        
        # 2. Resample to fixed length
        resampled = resample_stroke(norm_coords, NUM_POINTS)
        
        # 3. Flatten (20 * 3 = 60 features)
        feature_vector = resampled.flatten()
        
        features.append(feature_vector)
        labels.append(label)
        
    return np.array(features), np.array(labels)

def main():
    if not os.path.exists(CSV_FILE):
        print(f"❌ Could not find dataset at {CSV_FILE}")
        print("Please run data_collector.py first to gather some data.")
        return

    print("📊 Loading dataset...")
    df = pd.read_csv(CSV_FILE)
    
    print(f"Total rows: {len(df)}")
    
    X, y = extract_features(df)
    print(f"Total extracted strokes: {len(X)}")
    
    if len(X) < 10:
        print("⚠️ Not enough data. Please collect at least a few dozen strokes.")
        return
        
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    print("Class distribution:", np.unique(y, return_counts=True))
    
    # Train Model
    print("\n🧠 Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print(f"🎯 Model Accuracy: {acc*100:.2f}%")
    print("="*50)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save Model
    joblib.dump(clf, MODEL_FILE)
    print(f"\n✅ Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    main()
