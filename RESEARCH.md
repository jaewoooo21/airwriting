# State-of-the-Art in IMU Airwriting (2024-2025)

## 1. Overview
This document summarizes the latest advancements in IMU-based airwriting recognition and drift compensation as of 2024-2025. The focus is on implementing these findings into the `airwriting_imu_only` project to achieve robust, high-accuracy tracking and recognition.

## 2. Key Algorithms & Techniques

### A. Deep Learning for Recognition
Modern approaches have moved beyond simple DTW (Dynamic Time Warping) to complex deep learning architectures:
- **CNN + LSTM/BiLSTM**: The standard for temporal sequence modeling. CNNs extract spatial features from sensor data, while LSTMs handle the temporal dependencies.
- **Image Encoding**: A newer trend involves converting 1D IMU sensor streams into 2D images (e.g., using Gramian Angular Fields or Recurrence Plots) and feeding them into established image classification models like ResNet or DenseNet. This leverages the massive pre-training available for image models.
- **Contrastive Learning**: Techniques like **ECHWR** (Error-enhanced Contrastive Handwriting Recognition) improve feature separability, making the system more robust to writing style variations.

### B. Drift Compensation & Sensor Fusion
Drift remains the primary challenge for IMU-only systems.
- **Neural ZUPT (Zero-Velocity Update)**: Instead of simple threshold-based ZUPT, neural networks are trained to detect stationary phases even during subtle movements, allowing for more frequent and accurate velocity resets.
- **Magnetometer Integration**: In improved environments, magnetometers correct heading drift (yaw).
- **Physics-Informed Deep Learning**: Newer "gray-box" models combine traditional Kalman Filters (White Box) with Neural Networks (Black Box) to estimate residuals or adapt covariance matrices dynamically.

## 3. Recommendations for This Project
1.  **Enhance ZUPT**: Move from purely threshold-based ZUPT to a hybrid model that uses a lightweight ML model (already partially present in `ml/neural_zupt.py`) to gate the ZUPT updates.
2.  **Automated Validation**: All fusion logic changes must be verified against a "Ground Truth" dataset or a strict physics simulation (e.g., "return to start" tests).
3.  **Data Collection**: Collect a diverse dataset of airwriting samples to fine-tune any future ML models.
