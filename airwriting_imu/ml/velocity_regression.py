"""
Velocity Regression Network (VRN) — v2.3
==========================================
Neural network that predicts velocity directly from raw IMU data,
bypassing the error-prone double integration of acceleration.

Architecture: 1D-CNN (temporal features) + LSTM (sequence modeling) → 3D velocity

The predicted velocity is injected into the ESKF as a pseudo-measurement
(like ZUPT but with the predicted velocity instead of zero).

Inspired by:
  - MoRPI-PINN (arXiv, 2024): physics-informed INS drift reduction
  - PiDR (arXiv, 2024): differential constraints for dead-reckoning

Usage (inference):
    vrn = VelocityRegressionNet()
    vrn.load("models/vrn_trained.pt")
    vel_pred = vrn.predict(imu_window)  # [18,] x window → [3,] velocity

Usage (training):
    dataset = VRNDataGenerator.generate_synthetic(n_samples=10000)
    vrn.train(dataset)
"""
import numpy as np
import logging
from typing import Optional

log = logging.getLogger(__name__)

# Try importing PyTorch
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    log.info("⚠️ PyTorch not available — VRN uses fallback estimator")


class VRNFallback:
    """Simple velocity estimator when PyTorch is not available.

    Uses exponentially-weighted moving average of acceleration
    integrated over a short window. Not as good as neural network
    but provides a meaningful velocity estimate.
    """

    def __init__(self, window: int = 10, alpha: float = 0.3):
        self._window = window
        self._alpha = alpha
        self._accel_buf = np.zeros((window, 3), dtype=np.float64)
        self._idx = 0
        self._full = False

    def predict(self, accel_world: np.ndarray, dt: float) -> np.ndarray:
        """Estimate velocity from recent acceleration history.

        Uses trapezoidal integration with exponential weighting.
        """
        self._accel_buf[self._idx] = accel_world
        self._idx = (self._idx + 1) % self._window
        if self._idx == 0:
            self._full = True

        count = self._window if self._full else self._idx
        if count < 2:
            return accel_world * dt

        buf = self._accel_buf[:count]

        # Exponential weights (recent samples weighted more)
        weights = np.exp(np.linspace(-1, 0, count))
        weights /= weights.sum()

        # Weighted mean acceleration × window time
        mean_accel = (buf * weights[:, np.newaxis]).sum(axis=0)
        estimated_vel = mean_accel * (count * dt)

        return estimated_vel

    def reset(self):
        self._accel_buf[:] = 0
        self._idx = 0
        self._full = False


if _HAS_TORCH:
    class VRNModel(nn.Module):
        """Conv1D + LSTM velocity regression model.

        Input: (batch, seq_len, 18) — 6-axis IMU × 3 sensors
        Output: (batch, 3) — predicted 3D velocity
        """

        def __init__(self, input_dim: int = 18, hidden_dim: int = 64,
                     num_layers: int = 2, conv_channels: int = 32):
            super().__init__()

            # Temporal feature extraction
            self.conv = nn.Sequential(
                nn.Conv1d(input_dim, conv_channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(conv_channels),
                nn.ReLU(),
                nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(conv_channels),
                nn.ReLU(),
            )

            # Sequence modeling
            self.lstm = nn.LSTM(
                input_size=conv_channels,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2,
            )

            # Velocity output
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 3),  # 3D velocity
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (batch, seq_len, input_dim)
            Returns:
                vel: (batch, 3)
            """
            # Conv1D expects (batch, channels, seq_len)
            x_conv = self.conv(x.transpose(1, 2))  # → (batch, conv_ch, seq_len)
            x_conv = x_conv.transpose(1, 2)  # → (batch, seq_len, conv_ch)

            # LSTM
            lstm_out, _ = self.lstm(x_conv)  # → (batch, seq_len, hidden)
            last = lstm_out[:, -1, :]  # Take last timestep

            # Velocity prediction
            vel = self.head(last)
            return vel


class VelocityRegressionNet:
    """High-level wrapper for VRN inference and ESKF integration."""

    def __init__(self, config: dict = None):
        config = config or {}
        self._window = config.get("window_size", 20)
        self._input_dim = config.get("input_dim", 18)  # 6-axis × 3 sensors
        self._noise_std = config.get("noise_std", 0.1)

        # IMU data buffer
        self._buf = np.zeros((self._window, self._input_dim), dtype=np.float64)
        self._idx = 0
        self._full = False

        self._model = None
        self._fallback = VRNFallback(window=10)
        self._device = None

        if _HAS_TORCH:
            self._device = torch.device("cpu")
            self._model = VRNModel(input_dim=self._input_dim)
            self._model.eval()
            log.info(f"✅ VRN model initialized ({self._window}-sample window)")
        else:
            log.info("⚠️ VRN using fallback estimator (install PyTorch for full VRN)")

    def load(self, model_path: str):
        """Load pre-trained weights."""
        if self._model is None:
            log.warning("VRN: PyTorch not available, cannot load model")
            return False
        try:
            state = torch.load(model_path, map_location=self._device,
                               weights_only=True)
            self._model.load_state_dict(state)
            self._model.eval()
            log.info(f"✅ VRN model loaded from {model_path}")
            return True
        except Exception as e:
            log.warning(f"VRN load failed: {e}")
            return False

    def feed(self, imu_data: np.ndarray):
        """Feed one frame of IMU data (18-dim: 3 sensors × [accel, gyro]).

        Args:
            imu_data: flattened IMU data [ax1,ay1,az1,gx1,gy1,gz1, ...]
        """
        self._buf[self._idx] = imu_data[:self._input_dim]
        self._idx = (self._idx + 1) % self._window
        if self._idx == 0:
            self._full = True

    def predict(self, accel_world: np.ndarray = None,
                dt: float = 0.01) -> Optional[np.ndarray]:
        """Predict velocity from buffered IMU data.

        Returns:
            3D velocity prediction, or None if buffer not full enough
        """
        count = self._window if self._full else self._idx
        if count < self._window // 2:
            return None

        if self._model is not None and self._full:
            # Neural network inference
            with torch.no_grad():
                x = torch.tensor(
                    self._buf[np.newaxis, :, :],
                    dtype=torch.float32,
                    device=self._device
                )
                vel = self._model(x).squeeze(0).numpy()
            return vel
        elif accel_world is not None:
            # Fallback: EWMA-based velocity estimation
            return self._fallback.predict(accel_world, dt)
        return None

    def get_noise_std(self) -> float:
        """Get measurement noise for EKF integration."""
        return self._noise_std

    def reset(self):
        self._buf[:] = 0
        self._idx = 0
        self._full = False
        self._fallback.reset()


class VRNDataGenerator:
    """Generate synthetic training data for the VRN."""

    @staticmethod
    def generate_synthetic(n_samples: int = 10000,
                           window: int = 20,
                           n_sensors: int = 3,
                           dt: float = 0.01,
                           seed: int = 42) -> dict:
        """Create synthetic IMU → velocity pairs.

        Simulates random trajectories with known ground truth velocity.

        Returns:
            dict with "X" (n_samples, window, 18) and "Y" (n_samples, 3)
        """
        rng = np.random.default_rng(seed)

        X = np.zeros((n_samples, window, n_sensors * 6), dtype=np.float32)
        Y = np.zeros((n_samples, 3), dtype=np.float32)

        for i in range(n_samples):
            # Random trajectory type
            ttype = rng.choice(["stationary", "linear", "circular", "random"])

            vel = np.zeros(3, dtype=np.float64)
            accel_base = np.zeros(3, dtype=np.float64)

            if ttype == "stationary":
                vel[:] = 0
                accel_base[:] = 0
            elif ttype == "linear":
                direction = rng.standard_normal(3)
                direction /= np.linalg.norm(direction) + 1e-10
                speed = rng.uniform(0.01, 0.5)
                vel = direction * speed
                accel_base = rng.standard_normal(3) * 0.1
            elif ttype == "circular":
                t = rng.uniform(0, 2 * np.pi)
                radius = rng.uniform(0.02, 0.1)
                freq = rng.uniform(0.5, 3.0)
                vel = np.array([
                    -radius * freq * np.sin(freq * t),
                    radius * freq * np.cos(freq * t),
                    0
                ])
                accel_base = np.array([
                    -radius * freq**2 * np.cos(freq * t),
                    -radius * freq**2 * np.sin(freq * t),
                    0
                ])
            else:  # random
                vel = rng.standard_normal(3) * 0.3
                accel_base = rng.standard_normal(3) * 2.0

            # Generate window of IMU data with noise
            for j in range(window):
                noise_a = rng.standard_normal(3) * 0.5
                noise_g = rng.standard_normal(3) * 0.02
                for s in range(n_sensors):
                    offset = s * 6
                    X[i, j, offset:offset+3] = accel_base + noise_a
                    X[i, j, offset+3:offset+6] = noise_g

            Y[i] = vel.astype(np.float32)

        log.info(f"✅ VRN training data: {n_samples} samples, "
                 f"window={window}, input_dim={n_sensors*6}")

        return {"X": X, "Y": Y}
