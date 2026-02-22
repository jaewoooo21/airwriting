"""
Neural ZUPT — LSTM-based Adaptive Zero-Velocity Detector
=========================================================
Replaces rule-based threshold ZUPT with a learned detector.

Architecture:
  Input:  [batch, window_size, 6] (accel_xyz + gyro_xyz)
  → 2-layer BiLSTM (hidden=64)
  → FC(128) → ReLU → Dropout(0.3) → FC(1) → Sigmoid
  Output: probability of zero-velocity state (0..1)

Reference: AZUPT (IEEE) — adaptive ZUPT via neural networks.
Falls back to rule-based if no model is loaded.
"""
import numpy as np
import logging
from pathlib import Path
from collections import deque

log = logging.getLogger(__name__)

# Try importing torch; fallback gracefully
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    log.warning("PyTorch not available — Neural ZUPT will use rule-based fallback")


# ═══════════════════════════════════════════
# Model Definition
# ═══════════════════════════════════════════
if TORCH_AVAILABLE:
    class ZUPTNet(nn.Module):
        """BiLSTM network for zero-velocity detection."""

        def __init__(self, input_dim=6, hidden_dim=64, num_layers=2,
                     dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim, hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            """
            Args:
                x: [batch, seq_len, 6]
            Returns:
                prob: [batch, 1] — probability of stationary
            """
            lstm_out, _ = self.lstm(x)
            # Use last timestep output
            last = lstm_out[:, -1, :]
            return self.classifier(last)


# ═══════════════════════════════════════════
# Neural ZUPT Detector
# ═══════════════════════════════════════════
class NeuralZUPTDetector:
    """
    Adaptive ZUPT detector using LSTM.
    Falls back to rule-based detection if model unavailable.
    """

    def __init__(self, config: dict, model_path: str = None):
        self.window_size = config.get("window_size", 20)
        self.threshold = config.get("neural_threshold", 0.7)

        # Rule-based fallback params
        self.gyro_th = config.get("gyro_threshold", 0.05)
        self.avar_th = config.get("accel_variance_threshold", 0.3)

        # Ring buffer for IMU window
        self._buf = deque(maxlen=self.window_size)

        # Model
        self.model = None
        self.use_neural = False

        if TORCH_AVAILABLE and model_path:
            self._load_model(model_path)
        elif TORCH_AVAILABLE:
            # Check default path
            default_path = Path(__file__).parent.parent.parent / "models" / "zupt_net.pt"
            if default_path.exists():
                self._load_model(str(default_path))

        if not self.use_neural:
            log.info("⚠️ Neural ZUPT: using rule-based fallback")
        else:
            log.info("✅ Neural ZUPT: LSTM model loaded")

        # Stats
        self.n_neural = 0
        self.n_fallback = 0

    def _load_model(self, path: str):
        """Load pre-trained ZUPT model."""
        try:
            self.model = ZUPTNet()
            state = torch.load(path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)
            self.model.eval()
            self.use_neural = True
            log.info(f"✅ Loaded ZUPT model from {path}")
        except Exception as e:
            log.warning(f"Failed to load ZUPT model: {e}")
            self.model = None
            self.use_neural = False

    def detect(self, accel: np.ndarray, gyro: np.ndarray) -> tuple:
        """
        Detect zero-velocity state.

        Args:
            accel: accelerometer [3] (gravity-removed, world frame)
            gyro: gyroscope [3] (body frame)

        Returns:
            (is_stationary: bool, confidence: float)
        """
        sample = np.concatenate([accel, gyro])
        self._buf.append(sample)

        if len(self._buf) < self.window_size:
            return False, 0.0

        if self.use_neural:
            return self._detect_neural()
        else:
            return self._detect_rule(accel, gyro)

    def _detect_neural(self) -> tuple:
        """LSTM-based detection."""
        window = np.array(self._buf, dtype=np.float32)
        x = torch.from_numpy(window).unsqueeze(0)  # [1, W, 6]

        with torch.no_grad():
            prob = self.model(x).item()

        is_stationary = prob > self.threshold
        self.n_neural += 1
        return is_stationary, prob

    def _detect_rule(self, accel: np.ndarray, gyro: np.ndarray) -> tuple:
        """Rule-based fallback (original logic)."""
        # Gyro check
        gyro_mag = np.linalg.norm(gyro)
        if gyro_mag > self.gyro_th:
            self.n_fallback += 1
            return False, 0.0

        # Accel variance check
        window = np.array(self._buf)
        var_sum = np.var(window[:, :3], axis=0).sum()

        is_stationary = var_sum < self.avar_th
        confidence = max(0.0, 1.0 - var_sum / self.avar_th)
        self.n_fallback += 1
        return is_stationary, confidence

    def get_stats(self) -> dict:
        return {
            "mode": "neural" if self.use_neural else "rule-based",
            "n_neural": self.n_neural,
            "n_fallback": self.n_fallback,
        }

    def reset(self):
        self._buf.clear()
        self.n_neural = 0
        self.n_fallback = 0
