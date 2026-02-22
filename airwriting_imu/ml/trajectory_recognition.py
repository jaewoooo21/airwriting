"""
Trajectory-to-Image Encoder + CNN Recognition (v2.3)
=====================================================
Converts estimated 2D pen-tip trajectories into images and applies
a CNN for character recognition.

Inspired by ImAiR (IIT Delhi, 2024) which encodes IMU time-series
as images for airwriting recognition.

Pipeline:
  1. Trajectory buffer → 2D projection (use WritingPlane normal)
  2. Render to grayscale image (256×256)
  3. CNN inference → character class

Usage:
    encoder = TrajectoryEncoder(img_size=256)
    encoder.add_point(pos_2d)
    img = encoder.render()
    # Feed img to CNN for recognition
"""
import numpy as np
import logging
from typing import Optional

log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class TrajectoryEncoder:
    """Convert 2D trajectory to a grayscale image for CNN recognition."""

    def __init__(self, img_size: int = 64, line_width: int = 2,
                 margin: float = 0.1, max_points: int = 1000):
        """
        Args:
            img_size: output image resolution (square)
            line_width: line thickness in pixels
            margin: fraction of image to leave as border
            max_points: max trajectory points to buffer
        """
        self._size = img_size
        self._lw = line_width
        self._margin = margin
        self._max_pts = max_points

        self._points = []  # list of 2D points
        self._image = np.zeros((img_size, img_size), dtype=np.float32)

    def add_point(self, point_2d: np.ndarray):
        """Add a 2D point to the trajectory.

        Args:
            point_2d: [x, y] position on writing plane
        """
        self._points.append(point_2d[:2].copy())
        if len(self._points) > self._max_pts:
            self._points.pop(0)

    def project_3d(self, point_3d: np.ndarray,
                   plane_normal: np.ndarray) -> np.ndarray:
        """Project a 3D point onto the writing plane.

        Args:
            point_3d: 3D position [x, y, z]
            plane_normal: normal vector of the writing plane

        Returns:
            2D position on the plane
        """
        # Find two orthonormal vectors in the plane
        n = plane_normal / (np.linalg.norm(plane_normal) + 1e-10)

        # Find a vector not parallel to n
        if abs(n[0]) < 0.9:
            v1 = np.cross(n, np.array([1, 0, 0]))
        else:
            v1 = np.cross(n, np.array([0, 1, 0]))
        v1 /= np.linalg.norm(v1) + 1e-10
        v2 = np.cross(n, v1)
        v2 /= np.linalg.norm(v2) + 1e-10

        # Project
        return np.array([np.dot(point_3d, v1), np.dot(point_3d, v2)])

    def render(self) -> np.ndarray:
        """Render the trajectory as a grayscale image.

        Returns:
            (img_size, img_size) float32 array, values 0-1
        """
        self._image[:] = 0

        if len(self._points) < 2:
            return self._image.copy()

        pts = np.array(self._points)

        # Normalize to image coordinates
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        span = maxs - mins
        max_span = max(span[0], span[1], 1e-6)

        # Scale to fit within margin
        effective_size = self._size * (1 - 2 * self._margin)
        scale = effective_size / max_span
        offset = self._size * self._margin

        # Center the trajectory
        center_offset = (np.array([effective_size, effective_size]) -
                        span * scale) / 2

        # Draw lines between consecutive points
        for i in range(len(pts) - 1):
            p0 = (pts[i] - mins) * scale + offset + center_offset
            p1 = (pts[i + 1] - mins) * scale + offset + center_offset

            # Bresenham-like line drawing
            self._draw_line(
                int(p0[0]), int(p0[1]),
                int(p1[0]), int(p1[1])
            )

        return self._image.copy()

    def _draw_line(self, x0: int, y0: int, x1: int, y1: int):
        """Draw a line on the image with anti-aliasing."""
        n = max(abs(x1 - x0), abs(y1 - y0), 1)
        for t in range(n + 1):
            frac = t / n
            x = int(x0 + frac * (x1 - x0))
            y = int(y0 + frac * (y1 - y0))
            # Draw thick point
            for dx in range(-self._lw, self._lw + 1):
                for dy in range(-self._lw, self._lw + 1):
                    px, py = x + dx, y + dy
                    if 0 <= px < self._size and 0 <= py < self._size:
                        # Intensity falls off with distance
                        dist = (dx**2 + dy**2) ** 0.5
                        intensity = max(0, 1.0 - dist / (self._lw + 0.5))
                        self._image[py, px] = max(
                            self._image[py, px], intensity
                        )

    def clear(self):
        self._points.clear()
        self._image[:] = 0

    def get_point_count(self) -> int:
        return len(self._points)


if _HAS_TORCH:
    class CharacterRecognizer(nn.Module):
        """Lightweight CNN for character recognition from trajectory images.

        Architecture: 3× (Conv2d + BN + ReLU + MaxPool) → FC → softmax
        Input: (1, 64, 64) grayscale image
        Output: (num_classes,) probabilities
        """

        def __init__(self, num_classes: int = 36, img_size: int = 64):
            super().__init__()

            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),  # → 32×32

                nn.Conv2d(16, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),  # → 16×16

                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),  # → 8×8
            )

            feat_size = 64 * (img_size // 8) ** 2

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feat_size, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            """
            Args:
                x: (batch, 1, H, W) grayscale image
            Returns:
                logits: (batch, num_classes)
            """
            return self.classifier(self.features(x))


class TrajectoryRecognizer:
    """High-level wrapper combining encoder + CNN."""

    # Character map: 0-9 + A-Z
    CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def __init__(self, img_size: int = 64, num_classes: int = 36):
        self.encoder = TrajectoryEncoder(img_size=img_size)
        self._img_size = img_size
        self._num_classes = num_classes
        self._model = None

        if _HAS_TORCH:
            self._model = CharacterRecognizer(num_classes, img_size)
            self._model.eval()
            log.info(f"✅ TrajectoryRecognizer: {num_classes} classes, "
                     f"{img_size}×{img_size}")

    def recognize(self) -> Optional[dict]:
        """Recognize the current trajectory.

        Returns:
            dict with "char", "confidence", "image" or None
        """
        img = self.encoder.render()

        if self._model is None:
            return {"char": "?", "confidence": 0.0, "image": img}

        with torch.no_grad():
            x = torch.tensor(
                img[np.newaxis, np.newaxis, :, :],
                dtype=torch.float32
            )
            logits = self._model(x)
            probs = F.softmax(logits, dim=1).squeeze(0).numpy()

        idx = int(np.argmax(probs))
        return {
            "char": self.CLASSES[idx] if idx < len(self.CLASSES) else "?",
            "confidence": float(probs[idx]),
            "image": img,
        }

    def load(self, model_path: str) -> bool:
        """Load pre-trained model weights."""
        if self._model is None:
            return False
        try:
            state = torch.load(model_path, map_location="cpu",
                               weights_only=True)
            self._model.load_state_dict(state)
            self._model.eval()
            log.info(f"✅ Recognizer model loaded from {model_path}")
            return True
        except Exception as e:
            log.warning(f"Recognizer load failed: {e}")
            return False

    def reset(self):
        self.encoder.clear()
