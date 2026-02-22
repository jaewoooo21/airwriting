"""
Writing Plane Detector — PCA-based plane constraint (v2.2)
==========================================================
Automatically detects the dominant 2D plane from recent pen-tip positions
and suppresses the orthogonal (depth) component of acceleration.

Rationale:
  Airwriting naturally occurs on an implicit 2D surface. By detecting this
  plane via PCA and removing the normal-axis acceleration, we eliminate
  one axis of drift entirely — the axis where no intentional writing occurs.

Usage:
  plane = WritingPlaneDetector(buffer_size=100)
  ...
  plane.observe(position)       # feed positions
  if plane.is_ready():
      plane.constrain(accel_w)  # remove off-plane acceleration
"""
import numpy as np
import logging

log = logging.getLogger(__name__)


class WritingPlaneDetector:
    """Detect and constrain motion to the dominant writing plane."""

    def __init__(self, buffer_size: int = 100, min_spread: float = 0.005,
                 suppress_ratio: float = 0.8, absolute_lock: bool = False):
        """
        Args:
            buffer_size: number of recent positions for PCA
            min_spread: minimum position spread (m) before plane is reliable
            suppress_ratio: how much to suppress normal-axis accel (0=none, 1=full)
            absolute_lock: if True, hardcodes a vertical chalkboard (Z-axis normal)
                           and completely removes depth drift.
        """
        self._buf_size = buffer_size
        self._min_spread = min_spread
        self._suppress = suppress_ratio
        self._absolute_lock = absolute_lock

        self._buf = np.zeros((buffer_size, 3), dtype=np.float64)
        self._idx = 0
        self._full = False

        # In absolute lock mode, the plane is always the XZ plane (normal is Y)
        # This means we suppress forward/backward depth movement entirely.
        self._normal = np.array([0., 1., 0.], dtype=np.float64)
        self._ready = absolute_lock
        self._eigenvalues = np.zeros(3, dtype=np.float64)

        # Statistics
        self.n_constrain = 0
        self.n_recompute = 0

    def observe(self, position: np.ndarray):
        """Feed a new position into the buffer and periodically re-estimate plane."""
        if self._absolute_lock:
            return  # No PCA needed

        self._buf[self._idx] = position
        self._idx = (self._idx + 1) % self._buf_size
        if self._idx == 0:
            self._full = True

        # Recompute plane every buffer_size / 4 samples
        count = self._buf_size if self._full else self._idx
        if count >= 20 and count % (self._buf_size // 4) == 0:
            self._estimate_plane(count)

    def _estimate_plane(self, count: int):
        """PCA on recent positions to find the writing plane normal."""
        data = self._buf[:count]
        spread = np.max(data, axis=0) - np.min(data, axis=0)
        if np.max(spread) < self._min_spread:
            return  # Not enough movement yet

        # Center the data
        centered = data - np.mean(data, axis=0)

        # Covariance matrix (3×3)
        cov = (centered.T @ centered) / (count - 1)

        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(cov)
        # eigh returns sorted ascending, so eigvecs[:,0] is the smallest eigenvalue

        self._eigenvalues[:] = eigvals
        self._normal[:] = eigvecs[:, 0]  # normal = least-variance direction

        # Plane is reliable if smallest eigenvalue is much smaller than others
        ratio = eigvals[0] / (eigvals[1] + 1e-10)
        self._ready = ratio < 0.3  # Normal axis has < 30% of the next axis's variance

        self.n_recompute += 1
        if self._ready:
            log.debug(f"✏️ Writing plane normal={np.round(self._normal, 3)} "
                      f"ratio={ratio:.3f}")

    def is_ready(self) -> bool:
        """Whether a reliable writing plane has been detected."""
        return self._ready

    def constrain(self, accel_w: np.ndarray):
        """Remove the normal-axis component from world-frame acceleration (in-place).

        Args:
            accel_w: world-frame acceleration [m/s²], modified in-place
        """
        if not self._ready:
            return

        # Project acceleration onto plane normal
        normal_component = np.dot(accel_w, self._normal)

        # Suppress the normal component
        accel_w -= self._suppress * normal_component * self._normal
        self.n_constrain += 1

    def get_normal(self) -> np.ndarray:
        """Get the current plane normal vector."""
        return self._normal.copy()

    def get_stats(self) -> dict:
        return {
            "ready": self._ready,
            "normal": self._normal.tolist(),
            "eigenvalues": self._eigenvalues.tolist(),
            "n_constrain": self.n_constrain,
            "n_recompute": self.n_recompute,
        }

    def reset(self):
        self._buf[:] = 0
        self._idx = 0
        self._full = False
        self._ready = False
        self._normal[:] = [0, 0, 1]
        self.n_constrain = 0
        self.n_recompute = 0
