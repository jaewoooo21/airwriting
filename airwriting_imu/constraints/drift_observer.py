"""
Drift Observer — ZUPT position variance monitor (v2.2)
=====================================================
Monitors the variance of position during ZUPT-detected stationary periods.
If the pen is supposed to be still but position keeps drifting, this indicates
that the accelerometer bias estimate is stale or incorrect.

When drift is detected, signals that bias re-estimation is needed, allowing
the system to self-heal from temperature-induced bias shifts.

Usage:
  observer = DriftObserver(window=30, threshold=0.001)
  ...
  if observer.observe(position, zupt_active):
      # Bias needs recalibration!
      eskf.ba *= 0.5  # or trigger full recal
"""
import numpy as np
import logging

log = logging.getLogger(__name__)


class DriftObserver:
    """Monitor position drift during stationary periods to detect stale bias."""

    def __init__(self, window: int = 30, drift_threshold: float = 0.001,
                 correction_factor: float = 0.5):
        """
        Args:
            window: number of ZUPT positions to buffer
            drift_threshold: max allowed position variance (m²) during ZUPT
            correction_factor: how much to reduce ba when drift detected (0-1)
        """
        self._window = window
        self._threshold = drift_threshold
        self._correction_factor = correction_factor

        self._zupt_positions = []
        self._variance = np.zeros(3, dtype=np.float64)

        # Statistics
        self.n_drift_detected = 0
        self.n_corrections = 0
        self._last_drift_state = False

    def observe(self, position: np.ndarray, zupt_active: bool) -> bool:
        """Observe position during potential ZUPT, return True if drift detected.

        Args:
            position: current position estimate [m]
            zupt_active: whether ZUPT is currently active

        Returns:
            True if significant drift detected during ZUPT (bias needs refresh)
        """
        if not zupt_active:
            # Clear buffer when leaving ZUPT
            if self._zupt_positions:
                self._zupt_positions.clear()
            self._last_drift_state = False
            return False

        self._zupt_positions.append(position.copy())

        # Keep only recent window
        if len(self._zupt_positions) > self._window:
            self._zupt_positions.pop(0)

        # Need at least half the window to make a judgment
        if len(self._zupt_positions) < self._window // 2:
            return False

        # Compute position variance during ZUPT
        positions = np.array(self._zupt_positions)
        self._variance[:] = np.var(positions, axis=0)
        max_var = np.max(self._variance)

        drift_detected = max_var > self._threshold
        if drift_detected and not self._last_drift_state:
            self.n_drift_detected += 1
            log.warning(
                f"⚠️ Drift detected during ZUPT: var={max_var:.6f}m² "
                f"(threshold={self._threshold:.6f})"
            )
        self._last_drift_state = drift_detected
        return drift_detected

    def apply_correction(self, ba: np.ndarray, vel: np.ndarray):
        """Apply soft correction when drift is detected.

        Reduces accel bias estimate and zeroes velocity.

        Args:
            ba: accelerometer bias vector (modified in-place)
            vel: velocity vector (modified in-place)
        """
        ba *= self._correction_factor
        vel *= 0.0
        self.n_corrections += 1
        log.info(f"🔧 Drift correction applied: ba scaled by {self._correction_factor}")

    def get_stats(self) -> dict:
        return {
            "variance": self._variance.tolist(),
            "n_drift_detected": self.n_drift_detected,
            "n_corrections": self.n_corrections,
        }

    def reset(self):
        self._zupt_positions.clear()
        self._variance[:] = 0
        self.n_drift_detected = 0
        self.n_corrections = 0
        self._last_drift_state = False
