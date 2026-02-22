"""
Stroke-Level Loop Closure Detector (v2.3)
==========================================
Detects when the pen-tip trajectory closes a loop (returns near a
previously visited point within the same stroke) and applies a
position correction via EKF measurement update.

Many characters form closed or semi-closed loops (o, a, d, e, p, b, 0, 8).
By detecting these loops in real-time during writing, we can
correct accumulated drift without waiting for pen-up.

Usage:
  closure = LoopClosureDetector(min_loop_length=15, proximity_m=0.008)
  ...
  closure.track(position)  # feed every frame
  match = closure.detect()
  if match is not None:
      origin = match["origin"]
      # Apply position correction: pos → origin
"""
import numpy as np
import logging

log = logging.getLogger(__name__)


class LoopClosureDetector:
    """Detect trajectory loops within a stroke for drift correction."""

    def __init__(self, min_loop_length: int = 15,
                 proximity_m: float = 0.008,
                 cooldown: int = 20,
                 max_buffer: int = 500):
        """
        Args:
            min_loop_length: minimum samples between start and closure point
                             to avoid detecting small wiggles as loops
            proximity_m: distance threshold (m) to consider a loop closed
            cooldown: frames to wait after a closure before detecting again
            max_buffer: max trajectory buffer size (prevents memory growth)
        """
        self._min_loop = min_loop_length
        self._proximity = proximity_m
        self._cooldown_max = cooldown
        self._max_buf = max_buffer

        self._trajectory = []     # list of position snapshots
        self._cooldown = 0
        self._active = False

        # Statistics
        self.n_closures = 0

    def start_stroke(self):
        """Call when pen goes down — start tracking trajectory."""
        self._trajectory.clear()
        self._cooldown = 0
        self._active = True

    def end_stroke(self):
        """Call when pen goes up — stop tracking."""
        self._active = False
        self._trajectory.clear()

    def track(self, position: np.ndarray):
        """Feed a new position sample during active stroke.

        Args:
            position: current pen-tip position [m]
        """
        if not self._active:
            return

        self._trajectory.append(position.copy())

        # Cap buffer to prevent unbounded growth
        if len(self._trajectory) > self._max_buf:
            # Remove oldest quarter
            self._trajectory = self._trajectory[self._max_buf // 4:]

        # Tick cooldown
        if self._cooldown > 0:
            self._cooldown -= 1

    def detect(self) -> dict | None:
        """Check if current position closes a loop with an earlier point.

        Returns:
            dict with "origin" (the matched earlier position) and
            "loop_length" (number of samples in the loop), or None.
        """
        if not self._active or self._cooldown > 0:
            return None

        n = len(self._trajectory)
        if n < self._min_loop + 1:
            return None

        current = self._trajectory[-1]

        # Search older points (exclude recent min_loop frames)
        # Search in reverse from the oldest eligible point
        search_end = n - self._min_loop
        best_dist = self._proximity
        best_idx = -1

        for i in range(max(0, search_end - 100), search_end):
            dist = np.linalg.norm(current - self._trajectory[i])
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx >= 0:
            origin = self._trajectory[best_idx].copy()
            loop_len = n - 1 - best_idx
            self.n_closures += 1
            self._cooldown = self._cooldown_max

            log.info(f"🔄 Loop closure detected! dist={best_dist:.4f}m "
                     f"loop_len={loop_len} total={self.n_closures}")

            return {
                "origin": origin,
                "loop_length": loop_len,
                "distance": best_dist,
            }

        return None

    def get_stats(self) -> dict:
        return {
            "n_closures": self.n_closures,
            "active": self._active,
            "buffer_size": len(self._trajectory),
        }

    def reset(self):
        self._trajectory.clear()
        self._cooldown = 0
        self._active = False
        self.n_closures = 0
