"""
Biomechanical Constraints for Drift Mitigation
===============================================
Applies physical constraints based on human arm kinematics to reject
impossible states and reduce position drift.

Constraints:
  1. Velocity clamping: max hand speed ≈ 3 m/s during writing
  2. Acceleration clamping: max ≈ 30 m/s²
  3. Workspace boundary: pen tip cannot exceed skeleton chain reach
  4. Jerk limiting: sudden acceleration changes are suspicious
"""
import numpy as np
import logging

log = logging.getLogger(__name__)


class BiomechanicalConstraints:
    """Apply biomechanical constraints to estimated state."""

    def __init__(self, config: dict):
        self.max_velocity = config.get("max_velocity", 3.0)       # m/s
        self.max_acceleration = config.get("max_acceleration", 30.0)  # m/s²
        self.max_reach = config.get("max_reach", 0.51)            # m
        self.vel_decay = config.get("velocity_decay", 0.98)

        # Jerk limiting
        self.max_jerk = 500.0  # m/s³ (empirical limit for hand writing)
        self._last_accel = np.zeros(3, dtype=np.float64)
        self._last_accel_set = False

        # Statistics
        self.n_vel_clamp = 0
        self.n_acc_clamp = 0
        self.n_workspace_clamp = 0
        self.n_jerk_clamp = 0

        log.info(f"✅ Constraints: max_vel={self.max_velocity}m/s "
                 f"max_acc={self.max_acceleration}m/s² "
                 f"max_reach={self.max_reach}m")

    def constrain_accel(self, accel: np.ndarray, dt: float = 0.01) -> dict:
        """Apply constraints to acceleration (in-place)."""
        result = {"acc_clamped": False, "jerk_clamped": False}

        # ── 1. Acceleration clamping ──
        acc_mag = np.linalg.norm(accel)
        if acc_mag > self.max_acceleration:
            accel *= self.max_acceleration / acc_mag
            self.n_acc_clamp += 1
            result["acc_clamped"] = True

        # ── 2. Jerk limiting ──
        if self._last_accel_set and dt > 0:
            diff = accel - self._last_accel
            jerk = np.linalg.norm(diff) / dt
            if jerk > self.max_jerk:
                # Blend toward last accel
                alpha = self.max_jerk * dt / (np.linalg.norm(diff) + 1e-9)
                alpha = min(alpha, 1.0)
                accel[:] = self._last_accel + alpha * diff
                self.n_jerk_clamp += 1
                result["jerk_clamped"] = True
        self._last_accel[:] = accel
        self._last_accel_set = True

        return result

    def constrain_state(self, position: np.ndarray, velocity: np.ndarray,
                        origin: np.ndarray = None) -> tuple:
        """Check state constraints and return clamped values (no side constraints).
        
        Returns:
            (pos_new, vel_new, flags_dict)
        """
        result = {"vel_clamped": False, "workspace_clamped": False}
        pos_new = position.copy()
        vel_new = velocity.copy()

        # ── 3. Velocity clamping ──
        vel_mag = np.linalg.norm(vel_new)
        if vel_mag > self.max_velocity:
            vel_new *= self.max_velocity / vel_mag
            self.n_vel_clamp += 1
            result["vel_clamped"] = True

        # ── 4. Workspace boundary ──
        if origin is not None:
            diff = pos_new - origin
            displacement = np.linalg.norm(diff)
            if displacement > self.max_reach:
                # Pull back toward workspace boundary
                direction = diff / (displacement + 1e-9)
                pos_new[:] = origin + direction * self.max_reach
                self.n_workspace_clamp += 1
                result["workspace_clamped"] = True

        return pos_new, vel_new, result

    def apply(self, position: np.ndarray, velocity: np.ndarray,
              accel: np.ndarray, origin: np.ndarray = None,
              dt: float = 0.01) -> dict:
        """Legacy wrapper: modifies state in-place (Violates ESKF P-matrix!)."""
        r1 = self.constrain_accel(accel, dt)
        p_new, v_new, r2 = self.constrain_state(position, velocity, origin)
        
        # Modify in-place for legacy support
        position[:] = p_new
        velocity[:] = v_new
        
        return {**r1, **r2}

    def apply_velocity_decay(self, velocity: np.ndarray,
                             accel_magnitude: float):
        """
        Apply exponential velocity decay when acceleration is low.
        This prevents unbounded drift during quiet periods.
        """
        if accel_magnitude < 0.5:
            velocity *= self.vel_decay

    def get_stats(self) -> dict:
        return {
            "vel_clamp": self.n_vel_clamp,
            "acc_clamp": self.n_acc_clamp,
            "workspace_clamp": self.n_workspace_clamp,
            "jerk_clamp": self.n_jerk_clamp,
        }

    def reset(self):
        self.n_vel_clamp = 0
        self.n_acc_clamp = 0
        self.n_workspace_clamp = 0
        self.n_jerk_clamp = 0
        self._last_accel[:] = 0
        self._last_accel_set = False
