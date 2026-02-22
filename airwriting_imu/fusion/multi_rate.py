"""
Multi-Rate Processing Scheduler (v2.3)
=======================================
Manages different update rates for different pipeline stages:

  Fast  (200Hz): Gyro integration for attitude tracking
  Normal(100Hz): Full ESKF predict + ZUPT/ZARU
  Slow  (10Hz):  FK correction + Writing Plane PCA recompute

Inspired by IMU pre-integration from Invariant EKF literature.
Separating rates allows high-frequency attitude tracking without
the computational cost of running EKF corrections every frame.

Usage:
    scheduler = MultiRateScheduler(base_rate=100)
    ...
    for each_sample:
        if scheduler.should_fast():    # every 5ms
            integrate_gyro()
        if scheduler.should_normal():  # every 10ms
            eskf_update()
        if scheduler.should_slow():    # every 100ms
            fk_correction()
        scheduler.tick()
"""
import logging

log = logging.getLogger(__name__)


class MultiRateScheduler:
    """Schedule pipeline stages at different rates."""

    def __init__(self, base_rate_hz: int = 100,
                 fast_multiplier: int = 2,
                 slow_divisor: int = 10):
        """
        Args:
            base_rate_hz: the nominal sensor rate (defines "normal")
            fast_multiplier: fast layer runs this × faster than base
            slow_divisor: slow layer runs this × slower than base
        """
        self._base_hz = base_rate_hz
        self._fast_mult = fast_multiplier
        self._slow_div = slow_divisor

        self._tick = 0  # monotonic sample counter
        self._gyro_accum = None  # accumulated gyro for pre-integration

        # Pre-integration buffer (gyro × dt accumulated between normal ticks)
        self._gyro_buffer = []
        self._dt_buffer = []

        log.info(f"✅ MultiRate: fast={base_rate_hz*fast_multiplier}Hz "
                 f"normal={base_rate_hz}Hz slow={base_rate_hz//slow_divisor}Hz")

    def should_fast(self) -> bool:
        """Should run high-rate gyro integration (every tick)."""
        return True  # Fast layer runs every sample

    def should_normal(self) -> bool:
        """Should run normal ESKF update."""
        return self._tick % self._fast_mult == 0

    def should_slow(self) -> bool:
        """Should run slow corrections (FK, PCA)."""
        return self._tick % (self._fast_mult * self._slow_div) == 0

    def tick(self):
        """Advance the scheduler by one sample."""
        self._tick += 1

    def accumulate_gyro(self, gyro, dt: float):
        """Accumulate gyro data for pre-integration between normal updates.

        Args:
            gyro: gyroscope reading [rad/s]
            dt: time step [s]
        """
        self._gyro_buffer.append(gyro.copy())
        self._dt_buffer.append(dt)

    def get_preintegrated_gyro(self):
        """Get pre-integrated rotation since last normal update.

        Returns:
            (mean_gyro, total_dt) or (None, 0) if no data
        """
        if not self._gyro_buffer:
            return None, 0.0

        import numpy as np
        # Weighted mean gyro over accumulated period
        total_dt = sum(self._dt_buffer)
        if total_dt < 1e-10:
            return None, 0.0

        weighted_sum = np.zeros(3, dtype=np.float64)
        for g, d in zip(self._gyro_buffer, self._dt_buffer):
            weighted_sum += g * d

        mean_gyro = weighted_sum / total_dt

        # Clear buffer
        self._gyro_buffer.clear()
        self._dt_buffer.clear()

        return mean_gyro, total_dt

    def get_tick(self) -> int:
        """Current tick count."""
        return self._tick

    def get_rates(self) -> dict:
        return {
            "fast_hz": self._base_hz * self._fast_mult,
            "normal_hz": self._base_hz,
            "slow_hz": self._base_hz // self._slow_div,
        }

    def reset(self):
        self._tick = 0
        self._gyro_buffer.clear()
        self._dt_buffer.clear()
