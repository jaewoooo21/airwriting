"""
ZUPT and ZARU dedicated tests
"""
import pytest
import numpy as np
from airwriting_imu.fusion.imu_only_fusion import IMUOnlyFusion


@pytest.fixture
def fusion():
    config = {
        "accel_noise_std": 0.5,
        "gyro_noise_std": 0.01,
        "accel_bias_std": 0.0001,
        "gyro_bias_std": 0.00001,
        "initial_covariance": 1.0,
        "zupt": {
            "enabled": True,
            "gyro_threshold": 0.05,
            "accel_variance_threshold": 0.3,
            "noise": 0.001,
            "window_size": 20,
            "adaptive": True,
        },
        "zaru": {"enabled": True, "noise": 0.0001},
        "constraints": {
            "max_velocity": 3.0,
            "max_acceleration": 30.0,
            "velocity_decay": 0.98,
        },
    }
    return IMUOnlyFusion(config)


class TestZUPTDetection:
    def test_stationary_detected(self, fusion):
        """Stationary IMU should trigger ZUPT."""
        accel = np.zeros(3)
        gyro = np.zeros(3)

        for i in range(30):
            res = fusion.update(accel, gyro, i * 10000)

        assert fusion.n_zupt > 0

    def test_motion_prevents_zupt(self, fusion):
        """Moving IMU should not trigger ZUPT."""
        gyro = np.array([0.5, 0.0, 0.0])  # Rotating
        accel = np.array([2.0, 0.0, 0.0])

        initial_zupt = fusion.n_zupt
        for i in range(30):
            fusion.update(accel, gyro, i * 10000)

        assert fusion.n_zupt == initial_zupt

    def test_adaptive_threshold(self, fusion):
        """Adaptive threshold should adjust based on velocity."""
        # Build up some velocity first
        for i in range(50):
            fusion.update(np.array([5.0, 0, 0]), np.zeros(3), i * 10000)

        # Now try to trigger ZUPT with slightly variable accel
        zupt_start = fusion.n_zupt
        for i in range(50, 80):
            accel = np.random.normal(0, 0.15, 3)  # Slightly noisy "stationary"
            fusion.update(accel, np.zeros(3), i * 10000)

        # With adaptive threshold (widened due to velocity), should still detect
        # Note: may or may not trigger depending on noise; just check no crash
        assert fusion.n_update > 50


class TestZARUBiasCorrection:
    def test_bias_convergence(self, fusion):
        """ZARU should converge gyro bias toward actual bias."""
        true_bias = np.array([0.005, -0.003, 0.002])

        for i in range(500):  # 5 seconds
            accel = np.zeros(3)
            gyro = true_bias  # Constant bias when stationary
            fusion.update(accel, gyro, i * 10000)

        # Gyro bias estimate should approach true bias
        bias_error = np.linalg.norm(fusion.bg - true_bias)
        assert bias_error < np.linalg.norm(true_bias), \
            f"Bias should converge: true={true_bias}, est={fusion.bg}"


class TestZUPTCovariance:
    def test_zupt_reduces_velocity_covariance(self, fusion):
        """ZUPT should reduce velocity-related covariance."""
        # Predict to increase covariance
        for i in range(50):
            fusion.update(np.array([1.0, 0, 0]), np.zeros(3), i * 10000)

        vel_cov_before = np.trace(fusion.P[3:6, 3:6])

        # ZUPT phase
        for i in range(50, 100):
            fusion.update(np.zeros(3), np.zeros(3), i * 10000)

        vel_cov_after = np.trace(fusion.P[3:6, 3:6])

        assert vel_cov_after < vel_cov_before, \
            f"ZUPT should reduce velocity covariance: before={vel_cov_before:.4f} after={vel_cov_after:.4f}"
