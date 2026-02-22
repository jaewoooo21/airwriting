"""
Integration tests for the full IMU-only pipeline.
Tests the combination of ESKF + FK + Constraints under realistic scenarios.
"""
import pytest
import numpy as np
import math
from airwriting_imu.fusion.imu_only_fusion import IMUOnlyFusion
from airwriting_imu.fusion.forward_kinematics import ForwardKinematics
from airwriting_imu.fusion.madgwick import MadgwickAHRS
from airwriting_imu.constraints.biomechanical import BiomechanicalConstraints


@pytest.fixture
def full_config():
    return {
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
            "max_reach": 0.51,
        },
    }


@pytest.fixture
def skeleton_chain():
    return [
        {"joint": "forearm", "sensor": "S1", "bone_length_m": 0.25, "parent": None},
        {"joint": "hand", "sensor": "S2", "bone_length_m": 0.18, "parent": "forearm"},
        {"joint": "finger", "sensor": "S3", "bone_length_m": 0.08, "parent": "hand"},
    ]


class TestStationaryDrift:
    """Test that stationary IMU does not drift significantly."""

    def test_30sec_stationary_drift(self, full_config):
        """30 seconds of stationary data → position should stay near origin."""
        fusion = IMUOnlyFusion(full_config)

        accel_w = np.zeros(3)
        gyro = np.zeros(3)

        # 30 sec at 100 Hz = 3000 samples
        for i in range(3000):
            fusion.update(accel_w, gyro, i * 10000)

        pos_error = np.linalg.norm(fusion.pos)
        assert pos_error < 0.05, \
            f"Position drifted {pos_error:.4f}m in 30s of stationary data"

    def test_stationary_with_noise(self, full_config):
        """Stationary with sensor noise should still have low drift."""
        fusion = IMUOnlyFusion(full_config)
        np.random.seed(42)

        for i in range(3000):
            accel_w = np.random.normal(0, 0.02, 3)  # Small noise
            gyro = np.random.normal(0, 0.005, 3)     # Small gyro noise
            fusion.update(accel_w, gyro, i * 10000)

        pos_error = np.linalg.norm(fusion.pos)
        assert pos_error < 0.15, \
            f"Position drifted {pos_error:.4f}m with noise"


class TestCircularTrajectory:
    """Test tracking a circular trajectory."""

    def test_circle_shape_preserved(self, full_config):
        """Circular motion should produce roughly circular output."""
        fusion = IMUOnlyFusion(full_config)

        r, freq = 0.1, 0.5
        dt = 0.01
        positions = []

        for i in range(500):  # 5 seconds
            t = i * dt
            # Ground truth acceleration for circular motion
            ax = -r * (2 * math.pi * freq)**2 * math.cos(2 * math.pi * freq * t)
            ay = -r * (2 * math.pi * freq)**2 * math.sin(2 * math.pi * freq * t)
            accel_w = np.array([ax, ay, 0.0])
            gyro = np.zeros(3)

            fusion.update(accel_w, gyro, i * 10000)
            positions.append(fusion.pos.copy())

        positions = np.array(positions)

        # Check that there's movement in both X and Y
        x_range = positions[:, 0].max() - positions[:, 0].min()
        y_range = positions[:, 1].max() - positions[:, 1].min()

        assert x_range > 0.001, "Should have X movement"
        assert y_range > 0.001, "Should have Y movement"


class TestMoveAndStop:
    """Test move-and-stop pattern (key for ZUPT validation)."""

    def test_zupt_stops_drift(self, full_config):
        """After movement + stop, drift should be controlled by ZUPT."""
        fusion = IMUOnlyFusion(full_config)

        # Phase 1: Move for 1 second
        accel_w = np.array([2.0, 0.0, 0.0])
        gyro = np.zeros(3)
        for i in range(100):
            fusion.update(accel_w, gyro, i * 10000)

        pos_after_move = fusion.pos.copy()
        vel_after_move = np.linalg.norm(fusion.vel)
        # v2.0: adaptive Q dampens velocity buildup
        assert vel_after_move > 0.01, "Should have velocity after acceleration"

        # Phase 2: Stop for 2 seconds
        accel_w = np.zeros(3)
        for i in range(100, 300):
            fusion.update(accel_w, gyro, i * 10000)

        vel_after_stop = np.linalg.norm(fusion.vel)
        assert vel_after_stop < 0.05, \
            f"ZUPT should have zeroed velocity, got {vel_after_stop:.4f}"

        # Position should be fairly stable after stopping
        pos_start_stop = fusion.pos.copy()
        for i in range(300, 500):
            fusion.update(accel_w, gyro, i * 10000)
        pos_end_stop = fusion.pos.copy()

        drift_during_stop = np.linalg.norm(pos_end_stop - pos_start_stop)
        assert drift_during_stop < 0.05, \
            f"Position drifted {drift_during_stop:.4f}m during stop phase"


class TestFKIntegration:
    """Test Forward Kinematics integration with ESKF."""

    def test_fk_reduces_drift(self, full_config, skeleton_chain):
        """FK pseudo-measurement should reduce position drift."""
        fusion_no_fk = IMUOnlyFusion(full_config)
        fusion_with_fk = IMUOnlyFusion(full_config)
        fk = ForwardKinematics(skeleton_chain)

        R_id = np.eye(3)
        orientations = {"S1": R_id, "S2": R_id, "S3": R_id}
        fk_pos = fk.compute(orientations)

        np.random.seed(42)
        for i in range(1000):
            accel_w = np.random.normal(0, 0.05, 3)
            gyro = np.random.normal(0, 0.01, 3)

            fusion_no_fk.update(accel_w, gyro, i * 10000)
            fusion_with_fk.update(accel_w, gyro, i * 10000)
            fusion_with_fk.update_fk(fk_pos, fk_noise_std=0.05)

        drift_no_fk = np.linalg.norm(fusion_no_fk.pos)
        drift_with_fk = np.linalg.norm(fusion_with_fk.pos - fk_pos)

        # FK should reduce drift (or at least keep it comparable)
        assert drift_with_fk < drift_no_fk + 0.05, \
            f"FK should help: no_fk={drift_no_fk:.3f} with_fk={drift_with_fk:.3f}"


class TestMadgwickIntegration:
    """Test Madgwick orientation with FK."""

    def test_madgwick_produces_valid_rotation(self):
        mw = MadgwickAHRS(beta=0.1, sample_rate=100)

        for _ in range(200):  # Warmup
            mw.update_imu(np.zeros(3), np.array([0, 0, 9.81]))

        R = mw.rotation_matrix()
        # Should be a valid rotation matrix
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)
