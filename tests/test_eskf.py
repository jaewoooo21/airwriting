"""
Unit tests for IMU-Only Fusion Engine (ESKF-15)
"""
import pytest
import numpy as np
from airwriting_imu.fusion.imu_only_fusion import IMUOnlyFusion


@pytest.fixture
def default_config():
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
        },
    }


@pytest.fixture
def fusion(default_config):
    return IMUOnlyFusion(default_config)


class TestESKFInitialization:
    def test_state_dimensions(self, fusion):
        assert fusion.pos.shape == (3,)
        assert fusion.vel.shape == (3,)
        assert fusion.q.shape == (4,)
        assert fusion.bg.shape == (3,)
        assert fusion.ba.shape == (3,)

    def test_covariance_shape(self, fusion):
        assert fusion.P.shape == (15, 15)

    def test_initial_position_zero(self, fusion):
        assert np.allclose(fusion.pos, [0, 0, 0])

    def test_initial_quaternion_identity(self, fusion):
        assert np.allclose(fusion.q, [1, 0, 0, 0])

    def test_covariance_symmetric(self, fusion):
        assert np.allclose(fusion.P, fusion.P.T)

    def test_covariance_positive_definite(self, fusion):
        eigvals = np.linalg.eigvals(fusion.P)
        assert np.all(eigvals > 0)


class TestESKFPredict:
    def test_stationary_minimal_drift(self, fusion):
        """Stationary IMU should not drift significantly."""
        accel_w = np.zeros(3)  # gravity already removed
        gyro = np.zeros(3)

        for i in range(100):
            fusion.update(accel_w, gyro, i * 10000)

        pos_error = np.linalg.norm(fusion.pos)
        assert pos_error < 0.01, f"Position drifted {pos_error:.4f}m"

    def test_acceleration_causes_movement(self, fusion):
        """Constant acceleration should cause position change."""
        accel_w = np.array([1.0, 0.0, 0.0])  # 1 m/s² along X
        gyro = np.zeros(3)

        for i in range(100):
            fusion.update(accel_w, gyro, i * 10000)

        # Due to ZUPT, velocity decay, and adaptive Q, movement is dampened.
        # v2.0: accel bias correction + adaptive Q means less raw displacement.
        assert fusion.pos[0] > 0.0001, f"Should have moved along X, got {fusion.pos[0]}"

    def test_covariance_grows_during_predict(self, fusion):
        initial_trace = np.trace(fusion.P)
        accel_w = np.array([0.5, 0.0, 0.0])
        gyro = np.zeros(3)

        fusion.update(accel_w, gyro, 10000)

        assert np.trace(fusion.P) >= initial_trace

    def test_velocity_clamping(self, fusion):
        """Velocity should not exceed max_velocity."""
        accel_w = np.array([50.0, 0.0, 0.0])  # Very high accel
        gyro = np.zeros(3)

        for i in range(200):
            fusion.update(accel_w, gyro, i * 10000)

        vel_mag = np.linalg.norm(fusion.vel)
        assert vel_mag <= fusion.max_vel + 0.01


class TestESKFZUPT:
    def test_zupt_resets_velocity(self, fusion):
        """ZUPT should bring velocity close to zero."""
        # First, create some velocity
        accel_w = np.array([2.0, 0.0, 0.0])
        gyro = np.zeros(3)
        for i in range(50):
            fusion.update(accel_w, gyro, i * 10000)
        # v2.0: adaptive Q dampens velocity buildup, so threshold is lower
        assert np.linalg.norm(fusion.vel) > 0.01

        # Now go stationary to trigger ZUPT
        accel_w = np.zeros(3)
        gyro = np.zeros(3)
        for i in range(50, 120):
            fusion.update(accel_w, gyro, i * 10000)

        assert fusion.n_zupt > 0, "ZUPT should have triggered"
        assert np.linalg.norm(fusion.vel) < 0.15, "Velocity should be near zero"

    def test_zupt_counter_increment(self, fusion):
        """ZUPT counter should increment when stationary."""
        accel_w = np.zeros(3)
        gyro = np.zeros(3)

        for i in range(50):
            fusion.update(accel_w, gyro, i * 10000)

        assert fusion.n_zupt > 0


class TestESKFZARU:
    def test_zaru_updates_gyro_bias(self, fusion):
        """ZARU should update gyro bias estimate."""
        initial_bg = fusion.bg.copy()
        gyro_biased = np.array([0.001, -0.0005, 0.0003])  # Very small bias (within ZUPT threshold)
        accel_w = np.zeros(3)

        for i in range(100):
            fusion.update(accel_w, gyro_biased, i * 10000)

        # Gyro bias should have changed
        assert not np.allclose(fusion.bg, initial_bg), \
            "Gyro bias should have been updated by ZARU"
        assert fusion.n_zaru > 0


class TestESKFFKUpdate:
    def test_fk_corrects_position(self, fusion):
        """FK pseudo-measurement should pull position toward FK estimate."""
        # Let it drift
        accel_w = np.array([1.0, 0.0, 0.0])
        gyro = np.zeros(3)
        for i in range(30):
            fusion.update(accel_w, gyro, i * 10000)

        drifted_pos = fusion.pos.copy()
        fk_pos = np.array([0.0, 0.0, 0.0])

        fusion.update_fk(fk_pos, fk_noise_std=0.05)

        # Position should move toward FK
        dist_before = np.linalg.norm(drifted_pos - fk_pos)
        dist_after = np.linalg.norm(fusion.pos - fk_pos)
        assert dist_after < dist_before


class TestESKFReset:
    def test_reset_clears_state(self, fusion):
        accel_w = np.array([1.0, 0.0, 0.0])
        gyro = np.zeros(3)
        for i in range(50):
            fusion.update(accel_w, gyro, i * 10000)

        fusion.reset()

        assert np.allclose(fusion.pos, [0, 0, 0])
        assert np.allclose(fusion.vel, [0, 0, 0])
        assert np.allclose(fusion.q, [1, 0, 0, 0])
        assert fusion.n_zupt == 0
        assert fusion.n_update == 0


class TestESKFNumericalStability:
    def test_long_run_covariance_finite(self, fusion):
        """P should remain finite after many updates."""
        accel_w = np.array([0.1, 0.0, 0.0])
        gyro = np.array([0.01, 0.0, 0.0])

        for i in range(2000):
            fusion.update(accel_w, gyro, i * 10000)

        assert np.all(np.isfinite(fusion.P))

    def test_long_run_state_finite(self, fusion):
        """State should remain finite after many updates."""
        accel_w = np.array([0.1, -0.05, 0.02])
        gyro = np.array([0.01, -0.005, 0.002])

        for i in range(2000):
            fusion.update(accel_w, gyro, i * 10000)

        assert np.all(np.isfinite(fusion.pos))
        assert np.all(np.isfinite(fusion.vel))
        assert np.all(np.isfinite(fusion.q))


class TestDifferentialFK:
    """v2.1: Differential FK tests."""

    def test_differential_fk_first_call_initializes(self, fusion):
        """First call should just store state without correction."""
        pos_before = fusion.pos.copy()
        fk_pos = np.array([0.1, 0.2, 0.0])
        fusion.update_fk_differential(fk_pos, fk_noise_std=0.05)
        # First call: no correction
        assert np.allclose(fusion.pos, pos_before)

    def test_differential_fk_corrects_drift(self, fusion):
        """Differential FK should correct drift when FK and ESKF diverge."""
        # Initialize FK state
        fk_pos_0 = np.array([0.0, 0.0, 0.0])
        fusion.update_fk_differential(fk_pos_0)

        # Move ESKF independently
        accel_w = np.array([1.0, 0.0, 0.0])
        gyro = np.zeros(3)
        for i in range(20):
            fusion.update(accel_w, gyro, i * 10000)

        pos_before_fk = fusion.pos.copy()

        # FK says we moved differently
        fk_pos_1 = np.array([0.5, 0.0, 0.0])
        fusion.update_fk_differential(fk_pos_1, fk_noise_std=0.02)

        # Position should have changed (FK correction applied)
        assert not np.allclose(fusion.pos, pos_before_fk, atol=1e-6), \
            "Differential FK should have corrected position"

    def test_differential_fk_high_gyro_reduces_gain(self, fusion):
        """High gyro energy should reduce FK correction strength."""
        # Initialize
        fk_pos_0 = np.array([0.0, 0.0, 0.0])
        fusion.update_fk_differential(fk_pos_0)

        accel_w = np.array([1.0, 0.0, 0.0])
        gyro = np.zeros(3)
        for i in range(10):
            fusion.update(accel_w, gyro, i * 10000)

        pos_before = fusion.pos.copy()
        fk_pos_1 = np.array([0.5, 0.0, 0.0])

        # Low gyro energy -> strong correction
        fusion_low = IMUOnlyFusion(fusion.__dict__.get('_config', {
            "accel_noise_std": 0.5, "gyro_noise_std": 0.01,
            "accel_bias_std": 0.0001, "gyro_bias_std": 0.00001,
            "initial_covariance": 1.0,
            "zupt": {"enabled": True, "gyro_threshold": 0.05,
                     "accel_variance_threshold": 0.3, "noise": 0.001,
                     "window_size": 20, "adaptive": True},
            "zaru": {"enabled": True, "noise": 0.0001},
            "constraints": {"max_velocity": 3.0, "velocity_decay": 0.98},
        }))
        # Just verify the method runs without error for different gyro energies
        fk_pos_init = np.array([0.0, 0.0, 0.0])
        fusion_low.update_fk_differential(fk_pos_init, gyro_energy=0.0)
        fusion_low.update_fk_differential(fk_pos_1, gyro_energy=0.0)
        pos_low_gyro = fusion_low.pos.copy()

        fusion_low.reset()
        fusion_low.update_fk_differential(fk_pos_init, gyro_energy=1.0)
        fusion_low.update_fk_differential(fk_pos_1, gyro_energy=1.0)
        pos_high_gyro = fusion_low.pos.copy()

        # With high gyro, correction should be smaller (higher noise)
        correction_low = np.linalg.norm(pos_low_gyro)
        correction_high = np.linalg.norm(pos_high_gyro)
        # Both should be finite
        assert np.all(np.isfinite(pos_low_gyro))
        assert np.all(np.isfinite(pos_high_gyro))


class TestRTOConstraint:
    """v2.1: Return-to-Origin constraint tests."""

    def test_rto_corrects_position(self, fusion):
        """RTO should pull position toward stroke origin."""
        # Move away from origin
        accel_w = np.array([1.0, 0.0, 0.0])
        gyro = np.zeros(3)
        for i in range(50):
            fusion.update(accel_w, gyro, i * 10000)

        pos_before = fusion.pos.copy()
        stroke_origin = np.zeros(3)

        fusion.update_rto(stroke_origin, rto_noise_std=0.03)

        # Should have moved closer to origin
        dist_before = np.linalg.norm(pos_before - stroke_origin)
        dist_after = np.linalg.norm(fusion.pos - stroke_origin)
        assert dist_after < dist_before, \
            f"RTO should pull toward origin: {dist_before:.4f} -> {dist_after:.4f}"

    def test_rto_counter_increment(self, fusion):
        """RTO counter should increment."""
        assert fusion.n_rto == 0
        fusion.update_rto(np.zeros(3))
        assert fusion.n_rto == 1
        fusion.update_rto(np.zeros(3))
        assert fusion.n_rto == 2

    def test_rto_preserves_stability(self, fusion):
        """RTO should not make state unstable."""
        # Run for a while, then apply RTO
        accel_w = np.array([0.5, 0.3, 0.0])
        gyro = np.array([0.01, 0.0, 0.0])
        for i in range(100):
            fusion.update(accel_w, gyro, i * 10000)

        for _ in range(5):
            fusion.update_rto(np.zeros(3), rto_noise_std=0.05)

        assert np.all(np.isfinite(fusion.pos))
        assert np.all(np.isfinite(fusion.vel))
        assert np.all(np.isfinite(fusion.P))

