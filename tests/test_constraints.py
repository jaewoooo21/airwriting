"""
Unit tests for Biomechanical Constraints
"""
import pytest
import numpy as np
from airwriting_imu.constraints.biomechanical import BiomechanicalConstraints


@pytest.fixture
def constraints():
    return BiomechanicalConstraints({
        "max_velocity": 3.0,
        "max_acceleration": 30.0,
        "max_reach": 0.51,
        "velocity_decay": 0.98,
    })


class TestVelocityClamping:
    def test_normal_velocity_unchanged(self, constraints):
        pos = np.array([0.1, 0.1, 0.0])
        vel = np.array([1.0, 0.5, 0.0])
        accel = np.zeros(3)
        vel_before = vel.copy()

        constraints.apply(pos, vel, accel)

        np.testing.assert_allclose(vel, vel_before)

    def test_excessive_velocity_clamped(self, constraints):
        pos = np.array([0.0, 0.0, 0.0])
        vel = np.array([5.0, 0.0, 0.0])  # > max 3.0
        accel = np.zeros(3)

        result = constraints.apply(pos, vel, accel)

        assert result["vel_clamped"]
        assert np.linalg.norm(vel) <= 3.0 + 1e-6


class TestAccelerationClamping:
    def test_normal_accel_unchanged(self, constraints):
        pos = np.zeros(3)
        vel = np.zeros(3)
        accel = np.array([5.0, 0.0, 0.0])
        accel_before = accel.copy()

        constraints.apply(pos, vel, accel)

        np.testing.assert_allclose(accel, accel_before)

    def test_excessive_accel_clamped(self, constraints):
        pos = np.zeros(3)
        vel = np.zeros(3)
        accel = np.array([50.0, 0.0, 0.0])  # > max 30.0

        result = constraints.apply(pos, vel, accel)

        assert result["acc_clamped"]
        assert np.linalg.norm(accel) <= 30.0 + 1e-6


class TestWorkspaceBoundary:
    def test_within_workspace(self, constraints):
        pos = np.array([0.2, 0.1, 0.0])
        vel = np.zeros(3)
        accel = np.zeros(3)
        origin = np.zeros(3)

        result = constraints.apply(pos, vel, accel, origin=origin)

        assert not result["workspace_clamped"]

    def test_outside_workspace_clamped(self, constraints):
        pos = np.array([1.0, 0.0, 0.0])  # > max_reach 0.51
        vel = np.zeros(3)
        accel = np.zeros(3)
        origin = np.zeros(3)

        result = constraints.apply(pos, vel, accel, origin=origin)

        assert result["workspace_clamped"]
        assert np.linalg.norm(pos - origin) <= 0.51 + 1e-6


class TestVelocityDecay:
    def test_decay_reduces_velocity(self, constraints):
        vel = np.array([1.0, 0.5, 0.0])
        vel_before = vel.copy()

        constraints.apply_velocity_decay(vel, accel_magnitude=0.1)

        assert np.linalg.norm(vel) < np.linalg.norm(vel_before)

    def test_no_decay_during_acceleration(self, constraints):
        vel = np.array([1.0, 0.5, 0.0])
        vel_before = vel.copy()

        constraints.apply_velocity_decay(vel, accel_magnitude=5.0)

        np.testing.assert_allclose(vel, vel_before)

    def test_decay_converges_to_zero(self, constraints):
        vel = np.array([1.0, 0.0, 0.0])

        for _ in range(500):
            constraints.apply_velocity_decay(vel, accel_magnitude=0.0)

        assert np.linalg.norm(vel) < 0.001


class TestReset:
    def test_reset_clears_stats(self, constraints):
        pos = np.zeros(3)
        vel = np.array([5.0, 0.0, 0.0])
        accel = np.array([50.0, 0.0, 0.0])

        constraints.apply(pos, vel, accel)
        constraints.reset()

        stats = constraints.get_stats()
        assert all(v == 0 for v in stats.values())
