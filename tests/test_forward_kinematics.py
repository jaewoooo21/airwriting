"""
Unit tests for Forward Kinematics
"""
import pytest
import numpy as np
from airwriting_imu.fusion.forward_kinematics import ForwardKinematics


@pytest.fixture
def skeleton_chain():
    return [
        {"joint": "forearm", "sensor": "S1", "bone_length_m": 0.25, "parent": None},
        {"joint": "hand", "sensor": "S2", "bone_length_m": 0.18, "parent": "forearm"},
        {"joint": "finger", "sensor": "S3", "bone_length_m": 0.08, "parent": "hand"},
    ]


@pytest.fixture
def fk(skeleton_chain):
    return ForwardKinematics(skeleton_chain)


class TestFKInitialization:
    def test_segment_count(self, fk):
        assert len(fk.segments) == 3

    def test_max_reach(self, fk):
        assert abs(fk.get_max_reach() - 0.51) < 1e-6


class TestFKCompute:
    def test_identity_orientation(self, fk):
        """All identity rotations → pen tip at (0.51, 0, 0)."""
        R_id = np.eye(3)
        orientations = {"S1": R_id, "S2": R_id, "S3": R_id}

        pen_tip = fk.compute(orientations)

        expected = np.array([0.51, 0., 0.])
        np.testing.assert_allclose(pen_tip, expected, atol=1e-6)

    def test_90deg_forearm_rotation(self, fk):
        """S1 rotated 90° around Z → pen tip at (0, 0.51, 0)."""
        R_90z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        R_id = np.eye(3)

        orientations = {"S1": R_90z, "S2": R_90z, "S3": R_90z}
        pen_tip = fk.compute(orientations)

        expected = np.array([0., 0.51, 0.])
        np.testing.assert_allclose(pen_tip, expected, atol=1e-6)

    def test_mixed_rotations(self, fk):
        """Mixed rotations produce non-trivial position."""
        R_id = np.eye(3)
        R_90z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)

        orientations = {"S1": R_id, "S2": R_90z, "S3": R_id}
        pen_tip = fk.compute(orientations)

        # S1: (0.25, 0, 0), S2 rotates 90°: adds (0, 0.18, 0),
        # S3 identity: adds (0.08, 0, 0) (but in S2's frame... actually FK applies
        # each R independently, so S3 at identity means +X by 0.08)
        # pen_tip = (0.25 + 0.08, 0.18, 0) = (0.33, 0.18, 0)
        expected = np.array([0.33, 0.18, 0.])
        np.testing.assert_allclose(pen_tip, expected, atol=1e-6)

    def test_pen_within_reach(self, fk):
        """Pen tip should always be within max reach."""
        for _ in range(100):
            # Random rotation matrices
            orientations = {}
            for sid in ["S1", "S2", "S3"]:
                # Random orthogonal matrix via QR decomposition
                M = np.random.randn(3, 3)
                Q, _ = np.linalg.qr(M)
                if np.linalg.det(Q) < 0:
                    Q[:, 0] *= -1
                orientations[sid] = Q

            pen_tip = fk.compute(orientations)
            dist = np.linalg.norm(pen_tip - fk.origin)
            assert dist <= fk.get_max_reach() + 1e-6

    def test_joint_positions_populated(self, fk):
        """Joint positions should be available after compute."""
        R_id = np.eye(3)
        orientations = {"S1": R_id, "S2": R_id, "S3": R_id}

        fk.compute(orientations)
        joints = fk.get_joint_positions()

        assert "forearm" in joints
        assert "hand" in joints
        assert "finger" in joints

    def test_missing_sensor_fallback(self, fk):
        """Missing orientation should use fallback (straight forward)."""
        R_id = np.eye(3)
        orientations = {"S1": R_id}  # S2, S3 missing

        pen_tip = fk.compute(orientations)

        # Should still produce a valid position
        assert np.all(np.isfinite(pen_tip))
        # And extend straight in X
        expected = np.array([0.51, 0., 0.])
        np.testing.assert_allclose(pen_tip, expected, atol=1e-6)


class TestQuatToRotation:
    def test_identity_quaternion(self):
        q = np.array([1., 0., 0., 0.])
        R = ForwardKinematics.quat_to_rotation_matrix(q)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_90deg_z_rotation(self):
        # Quaternion for 90° rotation around Z: [cos(45°), 0, 0, sin(45°)]
        q = np.array([np.cos(np.pi/4), 0., 0., np.sin(np.pi/4)])
        R = ForwardKinematics.quat_to_rotation_matrix(q)

        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        np.testing.assert_allclose(R, expected, atol=1e-10)
