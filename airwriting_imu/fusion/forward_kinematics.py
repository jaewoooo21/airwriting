"""
Forward Kinematics for 3-IMU Skeleton Chain
============================================
Computes pen-tip position from orientation of each joint segment.

Skeleton chain:
  [Origin] → S1(forearm, 0.25m) → S2(hand, 0.18m) → S3(finger, 0.08m) → [pen tip]

Each segment's orientation is provided by its Madgwick filter quaternion.
The pen-tip position is the sum of rotated bone vectors along the chain.
"""
import numpy as np
import logging

log = logging.getLogger(__name__)


class ForwardKinematics:
    """Compute pen-tip position from joint orientations and bone lengths."""

    def __init__(self, skeleton_chain: list, origin: np.ndarray = None):
        """
        Args:
            skeleton_chain: list of dicts with keys:
                - joint: str (joint name)
                - sensor: str (sensor ID, e.g. "S1")
                - bone_length_m: float
                - parent: str or None
            origin: 3D origin position [m], defaults to [0, 0, 0]
        """
        self.chain = skeleton_chain
        self.origin = np.array(origin or [0., 0., 0.], dtype=np.float64)

        # Build ordered list of (sensor_id, bone_length)
        self.segments = []
        for joint in skeleton_chain:
            self.segments.append({
                "sensor": joint["sensor"],
                "length": joint["bone_length_m"],
                "joint": joint["joint"],
            })

        # Pre-allocate bone direction vector (default: forward = Y-axis)
        # Matches digital_twin.py and Unity conventions
        self._bone_dir = np.array([0., 1., 0.], dtype=np.float64)

        # Pre-allocate output
        self._pen_tip = np.zeros(3, dtype=np.float64)
        self._joint_positions = {}

        log.info(f"✅ FK: {len(self.segments)} segments, "
                 f"total reach = {sum(s['length'] for s in self.segments):.2f}m")

    def compute(self, orientations: dict) -> np.ndarray:
        """
        Compute pen-tip position from joint orientations.

        Args:
            orientations: dict mapping sensor_id → rotation_matrix (3×3 ndarray)
                         e.g. {"S1": R1, "S2": R2, "S3": R3}
        Returns:
            pen_tip: 3D position of pen tip [m]
        """
        pos = self.origin.copy()
        self._joint_positions.clear()

        for seg in self.segments:
            sid = seg["sensor"]
            length = seg["length"]

            if sid in orientations:
                R = orientations[sid]
                # Bone vector: rotate the base direction by joint orientation
                bone_vec = R @ (self._bone_dir * length)
            else:
                # Fallback: extend straight forward
                bone_vec = self._bone_dir * length

            pos = pos + bone_vec
            self._joint_positions[seg["joint"]] = pos.copy()

        self._pen_tip[:] = pos
        return self._pen_tip.copy()

    def get_joint_positions(self) -> dict:
        """Get all joint positions from last compute() call."""
        return {k: v.copy() for k, v in self._joint_positions.items()}

    def get_max_reach(self) -> float:
        """Maximum possible reach of the skeleton chain."""
        return sum(s["length"] for s in self.segments)

    @staticmethod
    def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion [w, x, y, z] to 3×3 rotation matrix.

        Args:
            q: quaternion [w, x, y, z]
        Returns:
            R: 3×3 rotation matrix
        """
        w, x, y, z = q
        xx = x * x; yy = y * y; zz = z * z
        xy = x * y; xz = x * z; yz = y * z
        wx = w * x; wy = w * y; wz = w * z

        R = np.array([
            [1 - 2*(yy + zz),  2*(xy - wz),     2*(xz + wy)],
            [2*(xy + wz),      1 - 2*(xx + zz),  2*(yz - wx)],
            [2*(xz - wy),      2*(yz + wx),      1 - 2*(xx + yy)],
        ], dtype=np.float64)
        return R
