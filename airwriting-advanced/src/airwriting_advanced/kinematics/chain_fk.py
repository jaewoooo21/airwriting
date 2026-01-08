from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from ..quaternion import q_rotate


@dataclass
class ArmPose:
    shoulder_w: np.ndarray  # (3,)
    elbow_w: np.ndarray
    wrist_w: np.ndarray
    tip_w: np.ndarray


class ArmKinematics3Link:
    def __init__(self, upper_arm_len: float, forearm_len: float, hand_len: float, seg_axis: np.ndarray):
        self.L1 = float(upper_arm_len)
        self.L2 = float(forearm_len)
        self.L3 = float(hand_len)
        a = np.asarray(seg_axis, dtype=float).reshape(3)
        n = float(np.linalg.norm(a))
        if n <= 0.0:
            self.axis = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            self.axis = a / n

    def forward(self, q_w_upper: np.ndarray, q_w_fore: np.ndarray, q_w_hand: np.ndarray,
                shoulder_w: np.ndarray | None = None) -> ArmPose:
        shoulder = np.zeros(3, dtype=float) if shoulder_w is None else np.asarray(shoulder_w, dtype=float).reshape(3)
        v1 = q_rotate(q_w_upper, self.axis) * self.L1
        elbow = shoulder + v1
        v2 = q_rotate(q_w_fore, self.axis) * self.L2
        wrist = elbow + v2
        v3 = q_rotate(q_w_hand, self.axis) * self.L3
        tip = wrist + v3
        return ArmPose(shoulder_w=shoulder, elbow_w=elbow, wrist_w=wrist, tip_w=tip)
