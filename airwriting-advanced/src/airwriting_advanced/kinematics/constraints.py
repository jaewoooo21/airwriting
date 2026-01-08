from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..quaternion import (
    q_mul,
    q_conj,
    q_slerp,
    q_normalize,
    swing_twist_decomposition,
    q_to_axis_angle,
    q_from_axis_angle,
)


def apply_elbow_hinge(
    q_w_upper: np.ndarray,
    q_w_fore: np.ndarray,
    elbow_axis_upper: np.ndarray,
    strength: float,
) -> np.ndarray:
    """
    Enforce elbow as hinge joint:
      - compute q_UF = q_UW* ⊗ q_FW? (relative from upper to fore)
      - keep only the twist around the elbow_axis_upper
      - soft blend using slerp

    strength:
      0 -> no change
      1 -> hard hinge
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return q_normalize(q_w_fore)

    q_u_f = q_mul(q_conj(q_w_upper), q_w_fore)
    swing, twist = swing_twist_decomposition(q_u_f, elbow_axis_upper)

    q_u_f_clean = q_slerp(q_u_f, twist, strength)
    return q_normalize(q_mul(q_w_upper, q_u_f_clean))


def apply_wrist_swing_limit(
    q_w_fore: np.ndarray,
    q_w_hand: np.ndarray,
    twist_axis_fore: np.ndarray,
    swing_limit_deg: float,
    strength: float,
) -> np.ndarray:
    """
    Keep hand orientation relative to forearm plausible by limiting swing angle.
    - decompose q_FH into swing + twist around twist_axis_fore
    - limit swing to swing_limit_deg
    - blend strength
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0 or swing_limit_deg <= 0.0:
        return q_normalize(q_w_hand)

    q_f_h = q_mul(q_conj(q_w_fore), q_w_hand)
    swing, twist = swing_twist_decomposition(q_f_h, twist_axis_fore)

    axis, angle = q_to_axis_angle(swing)
    limit = math.radians(float(swing_limit_deg))
    if angle > limit:
        swing_limited = q_from_axis_angle(axis, limit)
    else:
        swing_limited = swing

    q_f_h_clean = q_mul(swing_limited, twist)
    q_f_h_soft = q_slerp(q_f_h, q_f_h_clean, strength)
    return q_normalize(q_mul(q_w_fore, q_f_h_soft))
