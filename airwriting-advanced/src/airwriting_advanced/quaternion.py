from __future__ import annotations

import math
from typing import Tuple

import numpy as np


# Quaternion convention in this project:
# - ndarray shape (4,)
# - order: [w, x, y, z]
# - represents rotation from frame A to frame B when used as q_BA
# - vector rotation: v_B = q_BA ⊗ [0, v_A] ⊗ conj(q_BA)


def q_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if n <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def q_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def q_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = np.asarray(q1, dtype=float).reshape(4)
    q2 = np.asarray(q2, dtype=float).reshape(4)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=float)


def q_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = q_normalize(q)
    v = np.asarray(v, dtype=float).reshape(3)
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=float)
    out = q_mul(q_mul(q, qv), q_conj(q))
    return out[1:4]


def q_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float).reshape(3)
    n = float(np.linalg.norm(axis))
    if n <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    a = axis / n
    s = math.sin(angle_rad * 0.5)
    return q_normalize(np.array([math.cos(angle_rad * 0.5), a[0]*s, a[1]*s, a[2]*s], dtype=float))


def q_to_axis_angle(q: np.ndarray) -> Tuple[np.ndarray, float]:
    q = q_normalize(q)
    w = float(np.clip(q[0], -1.0, 1.0))
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(0.0, 1.0 - w*w))
    if s < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=float), 0.0
    axis = q[1:4] / s
    return axis, angle


def q_from_gyro(omega_rad_s: np.ndarray, dt: float) -> np.ndarray:
    """Small-angle integration quaternion from angular rate omega over dt."""
    omega = np.asarray(omega_rad_s, dtype=float).reshape(3)
    theta = float(np.linalg.norm(omega) * dt)
    if theta < 1e-12:
        # first order
        half = 0.5 * dt
        return q_normalize(np.array([1.0, omega[0]*half, omega[1]*half, omega[2]*half], dtype=float))
    axis = omega / float(np.linalg.norm(omega))
    return q_from_axis_angle(axis, theta)


def q_to_euler_zyx(q: np.ndarray) -> Tuple[float, float, float]:
    """Returns yaw(Z), pitch(Y), roll(X) in radians (Z-Y-X / yaw-pitch-roll)."""
    q = q_normalize(q)
    w, x, y, z = q

    # yaw (z)
    yaw = math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
    # pitch (y)
    sinp = 2.0*(w*y - z*x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi/2.0, sinp)
    else:
        pitch = math.asin(sinp)
    # roll (x)
    roll = math.atan2(2.0*(w*x + y*z), 1.0 - 2.0*(x*x + y*y))
    return yaw, pitch, roll


def q_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q_normalize(q0)
    q1 = q_normalize(q1)
    t = float(np.clip(t, 0.0, 1.0))

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t*(q1 - q0)
        return q_normalize(out)

    theta0 = math.acos(dot)
    sin_theta0 = math.sin(theta0)
    theta = theta0 * t
    s0 = math.sin(theta0 - theta) / sin_theta0
    s1 = math.sin(theta) / sin_theta0
    return q_normalize(s0*q0 + s1*q1)


def swing_twist_decomposition(q: np.ndarray, axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose q into swing * twist where twist is rotation about 'axis' (expressed in the same frame as q's vector part).
    Returns (swing, twist).
    """
    q = q_normalize(q)
    axis = np.asarray(axis, dtype=float).reshape(3)
    n = float(np.linalg.norm(axis))
    if n <= 0.0:
        return q, np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    a = axis / n

    w, x, y, z = q
    v = np.array([x, y, z], dtype=float)
    v_proj = a * float(np.dot(v, a))
    twist = q_normalize(np.array([w, v_proj[0], v_proj[1], v_proj[2]], dtype=float))
    swing = q_mul(q, q_conj(twist))
    return q_normalize(swing), twist
