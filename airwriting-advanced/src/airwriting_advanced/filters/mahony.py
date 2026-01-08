from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..quaternion import q_mul, q_normalize, q_conj


@dataclass
class MahonyParams:
    kp: float = 1.5
    ki: float = 0.05


class MahonyAHRS:
    """
    Mahony nonlinear complementary filter (6D/9D).
    Quaternion order: [w, x, y, z]
    """

    def __init__(self, sample_rate_hz: float, q0: Optional[np.ndarray] = None, params: Optional[MahonyParams] = None):
        self.dt = 1.0 / float(sample_rate_hz)
        self.params = params or MahonyParams()
        self.q = q_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=float) if q0 is None else q0)
        self.e_int = np.zeros(3, dtype=float)

    def reset(self, q0: Optional[np.ndarray] = None) -> None:
        self.q = q_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=float) if q0 is None else q0)
        self.e_int[:] = 0.0

    def update(self, gyr_rad_s: np.ndarray, acc_m_s2: np.ndarray, mag_uT: Optional[np.ndarray] = None) -> np.ndarray:
        gx, gy, gz = np.asarray(gyr_rad_s, dtype=float).reshape(3).tolist()
        ax, ay, az = np.asarray(acc_m_s2, dtype=float).reshape(3).tolist()

        # normalize acc
        a_norm = math.sqrt(ax*ax + ay*ay + az*az)
        if a_norm < 1e-9:
            return self.q.copy()
        ax, ay, az = ax/a_norm, ay/a_norm, az/a_norm

        # estimated gravity from quaternion
        q = self.q
        # g_body = R^T * [0,0,1]  (if world z up). Use quaternion rotate of world z into body frame:
        # body vector = q* ⊗ [0,v] ⊗ q
        z_world = np.array([0.0, 0.0, 1.0], dtype=float)
        g_est = self._rotate_world_to_body(q, z_world)

        # error = cross(acc_meas, g_est)
        e = np.cross(np.array([ax, ay, az], dtype=float), g_est)

        # optional mag for yaw correction: use measured mag vs estimated mag
        if mag_uT is not None:
            mx, my, mz = np.asarray(mag_uT, dtype=float).reshape(3).tolist()
            m_norm = math.sqrt(mx*mx + my*my + mz*mz)
            if m_norm > 1e-9:
                mx, my, mz = mx/m_norm, my/m_norm, mz/m_norm
                # compute reference direction of magnetic field in body frame
                # We'll use simple horizontal projection: error from heading.
                m_body = np.array([mx, my, mz], dtype=float)
                # rotate body mag to world frame, keep horizontal, rotate back
                m_world = self._rotate_body_to_world(q, m_body)
                m_world[2] = 0.0
                n = np.linalg.norm(m_world)
                if n > 1e-9:
                    m_world /= n
                    m_ref_body = self._rotate_world_to_body(q, m_world)
                    e_mag = np.cross(m_body, m_ref_body)
                    e = e + e_mag

        kp = self.params.kp
        ki = self.params.ki
        self.e_int += e * ki * self.dt
        omega = np.array([gx, gy, gz], dtype=float) + kp*e + self.e_int

        # integrate quaternion using omega
        q_dot = 0.5 * q_mul(q, np.array([0.0, omega[0], omega[1], omega[2]], dtype=float))
        self.q = q_normalize(q + q_dot * self.dt)
        return self.q.copy()

    @staticmethod
    def _rotate_world_to_body(q_wb: np.ndarray, v_world: np.ndarray) -> np.ndarray:
        # q_wb rotates world->body? Here we treat q as body->world in other modules,
        # but for this internal computation, use q_conj to invert.
        q = q_normalize(q_wb)
        v = np.asarray(v_world, dtype=float).reshape(3)
        qv = np.array([0.0, v[0], v[1], v[2]], dtype=float)
        out = q_mul(q_mul(q_conj(q), qv), q)  # world->body
        return out[1:4]

    @staticmethod
    def _rotate_body_to_world(q_wb: np.ndarray, v_body: np.ndarray) -> np.ndarray:
        q = q_normalize(q_wb)
        v = np.asarray(v_body, dtype=float).reshape(3)
        qv = np.array([0.0, v[0], v[1], v[2]], dtype=float)
        out = q_mul(q_mul(q, qv), q_conj(q))
        return out[1:4]
