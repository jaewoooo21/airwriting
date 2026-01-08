from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..quaternion import q_mul, q_normalize, q_conj


@dataclass
class MadgwickParams:
    beta: float = 0.10  # gradient descent gain
    zeta: float = 0.00  # gyro drift bias gain (optional)


class MadgwickAHRS:
    """
    Minimal Madgwick AHRS implementation (6D/9D).
    Quaternion order: [w, x, y, z]
    Inputs:
      - gyr: rad/s
      - acc: m/s^2
      - mag: uT (optional)
    """

    def __init__(self, sample_rate_hz: float, q0: Optional[np.ndarray] = None, params: Optional[MadgwickParams] = None):
        self.dt = 1.0 / float(sample_rate_hz)
        self.params = params or MadgwickParams()
        self.q = q_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=float) if q0 is None else q0)
        self.gyro_bias = np.zeros(3, dtype=float)

    def reset(self, q0: Optional[np.ndarray] = None) -> None:
        self.q = q_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=float) if q0 is None else q0)
        self.gyro_bias[:] = 0.0

    def update(self, gyr_rad_s: np.ndarray, acc_m_s2: np.ndarray, mag_uT: Optional[np.ndarray] = None) -> np.ndarray:
        q1, q2, q3, q4 = self.q  # w x y z
        gx, gy, gz = (np.asarray(gyr_rad_s, dtype=float).reshape(3) - self.gyro_bias).tolist()
        ax, ay, az = np.asarray(acc_m_s2, dtype=float).reshape(3).tolist()

        # normalize accelerometer
        a_norm = math.sqrt(ax*ax + ay*ay + az*az)
        if a_norm < 1e-9:
            return self.q
        ax, ay, az = ax/a_norm, ay/a_norm, az/a_norm

        if mag_uT is not None:
            mx, my, mz = np.asarray(mag_uT, dtype=float).reshape(3).tolist()
            m_norm = math.sqrt(mx*mx + my*my + mz*mz)
            if m_norm < 1e-9:
                mag_uT = None
            else:
                mx, my, mz = mx/m_norm, my/m_norm, mz/m_norm

        # Auxiliary variables to avoid repeated arithmetic
        _2q1 = 2.0*q1
        _2q2 = 2.0*q2
        _2q3 = 2.0*q3
        _2q4 = 2.0*q4
        _4q1 = 4.0*q1
        _4q2 = 4.0*q2
        _4q3 = 4.0*q3
        _8q2 = 8.0*q2
        _8q3 = 8.0*q3
        q1q1 = q1*q1
        q2q2 = q2*q2
        q3q3 = q3*q3
        q4q4 = q4*q4

        if mag_uT is None:
            # 6D: gradient descent corrective step (acc only)
            s1 = _4q1*q3q3 + _2q3*ax + _4q1*q2q2 - _2q2*ay
            s2 = _4q2*q4q4 - _2q4*ax + 4.0*q1q1*q2 - _2q1*ay - _4q2 + _8q2*q2q2 + _8q2*q3q3 + _4q2*az
            s3 = 4.0*q1q1*q3 + _2q1*ax + _4q3*q4q4 - _2q4*ay - _4q3 + _8q3*q2q2 + _8q3*q3q3 + _4q3*az
            s4 = 4.0*q2q2*q4 - _2q2*ax + 4.0*q3q3*q4 - _2q3*ay
            norm_s = math.sqrt(s1*s1 + s2*s2 + s3*s3 + s4*s4)
            if norm_s > 1e-12:
                s1, s2, s3, s4 = s1/norm_s, s2/norm_s, s3/norm_s, s4/norm_s

            # Quaternion derivative
            qDot1 = 0.5*(-q2*gx - q3*gy - q4*gz) - self.params.beta*s1
            qDot2 = 0.5*( q1*gx + q3*gz - q4*gy) - self.params.beta*s2
            qDot3 = 0.5*( q1*gy - q2*gz + q4*gx) - self.params.beta*s3
            qDot4 = 0.5*( q1*gz + q2*gy - q3*gx) - self.params.beta*s4

        else:
            # 9D: include magnetometer
            # Reference direction of Earth's magnetic field
            # Compute h = q ⊗ [0,m] ⊗ q*
            q = self.q
            m = np.array([0.0, mx, my, mz], dtype=float)
            h = q_mul(q_mul(q, m), q_conj(q))
            hx, hy, hz = h[1], h[2], h[3]
            bx = math.sqrt(hx*hx + hy*hy)
            bz = hz

            # Gradient descent step (from well-known Madgwick derivation)
            # Many terms; implement directly.
            _2bx = 2.0*bx
            _2bz = 2.0*bz
            _4bx = 4.0*bx
            _4bz = 4.0*bz

            s1 = (-_2q3*(2.0*(q2*q4 - q1*q3) - ax) + _2q2*(2.0*(q1*q2 + q3*q4) - ay)
                  - _2bz*q3*(_2bx*(0.5 - q3q3 - q4q4) + _2bz*(q2*q4 - q1*q3) - mx)
                  + (-_2bx*q4 + _2bz*q2)*(_2bx*(q2*q3 - q1*q4) + _2bz*(q1*q2 + q3*q4) - my)
                  + _2bx*q3*(_2bx*(q1*q3 + q2*q4) + _2bz*(0.5 - q2q2 - q3q3) - mz))
            s2 = (_2q4*(2.0*(q2*q4 - q1*q3) - ax) + _2q1*(2.0*(q1*q2 + q3*q4) - ay)
                  - 4.0*q2*(1 - 2.0*(q2q2 + q3q3) - az)
                  + _2bz*q4*(_2bx*(0.5 - q3q3 - q4q4) + _2bz*(q2*q4 - q1*q3) - mx)
                  + (_2bx*q3 + _2bz*q1)*(_2bx*(q2*q3 - q1*q4) + _2bz*(q1*q2 + q3*q4) - my)
                  + (_2bx*q4 - _4bz*q2)*(_2bx*(q1*q3 + q2*q4) + _2bz*(0.5 - q2q2 - q3q3) - mz))
            s3 = (-_2q1*(2.0*(q2*q4 - q1*q3) - ax) + _2q4*(2.0*(q1*q2 + q3*q4) - ay)
                  - 4.0*q3*(1 - 2.0*(q2q2 + q3q3) - az)
                  + (-_4bx*q3 - _2bz*q1)*(_2bx*(0.5 - q3q3 - q4q4) + _2bz*(q2*q4 - q1*q3) - mx)
                  + (_2bx*q2 + _2bz*q4)*(_2bx*(q2*q3 - q1*q4) + _2bz*(q1*q2 + q3*q4) - my)
                  + (_2bx*q1 - _4bz*q3)*(_2bx*(q1*q3 + q2*q4) + _2bz*(0.5 - q2q2 - q3q3) - mz))
            s4 = (_2q2*(2.0*(q2*q4 - q1*q3) - ax) + _2q3*(2.0*(q1*q2 + q3*q4) - ay)
                  + (-_4bx*q4 + _2bz*q2)*(_2bx*(0.5 - q3q3 - q4q4) + _2bz*(q2*q4 - q1*q3) - mx)
                  + (-_2bx*q1 + _2bz*q3)*(_2bx*(q2*q3 - q1*q4) + _2bz*(q1*q2 + q3*q4) - my)
                  + _2bx*q2*(_2bx*(q1*q3 + q2*q4) + _2bz*(0.5 - q2q2 - q3q3) - mz))

            norm_s = math.sqrt(s1*s1 + s2*s2 + s3*s3 + s4*s4)
            if norm_s > 1e-12:
                s1, s2, s3, s4 = s1/norm_s, s2/norm_s, s3/norm_s, s4/norm_s

            qDot1 = 0.5*(-q2*gx - q3*gy - q4*gz) - self.params.beta*s1
            qDot2 = 0.5*( q1*gx + q3*gz - q4*gy) - self.params.beta*s2
            qDot3 = 0.5*( q1*gy - q2*gz + q4*gx) - self.params.beta*s3
            qDot4 = 0.5*( q1*gz + q2*gy - q3*gx) - self.params.beta*s4

        # Integrate rate of change
        q1 += qDot1 * self.dt
        q2 += qDot2 * self.dt
        q3 += qDot3 * self.dt
        q4 += qDot4 * self.dt
        self.q = q_normalize(np.array([q1, q2, q3, q4], dtype=float))
        return self.q.copy()
