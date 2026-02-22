"""
MadgwickAHRS v3.2.1
Fix: np.radians 이중변환 제거 (입력은 이미 rad/s)
Fix: rotation_matrix 사전할당
"""
import numpy as np
import math


class MadgwickAHRS:
    def __init__(self, beta=0.1, sample_rate=100.0):
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.sample_period = 1.0 / sample_rate
        self._R = np.eye(3)
        self._s = np.zeros(4)
        self._qdot = np.zeros(4)

    def update_imu(self, gyro, accel, dt=None):
        if dt is None:
            dt = self.sample_period

        q = self.q
        # ✅ 수정: gyro는 이미 rad/s → 변환하지 않음
        # 펌웨어에서 raw_dps / 32.8 * (PI/180) 로 변환 완료
        gx, gy, gz = float(gyro[0]), float(gyro[1]), float(gyro[2])
        ax, ay, az = float(accel[0]), float(accel[1]), float(accel[2])

        # 가속도 정규화
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm < 1e-10:
            return q
        ax /= norm
        ay /= norm
        az /= norm

        q1, q2, q3, q4 = q  # w, x, y, z
        _2q1 = 2 * q1; _2q2 = 2 * q2; _2q3 = 2 * q3; _2q4 = 2 * q4
        _4q1 = 4 * q1; _4q2 = 4 * q2; _4q3 = 4 * q3
        _8q2 = 8 * q2; _8q3 = 8 * q3
        q1q1 = q1 * q1; q2q2 = q2 * q2; q3q3 = q3 * q3; q4q4 = q4 * q4

        # Gradient descent step
        s = self._s
        s[0] = _4q1 * q3q3 + _2q3 * ax + _4q1 * q2q2 - _2q2 * ay
        s[1] = (_4q2 * q4q4 - _2q4 * ax + 4 * q1q1 * q2 - _2q1 * ay
                - _4q2 + _8q2 * q2q2 + _8q2 * q3q3 + _4q2 * az)
        s[2] = (4 * q1q1 * q3 + _2q1 * ax + _4q3 * q4q4 - _2q4 * ay
                - _4q3 + _8q3 * q2q2 + _8q3 * q3q3 + _4q3 * az)
        s[3] = 4 * q2q2 * q4 - _2q2 * ax + 4 * q3q3 * q4 - _2q3 * ay

        s_norm = math.sqrt(s[0]**2 + s[1]**2 + s[2]**2 + s[3]**2)
        if s_norm > 1e-10:
            s /= s_norm

        # 쿼터니언 미분 (자이로 기반)
        qdot = self._qdot
        qdot[0] = 0.5 * (-q2 * gx - q3 * gy - q4 * gz)
        qdot[1] = 0.5 * ( q1 * gx + q3 * gz - q4 * gy)
        qdot[2] = 0.5 * ( q1 * gy - q2 * gz + q4 * gx)
        qdot[3] = 0.5 * ( q1 * gz + q2 * gy - q3 * gx)

        # 보정 적용
        qdot -= self.beta * s

        # 적분 + 정규화
        q = q + qdot * dt
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-10:
            q /= q_norm

        self.q = q
        return q

    def update_marg(self, gyro, accel, mag, dt=None):
        """9-axis MARG update: Gyro(rad/s), Accel(g or m/s²), Mag(any unit)"""
        if dt is None:
            dt = self.sample_period

        q = self.q
        gx, gy, gz = float(gyro[0]), float(gyro[1]), float(gyro[2])
        ax, ay, az = float(accel[0]), float(accel[1]), float(accel[2])
        mx, my, mz = float(mag[0]), float(mag[1]), float(mag[2])

        # Normalize accel
        normA = math.sqrt(ax*ax + ay*ay + az*az)
        if normA < 1e-10:
            return self.update_imu(gyro, accel, dt)
        ax /= normA; ay /= normA; az /= normA

        # Normalize mag
        normM = math.sqrt(mx*mx + my*my + mz*mz)
        if normM < 1e-10:
            return self.update_imu(gyro, accel, dt)
        mx /= normM; my /= normM; mz /= normM

        q1, q2, q3, q4 = q
        _2q1 = 2 * q1; _2q2 = 2 * q2; _2q3 = 2 * q3; _2q4 = 2 * q4
        _2q1q3 = 2 * q1 * q3; _2q3q4 = 2 * q3 * q4
        q1q1 = q1 * q1; q1q2 = q1 * q2; q1q3 = q1 * q3; q1q4 = q1 * q4
        q2q2 = q2 * q2; q2q3 = q2 * q3; q2q4 = q2 * q4
        q3q3 = q3 * q3; q3q4 = q3 * q4
        q4q4 = q4 * q4

        _2q1mx = 2 * q1 * mx; _2q1my = 2 * q1 * my; _2q1mz = 2 * q1 * mz
        _2q2mx = 2 * q2 * mx

        # Reference direction of Earth's magnetic field
        hx = mx * q1q1 - _2q1my * q4 + _2q1mz * q3 + mx * q2q2 + _2q2 * my * q3 + _2q2 * mz * q4 - mx * q3q3 - mx * q4q4
        hy = _2q1mx * q4 + my * q1q1 - _2q1mz * q2 + _2q2mx * q3 + my * q2q2 + my * q3q3 + _2q3 * mz * q4 - my * q4q4
        _2bx = math.sqrt(hx * hx + hy * hy)
        _2bz = -_2q1mx * q3 + _2q1my * q2 + mz * q1q1 + _2q2mx * q4 - mz * q2q2 + _2q3 * my * q4 - mz * q3q3 + mz * q4q4
        _4bx = 2 * _2bx; _4bz = 2 * _2bz
        
        # Gradient descent step
        s = self._s
        s[0] = -_2q3 * (2 * q2q4 - _2q1q3 - ax) + _2q2 * (2 * q1q2 + _2q3q4 - ay) - _2bz * q3 * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) + (-_2bx * q4 + _2bz * q2) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) + _2bx * q3 * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz)
        s[1] = _2q4 * (2 * q2q4 - _2q1q3 - ax) + _2q1 * (2 * q1q2 + _2q3q4 - ay) - 4 * q2 * (1 - 2 * q2q2 - 2 * q3q3 - az) + _2bz * q4 * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) + (_2bx * q3 + _2bz * q1) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) + (_2bx * q4 - _4bz * q2) * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz)
        s[2] = -_2q1 * (2 * q2q4 - _2q1q3 - ax) + _2q4 * (2 * q1q2 + _2q3q4 - ay) - 4 * q3 * (1 - 2 * q2q2 - 2 * q3q3 - az) + (-_4bx * q3 - _2bz * q1) * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) + (_2bx * q2 + _2bz * q4) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) + (_2bx * q1 - _4bz * q3) * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz)
        s[3] = _2q2 * (2 * q2q4 - _2q1q3 - ax) + _2q3 * (2 * q1q2 + _2q3q4 - ay) + (-_4bx * q4 + _2bz * q2) * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) + (-_2bx * q1 + _2bz * q3) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) + _2bx * q2 * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz)

        s_norm = math.sqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2] + s[3]*s[3])
        if s_norm > 1e-10:
            s /= s_norm

        # Quaternion rate of change (gyro)
        qdot = self._qdot
        qdot[0] = 0.5 * (-q2 * gx - q3 * gy - q4 * gz)
        qdot[1] = 0.5 * ( q1 * gx + q3 * gz - q4 * gy)
        qdot[2] = 0.5 * ( q1 * gy - q2 * gz + q4 * gx)
        qdot[3] = 0.5 * ( q1 * gz + q2 * gy - q3 * gx)

        # Apply feedback step
        qdot -= self.beta * s

        # Integrate and normalize
        q = q + qdot * dt
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-10:
            q /= q_norm

        self.q = q
        return q

    def rotation_matrix(self):
        """q → 3×3 회전행렬 (사전할당 재사용)"""
        w, x, y, z = self.q
        R = self._R
        xx = x*x; yy = y*y; zz = z*z
        xy = x*y; xz = x*z; yz = y*z
        wx = w*x; wy = w*y; wz = w*z
        R[0,0] = 1-2*(yy+zz); R[0,1] = 2*(xy-wz); R[0,2] = 2*(xz+wy)
        R[1,0] = 2*(xy+wz);   R[1,1] = 1-2*(xx+zz); R[1,2] = 2*(yz-wx)
        R[2,0] = 2*(xz-wy);   R[2,1] = 2*(yz+wx);   R[2,2] = 1-2*(xx+yy)
        return R

    def euler_deg(self):
        w, x, y, z = self.q
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = math.degrees(math.atan2(t0, t1))

        t2 = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
        pitch = math.degrees(math.asin(t2))

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.degrees(math.atan2(t3, t4))
        return np.array([roll, pitch, yaw])

    def reset(self):
        self.q[:] = [1, 0, 0, 0]
