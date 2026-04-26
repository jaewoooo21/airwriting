import numpy as np
from scipy.spatial.transform import Rotation

class ESKF:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.g = np.array([0, 0, 9.81])
        
        # Nominal State
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = Rotation.from_quat([0, 0, 0, 1]) # scalar-last [x,y,z,w]
        self.a_b = np.zeros(3)
        self.w_b = np.zeros(3)
        self.mag_ref_world = None
        
        # Error State Covariance (15x15)
        self.P = np.eye(15) * 0.01
        
        # Process Noise Covariances (Base values)
        self.Q_base = np.eye(12) 
        self.Q_base[0:3, 0:3] *= 0.05    # accel noise
        self.Q_base[3:6, 3:6] *= 0.01    # gyro noise
        self.Q_base[6:9, 6:9] *= 0.001   # accel bias random walk
        self.Q_base[9:12, 9:12] *= 0.0001# gyro bias random walk
        self.Q = np.copy(self.Q_base)
        
        # Measurement Covariance (ZUPT)
        self.R_zupt = np.eye(3) * 0.05
        self.R_zaru = np.eye(3) * 0.001
        
        # ZUPT Window
        self.window_size = 15
        self.a_win = []
        self.g_win = []
        
    def reset(self, initial_q=None, initial_bg=None, initial_mag=None):
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        if initial_q is not None:
            # calibration.py가 준 q_align은 World -> Sensor 회전입니다. 
            # ESKF 내부에서 self.q는 Sensor -> World 여야 하므로 역행렬(inv)을 취합니다.
            self.q = Rotation.from_quat(initial_q).inv()
        else:
            self.q = Rotation.from_quat([0, 0, 0, 1])
            
        self.a_b = np.zeros(3)
        if initial_bg is not None:
            self.w_b = initial_bg.copy()
        else:
            self.w_b = np.zeros(3)
            
        # [Phase 5] 지자기 참조점 (World Frame) 저장
        if initial_mag is not None:
            self.mag_ref_world = self.q.as_matrix() @ initial_mag
        else:
            self.mag_ref_world = None
        self.P = np.eye(15) * 0.01
        self.a_win.clear()
        self.g_win.clear()

    def _skew(self, v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def predict(self, accel, gyro, dt):
        # 1. Update Nominal State
        accel_true = accel - self.a_b
        gyro_true = gyro - self.w_b
        
        # Orientation update
        angle = np.linalg.norm(gyro_true) * dt
        if angle > 1e-8:
            axis = gyro_true / np.linalg.norm(gyro_true)
            dq = Rotation.from_rotvec(axis * angle)
            self.q = self.q * dq
            
        R = self.q.as_matrix()
        
        # Velocity & Position update
        accel_world = R @ accel_true - self.g
            
        self.p = self.p + self.v * dt + 0.5 * accel_world * (dt ** 2)
        self.v = self.v + accel_world * dt
        
        # 2. Update Error State Covariance
        Fx = np.eye(15)
        Fx[0:3, 3:6] = np.eye(3) * dt
        Fx[3:6, 6:9] = -R @ self._skew(accel_true) * dt
        Fx[3:6, 9:12] = -R * dt
        
        # rot error kinematics
        # approx exp(-[w]dt) ~ I - [w]dt
        Fx[6:9, 6:9] = np.eye(3) - self._skew(gyro_true) * dt
        Fx[6:9, 12:15] = -np.eye(3) * dt
        
        Fi = np.zeros((15, 12))
        Fi[3:6, 0:3] = R
        Fi[6:9, 3:6] = np.eye(3)
        Fi[9:12, 6:9] = np.eye(3)
        Fi[12:15, 9:12] = np.eye(3)
        
        self.P = Fx @ self.P @ Fx.T + (Fi * dt) @ self.Q @ (Fi * dt).T
        
        self.a_win.append(accel)
        self.g_win.append(gyro)
        if len(self.a_win) > self.window_size:
            self.a_win.pop(0)
            self.g_win.pop(0)
            
        # 4. Adaptive Q & Damping & Clamping
        # ZUPT 상태 점검 (predict() 직전에 밖에서 세팅될 수도 있으므로 여기서 방어적 처리)
        is_stationary = self.detect_zupt()
        if is_stationary:
            self.Q = self.Q_base * 0.1
            self.v *= 0.90 # 정지 상태 강력한 제동
        else:
            self.Q = self.Q_base * 2.0
            self.v *= 0.975 # 초당 약 ~48% 보존되도록 살짝 더 잡아줌 (너무 빠름 방지)
            
        # 속도 클램핑 (최대 2.0 m/s 제한, 인간 필기 속도 한계)
        v_norm = np.linalg.norm(self.v)
        if v_norm > 2.0:
            self.v = (self.v / v_norm) * 2.0

    def detect_zupt(self, accel_th=0.05, gyro_th=0.05):
        if len(self.a_win) < self.window_size:
            return False
            
        a_var = np.var(self.a_win, axis=0)
        g_var = np.var(self.g_win, axis=0)
        
        # SHOE-like Energy Score
        a_mean = np.mean(self.a_win, axis=0)
        a_n = np.linalg.norm(a_mean)
        is_gravity = abs(a_n - 9.81) < 0.5
        
        a_score = np.sum(a_var)
        g_score = np.sum(g_var)
        
        # 임계값: 파라미터는 분산의 합 기준
        if a_score < (accel_th * 3) and g_score < (gyro_th * 3) and is_gravity:
            return True
        return False

    def update_zupt(self, current_gyro=None):
        # [ZARU] 정지 상태 자이로 바이어스 보정
        if current_gyro is not None:
            H_zaru = np.zeros((3, 15))
            H_zaru[0:3, 12:15] = np.eye(3)
            # z_g is the negative of the expected measurement
            z_g = current_gyro - self.w_b
            
            K_g = self.P @ H_zaru.T @ np.linalg.inv(H_zaru @ self.P @ H_zaru.T + self.R_zaru)
            # [핵심] ZARU(자이로 바이어스 보정)가 회전각(q)을 미친듯이 꼬아버리는 참사 방지
            K_g[6:9, :] = 0
            dx_g = K_g @ z_g
            self._inject_error(dx_g)
            
        # [ZUPT] 정지 상태 속도 0 보정
        H_zupt = np.zeros((3, 15))
        H_zupt[0:3, 3:6] = np.eye(3)
        z_v = np.zeros(3) - self.v
        
        K_v = self.P @ H_zupt.T @ np.linalg.inv(H_zupt @ self.P @ H_zupt.T + self.R_zupt)
        # [핵심] ZUPT가 속도를 강제로 0으로 깎을 때, 필터가 "앗 자세(q)가 잘못되었구나!" 하고 
        # 회전각을 강제로 비틀어버리는 현상(버튼 뗄 때 튀는 현상)을 방지합니다.
        K_v[6:9, :] = 0  
        dx_v = K_v @ z_v
        self._inject_error(dx_v)
        
    def update_mag(self, current_mag):
        """[Phase 5] 지자기 센서를 이용한 9축 헤딩(Yaw) 고정"""
        if self.mag_ref_world is None:
            return
            
        m_w = self.q.as_matrix() @ current_mag
        
        # World 수평면(XY) 투영
        m_w_xy = m_w[:2]
        n_w = np.linalg.norm(m_w_xy)
        if n_w < 1e-4: return
        m_w_xy /= n_w
        
        m_ref_xy = self.mag_ref_world[:2]
        n_ref = np.linalg.norm(m_ref_xy)
        if n_ref < 1e-4: return
        m_ref_xy /= n_ref
        
        # 측정된 북쪽과 레퍼런스 북쪽 간의 각도 오차 계산
        yaw_err = np.arctan2(m_w_xy[1], m_w_xy[0]) - np.arctan2(m_ref_xy[1], m_ref_xy[0])
        yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi
        
        # 요동침(Spike) 방지를 위해 아주 약한 게인(0.01)으로 보정 (Complementary 방식)
        # 장시간 길게 문장 쓸 때 서서히 방향이 틀어지는 드리프트만 원천 방지
        gain = 0.01
        correction = np.array([0, 0, yaw_err * gain])
        
        # World 기반 회전 적용!
        dq = Rotation.from_rotvec(correction)
        self.q = dq * self.q

    def update_gravity_mahony(self, current_accel, alpha=0.002):
        """센서가 기울어질 때 발생하는 미세한 물리적 자이로 오차(Gravity-Bleed)를 보정하여 
        Pitch/Roll 위아래 드리프트를 영구적으로 차단합니다."""
        a_n = np.linalg.norm(current_accel)
        # 외부 가속이 심할 때(1G에서 멀어질 때)는 중력 보정을 스킵합니다
        if a_n < 9.0 or a_n > 11.0:
            return
            
        a_norm = current_accel / a_n
        
        # 센서 좌표계에서 예상되는 중력(Gravity) 방향
        g_sensor = self.q.inv().as_matrix() @ np.array([0.0, 0.0, 1.0])
        
        # 실제 가속도계가 가리키는 중력 방향과의 오차 (외적)
        err = np.cross(a_norm, g_sensor)
        
        # [핵심 수정] 자이로 바이어스(Drift) 자동 교정 로직 삭제
        # 이유: 손을 좌우로 흔들 때 발생하는 선형 가속도를 "센서가 기울어졌다"고 착각하고, 
        # 이에 대한 보상으로 자이로 센서 수치(w_b)를 왜곡시킵니다. 
        # 시간이 지날수록 축이 45도 이상 완전히 틀어져서, 좌우로 그어도 대각선으로 써지는 치명적 원인이 되었습니다.
        # 회전(q)를 중력 방향으로 아주 미세하게 당김
        if np.linalg.norm(err) > 1e-6:
            err_world = self.q.as_matrix() @ err
            dq = Rotation.from_rotvec(-err_world * alpha)
            self.q = dq * self.q

    def _inject_error(self, dx):
        self.p += dx[0:3]
        self.v += dx[3:6]
        
        angle = np.linalg.norm(dx[6:9])
        if angle > 1e-8:
            axis = dx[6:9] / angle
            dq = Rotation.from_rotvec(axis * angle)
            self.q = self.q * dq
            
        self.a_b += dx[9:12]
        self.w_b += dx[12:15]
