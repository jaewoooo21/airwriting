import numpy as np
from scipy.spatial.transform import Rotation

class Calibrator:
    def __init__(self, required_samples: int = 300):
        self.req_samples = required_samples
        self.samples = {
            's1_a': [], 's1_g': [],
            's2_a': [], 's2_g': [],
            's3_a': [], 's3_g': [], 's3_m': []
        }
        self.is_calibrated = False
        
        # Results
        self.bg = {'s1': np.zeros(3), 's2': np.zeros(3), 's3': np.zeros(3)}
        self.q_align = {'s1': np.array([0,0,0,1]), 's2': np.array([0,0,0,1]), 's3': np.array([0,0,0,1])}
        self.m_ref = {'s3': np.array([1,0,0])}  # 지자기 캘리브레이션 레퍼런스

    def add_sample(self, frame) -> bool:
        if self.is_calibrated:
            return True
            
        self.samples['s1_a'].append(frame.wrist_accel)
        self.samples['s1_g'].append(frame.wrist_gyro)
        if frame.hand_accel is not None:
            self.samples['s2_a'].append(frame.hand_accel)
            self.samples['s2_g'].append(frame.hand_gyro)
        self.samples['s3_a'].append(frame.finger_accel)
        self.samples['s3_g'].append(frame.finger_gyro)
        
        # [Phase 5] 지자기 센서 (나침반) 데이터 로깅
        if hasattr(frame, 'finger_mag'):
            self.samples['s3_m'].append(frame.finger_mag)
        
        if len(self.samples['s1_a']) >= self.req_samples:
            self._finalize()
            self.is_calibrated = True
            return True
        return False

    def _calc_alignment(self, acc_samples, gyr_samples):
        if len(acc_samples) == 0:
            return np.zeros(3), np.array([0, 0, 0, 1])
            
        acc = np.array(acc_samples)
        gyr = np.array(gyr_samples)
        
        bg = np.mean(gyr, axis=0)
        g_mean = np.mean(acc, axis=0)
        g_norm = np.linalg.norm(g_mean)
        
        if g_norm < 1e-6:
            return bg, np.array([0, 0, 0, 1])
            
        # 6-DOF Auto-Alignment (Gram-Schmidt Orthogonalization)
        # 1. Z축 (World UP)은 센서가 느끼는 중력 반대 방향(g_dir)으로 완벽하게 매핑
        g_dir = g_mean / g_norm
        z_col = g_dir
        
        # 2. Y축 (World FORWARD)는 센서의 물리적 Y축([0,1,0])을 사용자가 앞을 향해 뻗고 있다고 가정.
        #    이 물리적 Y축을 중력과 수직인 평면(수평면)에 투영(Projection)하여 진짜 Forward를 찾음.
        y_raw = np.array([0.0, 1.0, 0.0])
        y_proj = y_raw - np.dot(y_raw, z_col) * z_col
        y_norm = np.linalg.norm(y_proj)
        
        # 만약 센서를 수직(세워둔) 상태라 Y축 투영이 0에 가깝다면 X축을 대타로 사용
        if y_norm < 1e-4:
            x_raw = np.array([1.0, 0.0, 0.0])
            y_proj = x_raw - np.dot(x_raw, z_col) * z_col
            y_norm = np.linalg.norm(y_proj)
            
        y_col = y_proj / y_norm
        
        # 3. X축 (World RIGHT)는 Z와 Y의 외적으로 계산 (직교 보장)
        x_col = np.cross(y_col, z_col)
        x_col = x_col / np.linalg.norm(x_col)
        
        # 4. 회전 행렬 생성 (R = World -> Sensor 변환 매트릭스)
        R = np.column_stack((x_col, y_col, z_col))
        
        # Scipy Rotation을 이용해 쿼터니언(x,y,z,w)으로 변환
        from scipy.spatial.transform import Rotation
        try:
            q = Rotation.from_matrix(R).as_quat()
        except Exception:
            q = np.array([0, 0, 0, 1])
            
        return bg, q

    def _finalize(self):
        self.bg['s1'], self.q_align['s1'] = self._calc_alignment(self.samples['s1_a'], self.samples['s1_g'])
        self.bg['s2'], self.q_align['s2'] = self._calc_alignment(self.samples['s2_a'], self.samples['s2_g'])
        self.bg['s3'], self.q_align['s3'] = self._calc_alignment(self.samples['s3_a'], self.samples['s3_g'])
        
        # [Phase 5] 지자기 캘리브레이션 (초기 정면 헤딩 고정)
        if len(self.samples['s3_m']) > 0:
            m_mean = np.mean(self.samples['s3_m'], axis=0)
            if np.linalg.norm(m_mean) > 1e-6:
                self.m_ref['s3'] = m_mean / np.linalg.norm(m_mean)

    def reset(self):
        for k in self.samples.keys():
            self.samples[k].clear()
        self.is_calibrated = False

    @property
    def samples_accel(self):
        # UI 프로그래스용 헬퍼 프로퍼티
        return self.samples['s1_a']
