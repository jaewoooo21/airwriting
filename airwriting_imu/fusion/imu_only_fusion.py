"""
IMU-Only Fusion Engine — 15-State Error-State Kalman Filter (v2.0)
=================================================================
Pure-inertial position estimation with research-backed drift correction.

State vector (error-state):
  δx = [δp(3), δv(3), δθ(3), b_g(3), b_a(3)]  → 15 dims
  - δp: position error
  - δv: velocity error
  - δθ: attitude error (Rodrigues parameterization)
  - b_g: gyroscope bias
  - b_a: accelerometer bias

Drift mitigation layers:
  1. Accel bias correction in nominal predict (v2.0 fix)
  2. ZUPT — zero velocity update (energy-based SHOE detector, v2.0)
  3. ZARU — zero angular rate update when stationary
  4. Adaptive process noise Q (motion-dependent, v2.0)
  5. Velocity decay — exponential decay when no significant accel
  6. Velocity clamping — hard limit on max velocity
  7. FK pseudo-measurement (optional, applied externally)

References:
  - Joan Solà, "Quaternion kinematics for ESKF" (2017)
  - FIBA adaptive covariance for ZUPT-aided INS
  - SHOE (Stance Hypothesis Optimal Estimation) detector
"""
import numpy as np
import logging

log = logging.getLogger(__name__)

# Optional Neural ZUPT
try:
    from airwriting_imu.ml.neural_zupt import NeuralZUPTDetector
    _HAS_NEURAL_ZUPT = True
except ImportError:
    _HAS_NEURAL_ZUPT = False


class IMUOnlyFusion:
    """15-state Error-State Kalman Filter for IMU-only position tracking."""

    def __init__(self, config: dict):
        # ── Noise parameters ──
        self.sa = config.get("accel_noise_std", 0.5)
        self.sg = config.get("gyro_noise_std", 0.01)
        self.sa_b = config.get("accel_bias_std", 0.0001)
        self.sg_b = config.get("gyro_bias_std", 0.00001)

        # ── ZUPT config ──
        z_cfg = config.get("zupt", {})
        self.zupt_enabled = z_cfg.get("enabled", True)
        self.zupt_gyro_th = z_cfg.get("gyro_threshold", 0.05)
        self.zupt_avar_th = z_cfg.get("accel_variance_threshold", 0.3)
        self.zupt_noise = z_cfg.get("noise", 0.001)
        self.zupt_window = z_cfg.get("window_size", 20)
        self.zupt_adaptive = z_cfg.get("adaptive", True)

        # ── ZARU config ──
        zaru_cfg = config.get("zaru", {})
        self.zaru_enabled = zaru_cfg.get("enabled", True)
        self.zaru_noise = zaru_cfg.get("noise", 0.0001)

        # ── Constraint config ──
        c_cfg = config.get("constraints", {})
        self.max_vel = c_cfg.get("max_velocity", 3.0)
        self.max_acc = c_cfg.get("max_acceleration", 30.0)
        self.vel_decay = c_cfg.get("velocity_decay", 0.98)

        # ── Adaptive Q config (v2.0) ──
        aq_cfg = config.get("adaptive_q", {})
        self.adaptive_q_enabled = aq_cfg.get("enabled", True)
        self.adaptive_q_motion_scale = aq_cfg.get("motion_scale", 2.0)
        self.adaptive_q_static_scale = aq_cfg.get("static_scale", 0.1)
        self.adaptive_q_energy_th = aq_cfg.get("energy_threshold", 0.5)

        # ── Nominal state ──
        self.pos = np.zeros(3, dtype=np.float64)       # position  [m]
        self.vel = np.zeros(3, dtype=np.float64)       # velocity  [m/s]
        self.q = np.array([1., 0., 0., 0.], np.float64)  # quaternion [w,x,y,z]
        self.bg = np.zeros(3, dtype=np.float64)        # gyro bias [rad/s]
        self.ba = np.zeros(3, dtype=np.float64)        # accel bias [m/s²]

        # ── Error-state covariance P (15×15) ──
        init_cov = config.get("initial_covariance", 1.0)
        self.P = np.eye(15, dtype=np.float64) * init_cov
        # Tighter initial covariance for biases
        self.P[9:12, 9:12] *= 0.01    # gyro bias
        self.P[12:15, 12:15] *= 0.01  # accel bias

        # ── Pre-allocated matrices ──
        self._F = np.eye(15, dtype=np.float64)
        self._Q = np.zeros((15, 15), dtype=np.float64)
        self._I15 = np.eye(15, dtype=np.float64)

        # ZUPT measurement: H_zupt × δx = v_measured → rows for δv
        self._H_zupt = np.zeros((3, 15), dtype=np.float64)
        self._H_zupt[0, 3] = self._H_zupt[1, 4] = self._H_zupt[2, 5] = 1.0
        self._R_zupt = np.eye(3, dtype=np.float64) * self.zupt_noise

        # ZARU measurement: when stationary, gyro ≈ bg
        # We observe the gyro bias directly: z = gyro_measured, h(x) = bg
        # H_zaru selects the gyro bias states (indices 9-11)
        self._H_zaru = np.zeros((3, 15), dtype=np.float64)
        self._H_zaru[0, 9] = self._H_zaru[1, 10] = self._H_zaru[2, 11] = 1.0
        self._R_zaru = np.eye(3, dtype=np.float64) * self.zaru_noise

        # v2.3: Adaptive Measurement Noise (AMN)
        # When confidence is high (stationary), reduce R → stronger correction
        # When confidence is low (moving near threshold), increase R → gentler
        self._amn_enabled = z_cfg.get("adaptive_noise", True)
        self._amn_zupt_scale = 10.0   # R scale range: R_base to R_base*scale
        self._amn_zaru_scale = 5.0

        # FK pseudo-measurement: H_fk × δx = pos_fk - pos_nominal
        self._H_fk = np.zeros((3, 15), dtype=np.float64)
        self._H_fk[0, 0] = self._H_fk[1, 1] = self._H_fk[2, 2] = 1.0

        # ── ZUPT ring buffer (accel + gyro for SHOE detector) ──
        self._abuf = np.zeros((self.zupt_window, 3), dtype=np.float64)
        self._gbuf = np.zeros((self.zupt_window, 3), dtype=np.float64)
        self._aidx = 0
        self._afull = False

        # ── Neural ZUPT detector (optional) ──
        self._neural_zupt = None
        if _HAS_NEURAL_ZUPT and z_cfg.get("neural", False):
            model_path = z_cfg.get("model_path", None)
            self._neural_zupt = NeuralZUPTDetector(z_cfg, model_path)
            log.info("✅ Neural ZUPT detector enabled")

        # ── Timestamps ──
        self.last_ts = None
        self.initialized = True  # No UWB init needed; start at origin

        # ── P symmetrization ──
        self._sym_interval = 50
        self._sym_counter = 0

        # ── Statistics ──
        self.n_zupt = 0
        self.n_zaru = 0
        self.n_update = 0
        self.n_decay = 0
        self.n_rto = 0
        self.zupt_confidence = 0.0

        # FK state
        self._fk_prev = None
        self._pos_prev = None

        log.info("✅ IMUOnlyFusion: ESKF-15 initialized")

    # ════════════════════════════════════════════════
    # Main Update
    # ════════════════════════════════════════════════
    def update(self, accel_world: np.ndarray, gyro: np.ndarray,
               ts_us: int, R_body: np.ndarray = None) -> dict:
        """
        Process one IMU sample.

        Args:
            accel_world: gravity-removed acceleration in world frame [m/s²]
            gyro: angular velocity in body frame [rad/s]
            R_body: rotation matrix from body to world (for process model)
            ts_us: timestamp in microseconds
        Returns:
            dict with position, velocity, zupt_active, zaru_active
        """
        # ── dt calculation ──
        if self.last_ts is None:
            dt = 0.01
        else:
            d = ts_us - self.last_ts
            if d < 0:
                d += 2**32
            dt = np.clip(d / 1e6, 0.001, 0.5)
        self.last_ts = ts_us

        # ── 1. PREDICT (nominal state propagation) ──
        self._predict_nominal(accel_world, gyro, dt)

        # ── 2. PREDICT (error-state covariance) ──
        self._predict_error_cov(accel_world, gyro, dt)

        # ── 3. ZUPT (neural or rule-based) ──
        zupt_active = False
        self.zupt_confidence = 0.0
        if self.zupt_enabled:
            if self._neural_zupt is not None:
                zupt_active, self.zupt_confidence = self._neural_zupt.detect(
                    accel_world, gyro)
            else:
                zupt_active, self.zupt_confidence = self._zupt_check(
                    accel_world, gyro)
            if zupt_active:
                self._apply_zupt()
                self.n_zupt += 1

        # ── 4. ZARU ──
        zaru_active = False
        if self.zaru_enabled and zupt_active:
            # ZARU only when also stationary
            self._apply_zaru(gyro)
            zaru_active = True
            self.n_zaru += 1

        # ── 5. Adaptive velocity decay (v2.3: uses continuous ZUPT confidence) ──
        accel_mag = np.linalg.norm(accel_world)
        if accel_mag < 0.5 and not zupt_active:
            # v2.3: zupt_confidence is now a continuous 0-1 value from
            # SHOE metric, so we use it directly. Higher confidence
            # (closer to stationary) → stronger decay.
            effective_decay = self.vel_decay ** (1.0 + self.zupt_confidence)
            self.vel *= effective_decay
            self.n_decay += 1

        # ── 6. Velocity clamping ──
        vel_mag = np.linalg.norm(self.vel)
        if vel_mag > self.max_vel:
            self.vel *= self.max_vel / vel_mag

        # ── P symmetrization ──
        self._sym_counter += 1
        if self._sym_counter >= self._sym_interval:
            self.P = 0.5 * (self.P + self.P.T)
            self._sym_counter = 0

        self.n_update += 1
        return {
            "position": self.pos.copy(),
            "velocity": self.vel.copy(),
            "initialized": True,
            "zupt_active": zupt_active,
            "zaru_active": zaru_active,
            "zupt_confidence": self.zupt_confidence,
            "dt": dt,
        }

    # ════════════════════════════════════════════════
    # FK Pseudo-Measurement Update
    # ════════════════════════════════════════════════
    def update_fk(self, fk_position: np.ndarray, fk_noise_std: float = 0.05):
        """
        Apply Forward Kinematics position as pseudo-measurement.

        Args:
            fk_position: pen tip position from FK [m]
            fk_noise_std: noise standard deviation for FK measurement
        """
        H = self._H_fk
        R = np.eye(3, dtype=np.float64) * (fk_noise_std ** 2)

        # Innovation
        y = fk_position - self.pos

        S = H @ self.P @ H.T + R
        try:
            K = np.linalg.solve(S, (self.P @ H.T).T).T
        except np.linalg.LinAlgError:
            log.warning("FK update: solve failed")
            return

        # Error-state correction
        dx = K @ y
        self._inject_error(dx)

        # Covariance update (Joseph form)
        IKH = self._I15 - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

    def update_fk_differential(self, fk_position: np.ndarray,
                               gyro_energy: float = 0.0,
                               fk_noise_std: float = 0.05):
        """Apply Differential FK — shoulder-robust pseudo-measurement.

        v2.1 enhancement: Instead of absolute FK position, uses the *change*
        in FK position (ΔFK) to correct the ESKF position *change* (Δpos).
        This makes the filter robust to shoulder movement.

        Additionally, FK noise is scaled by gyro energy: faster joint
        rotation → less reliable FK → higher measurement noise.

        Args:
            fk_position: current FK pen-tip position [m]
            gyro_energy: ||gyro||² for FK confidence weighting
            fk_noise_std: base noise standard deviation
        """
        if self._fk_prev is None:
            self._fk_prev = fk_position.copy()
            self._pos_prev = self.pos.copy()
            return

        # Differential: how much FK moved vs how much ESKF moved
        delta_fk = fk_position - self._fk_prev
        delta_pos = self.pos - self._pos_prev

        # Innovation: difference between FK delta and ESKF delta
        y = delta_fk - delta_pos

        # FK confidence: lower when joints are rotating fast
        # High gyro energy → larger noise → less FK correction
        confidence_scale = 1.0 + 10.0 * min(gyro_energy, 1.0)
        R_noise = np.eye(3, dtype=np.float64) * (fk_noise_std * confidence_scale) ** 2

        H = self._H_fk  # Same H as absolute FK (position observation)
        S = H @ self.P @ H.T + R_noise
        try:
            K = np.linalg.solve(S, (self.P @ H.T).T).T
        except np.linalg.LinAlgError:
            log.warning("Differential FK: solve failed")
            self._fk_prev[:] = fk_position
            self._pos_prev[:] = self.pos
            return

        dx = K @ y
        self._inject_error(dx)

        IKH = self._I15 - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_noise @ K.T

        # Store for next differential
        self._fk_prev[:] = fk_position
        self._pos_prev[:] = self.pos

    def update_rto(self, stroke_origin: np.ndarray,
                   rto_noise_std: float = 0.03):
        """Return-to-Origin constraint for stroke-end drift correction.

        v2.1 enhancement: When a stroke ends (pen-up), assumes the pen
        returns near the stroke starting position. Injects a position
        pseudo-measurement toward the origin.

        Args:
            stroke_origin: position where current stroke started [m]
            rto_noise_std: noise std (lower = stronger correction)
        """
        H = self._H_fk  # Position observation
        R = np.eye(3, dtype=np.float64) * (rto_noise_std ** 2)

        y = stroke_origin - self.pos

        S = H @ self.P @ H.T + R
        try:
            K = np.linalg.solve(S, (self.P @ H.T).T).T
        except np.linalg.LinAlgError:
            log.warning("RTO update: solve failed")
            return

        dx = K @ y
        self._inject_error(dx)

        IKH = self._I15 - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

        self.n_rto += 1
        log.debug(f"RTO correction: |Δpos|={np.linalg.norm(dx[:3]):.4f}m")

    def update_velocity_measurement(self, vel_measured: np.ndarray,
                                     vel_noise_std: float = 0.1):
        """Apply velocity pseudo-measurement from VRN or other velocity source.

        v2.3: Uses the ZUPT H matrix (velocity observation) but with a
        non-zero measurement — the neural network's velocity prediction.

        Args:
            vel_measured: predicted velocity [m/s]
            vel_noise_std: measurement noise standard deviation
        """
        H = self._H_zupt  # Velocity observation
        R = np.eye(3, dtype=np.float64) * (vel_noise_std ** 2)

        # Innovation: measured velocity - current velocity
        y = vel_measured - self.vel

        S = H @ self.P @ H.T + R
        try:
            K = np.linalg.solve(S, (self.P @ H.T).T).T
        except np.linalg.LinAlgError:
            return

        dx = K @ y
        self._inject_error(dx)

        IKH = self._I15 - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T


    def update_position_measurement(self, pos_measured: np.ndarray,
                                    pos_noise_std: float = 0.05):
        """Apply absolute position pseudo-measurement.

        v2.4: Used for soft constraint enforcement (workspace boundary).
        """
        # H_fk selects position states (indices 0-2) -> [I, 0...]
        H = self._H_fk

        # Residual y = z - h(x)
        y = pos_measured - self.pos

        # R measurement noise
        R = np.eye(3, dtype=np.float64) * (pos_noise_std ** 2)

        # Kalman Update
        PHT = self.P @ H.T
        S = H @ PHT + R

        try:
            K = PHT @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        dx = K @ y
        self._inject_error(dx)

        IKH = self._I15 - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.n_update += 1


    # ════════════════════════════════════════════════
    # Nominal State Propagation
    # ════════════════════════════════════════════════
    def _predict_nominal(self, accel_w: np.ndarray, gyro: np.ndarray,
                         dt: float):
        """Propagate nominal state: pos, vel, quaternion, bias.

        v2.4 fix (C2): Accel bias (ba) is now modeled in BODY frame.
        Old behavior (world frame) caused drift when sensor rotated.
        """
        # ── Accel bias correction (Body Frame) ──
        # Compute rotation R from current quaternion
        w, x, y, z = self.q
        xx = x*x; yy = y*y; zz = z*z
        xy = x*y; xz = x*z; yz = y*z
        wx = w*x; wy = w*y; wz = w*z
        
        R = np.array([
            [1 - 2*(yy + zz),  2*(xy - wz),     2*(xz + wy)],
            [2*(xy + wz),      1 - 2*(xx + zz),  2*(yz - wx)],
            [2*(xz - wy),      2*(yz + wx),      1 - 2*(xx + yy)],
        ])

        # Accel in world = R * (a_body_true)
        # a_body_true = a_body_measured - ba
        # accel_w (input) ≈ R * a_body_measured
        # We want: accel_true = R * (a_body_measured - ba)
        #                     = R * a_body_measured - R * ba
        #                     = accel_w - R @ ba
        accel_corrected = accel_w - R @ self.ba

        # Position update
        self.pos += self.vel * dt + 0.5 * accel_corrected * dt * dt

        # Velocity update
        self.vel += accel_corrected * dt

        # Quaternion update (gyro - bias)
        omega = gyro - self.bg
        omega_mag = np.linalg.norm(omega)
        if omega_mag > 1e-10:
            half_angle = 0.5 * omega_mag * dt
            s = np.sin(half_angle) / omega_mag
            dq = np.array([
                np.cos(half_angle),
                omega[0] * s,
                omega[1] * s,
                omega[2] * s,
            ])
            self.q = self._quat_mult(self.q, dq)
            q_norm = np.linalg.norm(self.q)
            if q_norm > 1e-10:
                self.q /= q_norm

        # Biases: random walk (no nominal update)

    # ════════════════════════════════════════════════
    # Error-State Covariance Propagation
    # ════════════════════════════════════════════════
    def _predict_error_cov(self, accel_w: np.ndarray,
                           gyro: np.ndarray, dt: float):
        """Propagate error-state covariance P.

        v2.4 fix (C2): F[3:6, 12:15] uses -R * dt (body frame bias).
        v2.4 fix (P1): Optimized F reset using pre-allocated identity.
        """
        F = self._F
        F[:] = self._I15  # Reset (Optimized P1)

        dt2 = dt * dt

        # δp += δv × dt
        F[0, 3] = F[1, 4] = F[2, 5] = dt

        # Recompute R for Jacobian (reuse calculation from nominal if possible,
        # but for safety/independence we recompute here or assume R is small change)
        # Using current q is correct for linearization point.
        w, x, y, z = self.q
        xx = x*x; yy = y*y; zz = z*z
        xy = x*y; xz = x*z; yz = y*z
        wx = w*x; wy = w*y; wz = w*z
        
        R = np.array([
            [1 - 2*(yy + zz),  2*(xy - wz),     2*(xz + wy)],
            [2*(xy + wz),      1 - 2*(xx + zz),  2*(yz - wx)],
            [2*(xz - wy),      2*(yz + wx),      1 - 2*(xx + yy)],
        ])

        # δv += -[a×]×δθ×dt - R×δba×dt
        # v2.0 fix: always compute skew from world-frame accel
        skew_a = self._skew(accel_w)
        F[3:6, 6:9] = -skew_a * dt
        
        # v2.4 fix (C2): Velocity dependence on Accel Bias (Body Frame)
        # δv_dot = ... - R * δba
        F[3:6, 12:15] = -R * dt

        # δθ += -[ω×]×δθ×dt - δbg×dt
        # v2.0 fix: use actual bias-corrected gyro (was np.zeros(3))
        omega = gyro - self.bg
        F[6:9, 6:9] = np.eye(3) - self._skew(omega) * dt
        F[6:9, 9:12] = -np.eye(3) * dt

        # Process noise Q
        Q = self._Q
        Q[:] = 0
        sa2 = self.sa ** 2
        sg2 = self.sg ** 2
        sab2 = self.sa_b ** 2
        sgb2 = self.sg_b ** 2

        # Position noise (from accel integration)
        q_pp = 0.25 * sa2 * dt2 * dt2
        q_pv = 0.5 * sa2 * dt2 * dt
        q_vv = sa2 * dt2
        for i in range(3):
            Q[i, i] = q_pp
            Q[i, i + 3] = q_pv
            Q[i + 3, i] = q_pv
            Q[i + 3, i + 3] = q_vv

        # Attitude noise
        for i in range(3):
            Q[6 + i, 6 + i] = sg2 * dt2

        # Bias random walk
        for i in range(3):
            Q[9 + i, 9 + i] = sgb2 * dt
            Q[12 + i, 12 + i] = sab2 * dt

        # ── Adaptive Q scaling (v2.0 enhancement) ──
        if self.adaptive_q_enabled:
            accel_energy = np.dot(accel_w, accel_w)
            if accel_energy < self.adaptive_q_energy_th:
                # Stationary: tighten Q to reduce drift
                Q[:6, :6] *= self.adaptive_q_static_scale
            else:
                # Moving: increase Q to allow faster response
                Q[:6, :6] *= self.adaptive_q_motion_scale

        self.P = F @ self.P @ F.T + Q

    # ════════════════════════════════════════════════
    # ZUPT
    # ════════════════════════════════════════════════
    def _zupt_check(self, accel: np.ndarray, gyro: np.ndarray) -> tuple:
        """Check if IMU is stationary using SHOE energy detector + ring buffer.

        v2.3 enhancement: Returns (is_stationary, confidence) tuple where
        confidence is a continuous 0-1 value derived from the SHOE metric.
        This enables adaptive velocity decay even when ZUPT is not triggered.

        v2.0 enhancement: Uses SHOE-like combined energy metric:
          T = (1/W) * Σ [||a_k - g_ref||² + α * ||ω_k||²]
        """
        # Fill ring buffer
        idx = self._aidx
        self._abuf[idx] = accel
        self._gbuf[idx] = gyro
        self._aidx = (idx + 1) % self.zupt_window
        if self._aidx == 0:
            self._afull = True

        if not self._afull and self._aidx < 5:
            return False, 0.0

        cnt = self.zupt_window if self._afull else self._aidx

        # ── SHOE-like energy detector (v2.0) ──
        a_buf = self._abuf[:cnt]
        g_buf = self._gbuf[:cnt]

        # Accel energy: variance of magnitude (deviation from static)
        accel_var = np.var(a_buf, axis=0).sum()

        # Gyro energy: mean squared magnitude
        gyro_energy = np.mean(np.sum(g_buf ** 2, axis=1))

        # Combined SHOE metric (α = 1e3 balances accel [m/s²] vs gyro [rad/s])
        alpha = 1e3
        shoe_metric = accel_var + alpha * gyro_energy

        # Adaptive threshold
        threshold = self.zupt_avar_th
        if self.zupt_adaptive:
            vel_mag = np.linalg.norm(self.vel)
            threshold *= (1.0 + 0.5 * min(vel_mag, 1.0))

        # v2.3: continuous confidence (high when metric is well below threshold)
        # confidence = 1 when metric ≈ 0, gradually drops to 0 at threshold
        confidence = max(0.0, 1.0 - shoe_metric / threshold)

        return shoe_metric < threshold, confidence

    def _apply_zupt(self):
        """Apply ZUPT: velocity → 0.

        v2.3: R is adaptively scaled by zupt_confidence.
        High confidence → small R → strong velocity reset.
        Low confidence (near threshold) → large R → gentle correction.
        """
        H = self._H_zupt

        # v2.3 AMN: scale R inversely with confidence
        if self._amn_enabled and self.zupt_confidence > 0:
            # confidence=1 → scale=1 (strongest), confidence≈0 → scale=amn_scale
            noise_scale = 1.0 + self._amn_zupt_scale * (1.0 - self.zupt_confidence)
            R = self._R_zupt * noise_scale
        else:
            R = self._R_zupt

        y = -self.vel  # Innovation: measured_vel(0) - nominal_vel

        S = H @ self.P @ H.T + R
        try:
            K = np.linalg.solve(S, (self.P @ H.T).T).T
        except np.linalg.LinAlgError:
            self.vel *= 0.1  # Fallback
            return

        dx = K @ y
        self._inject_error(dx)

        IKH = self._I15 - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

    # ════════════════════════════════════════════════
    # ZARU
    # ════════════════════════════════════════════════
    def _apply_zaru(self, gyro: np.ndarray):
        """Apply ZARU: correct gyro bias when stationary.

        v2.3: R adaptively scaled like ZUPT.
        """
        H = self._H_zaru

        # v2.3 AMN: scale R inversely with confidence
        if self._amn_enabled and self.zupt_confidence > 0:
            noise_scale = 1.0 + self._amn_zaru_scale * (1.0 - self.zupt_confidence)
            R = self._R_zaru * noise_scale
        else:
            R = self._R_zaru

        # Innovation: observed gyro - current bias estimate
        y = gyro - self.bg

        S = H @ self.P @ H.T + R
        try:
            K = np.linalg.solve(S, (self.P @ H.T).T).T
        except np.linalg.LinAlgError:
            return

        dx = K @ y
        self._inject_error(dx)

        IKH = self._I15 - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

    # ════════════════════════════════════════════════
    # Error Injection
    # ════════════════════════════════════════════════
    def _inject_error(self, dx: np.ndarray):
        """Inject error-state correction into nominal state."""
        self.pos += dx[0:3]
        self.vel += dx[3:6]

        # Attitude correction via small-angle quaternion
        dtheta = dx[6:9]
        dtheta_mag = np.linalg.norm(dtheta)
        if dtheta_mag > 1e-10:
            half = 0.5 * dtheta_mag
            s = np.sin(half) / dtheta_mag
            dq = np.array([np.cos(half), dtheta[0]*s, dtheta[1]*s, dtheta[2]*s])
        else:
            dq = np.array([1.0, 0.5*dtheta[0], 0.5*dtheta[1], 0.5*dtheta[2]])
        self.q = self._quat_mult(self.q, dq)
        q_norm = np.linalg.norm(self.q)
        if q_norm > 1e-10:
            self.q /= q_norm

        self.bg += dx[9:12]
        self.ba += dx[12:15]

    # ════════════════════════════════════════════════
    # Utility
    # ════════════════════════════════════════════════
    @staticmethod
    def _quat_mult(q1, q2):
        """Hamilton quaternion multiplication [w,x,y,z]."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    @staticmethod
    def _skew(v):
        """Skew-symmetric matrix from 3-vector."""
        return np.array([
            [0,    -v[2],  v[1]],
            [v[2],  0,    -v[0]],
            [-v[1], v[0],  0   ],
        ], dtype=np.float64)

    def reset(self):
        """Reset to initial state."""
        self.pos[:] = 0
        self.vel[:] = 0
        self.q[:] = [1, 0, 0, 0]
        self.bg[:] = 0
        self.ba[:] = 0
        self.P[:] = np.eye(15) * 1.0
        self.P[9:12, 9:12] *= 0.01
        self.P[12:15, 12:15] *= 0.01
        self.last_ts = None
        self._aidx = 0
        self._afull = False
        self._abuf[:] = 0
        self._gbuf[:] = 0
        self._sym_counter = 0
        self.zupt_confidence = 0.0
        # v2.1: clear differential FK state
        if hasattr(self, '_fk_prev'):
            del self._fk_prev
        if hasattr(self, '_pos_prev'):
            del self._pos_prev
        self.n_zupt = self.n_zaru = self.n_update = self.n_decay = 0
        self.n_rto = 0
        log.info("🔄 IMUOnlyFusion reset")
