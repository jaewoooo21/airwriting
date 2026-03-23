"""
IMU-Only AirWriting Controller
===============================
Adapted from v3.2 controller — UWB parsing removed.
Pipeline: ESP32 UDP → Parse → Calibrate → Madgwick → World-Frame Accel
         → ESKF fusion + FK + Constraints → Unity/Dashboard output
"""
import socket, struct, threading, time, json, logging
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Optional

from airwriting_imu.fusion.imu_only_fusion import IMUOnlyFusion
from airwriting_imu.fusion.madgwick import MadgwickAHRS
from airwriting_imu.fusion.forward_kinematics import ForwardKinematics
from airwriting_imu.filters.one_euro import OneEuroFilter3D
from airwriting_imu.constraints.biomechanical import BiomechanicalConstraints
from airwriting_imu.constraints.writing_plane import WritingPlaneDetector
from airwriting_imu.constraints.drift_observer import DriftObserver
from airwriting_imu.constraints.loop_closure import LoopClosureDetector
from airwriting_imu.core.command_bus import CommandBus
from airwriting_imu.core.policy_engine import PolicyEngine

log = logging.getLogger(__name__)

_FMT_TS = struct.Struct("<I")
_FMT_6F = struct.Struct("<6f")
_SIDS = ("S1", "S2", "S3")
_OFFSETS = (5, 29, 53)


# NumPy 2.0 compatible JSON serializer
def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


class AirWritingIMUController:
    """IMU-only AirWriting controller — no UWB."""

    HEADER = 0xAA
    FOOTER = 0x55
    PKT_LEN = 79      # v1: no button
    PKT_LEN_V2 = 80   # v2: with button byte at offset 77
    PKT_LEN_V3 = 92   # v3: S1(6), S2(6), S3(9) + button + cksum

    def __init__(self, config):
        self.config = config
        self.running = False
        self._stop_flag = False

        # ── Network ──
        ports = config.network.get("ports", {})
        self._esp_port = ports.get("esp32_to_python", 12345)
        self._uni_port = ports.get("python_to_unity", 12346)
        self._dash_port = ports.get("python_to_dashboard", 12347)
        self._action_port = ports.get("python_to_action", 12348)
        self._ctrl_port = ports.get("python_control", 12350)
        self._uni_ip = config.network.get("unity", {}).get("ip", "127.0.0.1")
        self._dash_ip = config.network.get("dashboard", {}).get("ip", "127.0.0.1")

        # Track config directory for bias saving (v2.3)
        self._config_dir = getattr(config, '_config_dir', None)

        # ── Calibration ──
        cal = config.calibration or {}
        self._cal_target = cal.get("num_samples", 300)
        self._cal_timeout = cal.get("timeout_sec", 10)
        self._gravity = cal.get("gravity", 9.81)
        self._calibrated = False
        self._cal_buf = {s: {"a": [], "g": []} for s in config.imu_sensors}
        self._cal_t0 = None
        self._grav_world = np.array([0., 0., self._gravity], dtype=np.float64)

        # ── IMU-Only Fusion (ESKF) ──
        self.fusion = IMUOnlyFusion(config.fusion)

        # ── Biomechanical Constraints ──
        c_cfg = config.fusion.get("constraints", {})
        self.constraints = BiomechanicalConstraints(c_cfg)

        # ── Writing Plane Detector (v2.2) ──
        plane_cfg = config.fusion.get("writing_plane", {})
        self._plane_enabled = plane_cfg.get("enabled", True)
        self._plane = WritingPlaneDetector(
            buffer_size=plane_cfg.get("buffer_size", 100),
            min_spread=plane_cfg.get("min_spread", 0.005),
            suppress_ratio=plane_cfg.get("suppress_ratio", 1.0),
            absolute_lock=plane_cfg.get("absolute_lock", True),
        )

        # ── Drift Observer (v2.2) ──
        drift_cfg = config.fusion.get("drift_observer", {})
        self._drift_enabled = drift_cfg.get("enabled", True)
        self._drift_observer = DriftObserver(
            window=drift_cfg.get("window", 30),
            drift_threshold=drift_cfg.get("threshold", 0.001),
            correction_factor=drift_cfg.get("correction_factor", 0.5),
        )

        # ── Loop Closure Detector (v2.3) ──
        lc_cfg = config.fusion.get("loop_closure", {})
        self._lc_enabled = lc_cfg.get("enabled", True)
        self._loop_closure = LoopClosureDetector(
            min_loop_length=lc_cfg.get("min_loop_length", 15),
            proximity_m=lc_cfg.get("proximity_m", 0.008),
            cooldown=lc_cfg.get("cooldown", 20),
            max_buffer=lc_cfg.get("max_buffer", 500),
        )
        self._lc_noise_std = lc_cfg.get("noise_std", 0.02)

        # ── Forward Kinematics ──
        fk_cfg = config.fusion.get("forward_kinematics", {})
        self.fk_enabled = fk_cfg.get("enabled", True)
        self.fk_weight = fk_cfg.get("fk_weight", 0.3)
        self.fk = ForwardKinematics(config.skeleton_raw)

        # ── Madgwick filters (one per sensor) ──
        mc = config.madgwick or {}
        beta = mc.get("beta", 0.1)
        rate = mc.get("sample_rate", 100)
        self._mw_warmup_target = mc.get("warmup_samples", 200)
        self._mw = {s: MadgwickAHRS(beta, rate) for s in config.imu_sensors}
        self._mw_count = {s: 0 for s in config.imu_sensors}
        self._mw_ready = {s: False for s in config.imu_sensors}

        # ── Smoothing ──
        self._pos_filt = OneEuroFilter3D(rate, 1.0, 0.007)

        # ── Timing ──
        self._last_ts = {}
        self._frame = 0
        self._t_stats = 0.
        self._t_start = 0.

        # ── Unity buffer & rate limit ──
        self._ubuf = {}
        self._uni_interval = 1.0 / config.unity_send.get("rate_hz", 60)
        self._last_uni_send = 0.

        # ── Pre-allocated work buffers ──
        self._accel_w = np.empty(3, dtype=np.float64)

        self._cksum_err = 0
        self._debug_count = 5  # print first N packets for debugging

        # ── Pen state (v2.2: GPIO 15 button) ──
        self._pen_down = False
        self._pen_prev = False
        self._stroke_origin = np.zeros(3, dtype=np.float64)
        self._stroke_active = False
        self._stroke_positions = []  # v2.5: collect FK positions during stroke

        # ── Macro OS: Command Bus & Policy Engine (v3.0 Pivot) ──
        self._macro_os_enabled = False
        try:
            cfg_dir = getattr(config, '_config_dir', None)
            if cfg_dir:
                sys_path = Path(cfg_dir) / 'system.yaml'
                if sys_path.exists():
                    with open(sys_path, encoding='utf-8') as f:
                        sys_data = yaml.safe_load(f)
                    mos = sys_data.get('macro_os', {})
                    self._macro_os_enabled = mos.get('enabled', False)
                    if self._macro_os_enabled:
                        profiles = mos.get('profiles', {})
                        active_profile = mos.get('active_profile', 'RUNNING')
                        udp_ports = mos.get('udp_ports', {})
                        
                        # Initialize Policy Engine
                        self.policy_engine = PolicyEngine(profiles, initial_profile=active_profile)
                        
                        # Initialize Command Bus with Policy Engine for context-aware mapping
                        self.command_bus = CommandBus(self.policy_engine, udp_ports)
                        log.info(f"🌐 Macro OS enabled (Active: {active_profile}, Profiles: {list(profiles.keys())})")
        except Exception as e:
            log.warning(f"⚠️ Failed to initialize Macro OS: {e}")

        # ── Per-sensor gyro cache (v2.2: for FK confidence) ──
        self._gyro_cache = {s: np.zeros(3) for s in config.imu_sensors}

        log.info("✅ IMU-Only Controller init OK")

    # ════════════════════════════════════
    # Lifecycle
    # ════════════════════════════════════
    def start(self):
        self.running = True
        self._t_start = self._t_stats = time.monotonic()

        self._rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rx_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._rx_sock.bind(("0.0.0.0", self._esp_port))
        self._rx_sock.settimeout(0.5)

        self._tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self._ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._ctrl_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._ctrl_sock.bind(("0.0.0.0", self._ctrl_port))
        self._ctrl_sock.settimeout(0.1)

        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="ctrl-rx"
        )
        self._thread.start()

        self._ctrl_thread = threading.Thread(
            target=self._control_loop, daemon=True, name="ctrl-cmd"
        )
        self._ctrl_thread.start()
        self._cal_t0 = time.monotonic()
        log.info(f"🚀 Listening :{self._esp_port}  →Unity :{self._uni_port} (IMU-only)")

    def request_stop(self):
        self._stop_flag = True

        log.info(f"🛑 Stopped. frames={self._frame} cksum_err={self._cksum_err}")

    def _control_loop(self):
        """Listen for external commands (e.g. profile switching)."""
        while self.running and not self._stop_flag:
            try:
                data, addr = self._ctrl_sock.recvfrom(1024)
                msg = json.loads(data.decode('utf-8'))
                
                cmd = msg.get("command")
                if cmd == "SET_PROFILE":
                    profile = msg.get("profile")
                    if hasattr(self, "policy_engine"):
                        if self.policy_engine.set_profile(profile):
                            log.info(f"🎮 External Profile Switch: {profile}")
                            # Notify apps about the change
                            self._notify_profile_change(profile)
                
            except socket.timeout:
                continue
            except Exception as e:
                log.warning(f"Control loop error: {e}")
                
    def _notify_profile_change(self, profile: str):
        """Broadcast profile change to all clients."""
        msg = json.dumps({"type": "profile_change", "profile": profile}).encode('utf-8')
        try:
            self._tx_sock.sendto(msg, ("127.0.0.1", self._action_port)) # Web UI
            self._tx_sock.sendto(msg, ("127.0.0.1", 12349)) # Phone
        except:
            pass

    # ════════════════════════════════════
    # Rx Loop
    # ════════════════════════════════════
    def _loop(self):
        while self.running and not self._stop_flag:
            try:
                data, addr = self._rx_sock.recvfrom(256)
                self._esp_last_addr = addr
            except socket.timeout:
                self._check_cal_timeout()
                continue
            except OSError:
                break

            pkts = self._parse(data)
            if not pkts:
                continue

            self._send_dash(pkts)

            for p in pkts:
                if not self._calibrated:
                    self._cal_step(p)
                else:
                    self._process(p)

            now = time.monotonic()
            if (self._calibrated
                    and len(self._ubuf) > 0
                    and now - self._last_uni_send >= self._uni_interval):
                self._flush_unity()
                self._last_uni_send = now

            self._frame += 1
            if now - self._t_stats >= 5:
                self._stats(now)

        self.running = False

    # ════════════════════════════════════
    # Packet Parser (No UWB)
    # ════════════════════════════════════
    def _parse(self, data: bytes):
        n = len(data)

        # v3 format: [AA][4B ts][S1 24B][S2 24B][S3 36B][1B btn][1B cksum][55] = 92 bytes
        if n >= self.PKT_LEN_V3:
            if data[0] != self.HEADER or data[91] != self.FOOTER:
                return []
            ck = 0
            for i in range(1, 90):
                ck ^= data[i]
            if ck != data[90]:
                self._cksum_err += 1
                return []
                
            pen_raw = bool(data[89] & 0x01)
            if pen_raw:
                self._pen_down = True
                self._pen_up_counter = 0
            else:
                if not hasattr(self, '_pen_up_counter'):
                    self._pen_up_counter = 0
                self._pen_up_counter += 1
                if self._pen_up_counter > 4:  # ~50ms hysteresis
                    self._pen_down = False
            ts = _FMT_TS.unpack_from(data, 1)[0]
            
            # v2.5.2: Safely ignore out-of-order UDP packets
            if hasattr(self, '_last_rx_ts'):
                diff = ts - self._last_rx_ts
                # ESP32 millis() wraps 49 days, so only drop if older by less than ~10 sec
                if diff < 0 and diff > -10000:
                    return []
            self._last_rx_ts = ts
            
            pkts = []
            
            # S1 (Forearm, 6-axis)
            f1 = struct.unpack_from("<6f", data, 5)
            # Apply x=-x, y=-y mapping
            a1 = np.array([-f1[0], -f1[1], f1[2]], np.float64)
            g1 = np.array([-f1[3], -f1[4], f1[5]], np.float64)
            pkts.append({
                "sid": "S1", "ts": ts,
                "a": a1, "g": g1,
            })
            
            # S2 (Hand, 6-axis)
            f2 = struct.unpack_from("<6f", data, 29)
            # Apply x=-x, y=-y mapping
            a2 = np.array([-f2[0], -f2[1], f2[2]], np.float64)
            g2 = np.array([-f2[3], -f2[4], f2[5]], np.float64)
            pkts.append({
                "sid": "S2", "ts": ts,
                "a": a2, "g": g2,
            })
            
            # S3 (Finger/ICM-20948, 9-axis)
            f3 = struct.unpack_from("<9f", data, 53)
            # All sensors physically mounted identically: x=-x, y=-y, z=z
            # The direction the bone points is handled by Forward Kinematics
            a3 = np.array([-f3[0], -f3[1], f3[2]], np.float64)
            g3 = np.array([-f3[3], -f3[4], f3[5]], np.float64)
            m3 = np.array([-f3[6], -f3[7], f3[8]], np.float64)
            pkts.append({
                "sid": "S3", "ts": ts,
                "a": a3,
                "g": g3,
                "m": m3,
            })
            

            return pkts
            
        # v2.2: Support both old (79B) and new (80B, with button) formats
        elif n >= self.PKT_LEN_V2:
            # New format: [AA][4B ts][72B sensors][1B button][1B cksum][55]
            if data[0] != self.HEADER or data[79] != self.FOOTER:
                return []
            ck = 0
            for i in range(1, 78):
                ck ^= data[i]
            if ck != data[78]:
                self._cksum_err += 1
                return []
            button_raw = data[77]
            self._pen_down = bool(button_raw & 0x01)
        elif n >= self.PKT_LEN:
            # Old format: [AA][4B ts][72B sensors][1B cksum][55]
            if data[0] != self.HEADER or data[78] != self.FOOTER:
                return []
            ck = 0
            for i in range(1, 77):
                ck ^= data[i]
            if ck != data[77]:
                self._cksum_err += 1
                return []
        else:
            return []

        ts = _FMT_TS.unpack_from(data, 1)[0]

        pkts = []
        for sid, off in zip(_SIDS, _OFFSETS):
            f = _FMT_6F.unpack_from(data, off)
            ok = True
            for v in f:
                if v != v or abs(v) > 1e6:
                    ok = False
                    break
            if not ok:
                continue
            pkts.append({
                "sid": sid, "ts": ts,
                "a": np.array(f[:3], np.float64),
                "g": np.array(f[3:], np.float64),
            })

        if self._debug_count > 0:
            log.info(f"🐛 PARSE n={n} | pkts_sids={[p['sid'] for p in pkts]} | btn={self._pen_down}")
            self._debug_count -= 1

        return pkts

    # ════════════════════════════════════
    # Calibration (identical logic)
    # ════════════════════════════════════
    def _cal_step(self, p):
        sid = p["sid"]
        if sid not in self._cal_buf:
            return
        buf = self._cal_buf[sid]
        buf["a"].append(p["a"].copy())
        buf["g"].append(p["g"].copy())

        if "m" in p:
            self._mw[sid].update_marg(p["g"], p["a"], p["m"], 0.01)
        else:
            self._mw[sid].update_imu(p["g"], p["a"], 0.01)

        self._mw_count[sid] += 1

        c = len(buf["a"])
        if c == 1:
            log.info(f"📐 Cal {sid} … keep still!")
        if all(
            len(self._cal_buf[s]["a"]) >= self._cal_target
            for s in self._cal_buf
        ):
            self._cal_final()

    def _check_cal_timeout(self):
        if self._calibrated or self._cal_t0 is None:
            return
        if time.monotonic() - self._cal_t0 > self._cal_timeout:
            log.warning("⏰ Cal timeout — using partial data")
            self._cal_final()

    def _cal_final(self):
        log.info("─" * 45)
        for sid, buf in self._cal_buf.items():
            if not buf["a"]:
                continue
            aa = np.array(buf["a"])
            gg = np.array(buf["g"])
            ab = np.mean(aa, axis=0)
            # v2.1 fix: use actual gravity direction (robust to sensor tilt)
            grav_mag = np.linalg.norm(ab)
            if grav_mag > 1e-6:
                grav_dir = ab / grav_mag
                ab -= grav_dir * self._gravity
            else:
                ab[2] -= self._gravity  # fallback
            gb = np.mean(gg, axis=0)

            # v2.3: Calibration quality report
            a_std = np.std(aa, axis=0)
            g_std = np.std(gg, axis=0)
            quality = "✅ GOOD" if np.all(a_std < 0.5) and np.all(g_std < 0.02) else "⚠️ NOISY"
            log.info(
                f"  {sid}: A_bias={np.round(ab, 4)}  "
                f"G_bias={np.round(gb, 4)}  (n={len(aa)})  {quality}"
            )
            log.info(
                f"      A_std={np.round(a_std, 4)}  G_std={np.round(g_std, 5)}"
            )

            cfg = self.config.imu_sensors[sid]
            cfg.bias_accel = ab.tolist()
            cfg.bias_gyro = gb.tolist()

        self._calibrated = True
        for sid in self._mw_count:
            if self._mw_count[sid] >= self._mw_warmup_target:
                self._mw_ready[sid] = True

        # v2.3: Save calibrated bias to imu.yaml
        self._save_bias_to_yaml()

        log.info("✅ Calibration done")
        log.info("─" * 45)

    def _save_bias_to_yaml(self):
        """Save calibrated bias values back to imu.yaml.

        v2.3: Persists calibration across restarts.
        Uses safe YAML round-trip to preserve comments structure.
        """
        # Find imu.yaml
        if self._config_dir:
            imu_path = Path(self._config_dir) / "imu.yaml"
        else:
            # Try default paths
            root = Path(__file__).parent.parent.parent
            imu_path = root / "config" / "imu.yaml"

        if not imu_path.exists():
            log.warning(f"⚠️ Cannot save bias: {imu_path} not found")
            return

        try:
            with open(imu_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            sensors = data.get("sensors", {})
            for sid, cfg in self.config.imu_sensors.items():
                if sid in sensors:
                    if "bias" not in sensors[sid]:
                        sensors[sid]["bias"] = {}
                    sensors[sid]["bias"]["accel"] = [
                        round(v, 6) for v in cfg.bias_accel
                    ]
                    sensors[sid]["bias"]["gyro"] = [
                        round(v, 6) for v in cfg.bias_gyro
                    ]

            with open(imu_path, "w", encoding="utf-8") as f:
                f.write("# IMU Configuration v3.2 (bias auto-calibrated)\n\n")
                yaml.dump(data, f, default_flow_style=False,
                         allow_unicode=True, sort_keys=False)

            log.info(f"💾 Bias saved to {imu_path}")
        except Exception as e:
            log.warning(f"⚠️ Failed to save bias: {e}")

    def skip_calibration(self):
        """Skip runtime calibration and use pre-loaded bias from config.

        Used with --load-bias flag when imu.yaml already has calibrated values.
        """
        has_nonzero = False
        for sid, cfg in self.config.imu_sensors.items():
            if any(v != 0 for v in cfg.bias_accel) or \
               any(v != 0 for v in cfg.bias_gyro):
                has_nonzero = True
                log.info(f"  {sid}: A_bias={cfg.bias_accel}  G_bias={cfg.bias_gyro}")

        if not has_nonzero:
            log.warning("⚠️ All bias values are zero — calibration recommended!")
            return False

        self._calibrated = True
        log.info("✅ Using pre-loaded bias (skip calibration)")
        return True

    # ════════════════════════════════════
    # Per-sensor Processing (IMU-only)
    # ════════════════════════════════════
    def _process(self, p):
        sid = p["sid"]
        cfg = self.config.imu_sensors.get(sid)
        if cfg is None:
            return

        accel = p["a"] - np.asarray(cfg.bias_accel)
        gyro = p["g"] - np.asarray(cfg.bias_gyro)

        ts = p["ts"]
        if sid in self._last_ts:
            d = ts - self._last_ts[sid]
            if d < 0:
                d += 2**32
            dt = np.clip(d / 1e6, 0.001, 0.5)
        else:
            dt = 0.01
        self._last_ts[sid] = ts

        mw = self._mw[sid]
        
        if "m" in p:
            quat = mw.update_marg(gyro, accel, p["m"], dt)
        else:
            quat = mw.update_imu(gyro, accel, dt)

        if not self._mw_ready[sid]:
            self._mw_count[sid] += 1
            if self._mw_count[sid] >= self._mw_warmup_target:
                self._mw_ready[sid] = True
                log.info(f"✅ Madgwick {sid} converged ({self._mw_count[sid]})")

        pos = vel = None
        zupt = False
        zaru = False

        # Cache bias-corrected gyro for FK confidence (v2.2)
        self._gyro_cache[sid] = gyro.copy()

        if cfg.role == "finger" and self._mw_ready[sid]:
            R = mw.rotation_matrix()

            # World-frame acceleration (gravity removed)
            np.dot(R, accel, out=self._accel_w)
            accel_w = self._accel_w - self._grav_world

            # Apply biomechanical constraints to acceleration (Pre-update)
            # v2.4: Accel constraints are safe to apply directly to measurement
            self.constraints.constrain_accel(accel_w, dt=dt)

            # v2.2: Writing plane constraint (suppress off-plane accel)
            if self._plane_enabled and self._plane.is_ready():
                self._plane.constrain(accel_w)

            # ESKF fusion update
            res = self.fusion.update(accel_w, gyro, ts, R)
            pos = res["position"]
            vel = res["velocity"]
            zupt = res["zupt_active"]
            zaru = res.get("zaru_active", False)

            # v2.4 fix (C1): Apply state constraints via pseudo-measurements
            # This ensures ESKF covariance P remains consistent
            p_target, v_target, c_res = self.constraints.constrain_state(
                self.fusion.pos, self.fusion.vel, origin=self.fk.origin
            )
            
            if c_res["vel_clamped"]:
                # Soft constraint: measure velocity at limit
                self.fusion.update_velocity_measurement(v_target, vel_noise_std=0.1)
                
            if c_res["workspace_clamped"]:
                # Soft constraint: measure position at boundary
                self.fusion.update_position_measurement(p_target, pos_noise_std=0.1)
            
            # Update local pos/vel refs after potential corrections
            pos = self.fusion.pos
            vel = self.fusion.vel

            # v2.3: Feed current position to writing plane (after ESKF update)
            if self._plane_enabled:
                self._plane.observe(pos)

            # v2.2: Drift Observer — check for stale bias during ZUPT
            if self._drift_enabled:
                if self._drift_observer.observe(pos, zupt):
                    self._drift_observer.apply_correction(
                        self.fusion.ba, self.fusion.vel
                    )

            # ── Pen state edge detection (v2.2) ──
            if self._pen_down and not self._pen_prev:
                # Pen-down edge: start new stroke
                self._stroke_origin[:] = self.fusion.pos
                self._stroke_active = True
                self._stroke_positions = []  # v2.5: reset stroke buffer
                if self._lc_enabled:
                    self._loop_closure.start_stroke()
                log.info(f"✏️ Pen DOWN — stroke start at "
                         f"({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")
            elif not self._pen_down and self._pen_prev:
                # Pen-up edge: end stroke, apply RTO
                if self._stroke_active:
                    self.fusion.update_rto(
                        self._stroke_origin, rto_noise_std=0.05
                    )
                    pos = self.fusion.pos.copy()
                    self._stroke_active = False
                    if self._lc_enabled:
                        self._loop_closure.end_stroke()
                    log.info(f"✏️ Pen UP — RTO applied, pos="
                             f"({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")
                    # v2.5: Send stroke for recognition → action dispatch
                    if self._action_enabled and len(self._stroke_positions) > 5:
                        self._recognize_and_dispatch()
            self._pen_prev = self._pen_down

            # v2.3: Stroke-level loop closure detection
            if self._lc_enabled and self._stroke_active:
                self._loop_closure.track(pos)
                match = self._loop_closure.detect()
                if match is not None:
                    # Apply position correction toward matched origin
                    self.fusion.update_rto(
                        match["origin"],
                        rto_noise_std=self._lc_noise_std
                    )
                    pos = self.fusion.pos.copy()

            fk_pos = None
            
            # FK pseudo-measurement update (v2.1: differential mode)
            # v2.5 fix: allow FK to run even if S1/S2 are missing (use Identity fallback)
            if self.fk_enabled and self._mw_ready.get("S3", False):
                orientations = {}
                for s_id in self.config.imu_sensors:
                    if self._mw_ready.get(s_id, False):
                        orientations[s_id] = self._mw[s_id].rotation_matrix().copy()
                    else:
                        orientations[s_id] = np.eye(3)

                fk_pos = self.fk.compute(orientations)
                # v2.2 fix: use ALL sensors' gyro energy for FK confidence
                total_gyro_energy = sum(
                    np.dot(self._gyro_cache[s], self._gyro_cache[s])
                    for s in self.config.imu_sensors
                )
                self.fusion.update_fk_differential(
                    fk_pos, gyro_energy=total_gyro_energy,
                    fk_noise_std=0.05
                )
                pos = self.fusion.pos.copy()

            # Smooth output
            pos = self._pos_filt(pos)
            
            # Save FK position if we have it
            if fk_pos is not None:
                self._current_fk_pos = fk_pos.copy()

            # v2.5: Collect stroke positions during pen-down
            # Use FK pos to perfectly match JavaScript training data representation
            if self._pen_down:
                if fk_pos is not None:
                    self._stroke_positions.append(fk_pos.copy())
                elif pos is not None:
                    self._stroke_positions.append(pos.copy())

        # Update buffer
        self._ubuf[sid] = {
            "q": quat.tolist(),
            "e": mw.euler_deg().tolist(),
            "p": pos.tolist() if pos is not None else None,
            "v": vel.tolist() if vel is not None else None,
            "z": bool(zupt),
            "zaru": bool(zaru),
            "pen": bool(self._pen_down),
            "fk": self._current_fk_pos.tolist() if getattr(self, '_current_fk_pos', None) is not None else None,
        }

    # ════════════════════════════════════
    # Senders
    # ════════════════════════════════════
    def _flush_unity(self):
        b = self._ubuf
        s1 = b.get("S1", {})
        s2 = b.get("S2", {})
        s3 = b.get("S3", {})

        obj = {
            "t": "f",
            "ms": int(time.time() * 1000),
            "S1q": s1.get("q", [1, 0, 0, 0]),
            "S1e": s1.get("e", [0, 0, 0]),
            "S2q": s2.get("q", [1, 0, 0, 0]),
            "S2e": s2.get("e", [0, 0, 0]),
            "S3q": s3.get("q", [1, 0, 0, 0]),
            "S3e": s3.get("e", [0, 0, 0]),
            "S3p": s3.get("p"),
            "S3v": s3.get("v"),
            "S3z": s3.get("z", False),
            "S3zaru": s3.get("zaru", False),
            "pen": s3.get("pen", False),
            "S3fk": s3.get("fk", None),
        }
        try:
            raw = json.dumps(
                obj, separators=(",", ":"), default=_json_default
            ).encode()
            self._tx_sock.sendto(raw, (self._uni_ip, self._uni_port))
        except Exception as e:
            log.debug(f"Unity tx: {e}")

        self._ubuf.clear()

    def _send_dash(self, pkts):
        try:
            obj = {
                "t": "raw",
                "cal": self._calibrated,
                "s": {
                    p["sid"]: {
                        "a": p["a"].tolist(),
                        "g": p["g"].tolist(),
                    }
                    for p in pkts
                },
            }
            raw = json.dumps(
                obj, separators=(",", ":"), default=_json_default
            ).encode()
            self._tx_sock.sendto(raw, (self._dash_ip, self._dash_port))
        except Exception:
            pass

    def _recognize_and_dispatch(self):
        """v3.0: Run ML recognition and process through Policy Engine and Command Bus.
        
        Implements 'Context-Armed Silent Air Macro OS' logic:
        ML Label -> Policy Engine (Validation/Lane) -> Command Bus (Dispatch)
        """
        if not self._macro_os_enabled:
            return

        try:
            # Lazy-load ML engine
            if not hasattr(self, '_ml_engine'):
                from tools.ml_engine import MLEngine
                self._ml_engine = MLEngine()
                log.info("🧠 ML Engine loaded for Macro OS")

            stroke_data = np.array(self._stroke_positions)
            predictions = self._ml_engine.predict(stroke_data, top_n=1)

            if not predictions:
                log.debug("🤖 No prediction (insufficient data)")
                return

            label = predictions[0].get("label", "").upper()
            confidence = predictions[0].get("confidence", 0.0)
            
            # 1. Pass through Policy Engine
            validation = self.policy_engine.validate_action(label, confidence)
            
            if validation:
                lane = validation.get("lane", "REFLEX")
                profile = validation.get("profile", "GLOBAL")
                
                log.info(f"🎯 Policy Match: '{label}' ({confidence:.1%}) | Lane: {lane} | Profile: {profile}")

                # 2. Dispatch via Command Bus
                self.command_bus.dispatch(label, confidence, context=profile)
                
                # 3. OLED Visual Feedback (if available)
                if hasattr(self, '_esp_last_addr'):
                    # Success feedback: Label(Conf)
                    oled_msg = f"{label},{confidence*100:.1f}".encode('utf-8')
                    self._tx_sock.sendto(oled_msg, (self._esp_last_addr[0], 5555))
            else:
                log.info(f"🛑 Policy REJECT: '{label}' ({confidence:.1%}) in {self.policy_engine.active_profile}")
                # Optional: Send 'REJECT' to OLED if confidence was reasonable but policy failed
                if confidence > 0.4 and hasattr(self, '_esp_last_addr'):
                    oled_msg = f"REJECT,{confidence*100:.1f}".encode('utf-8')
                    self._tx_sock.sendto(oled_msg, (self._esp_last_addr[0], 5555))

        except Exception as e:
            log.warning(f"Macro OS Recognition error: {e}")

    def _stats(self, now):
        self._t_stats = now
        el = now - self._t_start
        fps = self._frame / el if el > 0 else 0
        fs = self.fusion
        p_str = f"({fs.pos[0]:.3f},{fs.pos[1]:.3f},{fs.pos[2]:.3f})"
        v_str = f"({fs.vel[0]:.3f},{fs.vel[1]:.3f},{fs.vel[2]:.3f})"
        st = "RUN" if self._calibrated else "CAL"
        plane_str = "✅" if self._plane.is_ready() else "⏳"
        drift_n = self._drift_observer.n_corrections if self._drift_enabled else 0
        log.info(
            f"📊 [{st}] f={self._frame} fps={fps:.0f} pos={p_str} vel={v_str} "
            f"zupt={fs.n_zupt} zaru={fs.n_zaru} decay={fs.n_decay} "
            f"rto={fs.n_rto} pen={'✏️' if self._pen_down else '🔵'} "
            f"plane={plane_str} drift_fix={drift_n} "
            f"ck_err={self._cksum_err}"
        )
