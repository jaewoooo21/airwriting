from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import yaml


def _np3(x: List[float]) -> np.ndarray:
    a = np.asarray(x, dtype=float).reshape(3)
    return a


def _np3i(x: List[int]) -> np.ndarray:
    a = np.asarray(x, dtype=int).reshape(3)
    return a


@dataclass
class SensorConfig:
    name: str
    i2c_bus: int
    i2c_addr: int
    has_mag: bool
    axis_map: np.ndarray  # (3,) int
    axis_sign: np.ndarray  # (3,) int
    mount_quat_wxyz: np.ndarray  # (4,)


@dataclass
class ArmModelConfig:
    upper_arm_len: float
    forearm_len: float
    hand_len: float
    seg_axis: np.ndarray  # (3,) float


@dataclass
class JointConstraintConfig:
    enable: bool
    elbow_axis: np.ndarray  # (3,) float (forearm local axis around which elbow flexes)
    elbow_strength: float
    wrist_swing_limit_deg: float
    wrist_strength: float


@dataclass
class PlaneConfig:
    enable: bool
    normal_w: np.ndarray  # (3,) float
    d: float
    u_w: np.ndarray  # (3,) float
    v_w: np.ndarray  # (3,) float


@dataclass
class StrokeConfig:
    speed_down_th: float
    speed_up_th: float
    min_down_time_s: float


@dataclass
class ScreenMappingConfig:
    enable: bool
    width: int
    height: int
    roll_scale: float
    pitch_scale: float
    roll_offset: float
    pitch_offset: float
    invert_x: bool
    invert_y: bool


@dataclass
class LoggingConfig:
    save_csv: bool
    out_dir: str


@dataclass
class MahonyConfig:
    kp: float = 0.8
    ki: float = 0.02


@dataclass
class MadgwickConfig:
    beta: float = 0.08


@dataclass
class FiltersConfig:
    attitude_estimator: str = "auto"  # auto | vqf | mahony | madgwick
    mahony: MahonyConfig = field(default_factory=MahonyConfig)
    madgwick: MadgwickConfig = field(default_factory=MadgwickConfig)


@dataclass
class CalibrationConfig:
    enable_mount_cal: bool = True
    mount_cal_seconds: float = 2.0


@dataclass
class HeadingAnchorConfig:
    enable: bool = True
    strength: float = 0.03


@dataclass
class ProjectConfig:
    sample_rate_hz: float
    sensors: Dict[str, SensorConfig]
    arm_model: ArmModelConfig
    joint_constraints: JointConstraintConfig
    writing_plane: PlaneConfig
    stroke: StrokeConfig
    screen_mapping: ScreenMappingConfig
    logging: LoggingConfig
    filters: FiltersConfig = field(default_factory=FiltersConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    heading_anchor: HeadingAnchorConfig = field(default_factory=HeadingAnchorConfig)


def load_config(path: str) -> ProjectConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    sample_rate_hz = float(raw.get("sample_rate_hz", 100.0))

    sensors: Dict[str, SensorConfig] = {}
    raw_sensors = raw.get("sensors", {})
    if not isinstance(raw_sensors, dict) or len(raw_sensors) == 0:
        raise ValueError("config.yaml: sensors block is missing/empty")

    for name, s in raw_sensors.items():
        sensors[name] = SensorConfig(
            name=str(name),
            i2c_bus=int(s["i2c_bus"]),
            i2c_addr=int(s["i2c_addr"], 0) if isinstance(s["i2c_addr"], str) else int(s["i2c_addr"]),
            has_mag=bool(s.get("has_mag", False)),
            axis_map=_np3i(s.get("axis_map", [0, 1, 2])),
            axis_sign=_np3i(s.get("axis_sign", [1, 1, 1])),
            mount_quat_wxyz=np.asarray(s.get("mount_quat_wxyz", [1, 0, 0, 0]), dtype=float).reshape(4),
        )

    am = raw.get("arm_model", {})
    arm_model = ArmModelConfig(
        upper_arm_len=float(am.get("upper_arm_len", 0.30)),
        forearm_len=float(am.get("forearm_len", 0.26)),
        hand_len=float(am.get("hand_len", 0.14)),
        seg_axis=_np3(am.get("seg_axis", [1.0, 0.0, 0.0])),
    )

    jc = raw.get("joint_constraints", {})
    joint_constraints = JointConstraintConfig(
        enable=bool(jc.get("enable", True)),
        elbow_axis=_np3(jc.get("elbow_axis", [0.0, 1.0, 0.0])),
        elbow_strength=float(jc.get("elbow_strength", 0.15)),
        wrist_swing_limit_deg=float(jc.get("wrist_swing_limit_deg", 70.0)),
        wrist_strength=float(jc.get("wrist_strength", 0.10)),
    )

    wp = raw.get("writing_plane", {})
    writing_plane = PlaneConfig(
        enable=bool(wp.get("enable", True)),
        normal_w=_np3(wp.get("normal_w", [0.0, 1.0, 0.0])),
        d=float(wp.get("d", 0.0)),
        u_w=_np3(wp.get("u_w", [1.0, 0.0, 0.0])),
        v_w=_np3(wp.get("v_w", [0.0, 0.0, 1.0])),
    )

    st = raw.get("stroke", {})
    stroke = StrokeConfig(
        speed_down_th=float(st.get("speed_down_th", 0.03)),
        speed_up_th=float(st.get("speed_up_th", 0.015)),
        min_down_time_s=float(st.get("min_down_time_s", 0.05)),
    )

    sm = raw.get("screen_mapping", {})
    screen_mapping = ScreenMappingConfig(
        enable=bool(sm.get("enable", True)),
        width=int(sm.get("width", 1280)),
        height=int(sm.get("height", 720)),
        roll_scale=float(sm.get("roll_scale", 420.0)),
        pitch_scale=float(sm.get("pitch_scale", 420.0)),
        roll_offset=float(sm.get("roll_offset", 0.0)),
        pitch_offset=float(sm.get("pitch_offset", 0.0)),
        invert_x=bool(sm.get("invert_x", False)),
        invert_y=bool(sm.get("invert_y", True)),
    )

    lg = raw.get("logging", {})
    logging = LoggingConfig(
        save_csv=bool(lg.get("save_csv", True)),
        out_dir=str(lg.get("out_dir", "runs")),
    )

    # optional blocks
    fr = raw.get("filters", {}) or {}
    filters = FiltersConfig(
        attitude_estimator=str(fr.get("attitude_estimator", "auto")),
        mahony=MahonyConfig(
            kp=float((fr.get("mahony", {}) or {}).get("kp", 0.8)),
            ki=float((fr.get("mahony", {}) or {}).get("ki", 0.02)),
        ),
        madgwick=MadgwickConfig(
            beta=float((fr.get("madgwick", {}) or {}).get("beta", 0.08)),
        ),
    )

    cal = raw.get("calibration", {}) or {}
    calibration = CalibrationConfig(
        enable_mount_cal=bool(cal.get("enable_mount_cal", True)),
        mount_cal_seconds=float(cal.get("mount_cal_seconds", 2.0)),
    )

    ha = raw.get("heading_anchor", {}) or {}
    heading_anchor = HeadingAnchorConfig(
        enable=bool(ha.get("enable", True)),
        strength=float(ha.get("strength", 0.03)),
    )

    return ProjectConfig(
        sample_rate_hz=sample_rate_hz,
        sensors=sensors,
        arm_model=arm_model,
        joint_constraints=joint_constraints,
        writing_plane=writing_plane,
        stroke=stroke,
        screen_mapping=screen_mapping,
        logging=logging,
        filters=filters,
        calibration=calibration,
        heading_anchor=heading_anchor,
    )
