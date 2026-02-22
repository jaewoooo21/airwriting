"""설정 로더 — IMU-Only (앵커 없음)"""
import yaml, logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class IMUSensorConfig:
    role: str
    sensor_type: str
    bus: int
    address: int
    sample_rate_hz: int = 100
    has_magnetometer: bool = False
    bias_accel: List[float] = field(default_factory=lambda: [0, 0, 0])
    bias_gyro: List[float] = field(default_factory=lambda: [0, 0, 0])


@dataclass
class SkeletonJoint:
    joint_name: str
    sensor_id: str
    bone_length_m: float
    parent: Optional[str] = None


@dataclass
class SystemConfig:
    imu_sensors: Dict[str, IMUSensorConfig] = field(default_factory=dict)
    skeleton: List[SkeletonJoint] = field(default_factory=list)
    skeleton_raw: List[dict] = field(default_factory=list)
    network: dict = field(default_factory=dict)
    fusion: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    madgwick: dict = field(default_factory=dict)
    unity_send: dict = field(default_factory=dict)


class ConfigLoader:
    _REQ = ("imu.yaml", "system.yaml")  # No anchors.yaml required

    def __init__(self, config_dir=None):
        if config_dir is None:
            root = Path(__file__).parent.parent.parent
            for c in [root / "config", root / "config" / "products"]:
                if (c / "imu.yaml").exists():
                    config_dir = c
                    break
            if config_dir is None:
                raise FileNotFoundError("config/ not found")
        self.dir = Path(config_dir)
        miss = [f for f in self._REQ if not (self.dir / f).exists()]
        if miss:
            raise FileNotFoundError(f"Missing: {miss}")

    def _yaml(self, name):
        with open(self.dir / name, encoding="utf-8") as f:
            d = yaml.safe_load(f)
        if d is None:
            raise ValueError(f"Empty: {name}")
        return d

    def load_all(self) -> SystemConfig:
        imu_raw = self._yaml("imu.yaml")
        sys_raw = self._yaml("system.yaml")

        # ── IMU sensors ──
        sensors = {}
        for name, c in imu_raw.get("sensors", {}).items():
            addr = c.get("address", 0x68)
            if isinstance(addr, str):
                addr = int(addr, 16)
            sensors[name] = IMUSensorConfig(
                role=c["role"],
                sensor_type=c["type"],
                bus=c["bus"],
                address=addr,
                sample_rate_hz=c.get("sample_rate_hz", 100),
                has_magnetometer=c.get("has_magnetometer", False),
                bias_accel=c.get("bias", {}).get("accel", [0, 0, 0]),
                bias_gyro=c.get("bias", {}).get("gyro", [0, 0, 0]),
            )

        # I2C conflict check
        seen = {}
        for n, s in sensors.items():
            k = (s.bus, s.address)
            if k in seen:
                raise ValueError(f"I2C conflict: {n} vs {seen[k]}")
            seen[k] = n

        # ── Skeleton chain ──
        skel = []
        skel_raw = []
        for j in imu_raw.get("skeleton_chain", []):
            sid = j["sensor"]
            if sid not in sensors:
                raise ValueError(f"Skeleton refs unknown sensor '{sid}'")
            skel.append(SkeletonJoint(
                j["joint"], sid, j["bone_length_m"], j.get("parent")))
            skel_raw.append({
                "joint": j["joint"],
                "sensor": sid,
                "bone_length_m": j["bone_length_m"],
                "parent": j.get("parent"),
            })

        # ── Port conflict check ──
        ports = list(sys_raw.get("network", {}).get("ports", {}).values())
        if len(ports) != len(set(ports)):
            raise ValueError(f"Duplicate ports: {ports}")

        cfg = SystemConfig(
            imu_sensors=sensors,
            skeleton=skel,
            skeleton_raw=skel_raw,
            network=sys_raw.get("network", {}),
            fusion=sys_raw.get("fusion", {}),
            calibration=sys_raw.get("calibration", {}),
            madgwick=sys_raw.get("madgwick", {}),
            unity_send=sys_raw.get("unity_send", {}),
        )
        # v2.3: attach config dir for bias saving
        cfg._config_dir = str(self.dir)
        log.info(f"✅ Config: {len(sensors)} sensors, "
                 f"{len(skel)} skeleton joints (IMU-only)")
        return cfg
