from __future__ import annotations

import argparse
import os
import time
from typing import Dict, Optional

import numpy as np

from ..config import load_config, ProjectConfig
from ..filters.vqf_wrapper import VQFWrapper
from ..filters.mahony import MahonyAHRS, MahonyParams
from ..filters.madgwick import MadgwickAHRS, MadgwickParams
from ..quaternion import (
    q_mul,
    q_normalize,
    q_conj,
    q_from_axis_angle,
    q_to_euler_zyx,
)
from ..sensors.imu_base import IMUDevice, IMUSample
from ..sensors.mpu6050_i2c import MPU6050I2C, MPU6050Settings
from ..sensors.icm20948_i2c import ICM20948I2C, ICM20948Settings
from ..kinematics.chain_fk import ArmKinematics3Link
from ..kinematics.constraints import apply_elbow_hinge, apply_wrist_swing_limit
from ..kinematics.plane import PlaneProjector
from ..kinematics.stroke import StrokeDetector
from ..utils.axis import remap_vec3
from ..utils.logger import RunLogger



def _screen_xy_from_quat(q_w: np.ndarray, cfg) -> tuple[float, float]:
    # debug cursor: roll/pitch -> screen
    yaw, pitch, roll = q_to_euler_zyx(q_w)
    x = cfg.width * 0.5 + (roll + cfg.roll_offset) * cfg.roll_scale
    y = cfg.height * 0.5 + (pitch + cfg.pitch_offset) * cfg.pitch_scale
    if cfg.invert_x:
        x = cfg.width - x
    if cfg.invert_y:
        y = cfg.height - y
    return float(x), float(y)


def _pick_attitude_estimator(cfg: ProjectConfig, has_mag: bool) -> str:
    sel = (cfg.filters.attitude_estimator or "auto").lower().strip()
    if sel in ("auto", "vqf", "mahony", "madgwick"):
        if sel == "vqf":
            return "vqf"
        if sel == "mahony":
            return "mahony"
        if sel == "madgwick":
            return "madgwick"
        # auto
        try:
            # VQFWrapper will throw if vqf is not installed
            _ = VQFWrapper(sample_rate_hz=cfg.sample_rate_hz, use_fast=True)
            return "vqf"
        except Exception:
            return "mahony"
    return "mahony"


def _mean_quaternion(quats: np.ndarray) -> np.ndarray:
    """
    Average quaternion via eigen method.
    quats: (N,4), each normalized
    """
    Q = np.asarray(quats, dtype=float).reshape(-1, 4)
    # make sign consistent (avoid antipodal cancellation)
    for i in range(1, Q.shape[0]):
        if np.dot(Q[0], Q[i]) < 0:
            Q[i] *= -1.0
    A = Q.T @ Q
    w, v = np.linalg.eigh(A)
    q = v[:, np.argmax(w)]
    return q_normalize(q)


def _apply_heading_anchor(
    q_target: np.ndarray,
    q_src: np.ndarray,
    strength: float,
) -> np.ndarray:
    """
    Pull q_src's yaw towards q_target's yaw (world Z-axis), by 'strength'.
    q_target, q_src: world<-frame
    """
    yaw_t, _, _ = q_to_euler_zyx(q_target)
    yaw_s, _, _ = q_to_euler_zyx(q_src)
    # wrap to [-pi, pi]
    dy = (yaw_t - yaw_s + np.pi) % (2.0 * np.pi) - np.pi
    q_corr = q_from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=float), dy * float(strength))
    return q_normalize(q_mul(q_corr, q_src))


def _read_and_remap(dev: IMUDevice, axis_map: np.ndarray, axis_sign: np.ndarray) -> IMUSample:
    s = dev.read()
    acc = remap_vec3(s.acc_m_s2, axis_map, axis_sign)
    gyr = remap_vec3(s.gyr_rad_s, axis_map, axis_sign)
    mag = None
    if s.mag_uT is not None:
        mag = remap_vec3(s.mag_uT, axis_map, axis_sign)
    return IMUSample(t=s.t, acc_m_s2=acc, gyr_rad_s=gyr, mag_uT=mag, temp_C=s.temp_C)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/config.yaml")
    ap.add_argument("--duration", type=float, default=0.0, help="0=run forever")
    ap.add_argument("--no-gui", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # device creation (네 구성: has_mag==False -> MPU6050, has_mag==True -> ICM20948)
    devices: Dict[str, IMUDevice] = {}
    for name, sc in cfg.sensors.items():
        if sc.has_mag:
            devices[name] = ICM20948I2C(
                name=name,
                bus=sc.i2c_bus,
                addr=sc.i2c_addr,
                settings=ICM20948Settings(sample_rate_hz=cfg.sample_rate_hz, enable_mag=True),
            )
        else:
            devices[name] = MPU6050I2C(
                name=name,
                bus=sc.i2c_bus,
                addr=sc.i2c_addr,
                settings=MPU6050Settings(sample_rate_hz=cfg.sample_rate_hz),
            )

    # attitude estimators
    estimators: Dict[str, object] = {}
    estimator_kind: Dict[str, str] = {}
    for name, sc in cfg.sensors.items():
        kind = _pick_attitude_estimator(cfg, sc.has_mag)
        estimator_kind[name] = kind
        if kind == "vqf":
            estimators[name] = VQFWrapper(sample_rate_hz=cfg.sample_rate_hz, use_fast=True)
        elif kind == "madgwick":
            estimators[name] = MadgwickAHRS(
                sample_rate_hz=cfg.sample_rate_hz,
                params=MadgwickParams(beta=cfg.filters.madgwick.beta),
            )
        else:
            estimators[name] = MahonyAHRS(
                sample_rate_hz=cfg.sample_rate_hz,
                params=MahonyParams(kp=cfg.filters.mahony.kp, ki=cfg.filters.mahony.ki),
            )

    # runtime mount correction
    mount_runtime: Dict[str, np.ndarray] = {n: sc.mount_quat_wxyz.copy() for n, sc in cfg.sensors.items()}

    # warm-up + optional mount calibration
    yaw_offset: Dict[str, float] = {n: 0.0 for n in cfg.sensors.keys()}
    if cfg.calibration.enable_mount_cal:
        t_end = time.time() + float(cfg.calibration.mount_cal_seconds)
        buf: Dict[str, list[np.ndarray]] = {n: [] for n in cfg.sensors.keys()}
        while time.time() < t_end:
            for name, sc in cfg.sensors.items():
                s = _read_and_remap(devices[name], sc.axis_map, sc.axis_sign)
                est = estimators[name]
                kind = estimator_kind[name]

                if kind == "vqf":
                    out = est.update(s.gyr_rad_s, s.acc_m_s2, s.mag_uT)  # type: ignore
                    q_ws = out.quat9D_wxyz if (sc.has_mag and out.quat9D_wxyz is not None) else out.quat6D_wxyz
                else:
                    # mahony/madgwick: update(gyr, acc, mag)
                    q_ws = est.update(s.gyr_rad_s, s.acc_m_s2, s.mag_uT)  # type: ignore

                q_ws = q_normalize(q_ws)
                buf[name].append(q_ws)

            time.sleep(0.0)

        # compute mount correction so that reference pose becomes identity in segment frame
        q_ref_seg: Dict[str, np.ndarray] = {}
        for name, sc in cfg.sensors.items():
            q_mean = _mean_quaternion(np.stack(buf[name], axis=0))
            # apply mount from config first -> reference in segment frame
            q_mean_seg = q_mul(q_mean, mount_runtime[name])
            q_ref_seg[name] = q_mean_seg

            # mount correction: want q_mean_seg -> identity => multiply by conj(q_mean)
            mount_runtime[name] = q_mul(q_conj(q_mean), mount_runtime[name])
            mount_runtime[name] = q_normalize(mount_runtime[name])

        # heading anchor yaw offsets (segment-frame quats)
        if cfg.heading_anchor.enable and "S3" in q_ref_seg:
            yaw_hand, _, _ = q_to_euler_zyx(q_ref_seg["S3"])
            for name in ("S1", "S2"):
                if name in q_ref_seg:
                    yaw_seg, _, _ = q_to_euler_zyx(q_ref_seg[name])
                    yaw_offset[name] = float(yaw_seg - yaw_hand)

    # components
    fk = ArmKinematics3Link(
        upper_arm_len=cfg.arm_model.upper_arm_len,
        forearm_len=cfg.arm_model.forearm_len,
        hand_len=cfg.arm_model.hand_len,
        seg_axis=cfg.arm_model.seg_axis,
    )
    projector = PlaneProjector(cfg.writing_plane.normal_w, cfg.writing_plane.d, cfg.writing_plane.u_w, cfg.writing_plane.v_w)
    stroke = StrokeDetector(
        speed_down_th=cfg.stroke.speed_down_th,
        speed_up_th=cfg.stroke.speed_up_th,
        min_down_time_s=cfg.stroke.min_down_time_s,
    )

    viewer = None
    if (not args.no_gui) and cfg.screen_mapping.enable:
        # lazy import: headless 환경에서도 import 에러 없이 실행 가능
        from .visualize_pygame import PygameViewer
        viewer = PygameViewer(width=cfg.screen_mapping.width, height=cfg.screen_mapping.height)

    logger: Optional[RunLogger] = None
    if cfg.logging.save_csv:
        logger = RunLogger(out_dir=cfg.logging.out_dir, prefix="run")

    # main loop
    t0 = time.time()
    q_w_seg: Dict[str, np.ndarray] = {n: np.array([1.0, 0.0, 0.0, 0.0], dtype=float) for n in cfg.sensors.keys()}
    imu_dbg: Dict[str, Dict[str, np.ndarray]] = {n: {"acc": np.zeros(3), "gyr": np.zeros(3)} for n in cfg.sensors.keys()}

    try:
        while True:
            now = time.time()
            if args.duration > 0.0 and (now - t0) > args.duration:
                break

            # read + attitude
            for name, sc in cfg.sensors.items():
                s = _read_and_remap(devices[name], sc.axis_map, sc.axis_sign)
                imu_dbg[name]["acc"] = s.acc_m_s2.copy()
                imu_dbg[name]["gyr"] = s.gyr_rad_s.copy()

                est = estimators[name]
                kind = estimator_kind[name]
                if kind == "vqf":
                    out = est.update(s.gyr_rad_s, s.acc_m_s2, s.mag_uT)  # type: ignore
                    q_ws = out.quat9D_wxyz if (sc.has_mag and out.quat9D_wxyz is not None) else out.quat6D_wxyz
                else:
                    q_ws = est.update(s.gyr_rad_s, s.acc_m_s2, s.mag_uT)  # type: ignore

                q_ws = q_normalize(q_ws)
                q_w_seg[name] = q_normalize(q_mul(q_ws, mount_runtime[name]))

            # need S1,S2,S3
            if not all(k in q_w_seg for k in ("S1", "S2", "S3")):
                raise RuntimeError("config.sensors must include S1,S2,S3")

            # 실물 체인 매핑(너가 말한 그대로)
            # - S2: 상완(Upper)
            # - S1: 전완(Fore)
            q_upper = q_w_seg["S2"]
            q_fore = q_w_seg["S1"]
            q_hand = q_w_seg["S3"]

            # constraints
            if cfg.joint_constraints.enable:
                q_fore = apply_elbow_hinge(
                    q_upper,
                    q_fore,
                    elbow_axis_upper=cfg.joint_constraints.elbow_axis,
                    strength=cfg.joint_constraints.elbow_strength,
                )
                q_hand = apply_wrist_swing_limit(
                    q_fore,
                    q_hand,
                    twist_axis_fore=cfg.arm_model.seg_axis,
                    swing_limit_deg=cfg.joint_constraints.wrist_swing_limit_deg,
                    strength=cfg.joint_constraints.wrist_strength,
                )

            # heading anchor (S3 heading -> pull S1,S2 yaw)
            if cfg.heading_anchor.enable:
                strength = float(cfg.heading_anchor.strength)
                # apply offsets from calibration so initial yaw relationship is kept
                if strength > 0.0:
                    # build a target quaternion with yaw offset by left-multiplying world yaw rot
                    yaw_hand, _, _ = q_to_euler_zyx(q_hand)
                    # 상완(S2) + 전완(S1) 둘 다를 손(S3) heading에 느리게 고정
                    for nm in ("S2", "S1"):
                        if nm in q_w_seg:
                            target_yaw = yaw_hand + yaw_offset.get(nm, 0.0)
                            q_target = q_mul(
                                q_from_axis_angle(np.array([0.0, 0.0, 1.0]), target_yaw - yaw_hand),
                                q_hand,
                            )
                            if nm == "S2":
                                q_upper = _apply_heading_anchor(q_target, q_upper, strength)
                            else:
                                q_fore = _apply_heading_anchor(q_target, q_fore, strength)

            # fk + plane + stroke
            pose = fk.forward(q_upper, q_fore, q_hand)
            _, uv = projector.project_point(pose.tip_w)
            down, speed = stroke.update(now, uv, pose.tip_w)

            # debug cursor from hand quat
            cursor_xy = _screen_xy_from_quat(q_hand, cfg.screen_mapping)

            if viewer is not None:
                viewer.draw(cursor_xy=cursor_xy, down=down)
                if not viewer.is_running:
                    break

            if logger is not None:
                logger.write_frame(t=now, imu=imu_dbg, tip_w=pose.tip_w, uv=uv, down=down, speed=speed)

            # pacing
            time.sleep(max(0.0, (1.0 / cfg.sample_rate_hz) * 0.35))

    finally:
        for d in devices.values():
            try:
                d.close()
            except Exception:
                pass
        if logger is not None:
            logger.close()
            print(f"saved: {logger.csv_path}")
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
