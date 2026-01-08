from __future__ import annotations

import argparse
from typing import Dict

import numpy as np

from ..config import load_config
from ..filters.vqf_wrapper import VQFWrapper
from ..filters.mahony import MahonyAHRS, MahonyParams
from ..filters.madgwick import MadgwickAHRS, MadgwickParams
from ..quaternion import q_mul, q_to_euler_zyx, q_normalize
from ..kinematics.chain_fk import ArmKinematics3Link
from ..kinematics.constraints import apply_elbow_hinge, apply_wrist_swing_limit
from ..kinematics.plane import PlaneProjector
from ..kinematics.stroke import StrokeDetector
from ..utils.logger import RunLogger
from ..sensors.csv_reader import CSVReplay
from ..utils.axis import remap_vec3



def _screen_xy_from_quat(q_w: np.ndarray, cfg) -> tuple[float, float]:
    yaw, pitch, roll = q_to_euler_zyx(q_w)
    x = cfg.width * 0.5 + (roll + cfg.roll_offset) * cfg.roll_scale
    y = cfg.height * 0.5 + (pitch + cfg.pitch_offset) * cfg.pitch_scale
    if cfg.invert_x:
        x = cfg.width - x
    if cfg.invert_y:
        y = cfg.height - y
    return float(x), float(y)


def _pick_estimator_kind(attitude_estimator: str) -> str:
    sel = (attitude_estimator or "auto").lower().strip()
    if sel in ("vqf", "mahony", "madgwick", "auto"):
        return sel
    return "auto"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/config.yaml")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--no-view", action="store_true")
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    sensor_names = list(cfg.sensors.keys())

    replay = CSVReplay(args.csv, sensor_names=sensor_names)

    # choose estimator per sensor (auto: try VQF -> fallback Mahony)
    filters: Dict[str, object] = {}
    kind_map: Dict[str, str] = {}
    sel = _pick_estimator_kind(cfg.filters.attitude_estimator)
    for name in sensor_names:
        if sel == "vqf" or sel == "auto":
            try:
                filters[name] = VQFWrapper(cfg.sample_rate_hz)
                kind_map[name] = "vqf"
                continue
            except Exception:
                pass
        if sel == "madgwick":
            filters[name] = MadgwickAHRS(cfg.sample_rate_hz, params=MadgwickParams(beta=cfg.filters.madgwick.beta))
            kind_map[name] = "madgwick"
        else:
            filters[name] = MahonyAHRS(cfg.sample_rate_hz, params=MahonyParams(kp=cfg.filters.mahony.kp, ki=cfg.filters.mahony.ki))
            kind_map[name] = "mahony"
    kin = ArmKinematics3Link(cfg.arm_model.upper_arm_len, cfg.arm_model.forearm_len, cfg.arm_model.hand_len, cfg.arm_model.seg_axis)
    projector = PlaneProjector(cfg.writing_plane.normal_w, cfg.writing_plane.d, cfg.writing_plane.u_w, cfg.writing_plane.v_w)
    strokes = StrokeDetector(cfg.stroke.speed_down_th, cfg.stroke.speed_up_th, cfg.stroke.min_down_time_s)

    viewer = None
    if cfg.screen_mapping.enable and (not args.no_view):
        # lazy import: pygame 없는 환경에서도 --no-view면 실행 가능
        from .visualize_pygame import PygameViewer
        viewer = PygameViewer(cfg.screen_mapping.width, cfg.screen_mapping.height, title="Replay Debug")

    out_dir = args.out_dir if args.out_dir is not None else cfg.logging.out_dir
    logger = RunLogger(out_dir, prefix="replay")

    try:
        for frame in replay.frames():
            q_w_seg: Dict[str, np.ndarray] = {}
            imu_vectors: Dict[str, Dict[str, np.ndarray]] = {}

            for name, samp in frame.samples.items():
                scfg = cfg.sensors[name]
                acc = remap_vec3(samp.acc_m_s2, scfg.axis_map, scfg.axis_sign)
                gyr = remap_vec3(samp.gyr_rad_s, scfg.axis_map, scfg.axis_sign)
                mag = remap_vec3(samp.mag_uT, scfg.axis_map, scfg.axis_sign) if (samp.mag_uT is not None and scfg.has_mag) else None
                imu_vectors[name] = {"acc": acc, "gyr": gyr}

                f = filters[name]
                kind = kind_map[name]
                if kind == "vqf":
                    out = f.update(gyr, acc, mag)  # type: ignore
                    q_ws = out.quat9D_wxyz if (scfg.has_mag and out.quat9D_wxyz is not None) else out.quat6D_wxyz
                else:
                    q_ws = f.update(gyr, acc, mag)  # type: ignore
                q_w_seg[name] = q_normalize(q_mul(q_ws, scfg.mount_quat_wxyz))

            # 실물 체인 매핑(너가 말한 그대로)
            # - S2: 상완(Upper)
            # - S1: 전완(Fore)
            q_u = q_w_seg["S2"]
            q_f = q_w_seg["S1"]
            q_h = q_w_seg["S3"]

            if cfg.joint_constraints.enable:
                q_f = apply_elbow_hinge(q_u, q_f, cfg.joint_constraints.elbow_axis, cfg.joint_constraints.elbow_strength)
                q_h = apply_wrist_swing_limit(q_f, q_h, cfg.arm_model.seg_axis,
                                              cfg.joint_constraints.wrist_swing_limit_deg, cfg.joint_constraints.wrist_strength)

            pose = kin.forward(q_u, q_f, q_h)
            _, uv = projector.project_point(pose.tip_w)
            down, speed = strokes.update(frame.t, uv, pose.tip_w)

            if viewer is not None:
                viewer.draw(_screen_xy_from_quat(q_h, cfg.screen_mapping), down)
                if not viewer.is_running:
                    break

            logger.write_frame(frame.t, imu_vectors, pose.tip_w, uv, down, speed)

    finally:
        logger.close()
        if viewer is not None:
            viewer.close()
        print(f"saved: {logger.csv_path}")


if __name__ == "__main__":
    main()
