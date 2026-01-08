from __future__ import annotations

import argparse
import time

from airwriting_advanced.config import load_config
from airwriting_advanced.sensors.mpu9250_i2c import MPU9250I2C, MPU9250Settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/config/config.yaml")
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args()

    cfg = load_config(args.config)

    imus = {}
    try:
        for name, s in cfg.sensors.items():
            settings = MPU9250Settings(sample_rate_hz=cfg.sample_rate_hz, enable_mag=bool(s.has_mag))
            imus[name] = MPU9250I2C(name=name, bus=s.i2c_bus, addr=s.i2c_addr, settings=settings)

        print("지금부터 가만히 있어. 바이어스 평균 잡을거임.")
        time.sleep(1.0)

        for name, imu in imus.items():
            print(f"[{name}] calibrating {args.seconds:.1f}s ...")
            imu.calibrate_bias(seconds=args.seconds)
            print(f"[{name}] done")

        print("완료. 이 바이어스는 현재 코드에서는 디바이스 객체 내부에만 적용됨(즉시효과).")
        print("진짜 영구적으로 쓰려면: 여기서 평균값 출력해서 config/config.yaml에 offsets로 저장하도록 바꿔도 됨.")
    finally:
        for imu in imus.values():
            imu.close()


if __name__ == "__main__":
    main()
