"""
main.py — IMU-Only AirWriting Entry Point (v2.3)
  python main.py                   # 실행 (auto calibration)
  python main.py --load-bias       # 이전 캘리브 사용
  python main.py --calibrate-only  # 캘리브만 실행 후 저장
  python main.py --config-check    # 설정 검증만
  python main.py --log-level DEBUG # 디버그
"""
import argparse, logging, signal, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from airwriting_imu.core.config_loader import ConfigLoader
from airwriting_imu.core.controller import AirWritingIMUController


def main():
    ap = argparse.ArgumentParser(description="AirWriting IMU-Only v2.3")
    ap.add_argument("--config-check", action="store_true",
                    help="Validate config and exit")
    ap.add_argument("--config-dir", default=None,
                    help="Config directory path")
    ap.add_argument("--load-bias", action="store_true",
                    help="Skip calibration, use bias from imu.yaml")
    ap.add_argument("--calibrate-only", action="store_true",
                    help="Run calibration, save bias, then exit")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler("airwriting_imu.log", encoding="utf-8")],
    )
    log = logging.getLogger("Main")
    log.info("=" * 55)
    log.info("  AirWriting IMU-Only v2.3")
    log.info("=" * 55)

    # ── Config ──
    try:
        cfg = ConfigLoader(args.config_dir).load_all()
    except Exception as e:
        log.error(f"Config error:\n{e}")
        sys.exit(1)

    if args.config_check:
        log.info("✅ Configuration OK")
        sys.exit(0)

    # ── Controller ──
    ctrl = None
    try:
        ctrl = AirWritingIMUController(cfg)

        # --load-bias: skip runtime calibration
        if args.load_bias:
            if not ctrl.skip_calibration():
                log.error("❌ Cannot skip calibration (no bias data)")
                sys.exit(1)

        signal.signal(signal.SIGINT, lambda *_: ctrl.request_stop())
        signal.signal(signal.SIGTERM, lambda *_: ctrl.request_stop())
        ctrl.start()

        if args.calibrate_only:
            log.info("📐 Calibrate-only mode: waiting for calibration...")
            while ctrl.running and not ctrl._calibrated:
                time.sleep(0.1)
            if ctrl._calibrated:
                log.info("✅ Calibration complete. Bias saved to imu.yaml.")
            else:
                log.warning("⚠️ Calibration did not complete.")
            ctrl.stop()
            sys.exit(0)

        log.info("✅ Running (IMU-only mode).  Ctrl+C to stop.")
        while ctrl.running:
            time.sleep(0.25)
    except Exception as e:
        log.error(f"Runtime error: {e}", exc_info=True)
    finally:
        if ctrl:
            ctrl.stop()


if __name__ == "__main__":
    main()
