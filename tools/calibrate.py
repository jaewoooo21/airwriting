"""
Standalone Calibration Tool
=============================
Collects IMU data from ESP32, computes bias, and saves to imu.yaml.

Usage:
  python tools/calibrate.py                  # default settings
  python tools/calibrate.py --samples 500    # more samples
  python tools/calibrate.py --port 12345     # custom port
"""
import socket, struct, time, sys, argparse
import numpy as np
import yaml
from pathlib import Path

HEADER = 0xAA
FOOTER = 0x55
FMT_TS = struct.Struct("<I")
FMT_6F = struct.Struct("<6f")
SIDS = ("S1", "S2", "S3")
OFFSETS = (5, 29, 53)
GRAVITY = 9.81


def parse_packet(data):
    """Parse ESP32 packet, return list of sensor readings."""
    n = len(data)
    if n >= 80:
        if data[0] != HEADER or data[79] != FOOTER:
            return []
        ck = 0
        for i in range(1, 78):
            ck ^= data[i]
        if ck != data[78]:
            return []
    elif n >= 79:
        if data[0] != HEADER or data[78] != FOOTER:
            return []
        ck = 0
        for i in range(1, 77):
            ck ^= data[i]
        if ck != data[77]:
            return []
    else:
        return []

    pkts = []
    for sid, off in zip(SIDS, OFFSETS):
        f = FMT_6F.unpack_from(data, off)
        if all(v == v and abs(v) < 1e6 for v in f):
            pkts.append({
                "sid": sid,
                "accel": np.array(f[:3]),
                "gyro": np.array(f[3:]),
            })
    return pkts


def main():
    ap = argparse.ArgumentParser(description="IMU Calibration Tool")
    ap.add_argument("--port", type=int, default=12345)
    ap.add_argument("--samples", type=int, default=300,
                    help="Number of samples to collect per sensor")
    ap.add_argument("--timeout", type=int, default=15,
                    help="Max wait time in seconds")
    ap.add_argument("--config-dir", default=None,
                    help="Config directory (default: auto-detect)")
    args = ap.parse_args()

    # Find config dir
    if args.config_dir:
        config_dir = Path(args.config_dir)
    else:
        root = Path(__file__).parent.parent
        config_dir = root / "config"

    imu_path = config_dir / "imu.yaml"
    if not imu_path.exists():
        print(f"❌ imu.yaml not found at {imu_path}")
        sys.exit(1)

    print("=" * 55)
    print("  📐 IMU Calibration Tool")
    print("=" * 55)
    print(f"  Port:     {args.port}")
    print(f"  Samples:  {args.samples} per sensor")
    print(f"  Config:   {imu_path}")
    print()
    print("  ⚠️  KEEP THE SENSOR COMPLETELY STILL!")
    print("  ⚠️  Place on a flat surface, do not touch.")
    print()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(1.0)

    buffers = {s: {"accel": [], "gyro": []} for s in SIDS}
    t_start = time.time()
    first_packet = False

    try:
        while time.time() - t_start < args.timeout:
            try:
                data, addr = sock.recvfrom(256)
            except socket.timeout:
                if not first_packet:
                    elapsed = time.time() - t_start
                    print(f"\r  ⏳ Waiting for ESP32... ({elapsed:.0f}s)", end="")
                continue

            if not first_packet:
                first_packet = True
                print(f"\n  ✅ Connected to {addr[0]}:{addr[1]}")

            pkts = parse_packet(data)
            for p in pkts:
                sid = p["sid"]
                buffers[sid]["accel"].append(p["accel"])
                buffers[sid]["gyro"].append(p["gyro"])

            # Progress
            min_count = min(len(buffers[s]["accel"]) for s in SIDS)
            pct = min(100, int(min_count / args.samples * 100))
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  [{bar}] {pct}% ({min_count}/{args.samples})", end="")

            if all(len(buffers[s]["accel"]) >= args.samples for s in SIDS):
                break

    except KeyboardInterrupt:
        print("\n  ⏹️ Interrupted")
    finally:
        sock.close()

    print()
    print()

    # Check if we got enough data
    counts = {s: len(buffers[s]["accel"]) for s in SIDS}
    if min(counts.values()) < 10:
        print("  ❌ Insufficient data collected!")
        print(f"     Counts: {counts}")
        sys.exit(1)

    # ── Compute bias ──
    print("─" * 55)
    print("  📊 Calibration Results")
    print("─" * 55)

    bias_data = {}
    all_good = True

    for sid in SIDS:
        aa = np.array(buffers[sid]["accel"][:args.samples])
        gg = np.array(buffers[sid]["gyro"][:args.samples])

        # Accel bias: remove gravity
        a_mean = np.mean(aa, axis=0)
        grav_mag = np.linalg.norm(a_mean)
        if grav_mag > 1e-6:
            grav_dir = a_mean / grav_mag
            a_bias = a_mean - grav_dir * GRAVITY
        else:
            a_bias = a_mean.copy()
            a_bias[2] -= GRAVITY

        g_bias = np.mean(gg, axis=0)
        a_std = np.std(aa, axis=0)
        g_std = np.std(gg, axis=0)

        # Quality
        noisy = np.any(a_std > 0.5) or np.any(g_std > 0.02)
        quality = "⚠️ NOISY" if noisy else "✅ GOOD"
        if noisy:
            all_good = False

        print(f"\n  {sid} ({counts[sid]} samples)  {quality}")
        print(f"    Accel bias: [{a_bias[0]:+.6f}, {a_bias[1]:+.6f}, {a_bias[2]:+.6f}]")
        print(f"    Gyro bias:  [{g_bias[0]:+.6f}, {g_bias[1]:+.6f}, {g_bias[2]:+.6f}]")
        print(f"    Accel std:  [{a_std[0]:.4f}, {a_std[1]:.4f}, {a_std[2]:.4f}]")
        print(f"    Gyro std:   [{g_std[0]:.5f}, {g_std[1]:.5f}, {g_std[2]:.5f}]")
        print(f"    |gravity|:  {grav_mag:.3f} m/s² (expected ~9.81)")

        bias_data[sid] = {
            "accel": [round(float(v), 6) for v in a_bias],
            "gyro": [round(float(v), 6) for v in g_bias],
        }

    # ── Save to imu.yaml ──
    print()
    print("─" * 55)

    if not all_good:
        print("  ⚠️ Some sensors have noisy data.")
        r = input("  Save anyway? (y/N): ").strip().lower()
        if r != 'y':
            print("  ❌ Calibration NOT saved.")
            sys.exit(0)

    try:
        with open(imu_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        sensors = data.get("sensors", {})
        for sid, bias in bias_data.items():
            if sid in sensors:
                if "bias" not in sensors[sid]:
                    sensors[sid]["bias"] = {}
                sensors[sid]["bias"]["accel"] = bias["accel"]
                sensors[sid]["bias"]["gyro"] = bias["gyro"]

        with open(imu_path, "w", encoding="utf-8") as f:
            f.write("# IMU Configuration v3.2 (bias auto-calibrated)\n\n")
            yaml.dump(data, f, default_flow_style=False,
                     allow_unicode=True, sort_keys=False)

        print(f"  💾 Bias saved to {imu_path}")
        print()
        print("  Next steps:")
        print("    python main.py --load-bias    # Run with saved calibration")
        print("    python main.py                # Re-calibrate at startup")
    except Exception as e:
        print(f"  ❌ Save failed: {e}")
        sys.exit(1)

    print("─" * 55)


if __name__ == "__main__":
    main()
