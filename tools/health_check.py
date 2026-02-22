"""
Health Check — ESP32 Connection & Sensor Quality Test
=====================================================
Listens for ESP32 packets and reports:
  - Connection status
  - Packet rate (Hz)
  - Per-sensor data ranges
  - Noise levels
  - Checksum error rate

Usage:
  python tools/health_check.py               # default port
  python tools/health_check.py --port 12345  # custom port
  python tools/health_check.py --duration 10 # test for 10 seconds
"""
import socket, struct, time, sys, argparse
import numpy as np

HEADER = 0xAA
FOOTER = 0x55
FMT_TS = struct.Struct("<I")
FMT_6F = struct.Struct("<6f")
SIDS = ("S1", "S2", "S3")
OFFSETS = (5, 29, 53)


def parse_packet(data):
    """Parse a single ESP32 packet."""
    n = len(data)
    button = None

    if n >= 80:
        if data[0] != HEADER or data[79] != FOOTER:
            return None, "header/footer"
        ck = 0
        for i in range(1, 78):
            ck ^= data[i]
        if ck != data[78]:
            return None, "checksum"
        button = data[77] & 0x01
    elif n >= 79:
        if data[0] != HEADER or data[78] != FOOTER:
            return None, "header/footer"
        ck = 0
        for i in range(1, 77):
            ck ^= data[i]
        if ck != data[77]:
            return None, "checksum"
    else:
        return None, f"size({n})"

    ts = FMT_TS.unpack_from(data, 1)[0]
    sensors = {}
    for sid, off in zip(SIDS, OFFSETS):
        f = FMT_6F.unpack_from(data, off)
        sensors[sid] = {
            "accel": np.array(f[:3]),
            "gyro": np.array(f[3:]),
        }

    return {"ts": ts, "sensors": sensors, "button": button}, None


def main():
    ap = argparse.ArgumentParser(description="ESP32 Health Check")
    ap.add_argument("--port", type=int, default=12345)
    ap.add_argument("--duration", type=int, default=5,
                    help="Test duration in seconds")
    args = ap.parse_args()

    print("=" * 55)
    print("  🏥 ESP32 Health Check")
    print("=" * 55)
    print(f"  Listening on port {args.port} for {args.duration}s...")
    print(f"  Make sure ESP32 is powered on and connected to WiFi.")
    print()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(1.0)

    # Accumulators
    n_packets = 0
    n_errors = {"checksum": 0, "header/footer": 0, "other": 0}
    sensor_data = {s: {"accel": [], "gyro": []} for s in SIDS}
    has_button = False
    t_start = time.time()
    first_packet = None

    try:
        while time.time() - t_start < args.duration:
            try:
                data, addr = sock.recvfrom(256)
            except socket.timeout:
                continue

            if first_packet is None:
                first_packet = time.time()
                print(f"  ✅ First packet from {addr[0]}:{addr[1]}")

            result, err = parse_packet(data)
            if err:
                n_errors[err] = n_errors.get(err, 0) + 1
                continue

            n_packets += 1
            if result["button"] is not None:
                has_button = True

            for sid, sdata in result["sensors"].items():
                sensor_data[sid]["accel"].append(sdata["accel"])
                sensor_data[sid]["gyro"].append(sdata["gyro"])

    except KeyboardInterrupt:
        pass
    finally:
        sock.close()

    elapsed = time.time() - t_start

    # ── Report ──
    print()
    print("─" * 55)
    print("  📊 Results")
    print("─" * 55)

    if n_packets == 0:
        print("  ❌ NO PACKETS RECEIVED!")
        print()
        print("  Troubleshooting:")
        print("    1. Is ESP32 powered on?")
        print("    2. Is ESP32 connected to same WiFi network?")
        print("    3. Is the correct IP configured in ESP32 firmware?")
        print(f"    4. Is port {args.port} blocked by firewall?")
        print("    5. Try: python tools/mock_esp32_imu.py (to test pipeline)")
        sys.exit(1)

    hz = n_packets / elapsed
    print(f"  Packets:    {n_packets:,}")
    print(f"  Rate:       {hz:.1f} Hz")
    print(f"  Duration:   {elapsed:.1f}s")
    print(f"  Errors:     {sum(n_errors.values())}")
    print(f"  Button:     {'✅ v2' if has_button else '⚠️ v1 (no button)'}")
    print()

    for sid in SIDS:
        a = np.array(sensor_data[sid]["accel"])
        g = np.array(sensor_data[sid]["gyro"])
        if len(a) == 0:
            print(f"  {sid}: ❌ No data!")
            continue

        a_mean = np.mean(a, axis=0)
        a_std = np.std(a, axis=0)
        g_mean = np.mean(g, axis=0)
        g_std = np.std(g, axis=0)

        # Quality assessment
        grav_mag = np.linalg.norm(a_mean)
        grav_ok = "✅" if abs(grav_mag - 9.81) < 1.0 else "⚠️"
        noise_ok = "✅" if np.all(a_std < 0.5) else "⚠️"

        print(f"  {sid} ({len(a)} samples):")
        print(f"    Accel mean: [{a_mean[0]:+.3f}, {a_mean[1]:+.3f}, {a_mean[2]:+.3f}]  "
              f"|g|={grav_mag:.2f} {grav_ok}")
        print(f"    Accel std:  [{a_std[0]:.4f}, {a_std[1]:.4f}, {a_std[2]:.4f}]  {noise_ok}")
        print(f"    Gyro mean:  [{g_mean[0]:+.5f}, {g_mean[1]:+.5f}, {g_mean[2]:+.5f}]")
        print(f"    Gyro std:   [{g_std[0]:.5f}, {g_std[1]:.5f}, {g_std[2]:.5f}]")
        print()

    # Overall verdict
    print("─" * 55)
    total_err = sum(n_errors.values())
    err_rate = total_err / max(n_packets + total_err, 1)
    if hz > 80 and err_rate < 0.01:
        print("  🎉 HEALTH CHECK PASSED — Ready for airwriting!")
    elif hz > 50:
        print("  ⚠️ MARGINAL — Rate is low, check WiFi signal")
    else:
        print("  ❌ FAILED — Insufficient data rate")
    print("─" * 55)


if __name__ == "__main__":
    main()
