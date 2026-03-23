"""
Health Check for ESP32 IMU packets.

Supports all packet formats currently used by the project:
  - v1: 79B
  - v2: 80B (button byte)
  - v3: 92B (S3 magnetometer)
"""
import argparse
import socket
import struct
import subprocess
import sys
import time

import numpy as np

HEADER = 0xAA
FOOTER = 0x55

PKT_LEN_V1 = 79
PKT_LEN_V2 = 80
PKT_LEN_V3 = 92

FMT_TS = struct.Struct("<I")
FMT_6F = struct.Struct("<6f")
FMT_9F = struct.Struct("<9f")


def parse_packet(data):
    """Parse a single ESP32 packet."""
    n = len(data)

    if n >= PKT_LEN_V3:
        if data[0] != HEADER or data[91] != FOOTER:
            return None, "header/footer"

        checksum = 0
        for i in range(1, 90):
            checksum ^= data[i]
        if checksum != data[90]:
            return None, "checksum"

        ts = FMT_TS.unpack_from(data, 1)[0]
        s1 = FMT_6F.unpack_from(data, 5)
        s2 = FMT_6F.unpack_from(data, 29)
        s3 = FMT_9F.unpack_from(data, 53)

        sensors = {
            "S1": {
                "accel": np.array([-s1[0], -s1[1], s1[2]], dtype=np.float64),
                "gyro": np.array([-s1[3], -s1[4], s1[5]], dtype=np.float64),
            },
            "S2": {
                "accel": np.array([-s2[0], -s2[1], s2[2]], dtype=np.float64),
                "gyro": np.array([-s2[3], -s2[4], s2[5]], dtype=np.float64),
            },
            "S3": {
                "accel": np.array([-s3[0], -s3[1], s3[2]], dtype=np.float64),
                "gyro": np.array([-s3[3], -s3[4], s3[5]], dtype=np.float64),
                "mag": np.array([-s3[6], -s3[7], s3[8]], dtype=np.float64),
            },
        }
        return {
            "ts": ts,
            "button": bool(data[89] & 0x01),
            "version": "v3",
            "sensors": sensors,
        }, None

    if n >= PKT_LEN_V2:
        if data[0] != HEADER or data[79] != FOOTER:
            return None, "header/footer"
        checksum = 0
        for i in range(1, 78):
            checksum ^= data[i]
        if checksum != data[78]:
            return None, "checksum"
        version = "v2"
        button = bool(data[77] & 0x01)
    elif n >= PKT_LEN_V1:
        if data[0] != HEADER or data[78] != FOOTER:
            return None, "header/footer"
        checksum = 0
        for i in range(1, 77):
            checksum ^= data[i]
        if checksum != data[77]:
            return None, "checksum"
        version = "v1"
        button = None
    else:
        return None, f"size({n})"

    ts = FMT_TS.unpack_from(data, 1)[0]
    sensors = {}
    for sid, off in zip(("S1", "S2", "S3"), (5, 29, 53)):
        values = FMT_6F.unpack_from(data, off)
        sensors[sid] = {
            "accel": np.array(values[:3], dtype=np.float64),
            "gyro": np.array(values[3:], dtype=np.float64),
        }

    return {
        "ts": ts,
        "button": button,
        "version": version,
        "sensors": sensors,
    }, None


def find_udp_port_owners(port):
    """Best-effort lookup for UDP listeners without admin privileges."""
    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "udp"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []

    owners = []
    needle = f":{port}"
    for line in result.stdout.splitlines():
        if needle not in line:
            continue
        parts = line.split()
        if len(parts) >= 4 and parts[1].endswith(needle):
            owners.append(parts[-1])
    return sorted(set(owners))


def main():
    ap = argparse.ArgumentParser(description="ESP32 Health Check")
    ap.add_argument("--port", type=int, default=12345)
    ap.add_argument("--duration", type=int, default=5)
    args = ap.parse_args()

    print("=" * 55)
    print("  ESP32 Health Check")
    print("=" * 55)
    print(f"  Listening on UDP :{args.port} for {args.duration}s...")
    print()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(1.0)

    n_packets = 0
    n_errors = {"checksum": 0, "header/footer": 0, "other": 0}
    sensor_data = {s: {"accel": [], "gyro": []} for s in ("S1", "S2", "S3")}
    has_button = False
    versions = {}
    first_sender = None
    t_start = time.time()

    try:
        while time.time() - t_start < args.duration:
            try:
                data, addr = sock.recvfrom(256)
            except socket.timeout:
                continue

            if first_sender is None:
                first_sender = addr
                print(f"  First packet from {addr[0]}:{addr[1]}")

            result, err = parse_packet(data)
            if err:
                n_errors[err] = n_errors.get(err, 0) + 1
                continue

            n_packets += 1
            if result["button"] is not None:
                has_button = True
            versions[result["version"]] = versions.get(result["version"], 0) + 1

            for sid, sdata in result["sensors"].items():
                sensor_data[sid]["accel"].append(sdata["accel"])
                sensor_data[sid]["gyro"].append(sdata["gyro"])
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()

    elapsed = time.time() - t_start

    print()
    print("=" * 55)
    print("  Results")
    print("=" * 55)

    if n_packets == 0:
        print("  NO PACKETS RECEIVED")
        print()
        owners = find_udp_port_owners(args.port)
        if owners:
            print(f"  Note: UDP {args.port} is already open by PID(s): {', '.join(owners)}")
            print("  On Windows, another listener can consume the packets first.")
            print()
        print("  Troubleshooting:")
        print("    1. Check ESP32 power")
        print("    2. Check Wi-Fi network match")
        print("    3. Check PC IP in firmware")
        print(f"    4. Check firewall for UDP {args.port}")
        print("    5. Try tools/mock_esp32_imu.py for pipeline-only verification")
        sys.exit(1)

    hz = n_packets / elapsed if elapsed > 0 else 0.0
    total_err = sum(n_errors.values())
    err_rate = total_err / max(n_packets + total_err, 1)

    print(f"  Packets:    {n_packets:,}")
    print(f"  Rate:       {hz:.1f} Hz")
    print(f"  Duration:   {elapsed:.1f}s")
    print(f"  Errors:     {total_err}")
    print(f"  Button:     {'yes' if has_button else 'no'}")
    if first_sender is not None:
        print(f"  Sender:     {first_sender[0]}:{first_sender[1]}")
    print(f"  Format:     {', '.join(f'{k}={v}' for k, v in sorted(versions.items()))}")
    print()

    for sid in ("S1", "S2", "S3"):
        accel = np.array(sensor_data[sid]["accel"])
        gyro = np.array(sensor_data[sid]["gyro"])
        if len(accel) == 0:
            print(f"  {sid}: no data")
            continue

        a_mean = np.mean(accel, axis=0)
        a_std = np.std(accel, axis=0)
        g_mean = np.mean(gyro, axis=0)
        g_std = np.std(gyro, axis=0)
        grav_mag = np.linalg.norm(a_mean)
        grav_ok = "ok" if abs(grav_mag - 9.81) < 1.0 else "warn"
        noise_ok = "ok" if np.all(a_std < 0.5) else "warn"

        print(f"  {sid} ({len(accel)} samples):")
        print(
            f"    Accel mean: [{a_mean[0]:+.3f}, {a_mean[1]:+.3f}, {a_mean[2]:+.3f}]"
            f"  |g|={grav_mag:.2f} {grav_ok}"
        )
        print(
            f"    Accel std:  [{a_std[0]:.4f}, {a_std[1]:.4f}, {a_std[2]:.4f}]  {noise_ok}"
        )
        print(
            f"    Gyro mean:  [{g_mean[0]:+.5f}, {g_mean[1]:+.5f}, {g_mean[2]:+.5f}]"
        )
        print(
            f"    Gyro std:   [{g_std[0]:.5f}, {g_std[1]:.5f}, {g_std[2]:.5f}]"
        )
        print()

    print("=" * 55)
    if hz > 80 and err_rate < 0.01:
        print("  HEALTH CHECK PASSED")
    elif hz > 50:
        print("  MARGINAL: rate is low, check Wi-Fi signal")
    else:
        print("  FAILED: insufficient data rate")
    print("=" * 55)


if __name__ == "__main__":
    main()
