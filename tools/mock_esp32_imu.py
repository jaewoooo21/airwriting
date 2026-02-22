"""
Mock ESP32 (IMU-Only) — No UWB packets
Generates synthetic IMU data for testing the IMU-only airwriting system.
Packet format: 79 bytes (HEADER + TS + 3×(6f) + CHECKSUM + FOOTER)
"""
import socket, struct, time, math, argparse
import numpy as np

HEADER = 0xAA
FOOTER = 0x55
FMT_TS = struct.Struct("<I")
FMT_6F = struct.Struct("<6f")


def trajectory(t, pattern):
    """Generate test trajectories."""
    if pattern == "circle":
        r, f = 0.15, 0.5
        return np.array([r * math.cos(2 * math.pi * f * t),
                         r * math.sin(2 * math.pi * f * t), 0.0])
    elif pattern == "line":
        return np.array([0.2 * math.sin(2 * math.pi * 0.3 * t), 0.0, 0.0])
    elif pattern == "writing":
        # Simple "L" shape pattern
        c = t % 4.0
        if c < 1:
            x, y = 0.15 * c, 0.0
        elif c < 2:
            x, y = 0.15, -0.1 * (c - 1)
        elif c < 3:
            x, y = 0.15 + 0.1 * (c - 2), -0.1
        else:
            x, y = 0.0, 0.0  # Return to origin (ZUPT opportunity)
        return np.array([x, y, 0.0])
    elif pattern == "zupt_test":
        # Alternating movement and stillness for ZUPT testing
        c = t % 4.0
        if c < 1:
            return np.array([0.1 * c, 0.0, 0.0])  # Move
        elif c < 2:
            return np.array([0.1, 0.0, 0.0])       # Still (ZUPT)
        elif c < 3:
            return np.array([0.1 + 0.1 * (c - 2), 0.05 * (c - 2), 0.0])  # Move
        else:
            return np.array([0.2, 0.05, 0.0])      # Still (ZUPT)
    else:
        return np.array([0.0, 0.0, 0.0])


def main():
    ap = argparse.ArgumentParser(description="Mock ESP32 (IMU-only)")
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=12345)
    ap.add_argument("--rate", type=int, default=100)
    ap.add_argument("--pattern", default="circle",
                    choices=["circle", "line", "writing", "static", "zupt_test"])
    ap.add_argument("--noise", type=float, default=0.1)
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dt = 1.0 / args.rate
    t0 = time.time()
    n = 0

    print(f"Mock ESP32 (IMU-only) -> {args.ip}:{args.port}  "
          f"pattern={args.pattern} rate={args.rate}Hz noise={args.noise}")

    try:
        while True:
            t = time.time() - t0
            ts = int(t * 1e6) & 0xFFFFFFFF

            pkt = bytearray(79)
            pkt[0] = HEADER
            FMT_TS.pack_into(pkt, 1, ts)

            for sid, off in zip(("S1", "S2", "S3"), (5, 29, 53)):
                a = np.array([0., 0., 9.81])
                g = np.zeros(3)

                if sid == "S3" and args.pattern != "static":
                    # Generate acceleration from trajectory
                    p_now = trajectory(t, args.pattern)
                    p_prev = trajectory(t - dt, args.pattern)
                    p_pp = trajectory(t - 2 * dt, args.pattern)
                    v1 = (p_now - p_prev) / dt
                    v0 = (p_prev - p_pp) / dt
                    a += (v1 - v0) / dt
                    # Small gyro variation
                    g[2] = 0.3 * math.sin(2 * math.pi * 0.5 * t)
                elif sid == "S2":
                    # Slight wrist rotation
                    g[0] = 0.1 * math.sin(2 * math.pi * 0.3 * t)
                elif sid == "S1":
                    # Minimal forearm movement
                    g[1] = 0.05 * math.sin(2 * math.pi * 0.2 * t)

                a += np.random.normal(0, args.noise, 3)
                g += np.random.normal(0, args.noise * 0.1, 3)

                FMT_6F.pack_into(pkt, off,
                                 a[0], a[1], a[2], g[0], g[1], g[2])

            # Checksum
            ck = 0
            for i in range(1, 77):
                ck ^= pkt[i]
            pkt[77] = ck
            pkt[78] = FOOTER

            sock.sendto(bytes(pkt), (args.ip, args.port))
            n += 1

            if n % 500 == 0:
                pos = trajectory(t, args.pattern)
                print(f"n={n:,}  t={t:.1f}s  "
                      f"pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")

            time.sleep(dt)

    except KeyboardInterrupt:
        print(f"\nTotal: {n:,} packets")
    sock.close()


if __name__ == "__main__":
    main()
