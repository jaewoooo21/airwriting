"""
Training Data Generator for Neural ZUPT
=========================================
Generates labeled IMU windows for training the ZUPT LSTM.

Labels:
  1.0 = stationary (zero velocity)
  0.0 = in motion

Uses trajectory patterns from mock_esp32 to create realistic
motion/stationary transitions.
"""
import numpy as np
import math
import json
import logging
import argparse
from pathlib import Path

log = logging.getLogger(__name__)


def generate_trajectory(t: float, pattern: str) -> tuple:
    """
    Generate position and ground-truth stationary label.

    Returns:
        (position [3], velocity [3], is_stationary: bool)
    """
    if pattern == "zupt_alternating":
        # 2s move, 2s stop, repeat
        cycle = t % 4.0
        if cycle < 2.0:
            # Moving
            f = 0.5
            pos = np.array([0.1 * math.sin(2 * math.pi * f * t),
                            0.05 * math.cos(2 * math.pi * f * t), 0.0])
            vel = np.array([0.1 * 2*math.pi*f * math.cos(2*math.pi*f*t),
                            -0.05 * 2*math.pi*f * math.sin(2*math.pi*f*t), 0.0])
            return pos, vel, False
        else:
            # Stationary
            return np.array([0.1 * math.sin(2*math.pi*0.5*2.0), 0.05*math.cos(2*math.pi*0.5*2.0), 0.0]), np.zeros(3), True

    elif pattern == "writing":
        # L-shape with pauses
        c = t % 6.0
        if c < 1.0:
            pos = np.array([0.15 * c, 0.0, 0.0])
            vel = np.array([0.15, 0.0, 0.0])
            return pos, vel, False
        elif c < 1.5:
            return np.array([0.15, 0.0, 0.0]), np.zeros(3), True  # Pause
        elif c < 2.5:
            t2 = c - 1.5
            pos = np.array([0.15, -0.1 * t2, 0.0])
            vel = np.array([0.0, -0.1, 0.0])
            return pos, vel, False
        elif c < 3.0:
            return np.array([0.15, -0.1, 0.0]), np.zeros(3), True  # Pause
        elif c < 4.0:
            t3 = c - 3.0
            pos = np.array([0.15 + 0.1 * t3, -0.1, 0.0])
            vel = np.array([0.1, 0.0, 0.0])
            return pos, vel, False
        else:
            return np.array([0.25, -0.1, 0.0]), np.zeros(3), True  # Pause

    elif pattern == "circle":
        r, f = 0.1, 0.5
        pos = np.array([r * math.cos(2*math.pi*f*t),
                        r * math.sin(2*math.pi*f*t), 0.0])
        vel = np.array([-r*2*math.pi*f*math.sin(2*math.pi*f*t),
                        r*2*math.pi*f*math.cos(2*math.pi*f*t), 0.0])
        return pos, vel, False

    elif pattern == "static":
        return np.zeros(3), np.zeros(3), True

    elif pattern == "fast_stop":
        # Quick motion then abrupt stop (hard case)
        c = t % 3.0
        if c < 0.5:
            speed = 2.0 * c  # Accelerating
            pos = np.array([0.5 * speed * c, 0.0, 0.0])
            vel = np.array([speed, 0.0, 0.0])
            return pos, vel, False
        elif c < 0.7:
            # Decelerating
            return np.array([0.25, 0.0, 0.0]), np.array([0.5, 0.0, 0.0]), False
        else:
            return np.array([0.3, 0.0, 0.0]), np.zeros(3), True

    return np.zeros(3), np.zeros(3), True


def generate_imu_from_trajectory(pos: np.ndarray, vel: np.ndarray,
                                 prev_vel: np.ndarray, dt: float,
                                 is_stationary: bool,
                                 noise_accel: float = 0.1,
                                 noise_gyro: float = 0.02) -> tuple:
    """
    Generate synthetic IMU readings from trajectory.

    Returns:
        (accel [3], gyro [3])
    """
    # Acceleration from velocity difference
    if dt > 0:
        accel = (vel - prev_vel) / dt
    else:
        accel = np.zeros(3)

    # Add gravity component (sensor measures gravity when stationary)
    # Since we work in gravity-removed space, just add noise
    accel += np.random.normal(0, noise_accel if not is_stationary else noise_accel * 0.3, 3)

    # Gyro: small rotation matching motion
    if is_stationary:
        gyro = np.random.normal(0, noise_gyro * 0.2, 3)
    else:
        gyro = np.random.normal(0, noise_gyro, 3)
        gyro[2] += 0.1 * np.linalg.norm(vel)  # Yaw correlates with velocity

    return accel, gyro


def generate_dataset(num_seconds: float = 60.0, sample_rate: int = 100,
                     window_size: int = 20, patterns: list = None,
                     noise_level: float = 0.1) -> tuple:
    """
    Generate complete training dataset.

    Returns:
        X: [N, window_size, 6] — IMU windows
        y: [N, 1] — labels (1=stationary, 0=moving)
    """
    if patterns is None:
        patterns = ["zupt_alternating", "writing", "circle", "static", "fast_stop"]

    all_X = []
    all_y = []

    for pattern in patterns:
        dt = 1.0 / sample_rate
        n_samples = int(num_seconds * sample_rate)

        # Generate trajectory
        buffer = []
        labels = []
        prev_vel = np.zeros(3)

        for i in range(n_samples):
            t = i * dt
            pos, vel, is_stationary = generate_trajectory(t, pattern)
            accel, gyro = generate_imu_from_trajectory(
                pos, vel, prev_vel, dt, is_stationary,
                noise_accel=noise_level, noise_gyro=noise_level * 0.2
            )
            prev_vel = vel.copy()

            sample = np.concatenate([accel, gyro])  # [6]
            buffer.append(sample)
            labels.append(1.0 if is_stationary else 0.0)

        # Create sliding windows
        for i in range(window_size, len(buffer)):
            window = np.array(buffer[i - window_size:i], dtype=np.float32)
            label = labels[i - 1]  # Label of last sample in window
            all_X.append(window)
            all_y.append(label)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32).reshape(-1, 1)

    # Balance dataset (undersample majority class)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos > 0 and n_neg > 0:
        minority = min(n_pos, n_neg)
        pos_idx = np.where(y.flatten() == 1.0)[0]
        neg_idx = np.where(y.flatten() == 0.0)[0]

        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)

        # Take equal samples (with some oversampling of minority)
        target = max(n_pos, n_neg)
        if n_pos < n_neg:
            pos_idx = np.tile(pos_idx, (target // n_pos + 1))[:target]
            neg_idx = neg_idx[:target]
        else:
            neg_idx = np.tile(neg_idx, (target // n_neg + 1))[:target]
            pos_idx = pos_idx[:target]

        idx = np.concatenate([pos_idx, neg_idx])
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

    log.info(f"Generated dataset: {X.shape[0]} samples, "
             f"{int(y.sum())} stationary, {len(y) - int(y.sum())} moving")
    return X, y


def save_dataset(X: np.ndarray, y: np.ndarray, path: str):
    """Save dataset as numpy arrays."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    np.save(p / "X.npy", X)
    np.save(p / "y.npy", y)
    meta = {"n_samples": len(X), "window_size": X.shape[1],
            "n_features": X.shape[2], "n_stationary": int(y.sum()),
            "n_moving": len(y) - int(y.sum())}
    with open(p / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"💾 Saved to {p}: {meta}")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    ap = argparse.ArgumentParser(description="Generate Neural ZUPT training data")
    ap.add_argument("--duration", type=float, default=120.0,
                    help="Seconds per pattern")
    ap.add_argument("--rate", type=int, default=100)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--output", default="data/zupt_training",
                    help="Output directory")
    args = ap.parse_args()

    np.random.seed(42)

    X, y = generate_dataset(
        num_seconds=args.duration,
        sample_rate=args.rate,
        window_size=args.window,
        noise_level=args.noise,
    )

    save_dataset(X, y, args.output)
    print(f"\n✅ Dataset: {X.shape[0]:,} windows, "
          f"{X.shape[1]}×{X.shape[2]} features")
    print(f"   Stationary: {int(y.sum()):,} ({100*y.mean():.1f}%)")
    print(f"   Moving:     {len(y)-int(y.sum()):,} ({100*(1-y.mean()):.1f}%)")


if __name__ == "__main__":
    main()
