from __future__ import annotations

import argparse
import time

from .replay_csv import main as replay_main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="data/samples/sample.csv")
    ap.add_argument("--config", type=str, default="config/config.yaml")
    ap.add_argument("--seconds", type=float, default=5.0)
    args, unknown = ap.parse_known_args()

    # run replay for a bounded time by passing through to replay_csv with --no-view
    # (replay_csv supports --out-dir, etc.)
    # We call it as a module-level function by temporarily patching sys.argv.
    import sys
    argv0 = sys.argv[:]
    try:
        sys.argv = ["replay_csv", "--config", args.config, "--csv", args.csv, "--no-view", "--out-dir", "runs/benchmark"]
        t0 = time.time()
        replay_main()
        dt = time.time() - t0
        print(f"benchmark finished in {dt:.3f}s (see runs/benchmark)")
    finally:
        sys.argv = argv0


if __name__ == "__main__":
    main()
