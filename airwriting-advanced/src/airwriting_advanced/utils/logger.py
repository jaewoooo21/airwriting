from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class RunLogger:
    out_dir: str
    prefix: str = "run"
    csv_path: Optional[str] = None

    def __post_init__(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(self.out_dir, f"{self.prefix}_{ts}.csv")
        self._f = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow([
            "t",
            "S1_ax","S1_ay","S1_az","S1_gx","S1_gy","S1_gz",
            "S2_ax","S2_ay","S2_az","S2_gx","S2_gy","S2_gz",
            "S3_ax","S3_ay","S3_az","S3_gx","S3_gy","S3_gz",
            "tip_x","tip_y","tip_z",
            "uv_x","uv_y",
            "down","speed",
        ])
        self._f.flush()

    def write_frame(self, t: float,
                    imu: Dict[str, Dict[str, np.ndarray]],
                    tip_w: np.ndarray,
                    uv: np.ndarray,
                    down: bool,
                    speed: float) -> None:
        row = [float(t)]
        for s in ["S1","S2","S3"]:
            a = imu[s]["acc"]
            g = imu[s]["gyr"]
            row += [float(a[0]), float(a[1]), float(a[2]), float(g[0]), float(g[1]), float(g[2])]
        tip_w = np.asarray(tip_w, dtype=float).reshape(3)
        uv = np.asarray(uv, dtype=float).reshape(2)
        row += [float(tip_w[0]), float(tip_w[1]), float(tip_w[2]), float(uv[0]), float(uv[1]), int(down), float(speed)]
        self._w.writerow(row)
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass
