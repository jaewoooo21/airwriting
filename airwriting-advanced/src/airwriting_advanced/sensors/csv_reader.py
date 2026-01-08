from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, List, Tuple

import numpy as np

from .imu_base import IMUSample


@dataclass
class MultiIMUFrame:
    t: float
    samples: Dict[str, IMUSample]


def _try_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


class CSVReplay:
    """
    CSV replay with auto-detection.

    Supported formats:

    (A) long format:
        t,sensor,ax,ay,az,gx,gy,gz,mx,my,mz
        1700000000.00,S1,.......

    (B) wide format:
        t,S1_ax,S1_ay,S1_az,S1_gx,S1_gy,S1_gz,S1_mx,...,S2_ax,...,S3_ax,...

    Units expected:
        - acc: m/s^2
        - gyro: rad/s
        - mag: uT (optional)
    """

    def __init__(self, csv_path: str, sensor_names: List[str]):
        self.csv_path = csv_path
        self.sensor_names = sensor_names
        self._rows = self._load_rows()
        if not self._rows:
            raise ValueError(f"CSV is empty: {csv_path}")
        self._mode = self._detect_mode(self._rows[0].keys())

    def _load_rows(self) -> List[Dict[str, str]]:
        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            return [row for row in r]

    @staticmethod
    def _detect_mode(keys) -> str:
        keys = set(keys)
        if "sensor" in keys and "t" in keys:
            return "long"
        if "t" in keys:
            return "wide"
        raise ValueError("CSV must contain 't' column (and optionally 'sensor').")

    def frames(self) -> Iterator[MultiIMUFrame]:
        if self._mode == "long":
            yield from self._frames_long()
        else:
            yield from self._frames_wide()

    def _frames_long(self) -> Iterator[MultiIMUFrame]:
        # Group by timestamp (exact string match). If your CSV has different rates, pre-resample.
        current_t = None
        bucket: Dict[str, IMUSample] = {}
        for row in self._rows:
            t = float(row["t"])
            sname = row["sensor"]
            if current_t is None:
                current_t = t
            if t != current_t:
                if bucket:
                    yield MultiIMUFrame(t=current_t, samples=bucket)
                current_t = t
                bucket = {}
            bucket[sname] = IMUSample(
                t=t,
                acc_m_s2=np.array([float(row["ax"]), float(row["ay"]), float(row["az"])], dtype=float),
                gyr_rad_s=np.array([float(row["gx"]), float(row["gy"]), float(row["gz"])], dtype=float),
                mag_uT=(np.array([float(row["mx"]), float(row["my"]), float(row["mz"])], dtype=float)
                        if ("mx" in row and row["mx"] not in ("", None)) else None),
            )
        if bucket and current_t is not None:
            yield MultiIMUFrame(t=current_t, samples=bucket)

    def _frames_wide(self) -> Iterator[MultiIMUFrame]:
        for row in self._rows:
            t = float(row["t"])
            samples: Dict[str, IMUSample] = {}
            for s in self.sensor_names:
                def col(x): return f"{s}_{x}"
                ax = float(row[col("ax")])
                ay = float(row[col("ay")])
                az = float(row[col("az")])
                gx = float(row[col("gx")])
                gy = float(row[col("gy")])
                gz = float(row[col("gz")])
                mx = row.get(col("mx"), "")
                my = row.get(col("my"), "")
                mz = row.get(col("mz"), "")
                mag = None
                if mx != "" and my != "" and mz != "":
                    mag = np.array([float(mx), float(my), float(mz)], dtype=float)
                samples[s] = IMUSample(
                    t=t,
                    acc_m_s2=np.array([ax, ay, az], dtype=float),
                    gyr_rad_s=np.array([gx, gy, gz], dtype=float),
                    mag_uT=mag,
                )
            yield MultiIMUFrame(t=t, samples=samples)
