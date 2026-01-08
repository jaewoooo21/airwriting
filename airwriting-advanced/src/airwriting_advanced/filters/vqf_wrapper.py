from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np

try:
    from vqf import VQF, PyVQF
except Exception:  # pragma: no cover
    VQF = None  # type: ignore
    PyVQF = None  # type: ignore

from ..quaternion import q_normalize


@dataclass
class VQFOutput:
    quat6D_wxyz: np.ndarray
    quat9D_wxyz: Optional[np.ndarray]
    rest_detected: bool
    mag_dist_detected: bool
    bias_rad_s: np.ndarray  # (3,)


class VQFWrapper:
    """
    Thin wrapper around the official VQF implementation.
    - quat6D: magnetometer-free, drift-free roll/pitch + bias-est. yaw still can drift.
    - quat9D: uses magnetometer, includes mag disturbance detection/rejection if enabled.
    """

    def __init__(self, sample_rate_hz: float, *, use_fast: bool = True, params: Optional[Dict[str, Any]] = None):
        if VQF is None and PyVQF is None:
            raise ImportError("vqf is not installed. Install with: pip install vqf")

        self.Ts = 1.0 / float(sample_rate_hz)
        self.params = params or {}

        if use_fast and VQF is not None:
            self.vqf = VQF(self.Ts, **self.params)
        else:
            if PyVQF is None:
                raise ImportError("vqf.PyVQF not available (vqf not installed correctly).")
            self.vqf = PyVQF(self.Ts, **self.params)

    def reset(self) -> None:
        self.vqf.resetState()

    def update(self, gyr_rad_s: np.ndarray, acc_m_s2: np.ndarray, mag_uT: Optional[np.ndarray] = None) -> VQFOutput:
        gyr = np.asarray(gyr_rad_s, dtype=float).reshape(3)
        acc = np.asarray(acc_m_s2, dtype=float).reshape(3)
        if mag_uT is None:
            self.vqf.update(gyr, acc)
        else:
            mag = np.asarray(mag_uT, dtype=float).reshape(3)
            self.vqf.update(gyr, acc, mag)

        quat6 = np.asarray(self.vqf.getQuat6D(), dtype=float).reshape(4)
        quat9 = None
        if mag_uT is not None:
            quat9 = np.asarray(self.vqf.getQuat9D(), dtype=float).reshape(4)

        bias = np.asarray(self.vqf.getBiasEstimate(), dtype=float).reshape(-1)
        # getBiasEstimate returns [bx, by, bz, sigma] or separate? normalize to first 3.
        bias3 = bias[:3].copy()

        return VQFOutput(
            quat6D_wxyz=q_normalize(quat6),
            quat9D_wxyz=q_normalize(quat9) if quat9 is not None else None,
            rest_detected=bool(self.vqf.getRestDetected()),
            mag_dist_detected=bool(self.vqf.getMagDistDetected()),
            bias_rad_s=bias3,
        )
