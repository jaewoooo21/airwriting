from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass
class IMUSample:
    t: float
    acc_m_s2: np.ndarray  # (3,)
    gyr_rad_s: np.ndarray  # (3,)
    mag_uT: Optional[np.ndarray]  # (3,) or None
    temp_C: Optional[float] = None


class IMUDevice(Protocol):
    name: str

    def read(self) -> IMUSample:
        ...

    def close(self) -> None:
        ...
