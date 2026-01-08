from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import serial


@dataclass
class UWBRange:
    t: float
    distance_m: float
    anchor_id: Optional[str] = None
    tag_id: Optional[str] = None


class UWBSerialReader:
    """
    Extremely small, format-tolerant UWB range reader.

    Expected line formats (examples):
      - "RANGE,2.341"
      - "RANGE,anchorA,tag1,2.341"
      - "2.341"
    """

    def __init__(self, port: str, baud: int = 115200, timeout: float = 0.2):
        self.ser = serial.Serial(port, baud, timeout=timeout)

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass

    def read(self) -> Optional[UWBRange]:
        line = self.ser.readline()
        if not line:
            return None
        try:
            s = line.decode("utf-8", errors="ignore").strip()
        except Exception:
            return None
        if not s:
            return None
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        try:
            if len(parts) == 1:
                d = float(parts[0])
                return UWBRange(t=time.time(), distance_m=d)
            if len(parts) == 2 and parts[0].upper() == "RANGE":
                d = float(parts[1])
                return UWBRange(t=time.time(), distance_m=d)
            if len(parts) >= 4 and parts[0].upper() == "RANGE":
                anchor_id = parts[1]
                tag_id = parts[2]
                d = float(parts[3])
                return UWBRange(t=time.time(), distance_m=d, anchor_id=anchor_id, tag_id=tag_id)
        except Exception:
            return None
        return None
