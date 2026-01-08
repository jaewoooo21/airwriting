from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Stroke:
    t0: float
    points_uv: List[np.ndarray] = field(default_factory=list)
    points_w: List[np.ndarray] = field(default_factory=list)
    times: List[float] = field(default_factory=list)


class StrokeDetector:
    def __init__(self, speed_down_th: float, speed_up_th: float, min_down_time_s: float):
        self.speed_down_th = float(speed_down_th)
        self.speed_up_th = float(speed_up_th)
        self.min_down_time_s = float(min_down_time_s)

        self.is_down = False
        self.down_t0: Optional[float] = None
        self.last_uv: Optional[np.ndarray] = None
        self.last_t: Optional[float] = None
        self.current: Optional[Stroke] = None
        self.finished: List[Stroke] = []

    def update(self, t: float, uv: np.ndarray, point_w: np.ndarray) -> Tuple[bool, float]:
        uv = np.asarray(uv, dtype=float).reshape(2)
        point_w = np.asarray(point_w, dtype=float).reshape(3)

        speed = 0.0
        if self.last_uv is not None and self.last_t is not None:
            dt = max(1e-6, float(t - self.last_t))
            speed = float(np.linalg.norm(uv - self.last_uv) / dt)

        # state machine
        if not self.is_down:
            if speed >= self.speed_down_th:
                self.is_down = True
                self.down_t0 = t
                self.current = Stroke(t0=t)
        else:
            assert self.down_t0 is not None and self.current is not None
            down_time = t - self.down_t0
            if down_time >= self.min_down_time_s and speed <= self.speed_up_th:
                # lift
                self.is_down = False
                self.finished.append(self.current)
                self.current = None
                self.down_t0 = None

        # record if down
        if self.is_down and self.current is not None:
            self.current.points_uv.append(uv.copy())
            self.current.points_w.append(point_w.copy())
            self.current.times.append(float(t))

        self.last_uv = uv.copy()
        self.last_t = float(t)
        return self.is_down, speed

    def pop_finished(self) -> List[Stroke]:
        out = self.finished
        self.finished = []
        return out
