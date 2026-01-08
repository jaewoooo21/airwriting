from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass
class Plane:
    n: np.ndarray  # unit normal (3,)
    d: float       # plane: n·x + d = 0
    u: np.ndarray  # basis (3,)
    v: np.ndarray  # basis (3,)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return v / n


class PlaneProjector:
    def __init__(self, normal_w: np.ndarray, d: float, u_w: np.ndarray, v_w: np.ndarray):
        n = _unit(normal_w)
        u = np.asarray(u_w, dtype=float).reshape(3)
        # make u orthogonal to n
        u = u - n * float(np.dot(n, u))
        u = _unit(u)
        v = np.asarray(v_w, dtype=float).reshape(3)
        v = v - n * float(np.dot(n, v))
        v = v - u * float(np.dot(u, v))
        v = _unit(v)
        self.plane = Plane(n=n, d=float(d), u=u, v=v)

    def signed_distance(self, x_w: np.ndarray) -> float:
        x = np.asarray(x_w, dtype=float).reshape(3)
        return float(np.dot(self.plane.n, x) + self.plane.d)

    def project_point(self, x_w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (x_plane_w, uv) where uv is 2D coords in plane basis."""
        x = np.asarray(x_w, dtype=float).reshape(3)
        dist = self.signed_distance(x)
        x_plane = x - self.plane.n * dist
        uv = np.array([float(np.dot(self.plane.u, x_plane)), float(np.dot(self.plane.v, x_plane))], dtype=float)
        return x_plane, uv

    @staticmethod
    def fit_from_points(points_w: Iterable[np.ndarray]) -> Plane:
        pts = np.stack([np.asarray(p, dtype=float).reshape(3) for p in points_w], axis=0)
        c = np.mean(pts, axis=0)
        X = pts - c
        # PCA: smallest singular vector is normal
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        n = vt[-1, :]
        n = _unit(n)
        d = -float(np.dot(n, c))
        # choose u from first principal component
        u = _unit(vt[0, :])
        u = u - n * float(np.dot(n, u))
        u = _unit(u)
        v = np.cross(n, u)
        v = _unit(v)
        return Plane(n=n, d=d, u=u, v=v)
