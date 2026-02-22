"""1€ Filter v3.2 — 출력 배열 사전할당"""
import numpy as np
from math import pi


class OneEuroFilter:
    __slots__ = ("_freq", "_mincutoff", "_beta", "_dcutoff",
                 "_x", "_dx", "_init")

    def __init__(self, freq=100., min_cutoff=1., beta=0.007, d_cutoff=1.):
        self._freq = freq
        self._mincutoff = min_cutoff
        self._beta = beta
        self._dcutoff = d_cutoff
        self._x = 0.
        self._dx = 0.
        self._init = False

    @staticmethod
    def _alpha(cutoff, freq):
        tau = 1.0 / (2 * pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        if not self._init:
            self._x = x
            self._dx = 0.
            self._init = True
            return x
        dx = (x - self._x) * self._freq
        a_d = self._alpha(self._dcutoff, self._freq)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self._mincutoff + self._beta * abs(self._dx)
        a = self._alpha(cutoff, self._freq)
        self._x = a * x + (1 - a) * self._x
        return self._x

    def reset(self):
        self._init = False


class OneEuroFilter3D:
    __slots__ = ("_f", "_out")

    def __init__(self, freq=100., min_cutoff=1., beta=0.007):
        self._f = [OneEuroFilter(freq, min_cutoff, beta) for _ in range(3)]
        self._out = np.empty(3, dtype=np.float64)  # ✅ 사전할당

    def __call__(self, v):
        self._out[0] = self._f[0](v[0])
        self._out[1] = self._f[1](v[1])
        self._out[2] = self._f[2](v[2])
        return self._out.copy()

    def reset(self):
        for f in self._f:
            f.reset()
