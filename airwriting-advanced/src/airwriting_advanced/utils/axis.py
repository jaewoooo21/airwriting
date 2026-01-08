from __future__ import annotations

import numpy as np


def remap_vec3(v: np.ndarray, axis_map: np.ndarray, axis_sign: np.ndarray) -> np.ndarray:
    """
    v: (3,) in sensor raw axis order
    axis_map: e.g. [0,1,2] means x->x,y->y,z->z ; [1,0,2] means swap x/y
    axis_sign: e.g. [1,-1,1] flips y

    returns: (3,) remapped
    """
    vv = np.asarray(v, dtype=float).reshape(3)
    am = np.asarray(axis_map, dtype=int).reshape(3)
    sg = np.asarray(axis_sign, dtype=int).reshape(3)
    return vv[am] * sg
