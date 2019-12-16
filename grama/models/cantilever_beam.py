__all__ = ["make_cantilever_beam"]

import numpy as np
from .. import core
from .. import compositions as cp
from collections import OrderedDict as od
from numpy import sqrt, array, Inf
from scipy.stats import norm

LENGTH = 100
D_MAX  = 2.2535

MU_H   = 500.
MU_V   = 1000.
MU_E   = 2.9e7
MU_Y   = 40000.

TAU_H  = 100.
TAU_V  = 100.
TAU_E  = 1.45e6
TAU_Y  = 2000.

def function_beam(x):
    w, t, H, V, E, Y = x

    return array([
        w * t,
        Y - 600 * V / w / t**2 - 600 * H / w**2 / t,
        D_MAX - np.float64(4) * LENGTH**3 / E / w / t * sqrt(
            V**2 / t**4 + H**2 / w**4
        )
    ])

def function_area(x):
    w, t = x
    return w * t

def function_stress(x):
    w, t, H, V, E, Y = x
    return Y - 600 * V / w / t**2 - 600 * H / w**2 / t

def function_displacement(x):
    w, t, H, V, E, Y = x
    return D_MAX - np.float64(4) * LENGTH**3 / E / w / t * sqrt(
        V**2 / t**4 + H**2 / w**4
    )

def make_cantilever_beam():
    md = core.Model(name = "Cantilever Beam") >> \
         cp.cp_function(
             fun=function_area,
             var=["w", "t"],
             out=["c_area"],
             name="cross-sectional area"
         ) >> \
         cp.cp_function(
             fun=function_stress,
             var=["w", "t", "H", "V", "E", "Y"],
             out=["g_stress"],
             name="limit state: stress"
         ) >> \
         cp.cp_function(
             fun=function_displacement,
             var=["w", "t", "H", "V", "E", "Y"],
             out=["g_disp"],
             name="limit state: displacement"
         ) >> \
         cp.cp_bounds(
             w=(2, 4),
             t=(2, 4)
         ) >> \
         cp.cp_marginals(
             H={"dist": "norm", "loc": MU_H, "scale": TAU_H, "sign": +1},
             V={"dist": "norm", "loc": MU_V, "scale": TAU_V, "sign": +1},
             E={"dist": "norm", "loc": MU_E, "scale": TAU_E, "sign":  0},
             Y={"dist": "norm", "loc": MU_Y, "scale": TAU_Y, "sign": -1}
         )

    return md
