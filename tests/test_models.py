import numpy as np
import pandas as pd
import unittest
import io
import sys

from context import grama as gr
from context import models

## Test the built-in models
##################################################
class TestModels(unittest.TestCase):

    def setUp(self):
        pass

    def test_make(self):
        ## Models build
        md_cantilever_beam = models.make_cantilever_beam()
        md_ishigami = models.make_ishigami()
        md_linear_normal = models.make_linear_normal()
        md_plane_laminate = models.make_composite_plate_tension([0])
        md_plate_buckling = models.make_plate_buckle()
        md_poly = models.make_poly()
        md_test = models.make_test()
        md_trajectory_linear = models.make_trajectory_linear()

        ## Models evaluate
        df_cantilever = md_cantilever_beam >> gr.ev_nominal(df_det="nom")
        df_ishigami = md_ishigami >> gr.ev_nominal(df_det="nom")
        df_ln = md_linear_normal >> gr.ev_nominal(df_det="nom")
        df_plane = md_plane_laminate >> gr.ev_nominal(df_det="nom")
        df_plate = md_plate_buckling >> gr.ev_nominal(df_det="nom")
        df_poly = md_poly >> gr.ev_nominal(df_det="nom")
        df_test = md_test >> gr.ev_nominal(df_det="nom")
        df_traj = md_trajectory_linear >> gr.ev_nominal(df_det="nom")
