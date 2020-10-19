# -*- coding: utf-8 -*-
"""
Natalia Rubio
Oct 2020
Two-Level Monte Carlo Sub-Estimator Grama Model

model inputs:
    lev -> level on which to sub-estimate
    x -> point at which to evaluate
model outputs:
    P -> model evaluation at x
    cost -> total cost of model evaluation
"""
def make_tlmc_model_2f1m():
    
    import numpy as np
    import grama as gr
    
    def fun_lev0(x): # evaluate level 0 function at x, record cost
            P = x
            cost = 1
            return P, cost
    
    def fun_lev1(x): # evaluate level 1 function at x, record cost
            P = np.sin(x)
            cost = 2
            return P, cost

    md = gr.Model(name = "tlmc_model") >> \
    gr.cp_function(
        fun = fun_lev0,
        var = ["x"],
        out = ["P0" , "cost0"],
        name = ["level 0 function"] ) >> \
    gr.cp_function(
        fun = fun_lev1,
        var = ["x"],
        out = ["P1" , "cost1"],
        name = ["level 1 function"] ) >> \
    gr.cp_marginals(
        x = {"dist": "norm", "loc": 0, "scale": 1, "sign": +1}) >> \
    gr.cp_copula_independence
    
    return md
