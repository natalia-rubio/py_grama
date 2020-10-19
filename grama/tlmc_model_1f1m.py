# -*- coding: utf-8 -*-
"""
Natalia Rubio
Oct 2020
Two-Level Monte Carlo Sub-Estimator Grama Model

(one function, one model)

model inputs:
    level -> level on which to sub-estimate
    x -> point at which to evaluate
model outputs:
    P -> model evaluation at x
    cost -> total cost of model evaluation
"""
def make_tlmc_model_1f1m():
    
    import numpy as np
    import grama as gr
    
    def fun_lev(args): # evaluate level "lev" function at x, record cost
        level, x = args
        def fun_lev0(x): # evaluate level 0 function at x, record cost
            P = x
            cost = 1
            return P, cost
    
        def fun_lev1(x): # evaluate level 1 function at x, record cost
            P = np.sin(x)
            cost = 2
            return P, cost
    
        if level == 0:
            fun = fun_lev0
        elif level == 1:
            fun = fun_lev1
        else: 
            raise ValueError ('Input level too high')
        P, cost = fun(x)
            
        return P, cost

    md = gr.Model(name = "tlmc_model_1f1m") >> \
    gr.cp_function(
        fun = fun_lev,
        var = ["level", "x"],
        out = ["P" , "cost"],
        name = ["level function"] ) >> \
    gr.cp_marginals(
        x = {"dist": "norm", "loc": 0, "scale": 1, "sign": +1}) >> \
    gr.cp_copula_independence
    
    return md

    

