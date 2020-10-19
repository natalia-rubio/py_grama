# -*- coding: utf-8 -*-
"""
Natalia Rubio
Oct 2020
Two-Level Monte Carlo Estimator

inputs:
    md1f1m -> grama model (one function, one model)
        md inputs:
                lev -> level on which to sub-estimate
                x -> point at which to evaluate
        model outputs:
                P -> model evaluation at x
                cost -> total cost of model evaluation
    N0 -> initial number of samples to evaluate on both levels
    eps -> desired overall standard deviation

outputs:
    P -> estimator result
    Nlev -> total samples per level
    Vlev -> variance per level
    Clev -> cost per level
    its -> number of iterations to reach target standard deviation
"""
def tlmc_1f1m(md, N0, eps):
    
    import numpy as np
    import grama as gr
    X = gr.Intention()
    
    md1f1m = gr.make_tlmc_model_1f1m()
    
    # Check that md is OK --> same # inputs/outputs
    # Check inputs
    try:
        r = md.functions[0].func(0,0)
    except TypeError:
        print('Input model must have 2 inputs: level and point at which to evaluate.')
    
    # Check outputs
    r = md.functions[0].func(0,0)
    if len(r) != 2:
        raise ValueError ('Level 0 function must have 2 outputs: result and cost.')
        r = md.functions[0].func(1,0)
    if len(r) != 2:
        raise ValueError ('Level 1 function must have 2 outputs: result and cost.')
        
    # Check that md has 1 function
    if len(md.functions) != 1:
        raise ValueError('Input model must have 1 function.')
    

    # make sure N0 and eps are greater than 0
    if ((N0 <= 0) | (eps <= 0)): # make sure N0 and eps are greater than 0
        raise ValueError('N0 and eps must be > 0.') 
        
        
        
    its = 0 # initialize iteration counter
        
    Nlev = np.zeros((1,2)) # samples taken per level (initialize)
    dNlev = np.array([[N0, N0]]) # samples left to take per level (initialize)
    Vlev = np.zeros((1,2)) # variance per level (initialize)
    sumlev = np.zeros((2,2)) # sample results per level (initialize)
    costlev = np.zeros((1,2)) # total cost per level (initialize)

    while np.sum(dNlev) > 0: # check if there are samples left to be evaluated
        for lev in range(2):
            if dNlev[0,lev] > 0: # check if there are samples to be evaluated on level 'lev'                
                df_mc_lev = md1f1m >> gr.ev_monte_carlo(n=dNlev[0,lev], df_det = gr.df_make(level = lev))              
                if lev > 0:
                    df_prev = df_mc_lev >> gr.tf_select(gr.columns_between("x","level")) >> gr.tf_mutate(level = X.level-1)
                    df_mc_lev_prev = md1f1m >> gr.ev_df(df_prev)
                    Y = df_mc_lev.P - df_mc_lev_prev.P
                    C = sum(df_mc_lev.cost) + sum(df_mc_lev_prev.cost)
                else: 
                    Y = df_mc_lev.P
                    C = sum(df_mc_lev.cost)
                    
                cost = C
                sums = [sum(Y), sum(Y**2)]

                Nlev[0,lev] = Nlev[0,lev] + dNlev[0,lev] # update samples taken on level 'lev'
                sumlev[0, lev] = sumlev[0, lev] + sums[0] # update sample results on level 'lev'
                sumlev[1, lev] = sumlev[1, lev] + sums[1] # update sample results on level 'lev'
                costlev[0, lev] = costlev[0, lev] + cost # update total cost on level 'lev'
                
        mlev = np.abs(sumlev[0,:]/Nlev) # expected value per level
        Vlev = np.maximum(0, (sumlev[1,:]/Nlev - mlev**2)) # variance per level
        Clev = costlev/Nlev # cost per result per level
        
        mu = eps**(-2) * sum(np.sqrt(Vlev*Clev)) # Lagrange multiplier to minimize variance for a fixed cost
        Ns = np.ceil(mu * np.sqrt(Vlev/Clev)) # optimal number of samples per level
        dNlev = np.maximum(0, Ns-Nlev) # update samples left to take per level
        its += 1 # update counter
    
    P = np.sum(sumlev[0,:]/Nlev)# evaluate two-level estimator
    return P, Nlev, Vlev, its