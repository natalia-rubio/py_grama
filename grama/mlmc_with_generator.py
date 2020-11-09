# -*- coding: utf-8 -*-
"""
Natalia Rubio
Nov 2020
Multi-Level Monte Carlo Estimator with py_grama
------------------------------------------------------
heat_equation:
    Generator function representing temperature as a function of distance x along a bar.  
    Both ends of the bar have a fixed temperature and there is a heat source in the bar represented by -3*sin(3 pi x / 2).
    The resolution of the generated function changes with the number of elements.
inputs:
    N: number of elements in rod
outputs:
    f_temp : (function) linear interpolation of solution to heat equation over at N+1 points across the bar
            f_temp(x) = approximated temperature at x
------------------------------------------------------
md_gen:
    Generates a list of models at different resolutions (levels) to be used in a multi-level Monte Carlo estimator.
inputs:
    fun_gen: generator function that produces new functions at different resolutions
    num_levs: number of models to be generated
outputs:
    models: list of models, each containing a function at a different resolution
    costs: cost of each model (based on order of resolution)
-----------------------------------------------------------
mlmc_gen:
    Multi-level Monte Carlo estimator (based heavily on M Giles code).  
    Takes in a generator funtion and generates models for desired levels.
inputs:
    N0: initital number of samples on each level
    eps: target standard deviation
    fun_gen: generator function
    num_levs: number of levels to include in estimator
outputs:
    P: estimator result
    Nlev: number of samples on each level
    Vlev: variance on each level
    its: iteration count
-----------------------------------------------------------
"""
def heat_equation(N):
    import numpy as np
    from scipy.interpolate import interp1d
    if np.mod(N,2) != 0:
        raise ValueError ('N must be even to have a point at 0.')
    T_o = 0 # temperature at x = 0
    T_f = 1 # temperature at x = 1
    A = -2*np.eye(N-1) + np.diag(np.ones(N-2),1) + np.diag(np.ones(N-2),-1)
    b = (N**-2)*(-3)*np.sin((3*np.pi/2)*np.linspace(1/N,(N-1)/N,(N-1)))
    b[0] = b[0]-T_o
    b[N-2] = b[N-2]-T_f
    T = np.linalg.solve(A,b)
    T = np.concatenate(([T_o , T_o], T, [T_f , T_f]),axis=0)
    x_vec =np.linspace(1/N,(N-1)/N,(N-1))
    x_vec = np.concatenate(([-100 , 0], x_vec, [1 , 101]),axis=0)
    f_temp = interp1d(x_vec,T)
    return f_temp
def md_gen(fun_gen, num_levs):
    import grama as gr
    models = list() 
    costs = list()
    for i in range(num_levs):   
        md = gr.Model(name = ("md{}".format(i))) >> \
        gr.cp_function(
            fun = fun_gen(10**(i+1)),
            var = ["x"],
            out = ["P"],
            name = ["level 0 function"] ) >> \
        gr.cp_marginals(
            x = {"dist": "norm", "loc": 0.5, "scale": 0.2, "sign": +1}) >> \
        gr.cp_copula_independence
            
        models.append(md)
        md_cost = i+1
        costs.append(md_cost)
    return models, costs    
def mlmc_gen(N0, eps, fun_gen, num_levs):
    import numpy as np
    import grama as gr
    
    # generate models
    [models, costs] = md_gen(fun_gen, num_levs)
    
    # from timeit import default_timer as timer
    # costs = list()
    # for i in range(num_levs):
    #     start = timer()
    #     models[i] >> gr.ev_monte_carlo(10)
    #     end = timer()
    #     md_cost = end - start
    #     costs.append(md_cost)
        
    # print("costs: ",costs)
    # print(models)
        
    its = 0 # initialize iteration counter
        
    Nlev = np.zeros(num_levs) # samples taken per level (initialize)
    dNlev = N0*np.ones(num_levs) # samples left to take per level (initialize)
    sumlev = np.zeros((2,num_levs)) # sample results per level (initialize)
    costlev = np.zeros(num_levs) # total cost per level (initialize)

    while np.sum(dNlev) > 0: # check if there are samples left to be evaluated
        for lev in range(num_levs):
            if dNlev[lev] > 0: # check if there are samples to be evaluated on level 'lev'                
                df_mc_lev = models[lev] >> gr.ev_monte_carlo(dNlev[lev]) 
                if lev > 0:
                    df_prev = df_mc_lev >> gr.tf_select(gr.starts_with("x"))
                    df_mc_lev_prev = models[lev-1] >> gr.ev_df(df_prev)
                    Y = df_mc_lev.P - df_mc_lev_prev.P
                    cost = (costs[lev]+costs[lev-1])*dNlev[lev]
                else:
                    Y = df_mc_lev.P
                    cost = costs[lev]*dNlev[lev]
                    
                sums = [Y.sum(), (Y**2).sum()]

                Nlev[lev] = Nlev[lev] + dNlev[lev] # update samples taken on level 'lev'
                sumlev[0, lev] = sumlev[0, lev] + sums[0] # update sample results on level 'lev'
                sumlev[1, lev] = sumlev[1, lev] + sums[1] # update sample results on level 'lev'
                costlev[lev] = costlev[lev] + cost # update total cost on level 'lev'
                
        mlev = np.abs(sumlev[0,:]/Nlev) # expected value per level
        Vlev = np.maximum(0, (sumlev[1,:]/Nlev - mlev**2)) # variance per level
        Clev = costlev/Nlev # cost per result per level
        
        mu = eps**(-2) * sum(np.sqrt(Vlev*Clev)) # Lagrange multiplier to minimize variance for a fixed cost
        Ns = np.ceil(mu * np.sqrt(Vlev/Clev)) # optimal number of samples per level
        dNlev = np.maximum(0, Ns-Nlev) # update samples left to take per level
        its += 1 # update counter
    
    P = np.sum(sumlev[0,:]/Nlev)# evaluate two-level estimator
    return P, Nlev, Vlev, its