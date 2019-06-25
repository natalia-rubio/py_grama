##
#
# References:
# C.T. Sun "Mechanics of Aircraft Structures" (1998)

# from .. import core
from grama import core
import numpy as np
import itertools

## Composite properties
##################################################
# Inspired by Arevo data
E1_M   = 114e9
E2_M   =   7e9
G12_M  =   4e9
nu12_M = 0.45

E1_CV    = np.sqrt(0.02)
E2_CV    = np.sqrt(0.08)
G12_CV   = np.sqrt(0.1)
nu12_SIG = nu12_M * np.sqrt(0.08)

T_NOM    = 1e-3            # Nominal thickness
T_PM     = T_NOM * 0.01    # +/- ply thickness
THETA_PM = 3 * np.pi / 180 # +/- angle tolerance

SIG_11_T_M = 1.4e9 # Tensile strength
SIG_11_C_M = 0.5e9 # Compressive strength
SIG_12_M_M =  62e6 # Interlaminar strength

SIG_11_T_CV = 0.06
SIG_11_C_CV = 0.06
SIG_12_M_CV = 0.07

## Laminate Composite Analysis
##################################################
# Construct stress transformation matrix
def make_Ts(theta):
    """
    Usage
        Ts    = make_Ts(theta)
    Arguments
        theta = angle of rotation; in radians
    Returns
        Ts    = stress transformation matrix
    """
    C = np.cos(theta)
    S = np.sin(theta)

    return np.array([
        [  C**2,  S**2,   2 * S * C],
        [  S**2,  C**2,  -2 * S * C],
        [-S * C, S * C, C**2 - S**2]
    ])

# Construct strain transformation matrix
def make_Te(theta):
    """
    Usage
        Te    = make_Te(theta)
    Arguments
        theta = angle of rotation; in radians
    Returns
        Te    = strain transformation matrix
    """
    C = np.cos(theta)
    S = np.sin(theta)

    return np.array([
        [      C**2,      S**2,       S * C],
        [      S**2,      C**2,      -S * C],
        [-2 * S * C, 2 * S * C, C**2 - S**2]
    ])

# Construct stiffness matrix in lamina coordinate system
def make_Q(param):
    """
    Usage
        Q     = make_Q(param)
    Arguments
        param = array of parameters
              = [E1, E2, nu12, G12]
    Returns
        Q     = stiffness matrix in lamina coordinate system
    """
    E1, E2, nu12, G12 = param

    nu21 = nu12 * E2 / E1

    Q11 = E1 / (1 - nu12 * nu21)
    Q12 = nu12 * E2 / (1 - nu12 * nu21)
    Q21 = Q12
    Q22 = E2 / (1 - nu12 * nu21)
    Q66 = G12

    return np.array([
        [ Q11, Q12,   0],
        [ Q21, Q22,   0],
        [   0,   0, Q66]
    ])

# Construct stiffness matrix in global coordinate system
def make_Qb(param, theta):
    """
    Usage
        Qb    = make_Qb(param)
    Arguments
        param = array of parameters
              = [E1, E2, nu12, G12]
        theta = lamina angle; in radians
    Returns
        Qb    = stiffness matrix in global coordinate system
    """
    Ts_inv = make_Ts(-theta)
    Te     = make_Te(+theta)
    Q      = make_Q(param)

    return np.dot(Ts_inv, np.dot(Q, Te))

# Construct extensional stiffness matrix
def make_A(Param, Theta, T):
    """
    Usage
        A     = make_A(Param, Theta, T)
    Arguments
        Param = array of array of parameters
              = [[E1, E2, nu12, G12]_1,
                          . . .
                 [E1, E2, nu12, G12]_k]
        Theta = array of lamina angles
              = [theta_1, ..., theta_k]
        T     = array of thicknesses
              = [t_1, ..., t_k]
    Returns
        A     = extensional stiffness matrix

    @pre len(Param) == len(Theta)
    @pre len(Param) == len(T)
    """
    n_k = len(Param)
    Qb_all = np.zeros((3, 3, n_k))

    for ind in range(n_k):
        Qb_all[:, :, ind] = make_Qb(Param[ind], Theta[ind]) * T[ind]

    return np.sum(Qb_all, axis = 2)

# Compute stresses under uniaxial tension
def uniaxial_stresses(Param, Theta, T):
    """
    Return the lamina stresses for a unit uniaxial tension along the fiber direction

    Usage
        Stresses = uniaxial_stresse(Param, Theta, T)
    Arguments
        Param = array of array of parameters
              = [[E1, E2, nu12, G12]_1,
                          . . .
                 [E1, E2, nu12, G12]_k]
        Theta = array of lamina angles
              = [theta_1, ..., theta_k]
        T     = array of thicknesses
              = [t_1, ..., t_k]
    Returns
        Stresses = state of stress in each lamina
                 = [[\sigma_11, \sigma_22, \sigma_12]_1
                                 . . .

                 =  [\sigma_11, \sigma_22, \sigma_12]_k]
    """
    ## Solve for state of strain with uniaxial loading
    Ab = make_A(Param, Theta, T)
    strains = np.linalg.solve(Ab, np.array([1, 0, 0]))

    ## Solve for each lamina state of strain in local coordinates
    n_k      = len(Param)
    Stresses = np.zeros((n_k, 3))
    for ind in range(n_k):
        ## Solve in global coordinates
        stress_global = np.dot(make_Qb(Param[ind], Theta[ind]), strains)
        ## Convert to lamina coordinates
        Stresses[ind] = np.dot(make_Ts(Theta[ind]), stress_global)

    return Stresses

# Uniaxial tension limit state functions
def uniaxial_stress_limit(X):
    """
    Evaluate stress limit states for uniaxial tension. Dimensionality of problem
    is inferred from size of X.

    Usage
        g_stress = uniaxial_stress_limit(X)
    Arguments
        X = array of composite laminate properties
          = [[E1, E2, nu12, G12, theta, t, \sigma_11_tensile, \sigma_11_comp, \sigma_12_max]_1
                                  .  .  .
          =  [E1, E2, nu12, G12, theta, t, \sigma_11_tensile, \sigma_11_comp, \sigma_12_max]_k]
    Returns
        g_stress = array of limit state values
                 = [g_sigma_11_1, g_sigma_12_1,
                            .  .  .
                    g_sigma_11_k, g_sigma_12_k]
    @pre X.shape == (k, 8)
    """
    k = X.shape[0]
    ## Unpack inputs
    Param = X[:, 0:4]
    Theta = X[:, 4]
    T     = X[:, 5]
    Sigma_max = X[:, 6:]

    ## Evaluate stress
    ## TODO: Scale stresses by given load
    Stresses = uniaxial_stresses(Param, Theta, T)

    ## Construct limit state
    g_limit = np.zeros()
    g_limit[:, (0, 2)] = Sigma_max[:, (0,2)] - Stresses[:, (0,2)]
    g_limit[:, 1]      = Sigma_max[:, 1]     + Stresses[:, 1]

    return g_limit

## Random variable model
##################################################
# Helper function to create domain for given ply
def make_domain(
        Theta_nom,
        T_nom= T_NOM
):
    """
    Helper function to construct domain object for composite panel

    Usage
        domain = make_domain(Theta_nom)
        domain = make_domain(Theta_nom, T_nom= T_nom)
    Arguments
        Theta_nom = nominal lamina angles; determines dimensionality of problem
                  = [theta_1, ..., theta_k]
    Keyword Arguments
        t_nom     = nominal ply thicknesses
                  = [t_1, ..., t_k] OR
                  = t_nom
    Returns
        domain = grama domain object
    """
    k = len(Theta_nom)
    inputs = list(itertools.chain.from_iterable([
        ["E1_{}".format(i),
         "E2_{}".format(i),
         "nu12_{}".format(i),
         "G12_{}".format(i),
         "theta_{}".format(i),
         "t_{}".format(i),
         "sigma_11_tensile_{}".format(i),
         "sigma_11_comp_{}".format(i),
         "sigma_12_max_{}".format(i)] for i in range(k)
    ]))
    bounds = list(itertools.chain.from_iterable([
       [{"E1_{}".format(i): [0, +np.Inf]},
        {"E2_{}".format(i): [0, +np.Inf]},
        {"nu12_{}".format(i): [-np.Inf, +np.Inf]},
        {"G12_{}".format(i): [0, +np.Inf]},
        {"theta_{}".format(i): [-np.pi/2, +np.pi/2]},
        {"t_{}".format(i): [-np.Inf, +np.Inf]},
        {"sigma_11_tensile_{}".format(i): [0, +np.Inf]},
        {"sigma_11_comp_{}".format(i): [0, +np.Inf]},
        {"sigma_12_max_{}".format(i): [0, +np.Inf]}] for i in range(k)
    ]))

    return core.domain_(
        hypercube = True,
        inputs = inputs,
        bounds = bounds
    )

# Helper function to create density for given ply
def make_density(
        Theta_nom,
        T_nom = T_NOM,
):
    """
    Helper function to construct density object for composite panel

    Usage
        density = make_density(Theta_nom)
        density = make_density(Theta_nom, t_nom= t_nom)
    Arguments
        Theta_nom = nominal lamina angles; determines dimensionality of problem
                      = [theta_1, ..., theta_k]
    Keyword Arguments
        t_nom     = nominal ply thicknesses
                      = [t_1, ..., t_k] OR
                      = t_nom
    Returns
        density = grama density object
    """
    ## Setup
    k = len(Theta_nom)
    if not isinstance(T_nom, list):
        T_nom= [T_nom] * k

    ## TODO: Implement pdf
    pdf = lambda X: 1
    pdf_factors = list(itertools.chain.from_iterable([
        ["lognorm",                   # E1
         "lognorm",                   # E2
         "norm",                      # nu12
         "lognorm",                   # G12
         "unif",                      # theta
         "unif",                      # t
         "lognorm",                   # sigma_11_tensile
         "lognorm",                   # sigma_11_comp
         "lognorm"] for i in range(k) # sigma_12_max
    ]))
    pdf_param = list(itertools.chain.from_iterable([
       [{"loc": E1_M, "s": E1_CV, "scale": 1},                                # E1
        {"loc": E2_M, "s": E2_CV, "scale": 1},                                # E2
        {"loc": nu12_M, "scale": nu12_SIG},                                   # nu12
        {"loc": G12_M, "s": G12_CV, "scale": 1},                              # G12
        {"lower": Theta_nom[i] - THETA_PM, "upper": Theta_nom[i] + THETA_PM}, # theta
        {"lower": T_nom[i] - T_PM, "upper": T_nom[i] + T_PM},                 # t
        {"loc": SIG_11_T_M, "s": SIG_11_T_CV, "scale": 1}, # sigma_11_tensile
        {"loc": SIG_11_C_M, "s": SIG_11_C_CV, "scale": 1}, # sigma_11_comp
        {"loc": SIG_12_M_M, "s": SIG_12_M_CV, "scale": 1}] for i in range(k) # sigma_12_max
    ]))
    ## TODO: Determine proper "conservative" quantile directions!
    pdf_qt_flip = list(itertools.chain.from_iterable([
        [0,                   # E1
         0,                   # E2
         0,                   # nu12
         0,                   # G12
         0,                   # theta
         0,                   # t
         0,                   # sigma_11_tensile
         0,                   # sigma_11_comp
         0] for i in range(k) # sigma_12_max
    ]))
    return core.density_(
        pdf         = pdf,
        pdf_factors = pdf_factors,
        pdf_param   = pdf_param,
        pdf_qt_flip = pdf_qt_flip
    )

## Model class
##################################################

class model_composite_plate_tension(core.model_):
    def __init__(self, Theta_nom, T_nom = T_NOM):
        k = len(Theta_nom)

        super().__init__(
            name = "Composite Plate in Tension",
            function = lambda X: uniaxial_stress_limit(X),
            outputs = list(itertools.chain.from_iterable([
                ["g_sigma_11_{}".format(i), "g_sigma_12_{}".format(i)] for i in range(k)
            ])),
            domain = make_domain(Theta_nom, T_nom = T_nom),
            density = make_density(Theta_nom, T_nom = T_nom)
        )

## Verification
##################################################
if __name__ == "__main__":
    ## Verify the stiffness matrix
    E1   = 140e9
    E2   = 10e9
    G12  = 6.9e9
    nu12 = 0.3

    param = [E1, E2, nu12, G12]
    theta = 45. * np.pi / 180

    Qb = make_Qb(param, theta)
    Sb = np.linalg.inv(Qb)

    ## Example 8.4
    E1   = 180e9    # Pa
    E2   =  10e9    # Pa
    G12  =   7e9    # Pa
    nu12 = 0.3      # [-]
    t    = 0.127e-3 # m

    Theta = np.array([+45, -45, 0, 90, 90, 0, -45, +45]) * np.pi / 180
    Param = np.tile([E1, E2, nu12, G12], reps = (len(Theta), 1))
    T     = np.array([t] * len(Theta))

    A = make_A(Param, Theta, T)

    # Check values; close to Sun (1998)
    print(A / t)

    # Uniaxial (unit) tension
    strain_uniaxial = np.linalg.solve(A, np.array([1, 0, 0]))
    stress_uniaxial = uniaxial_stresses(Param, Theta, T)
