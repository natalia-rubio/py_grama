__all__ = [
    "comp_function",
    "cp_function",
    "comp_bounds",
    "cp_bounds",
    "comp_marginals",
    "cp_marginals"
]

from .. import core
from .. import evals
from .. import fitting
from ..tools import pipe
from toolz import curry

## Model Building Interface (MBI) tools
##################################################
# Add a lambda function
# -------------------------
@curry
def comp_function(
    model, fun=None, var=None, out=None, name=None
):
    """Add a function to a model.

    @param model Model to compose
    @param fun Function taking R^d -> R^r
    @param var List of variable names or number of inputs
    @param out List of output names or number of outputs

    @type model gr.model
    @type fun function
    @type var list(string)
    @param out list(string)

    @returns New model with added function
    @rtype gr.model

    @pre (len(var) == d) | (var == d)
    @pre (len(out) == r) | (var == r)
    """
    model_new = model.copy()

    # Check inputs
    if fun is None:
        raise ValueError("`fun` must be a valid function")

    if name is None:
        name = fun.__name__

    # Create variable names, if necessary
    if isinstance(var, int):
        i0 = model_new.n_var
        i1 = model_new.n_var + var
        var = ["x{}".format(i) for i in range(i0, i1)]
    elif var is None:
        raise ValueError("`var` must be list or int")

    # Create output names, if necessary
    if isinstance(out, int):
        i0 = model_new.n_out
        i1 = model_new.n_out + out
        out = ["y{}".format(i) for i in range(i0, i1)]
    elif out is None:
        raise ValueError("`out` must be list or int")

    ## Add new function
    model_new.functions.append(
        core.Function(fun, var, out, name)
    )

    model_new.update()
    return model_new

@pipe
def cp_function(*args, **kwargs):
    return comp_function(*args, **kwargs)

# Add bounds
# -------------------------
@curry
def comp_bounds(model, **kwargs):
    """Add variable bounds to a model.

    Bounds are specified by iterable; the model variable name is specified by the
    keyword argument name.

    @param model Model to modify
    @param var Bound information

    @type model gr.model
    @type var iterable

    @returns Model with new marginals
    @rtype gr.model

    @pre len(var) >= 2

    Examples

    import grama as gr
    md = gr.Model() >> \
        cp_function(
            lambda x: x[0] + x[1],
            var=["x0", "x1"],
            out=1
        ) >> \
        cp_bounds(
            x0=(-1, 1),
            x1=(0, np.inf)
        )

    """
    new_model = model.copy()

    ## Parse keyword arguments
    for key, value in kwargs.items():
        ## Add new bound
        new_model.domain.bounds[key] = [value[0], value[1]]

    new_model.update()
    return new_model

@pipe
def cp_bounds(*args, **kwargs):
    return comp_bounds(*args, **kwargs)

# Add marginals
# -------------------------
@curry
def comp_marginals(model, **kwargs):
    """Add marginals to a model.

    Marginals are specified by dictionary entries; the model
    variable name is specified by the keyword argument name.

    @param model Model to modify
    @param var Marginal information

    @type model gr.model
    @type var dict

    @returns Model with new marginals
    @rtype gr.model

    TODO:
    - Implement marginals other than MarginalNamed

    Examples

    import grama as gr
    print(gr.valid_dist.keys()) # Print list of implemented marginals

    md = gr.Model() >> \
        cp_function(
            lambda x: x[0] + x[1],
            var=["x0", "x1"],
            out=1
        ) >> \
        cp_marginals(
            x0={"dist": "norm", "loc": 0, "scale": 1}
        )

    """
    new_model = model.copy()

    ## Parse keyword arguments
    for key, value in kwargs.items():
        value_copy = value.copy()

        ## Check for named marginal
        try:
            dist = value_copy.pop("dist")
        except KeyError:
            raise NotImplementationError("Non-named marginals not implemented; please provide a valid 'dist' key")

        try:
            sign = value_copy.pop("sign")
        except KeyError:
            sign = 0

        new_model.density.marginals[key] = core.MarginalNamed(
            sign=sign,
            d_name=dist,
            d_param=value_copy
        )

    new_model.update()
    return new_model

@pipe
def cp_marginals(*args, **kwargs):
    return comp_marginals(*args, **kwargs)