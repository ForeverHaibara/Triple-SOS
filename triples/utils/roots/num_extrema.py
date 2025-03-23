from typing import List, Union, Tuple, Dict, Callable

import numpy as np
import sympy as sp
from sympy import Poly, Expr, Symbol, lambdify
from scipy.stats.qmc import Sobol, Halton
from scipy.optimize import OptimizeResult, shgo

from .extrema import _infer_symbols
import warnings

def _wrap_func(func, symbols, vector_input=True, jac=False, **kwargs):
    if not isinstance(func, Expr):
        func = func.as_expr()
    if not jac:
        _new_func = lambdify(symbols, func, **kwargs)
        if not vector_input:
            new_func = _new_func
        else:
            new_func = lambda args: _new_func(*args)
    else:
        diff = [lambdify(symbols, func.diff(s)) for s in symbols]
        if not vector_input:
            new_func = lambda args: np.array([_(args) for _ in diff])
        else:
            new_func = lambda args: np.array([_(*args) for _ in diff])
    return new_func


def _update_dict(d1: Dict, d2: Dict) -> Dict:
    """Update dictionary d1 with d2 recursively. The modification is in-place on d1."""
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1:
            d1[k] = _update_dict(d1[k], v)
        else:
            d1[k] = v
    return d1

def _numeric_optimize_shgo(poly: Union[Poly, Expr], ineq_constraints: List[Union[Poly, Expr]] = [], eq_constraints: List[Union[Poly, Expr]] = [],
        symbols: List[Symbol] = None, shgo_kwargs: Dict = {}
    ) -> OptimizeResult:
    """
    Internal function to numerically optimize a polynomial with given inequality and equality constraints
    by calling scipy.optimize.shgo.
    """
    
    f = _wrap_func(poly, symbols, vector_input=True)
    f_jac = _wrap_func(poly, symbols, vector_input=True, jac=True)

    def _to_shgo_con(con, type='ineq'):
        return {'type': type, 'fun': _wrap_func(con, symbols, vector_input=True),
                'jac': _wrap_func(con, symbols, vector_input=True, jac=True)}
    constraints = [_to_shgo_con(_, 'ineq') for _ in ineq_constraints] \
                + [_to_shgo_con(_, 'eq') for _ in eq_constraints]

    _default_shgo_kwargs = { # default values
        'n': 256,
        'iters': 5,
        'sampling_method': 'sobol',
        'options': {'jac': f_jac, 'maxiter': 4}
    }
    shgo_kwargs = _update_dict(_default_shgo_kwargs, shgo_kwargs)

    with warnings.catch_warnings():
        # disable RuntimeWarning like invalid zero division
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r".*_shgo")
        try:
            result = shgo(f, bounds=[(0, 1) for _ in symbols],
                        constraints=constraints, **_default_shgo_kwargs)
        except Exception as e:
            if isinstance(e, (TypeError, ValueError, KeyboardInterrupt)):
                raise e
            return []

    return result

def numeric_optimize_poly(poly: Union[Poly, Expr], ineq_constraints: List[Union[Poly, Expr]] = [], eq_constraints: List[Union[Poly, Expr]] = [],
        symbols: List[Symbol] = None, objective: str = 'min', return_dict: bool = False, shgo_kwargs: Dict = {}
    ) -> Union[List[Tuple[Expr]], List[Dict[Symbol, Expr]]]:
    """
    Numerically polynomial optimize a polynomial with given inequality and equality constraints
    using heuristic methods. It uses incomplete algorithm to balance the efficiency
    and effectiveness.
    """
    symbols = _infer_symbols(symbols, poly, ineq_constraints, eq_constraints)
    if len(symbols) == 0:
        return []
    if not (objective in ('min', 'max', 'all')):
        raise ValueError('Objective must be either "min" or "max" or "all".')

    if objective == 'max':
        poly = -poly


    if poly in eq_constraints:
        # This often happens when solving for the equality case of a nonnegative poly.
        eq_constraints = eq_constraints.copy()
        eq_constraints.remove(poly)

        shgo_kwargs = _update_dict({'options': {'f_min': 0}}, shgo_kwargs)

    result = _numeric_optimize_shgo(poly, ineq_constraints, eq_constraints, symbols, shgo_kwargs)

    points = result.xl
    if return_dict:
        points = [dict(zip(symbols, _)) for _ in points]
    return points