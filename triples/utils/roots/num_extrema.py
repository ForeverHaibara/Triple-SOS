"""
This module provides heuristic functions to optimize a polynomial
with inequality and equality constraints NUMERICALLY.
"""
from typing import List, Union, Tuple, Dict, Callable
import warnings

import numpy as np
import sympy as sp
from sympy import Poly, Expr, Symbol, lambdify
from scipy.optimize import OptimizeResult, shgo

from .extrema import _infer_symbols, polylize_input
from .roots import Root

DEFAULT_SHGO_KWARGS = { # default values
    'n': 256,
    'iters': 2,
    'sampling_method': 'sobol',
    'options': {'maxiter': 4}
}

class NumerFuncWrapper:
    def __init__(self, symbols, free_symbols=None, embedding=None, **kwargs):
        self.symbols = symbols
        self.free_symbols = free_symbols
        self.embedding = embedding
        self.free_symbols_to_symbols = self._make_embedding(embedding, **kwargs)
        self.jacobian = self._make_jacobian(embedding, **kwargs)

    def __len__(self):
        """Number of free variables."""
        return len(self.symbols) if self.free_symbols is None else len(self.free_symbols)

    def _make_embedding(self, embedding, **kwargs):
        symbols, free_symbols = self.symbols, self.free_symbols
        if free_symbols is None:
            return None
        embedding = {k: lambdify(free_symbols, e, **kwargs) for k, e in embedding.items()}
        embedding_funcs = []
        for i, symbol in enumerate(symbols):
            if symbol in embedding:
                func = embedding[symbol]
                embedding_funcs.append(lambda args, f=func: f(*args))
            elif symbol in free_symbols:
                ind = free_symbols.index(symbol)
                embedding_funcs.append(lambda args, i=ind: args[i])
            else:
                raise ValueError(f'Undefined non-free symbol {symbol}.')
        new_func = lambda x: np.array([f(x) for f in embedding_funcs])
        return new_func

    def _make_jacobian(self, embedding, **kwargs):
        if self.free_symbols is None:
            return None
        diff_wrt = []
        embedding = embedding.copy()
        for k in self.symbols:
            if not (k in embedding):
                embedding[k] = k
        for v in self.free_symbols:
            embedding_diff = {k: e.diff(v) for k, e in embedding.items()}
            diff_wrt.append(self._make_embedding(embedding_diff, **kwargs))
        new_func = lambda x: np.vstack([f(x) for f in diff_wrt])
        return new_func

    def wrap(self, func, jac=False, **kwargs):
        if not isinstance(func, Expr):
            func = func.as_expr()
        _wrapper = self._wrap_f if not jac else self._wrap_jac
        return _wrapper(func, **kwargs)

    def _wrap_f(self, f, **kwargs):
        symbols = self.symbols
        _new_func = lambdify(symbols, f, **kwargs)
        if self.free_symbols is None:
            new_func = lambda args: _new_func(*args)
        else:
            new_func = lambda args: _new_func(*self.free_symbols_to_symbols(args))
        return new_func

    def _wrap_jac(self, f, **kwargs):
        symbols = self.symbols
        jac = [lambdify(symbols, f.diff(s)) for s in symbols]
        if self.free_symbols is None:
            new_func = lambda args: np.array([_(*args) for _ in jac])
        else:
            def new_func(args):
                A = self.jacobian(args)
                new_args = self.free_symbols_to_symbols(args)
                b = np.array([_(*new_args) for _ in jac])
                return A @ b
        return new_func


def _update_dict(d1: Dict, d2: Dict) -> Dict:
    """Write dict d2 to d1 recursively. The modification is in-place."""
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1:
            d1[k] = _update_dict(d1[k], v)
        else:
            d1[k] = v
    return d1

def _numeric_optimize_shgo(poly: Union[Poly, Expr], ineq_constraints: List[Union[Poly, Expr]], eq_constraints: List[Union[Poly, Expr]],
        symbols: List[Symbol], free_symbols: List[Symbol]=None, embedding: Dict[Symbol, Expr]=None,
        restore: bool=True, shgo_kwargs: Dict = {}
    ) -> OptimizeResult:
    """
    Internal function to numerically optimize a polynomial with given inequality and equality constraints
    by calling scipy.optimize.shgo.
    """
    if isinstance(poly, Poly) and poly.total_degree() <= 0:
        res = OptimizeResult(nit=0, nfev=0, nlfev=0, nljev=0, nlhev=0, success=True,
                status='success', message='Optimization terminated successfully.')
        nvars = len(symbols)
        res.x = np.array([0.] * nvars)
        res.fun = float(poly.coeff_monomial((0,)*nvars))
        res.xl = res.x.reshape((1, nvars))
        res.funl = np.array([res.fun])
        return res

    wrapper = NumerFuncWrapper(symbols, free_symbols=free_symbols, embedding=embedding)

    f = wrapper.wrap(poly)
    f_jac = wrapper.wrap(poly, jac=True)

    def _to_shgo_con(con, type='ineq'):
        return {'type': type, 'fun': wrapper.wrap(con),
                'jac': wrapper.wrap(con, jac=True)}
    constraints = [_to_shgo_con(_, 'ineq') for _ in ineq_constraints] \
                + [_to_shgo_con(_, 'eq') for _ in eq_constraints]

    shgo_kwargs = _update_dict(
        _update_dict({'options': {'jac': f_jac}}, DEFAULT_SHGO_KWARGS), shgo_kwargs
    )

    with warnings.catch_warnings():
        # disable RuntimeWarning like invalid zero division
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r".*_shgo")
        # try:
        result = shgo(f, bounds=[(0, 1)] * len(wrapper),
                    constraints=constraints, **shgo_kwargs)
        # except Exception as e:
        #     if isinstance(e, (TypeError, ValueError, KeyboardInterrupt)):
        #         raise e
        #     return []

    if restore and (wrapper.free_symbols is not None):
        if hasattr(result, 'x'):
            result.x = wrapper.free_symbols_to_symbols(result.x)
        if hasattr(result, 'xl'):
            result.xl = wrapper.free_symbols_to_symbols(result.xl.T).T

    return result


def _numeric_optimize_poly(poly: Poly, ineq_constraints: List[Poly], eq_constraints: List[Poly], symbols: List[Symbol],
        shgo_kwargs: Dict = {}
    ) -> OptimizeResult:
    """
    Internal function to optimize a polynomial with inequality and equality constraints.
    """
    free_symbols, embedding = None, None

    if poly.is_homogeneous and all(_.is_homogeneous for _ in ineq_constraints)\
        and all(_.is_homogeneous for _ in eq_constraints):
        gen = symbols[-1]
        free_symbols = symbols[:len(symbols)-1]
        embedding = {gen: 1 - sum(symbols[:-1])}

    result = _numeric_optimize_shgo(poly, ineq_constraints, eq_constraints, symbols,
                    free_symbols=free_symbols, embedding=embedding, shgo_kwargs=shgo_kwargs)
    return result


def numeric_optimize_poly(poly: Union[Poly, Expr], ineq_constraints: List[Union[Poly, Expr]] = [], eq_constraints: List[Union[Poly, Expr]] = [],
        symbols: List[Symbol] = None, objective: str = 'min', return_type: str='tuple', shgo_kwargs: Dict = {}
    ) -> List[Root]:
    """
    Numerically polynomial optimize a polynomial with given inequality and equality constraints
    using heuristic methods. It uses incomplete algorithm to balance the efficiency
    and effectiveness.

    Parameters
    ----------
    poly : Poly or Expr
        The polynomial to be optimized.
    ineq_constraints : List[Poly or Expr]
        The inequality constraints, G1, G2, ... (>= 0).
    eq_constraints : List[Poly or Expr]
        The equality constraints, H1, H2, ... (== 0).
    symbols : List[Symbol]
        The symbols to optimize on. If None, it is inferred from the polynomial and constraints.
    objective : str
        The objective to optimize. Either 'min' or 'max' or 'all'.
        When 'min', the function returns the global minimizers
        When 'max', the function returns the global maximizers.
        When 'all', the function returns all recognized extrema.
    return_type : str
        The returned type, should be one of "tuple", "root", "dict" or "result".
    shgo_kwargs: Dict
        Extra parameters passed into `scipy.optimize.shgo` that overwrite
        the default configurations.
    """
    symbols = _infer_symbols(symbols, poly, ineq_constraints, eq_constraints)
    if len(symbols) == 0:
        return []
    if not (objective in ('min', 'max', 'all')):
        raise ValueError('Objective must be either "min" or "max" or "all".')
    if not (return_type in ('root', 'tuple', 'dict', 'result')):
        raise ValueError('Return type must be either "root" or "tuple" or "dict" or "result".')

    poly, ineq_constraints, eq_constraints = polylize_input(
        poly, ineq_constraints, eq_constraints, symbols=symbols,
        check_poly=lambda p: p.domain.is_Numerical
    )

    if poly in eq_constraints:
        # This often happens when solving for the equality case of a nonnegative poly.
        eq_constraints = eq_constraints.copy()
        eq_constraints.remove(poly)

        shgo_kwargs = _update_dict({'options': {'f_min': 0}}, shgo_kwargs)

    if objective == 'max':
        poly = -poly

    result = _numeric_optimize_poly(poly, ineq_constraints, eq_constraints, symbols,
                                    shgo_kwargs=shgo_kwargs)

    if return_type == 'result':
        return result

    points = result.xl.tolist() if hasattr(result, 'xl') else []
    points = [tuple(_) for _ in points]

    if return_type == 'root':
        points = [Root(_) for _ in points]
    elif return_type == 'dict':
        points = [dict(zip(symbols, _)) for _ in points]
    return points