"""
This module provides heuristic functions to optimize a polynomial
with inequality and equality constraints NUMERICALLY.
"""
from typing import List, Union, Tuple, Dict, Optional, Callable
from functools import partial
import warnings

import numpy as np
from sympy import Poly, Expr, MatrixBase, Symbol, lambdify, nextprime
from sympy import __version__ as SYMPY_VERSION
from sympy.external.importtools import version_tuple
from sympy.combinatorics import Permutation, PermutationGroup
from scipy import __version__ as SCIPY_VERSION

class OptimizeResult: # this is only for type hint
    ...

from .extrema import _infer_symbols, polylize_input
from .roots import Root
from .root_list import RootList

DEFAULT_SHGO_KWARGS = { # default values
    'n': 256,
    'iters': 2,
    'sampling_method': 'sobol',
    'options': {'maxiter': 4}
}

if tuple(version_tuple(SYMPY_VERSION)) >= tuple(version_tuple('1.12')):
    anonymous_lambdify = partial(lambdify, docstring_limit=-1)
else:
    anonymous_lambdify = lambdify

USE_SCIPY_HALTON = True
if tuple(version_tuple(SCIPY_VERSION)) < tuple(version_tuple('1.7')):
    USE_SCIPY_HALTON = False
class HaltonManual:
    """Implemented Halton sampler for scipy version < 1.7."""
    def __init__(self, d, scramble=False):
        self.d = d
        self.scramble = scramble

    def random(self, n):
        def van_der_corput(n, base=2):
            sequence = []
            for i in range(n):
                num = i
                value = 0
                denominator = 1
                while num > 0:
                    num, remainder = divmod(num, base)
                    denominator *= base
                    value += remainder / denominator
                sequence.append(value)
            return np.array(sequence)

        samples = np.zeros((n, self.d))
        base = 2
        for i in range(self.d):
            if i > 0:
                base = int(nextprime(base))
            samples[:, i] = van_der_corput(n, base)

        if self.scramble:
            for i in range(self.d):
                np.random.shuffle(samples[:, i])
        return samples


class NumerFunc:
    """Class of callable functions with automatic gradient computation."""
    def __init__(self, f, g = None):
        self.f = f
        self.g = g
    def __call__(self, x):
        return self.f(x)
    @property
    def grad(self):
        return self.g
    def __add__(self, other):
        if isinstance(other, (float, int)):
            return NumerFunc(lambda x: self.f(x) + other, self.g)
        elif isinstance(other, NumerFunc):
            return NumerFunc(lambda x: self.f(x) + other.f(x), lambda x: self.g(x) + other.g(x))
        else:
            raise TypeError(f'Cannot add {type(other)} to NumerFunc.')
    def __radd__(self, other):
        return self.__add__(other)
    def __sub__(self, other):
        if isinstance(other, (float, int)):
            return NumerFunc(lambda x: self.f(x) - other, self.g)
        elif isinstance(other, NumerFunc):
            return NumerFunc(lambda x: self.f(x) - other.f(x), lambda x: self.g(x) - other.g(x))
        else:
            raise TypeError(f'Cannot subtract {type(other)} from NumerFunc.')
    def __rsub__(v, u):
        if isinstance(u, (float, int)):
            return NumerFunc(lambda x: u - v.f(x), lambda x: -v.g(x))
        elif isinstance(u, NumerFunc):
            return NumerFunc(lambda x: u.f(x) - v.f(x), lambda x: u.g(x) - v.g(x))
        else:
            raise TypeError(f'Cannot subtract NumerFunc from {type(u)}.')
    def __neg__(self):
        return NumerFunc(lambda x: -self.f(x), lambda x: -self.g(x))
    def __mul__(u, v):
        if isinstance(v, (float, int)):
            return NumerFunc(lambda x: u.f(x) * v, lambda x: u.g(x) * v)
        elif isinstance(v, NumerFunc):
            return NumerFunc(lambda x: u.f(x) * v.f(x), lambda x: u.g(x) * v.f(x) + u.f(x) * v.g(x))
        else:
            raise TypeError(f'Cannot multiply {type(v)} with NumerFunc.')
    def __rmul__(self, other):
        return self.__mul__(other)
    def __truediv__(u, v):
        if isinstance(v, (float, int)):
            return NumerFunc(lambda x: u.f(x) / v, lambda x: u.g(x) / v)
        elif isinstance(v, NumerFunc):
            def new_g(x):
                vf = v.f(x)
                return (u.g(x) * vf - u.f(x) * v.g(x)) / vf**2
            return NumerFunc(lambda x: u.f(x) / v.f(x), new_g)
        else:
            raise TypeError(f'Cannot divide NumerFunc by {type(v)}.')
    def __rtruediv__(v, u):
        if isinstance(u, (float, int)):
            return NumerFunc(lambda x: u / v.f(x), lambda x: -u / v.f(x)**2 * v.g(x))
        elif isinstance(u, NumerFunc):
            def new_g(x):
                vf = v.f(x)
                return (u.g(x) * vf - u.f(x) * v.g(x)) / vf**2
            return NumerFunc(lambda x: u.f(x) / v.f(x), new_g)
    def __pow__(u, v):
        if isinstance(v, (float, int)):
            return NumerFunc(lambda x: u.f(x) ** v, lambda x: v * u.f(x) ** (v - 1) * u.g(x))
        elif isinstance(v, NumerFunc):
            def new_g(x):
                vf = v.f(x)
                return (vf * u.f(x) ** (vf - 1) * u.g(x) + v.g(x) * u.f(x) ** vf * np.log(u.f(x)))
            return NumerFunc(lambda x: u.f(x) ** v.f(x), new_g)
        else:
            raise TypeError(f'Cannot raise NumerFunc to {type(v)}.')

    @classmethod
    def sum(cls, funcs: List['NumerFunc']) -> 'NumerFunc':
        """Create a new NumerFunc object so that f1(x) = sum(f0(x) for f0 in funcs)."""
        return NumerFunc(lambda x: sum(f.f(x) for f in funcs), lambda x: sum(f.g(x) for f in funcs))

    def log(self):
        return NumerFunc(lambda x: np.log(self.f(x)), lambda x: self.g(x) / self.f(x))

    @classmethod
    def vectorize(cls, funcs: List['NumerFunc']) -> 'NumerFunc':
        """Create a new NumerFunc object so that f1(x) = np.array([func(x) for func in funcs])."""
        return NumerFunc(lambda x: np.array([f(x) for f in funcs]),
                lambda x: np.vstack([f.g(x) for f in funcs]).T)

    def compose(self, funcs: Optional[Union['NumerFunc', List['NumerFunc']]]) -> 'NumerFunc':
        """Create a new NumerFunc object so that f1(x) = f0([func(x) for func in funcs])."""
        if funcs is None:
            return self
        func = NumerFunc.vectorize(funcs) if not isinstance(funcs, NumerFunc) else funcs
        return NumerFunc(lambda x: self.f(func(x)), lambda x: func.g(x) @ self.g(func(x)))

    @classmethod
    def wrap(cls, expr: Union[Expr, List[Expr]], symbols: List[Symbol]) -> 'NumerFunc':
        """
        Convert sympy expressions to numerical functions.

        Parameters
        ----------
        expr : sympy.Expr
            The expression to convert.
        symbols : tuple of sympy.Symbol
            The symbols in the expression.
        """
        def _lambdify(symbols, expr):
            if isinstance(expr, Poly):
                expr = expr.as_expr()
                # expr = sp.horner(expr)
            else:
                expr = expr.doit()
            # TODO: avoid converting to expr for polynomials
            return anonymous_lambdify(symbols, expr, modules='numpy')
        def _wrap_single(expr, symbols):
            f = _lambdify(symbols, expr)
            fdiff = [_lambdify(symbols, expr.diff(_)) for _ in symbols]
            f0 = NumerFunc(lambda x: f(*x), lambda x: np.array([df(*x) for df in fdiff]))
            return f0

        if isinstance(expr, (list, tuple, MatrixBase)):
            fs = [_wrap_single(e, symbols) for e in expr]
            return NumerFunc.vectorize(fs)
        elif isinstance(expr, (Expr, Poly)):
            return _wrap_single(expr, symbols)

        raise TypeError(f"Unsupported type {type(expr)} for wrapping.")

    @classmethod
    def index(cls, i):
        """Create a new NumerFunc object so that f(x) = x[i]."""
        def new_g(x):
            g = np.zeros_like(x)
            g[i] = 1
            return g
        return NumerFunc(lambda x: x[i], new_g)

    @classmethod
    def identity(cls):
        """Create a new NumerFunc object so that f(x) = x."""
        return NumerFunc(lambda x: x, lambda x: np.eye(x.shape[0]))

    @classmethod
    def affine(cls, A, b):
        """Create a new NumerFunc object so that f(x) = A @ x + b."""
        A, b = np.array(A), np.array(b)
        if len(A.shape) == 1:
            return NumerFunc(lambda x: np.dot(A, x) + b, lambda x: A)
        return NumerFunc(lambda x: A @ x + b, lambda x: A.T)

    def compose_affine(self, A, b=None):
        """Create a new NumerFunc object so that f1(x) = f0(A @ x + b)."""
        A = np.array(A)
        b = np.array(b) if b is not None else 0
        return NumerFunc(lambda x: self.f(A @ x + b), lambda x: A.T @ self.g(A @ x + b))

    def permute(self, perm: List[int]) -> 'NumerFunc':
        """Create a new NumerFunc object so that f1(x) = f0(x[perm])."""
        inv_perm = np.argsort(perm)
        return NumerFunc(lambda x: self.f(x[perm]), lambda x: self.g(x[perm])[inv_perm])


def _update_dict(d1: Dict, d2: Dict) -> Dict:
    """Write dict d2 to d1 recursively. The modification is in-place."""
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1:
            d1[k] = _update_dict(d1[k], v)
        else:
            d1[k] = v
    return d1


def numeric_optimize_skew_symmetry(
    poly: Union[Expr, Poly],
    symbols: List[Symbol],
    perm_group: Union[List[List[int]], Permutation, PermutationGroup],
    num: int = 5, is_homogeneous: Optional[bool] = None,
    points: Optional[np.ndarray]=None, optimizer: Optional[Callable]=None,
) -> List[np.ndarray]:
    """
    Numerically optimize a polynomial with no constraints by exploiting the skew-symmetry
    of the polynomial.

    Parameters
    ----------
    poly : sympy.Poly or sympy.Expr
        The polynomial to optimize.
    symbols : tuple of sympy.Symbol
        The symbols in the polynomial.
    perm_group : list of permutations or sympy.PermutationGroup
        Explore the space by permuting the symbols.
        The polynomial should NOT be symmetric with the given permutations
        to highlight the biases in the values across permutations.
    num : int, optional
        Number of extrema to find. Default is 5.
    is_homogeneous : bool, optional
        Whether the objective is homogeneous. If None, it is inferred from the polynomial.
        If the `poly` argument is a sympy expression, it will not be inferred.
    points : np.ndarray, optional
        Initial points to start the optimization. If None, points are generated
        using Halton sequence. If given, it should be a matrix of shape (N, len(symbols)).
    optimizer : callable, optional
        The optimizer to use for finding a local minima given a initial point.
        If None, it uses scipy.optimize.minimize with BFGS method.
        The optimizer should have signature `optimizer(f, x0, jac) -> np.ndarray`.

    Returns
    -------
    extrema : list of np.ndarray
        The extrema found. Each element is a tuple of the coordinates of the extrema.

    Examples
    --------
    >>> from sympy.abc import a, b, c
    >>> from sympy.combinatorics import SymmetricGroup
    >>> p = ((a**2+b**2+c**2)**2-3*(a**3*b+b**3*c+c**3*a)).as_poly(a,b,c)
    >>> xs = numeric_optimize_skew_symmetry(p, (a,b,c), SymmetricGroup(3), num=3); xs # doctest: +SKIP
    [(0.543133976714614, 0.3492916988562254, 0.10757432442916048),
     (0.5431339850218285, 0.34929170314855795, 0.10757431182961352),
     (0.33333334444005286, 0.33333333749340305, 0.3333333180665441)]
    >>> p(*xs[0]) # doctest: +SKIP
    1.27502175484295e-16
    """
    if isinstance(perm_group, PermutationGroup):
        perm_group = list(perm_group.elements)
    elif isinstance(perm_group, Permutation):
        perm_group = [perm_group]
    elif not isinstance(perm_group, list):
        raise TypeError("Perm_group must be a list, a Permutation, or a PermutationGroup object.")
    perm_group = [_.array_form if isinstance(_, Permutation) else _ for _ in perm_group]
    if len(perm_group) == 0:
        raise ValueError("Perm_group must not be empty.")

    if is_homogeneous is None:
        is_homogeneous = poly.is_homogeneous if hasattr(poly, 'is_homogeneous') else False # type: ignore

    f = NumerFunc.wrap(poly, symbols)
    if points is None: # generate points using Halton sequence
        if USE_SCIPY_HALTON:
            from scipy.stats.qmc import Halton
        else:
            Halton = HaltonManual
        points = Halton(d=len(symbols), scramble=False).random(64)[1:]
        if is_homogeneous: # normalize the points
            points = points / (np.abs(points.sum(axis=1, keepdims=True)) + 1e-8)

    permed_values = []
    for perm in perm_group:
        permed_values.append(f(points[:,perm].T))
    permed_values = np.array(permed_values)
    permed_values -= np.min(permed_values) - 1e-8 # shift every value to be positive
    values = np.min(permed_values, axis=0)
    ratio = (values / permed_values.mean(axis=0))


    fdenom = NumerFunc.sum([f.permute(perm) for perm in perm_group])
    f2_ = f / fdenom
    if is_homogeneous:
        affine_map = np.eye(len(symbols), len(symbols)-1)
        affine_map[-1,:] = -1
        affine_b = np.zeros(len(symbols))
        affine_b[-1] = 1
        f2 = f2_.compose_affine(affine_map, affine_b)
    else:
        f2 = f2_

    def _solve_local_minima(f: Callable, x0: np.ndarray, jac: Optional[Callable]=None):
        from scipy.optimize import minimize
        with warnings.catch_warnings():
            # disable RuntimeWarning like invalid zero division
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            res = minimize(f, x0, jac=jac, method='BFGS', tol=1e-6)
            if res is not None and hasattr(res, 'x'):
                return res.x
            return x0
    if optimizer is None:
        optimizer = _solve_local_minima

    inds = np.argsort(ratio)
    extrema = []
    for ind in inds[:num]:
        perm_ind = np.argmin(permed_values[:,ind])
        x0 = points[ind][perm_group[perm_ind]]
        # print('Starting from', x0, 'with value =', values[ind], 'ratio =', ratio[ind], 'f2 =', f2_(x0), fdenom(x0), fdenom.g(x0))
        if is_homogeneous:
            x0 = x0[:-1]
        extrema.append(tuple(optimizer(f2, x0, jac=f2.g)))
    if is_homogeneous:
        extrema = [tuple((*_, 1 - sum(_))) for _ in extrema]
    return extrema


def _numeric_optimize_shgo(
    poly: Union[Poly, Expr],
    ineq_constraints: List[Union[Poly, Expr]],
    eq_constraints: List[Union[Poly, Expr]],
    symbols: List[Symbol],
    free_symbols: List[Symbol]=None,
    embedding: Dict[Symbol, Expr]=None,
    restore: bool=True,
    shgo_kwargs: Dict = {}
) -> OptimizeResult:
    """
    Internal function to numerically optimize a polynomial with given inequality and equality constraints
    by calling scipy.optimize.shgo.
    """
    from scipy.optimize import OptimizeResult, shgo
    if isinstance(poly, Poly) and poly.total_degree() <= 0:
        res = OptimizeResult(nit=0, nfev=0, nlfev=0, nljev=0, nlhev=0, success=True,
                status='success', message='Optimization terminated successfully.')
        nvars = len(symbols)
        res.x = np.array([0.] * nvars)
        res.fun = float(poly.coeff_monomial((0,)*nvars))
        res.xl = res.x.reshape((1, nvars))
        res.funl = np.array([res.fun])
        return res

    embedding_f = None
    if free_symbols is not None:
        embedding_list = [embedding.get(s, s) for s in symbols]
        embedding_f = NumerFunc.wrap(embedding_list, free_symbols)

    f = NumerFunc.wrap(poly, symbols).compose(embedding_f)

    def _to_shgo_con(con, type='ineq'):
        con_f = NumerFunc.wrap(con, symbols).compose(embedding_f)
        return {'type': type, 'fun': con_f, 'jac': con_f.g}
    constraints = [_to_shgo_con(_, 'ineq') for _ in ineq_constraints] \
                + [_to_shgo_con(_, 'eq') for _ in eq_constraints]

    shgo_kwargs = _update_dict(
        _update_dict({'options': {'jac': f.g}}, DEFAULT_SHGO_KWARGS), shgo_kwargs
    )

    with warnings.catch_warnings():
        # disable RuntimeWarning like invalid zero division
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r".*_shgo")
        # try:
        dof = len(free_symbols) if free_symbols is not None else len(symbols)
        result = shgo(f, bounds=[(0, 1)] * dof,
                    constraints=constraints, **shgo_kwargs)
        # except Exception as e:
        #     if isinstance(e, (TypeError, ValueError, KeyboardInterrupt)):
        #         raise e
        #     return []

    if restore and (free_symbols is not None):
        if hasattr(result, 'x'):
            result.x = embedding_f(result.x)
        if hasattr(result, 'xl'):
            result.xl = embedding_f(result.xl.T).T

    return result


def _numeric_optimize_poly(
    poly: Poly,
    ineq_constraints: List[Poly],
    eq_constraints: List[Poly],
    symbols: List[Symbol],
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


def numeric_optimize_poly(
    poly: Union[Poly, Expr],
    ineq_constraints: List[Union[Poly, Expr]] = [],
    eq_constraints: List[Union[Poly, Expr]] = [],
    symbols: Optional[List[Symbol]] = None,
    objective: str = 'min',
    return_type: str='tuple',
    shgo_kwargs: Dict = {}
) -> List[Root]:
    """
    Numerically polynomial optimize a polynomial with given inequality and equality constraints
    using heuristic methods. It is HIGHLY EXPERIMENTAL and may be unstable for complicated systems.

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
        points = RootList(symbols, [Root(_) for _ in points])
    elif return_type == 'dict':
        points = [dict(zip(symbols, _)) for _ in points]
    return points
