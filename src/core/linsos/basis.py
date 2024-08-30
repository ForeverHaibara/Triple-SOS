from typing import Union, Tuple, List, Callable

import sympy as sp
import numpy as np
from sympy.combinatorics import PermutationGroup, Permutation
from sympy.core.singleton import S

from ...utils import arraylize, arraylize_sp, MonomialReduction

class CallableExpr():
    """
    Callable expression is a wrapper of sympy expression that can be called with symbols,
    it is more like a function.

    Example
    ========
    >>> CallableExpr.from_expr(a**3*b**2, (a,b))((x,y))
    x**3*y**2
    """
    __slots__ = ['_func']
    def __init__(self, func: Callable[[Tuple[sp.Symbol, ...]], sp.Expr]):
        self._func = func
    def __call__(self, symbols: Tuple[sp.Symbol, ...]) -> sp.Expr:
        return self._func(symbols)

    @classmethod
    def from_expr(cls, expr: sp.Expr, symbols: Tuple[sp.Symbol, ...]) -> 'CallableExpr':
        return cls(lambda s: expr.xreplace(dict(zip(symbols, s))))

    def default(self, nvars: int) -> sp.Expr:
        """
        Get the defaulted value of the expression given nvars.
        """
        symbols = sp.symbols(f'x:{nvars}')
        return self._func(symbols)


class LinearBasis():
    def nvars(self) -> int:
        raise NotImplementedError
    def _get_default_symbols(self) -> Tuple[sp.Symbol, ...]:
        return tuple(sp.symbols(f'x:{self.nvars()}'))
    def as_expr(self, symbols) -> sp.Expr:
        raise NotImplementedError
    def as_poly(self, symbols) -> sp.Poly:
        return self.as_expr(symbols).as_poly(symbols)
    def degree(self) -> int:
        return self.as_poly(self._get_default_symbols()).total_degree()
    def as_array_np(self, **kwargs) -> np.ndarray:
        return arraylize(self.as_poly(self._get_default_symbols()), **kwargs)
    def as_array_sp(self, **kwargs) -> sp.Matrix:
        return arraylize_sp(self.as_poly(self._get_default_symbols()), **kwargs)

class LinearBasisExpr(LinearBasis):
    __slots__ = ['_expr', '_symbols']
    def __init__(self, expr: sp.Expr, symbols: Tuple[int, ...]):
        self._expr = expr.as_expr()
        self._symbols = symbols
    def nvars(self) -> int:
        return len(self._symbols)
    def as_expr(self, symbols) -> sp.Expr:
        return self._expr.xreplace(dict(zip(self._symbols, symbols)))

class LinearBasisTangent(LinearBasis):
    __slots__ = ['_powers', '_tangent']
    def __init__(self, powers: Tuple[int, ...], tangent: sp.Expr, symbols: Tuple[sp.Symbol, ...]):
        self._powers = powers
        self._tangent = CallableExpr.from_expr(tangent, symbols)
    @property
    def powers(self) -> Tuple[int, ...]:
        return self._powers
    @property
    def tangent(self) -> CallableExpr:
        return self._tangent
    def nvars(self) -> int:
        return len(self._powers)
    def as_expr(self, symbols) -> sp.Expr:
        return sp.Mul(*(x**i for x, i in zip(symbols, self._powers))) * self._tangent(symbols)

    @classmethod
    def from_callable_expr(cls, powers: Tuple[int, ...], tangent: CallableExpr) -> 'LinearBasisTangent':
        """
        Create a LinearBasisTangent from powers and a callable expression. This is intended for
        internal use only.
        """
        obj = cls.__new__(cls)
        obj._powers = powers
        obj._tangent = tangent
        return obj

    @classmethod
    def generate(self, tangent: sp.Expr, symbols: Tuple[int, ...], degree: int, require_equal: bool = True) -> List['LinearBasisTangent']:
        """
        Generate all possible linear bases of the form x1^a1 * x2^a2 * ... * xn^an * tangent
        with total degree == degree or total degree <= degree.        
        """
        degree = degree - tangent.as_poly(symbols).total_degree()
        if degree < 0:
            return []
        tangent = CallableExpr.from_expr(tangent, symbols)
        return [LinearBasisTangent.from_callable_expr(p, tangent) for p in \
                _degree_combinations([1 for _ in symbols], degree, require_equal=require_equal)]


def _degree_combinations(d_list: List[int], degree: int, require_equal = False) -> List[Tuple[int, ...]]:
    """
    Find a1, a2, ..., an such that a1*d1 + a2*d2 + ... + an*dn <= degree.
    """
    n = len(d_list)
    if n == 0:
        return []

    powers = []
    i = 0
    current_degree = 0
    current_powers = [0 for _ in range(n)]
    while True:
        if i == n - 1:
            if degree >= current_degree:
                if not require_equal:
                    for j in range(1 + (degree - current_degree)//d_list[i]):
                        current_powers[i] = j
                        powers.append(tuple(current_powers))
                elif (degree - current_degree) % d_list[i] == 0:
                    current_powers[i] = (degree - current_degree) // d_list[i]
                    powers.append(tuple(current_powers))
            i -= 1
            current_powers[i] += 1
            current_degree += d_list[i]
        else:
            if current_degree > degree:
                # reset the current power
                current_degree -= d_list[i] * current_powers[i]
                current_powers[i] = 0
                i -= 1
                if i < 0:
                    break
                current_powers[i] += 1
                current_degree += d_list[i]
            else:
                i += 1
    return powers

def cross_exprs(exprs: List[sp.Expr], symbols: Tuple[sp.Symbol, ...], degree: int) -> List[sp.Expr]:
    """
    Generate cross products of exprs within given degree.
    """
    polys = [_.as_poly(symbols) for _ in exprs]
    poly_degrees = [_.total_degree() for _ in polys]

    # remove zero-degree polynomials
    polys = [p for p, d in zip(polys, poly_degrees) if d > 0]
    poly_degrees = [d for d in poly_degrees if d > 0]
    if len(polys) == 0:
        return []

    # find all a1*d1 + a2*d2 + ... + an*dn <= degree
    powers = _degree_combinations(poly_degrees, degree)
    # map the powers to expressions
    return [sp.Mul(*(x**i for x, i in zip(exprs, p))) for p in powers]

def quadratic_difference(symbols: Tuple[sp.Symbol, ...]) -> List[sp.Expr]:
    """
    Generate all expressions of the form (ai - aj)^2

    Example
    ========
    >>> quadratic_difference((a, b, c))
    [(a - b)**2, (a - c)**2, (b - c)**2]
    """
    exprs = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            exprs.append((symbols[i] - symbols[j])**2)
    return exprs



def _define_common_tangents() -> List[sp.Expr]:
    # define keys of basis that should be cached
    a, b, c, d = sp.symbols('x:4')
    return [
        S.One,
        (a**2 - b*c)**2, (b**2 - a*c)**2, (c**2 - a*b)**2,
        (a**3 - b*c**2)**2, (a**3 - b**2*c)**2, (b**3 - a*c**2)**2,
        (b**3 - a**2*c)**2, (c**3 - a*b**2)**2, (c**3 - a**2*b)**2,
    ]

_CACHED_BASIS_TO_MATRIX = dict((k, {}) for k in _define_common_tangents())

def multiple_basis_to_matrix(
        tangent: sp.Expr, symbols: Tuple[int, ...], degree: int,
        basis: List[LinearBasisTangent], symmetry: Union[MonomialReduction, PermutationGroup]
    ) -> np.ndarray:
    """
    Convert multiple tangents to a matrix. It uses cache to store the intermediate results.
    This is an optimized implementation of the following code:

    ```python
        def func(tangent, symbols, degree, symmetry):
            basis, _diff_tangents = [], quadratic_difference(symbols)
            d = tangent.as_poly(symbols).total_degree()
            cross_tangents = cross_exprs(_diff_tangents, symbols, degree - d)
            for t in cross_tangents:
                basis += LinearBasisTangent.generate(t * tangent, symbols, degree)
            mat = np.array([x.as_array_np(expand_cyc=True, symmetry=symmetry) for x in basis])
            return mat
    ```
    """
    def _get_default():
        return np.array([x.as_array_np(expand_cyc=True, symmetry=symmetry) for x in basis])

    if isinstance(symmetry, MonomialReduction):
        perm_group = symmetry.to_perm_group(len(symbols))
    elif isinstance(symmetry, PermutationGroup):
        perm_group = symmetry
        if symmetry.is_trivial and symmetry.degree != len(symbols):
            perm_group = PermutationGroup(Permutation(list(range(len(symbols)))))

    callable_tangent = CallableExpr.from_expr(tangent, symbols)
    std_tangent = callable_tangent.default(len(symbols))

    # check whether (std_tangent, degree, perm_group) is in the key of cache
    cache = _CACHED_BASIS_TO_MATRIX.get(std_tangent)
    if cache is None:
        # do not cache the result, fallback to the original code
        return _get_default()

    v = cache.get((degree, perm_group))
    if v is not None:
        return v
    else:
        cache[(degree, perm_group)] = _get_default()
        return cache[(degree, perm_group)]

    # fallback to the original code
    return _get_default()