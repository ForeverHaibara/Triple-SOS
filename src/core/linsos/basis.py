from typing import Any, Union, Tuple, List, Dict, Callable, Optional

import sympy as sp
import numpy as np
from sympy.combinatorics import PermutationGroup, Permutation
from sympy.core.singleton import S

from ...utils import arraylize, arraylize_sp, MonomialReduction, MonomialPerm

class _callable_expr():
    """
    Callable expression is a wrapper of sympy expression that can be called with symbols,
    it is more like a function. It accepts an addition kwarg poly=True/False.

    Example
    ========
    >>> _callable_expr.from_expr(a**3*b**2, (a,b))((x,y))
    x**3*y**2

    >>> e = _callable_expr.from_expr(sp.Function("F")(a,b,c), (a,b,c), (a**3+b**3+c**3).as_poly(a,b,c))
    >>> e((a,b,c))
    F(a,b,c)
    >>> e((a,b,c), poly=True)
    Poly(a**3 + b**3 + c**3, a, b, c, domain='ZZ')
    
    """
    __slots__ = ['_func']
    def __init__(self, func: Callable[[Tuple[sp.Symbol, ...], Any], sp.Expr]):
        self._func = func
    def __call__(self, symbols: Tuple[sp.Symbol, ...], *args, **kwargs) -> sp.Expr:
        return self._func(symbols, *args, **kwargs)

    @classmethod
    def from_expr(cls, expr: sp.Expr, symbols: Tuple[sp.Symbol, ...], p: Optional[sp.Poly] = None) -> '_callable_expr':
        if p is None:
            def func(s, poly=False):
                e = expr.xreplace(dict(zip(symbols, s)))
                if poly: e = e.as_poly(s)
                return e
        else:
            def func(s, poly=False):
                if not poly:
                    return expr.xreplace(dict(zip(symbols, s)))
                return p.as_expr().xreplace(dict(zip(symbols, s))).as_poly(s)
        return cls(func)

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
        return self.as_expr(symbols).doit().as_poly(symbols)
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
    _degree_step = 1
    __slots__ = ['_powers', '_tangent']
    def __init__(self, powers: Tuple[int, ...], tangent: sp.Expr, symbols: Tuple[sp.Symbol, ...]):
        self._powers = powers
        self._tangent = _callable_expr.from_expr(tangent, symbols)
    @property
    def powers(self) -> Tuple[int, ...]:
        return self._powers
    @property
    def tangent(self) -> _callable_expr:
        return self._tangent
    def nvars(self) -> int:
        return len(self._powers)
    def as_expr(self, symbols) -> sp.Expr:
        return sp.Mul(*(x**i for x, i in zip(symbols, self._powers))) * self._tangent(symbols).as_expr()
    def as_poly(self, symbols) -> sp.Poly:
        return sp.Poly.from_dict({self._powers: 1}, symbols) * self._tangent(symbols, poly=True)
    def __neg__(self) -> 'LinearBasisTangent':
        return self.__class__.from_callable_expr(self._powers, lambda *args, **kwargs: -self._tangent(*args, **kwargs).as_expr())

    def to_even(self, symbols: List[sp.Expr]) -> 'LinearBasisTangentEven':
        """
        Convert the linear basis to an even basis.
        """
        rem_powers = tuple(d % 2 for d in self._powers)
        even_powers = tuple(d - r for d, r in zip(self._powers, rem_powers))
        def _new_tangent(s, poly=False):
            if poly: return self._tangent(s, poly=True)
            monom = sp.Mul(*(symbols[i] for i, d in enumerate(rem_powers) if d))
            return self._tangent(s, poly=False).as_expr() * monom
        return LinearBasisTangentEven.from_callable_expr(even_powers, _callable_expr(_new_tangent))

    @classmethod
    def from_callable_expr(cls, powers: Tuple[int, ...], tangent: _callable_expr) -> 'LinearBasisTangent':
        """
        Create a LinearBasisTangent from powers and a callable expression. This is intended for
        internal use only.
        """
        obj = cls.__new__(cls)
        obj._powers = powers
        obj._tangent = tangent
        return obj

    @classmethod
    def generate(cls, tangent: sp.Expr, symbols: Tuple[int, ...], degree: int, tangent_p: Optional[sp.Poly] = None, require_equal: bool = True) -> List['LinearBasisTangent']:
        """
        Generate all possible linear bases of the form x1^a1 * x2^a2 * ... * xn^an * tangent
        with total degree == degree or total degree <= degree.
        """
        if tangent_p is None:
            tangent_degree = tangent.as_poly(symbols).total_degree()
        else:
            tangent_degree = tangent_p.total_degree()
        degree = degree - tangent_degree
        step = cls._degree_step
        if degree < 0 or degree % step != 0:
            return []
        tangent = _callable_expr.from_expr(tangent, symbols, p=tangent_p)
        return [LinearBasisTangent.from_callable_expr(tuple(i*step for i in comb), tangent) for comb in \
                _degree_combinations([cls._degree_step] * len(symbols), degree, require_equal=require_equal)]

    @classmethod
    def generate_quad_diff(cls, 
            tangent: sp.Expr, symbols: Tuple[int, ...], degree: int, symmetry: PermutationGroup,
            tangent_p: Optional[sp.Poly] = None, quad_diff: bool = True
        ) -> Tuple[List['LinearBasisTangent'], np.ndarray]:
        """
        Generate all possible linear bases of the form x1^a1 * x2^a2 * ... * xn^an * (x1-x2)^(2b_12) * ... * (xi-xj)^(2b_ij) * tangent
        with total degree == degree.
        Also, return the matrix representation of the bases.
        """
        basis, mat, perm_group = None, None, None
        cache = _get_tangent_cache_key(cls, tangent, symbols) if quad_diff else None
        if cache is not None:
            perm_group = symmetry.to_perm_group(len(symbols)) if isinstance(symmetry, MonomialReduction) else symmetry
            basis = cache.get((degree, len(symbols)))
            if basis is not None:
                mat = cache.get((degree, perm_group))
                if mat is not None:
                    return basis, mat

        if not isinstance(tangent_p, sp.Poly):
            p = tangent.as_poly(symbols)
        else:
            p = tangent_p
        d = p.total_degree()
        if p.is_zero or len(p.free_symbols_in_domain) or d > degree:
            return [], np.array([], dtype='float')

        if quad_diff:
            quad_diff = quadratic_difference(symbols)
            cross_tangents = cross_exprs(quad_diff, symbols, degree - d)
        else:
            cross_tangents = [S.One]

        if basis is None:
            # no cache, generate the bases first
            basis = []
            for t in cross_tangents:
                p2 = t.as_poly(symbols) * p
                basis += cls.generate(t * tangent, symbols, degree, tangent_p=p2, require_equal=True)

        if mat is None:
            # convert the bases to matrix
            # mat = np.array([x.as_array_np(expand_cyc=True, symmetry=symmetry) for x in basis]) # too slow
            mat = [0] * len(basis)
            step = cls._degree_step
            def tuple_sum(t1: Tuple[int, ...], t2: Tuple[int, ...]) -> Tuple[int, ...]:
                return tuple(x + y for x, y in zip(t1, t2))

            mat_ind = 0
            if not isinstance(symmetry, MonomialReduction):
                symmetry = MonomialPerm(symmetry)
            poly_from_dict = sp.Poly.from_dict
            for t in cross_tangents:
                p2 = t.doit().as_poly(symbols) * p
                p2dict = p2.as_dict()
                for power in  _degree_combinations([cls._degree_step] * len(symbols), degree - p2.homogeneous_order(), require_equal=True):
                    power = tuple(i*step for i in power)
                    new_p_dict = dict((tuple_sum(power, k), v) for k, v in p2dict.items())
                    new_p = poly_from_dict(new_p_dict, symbols)
                    mat[mat_ind] = symmetry.arraylize(new_p, expand_cyc=True)
                    mat_ind += 1

            mat = np.vstack(mat) if len(mat) > 0 else np.array([], dtype='float')

        if cache is not None:
            # cache the result
            cache[(degree, len(symbols))] = basis
            cache[(degree, perm_group)] = mat

        return basis, mat


class LinearBasisTangentEven(LinearBasisTangent):
    """
    Ensure the degree of each monomial is even.
    """
    _degree_step = 2


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
    symbols = sorted(list(symbols), key=lambda x: x.name)
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            exprs.append((symbols[i] - symbols[j])**2)
    return exprs



def _define_common_tangents() -> List[sp.Expr]:
    # define keys of quad_diff bases that should be cached
    a, b, c, d = sp.symbols('x:4')
    return [
        S.One,
        (a**2 - b*c)**2, (b**2 - a*c)**2, (c**2 - a*b)**2,
        (a**3 - b*c**2)**2, (a**3 - b**2*c)**2, (b**3 - a*c**2)**2,
        (b**3 - a**2*c)**2, (c**3 - a*b**2)**2, (c**3 - a**2*b)**2,
    ]

_CACHED_TANGENT_BASIS = dict((k, {}) for k in _define_common_tangents())
_CACHED_TANGENT_BASIS_EVEN = dict((k, {}) for k in _define_common_tangents())

def _get_tangent_cache_key(cls, tangent: sp.Expr, symbols: Tuple[int, ...]) -> Optional[Dict]:
    """
    Given a tangent and symbols, return the cache key if it is in the cache.
    """
    callable_tangent = _callable_expr.from_expr(tangent, symbols)
    std_tangent = callable_tangent.default(len(symbols))
    cache = _CACHED_TANGENT_BASIS if cls is LinearBasisTangent else _CACHED_TANGENT_BASIS_EVEN
    return cache.get(std_tangent)
