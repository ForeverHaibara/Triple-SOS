from typing import Tuple, List

import sympy as sp
import numpy as np

from ...utils import arraylize, arraylize_sp

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
    def __init__(self, expr: sp.Expr, symbols: Tuple[int, ...]):
        self._expr = expr.as_expr()
        self._symbols = symbols
    def nvars(self) -> int:
        return len(self._symbols)
    def as_expr(self, symbols) -> sp.Expr:
        return self._expr.xreplace(dict(zip(self._symbols, symbols)))

class LinearBasisTangent(LinearBasis):
    def __init__(self, powers: Tuple[int, ...], tangent: sp.Expr, symbols: Tuple[int, ...]):
        self._powers = powers
        self._tangent = tangent
        self._symbols = symbols
    @property
    def powers(self) -> Tuple[int, ...]:
        return self._powers
    @property
    def tangent(self) -> sp.Expr:
        return self._tangent
    def nvars(self) -> int:
        return len(self._powers)
    def as_expr(self, symbols) -> sp.Expr:
        return sp.Mul(*(x**i for x, i in zip(symbols, self._powers))) *\
                       self._tangent.xreplace(dict(zip(self._symbols, symbols)))
    @classmethod
    def generate(self, tangent: sp.Expr, symbols: Tuple[int, ...], degree: int, require_equal: bool = True) -> List['LinearBasisTangent']:
        """
        Generate all possible linear bases of the form x1^a1 * x2^a2 * ... * xn^an * tangent
        with total degree == degree or total degree <= degree.        
        """
        degree = degree - tangent.as_poly(symbols).total_degree()
        if degree < 0:
            return []
        return [LinearBasisTangent(p, tangent, symbols) for p in 
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

def diff_tangents(symbols: Tuple[sp.Symbol, ...]) -> List[sp.Expr]:
    """
    Generate all possible tangents of the form (ai - aj)^2
    """
    exprs = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            exprs.append((symbols[i] - symbols[j])**2)
    return exprs
