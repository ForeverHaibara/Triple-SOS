from functools import reduce
from typing import Generator, Dict, Tuple, List

import sympy as sp
from sympy.core.singleton import S
from sympy.combinatorics import PermutationGroup, CyclicGroup

from .basis import LinearBasis, quadratic_difference
from ...utils import MonomialReduction, generate_expr


class LinearBasisMultiplier(LinearBasis):
    """
    For example, if we want to find

        CyclicSum(x0 * a**2 + x1 * a*b) * f(a,b,c) = CyclicSum(y1 * g1(a,b,c) + y2 * g2(a,b,c) + ...)

    then it is equivalent to

        RHS + x0 * CyclicSum(-a**2) * f(a,b,c) + x1 * CyclicSum(-a*b) * f(a,b,c) = 0.

    This converts the problem to a usual linear programming by adding the basis
    CyclicSum(-a**2)*f and CyclicSum(-a*b)*f to the linear programming.
    """
    def __init__(self, poly, multiplier):
        self.poly = poly
        self.multiplier = multiplier
    def nvars(self) -> int:
        return len(self.poly.gens)
    def as_poly(self, symbols) -> sp.Poly:
        poly = (self.poly * (-self.multiplier.doit().as_poly(self.poly.gens)))
        return poly.xreplace(dict(zip(self.poly.gens, symbols)))
    def as_expr(self, symbols) -> sp.Expr:
        return self.as_poly(symbols).as_expr()


def lift_degree(
        poly: sp.Poly,
        var_signs: List[int],
        # ineq_constraints: List[sp.Expr],
        symmetry: MonomialReduction,
        degree_limit: int = 12,
        lift_degree_limit: int = 4
    ) -> Generator[Dict, None, None]:
    """
    Hilbert's problem has shown that not every positive polynomial can be written as a sum of squares.
    However, we can write it as sum of rational functions. As a result, we can write
    f(a,b,c) = g(a,b,c) / h(a,b,c) where g and h are both positive. In other words,
    f(a,b,c) * h(a,b,c) = g(a,b,c).

    In practice, we can try out h(a,b,c) = \sum a, h(a,b,c) = \sum (a^2-ab + xab) and so on.
    This `lift_degree` function would generate the h(a,b,c) and associated information.

    Parameters
    ----------
    poly: sp.polys.Poly
        The target polynomial.
    var_signs: List[int]
        Signs of each variable. 1 for nonnegative, -1 for nonpositive, 0 for unrestricted.
    symmetry: MonomialReduction
        The symmetry of the polynomial.
    degree_limit: int
        When the degree of f(a,b,c) * h(a,b,c) is larger than this limit, 
        we stop to save computation resources.
    lift_degree_limit: int
        The degree of h(a,b,c) is at most this limit.

    Yields
    ----------
    Dict containing following items:
        poly: sp.polys.Poly
            The f(a,b,c) * h(a,b,c).
        multiplier: sp.Expr
            The multiplier h(a,b,c).
        basis: List[LinearBasisMultiplier]
            The additional basis to be added to the linear programming. 
            See details in `LinearBasisMultiplier`.
        degree: int
            The degree of f(a,b,c) * h(a,b,c).
        add_degree: int
            The degree of h(a,b,c).
    """
    n = poly.total_degree()
    nvars = len(poly.gens)
    symbols = poly.gens
    n_plus = 0

    while n + n_plus <= degree_limit and n_plus <= lift_degree_limit:
        multipliers = _get_multipliers(symbols, var_signs, n_plus, symmetry=symmetry)
        basis = [LinearBasisMultiplier(poly, multiplier) for multiplier in multipliers]

        if len(basis) > 0:
            yield {
                'basis': basis,
                'degree': n + n_plus,
                'add_degree': n_plus
            }

        n_plus += 1



def _get_multipliers(symbols: Tuple[sp.Symbol, ...], var_signs: List[int], n_plus: int,
                        symmetry: MonomialReduction) -> Tuple[Dict[int, sp.Expr], Dict[int, sp.Expr]]:
    nvars = len(symbols)

    multipliers = None
    if n_plus == 2:
        multipliers = quadratic_difference(symbols)
        psd_vars = [s for s, sign in zip(symbols, var_signs) if sign == 1]
        neg_vars = [s for s, sign in zip(symbols, var_signs) if sign == -1]
        def cross_mul(x):
            return [x[i] * x[j] for i in range(len(x)) for j in range(i+1, len(x))]
        multipliers.extend(cross_mul(psd_vars))
        multipliers.extend(cross_mul(neg_vars))

    if nvars == 3:
        ...
        # if (perm_group.is_alternating or perm_group.is_symmetric):
        #     multipliers = [sp.Mul(*(s**i for s, i in zip(symbols, power))) 
        #                     for power in generate_expr(nvars, n_plus, symmetry=symmetry)[1]]
    if multipliers is None:
        # default case
        def get_sign(power):
            return reduce(lambda x, y: x*y, [v**i for v, i in zip(var_signs, power)])
        multipliers = [sp.Mul(*(s**i for s, i in zip(symbols, power))) 
                            for power in generate_expr(nvars, n_plus, symmetry=symmetry)[1]
                            if get_sign(power) > 0]

    return multipliers