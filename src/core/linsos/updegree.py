from typing import Generator, Dict, Tuple

import sympy as sp
from sympy.core.singleton import S
from sympy.combinatorics import PermutationGroup, CyclicGroup

from .basis import LinearBasis
from ...utils import MonomialPerm, MonomialCyclic, MonomialReduction, generate_expr


class LinearBasisMultiplier(LinearBasis):
    r"""
    For example, if we want to find
    \sum (a^2 + x0*ab) * f(a,b,c) = \sum (x1 * g1(a,b,c) + x2 * g2(a,b,c) + ...)
    then it is equivalent to
    \sum (a^2 - ab) * f(a,b,c) = RHS + x0 * \sum -ab * f(a,b,c).

    This converts the problem to a usual linear programming by adding a basis \sum -ab * f(a,b,c).
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
        symmetry: PermutationGroup = PermutationGroup(),
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
    degree_limit: int
        When the degree of f(a,b,c) * h(a,b,c) is larger than this limit, 
        we stop to save computation resources.
    is_cyc: bool
        Whether the polynomial is cyclic or not.

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
        multipliers = [sp.Mul(*(s**i for s, i in zip(symbols, power))) 
                            for power in generate_expr(nvars, n_plus, symmetry=symmetry)[1]]
        basis = [LinearBasisMultiplier(poly, multiplier) for multiplier in multipliers]

        yield {
            'basis': basis,
            'degree': n + n_plus,
            'add_degree': n_plus
        }

        n_plus += 1



# def _get_multipliers(symbols: Tuple[sp.Symbol, ...], symmetry: PermutationGroup) -> Tuple[Dict[int, sp.Expr], Dict[int, sp.Expr]]:
#     nvars = len(symbols)
#     if isinstance(symmetry, MonomialPerm):
#         symmetry = symmetry.perm_group
#     elif isinstance(symmetry, MonomialCyclic):
#         symmetry = CyclicGroup(nvars)
#     elif isinstance(symmetry, MonomialReduction):
#         symmetry = PermutationGroup()

#     multipliers, adjustment_multipliers = {}, {}
#     if nvars == 3:
#         a, b, c = symbols
#         if (symmetry.is_cyclic and symmetry.order() == 3) or symmetry.is_symmetric:
#             multipliers = {
#                 1: a,
#                 2: (a - b)**2,
#                 3: a*(a-b)*(a-c),
#                 4: (b-c)**2*(b+c-a)**2
#             }
#             adjustment_multipliers = {
#                 1: [],
#                 2: [a*b],
#                 3: [a**2*b, a*b**2, a*b*c],
#                 4: [a**3*b, a*b**3, a**2*b*c, b*c*(b-c)**2]
#             }
    

#     if len(multipliers) == 0 and nvars > 1:
#         multipliers = {
#             1: symbols[0],
#             2: (symbols[0] - symbols[1])**2
#         }
#         # adjustment_multipliers = {
#         #     1: symbols[1:]
#         #     2: [(symbols[1] - symbols[2])**2, (symbols[0] - symbols[2])**2, symbols[0]*symbols[1], symbols[0]*symbols[2], symbols[1]*symbols[2]]
#         # }