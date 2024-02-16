import sympy as sp
from sympy.core.singleton import S

from .basis import LinearBasis, LinearBasisCyclic, a, b, c
from ...utils.polytools import deg
from ...utils.expression.cyclic import CyclicSum, CyclicProduct


class LinearBasisMultiplier(LinearBasis):
    r"""
    For example, if we want to find
    \sum (a^2 + x0*ab) * f(a,b,c) = \sum (x1 * g1(a,b,c) + x2 * g2(a,b,c) + ...)
    then it is equivalent to
    \sum (a^2 - ab) * f(a,b,c) = RHS + x0 * \sum -ab * f(a,b,c).

    This converts the problem to an usual linear programming by adding a basis \sum -ab * f(a,b,c).
    """
    is_cyc = False
    def __init__(self, poly, multiplier):
        self.poly = poly
        self.multiplier = multiplier
        self.expr_ = poly * (-multiplier.doit().as_poly(a,b,c))
        self.array_ = None
        self.array_sp_ = None

class LinearBasisMultiplierCyclic(LinearBasisCyclic, LinearBasisMultiplier):
    r"""
    For example, if we want to find
    \sum (a^2 + x0*ab) * f(a,b,c) = \sum (x1 * g1(a,b,c) + x2 * g2(a,b,c) + ...)
    then it is equivalent to
    \sum (a^2 - ab) * f(a,b,c) = RHS + x0 * \sum -ab * f(a,b,c).

    This converts the problem to an usual linear programming by adding a basis \sum -ab * f(a,b,c).
    """
    is_cyc = True
    def __init__(self, *args, **kwargs):
        super(LinearBasisMultiplier).__init__(*args, **kwargs)


def higher_degree(
        poly: sp.polys.Poly,
        degree_limit: int = 12,
        is_cyc: bool = True
    ):
    """
    Hilbert's problem has shown that not every positive polynomial can be written as a sum of squares.
    However, we can write it as sum of rational functions. As a result, we can write
    f(a,b,c) = g(a,b,c) / h(a,b,c) where g and h are both positive. In other words,
    f(a,b,c) * h(a,b,c) = g(a,b,c).

    In practice, we can try out h(a,b,c) = \sum a, h(a,b,c) = \sum (a^2-ab + xab) and so on.
    This `higher_degree` function would generate the h(a,b,c) and associated information.

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
    n = deg(poly)
    n_plus = 0
    
    multipliers = {
        1: CyclicSum(a),
        2: CyclicSum(a**2 - a*b),
        3: CyclicSum(a*(a-b)*(a-c)),
        4: CyclicSum(a**2*(a-b)*(a-c)),
    }

    adjustment_multipliers = {
        1: [],
        2: [CyclicSum(a*b)],
        3: [CyclicSum(a**2*b), CyclicSum(a*b**2), CyclicSum(a*b*c)],
        4: [CyclicSum(a**3*b), CyclicSum(a*b**3), CyclicSum(a**2*b*c), CyclicSum(a*b*(a-b)**2)],
    }

    while n + n_plus <= degree_limit and n_plus <= 4:
        if n_plus == 0:
            yield {
                'poly': poly,
                'multiplier': S.One,
                'basis': [],
                'degree': n,
                'add_degree': 0
            }
            n_plus += 1
            continue

        basis_type = LinearBasisMultiplierCyclic if is_cyc else LinearBasisMultiplier

        new_poly = poly * multipliers[n_plus].doit().as_poly(a,b,c)
        adjustment_basis = [
            basis_type(poly, multiplier) for multiplier in adjustment_multipliers[n_plus]
        ]

        yield {
            'poly': new_poly,
            'multiplier': multipliers[n_plus],
            'basis': adjustment_basis,
            'degree': n + n_plus,
            'add_degree': n_plus
        }

        n_plus += 1