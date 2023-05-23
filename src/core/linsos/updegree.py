# problem: what is input and what is output

# IDEA: Construct special basis:
# 

import sympy as sp
from sympy.core.singleton import S

from .basis import LinearBasisCyclic, a, b, c
from ...utils.polytools import deg
from ...utils.expression.cyclic import CyclicSum, CyclicProduct

class LinearBasisMultiplier(LinearBasisCyclic):
    r"""
    For example, if we want to find
    $\sum (a^2 + x0*ab) * f(a,b,c) = \sum (x1 * g1(a,b,c) + x2 * g2(a,b,c) + ...)$
    then it is equivalent to
    $\sum (a^2 - ab) * f(a,b,c) = RHS + x0 * \sum -ab * f(a,b,c)$.

    This converts the problem to a usual linear programming by adding a basis $\sum -ab * f(a,b,c)$.
    """
    def __init__(self, poly, multiplier):
        self.poly = poly
        self.multiplier = multiplier
        self.expr_ = poly * (-multiplier).as_poly(a,b,c)


def higher_degree(poly, degree_limit = 12):
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
        4: [CyclicSum(a**2*b*c), CyclicSum(a*b*(a-b)**2)],
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

        new_poly = poly * multipliers[n_plus].doit().as_poly(a,b,c)
        adjustment_basis = [
            LinearBasisMultiplier(poly, multiplier) for multiplier in adjustment_multipliers[n_plus]
        ]

        yield {
            'poly': new_poly,
            'multiplier': multipliers[n_plus],
            'basis': adjustment_basis,
            'degree': n + n_plus,
            'add_degree': n_plus
        }

        n_plus += 1