from typing import Dict

import sympy as sp

from ...utils.polytools import deg
from ...utils.basis_generator import generate_expr
from ...utils.expression.cyclic import CyclicSum, CyclicProduct
from ...utils.expression.solution import SolutionSimple, congruence
from ...utils.roots.rationalize import cancel_denominator

class SolutionSDP(SolutionSimple):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def is_equal(self):
        return True

def _matrix_as_expr(
        M: sp.Matrix, 
        multiplier: sp.Expr, 
        cyc: bool = True, 
        cancel: bool = True
    ) -> sp.Expr:
    """
    Helper function to rewrite a single semipositive definite matrix 
    to sum of squares.

    Parameters
    ----------
    M : sp.Matrix
        The matrix to be rewritten.
    multiplier : sp.Expr
        The multiplier of the expression. For example, if `M` represents 
        `(a^2-b^2)^2 + (a^2-2ab+c^2)^2` while `multiplier = ab`, 
        then the result should be `ab(a^2-b^2)^2 + ab(a^2-2ab+c^2)^2`.
    cyc : bool
        Whether add a cyclic sum to the expression. Defaults to True.
    cancel : bool
        Whether cancel the denominator of the expression. Defaults to True.

    Returns
    -------
    sp.Expr
        The expression of matrix as sum of squares.
    """
    degree = round((2*M.shape[0] + .25)**.5 - 1.5)

    a, b, c = sp.symbols('a b c')
    monoms = generate_expr(degree, cyc = 0)[1]

    if not cyc:
        as_cyc = lambda x: multiplier * x
    else:
        if multiplier == CyclicProduct(a):
            as_cyc = lambda x: multiplier * CyclicSum(x)
        else:
            as_cyc = lambda x: CyclicSum(multiplier * x)

    expr = sp.S(0)
    U, S = congruence(M)

    for i, s in enumerate(S):
        if s == 0:
            continue
        val = sp.S(0)
        if cancel:
            r = cancel_denominator(U[i,i:])
        for j in range(i, len(monoms)):
            monom = monoms[j]
            val += U[i,j] / r * a**monom[0] * b**monom[1] * c**monom[2]
        expr += (s * r**2) * as_cyc(val**2)

    return expr
    

def create_solution_from_M(
        poly: sp.Expr,
        Ms: Dict[str, sp.Matrix],
        cancel: bool = True
    ) -> SolutionSDP:
    """
    Create SDP solution from symmetric matrices.

    Parameters
    ----------
    poly : sp.Expr
        The polynomial to be solved.
    Ms : Dict[sp.Matrix]
        The symmetric matrices. `Ms` should have keys 'major', 'minor' and 
        'multiplier'.
    cancel : bool, optional
        Whether to cancel the denominator. The default is True.

    Returns
    -------
    SolutionSDP
        The SDP solution.
    """
    degree = deg(poly)

    a, b, c = sp.symbols('a b c')
    expr = sp.S(0)
    for key, M in Ms.items():
        if M is None:
            continue
        minor = key == 'minor'

        # after it gets cyclic, it will be three times
        # so we require it to be divided by 3 in advance
        if not ((degree % 2) ^ minor):
            M = M / 3

        # e.g. if f(a,b,c) = sum(a*g(a,b,c)^2 + a*h(a,b,c)^2 + ...)
        # then here multiplier = a
        multiplier = {
            (0, 0): 1,
            (0, 1): a * b,
            (1, 0): a,
            (1, 1): CyclicProduct(a)
        }[(degree % 2, minor)]

        expr += _matrix_as_expr(
            M, 
            multiplier, 
            cyc = True, 
            cancel = cancel
        )
    return SolutionSDP(
        problem = poly,
        numerator = expr,
        is_equal = True
    )
