from typing import Tuple, List, Optional

import sympy as sp

from .representation import (
    sym_representation,
    TRANSLATION_POSITIVE, TRANSLATION_REAL,
    prove_numerator
)
from .solution import SolutionSymmetric, SolutionSymmetricSimple
from ..shared import sanitize_input
from ...utils import Coeff


def nonnegative_vars(ineq_constraints: List[sp.Poly], symbols: Tuple[sp.Symbol, ...]) -> List[bool]:
    """
    Infer the nonnegativity of each variable from the inequality constraints.
    """
    nonnegative = set()
    for ineq in ineq_constraints:
        if ineq.is_monomial and ineq.total_degree() == 1 and ineq.LC() >= 0:
            nonnegative.update(ineq.free_symbols)
    return [1 if s in nonnegative else 0 for s in symbols]

@sanitize_input(homogenize=True)
def SymmetricSOS(
        poly: sp.Poly,
        ineq_constraints: List[sp.Poly] = [],
        eq_constraints: List[sp.Poly] = [],
    ) -> Optional[SolutionSymmetricSimple]:
    """
    Represent a 3-var symmetric polynomial in SOS using special
    changes of variables. The algorithm is described in [1].

    Parameters
    ----------
    poly: sp.Poly
        The polynomial to perform SOS on.
    ineq_constraints: List[sp.Poly]
        Inequality constraints to the problem. This assumes g_1(x) >= 0, g_2(x) >= 0, ...
    eq_constraints: List[sp.Poly]
        Equality constraints to the problem. This assumes h_1(x) = 0, h_2(x) = 0, ...

    Returns
    -------
    SolutionSymmetricSimple
        The solution of the problem.
        
    Reference
    -------
    [1] https://zhuanlan.zhihu.com/p/616532245
    """

    # check symmetricity here # and (1,1,1) == 0
    if (len(poly.gens) != 3 or not (poly.domain in (sp.ZZ, sp.QQ))):
        return None
    if not poly.is_homogeneous or not Coeff(poly).is_symmetric():
        return None
    # if poly(1,1,1) != 0:
    #     return None

    positives = [False]
    if all(nonnegative_vars(ineq_constraints, poly.gens)):
        positives.append(True)

    for positive in positives:
        if poly.homogeneous_order() % 2 != 0 and not positive:
            # cannot be positive on the whole plane R^3
            continue

        numerator, denominator = sym_representation(poly, is_pqr = False, positive = positive, return_poly = True)
        numerator = prove_numerator(numerator, positive = positive)
        if numerator is None:
            continue
        expr = numerator / denominator

        expr = expr.subs(
            TRANSLATION_POSITIVE if positive else TRANSLATION_REAL, simultaneous=True
        )
        expr = expr.xreplace(dict(zip(sp.symbols("a b c"), poly.gens)))

        solution = SolutionSymmetric(
            problem = poly,
            solution = expr,
            is_equal = True
        ).as_simple_solution()

        return solution

    return None