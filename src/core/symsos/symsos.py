from typing import Tuple, Union, Optional

import sympy as sp

from .representation import (
    sym_representation,
    TRANSLATION_POSITIVE, TRANSLATION_REAL,
    prove_numerator
)
from .solution import SolutionSymmetric, SolutionSymmetricSimple
from ...utils import deg, verify_hom_cyclic, verify_is_symmetric

def SymmetricSOS(
        poly: sp.Poly,
        positive: Optional[bool] = None,
        **kwargs
    ) -> Optional[SolutionSymmetricSimple]:
    """
    Represent a symmetric polynomial in SOS using special
    changes of variables. The algorithm is described in [1].

    Parameters
    ----------
    poly : sympy.Poly
        The polynomial to be represented.
    positive : optional[bool]
        Perform SOS on positive real numbers or real numbers.
        If it is None, it will automatically try both.

    Returns
    -------
    SolutionSymmetricSimple
        The solution of the problem.
        
    Reference
    -------
    [1] https://zhuanlan.zhihu.com/p/616532245
    """

    # check symmetricity here # and (1,1,1) == 0
    if (not (poly.domain in (sp.polys.ZZ, sp.polys.QQ))):
        return None
    if (not all(verify_hom_cyclic(poly))) or (not verify_is_symmetric(poly)):
        return None
    # if poly(1,1,1) != 0:
    #     return None

    if positive is None:
        positives = [False, True] if deg(poly) % 2 == 0 else [True]
    else:
        positives = [positive]

    for positive in positives:
        numerator, denominator = sym_representation(poly, positive = positive, return_poly = True)
        numerator = prove_numerator(numerator, positive = positive)
        if numerator is None:
            continue
        expr = numerator / denominator

        expr = expr.subs(
            TRANSLATION_POSITIVE if positive else TRANSLATION_REAL
        )

        solution = SolutionSymmetric(
            problem = poly,
            solution = expr,
            is_equal = True
        ).as_simple_solution()

        return solution

    return None