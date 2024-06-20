from typing import Union, Dict, Optional, Any

import sympy as sp
from sympy.core.symbol import uniquely_named_symbol

from .solution import SolutionStructural, SolutionStructuralSimple
from .ternary import structural_sos_3vars
from .quarternary import structural_sos_4vars
from ..symsos import prove_univariate


def StructuralSOS(
        poly: sp.Poly,
        rootsinfo: Optional[Any] = None,
        real: bool = True,
    ) -> SolutionStructuralSimple:
    """
    Main function of structural SOS. For a given polynomial, if it is in a well-known form,
    then we can solve it directly. For example, quartic 3-var cyclic polynomials have a complete
    algorithm.

    Params
    -------
    poly: sp.polys.polytools.Poly
        The target polynomial.
    real: bool
        Whether solve inequality on real numbers rather on nonnegative reals. This may
        requires lifting the degree.

    Returns
    -------
    solution: SolutionStructuralSimple

    """
    is_hom = poly.is_homogeneous
    homogenizer = None
    original_poly = poly
    if not is_hom:
        # create a symbol for homogenization
        homogenizer = uniquely_named_symbol('t', sp.Tuple(*original_poly.free_symbols))
        poly = original_poly.homogenize(homogenizer)
        
    nvars = len(poly.gens)
    solution = None


    ########################################
    #              main solver
    ########################################
    if nvars == 2:
        # two cases: homogeneous bivariate or univariate before homogenization
        a, b = poly.gens
        d = poly.total_degree()
        solution = prove_univariate(original_poly.subs(b, 1)).xreplace({a: a/b}).together() * b**d
    elif nvars == 3:
        solution = structural_sos_3vars(poly, real = real)
    elif nvars == 4:
        solution = structural_sos_4vars(poly, real = real)

    if solution is None:
        return None
    if not is_hom:
        # substitute the original variables
        solution = solution.subs(homogenizer, 1)

    solution = SolutionStructural(problem = original_poly, solution = solution, is_equal = True)
    solution = solution.as_simple_solution()
    return solution