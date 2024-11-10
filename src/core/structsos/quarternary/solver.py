from typing import Union, Dict, Optional, Any

import sympy as sp

from .cubic import quarternary_cubic_symmetric
from .quartic import quarternary_quartic

from ..utils import Coeff, PolynomialNonpositiveError, PolynomialUnsolvableError
from ..sparse import sos_struct_common, sos_struct_degree_specified_solver
from ..solution import SolutionStructural


SOLVERS_CYCLIC = {
    4: quarternary_quartic,
}

SOLVERS_SYMMETRIC = {
    3: quarternary_cubic_symmetric,
    4: quarternary_quartic,
}

def _structural_sos_4vars_symmetric(
        coeff: Union[sp.Poly, Coeff, Dict],
        real: bool = True
    ):
    """
    Internal function to solve a 4-var homogeneous symmetric polynomial using structural SOS.
    The function assumes the polynomial is wrt. (a, b, c, d).
    It does not check the homogeneous / cyclic property of the polynomial to save time.
    """
    if not isinstance(coeff, Coeff):
        coeff = Coeff(coeff)
    
    return sos_struct_common(coeff,
        sos_struct_degree_specified_solver(SOLVERS_SYMMETRIC, homogeneous=True),
        real=real
    )

def _structural_sos_4vars_cyclic(
        coeff: Union[sp.Poly, Coeff, Dict],
        real: bool = True
    ):
    """
    Internal function to solve a 4-var homogeneous symmetric polynomial using structural SOS.
    The function assumes the polynomial is wrt. (a, b, c, d).
    It does not check the homogeneous / cyclic property of the polynomial to save time.
    """
    if not isinstance(coeff, Coeff):
        coeff = Coeff(coeff)
    
    return sos_struct_common(coeff,
        sos_struct_degree_specified_solver(SOLVERS_CYCLIC, homogeneous=True),
        real=real
    )



def structural_sos_4vars(poly: sp.Poly, ineq_constraints: Dict[sp.Poly, sp.Expr] = {}, eq_constraints: Dict[sp.Poly, sp.Expr] = {}) -> sp.Expr:
    """
    Main function of structural SOS for 4-var homogeneous polynomials. It first assumes the polynomial
    has variables (a,b,c) and latter substitutes the variables with the original ones.
    """
    if len(poly.gens) != 4: # should not happen
        raise ValueError("structural_sos_3vars only supports 4-var polynomials.")
    if not poly.is_homogeneous: # should not happen
        raise ValueError("structural_sos_3vars only supports homogeneous polynomials.")

    coeff = Coeff(poly)
    is_cyc = 0
    solution = None
    # is_cyc = is_cyc or (3 if coeff.is_cyclic() else 0)
    if coeff.is_symmetric():
        is_cyc = 4
    elif coeff.is_cyclic():
        is_cyc = 3

    try:
        if is_cyc == 4:
            solution = _structural_sos_4vars_symmetric(coeff, real = 1)
        elif is_cyc == 3:
            solution = _structural_sos_4vars_cyclic(coeff, real = 1)
    except (PolynomialNonpositiveError, PolynomialUnsolvableError):
        return None

    if solution is None:
        return None

    if poly.gens != (sp.symbols("a b c d")):
        solution = solution.xreplace(dict(zip(sp.symbols("a b c d"), poly.gens)))
    solution = SolutionStructural._rewrite_symbols_to_constraints(solution, ineq_constraints)
    return solution