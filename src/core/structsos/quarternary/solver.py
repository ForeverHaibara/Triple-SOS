from typing import Union, Dict, Optional, Any
from functools import partial

import sympy as sp

from .cubic import quarternary_cubic_symmetric

from ..utils import Coeff, PolynomialNonpositiveError, PolynomialUnsolvableError
from ..sparse import sos_struct_extract_factors

def _null_solver(*args, **kwargs):
    return None

SOLVERS_SYMMETRIC = {
    3: quarternary_cubic_symmetric
}

def _structural_sos_4vars(
        coeff: Union[sp.Poly, Coeff, Dict],
        real: bool = True,
        is_cyc: int = 0,
    ) -> sp.Expr:
    """
    Perform structural sos on a 3-var polynomial and returns an sympy expression.
    This function could be called for recurrsive purpose.
    """

    if not isinstance(coeff, Coeff):
        coeff = Coeff(coeff)

    partial_recur = partial(_structural_sos_4vars, is_cyc = is_cyc)

    if is_cyc == 4:
        prior_solver = _null_solver
        solvers = SOLVERS_SYMMETRIC
        heuristic_solver = _null_solver
    else:
        prior_solver = _null_solver
        solvers = dict()
        heuristic_solver = _null_solver

    solution = None
    degree = coeff.degree()
    try:
        solution = sos_struct_extract_factors(coeff, recurrsion = partial_recur, real = real)

        # first try sparse cases
        if solution is None:
            solution = prior_solver(coeff, recurrsion = partial_recur, real = real)

        if solution is None:
            solver = solvers.get(degree, None)
            if solver is not None:
                solution = solver(coeff, recurrsion = partial_recur, real = real)
    except PolynomialUnsolvableError as e:
        # When we are sure that the polynomial is nonpositive,
        # we can return None directly.
        if isinstance(e, PolynomialNonpositiveError):
            return None

    # If the polynomial is not solved yet, we can try heuristic solver.
    try:
        if solution is None and degree > 6:
            solution = heuristic_solver(coeff, recurrsion = partial_recur)
    except PolynomialUnsolvableError:
        return None

    return solution


def structural_sos_4vars(
        poly: sp.Poly,
        real: bool = True,
    ) -> sp.Expr:
    """
    Main function of structural SOS for 4-var homogeneous polynomials. It first assumes the polynomial
    has variables (a,b,c) and latter substitutes the variables with the original ones.
    """
    if len(poly.gens) != 4:
        raise ValueError("structural_sos_3vars only supports 4-var polynomials.")
    if not poly.is_homogeneous:
        raise ValueError("structural_sos_3vars only supports homogeneous polynomials.")

    coeff = Coeff(poly)
    is_cyc = 4 if coeff.is_symmetric() else 0
    is_cyc = is_cyc or (3 if coeff.is_cyclic() else 0)

    try:
        solution = _structural_sos_4vars(coeff, real = real, is_cyc = is_cyc)
    except (PolynomialNonpositiveError, PolynomialUnsolvableError):
        return None

    if solution is None:
        return None

    solution = solution.xreplace(dict(zip(sp.symbols("a b c d"), poly.gens)))
    return solution