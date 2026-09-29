from typing import TYPE_CHECKING, Dict, Optional, Union

from .linear import structsos_nvars_linear
from .quadratic import structsos_nvars_quadratic
from .quartic import structsos_nvars_quartic_symmetric
from ..utils import structsos_extract_factors
from ...preprocess.signs import sign_sos
from ....utils.expressions import Coeff

if TYPE_CHECKING:
    from sympy import Expr, Poly

    from ...problem import InequalityProblem

SOLVERS = {
    1: structsos_nvars_linear,
    2: structsos_nvars_quadratic,
}

SOLVERS_SYMMETRIC = {
    **SOLVERS,
    4: structsos_nvars_quartic_symmetric,
}

@structsos_extract_factors
def _structural_sos_nvars_symmetric(
    coeff: Union["Poly", Coeff, Dict],
    real: int = 1
):
    """
    Internal function to solve an n-var homogeneous symmetric polynomial using structural SOS.
    It does not check the homogeneous / cyclic property of the polynomial to save time.
    """
    if not isinstance(coeff, Coeff):
        coeff = Coeff(coeff)

    degree = coeff.total_degree()
    if degree % 2 == 1 and real >= 2:
        return None

    solver = SOLVERS_SYMMETRIC.get(degree)
    if solver is not None:
        return solver(coeff, real=real)

@structsos_extract_factors
def _structural_sos_nvars_general(
    coeff: Union["Poly", Coeff, Dict],
    real: int = 1
) -> Optional["Expr"]:
    if not isinstance(coeff, Coeff):
        coeff = Coeff(coeff)
    degree = coeff.total_degree()
    if degree % 2 == 1 and real >= 2:
        return None

    solver = SOLVERS.get(degree)
    if solver is not None:
        return solver(coeff, real=real)


def structural_sos_nvars(
    problem: "InequalityProblem"
) -> Optional["Expr"]:
    """
    Main function of structural SOS for n-var homogeneous polynomials.
    """
    poly: "Poly" = problem.expr

    if not poly.is_homogeneous: # should not happen
        raise ValueError("structural_sos_nvars only supports homogeneous polynomials.")

    signs = problem.get_symbol_signs()
    is_pos = lambda x: (x is not None) and x >= 0
    r_plus = all(is_pos(signs.get(x, (-1, -1))[0]) for x in poly.gens)

    if (not r_plus) and poly.total_degree() % 2 == 1:
        # TODO: try to disprove the problem
        return None

    coeff = Coeff(poly)
    solution = None
    func = None
    if coeff.is_symmetric():
        func = _structural_sos_nvars_symmetric
    else:
        func = _structural_sos_nvars_general

    solution = func(coeff, real = 1)

    if solution is None:
        return None

    ####################################################################
    # replace assumed-nonnegative symbols with inequality constraints
    ####################################################################
    solution = sign_sos(solution, signs)
    return solution
