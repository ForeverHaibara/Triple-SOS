from typing import Union, Dict, Optional, TYPE_CHECKING

from sympy import Function
from sympy.core.symbol import uniquely_named_symbol

from .quartic import structsos_nvars_quartic_symmetric
from ..sparse import structsos_common, structsos_degree_specified_solver
from ...solution import extract_undetermined_exprs
from ....utils import Coeff

if TYPE_CHECKING:
    from sympy import Poly, Expr
    from ...problem import InequalityProblem

SOLVERS_SYMMETRIC = {
    4: structsos_nvars_quartic_symmetric,
}

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

    return structsos_common(coeff,
        structsos_degree_specified_solver(SOLVERS_SYMMETRIC, homogeneous=True),
        real=real
    )

def _structural_sos_nvars_general(
    coeff: Union["Poly", Coeff, Dict],
    real: int = 1
) -> Optional["Expr"]:
    if not isinstance(coeff, Coeff):
        coeff = Coeff(coeff)
    return structsos_common(coeff,
        structsos_degree_specified_solver({}, homogeneous=True),
        real=real
    )


def structural_sos_nvars(
    problem: "InequalityProblem"
) -> Optional["Expr"]:
    """
    Main function of structural SOS for n-var homogeneous polynomials.
    """
    poly: "Poly" = problem.expr
    ineq_constraints = problem.ineq_constraints
    # eq_constraints = problem.eq_constraints

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
    func_name = uniquely_named_symbol('G', poly.gens + tuple(ineq_constraints.values()))
    func = Function(func_name)
    solution = extract_undetermined_exprs(solution, func)
    if solution is None:
        return None

    replacement = {func(x): v for x, (sgn, v) in signs.items() if is_pos(sgn)}
    solution = solution.xreplace(replacement)

    if solution.has(func):
        # unhandled nonnegative symbols -> not a valid solution
        return None

    return solution
