from typing import Union, Dict, Optional

from sympy import Poly, Expr, Function
from sympy.core.symbol import uniquely_named_symbol

from .quartic import sos_struct_nvars_quartic_symmetric
from ..sparse import sos_struct_common, sos_struct_degree_specified_solver
from ..solution import SolutionStructural
from ....utils import Coeff

SOLVERS_SYMMETRIC = {
    4: sos_struct_nvars_quartic_symmetric,
}

def _structural_sos_nvars_symmetric(
    coeff: Union[Poly, Coeff, Dict],
    real: int = 1
):
    """
    Internal function to solve an n-var homogeneous symmetric polynomial using structural SOS.
    It does not check the homogeneous / cyclic property of the polynomial to save time.
    """
    if not isinstance(coeff, Coeff):
        coeff = Coeff(coeff)

    return sos_struct_common(coeff,
        sos_struct_degree_specified_solver(SOLVERS_SYMMETRIC, homogeneous=True),
        real=real
    )

def _structural_sos_nvars_general(
    coeff: Union[Poly, Coeff, Dict],
    real: int = 1
) -> Optional[Expr]:
    if not isinstance(coeff, Coeff):
        coeff = Coeff(coeff)
    return sos_struct_common(coeff,
        sos_struct_degree_specified_solver({}, homogeneous=True),
        real=real
    )


def structural_sos_nvars(
    poly: Poly,
    ineq_constraints: Dict[Poly, Expr] = {},
    eq_constraints: Dict[Poly, Expr] = {}
) -> Optional[Expr]:
    """
    Main function of structural SOS for n-var homogeneous polynomials.
    """
    if not poly.is_homogeneous: # should not happen
        raise ValueError("structural_sos_nvars only supports homogeneous polynomials.")

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
    solution = SolutionStructural._extract_nonnegative_exprs(solution, func_name=func_name)
    if solution is None:
        return None

    replacement = {}
    for k, v in ineq_constraints.items():
        if len(k.free_symbols) == 1 and k.is_monomial and k.LC() >= 0:
            replacement[func(k.free_symbols.pop())] = v/k.LC()
    solution = solution.xreplace(replacement)

    if solution.has(func):
        # unhandled nonnegative symbols -> not a valid solution
        return None

    return solution
