from typing import Union, Dict, Optional, Any

from sympy import Poly, Expr, Function
from sympy.core.symbol import uniquely_named_symbol
from sympy.combinatorics import PermutationGroup, Permutation

from .cubic import quaternary_cubic_symmetric, _quaternary_cubic_partial_symmetric
from .quartic import quaternary_quartic
from .quintic import quaternary_quintic_symmetric

from ..utils import Coeff, PolynomialNonpositiveError, PolynomialUnsolvableError
from ..sparse import sos_struct_common, sos_struct_degree_specified_solver
from ..solution import SolutionStructural


SOLVERS_CYCLIC = {
    4: quaternary_quartic,
}

SOLVERS_SYMMETRIC = {
    3: quaternary_cubic_symmetric,
    4: quaternary_quartic,
    5: quaternary_quintic_symmetric,
}

SOLVERS_SYMMETRIC_NONHOM = {
    3: _quaternary_cubic_partial_symmetric,
}

def _structural_sos_4vars_symmetric(
        coeff: Union[Poly, Coeff, Dict],
        real: int = 1
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
        coeff: Union[Poly, Coeff, Dict],
        real: int = 1
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

def _structural_sos_4vars_partial_symmetric(
        coeff: Union[Poly, Coeff, Dict],
        real: int = 1
    ) -> Optional[Expr]:
    """
    Internal function to solve a 4-var homogeneous partial symmetric polynomial using structural SOS.
    The function assumes the polynomial is wrt. (a, b, c, d). The permutation group is
    PermutationGroup(Permutation([1,2,0,3]))
    It is also symmetric with respect to a, b, c if we set d = 1.
    It does not check the homogeneous / cyclic property of the polynomial to save time.
    """
    if not isinstance(coeff, Coeff):
        coeff = Coeff(coeff)
    return sos_struct_common(coeff,
        sos_struct_degree_specified_solver(SOLVERS_SYMMETRIC_NONHOM, homogeneous=True),
        real=real
    )



def structural_sos_4vars(poly: Poly, ineq_constraints: Dict[Poly, Expr] = {}, eq_constraints: Dict[Poly, Expr] = {}) -> Expr:
    """
    Main function of structural SOS for 4-var homogeneous polynomials. It first assumes the polynomial
    has variables (a,b,c) and latter substitutes the variables with the original ones.
    """
    if len(poly.gens) != 4: # should not happen
        raise ValueError("structural_sos_3vars only supports 4-var polynomials.")
    if not poly.is_homogeneous: # should not happen
        raise ValueError("structural_sos_3vars only supports homogeneous polynomials.")

    coeff = Coeff(poly)
    solution = None
    func = None
    if coeff.is_symmetric():
        func = _structural_sos_4vars_symmetric
    elif coeff.is_cyclic():
        func = _structural_sos_4vars_cyclic
    else:
        if coeff.is_cyclic(PermutationGroup(Permutation([1,2,0,3]), Permutation([1,0,2,3]))):
            func = _structural_sos_4vars_partial_symmetric

    try:
        if func is not None:
            solution = func(coeff, real = 1)
    except (PolynomialNonpositiveError, PolynomialUnsolvableError):
        return None

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
