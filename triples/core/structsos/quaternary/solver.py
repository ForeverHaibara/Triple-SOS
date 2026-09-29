from typing import TYPE_CHECKING, Dict, Optional, Union

from sympy import Mul
from sympy.combinatorics import Permutation, PermutationGroup

from .cubic import (_quaternary_cubic_partial_symmetric, quaternary_cubic_symmetric,
                    quaternary_cubic_partial_constrained)
from .dense_symmetric import quaternary_dense_dihedral, quaternary_dense_symmetric
from .quartic import quaternary_quartic
from .quartic_symmetric import quaternary_quartic_symmetric
from .quintic import quaternary_quintic_symmetric
from ..nvars.linear import structsos_nvars_linear
from ..nvars.quadratic import structsos_nvars_quadratic

from ..utils import PolynomialNonpositiveError, PolynomialUnsolvableError, structsos_extract_factors
from ...preprocess.signs import sign_sos
from ....utils.expressions import Coeff

if TYPE_CHECKING:
    from sympy import Expr, Poly

    from ...problem import InequalityProblem


SOLVERS_CYCLIC = {
    1: structsos_nvars_linear,
    2: structsos_nvars_quadratic,
    4: quaternary_quartic,
}

SOLVERS_SYMMETRIC = {
    1: structsos_nvars_linear,
    2: structsos_nvars_quadratic,
    3: quaternary_cubic_symmetric,
    4: quaternary_quartic_symmetric,
    5: quaternary_quintic_symmetric,
}

SOLVERS_SYMMETRIC_NONHOM = {
    1: structsos_nvars_linear,
    2: structsos_nvars_quadratic,
    3: _quaternary_cubic_partial_symmetric,
}

SOLVERS_CONSTRAINED = {
    3: quaternary_cubic_partial_constrained,
}

@structsos_extract_factors
def _structural_sos_4vars_symmetric(
    coeff: Union["Poly", Coeff, Dict],
    real: int = 1
) -> Optional["Expr"]:
    """
    Internal function to solve a 4-var homogeneous symmetric polynomial using structural SOS.
    It does not check the homogeneous / cyclic property of the polynomial to save time.
    """
    if not isinstance(coeff, Coeff):
        coeff = Coeff(coeff)

    degree = coeff.total_degree()
    if degree % 2 == 1 and real >= 2:
        return None

    solvers = [
        SOLVERS_SYMMETRIC.get(degree),
        quaternary_dense_symmetric,
    ]
    for solver in solvers:
        if solver is None:
            continue
        solution = solver(coeff, real=real)
        if solution is not None:
            return solution


@structsos_extract_factors
def _structural_sos_4vars_cyclic(
    coeff: Union["Poly", Coeff, Dict],
    real: int = 1
) -> Optional["Expr"]:
    """
    Internal function to solve a 4-var homogeneous cyclic polynomial using structural SOS.
    It does not check the homogeneous / cyclic property of the polynomial to save time.
    """
    if not isinstance(coeff, Coeff):
        coeff = Coeff(coeff)

    degree = coeff.total_degree()
    if degree % 2 == 1 and real >= 2:
        return None

    solver = SOLVERS_CYCLIC.get(degree)
    if solver is not None:
        return solver(coeff, real=real)

@structsos_extract_factors
def _structural_sos_4vars_partial_symmetric(
    coeff: Union["Poly", Coeff, Dict],
    real: int = 1
) -> Optional["Expr"]:
    """
    Internal function to solve a 4-var homogeneous partial symmetric polynomial using structural SOS.
    The function assumes the polynomial has group symmetry `PermutationGroup(Permutation([1,2,0,3]))`
    It is also symmetric with respect to a, b, c if we set d = 1.
    It does not check the homogeneous / cyclic property of the polynomial to save time.
    """
    if not isinstance(coeff, Coeff):
        coeff = Coeff(coeff)
    degree = coeff.total_degree()
    if degree % 2 == 1 and real >= 2:
        return None

    solver = SOLVERS_SYMMETRIC_NONHOM.get(degree)
    if solver is not None:
        return solver(coeff, real=real)

def _structural_sos_4vars_dihedral(
    coeff: Union["Poly", Coeff, Dict],
    real: int = 1
) -> Optional["Expr"]:
    """
    Internal function to solve a 4-var homogeneous dihedral polynomial using structural SOS.
    It does not check the homogeneous / dihedral property of the polynomial to save time.

    The permutation group is assumed to be [[2,3,0,1], [1,0,2,3]].
    """
    if not isinstance(coeff, Coeff):
        coeff = Coeff(coeff)
    if coeff.total_degree() <= 2:
        sol = structsos_nvars_quadratic(coeff)
        if sol is not None:
            return sol

    poly = coeff.as_poly()
    const, factors = poly.factor_list()

    if const < 0:
        return None

    factor_sols = [const]
    dih = PermutationGroup(Permutation([2,3,0,1]), Permutation([1,0,2,3]))
    for factor, mul in factors:
        if mul % 2 == 1:
            factor_coeff = Coeff(factor)
            if not factor_coeff.is_cyclic(dih):
                # factors of a D4-symmetric polynomial might not
                # still be cyclic under D4
                # e.g. a*b*c*d
                wrap = factor_coeff.wrap
                if all(wrap(z) >= 0 for z in factor_coeff.coeffs()):
                    factor_sols.append(factor.as_expr() ** mul)
                    continue
                return None
            sol = quaternary_dense_dihedral(factor_coeff)
            if sol is None:
                return None
            factor_sols.append(sol ** mul)
        else:
            factor_sols.append(factor.as_expr() ** mul)

    return Mul(*factor_sols)


def structural_sos_4vars(
    problem: "InequalityProblem"
) -> Optional["Expr"]:
    """
    Main function of structural SOS for 4-var homogeneous polynomials.
    """
    poly: "Poly" = problem.expr

    if len(poly.gens) != 4: # should not happen
        raise ValueError("structural_sos_4vars only supports 4-var polynomials.")
    if not poly.is_homogeneous: # should not happen
        raise ValueError("structural_sos_4vars only supports homogeneous polynomials.")

    # check whether the variables are in the nonnegative orthant
    signs = problem.get_symbol_signs()
    is_pos = lambda x: (x is not None) and x >= 0
    r_plus = all(is_pos(signs.get(x, (-1, -1))[0]) for x in poly.gens)

    if r_plus or poly.total_degree() % 2 == 0:
        coeff = Coeff(poly)
        solution = None
        func = None
        if coeff.is_symmetric():
            func = _structural_sos_4vars_symmetric
        elif coeff.is_cyclic():
            func = _structural_sos_4vars_cyclic
        else:
            pg = PermutationGroup(Permutation([1,2,0,3]), Permutation([1,0,2,3]))
            if coeff.is_cyclic(pg):
                func = _structural_sos_4vars_partial_symmetric

            # TODO: dihedral belongs to cyclic
            pg = PermutationGroup(Permutation([2,3,0,1]), Permutation([1,0,2,3]))
            if coeff.is_cyclic(pg):
                func = _structural_sos_4vars_dihedral

        try:
            if func is not None:
                solution = func(coeff, real = 1)
        except (PolynomialNonpositiveError, PolynomialUnsolvableError):
            return None

        if solution is not None:
            ####################################################################
            # replace assumed-nonnegative symbols with inequality constraints
            ####################################################################
            solution = sign_sos(solution, signs)
            return solution

    if len(problem.ineq_constraints) or len(problem.eq_constraints):
        func = SOLVERS_CONSTRAINED.get(poly.total_degree(), None)
        if func is not None:
            solution = func(problem)
            if solution is not None:
                return solution
