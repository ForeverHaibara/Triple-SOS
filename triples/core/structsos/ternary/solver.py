from typing import Union, Dict, Optional, TYPE_CHECKING

from sympy import Function
from sympy.core.symbol import uniquely_named_symbol

from .sparse  import structsos_sparse, structsos_heuristic
from .dense_symmetric import structsos_ternary_dense_partial_symmetric
from .quadratic import structsos_quadratic, structsos_acyclic_quadratic
from .cubic   import structsos_cubic, structsos_acyclic_cubic
from .quartic import structsos_quartic, structsos_acyclic_quartic
from .quintic import structsos_quintic
from .sextic  import structsos_sextic
from .septic  import structsos_septic
from .octic   import structsos_octic
from .nonic   import structsos_nonic
from .acyclic import structsos_acyclic_sparse

from ..utils import Coeff, PolynomialNonpositiveError, PolynomialUnsolvableError
from ..sparse import structsos_common, structsos_degree_specified_solver
from ...solution import extract_undetermined_exprs
from ....sdp.arithmetic import rep_matrix_from_dict, permute_matrix_rows

if TYPE_CHECKING:
    from sympy import Poly, Expr
    from ...problem import InequalityProblem

SOLVERS = {
    2: structsos_quadratic,
    3: structsos_cubic,
    4: structsos_quartic,
    5: structsos_quintic,
    6: structsos_sextic,
    7: structsos_septic,
    8: structsos_octic,
    9: structsos_nonic,
}

SOLVERS_ACYCLIC = {
    2: structsos_acyclic_quadratic,
    3: structsos_acyclic_cubic,
    4: structsos_acyclic_quartic
}


def _is_cyclic_mat(M):
    if M.shape[0] != M.shape[1]:
        return False
    ddm = M._rep.rep.to_ddm()
    n = M.shape[0]
    for i in range(n):
        for j in range(n):
            if ddm[i][j] != ddm[(i+1)%n][(j+1)%n]:
                return False
    return True


def _structural_sos_3vars_cyclic(
    coeff: Union["Poly", Coeff, Dict],
    real: int = 1
) -> Optional["Expr"]:
    """
    Internal function to solve a 3-var homogeneous cyclic polynomial
    using structural SOS. It does not check the homogeneous / cyclic
    property of the polynomial to save time.

    Parameters
    ----------
    coeff : Union["Poly", Coeff, Dict]
        The polynomial to solve.
    real : int, optional
        If 2, it demands only solutions with variables in R.
        If 1, it demands solutions with variables in R in prior.
        If 0, it demands solutions with variables in R+.
    """
    if not isinstance(coeff, Coeff):
        coeff = Coeff(coeff)

    return structsos_common(coeff,
        structsos_sparse,
        structsos_degree_specified_solver(SOLVERS, homogeneous=True),
        structsos_heuristic,
        real=real
    )

def _structural_sos_3vars_acyclic(
    coeff: Union["Poly", Coeff, Dict],
    real: int = 1
) -> Optional["Expr"]:
    """
    Internal function to solve a 3-var homogeneous acyclic polynomial
    using structural SOS. It does not check the homogeneous / cyclic
    property of the polynomial to save time.

    Parameters
    ----------
    coeff : Union["Poly", Coeff, Dict]
        The polynomial to solve.
    real : int, optional
        If 2, it demands only solutions with variables in R.
        If 1, it demands solutions with variables in R in prior.
        If 0, it demands solutions with variables in R+.
    """
    if not isinstance(coeff, Coeff):
        coeff = Coeff(coeff)

    return structsos_common(coeff,
        structsos_acyclic_sparse,
        structsos_degree_specified_solver(SOLVERS_ACYCLIC, homogeneous=True),
        structsos_ternary_dense_partial_symmetric,
        real=real
    )


def structural_sos_3vars(
    problem: "InequalityProblem"
) -> Optional["Expr"]:
    """
    Main function of structural SOS for 3-var homogeneous polynomials.
    """
    poly: "Poly" = problem.expr
    gens = problem.gens
    ineq_constraints = problem.ineq_constraints
    # eq_constraints = problem.eq_constraints

    if len(gens) != 3: # should not happen
        return None
    if poly.domain.is_EX or poly.domain.is_EXRAW:
        return None

    is_hom = poly.is_homogeneous
    if not is_hom: # should not happen
        return None


    # get linear constraints and stack them as a matrix
    nvars = len(gens) # == 3
    linear_ineqs: Dict["Poly", "Expr"] = {k: v for k, v in ineq_constraints.items()
                    if k.total_degree() == 1 and k.coeff_monomial((0,)*nvars) == 0}
    if linear_ineqs:
        dom = next(iter(linear_ineqs)).domain.get_field()
        for ineq in linear_ineqs:
            dom = dom.unify(ineq.domain)
        linear_ineqs = {ineq.set_domain(dom) for ineq in linear_ineqs}
        dod = {i: {m.index(1): v for m, v in ineq.rep.terms()}
               for i, ineq in enumerate(linear_ineqs)}
        mat = rep_matrix_from_dict(dod, (len(dod), nvars), dom)
        if mat._rep.rank() == nvars:
            if mat.shape[0] == nvars:
                if mat._rep.nnz() == nvars:
                    # permutation matrix
                    # TODO: flip the sign of the entries so
                    # that all entries are nonnegative
                    pass
                else:
                    # change the variables

                    # try to rearrange the matrix so that it is cyclic
                    if not _is_cyclic_mat(mat):
                        mat2 = permute_matrix_rows(mat, [0, 2, 1])
                        if _is_cyclic_mat(mat2):
                            mat = mat2

                    inv_mat = mat._fromrep(mat._rep.inv())
                    # TODO: try not to change the variable first
                    dot = lambda i, j: sum(ik * jk for ik, jk in zip(i, j))

                    new_problem, restore = problem.transform(
                        {g: dot(row, gens) for g, row in zip(gens, inv_mat.tolist())},
                        {g: dot(row, gens) for g, row in zip(gens, mat.tolist())}
                    )
                    new_problem = new_problem.remove_redundancy().polylize()
                    return restore(structural_sos_3vars(new_problem))
            else:
                # TODO: handle the case when mat.shape[0] > 3:
                # e.g., use LP to determine the active linear constraints
                pass


    # check whether the variables are in the nonnegative orthant
    signs = problem.get_symbol_signs()
    is_pos = lambda x: (x is not None) and x >= 0
    r_plus = all(is_pos(signs.get(x, (-1, -1))[0]) for x in poly.gens)

    if (not r_plus) and poly.total_degree() % 2 == 1:
        # TODO: try to disprove the problem
        return None


    coeff_poly = Coeff(poly)
    is_cyc = coeff_poly.is_cyclic()
    func = _structural_sos_3vars_cyclic if is_cyc\
        else _structural_sos_3vars_acyclic

    try:
        param_real = 1 if r_plus else 2
        solution = func(coeff_poly, real = param_real)
    except (PolynomialNonpositiveError, PolynomialUnsolvableError):
        return None

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
