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
from ..solution import SolutionStructural

if TYPE_CHECKING:
    from sympy import Poly, Expr

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


def _structural_sos_3vars_cyclic(
    coeff: Union["Poly", Coeff, Dict],
    real: bool = True
) -> Optional["Expr"]:
    """
    Internal function to solve a 3-var homogeneous cyclic polynomial using structural SOS.
    It does not check the homogeneous / cyclic property of the polynomial to save time.
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
    real: bool = True
) -> Optional["Expr"]:
    """
    Internal function to solve a 3-var homogeneous acyclic polynomial using structural SOS.
    It does not check the homogeneous / cyclic property of the polynomial to save time.
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
    poly,
    ineq_constraints: Dict["Poly", "Expr"] = {},
    eq_constraints: Dict["Poly", "Expr"] = {}
) -> Optional["Expr"]:
    """
    Main function of structural SOS for 3-var homogeneous polynomials.
    """
    if len(poly.gens) != 3: # should not happen
        raise ValueError("structural_sos_3vars only supports 3-var polynomials.")

    is_hom = poly.is_homogeneous
    if not is_hom: # should not happen
        raise ValueError("structural_sos_3vars only supports homogeneous polynomials.")

    coeff_poly = Coeff(poly)
    is_cyc = coeff_poly.is_cyclic()
    if len(ineq_constraints) == 0 and len(eq_constraints) == 0 and poly.total_degree() % 2 == 1:
        return

    if is_cyc:
        func = _structural_sos_3vars_cyclic
    else:
        func = _structural_sos_3vars_acyclic

    try:
        solution = func(coeff_poly, real = 1)
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
