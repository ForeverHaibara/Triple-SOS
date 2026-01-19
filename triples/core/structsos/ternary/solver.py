from typing import Union, Dict, Optional, Any

from sympy import Poly, Expr, Function
from sympy.core.symbol import uniquely_named_symbol

from .sparse  import sos_struct_sparse, sos_struct_heuristic
from .quadratic import sos_struct_quadratic, sos_struct_acyclic_quadratic
from .cubic   import sos_struct_cubic, sos_struct_acyclic_cubic
from .quartic import sos_struct_quartic, sos_struct_acyclic_quartic
from .quintic import sos_struct_quintic
from .sextic  import sos_struct_sextic
from .septic  import sos_struct_septic
from .octic   import sos_struct_octic
from .nonic   import sos_struct_nonic
from .acyclic import sos_struct_acyclic_sparse

from ..utils import Coeff, PolynomialNonpositiveError, PolynomialUnsolvableError
from ..sparse import sos_struct_common, sos_struct_degree_specified_solver
from ..solution import SolutionStructural

SOLVERS = {
    2: sos_struct_quadratic,
    3: sos_struct_cubic,
    4: sos_struct_quartic,
    5: sos_struct_quintic,
    6: sos_struct_sextic,
    7: sos_struct_septic,
    8: sos_struct_octic,
    9: sos_struct_nonic,
}

SOLVERS_ACYCLIC = {
    2: sos_struct_acyclic_quadratic,
    3: sos_struct_acyclic_cubic,
    4: sos_struct_acyclic_quartic
}


def _structural_sos_3vars_cyclic(
    coeff: Union[Poly, Coeff, Dict],
    real: bool = True
) -> Optional[Expr]:
    """
    Internal function to solve a 3-var homogeneous cyclic polynomial using structural SOS.
    It does not check the homogeneous / cyclic property of the polynomial to save time.
    """
    if not isinstance(coeff, Coeff):
        coeff = Coeff(coeff)

    return sos_struct_common(coeff,
        sos_struct_sparse,
        sos_struct_degree_specified_solver(SOLVERS, homogeneous=True),
        sos_struct_heuristic,
        real=real
    )

def _structural_sos_3vars_acyclic(
    coeff: Union[Poly, Coeff, Dict],
    real: bool = True
) -> Optional[Expr]:
    """
    Internal function to solve a 3-var homogeneous acyclic polynomial using structural SOS.
    It does not check the homogeneous / cyclic property of the polynomial to save time.
    """
    if not isinstance(coeff, Coeff):
        coeff = Coeff(coeff)

    return sos_struct_common(coeff,
        sos_struct_acyclic_sparse,
        sos_struct_degree_specified_solver(SOLVERS_ACYCLIC, homogeneous=True),
        real=real
    )

def structural_sos_3vars(
    poly,
    ineq_constraints: Dict[Poly, Expr] = {},
    eq_constraints: Dict[Poly, Expr] = {}
) -> Optional[Expr]:
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
