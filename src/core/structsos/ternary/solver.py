from typing import Union, Dict, Optional, Any
from functools import partial

import sympy as sp

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
from .nonhomogeneous import sos_struct_nonhomogeneous

from ..utils import Coeff, PolynomialNonpositiveError, PolynomialUnsolvableError
from ..sparse import sos_struct_extract_factors
from ....utils import verify_hom_cyclic

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


def _structural_sos_3vars(
        coeff: Union[sp.Poly, Coeff, Dict],
        real: bool = True,
        is_cyc: bool = True
    ) -> sp.Expr:
    """
    Perform structural sos on a 3-var polynomial and returns an sympy expression.
    This function could be called for recurrsive purpose.
    """

    if not isinstance(coeff, Coeff):
        coeff = Coeff(coeff)

    partial_recur = partial(_structural_sos_3vars, is_cyc = is_cyc)

    if is_cyc:
        prior_solver = sos_struct_sparse
        solvers = SOLVERS
        heuristic_solver = sos_struct_heuristic
    else:
        prior_solver = sos_struct_acyclic_sparse
        solvers = SOLVERS_ACYCLIC
        heuristic_solver = lambda *args, **kwargs: None

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


def structural_sos_3vars(
        poly: sp.Poly,
        real: bool = True,
    ) -> sp.Expr:
    """
    Main function of structural SOS for 3-var homogeneous polynomials. It first assumes the polynomial
    has variables (a,b,c) and latter substitutes the variables with the original ones.
    """
    if len(poly.gens) != 3:
        raise ValueError("structural_sos_3vars only supports 3-var polynomials.")

    is_hom, is_cyc = verify_hom_cyclic(poly)
    if not is_hom:
        raise ValueError("structural_sos_3vars only supports homogeneous polynomials.")

    try:
        solution = _structural_sos_3vars(Coeff(poly), real = real, is_cyc = is_cyc)
    except (PolynomialNonpositiveError, PolynomialUnsolvableError):
        return None

    if solution is None:
        return None

    solution = solution.xreplace(dict(zip(sp.symbols("a b c"), poly.gens)))
    return solution


def structural_sos_3vars_nonhomogeneous(
        poly: sp.Poly,
        real: bool = True,
    ) -> sp.Expr:
    """
    Main function of structural SOS for 3-var nonhomogeneous polynomials. It first assumes the polynomial
    has variables (a,b,c) and latter substitutes the variables with the original ones.

    The function is designed to exploit the symmetry of the polynomial. It will return
    cyclic solutions if possible.
    """
    if len(poly.gens) != 3:
        raise ValueError("structural_sos_3vars only supports 3-var polynomials.")

    try:
        solution = sos_struct_nonhomogeneous(Coeff(poly))
    except (PolynomialNonpositiveError, PolynomialUnsolvableError):
        return None

    if solution is None:
        return None

    solution = solution.xreplace(dict(zip(sp.symbols("a b c"), poly.gens)))
    return solution