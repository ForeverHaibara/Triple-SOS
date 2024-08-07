from typing import Union, Dict, Optional, Any

import sympy as sp

from .sparse  import sos_struct_sparse, sos_struct_heuristic
from .quadratic import sos_struct_quadratic, sos_struct_acyclic_quadratic
from .cubic   import sos_struct_cubic, sos_struct_acyclic_cubic, _sos_struct_nonhom_cubic_symmetric
from .quartic import sos_struct_quartic, sos_struct_acyclic_quartic
from .quintic import sos_struct_quintic
from .sextic  import sos_struct_sextic
from .septic  import sos_struct_septic
from .octic   import sos_struct_octic
from .nonic   import sos_struct_nonic
from .acyclic import sos_struct_acyclic_sparse

from ..utils import Coeff, PolynomialNonpositiveError, PolynomialUnsolvableError
from ..sparse import sos_struct_common, sos_struct_degree_specified_solver
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

SOLVERS_NONHOM = {
    3: _sos_struct_nonhom_cubic_symmetric
}


def _structural_sos_3vars_cyclic(
        coeff: Union[sp.Poly, Coeff, Dict],
        real: bool = True
    ):
    """
    Internal function to solve a 3-var homogeneous cyclic polynomial using structural SOS.
    The function assumes the polynomial is wrt. (a, b, c).
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
        coeff: Union[sp.Poly, Coeff, Dict],
        real: bool = True
    ):
    """
    Internal function to solve a 3-var homogeneous acyclic polynomial using structural SOS.
    The function assumes the polynomial is wrt. (a, b, c).
    It does not check the homogeneous / cyclic property of the polynomial to save time.
    """
    if not isinstance(coeff, Coeff):
        coeff = Coeff(coeff)

    return sos_struct_common(coeff,
        sos_struct_acyclic_sparse,
        sos_struct_degree_specified_solver(SOLVERS_ACYCLIC, homogeneous=True),
        real=real
    )

def _structural_sos_3vars_nonhom(
        coeff: Union[sp.Poly, Coeff, Dict],
        real: bool = True
    ):
    """
    Internal function to solve a 3-var nonhomogeneous polynomial using structural SOS.
    The function assumes the polynomial is wrt. (a, b, c).
    It does not check the homogeneous / cyclic property of the polynomial to save time.
    """
    if not isinstance(coeff, Coeff):
        coeff = Coeff(coeff)

    return sos_struct_common(coeff,
        sos_struct_degree_specified_solver(SOLVERS_NONHOM, homogeneous=False),
        real=real
    )


def structural_sos_3vars(
        poly: sp.Poly,
        real: bool = True,
    ) -> sp.Expr:
    """
    Main function of structural SOS for 3-var homogeneous polynomials. 
    It first assumes the polynomial has variables (a,b,c) and
    latter substitutes the variables with the original ones.
    """
    if len(poly.gens) != 3:
        raise ValueError("structural_sos_3vars only supports 3-var polynomials.")

    is_hom, is_cyc = verify_hom_cyclic(poly)
    if not is_hom:
        raise ValueError("structural_sos_3vars only supports homogeneous polynomials.")

    if is_cyc:
        func = _structural_sos_3vars_cyclic
    else:
        func = _structural_sos_3vars_acyclic

    try:
        solution = func(Coeff(poly), real = real)
    except (PolynomialNonpositiveError, PolynomialUnsolvableError):
        return None

    if solution is None:
        return None

    if poly.gens != (sp.symbols("a b c")):
        solution = solution.xreplace(dict(zip(sp.symbols("a b c"), poly.gens)))
    return solution


def structural_sos_3vars_nonhom(
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

    coeff = Coeff(poly)
    is_sym = coeff.is_symmetric()
    if not is_sym:
        # give up
        return None

    try:
        solution = _structural_sos_3vars_nonhom(coeff, real = real)
    except (PolynomialNonpositiveError, PolynomialUnsolvableError):
        return None

    if solution is None:
        return None

    solution = solution.xreplace(dict(zip(sp.symbols("a b c"), poly.gens)))
    return solution