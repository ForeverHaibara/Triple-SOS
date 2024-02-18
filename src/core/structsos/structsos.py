from typing import Union, Dict, Optional, Any
from functools import partial

import sympy as sp

from .solution import SolutionStructural, SolutionStructuralSimple

from .utils import Coeff, PolynomialNonpositiveError, PolynomialUnsolvableError
from .sparse  import sos_struct_sparse, sos_struct_heuristic, sos_struct_extract_factors
from .quadratic import sos_struct_quadratic, sos_struct_acyclic_quadratic
from .cubic   import sos_struct_cubic
from .quartic import sos_struct_quartic
from .quintic import sos_struct_quintic
from .sextic  import sos_struct_sextic
from .septic  import sos_struct_septic
from .octic   import sos_struct_octic
from .nonic   import sos_struct_nonic
from .acyclic import sos_struct_acyclic_sparse

from ...utils import verify_hom_cyclic


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
}

def _structural_sos_handler(
        coeff: Union[sp.polys.Poly, Coeff, Dict],
        real: bool = True,
        is_cyc: bool = True
    ) -> sp.Expr:
    """
    Perform structural sos and returns an sympy expression. This function could be called 
    for recurrsive purpose. The outer function `StructuralSOS` will wrap the expression 
    to a solution object.
    """

    if not isinstance(coeff, Coeff):
        coeff = Coeff(coeff)


    partial_recur = partial(_structural_sos_handler, is_cyc = is_cyc)

    if is_cyc:
        prior_solver = sos_struct_sparse
        solvers = SOLVERS
        heuristic_solver = sos_struct_heuristic
    else:
        prior_solver = sos_struct_acyclic_sparse
        solvers = SOLVERS_ACYCLIC
        heuristic_solver = lambda *args, **kwargs: None

    try:
        solution = sos_struct_extract_factors(coeff, recurrsion = partial_recur, real = real)

        degree = coeff.degree()

        # first try sparse cases
        if solution is None:
            solution = prior_solver(coeff, recurrsion = partial_recur, real = real)

        if solution is None:
            solver = solvers.get(degree, None)
            if solver is not None:
                solution = solver(coeff, recurrsion = partial_recur, real = real)

        if solution is None and degree > 6:
            solution = heuristic_solver(coeff, recurrsion = partial_recur)

    except PolynomialUnsolvableError:
        return None

    return solution



def StructuralSOS(
        poly: sp.Poly,
        rootsinfo: Optional[Any] = None,
        real: bool = True,
    ) -> SolutionStructuralSimple:
    """
    Main function of structural SOS. For a given polynomial, if it is in a well-known form,
    then we can solve it directly. For example, quartic 3-var cyclic polynomials have a complete
    algorithm.

    Params
    -------
    poly: sp.polys.polytools.Poly
        The target polynomial.
    real: bool
        Whether solve inequality on real numbers rather on nonnegative reals. This may
        requires lifting the degree.

    Returns
    -------
    solution: SolutionStructuralSimple

    """
    is_hom, is_cyc = verify_hom_cyclic(poly)
    if not is_hom:
        return None

    try:
        solution = _structural_sos_handler(Coeff(poly), real = real, is_cyc = is_cyc)
    except (PolynomialNonpositiveError, PolynomialUnsolvableError):
        return None

    if solution is None:
        return None


    solution = SolutionStructural(problem = poly, solution = solution, is_equal = True)
    solution = solution.as_simple_solution()
    return solution