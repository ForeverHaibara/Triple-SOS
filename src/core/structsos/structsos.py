from typing import Union, Dict, Optional, Any

import sympy as sp

from .solution import SolutionStructural, SolutionStructuralSimple

from .utils import Coeff
from .sparse  import sos_struct_sparse, sos_struct_heuristic
from .cubic   import sos_struct_cubic
from .quartic import sos_struct_quartic
from .quintic import sos_struct_quintic
from .sextic  import sos_struct_sextic
from .septic  import sos_struct_septic
from .octic   import sos_struct_octic
from .nonic   import sos_struct_nonic
from .acyclic import sos_struct_acyclic

from ...utils import verify_hom_cyclic


SOLVERS = {
    3: sos_struct_cubic,
    4: sos_struct_quartic,
    5: sos_struct_quintic,
    6: sos_struct_sextic,
    7: sos_struct_septic,
    8: sos_struct_octic,
    9: sos_struct_nonic,
}

def _structural_sos_handler(
        coeff: Union[sp.polys.Poly, Coeff, Dict],
        real = True,
    ) -> sp.Expr:
    """
    Perform structural sos and returns an sympy expression. This function could be called 
    for recurrsive purpose. The outer function `StructuralSOS` will wrap the expression 
    to a solution object.
    """

    if not isinstance(coeff, Coeff):
        coeff = Coeff(coeff)

    degree = coeff.degree()

    # first try sparse cases
    solution = sos_struct_sparse(coeff, recurrsion = _structural_sos_handler, real = real)

    if solution is None:
        solver = SOLVERS.get(degree, None)
        if solver is not None:
            solution = SOLVERS[degree](coeff, recurrsion = _structural_sos_handler, real = real)

    if solution is None and degree > 6:
        solution = sos_struct_heuristic(coeff, recurrsion = _structural_sos_handler)

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

    if is_cyc:
        solution = _structural_sos_handler(Coeff(poly), real = real)
    else:
        solution = sos_struct_acyclic(Coeff(poly), real = real)

    if solution is None:
        return None


    solution = SolutionStructural(problem = poly, solution = solution, is_equal = True)
    solution = solution.as_simple_solution()
    return solution