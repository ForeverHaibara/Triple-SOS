import sympy as sp

from .solution import SolutionStructural, SolutionStructuralSimple

from .utils import Coeff
from .sparse  import sos_struct_sparse
from .cubic   import sos_struct_cubic
from .quartic import sos_struct_quartic
from .quintic import sos_struct_quintic
from .sextic  import sos_struct_sextic
from .septic  import sos_struct_septic
from .octic   import sos_struct_octic
from .nonic   import sos_struct_nonic

from ...utils.polytools import deg



def _structural_sos_handler(
        poly,
    ) -> sp.Expr:
    """
    Perform structural sos and returns an sympy expression. This function could be called 
    for recurrsive purpose. The outer function `StructuralSOS` will wrap the expression 
    to a solution object.
    """

    coeff = Coeff(poly)

    # first try sparse cases
    solution = sos_struct_sparse(poly, coeff, recurrsion = _structural_sos_handler)

    if solution is None:
        SOLVERS = {
            3: sos_struct_cubic,
            4: sos_struct_quartic,
            5: sos_struct_quintic,
            6: sos_struct_sextic,
            7: sos_struct_septic,
            8: sos_struct_octic,
            9: sos_struct_nonic,
        }

        degree = deg(poly)
        solver = SOLVERS.get(degree, None)
        if solver is not None:
            solution = SOLVERS[degree](poly, coeff, recurrsion = _structural_sos_handler)

    if solution is None:
        return None

    return solution



def StructuralSOS(
        poly,
        rootsinfo = None,
    ):
    """
    Main function of structural SOS. For a given polynomial, if it is in a well-known form,
    then we can solve it directly. For example, quartic 3-var cyclic polynomials have a complete
    algorithm.

    Params
    -------
    poly: sp.polys.polytools.Poly
        The target polynomial.

    Returns
    -------
    solution: SolutionStructuralSimple

    """
    solution = _structural_sos_handler(poly)
    if solution is None:
        return None

    solution = SolutionStructural(problem = poly, solution = solution, is_equal = True)
    solution = solution.as_simple_solution()
    return solution