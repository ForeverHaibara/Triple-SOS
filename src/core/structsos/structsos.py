from .solution import SolutionStructural, SolutionStructuralSimple

from .utils import _make_coeffs
from .sparse  import _sos_struct_sparse
from .quartic import _sos_struct_quartic
from .quintic import _sos_struct_quintic
# from .sextic  import _sos_struct_sextic
# from .septic  import _sos_struct_septic
# from .octic   import _sos_struct_octic

from ...utils.polytools import deg



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
    coeff = _make_coeffs(poly)

    # first try sparse cases
    solution = _sos_struct_sparse(poly, coeff, recurrsion = StructuralSOS)

    if solution is None:
        SOLVERS = {
            4: _sos_struct_quartic,
            5: _sos_struct_quintic,
            # 6: _sos_struct_sextic,
            # 7: _sos_struct_septic,
            # 8: _sos_struct_octic,
        }

        degree = deg(poly)
        solver = SOLVERS.get(degree, None)
        if solver is not None:
            solution = SOLVERS[degree](poly, coeff, recurrsion = StructuralSOS)

    if solution is None:
        return None

    solution = SolutionStructural(problem = poly, solution = solution, is_equal = True)
    solution = solution.as_simple_solution()
    return solution