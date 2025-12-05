from .octic_symmetric import sos_struct_octic_symmetric

from .utils import (
    Coeff, SS, inverse_substitution
)

def sos_struct_octic(coeff: Coeff, real = True):
    # first try symmetric case
    solution = sos_struct_octic_symmetric(coeff, real=real)
    if solution is not None:
        return solution

    if not coeff.is_rational:
        return None

    if any(coeff(_) for _ in [(8,0,0), (7,1,0), (7,0,1)]):
        return None

    if not any(coeff(_) for _ in [(6,2,0), (6,1,1), (6,0,2)]):
        a, b, c = coeff.gens
        CyclicSum = coeff.cyclic_sum
        # equivalent to degree-7 hexagon when applying (a,b,c) -> (1/a,1/b,1/c)
        poly2 = [((i, j, 7-i-j), coeff((i+j-2,5-i,5-j))) for i in range(5, -1, -1) for j in range(5, -1, -1) if 0 <= 7-i-j <= 5]
        poly2 = coeff.from_dict(dict(poly2))
        solution = SS.structsos.ternary._structural_sos_3vars_cyclic(poly2)

        if solution is not None:
            # unrobust method handling fraction
            return inverse_substitution(coeff, solution, factor_degree = 2)

    return None
