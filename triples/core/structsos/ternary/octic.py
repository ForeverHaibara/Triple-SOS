import sympy as sp

from .octic_symmetric import sos_struct_octic_symmetric

from .utils import (
    SS, CyclicSum, CyclicProduct, inverse_substitution
)

def sos_struct_octic(coeff, real = True):
    # first try symmetric case
    solution = sos_struct_octic_symmetric(coeff, real=real)
    if solution is not None:
        return solution

    if not coeff.is_rational:
        return None

    if any(coeff(_) for _ in [(8,0,0), (7,1,0), (7,0,1)]):
        return None

    if not any(coeff(_) for _ in [(6,2,0), (6,1,1), (6,0,2)]):
        a, b, c = sp.symbols('a b c')
        # equivalent to degree-7 hexagon when applying (a,b,c) -> (1/a,1/b,1/c)
        poly2 = coeff((0,3,5))*a**5*b**2 + coeff((1,2,5))*a**4*b**3 + coeff((2,1,5))*a**3*b**4 + coeff((3,0,5))*a**2*b**5\
                + coeff((3,3,2))*a**2*b**2*c**3+coeff((0,4,4))*a**5*b*c+coeff((1,3,4))*a**4*b**2*c+coeff((2,2,4))*a**3*b**3*c+coeff((3,1,4))*a**2*b**4*c
        poly2 = CyclicSum(poly2).doit().as_poly(a,b,c)
        solution = SS.structsos.ternary._structural_sos_3vars_cyclic(poly2)

        if solution is not None:
            # unrobust method handling fraction
            return inverse_substitution(solution, factor_degree = 2)

    return None
