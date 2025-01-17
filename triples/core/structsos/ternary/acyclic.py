import sympy as sp

from .quadratic import sos_struct_acyclic_quadratic

def sos_struct_acyclic_sparse(coeff, real = True):
    """
    Solve acyclic 3-var polynomial inequalities.
    """
    a, b, c = sp.symbols("a b c")

    degree = coeff.degree()
    if degree == 1:
        c1, c2, c3 = coeff((1,0,0)), coeff((0,1,0)), coeff((0,0,1))
        if c1 >= 0 and c2 >= 0 and c3 >= 0:
            return c1*a + c2*b + c3*c
        return None
    return None

