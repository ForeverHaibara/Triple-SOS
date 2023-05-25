import sympy as sp

from ...utils.roots.rationalize import rationalize
from ...utils.expression.cyclic import CyclicSum, CyclicProduct

class Coeff():
    def __init__(self, coeffs):
        self.coeffs = coeffs

    def __call__(self, *x):
        if len(x) == 1:
            # x is ((a,b,c), )
            x = x[0]
        return self.coeffs.get(x, sp.S(0))

    def __len__(self):
        return len(self.coeffs)


def _make_coeffs(poly):
    coeffs = {}

    for monom, coeff in poly.terms():
        if not isinstance(coeff, sp.Rational): #isinstance(coeff, sp.Float): # and degree > 4
            coeff = rationalize(coeff, reliable = True)
            # coeff = coeff.as_numer_denom()
        coeffs[monom] = coeff
    
    return Coeff(coeffs)


def _sum_y_exprs(y, exprs) -> sp.Expr:
    return sum(v * expr for v, expr in zip(y, exprs) if v != 0)