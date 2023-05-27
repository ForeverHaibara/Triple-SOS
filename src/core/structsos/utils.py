import sympy as sp

from ...utils.roots.rationalize import rationalize, square_perturbation
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
    """
    Construct a Coeff class from the coeffs of a polynomial.
    """
    coeffs = {}

    for monom, coeff in poly.terms():
        if not isinstance(coeff, sp.Rational): #isinstance(coeff, sp.Float): # and degree > 4
            coeff = rationalize(coeff, reliable = True)
            # coeff = coeff.as_numer_denom()
        coeffs[monom] = coeff
    
    return Coeff(coeffs)


def _sum_y_exprs(y, exprs) -> sp.Expr:
    return sum(v * expr for v, expr in zip(y, exprs) if v != 0)


def _try_perturbations(
        poly,
        p,
        q,
        perturbation,
        recurrsion = None,
        times = 4,
        **kwargs
    ):
    """
    Try subtracting t * perturbation from poly and perform recurrsive trials.
    The subtracted t satisfies that (p - t) / (q - t) is a square

    This is possibly helpful for deriving rational sum-of-square solution.
    """
    a, b, c = sp.symbols('a b c')
    perturbation_poly = perturbation.doit().as_poly(a,b,c)
    for t in square_perturbation(p, q, times = times):
        poly2 = poly - t * perturbation_poly
        solution = recurrsion(poly2)
        if solution is not None:
            return solution + t * perturbation
    return None