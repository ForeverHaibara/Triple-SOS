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


def inverse_substitution(expr: sp.Expr, factor_degree: int = 0) -> sp.Expr:
    """
    Substitute a <- b * c, b <- c * a, c <- a * b into expr.
    Then the function extract the common factor of the expression, usually (abc)^k.
    Finally the expression is divided by (abc)^(factor_degree).
    """
    a, b, c = sp.symbols('a b c')
    expr = sp.together(expr.xreplace({a:b*c,b:c*a,c:a*b}))

    def _try_factor(expr):
        if isinstance(expr, (sp.Add, sp.Mul, sp.Pow)):
            return expr.func(*[_try_factor(arg) for arg in expr.args])
        elif isinstance(expr, CyclicSum):
            # Sum(a**3*b**2*c**2*(...)**2)
            if isinstance(expr.args[0], sp.Mul):
                args2 = expr.args[0].args
                symbol_degrees = {}
                other_args = []
                for s in args2:
                    if s in (a,b,c):
                        symbol_degrees[s] = 1
                    elif isinstance(s, sp.Pow) and s.base in (a,b,c):
                        symbol_degrees[s.base] = s.exp
                    else:
                        other_args.append(s)
                if len(symbol_degrees) == 3:
                    degree = min(symbol_degrees.values())
                    da, db, dc = symbol_degrees[a], symbol_degrees[b], symbol_degrees[c]
                    da, db, dc = da - degree, db - degree, dc - degree
                    other_args.extend([a**da, b**db, c**dc])
                    return CyclicSum(sp.Mul(*other_args)) * CyclicProduct(a) ** degree
        elif isinstance(expr, CyclicProduct):
            # Product(a**2) = Product(a) ** 2
            if isinstance(expr.args[0], sp.Pow) and expr.args[0].base in (a,b,c):
                return CyclicProduct(expr.args[0].base) ** expr.args[0].exp
        return expr
    
    expr = sp.together(_try_factor(expr))
    if factor_degree != 0:
        expr = expr / CyclicProduct(a) ** factor_degree
    return expr
