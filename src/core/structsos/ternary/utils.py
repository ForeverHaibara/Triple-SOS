import sympy as sp

from ..utils import (
    Coeff, 
    radsimp, sum_y_exprs, rationalize_func, quadratic_weighting, zip_longest,
    StructuralSOSError, PolynomialNonpositiveError, PolynomialUnsolvableError
)

from ...symsos import prove_univariate
from ...solver import SS
from ....utils.roots.rationalize import (
    nroots, rationalize, rationalize_bound, univariate_intervals,
    square_perturbation, cancel_denominator, common_region_of_conics
)
from ....utils.expression import congruence, CyclicExpr, CyclicSum, CyclicProduct


def reflect_expression(expr: sp.Expr) -> sp.Expr:
    """
    Exchange b and c in a three-var cyclic expression.

    Examples
    ----------
    >>> CyclicExpr.PRINT_FULL = True
    >>> reflect_expression(CyclicSum(a**2*b**3*c**4))
    CyclicSum(a**2*b**4*c**3, (a, b, c), PermutationGroup([
        (0 1 2)]))
    """
    if isinstance(expr, CyclicExpr):
        return expr.func(reflect_expression(expr.args[0]), *expr.args[1:])
    if not expr.has(CyclicExpr):
        b, c = sp.symbols('b c')
        return expr.xreplace({b:c, c:b})
    return expr.func(*[reflect_expression(a) for a in expr.args])


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


def try_perturbations(
        poly, p, q, perturbation, times = 4, **kwargs
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
        solution = SS.structsos.ternary._structural_sos_3vars_cyclic(Coeff(poly2))
        if solution is not None:
            return solution + t * perturbation
    return None