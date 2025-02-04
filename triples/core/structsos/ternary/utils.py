from typing import Callable, Union
from functools import wraps

import sympy as sp

from ..utils import (
    Coeff, 
    radsimp, sum_y_exprs, rationalize_func, quadratic_weighting, zip_longest, congruence,
    StructuralSOSError, PolynomialNonpositiveError, PolynomialUnsolvableError
)

from ..pivoting import prove_univariate, prove_univariate_interval
from ...shared import SS
from ....utils.roots.rationalize import (
    nroots, rationalize, rationalize_bound, univariate_intervals,
    square_perturbation, cancel_denominator, common_region_of_conics
)
from ....utils import CyclicExpr, CyclicSum, CyclicProduct


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


def sos_struct_handle_uncentered(solver: Callable) -> Callable:
    """
    A decorator for structural SOS with uncentered polynomial handling.
    It only supports cyclic polynomials.
    """
    @wraps(solver)
    def _wrapped_solver(poly: Union[sp.Poly, Coeff], *args, **kwargs):
        bias = 0
        if isinstance(poly, Coeff):
            bias = poly.poly111()
        elif isinstance(poly, sp.Poly):
            bias = poly(1,1,1)
            poly = Coeff(poly)
        else:
            raise ValueError("Unsupported polynomial type. Expected Coeff or Poly, but received %s." % type(poly))
        if bias < 0:
            # raise PolynomialNonpositiveError#("The polynomial is nonpositive.")
            return None

        d = poly.total_degree()
        dd3, dm3 = divmod(d,3)
        i, j, k = dd3, dd3+(1 if dm3>=2 else 0), dd3+(1 if dm3 else 0)
        coeff = poly.coeffs.copy()
        if not ((i,j,k) in coeff):
            coeff[(i,j,k)] = 0
            coeff[(j,k,i)] = 0
            coeff[(k,i,j)] = 0
        coeff[(i,j,k)] -= bias/3
        coeff[(j,k,i)] -= bias/3
        coeff[(k,i,j)] -= bias/3

        new_poly = Coeff(coeff)
        solution = solver(new_poly, *args, **kwargs)
        if solution is not None:
            a, b, c = sp.symbols('a b c')
            if dm3 == 0:
                solution += bias * CyclicProduct(a**i)
            else:
                solution += bias/3 * CyclicSum(a**i*b**j*c**k)
        return solution
    return _wrapped_solver


class CommonExpr:
    """
    Store commonly used expressions for structural SOS.    
    """
    abc = sp.symbols("a b c")
    a, b, c = abc
    @classmethod
    def schur(cls, n):
        """
        Solve s(a^(n-2)*(a-b)*(a-c)) when n > 0
        """
        if n < 2:
            return
        a, b, c = cls.abc
        if n == 2:
            return CyclicSum((a - b)**2)/2
        elif n == 3:
            return (CyclicSum((a-b)**2*(a+b-c)**2) + 2*CyclicSum(a*b*(a-b)**2))/(2*CyclicSum(a))
        elif n == 5:
            return 2*(CyclicSum(a**3*(a-b)**2*(a-c)**2)+CyclicSum(a)*CyclicProduct((a-b)**2))/CyclicSum((a-b)**2)

    @classmethod
    def schurinv(cls, n):
        """
        Solve s(b^((n-2)/2)*c^((n-2)/2)*(a-b)*(a-c)) when n > 0
        """
        a, b, c = cls.abc
        if n == 2:
            return CyclicSum((a-b)**2)
        elif n == 4:
            return CyclicSum(a**2*(b-c)**2/2)
        elif n == 6:
            return (CyclicSum(c*(a-b)**2*(a*c+b*c-a*b)**2)*2 + CyclicProduct(a)*CyclicSum(a**2*(b-c)**2))/\
                (2*CyclicSum(a))

    @classmethod
    def quadratic(cls, x, y):
        """
        Solve s(x*a**2 + y*a*b)
        """
        if x < 0 or x + y < 0:
            return
        x, y = sp.S(x), sp.S(y)
        a, b, c = cls.abc
        if x == 0:
            return y * CyclicSum(a*b)
        if y > 2 * x:
            return CyclicSum(x * a**2 + y * b*c)
        w1 = (2*x - y) / 3
        w2 = x - w1
        return w1 / 2 * CyclicSum((a-b)**2) + w2 * CyclicSum(a)**2


    
    _SPECIAL_AMGMS = {
        ((2,0,1),(1,1,1)): CyclicSum(c*(2*a+c)*(a-b)**2)/(2*CyclicSum(a)),
        ((2,1,0),(1,1,1)): CyclicSum(b*(2*a+b)*(a-c)**2)/(2*CyclicSum(a)),
        ((6,0,0),(4,1,1)): CyclicSum(a**2)*CyclicSum((a**2-b**2)**2)/4 + CyclicSum(a**2*(a**2-b*c)**2)/2,
        ((6,0,0),(4,2,0)): CyclicSum((a**2-b**2)**2*(2*a**2+b**2))/3,
        ((6,0,0),(4,0,2)): CyclicSum((a**2-c**2)**2*(2*a**2+c**2))/3,
        ((6,0,0),(5,1,0)): CyclicSum((a**2-b**2)**2*(2*a**2+b**2))/6 + CyclicSum(a**4*(a-b)**2)/2,
        ((6,0,0),(5,0,1)): CyclicSum((a**2-c**2)**2*(2*a**2+c**2))/6 + CyclicSum(a**4*(a-c)**2)/2,
        ((6,0,0),(3,2,1)): CyclicSum((a**2-b**2)**2*(2*a**2+b**2))/6 + CyclicSum((a**3-b**2*c)**2)/2,
        ((6,0,0),(3,1,2)): CyclicSum((a**2-c**2)**2*(2*a**2+c**2))/6 + CyclicSum((a**3-b*c**2)**2)/2,
        ((8,0,0),(7,1,0)): CyclicSum((a**4-b**4)**2)/8 + CyclicSum(a**4*(a**2-b**2)**2)/4 + CyclicSum(a**6*(a-b)**2)/2,
        ((8,0,0),(7,0,1)): CyclicSum((a**4-b**4)**2)/8 + CyclicSum(a**4*(a**2-c**2)**2)/4 + CyclicSum(a**6*(a-c)**2)/2,
        ((8,0,0),(5,0,3)): CyclicSum((a**4-b**4)**2)/8 + CyclicSum(a**4*(a**2-b**2)**2)/4 + CyclicSum(b**2*(a**3-b**3)**2)/2,
        ((8,0,0),(5,3,0)): CyclicSum((a**4-b**4)**2)/8 + CyclicSum(a**4*(a**2-c**2)**2)/4 + CyclicSum(c**2*(a**3-c**3)**2)/2,
        ((8,0,0),(6,1,1)): CyclicSum((a**4-a**2*b*c)**2)/2 + CyclicSum((a**4-b**4)**2)/4 + CyclicSum(a**4*(b**2-c**2)**2)/4,
        ((8,0,0),(5,2,1)): CyclicSum((a**4-b**4)**2)/4 + CyclicSum(a**4*(b**2-c**2)**2)/4 + CyclicSum(a**2*(a**3-b**2*c)**2)/2,
        ((8,0,0),(5,1,2)): CyclicSum((a**4-b**4)**2)/4 + CyclicSum(a**4*(b**2-c**2)**2)/4 + CyclicSum(a**2*(a**3-b*c**2)**2)/2,
        ((8,0,0),(4,3,1)): CyclicSum((a**4-b**4)**2)/8 + CyclicSum(a**4*(a**2-b**2)**2)/4 + CyclicSum((a**4-b**3*c)**2)/2,
        ((8,0,0),(4,1,3)): CyclicSum((a**4-b**4)**2)/8 + CyclicSum(a**4*(a**2-c**2)**2)/4 + CyclicSum((a**4-b*c**3)**2)/2,
        ((8,0,0),(3,3,2)): CyclicSum((a**4-b**4)**2)/2 + CyclicSum(a**4*(b**2-c**2)**2)/2 + CyclicProduct(a**2)*CyclicSum((a-b)**2)/2,
    }
    @classmethod
    def amgm(cls, d1, d2):
        def _std(d):
            return max((d, (d[1],d[2],d[0]), (d[2],d[0],d[1])))
        return cls._SPECIAL_AMGMS.get((_std(d1), _std(d2)))