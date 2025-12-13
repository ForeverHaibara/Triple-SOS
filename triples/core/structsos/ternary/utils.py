from typing import Tuple, Callable, Optional, Union
from functools import wraps

import sympy as sp
from sympy import Poly, Expr, Symbol, Add, Mul, Pow
from sympy.combinatorics import CyclicGroup

from ..utils import (
    Coeff, DomainExpr,
    radsimp, sum_y_exprs, rationalize_func, quadratic_weighting, zip_longest, congruence,
    StructuralSOSError, PolynomialNonpositiveError, PolynomialUnsolvableError
)

from ....utils import (
    nroots, rationalize, rationalize_bound, univariate_intervals,
    square_perturbation, cancel_denominator, common_region_of_conics,
    CyclicExpr, CyclicSum, CyclicProduct
)

def align_cyclic_group(expr: Optional[Expr], gens: Tuple[Symbol, ...]) -> Expr:
    """
    Replace `CyclicSum(F(a, b, c), (a, c, b))` to `CyclicSum(F(a, b, c), (a, b, c))`.
    Note that this does not change the value of the expression.
    The function depends on the fact that `CyclicExpr` automatically sorts the
    symbols and might change its behaviour accordingly in the future.
    """
    if expr is None:
        return None
    cg = CyclicGroup(3)
    gens = CyclicSum(gens[0], gens).args[1] # sort the gens by default
    wrong_gens = (gens[0], gens[2], gens[1])
    def _recur(expr):
        if isinstance(expr, CyclicExpr):
            if expr.args[1] == wrong_gens and expr.args[2] == cg:
                return expr.func(expr.args[0], gens, cg)
        if not expr.has(CyclicExpr):
            return expr
        return expr.func(*[_recur(a) for a in expr.args])
    return _recur(expr)


def inverse_substitution(coeff: Coeff, expr: Expr, factor_degree: int = 0) -> Expr:
    """
    Substitute a <- b * c, b <- c * a, c <- a * b into expr.
    Then the function extract the common factor of the expression, usually (abc)^k.
    Finally the expression is divided by (abc)^(factor_degree).
    """
    a, b, c = coeff.gens
    expr = sp.together(expr.xreplace({a:b*c,b:c*a,c:a*b}))

    def _try_factor(expr):
        if isinstance(expr, (Add, Mul, Pow)):
            return expr.func(*[_try_factor(arg) for arg in expr.args])
        elif isinstance(expr, CyclicSum):
            # Sum(a**3*b**2*c**2*(...)**2)
            if isinstance(expr.args[0], Mul):
                args2 = expr.args[0].args
                symbol_degrees = {}
                other_args = []
                for s in args2:
                    if s in (a,b,c):
                        symbol_degrees[s] = 1
                    elif isinstance(s, Pow) and s.base in (a,b,c):
                        symbol_degrees[s.base] = s.exp
                    else:
                        other_args.append(s)
                if len(symbol_degrees) == 3:
                    degree = min(symbol_degrees.values())
                    da, db, dc = symbol_degrees[a], symbol_degrees[b], symbol_degrees[c]
                    da, db, dc = da - degree, db - degree, dc - degree
                    other_args.extend([a**da, b**db, c**dc])
                    return CyclicSum(Mul(*other_args), (a,b,c)) * CyclicProduct(a, (a,b,c)) ** degree
        elif isinstance(expr, CyclicProduct):
            # Product(a**2) = Product(a) ** 2
            if isinstance(expr.args[0], Pow) and expr.args[0].base in (a,b,c):
                return CyclicProduct(expr.args[0].base, (a,b,c)) ** expr.args[0].exp
        return expr

    expr = sp.together(_try_factor(expr))
    if factor_degree != 0:
        expr = expr / CyclicProduct(a, (a,b,c)) ** factor_degree
    return expr


def try_perturbations(
        poly, p, q, perturbation, times = 4, **kwargs
    ):
    """
    Try subtracting t * perturbation from poly and perform recurrsive trials.
    The subtracted t satisfies that (p - t) / (q - t) is a square

    This is possibly helpful for deriving rational sum-of-square solution.
    """
    from .solver import _structural_sos_3vars_cyclic
    perturbation_poly = perturbation.doit().as_poly(poly.gens)
    for t in square_perturbation(p, q, times = times):
        poly2 = poly - t * perturbation_poly
        solution = _structural_sos_3vars_cyclic(Coeff(poly2))
        if solution is not None:
            return solution + t * perturbation
    return None


def sos_struct_handle_uncentered(solver: Callable) -> Callable:
    """
    A decorator for structural SOS with uncentered polynomial handling.
    It only supports cyclic polynomials.
    """
    @wraps(solver)
    def _wrapped_solver(poly: Union[Poly, Coeff], *args, **kwargs):
        bias = 0
        if isinstance(poly, Coeff):
            bias = poly.poly111()
        elif isinstance(poly, Poly):
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
        coeff = dict(poly.terms())
        if not ((i,j,k) in coeff):
            coeff[(i,j,k)] = 0
            coeff[(j,k,i)] = 0
            coeff[(k,i,j)] = 0
        coeff[(i,j,k)] -= bias/3
        coeff[(j,k,i)] -= bias/3
        coeff[(k,i,j)] -= bias/3

        new_poly = poly.from_dict(coeff)
        solution = solver(new_poly, *args, **kwargs)
        if solution is not None:
            a, b, c = poly.gens
            CyclicSum, CyclicProduct = poly.cyclic_sum, poly.cyclic_product
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
    def schur(cls, n, symbols = None):
        """
        Solve s(a^(n-2)*(a-b)*(a-c)) when n > 0
        """
        if n < 2:
            return

        a, b, c = cls.abc if symbols is None else symbols
        symbols = (a, b, c)
        cyc_sum = lambda z: CyclicSum(z, symbols)
        cyc_prod = lambda z: CyclicProduct(z, symbols)

        if n == 2:
            return cyc_sum((a - b)**2)/2
        elif n == 3:
            return (cyc_sum((a-b)**2*(a+b-c)**2) + 2*cyc_sum(a*b*(a-b)**2))/(2*cyc_sum(a))
        elif n == 5:
            return 2*(cyc_sum(a**3*(a-b)**2*(a-c)**2)+cyc_sum(a)*cyc_prod((a-b)**2))/cyc_sum((a-b)**2)

    @classmethod
    def schurinv(cls, n, symbols = None):
        """
        Solve s(b^((n-2)/2)*c^((n-2)/2)*(a-b)*(a-c)) when n > 0
        """
        a, b, c = cls.abc if symbols is None else symbols
        symbols = (a, b, c)
        cyc_sum = lambda z: CyclicSum(z, symbols)
        cyc_prod = lambda z: CyclicProduct(z, symbols)

        if n == 2:
            return cyc_sum((a-b)**2)
        elif n == 4:
            return cyc_sum(a**2*(b-c)**2/2)
        elif n == 6:
            return (cyc_sum(c*(a-b)**2*(a*c+b*c-a*b)**2)*2 + cyc_prod(a)*cyc_sum(a**2*(b-c)**2))/\
                (2*cyc_sum(a))

    @classmethod
    def quadratic(cls, x, y, symbols = None):
        """
        Solve s(x*a**2 + y*a*b)
        """
        if x < 0 or x + y < 0:
            return
        x, y = sp.S(x), sp.S(y)

        a, b, c = cls.abc if symbols is None else symbols
        symbols = (a, b, c)
        cyc_sum = lambda z: CyclicSum(z, symbols)

        if x == 0:
            return y * cyc_sum(a*b)
        if y == 0:
            return x * cyc_sum(a**2)
        if x == y:
            return x/2 * cyc_sum((a+b)**2)
        if y > 2 * x:
            return cyc_sum(x * a**2 + y * b*c)
        w1 = (2*x - y) / 3
        w2 = x - w1
        return w1 / 2 * cyc_sum((a-b)**2) + w2 * cyc_sum(a)**2


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
    def amgm(cls, d1, d2, symbols = None):
        def _std(d):
            return max((d, (d[1],d[2],d[0]), (d[2],d[0],d[1])))
        v = cls._SPECIAL_AMGMS.get((_std(d1), _std(d2)))
        if v is None:
            return None

        if symbols is not None:
            v = v.xreplace(dict(zip(cls.abc, symbols)))
        return v
