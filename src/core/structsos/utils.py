from typing import Union, List, Dict, Callable

import sympy as sp
from sympy.core.singleton import S

from ..symsos import prove_univariate
from ...utils.roots.rationalize import rationalize, rationalize_bound, square_perturbation, cancel_denominator
from ...utils.roots.findroot import nroots
from ...utils.expression.cyclic import CyclicSum, CyclicProduct

class Coeff():
    """
    A standard class for representing a polynomial with coefficients.
    """
    def __init__(self, coeffs: Union[sp.polys.Poly, Dict], is_rational: bool = True):
        if isinstance(coeffs, sp.polys.Poly):
            poly = coeffs
            coeffs = {}
            for monom, coeff in poly.terms():
                if not isinstance(coeff, sp.Rational): #isinstance(coeff, sp.Float): # and degree > 4
                    if isinstance(coeff, sp.Float):
                        coeff = rationalize(coeff, reliable = True)
                    else:
                        is_rational = False
                    # coeff = coeff.as_numer_denom()
                coeffs[monom] = coeff
            
        self.coeffs = coeffs
        self.is_rational = is_rational

    def __call__(self, *x) -> sp.Expr:
        """
        Coeff((i,j,k)) -> returns the coefficient of a^i * b^j * c^k.
        """
        if len(x) == 1:
            # x is ((a,b,c), )
            x = x[0]
        return self.coeffs.get(x, sp.S(0))

    def __len__(self) -> int:
        """
        Number of coefficients. Sometimes the zero coefficients are not included.
        """
        return len(self.coeffs)

    def reflect(self):
        """
        Reflect the coefficients of a, b, c with respect to a,b.
        Returns a deepcopy.
        """
        reflected_coeffs = dict([((j,i,k), v) for (i,j,k), v in self.coeffs.items()])
        new_coeff = Coeff(reflected_coeffs, is_rational = self.is_rational)
        return new_coeff


def sum_y_exprs(y: List[sp.Expr], exprs: List[sp.Expr]) -> sp.Expr:
    """
    Return sum(y_i * expr_i).
    """
    return sum(v * expr for v, expr in zip(y, exprs) if v != 0)


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


def quadratic_weighting(c1, c2, c3, a = None, b = None, formal = False) -> Union[sp.Expr, List]:
    """
    Give solution to c1*a^2 + c2*b^2 + c3*a*b >= 0.

    Parameters
    -------
    c1, c2, c3: sp.Expr
        Coefficients of the quadratic form.
    a, b: sp.Expr
        The basis of the quadratic form.
    formal: bool
        Whether return a list or a sympy expression.
    
    Returns
    -------
    If formal == True, return a list [(w1, x1), (w2, x2), ...] so that sum(w_i * x_i**2) equals to the result.
    If formal == False, return the sympy expression of the result.
    """
    if 4*c1*c2 < c3**2:
        return None
    if a is None:
        a = sp.symbols('a')
    if b is None:
        b = sp.symbols('b')

    if c1 == 0:
        result = [(c2, b)]
    elif c2 == 0:
        result = [(c1, a)]
    else:
        ratio = c3/c1/2
        result = [(c1, a + ratio*b), (c2 - ratio**2*c1, b)]
    
    if formal:
        return result
    return sum(wi * xi**2 for wi, xi in result)


def radsimp(expr: Union[sp.Expr, List[sp.Expr]]) -> sp.Expr:
    """
    Rationalize the denominator by removing square roots. Wrapper of sympy.radsimp.
    Also refer to sympy.simplify.
    """
    if isinstance(expr, (list, tuple)):
        return [radsimp(e) for e in expr]
    if not isinstance(expr, sp.Expr):
        expr = sp.sympify(expr)

    numer, denom = expr.as_numer_denom()
    n, d = sp.fraction(sp.radsimp(1/denom, symbolic=False, max_terms=1))
    # if n is not S.One:
    expr = (numer*n).expand()/d
    return expr


def try_perturbations(
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
