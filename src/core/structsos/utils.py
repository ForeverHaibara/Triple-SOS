from typing import Union, List, Dict, Callable, Optional

import sympy as sp
from sympy.core.singleton import S

from ..symsos import prove_univariate
from ...utils.roots.rationalize import rationalize, rationalize_bound, square_perturbation, cancel_denominator
from ...utils.roots.findroot import nroots
from ...utils import congruence, CyclicSum, CyclicProduct

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

    def reflect(self) -> 'Coeff':
        """
        Reflect the coefficients of a, b, c with respect to a,b.
        Returns a deepcopy.
        """
        reflected_coeffs = dict([((j,i,k), v) for (i,j,k), v in self.coeffs.items()])
        new_coeff = Coeff(reflected_coeffs, is_rational = self.is_rational)
        return new_coeff

    def clear_zero(self) -> None:
        """
        Clear the coefficients that are zero.
        """
        self.coeffs = {k:v for k,v in self.coeffs.items() if v != 0}

    def as_poly(self, *args) -> sp.Poly:
        """
        Return the polynomial of a, b, c.
        """
        if len(args) == 0:
            args = sp.symbols('a b c')
        return sp.polys.Poly.from_dict(self.coeffs, gens = args)

    def degree(self) -> int:
        """
        Return the degree of the polynomial.
        """
        if len(self.coeffs) == 0:
            return 0
        for k in self.coeffs:
            return sum(k)

    @property
    def is_zero(self) -> bool:
        """
        Whether the polynomial is zero.
        """
        return len(self.coeffs) == 0

    def poly111(self) -> sp.Expr:
        """
        Evalutate the polynomial at (a,b,c) = (1,1,1).
        """
        return sum(self.coeffs.values())

    def items(self):
        return self.coeffs.items()

    def __operator__(self, other, operator) -> 'Coeff':
        new_coeffs = self.coeffs.copy()
        for k, v2 in other.items():
            v1 = self(k)
            v3 = operator(v1, v2)
            if v3 == 0 and v1 != 0:
                del new_coeffs[k]
            elif v3 != 0:
                new_coeffs[k] = v3
        new_coeffs = dict(sorted(new_coeffs.items(), reverse=True))
        other_rational = (not isinstance(other, Coeff)) or other.is_rational
        is_rational = self.is_rational and other_rational
        return Coeff(new_coeffs, is_rational = is_rational)

    def __add__(self, other) -> 'Coeff':
        return self.__operator__(other, lambda x, y: x + y)

    def __sub__(self, other) -> 'Coeff':
        return self.__operator__(other, lambda x, y: x - y)

    # def __mul__(self, other) -> 'Coeff':
    #     return self.__operator__(other, lambda x, y: x * y)

    # def __truediv__(self, other) -> 'Coeff':
    #     return self.__operator__(other, lambda x, y: x / y)

    def __pow__(self, other) -> 'Coeff':
        return self.__operator__(other, lambda x, y: x ** y)



def radsimp(expr: Union[sp.Expr, List[sp.Expr]]) -> sp.Expr:
    """
    Rationalize the denominator by removing square roots. Wrapper of sympy.radsimp.
    Also refer to sympy.simplify.
    """
    if isinstance(expr, (list, tuple)):
        return [radsimp(e) for e in expr]
    if not isinstance(expr, sp.Expr):
        expr = sp.sympify(expr)
    if isinstance(expr, sp.Rational):
        return expr

    numer, denom = expr.as_numer_denom()
    n, d = sp.fraction(sp.radsimp(1/denom, symbolic=False, max_terms=1))
    # if n is not S.One:
    expr = (numer*n).expand()/d
    return expr


def sum_y_exprs(y: List[sp.Expr], exprs: List[sp.Expr]) -> sp.Expr:
    """
    Return sum(y_i * expr_i).
    """
    return sum(v * expr for v, expr in zip(y, exprs) if v != 0)


def rationalize_func(
        poly: sp.Poly,
        validation: Callable[[sp.Rational], bool],
        direction: int = 0,
    ) -> Optional[sp.Rational]:
    """
    Find a rational number near the roots of poly that satisfies certain conditions.

    Parameters
    ----------
    poly : sp.Poly
        Initial value are near to the roots of the polynomial.
    validation : Callable
        Return True if validation(..) >= 0.
    direction : int
        When direction = 1, requires poly(..) >= 0. When direction = -1, requires
        poly(..) <= 0. When direction = 0 (defaulted), no addition requirement is imposed.

    Returns
    ----------
    t : sp.Rational
        Proper rational number that satisfies the validation conditions.
        Return None if no such t is found.
    """
    for t_ in nroots(poly, method = 'factor', real = True):
        if validation(t_):
            break
    else:
        return None

    if isinstance(t_, sp.Rational):
        return t_
    else:
        # make a perturbation
        if direction != 0:
            direction_t = direction if poly.diff()(t_) >= 0 else -direction
            validation_ = lambda t: sp.sign(poly(t)) * direction >= 0 and validation(t)
        else:
            direction_t = 0
            validation_ = validation

        for t__ in rationalize_bound(t_, direction = direction_t, compulsory = True):
            if validation_(t__):
                return t__


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


def quadratic_weighting(
        c1: sp.Rational,
        c2: sp.Rational,
        c3: sp.Rational,
        a: Optional[sp.Expr] = None,
        b: Optional[sp.Expr] = None,
        mapping: Optional[Callable[[sp.Rational, sp.Rational], sp.Expr]] = None,
        formal: bool = False
    ) -> Union[sp.Expr, List]:
    """
    Give solution to c1*a^2 + c2*a*b + c3*b^2 >= 0 where a,b in R.

    Parameters
    ----------
    c1, c2, c3 : sp.Expr
        Coefficients of the quadratic form.
    a, b : sp.Expr
        The basis of the quadratic form.
    mapping : Callable
        A function that receives two inputs, x, y, and
        outputs the desired (x*a + y*b)**2. Default is
        mapping = lambda x, y: (x*a + y*b)**2.
        If mapping is not None, it overrides the parameters a, b.
    formal : bool
        If True, return a list [(w1, (x1,y1))] so that sum(w_i * (x_i*a + y_i*b)**2) equals to the result.
        If False, return the sympy expression of the result.
        If formal == True, it overrides the mapping parameter.
    
    Returns
    ----------
    result : Union[sp.Expr, List]
        If formal = False, return the sympy expression of the result.
        If formal = True, return a list [(w1, (x1,y1))] so that sum(w_i * (x_i*a + y_i*b)**2) equals to the result.
    """
    if 4*c1*c3 < c2**2 or c1 < 0 or c3 < 0:
        return None
    c1, c2, c3 = radsimp(c1), radsimp(c2), radsimp(c3)

    a = a or sp.Symbol('a')
    b = b or sp.Symbol('b')
    mapping = mapping or (lambda x, y: (x*a + y*b)**2)

    if c1 == 0:
        result = [(c3, (sp.S(0), sp.S(1)))]
    elif c3 == 0:
        result = [(c1, (sp.S(1), sp.S(0)))]
    else:
        # ratio = c2/c3/2
        # result = [(c3, b + ratio*a), (c1 - ratio**2*c3, a)]
        ratio = radsimp(sp.S(c2)/c1/2)
        result = [(c1, (sp.S(1), ratio)), (c3 - ratio**2*c1, (sp.S(0), sp.S(1)))]

    if formal:
        return result

    return sum(radsimp(wi) * mapping(*xi) for wi, xi in result)


def zip_longest(*args):
    """
    Zip longest generators and pad the length with the final element.
    """
    if len(args) == 0: return
    args = [iter(arg) for arg in args]
    lasts = [None] * len(args)
    stops = [False] * len(args)
    while True:
        for i, gen in enumerate(args):
            if stops[i]:
                continue
            try:
                lasts[i] = next(gen)
            except StopIteration:
                stops[i] = True
                if all(stops):
                    return
        yield tuple(lasts)


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
        solution = recurrsion(Coeff(poly2))
        if solution is not None:
            return solution + t * perturbation
    return None
