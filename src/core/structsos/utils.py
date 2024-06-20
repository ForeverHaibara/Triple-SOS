from typing import Tuple, Union, List, Dict, Callable, Optional

import sympy as sp
from sympy.utilities.iterables import iterable

from ...utils.roots.rationalize import nroots, rationalize, rationalize_bound

class Coeff():
    """
    A standard class for representing a polynomial with coefficients.
    """
    def __init__(self, coeffs: Union[sp.polys.Poly, Dict], is_rational: bool = True):
        if isinstance(coeffs, sp.Poly):
            self._nvars = len(coeffs.gens)
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
        else:
            self._nvars = len(next(iter(coeffs.keys()))) if len(coeffs) > 0 else 0
            
        self.coeffs = coeffs
        self.is_rational = is_rational

    @property
    def nvars(self) -> int:
        return self._nvars

    def __call__(self, *x) -> sp.Expr:
        """
        Coeff((i,j,k)) -> returns the coefficient of a^i * b^j * c^k.
        """
        if len(x) == 1 and iterable(x[0]):
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
        if self.is_zero:
            return Coeff({}, is_rational = True)

        assert self.nvars == 3, "The number of variables must be 3."

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
        Return the polynomial of given variables. If args is not given, it uses.
        """
        if len(args) == 0:
            if self.is_zero:
                args = sp.symbols('a')
            else:
                args = sp.symbols(f'a:{chr(96+self.nvars)}')
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
        Evalutate the polynomial at (1,1,...,1).
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

    def cancel_abc(self) -> Tuple[List[int], 'Coeff']:
        """
        Assume poly = a^i*b^j*c^k * poly2.
        Return ((i,j,k), Coeff(poly2)).
        """
        if self.is_zero:
            return ((0,) * self.nvars, self)
        all_monoms = list(self.coeffs.keys())
        d = self.degree() + 1
        common = [d] * self.nvars
        for monom in all_monoms:
            common = [min(i, j) for i, j in zip(common, monom)]
            if all(_ == 0 for _ in common):
                return ((0,) * self.nvars, self)

        common = tuple(common)
        new_coeff = Coeff({tuple([i - j for i, j in zip(monom, common)]): _ for monom, _ in self.coeffs.items() if _ != 0})
        new_coeff.is_rational = self.is_rational
        return common, new_coeff


    def cancel_k(self) -> Tuple[int, 'Coeff']:
        """
        Assume poly = Sum_{uvw}(x_{uvw} * a^{d*u} * b^{d*v} * c^{d*w}).
        Write poly2 = Sum_{uvw}(x_{uvw} * a^{u} * b^{v} * c^{w}).
        Return (d, Coeff(poly2))
        """
        if self.is_zero:
            return (1, self)
        all_monoms = list(self.coeffs.keys())
        d = 0
        for monom in all_monoms:
            for u in monom:
                d = sp.gcd(d, u)
                if d == 1:
                    return (1, self)

        d = int(d)
        if d == 0:
            return 0, self

        new_coeff = Coeff({tuple([i//d for i in monom]): _ for monom, _ in self.coeffs.items() if _ != 0})
        new_coeff.is_rational = self.is_rational
        return d, new_coeff


class StructuralSOSError(Exception): ...

class PolynomialUnsolvableError(StructuralSOSError): ...

class PolynomialNonpositiveError(PolynomialUnsolvableError): ...



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


# def radsimp_together(x: sp.Expr) -> sp.Expr:
#     """
#     Wrapper of sympy.together and radsimp.

#     >>> sp.together((x + y)/(1+sp.sqrt(3)))
#     (x + y)/(1 + sqrt(3))

#     >>> radsimp_together(((x + y)/(1+sp.sqrt(3))))
#     (-1 + sqrt(3))*(x + y)/2
#     """
#     f1, f2 = x.together().as_numer_denom()
#     x1, y1 = f1.as_coeff_Mul()
#     if f2.is_constant():
#         x1, f2 = radsimp(x1/f2).as_numer_denom()
#         return x1 * y1 / f2

#     x2, y2 = f2.as_coeff_Mul()
#     return radsimp(x1/x2) * y1/y2


def sum_y_exprs(y: List[sp.Expr], exprs: List[sp.Expr]) -> sp.Expr:
    """
    Return sum(y_i * expr_i).
    """
    def _mul(v, expr):
        if v == 0: return 0
        x, f = (v * expr).radsimp(symbolic=False).together().as_coeff_Mul()
        return radsimp(x) * f
    return sum(_mul(*args) for args in zip(y, exprs))


def rationalize_func(
        poly: Union[sp.Poly, sp.Rational],
        validation: Callable[[sp.Rational], bool],
        validation_initial: Optional[Callable[[sp.Rational], bool]] = None,
        direction: int = 0,
    ) -> Optional[sp.Rational]:
    """
    Find a rational number near the roots of poly that satisfies certain conditions.

    Parameters
    ----------
    poly : Union[sp.Poly, sp.Rational]
        Initial values are near to the roots of the polynomial.
    validation : Callable
        Return True if validation(..) >= 0.
    validation_initial : Optional[Callable]
        The function first uses numerical roots of the poly, and it
        might not satisfy the validation function because of the numerical error.
        Configure this function to roughly test whether a root is proper.
        When None, it uses the validation function as default.
    direction : int
        When direction = 1, requires poly(..) >= 0. When direction = -1, requires
        poly(..) <= 0. When direction = 0 (defaulted), no addition requirement is imposed.

    Returns
    ----------
    t : sp.Rational
        Proper rational number that satisfies the validation conditions.
        Return None if no such t is found.
    """
    validation_initial = validation_initial or validation

    if isinstance(poly, sp.Poly):
        candidates = nroots(poly, method = 'factor', real = True)
        poly_diff = poly.diff()
        if direction != 0:
            def direction_t(t):
                return direction if poly_diff(t) >= 0 else -direction
            def validation_t(t):
                return sp.sign(poly(t)) * direction >= 0 and validation(t)
        else:
            direction_t = lambda t: 0
            validation_t = lambda t: validation(t)

    elif isinstance(poly, (int, float, sp.Rational)):
        candidates = [poly]
        direction_t = lambda t: direction
        validation_t = lambda t: validation(t)


    for t in candidates:
        if isinstance(t, sp.Rational):
            if validation(t):
                return t    
        elif validation_initial(t):
            # make a perturbation
            for t_ in rationalize_bound(t, direction = direction_t(t), compulsory = True):
                if validation_t(t_):
                    return t_


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
        If 4*c1*c3 < c2**2 or c1 < 0 or c3 < 0, return None.
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