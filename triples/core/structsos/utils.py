from typing import Union, Tuple, List, Dict, Callable, Optional

import sympy as sp

from ..shared import uniquely_named_symbol, SS
from ...sdp import congruence as _congruence
from ...utils.expression import Coeff, CyclicSum, CyclicProduct
from ...utils.roots import nroots, rationalize_bound

class StructuralSOSError(Exception): ...

class PolynomialUnsolvableError(StructuralSOSError): ...

class PolynomialNonpositiveError(PolynomialUnsolvableError): ...


def radsimp(expr: Union[sp.Expr, List[sp.Expr]]) -> sp.Expr:
    """
    Rationalize the denominator by removing square roots. Wrapper of sympy.radsimp.
    Also refer to sympy.simplify.
    """
    if isinstance(expr, (list, tuple, sp.MatrixBase)):
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

def congruence(M: sp.Matrix) -> Tuple[sp.Matrix, sp.Matrix]:
    """
    Decompose a positive semidefinite matrix M as M = U.T @ S @ U. Returns U and S.
    """
    def signfunc(x):
        # handle nested radicals also, e.g. (sqrt(2)+2)/(sqrt(2)+1)- sqrt(2) == 0
        x = radsimp(x)
        if x > 0: return 1
        if x == 0: return 0
        return -1
    return _congruence(M, signfunc=signfunc)

def intervals(polys: List[sp.Poly]) -> List[sp.Expr]:
    """
    Return points where the polynomials change their signs.
    When one of the polynomials is not in QQ or ZZ, return [].
    If no signs are changed, return [0].
    """
    if len(polys) == 0:
        return [sp.S(0)]
    if any(_.domain not in [sp.QQ,sp.ZZ] for _ in polys):
        return []
    ret = []
    pre = None
    for (l,r), mul in sp.intervals(polys):
        if l != pre:
            ret.append(l)
            pre = l
        if r != pre:
            ret.append(r)
            pre = r
    if len(ret):
        return ret
    return [sp.S(0)]


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

def has_gen(gen: sp.Symbol, *args):
    """
    Test whether a symbol is involved in a (list of) polynomial(s).
    """
    to_iter = lambda x: (x,) if isinstance(x, sp.Poly) else x
    return any(any(gen in p.free_symbols for p in arg) for arg in map(to_iter, args))


def clear_free_symbols(poly: sp.Poly, ineq_constraints: Dict[sp.Poly, sp.Expr] = {}, eq_constraints: Dict[sp.Poly, sp.Expr] = {}) -> Tuple[sp.Poly, Dict[sp.Poly, sp.Expr], Dict[sp.Poly, sp.Expr]]:
    """
    Clear nuisance free symbols from the polynomial and constraints.
    For example, if we want to solve x>=4 with constraints x>=y, x*y>=4, y>=0, a>=0. Then
    we can remove the symbol "a" from the constraints. But we cannot remove the symbol "y"
    even though it is not in the polynomial, as it is correlated with "x".

    TODO: to be moved to shared.py?
    """
    # Construct the "correlation" graph of the free symbols: symbols that are path-connected to free
    # vars in the polynomial are considered as active symbols.
    gens = poly.gens
    ufs = dict((gen, gen) for i, gen in enumerate(gens))
    def ufsfind(x):
        if ufs[x] == x:
            return x
        ufs[x] = ufsfind(ufs[x])
        return ufs[x]
    def ufsunion(p):
        v = p.free_symbols
        if len(v):
            x0 = v.pop()
            y0 = ufsfind(x0)
            for x in v:
                ufs[ufsfind(x)] = y0
    ufsunion(poly)
    for p in ineq_constraints:
        ufsunion(p)
    for p in eq_constraints:
        ufsunion(p)

    # Using .free_symbols is not the same as .gens. Because free_symbols exclude 0-degree gens.
    active_gens = set(ufsfind(gen) for gen in poly.free_symbols)
    active_gens = [x for x in gens if ufsfind(x) in active_gens]
    if len(active_gens) == 0:
        active_gens = (gens[0],) # the polynomial is a constant, but we need a gen to create a poly

    def is_active(p):
        return len(p.free_symbols.intersection(active_gens))

    poly = poly.as_poly(active_gens)
    ineq_constraints = {p.as_poly(active_gens): e for p, e in ineq_constraints.items() if is_active(p)}
    eq_constraints = {p.as_poly(active_gens): e for p, e in eq_constraints.items() if is_active(p)}
    return poly, ineq_constraints, eq_constraints
