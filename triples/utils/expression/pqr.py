from typing import Tuple, List, Optional

import sympy as sp

def _get_pqr_symbols(symbols: Optional[List[sp.Symbol]] = None) -> Tuple[sp.Symbol, sp.Symbol, sp.Symbol]:
    """Return p,q,r from symbols. If None, create new symbols."""
    if symbols is not None:
        p, q, r = symbols
    else:
        p, q, r = sp.symbols('p q r')
    return p, q, r

def pqr_sym(poly: sp.Poly, symbols: Optional[List[sp.Symbol]] = None) -> sp.Expr:
    """
    Express a 3-variable symmetric polynomial P in p,q,r form,
    such that P(p,q,r) = f(p,q,r) + (a-b)*(b-c)*(c-a) * g(p,q,r)
    where p = a+b+c, q = ab+bc+ca, r = abc.

    Parameters
    ----------
    poly : sp.Poly
        A 3-variable cyclic polynomial.
    symbols : Optional[List[sp.Symbol]]
        If None, use p,q,r to stand for the new vars.
        Otherwise, use the given list of symbols.

    Returns
    ----------
    part_sym : sp.Expr
        The f(p,q,r) part.
    part_cyc : sp.Expr
        The g(p,q,r) part.

    Examples
    ----------
    >>> a, b, c, x, y, z = sp.symbols('a b c x y z')
    >>> pqr_sym(((a-b)**2*(b-c)**2*(c-a)**2).as_poly(a,b,c))
    -4*p**3*r + p**2*q**2 + 18*p*q*r - 4*q**3 - 27*r**2
    >>> pqr_sym(((a-b)**2*(b-c)**2*(c-a)**2).as_poly(a,b,c), (x,y,z))
    -4*x**3*z + x**2*y**2 + 18*x*y*z - 4*y**3 - 27*z**2
    """
    if len(poly.gens) != 3 and symbols is None:
        raise ValueError("Symbols must be manually specified for non-3-variable polynomials.")
    symbols = _get_pqr_symbols(symbols)
    if len(poly.gens) != len(symbols):
        raise ValueError("Symbols must match the number of variables in the polynomial.")

    f, g, rule = sp.symmetrize(poly.as_expr(), poly.gens, formal=True)
    if g != 0:
        raise ValueError("The polynomial is not symmetric.")
    replacement = dict(zip([_[0] for _ in rule], symbols))
    f = f.xreplace(replacement)
    return f


def pqr_cyc(poly: sp.Poly, symbols: Optional[List[sp.Symbol]] = None) -> Tuple[sp.Expr, sp.Expr]:
    """
    Express a 3-variable cyclic polynomial P in p,q,r form,
    such that P(p,q,r) = f(p,q,r) + (a-b)*(b-c)*(c-a) * g(p,q,r)
    where p = a+b+c, q = ab+bc+ca, r = abc.

    Parameters
    ----------
    poly : sp.Poly
        A 3-variable cyclic polynomial.
    symbols : Optional[List[sp.Symbol]]
        If None, use p,q,r to stand for the new vars.
        Otherwise, use the given list of symbols.

    Returns
    ----------
    part_sym : sp.Expr
        The f(p,q,r) part.
    part_cyc : sp.Expr
        The g(p,q,r) part.

    Examples
    ----------
    >>> a, b, c = sp.symbols('a b c')
    >>> pqr_cyc((a**2*b+b**2*c+c**2*a).as_poly(a,b,c))
    (p*q/2 - 3*r/2, -1/2)
    """
    if len(poly.gens) != 3:
        raise ValueError("The polynomial must be a 3-variable polynomial.")

    a, b, c = poly.gens
    p, q, r = _get_pqr_symbols(symbols)
    f0 = ((poly + sp.Poly.new(poly.reorder(b,a,c).rep,a,b,c))/2).as_poly(a,b,c)
    f1 = ((poly - sp.Poly.new(poly.reorder(b,a,c).rep,a,b,c))/2).as_poly(a,b,c).div(
        ((a-b)*(b-c)*(c-a)).as_poly(a,b,c)
    )
    if not f1[1].is_zero:
        raise ValueError("The polynomial is not cyclic.")
    try:
        f0 = pqr_sym(f0, (p,q,r))
        f1 = pqr_sym(f1[0], (p,q,r))
    except ValueError:
        raise ValueError("The polynomial is not cyclic.")
    return f0, f1


def pqr_ker(symbols: Optional[List[sp.Symbol]] = None) -> sp.Expr:
    """
    Return the pqr representation of ((a-b)*(b-c)*(c-a))^2.

    It should be -4*p**3*r + p**2*q**2 + 18*p*q*r - 4*q**3 - 27*r**2.
    """
    p, q, r = _get_pqr_symbols(symbols)
    return -4*p**3*r + p**2*q**2 + 18*p*q*r - 4*q**3 - 27*r**2

# def pqr_pqrt(a, b, c = 1) -> Tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
#     """
#     Compute the p,q,r,t with p = 1 given a, b, c.
#     """
#     a, b, c = sp.S(a), sp.S(b), sp.S(c)
#     w = c + a + b
#     q = (a*c + a*b + b*c) / w**2
#     return sp.S(1), q, a*b*c/w**3, sp.sqrt(1-3*q)