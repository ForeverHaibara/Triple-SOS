from typing import Tuple, Optional

from sympy import Poly, Symbol
from sympy.polys.rings import PolyElement
from sympy.utilities import subsets


def _symmetric_poly(self, n):
    """
    Return the elementary symmetric polynomial of degree *n* over
    this ring's generators.
    """
    if n < 0 or n > self.ngens:
        raise ValueError("Cannot generate symmetric polynomial of order %s for %s" % (n, self.gens))
    elif not n:
        return self.one
    else:
        poly = self.zero
        for s in subsets(range(self.ngens), int(n)):
            monom = tuple(int(i in s) for i in range(self.ngens))
            poly += self.term_new(monom, self.domain.one)
        return poly

def _symmetrize(self: PolyElement):
    """
    `PolyElement.symmetrize` was added since SymPy 1.12. This
    function is copied from the SymPy source code for version
    compatibility.
    """
    f = self.copy()
    ring = f.ring
    n = ring.ngens

    if not n:
        return f, ring.zero, []

    polys = [_symmetric_poly(ring, i+1) for i in range(n)]

    poly_powers = {}
    def get_poly_power(i, n):
        if (i, n) not in poly_powers:
            poly_powers[(i, n)] = polys[i]**n
        return poly_powers[(i, n)]

    indices = list(range(n - 1))
    weights = list(range(n, 0, -1))

    symmetric = ring.zero

    while f:
        _height, _monom, _coeff = -1, None, None

        for i, (monom, coeff) in enumerate(f.terms()):
            if all(monom[i] >= monom[i + 1] for i in indices):
                height = max(n*m for n, m in zip(weights, monom))

                if height > _height:
                    _height, _monom, _coeff = height, monom, coeff

        if _height != -1:
            monom, coeff = _monom, _coeff
        else:
            break

        exponents = []
        for m1, m2 in zip(monom, monom[1:] + (0,)): # type: ignore
            exponents.append(m1 - m2)

        symmetric += ring.term_new(tuple(exponents), coeff)

        product = coeff
        for i, n in enumerate(exponents):
            product *= get_poly_power(i, n)
        f -= product

    mapping = list(zip(ring.gens, polys))

    return symmetric, f, mapping


def _get_pqr_symbols(symbols: Optional[Tuple[Symbol, Symbol, Symbol]] = None) -> Tuple[Symbol, Symbol, Symbol]:
    """Return p,q,r from symbols. If None, create new symbols."""
    if symbols is not None:
        return symbols
    else:
        from sympy import symbols as sp_symbols
        return sp_symbols('p q r')

def pqr_sym(poly: Poly, symbols: Optional[Tuple[Symbol, ...]] = None) -> Poly:
    """
    Express an n-variable symmetric polynomial in its
    elementary symmetric polynomials.

    Parameters
    ----------
    poly : Poly
        An n-variable symmetric polynomial.
    symbols : Optional[Tuple[Symbol, ...]]
        If None, use the original variables for the new vars.
        Otherwise, use the given list of symbols.

    Returns
    -------
    part_sym : Poly
        The function in the bases of its elementary symmetric polynomials.

    Examples
    --------
    >>> from sympy.abc import a, b, c, x, y, z
    >>> pqr_sym(((a-b)**2*(b-c)**2*(c-a)**2).as_poly(a,b,c))
    Poly(-4*a**3*c + a**2*b**2 + 18*a*b*c - 4*b**3 - 27*c**2, a, b, c, domain='ZZ')
    >>> p0 = pqr_sym(((a-b)**2*(b-c)**2*(c-a)**2).as_poly(a,b,c), (x,y,z)); p0
    Poly(-4*x**3*z + x**2*y**2 + 18*x*y*z - 4*y**3 - 27*z**2, x, y, z, domain='ZZ')

    >>> p0(a+b+c, a*b+b*c+c*a, a*b*c).factor()
    (a - b)**2*(a - c)**2*(b - c)**2
    """
    if symbols is None:
        symbols = poly.gens # type: ignore
    elif len(poly.gens) != len(symbols):
        raise ValueError("Symbols must match the number of variables in the polynomial.")

    # f, g, rule = symmetrize(poly.as_expr(), poly.gens, formal=True)

    ring = poly.domain[tuple(poly.gens)]
    p = ring(poly.rep.to_dict())
    f, g, rule = _symmetrize(p)
    if g != 0:
        raise ValueError("The polynomial is not symmetric.")
    # return f.as_expr(*symbols)

    return Poly.from_dict(f.to_dict(), symbols, domain = poly.domain)


def pqr_cyc(poly: Poly, symbols: Optional[Tuple[Symbol, ...]] = None) -> Tuple[Poly, Poly]:
    """
    Express a 3-variable cyclic polynomial P in p,q,r form,
    such that P(p,q,r) = f(p,q,r) + (a-b)*(b-c)*(c-a) * g(p,q,r)
    where p = a+b+c, q = ab+bc+ca, r = abc.

    Parameters
    ----------
    poly : Poly
        A 3-variable cyclic polynomial.
    symbols : Optional[Tuple[Symbol, ...]]
        If None, use p,q,r to stand for the new vars.
        Otherwise, use the given list of symbols.

    Returns
    -------
    part_sym : Poly
        The f(p,q,r) part.
    part_cyc : Poly
        The g(p,q,r) part.

    Examples
    --------
    >>> from sympy.abc import a, b, c, x, y, z
    >>> pqr_cyc((a**2*b+b**2*c+c**2*a).as_poly(a,b,c))
    (Poly(1/2*a*b - 3/2*c, a, b, c, domain='QQ'), Poly(-1/2, a, b, c, domain='QQ'))
    >>> pqr_cyc((a**2*b+b**2*c+c**2*a).as_poly(a,b,c), (x,y,z))
    (Poly(1/2*x*y - 3/2*z, x, y, z, domain='QQ'), Poly(-1/2, x, y, z, domain='QQ'))
    """
    if len(poly.gens) != 3:
        raise ValueError("The polynomial must be a 3-variable polynomial.")

    a, b, c = poly.gens
    if not (poly.domain.is_Composite and poly.domain.domain.is_Field): # type: ignore
        poly = poly.to_field()
    half = poly.domain.one/2
    q = Poly.new(poly.reorder(b,a,c).rep,a,b,c)
    f0 = (poly + q).mul_ground(half)
    f1 = (poly - q).mul_ground(half).div(
        ((a-b)*(b-c)*(c-a)).as_poly(a,b,c)
    )
    if not f1[1].is_zero:
        raise ValueError("The polynomial is not cyclic.")
    try:
        f0 = pqr_sym(f0, symbols)
        f1 = pqr_sym(f1[0], symbols)
    except ValueError:
        raise ValueError("The polynomial is not cyclic.")
    return f0, f1


def pqr_ker(symbols: Optional[Tuple[Symbol, Symbol, Symbol]] = None) -> Poly:
    """
    Return the pqr representation of `((a-b)*(b-c)*(c-a))^2`.

    It should be `-4*p**3*r + p**2*q**2 + 18*p*q*r - 4*q**3 - 27*r**2`.
    """
    p, q, r = _get_pqr_symbols(symbols)
    return Poly(-4*p**3*r + p**2*q**2 + 18*p*q*r - 4*q**3 - 27*r**2, p, q, r)

# def pqr_pqrt(a, b, c = 1) -> Tuple[Expr, Expr, Expr, Expr]:
#     """
#     Compute the p,q,r,t with p = 1 given a, b, c.
#     """
#     a, b, c = sympify(a), sympify(b), sympify(c)
#     w = c + a + b
#     q = (a*c + a*b + b*c) / w**2
#     return sympify(1), q, a*b*c/w**3, sqrt(1-3*q)
