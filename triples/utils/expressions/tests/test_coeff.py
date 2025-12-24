from ..coeff import PartialOrderElement, Coeff

from sympy import Poly, Expr, Rational, ZZ, QQ, RR, EX, E, sqrt
from sympy.abc import a, b, c

import pytest

def _get_test_cases():
    cases = []

    polys = [
        Poly(3*(a+b+c)**2/4 - 5*b**2, a, b, c, domain = QQ),
        Poly(1.2*a**2*b + 4/7*c**2 - 3/11*a*b*c, a, b, c, domain = RR),
        Poly(sqrt(7)*a**2*b + 2*c + (3 - sqrt(7)/11)*b, a, b, c,
            domain = QQ.algebraic_field(sqrt(7))),
        Poly((c + 2)*(a**2 + b)/(c**2 + 1) - 4*(b**2*a + c)/(3*c),
            a, b, domain = ZZ[c].get_field()),
        Poly((E + 2)*(a**2 + b)/(E**2 + 1) - 4*(b**2*a + E)/(3*E),
            b, a, domain = QQ[E].get_field()),
        Poly(3*(a+b+c)**2/4 - 5*b**2, b, c, domain = EX),
    ]
    coeffs = [Coeff(p) for p in polys]

    # python integers # and QQ.rational
    for l in (0, 6): #, QQ(0), QQ(6), QQ(-71, 97)):
        for cf in coeffs:
            lm = next(iter(cf.keys()))
            r = cf(lm)
            cases.append((l, r, type(r), True))

    # self and self
    for cf in coeffs:
        lm = next(iter(cf.keys()))
        r = cf(lm)
        cases.append((r*2/5, 3-r/4, type(r), True))
        cases.append((r*0, 3-r/4, type(r), True))

    # self and domain element
    for cf in coeffs:
        lm = next(iter(cf.keys()))
        r = cf(lm)
        cases.append((r, cf.domain.zero, type(r), False))
        cases.append((r, cf.domain.one, type(r), False))

    # sympy exprs
    # for l in (Rational(0), Rational(6), Rational(-71,97)):
    for l in (a, c, E, a**2/2+b):
        for cf in coeffs:
            lm = next(iter(cf.keys()))
            r = cf(lm)
            cases.append((l, r, Expr, True))
    return cases

@pytest.mark.parametrize("l,r,t,rev", _get_test_cases())
def test_partial_order_element_operators(l, r, t, rev):
    if issubclass(t, Expr):
        t = Expr
    assert isinstance(l + r, t)
    assert isinstance(l - r, t)
    assert isinstance(l * r, t)
    if r != 0 and ((not hasattr(r, 'is_zero')) or (not r.is_zero)):
        assert isinstance(l / r, t)

    if not rev:
        return
    assert isinstance(r + l, t)
    assert isinstance(r - l, t)
    assert isinstance(r * l, t)
    if l != 0 and ((not hasattr(l, 'is_zero')) or (not l.is_zero)):
        assert isinstance(r / l, t)
