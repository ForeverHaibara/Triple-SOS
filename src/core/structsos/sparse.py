from math import gcd

import sympy as sp

from .utils import CyclicSum, CyclicProduct
from .quartic import sos_struct_quartic
from ...utils.polytools import deg


def sos_struct_sparse(poly, coeff, recurrsion):
    if len(coeff) > 6:
        return None

    degree = deg(poly)
    if degree < 5:
        if degree == 0:
            return sp.S(0)
        elif degree == 1:
            return poly.as_expr()
        elif degree == 2:
            return sos_struct_quadratic(coeff)
        elif degree == 4:
            # quartic should be handled by _sos_struct_quartic
            # because it presents proof for real numbers
            return sos_struct_quartic(poly, coeff, recurrsion)

    monoms = list(coeff.coeffs.keys())
    a, b, c = sp.symbols('a b c')
    if len(coeff) == 1:
        # e.g.  abc
        if coeff(monoms[0]) >= 0:
            # we must have i == j == k as the polynomial is cyclic
            i, j, k = monoms[0]
            return coeff(monoms[0]) * CyclicProduct(a**i)

    elif len(coeff) == 3:
        # e.g. (a2b + b2c + c2a)
        if coeff(monoms[0]) >= 0:
            i, j, k = monoms[0]
            return coeff(monoms[0]) * CyclicSum(a**i * b**j * c**k)

    elif len(coeff) == 4:
        # e.g. (a2b + b2c + c2a - 8/3abc)
        n = degree // 3
        if coeff(monoms[0]) >= 0 and coeff(monoms[0])*3 + coeff(n, n, n) >= 0:
            i, j, k = monoms[0]
            return coeff(monoms[0]) * CyclicSum(a**i * b**j * c**k - CyclicProduct(a**n))\
                + (coeff(monoms[0]) * 3 + coeff(n, n, n)) * CyclicProduct(a**n)

    elif len(coeff) == 6:
        # e.g. s(a5b4 - a4b4c)
        monoms = [i for i in monoms if (i[0]>i[1] and i[0]>i[2]) or (i[0]==i[1] and i[0]>i[2])]
        monoms = sorted(monoms)
        small , large = monoms[0], monoms[1]
        if coeff(small) >= 0 and coeff(large) >= 0:
            return coeff(small) * CyclicSum(a**small[0] * b**small[1] * c**small[2]) \
                + coeff(large) * CyclicSum(a**large[0] * b**large[1] * c**large[2])

        elif coeff(large) >= 0 and coeff(large) + coeff(small) >= 0:
            # AM-GM inequality
            det = 3*large[0]*large[1]*large[2] - (large[0]**3+large[1]**3+large[2]**3)
            deta = small[0]*(large[1]*large[2]-large[0]**2)+small[1]*(large[2]*large[0]-large[1]**2)+small[2]*(large[0]*large[1]-large[2]**2)
            detb = small[0]*(large[2]*large[0]-large[1]**2)+small[1]*(large[0]*large[1]-large[2]**2)+small[2]*(large[1]*large[2]-large[0]**2)
            detc = small[0]*(large[0]*large[1]-large[2]**2)+small[1]*(large[1]*large[2]-large[0]**2)+small[2]*(large[2]*large[0]-large[1]**2)
            det, deta, detb, detc = -det, -deta, -detb, -detc

            if det > 0 and deta >= 0 and detb >= 0 and detc >= 0:
                d = gcd(det, gcd(deta, gcd(detb, detc)))
                det, deta, detb, detc = det//d, deta//d, detb//d, detc//d
                
                am_gm = deta*a**large[0]*b**large[1]*c**large[2] + detb*a**large[1]*b**large[2]*c**large[0] + detc*a**large[2]*b**large[0]*c**large[1] - det*a**small[0]*b**small[1]*c**small[2]
                
                return coeff(large)/det * CyclicSum(am_gm) + (coeff(large) + coeff(small)) * CyclicSum(a**small[0] * b**small[1] * c**small[2])

    return None


def sos_struct_quadratic(coeff):
    """
    Solve quadratic problems. It must be in the form $\sum (a^2 + xab)$ where x >= -1.
    However, we shall also handle cases for real numbers.
    """

    y, x = coeff((2,0,0)), coeff((1,1,0))
    if x + y < 0 or y < 0:
        return None

    a, b, c = sp.symbols('a b c')
    if y == 0:
        return CyclicSum(a*b) * x

    if x > 2 * y:
        return CyclicSum(y * a**2 + x * a*b)

    # real numbers
    # should be a linear combination of s(a2-ab) and s(a)2
    # w1 + w2 = y
    # -w1 + 2w2 = x
    w1 = (2*y - x) / 3
    w2 = y - w1
    return w1 / 2 * CyclicSum((a-b)**2) + w2 * CyclicSum(a)**2