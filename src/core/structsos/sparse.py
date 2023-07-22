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
            if i % 3 or j % 3 or k % 3:
                p1 = coeff(monoms[0]) * CyclicSum(a**i * b**j * c**k - CyclicProduct(a**n))
            else:
                # special case is i%3==j%3==k%3 == 0, and then we can factor the AM-GM
                ker1 = a**(i//3)*b**(j//3)*c**(k//3)
                ker2 = a**(j//3)*b**(k//3)*c**(i//3)
                p1 = coeff(monoms[0])/2 * CyclicSum(ker1) * CyclicSum((ker1 - ker2)**2)

            return p1 + (coeff(monoms[0]) * 3 + coeff(n, n, n)) * CyclicProduct(a**n)

    elif len(coeff) == 6:
        # AM-GM
        # e.g. s(a5b4 - a4b4c)
        monoms = [i for i in monoms if (i[0]>i[1] and i[0]>i[2]) or (i[0]==i[1] and i[0]>i[2])]
        monoms = sorted(monoms)
        small, large = monoms[0], monoms[1]

        return _sos_struct_sparse_amgm(coeff, small, large)
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


def _sos_struct_sparse_amgm(coeff, small, large):
    """
    Solve 
    \sum coeff(large) * a^u*b^v*c^w + \sum coeff(small) * a^x*b^y*c^z >= 0
    where triangle Cyclic(x,y,z) is contained in the triangle Cyclic(u,v,w).
    Also, |x-y|+|y-z|+|z-x| > 0.

    In general this is only an AM-GM inequality.

    However, we shall handle special cases more carerfully. Because AM-GM
    is sometimes not so beautiful as sum of squares.
    For example,
    s(a6-a4bc) = s(a2)s((a2-b2)2)/4+s(a2(a2-bc)2)/2 >= 0 for all real numbers a,b,c.
    """
    if coeff(large) < 0 or coeff(large) + coeff(small) < 0:
        return None
    a, b, c = sp.symbols('a b c')
    
    def _mean(a, b):
        return ((a[0]+b[0])//2, (a[1]+b[1])//2, (a[2]+b[2])//2)
    def _multiple(a, b):
        return (2*b[0]-a[0], 2*b[1]-a[1], 2*b[2]-a[2])

    if large[0] % 2 == large[1] % 2 and large[1] % 2 == large[2] % 2:
        # detect special case: small is the midpoint of large
        # in this case, we can provide sum-of-square for real numbers
        u,v,w = large
        if small == _mean(large, (v,w,u))\
            or small == _mean(large, (w,u,v))\
            or small == _mean((v,w,u),(w,u,v)):
            # midpoint
            x, y = coeff(small), coeff(large)
            if 2*y >= x:
                prefix = CyclicProduct(a) if u % 2 == 1 else sp.S(1)
                p1 = a**(u//2) * b**(v//2) * c**(w//2)
                p2 = a**(v//2) * b**(w//2) * c**(u//2)
                w1 = (2*y - x) / 3
                w2 = y - w1
                return (w1/2) * prefix * CyclicSum((p1 - p2)**2) + w2 * prefix * CyclicSum(p1)**2
                
                # if x + y >= 0 but 2*y < x (indicating x, y >= 0 both)
                # fall to the normal mode below                    

    # we can handle case for real numbers easily without highering the degree
    if large == (6,0,0):
        if small == (4,1,1):
            # s(a6-a4bc) = s(a2)s((a2-b2)2)/4+s(a2(a2-bc)2)/2
            if coeff(small) <= 0:
                # it is a linear combination of s(a6-a4bc) and s(a6)
                w1 = -coeff(small)
                w2 = coeff(large) + coeff(small)
                return w1 / 4 * CyclicSum(a**2)*CyclicSum((a**2-b**2)**2) + w1 / 2 * CyclicSum(a**2*(a**2 - b*c)**2)\
                    + w2 * CyclicSum(a**6)
        elif small == (4,2,0):
            # s(a6-a4b2) = 2/3s(a2(a2-b2)2)+1/3s(a2(a2-c2)2)
            return coeff(large)*2/3 * CyclicSum(a**2*(a**2-b**2)**2) + coeff(large)/3 * CyclicSum(a**2*(a**2-c**2)**2)\
                    + (coeff(large) + coeff(small)) * CyclicSum(a**4*b**2)
        elif small == (4,0,2):
            return coeff(large)*2/3 * CyclicSum(a**2*(a**2-c**2)**2) + coeff(large)/3 * CyclicSum(a**2*(a**2-b**2)**2)\
                    + (coeff(large) + coeff(small)) * CyclicSum(a**4*c**2)

        # s(a6-a5b)-1/2s(a2(a2-b2)(a2-c2))-1/2s(a4(a-b)2) = s(a4c2-a2b2c2)/2
        # however s(a4c2-a2b2c2) >= 0 needs higher degree


    if True:
        # extend each side of the small triangle by two times to obtain the large
        # e.g. s(a3b-a2bc) = s(ac(b-c)2)
        # e.g. s(a5c-a3bc2) = s(ac(a2-bc)2)
        u,v,w = large
        twice = _multiple(small, (small[1], small[2], small[0]))
        if twice in ((u,v,w),(v,w,u),(w,u,v)):
            prefix = a**(twice[0]%2)*b**(twice[1]%2)*c**(twice[2]%2)
            p1 = a**(twice[0]//2)*b**(twice[1]//2)*c**(twice[2]//2)
            p2 = a**(small[0]//2)*b**(small[1]//2)*c**(small[2]//2)
            return coeff(large) * CyclicSum(prefix * (p1 - p2)**2)\
                + (coeff(large) + coeff(small)) * CyclicSum(a**small[0]*b**small[1]*c**small[2])
        twice = _multiple((small[1], small[2], small[0]), small)
        if twice in ((u,v,w),(v,w,u),(w,u,v)):
            prefix = a**(twice[0]%2)*b**(twice[1]%2)*c**(twice[2]%2)
            p1 = a**(twice[0]//2)*b**(twice[1]//2)*c**(twice[2]//2)
            p2 = a**(small[1]//2)*b**(small[2]//2)*c**(small[0]//2)
            return coeff(large) * CyclicSum(prefix * (p1 - p2)**2)\
                + (coeff(large) + coeff(small)) * CyclicSum(a**small[0]*b**small[1]*c**small[2])
        
        

    # now we use general method
    if coeff(small) >= 0:
        return coeff(small) * CyclicSum(a**small[0] * b**small[1] * c**small[2]) \
            + coeff(large) * CyclicSum(a**large[0] * b**large[1] * c**large[2])

    else:
        # AM-GM inequality
        x, y, z = small
        u, v, w = large
        det = 3*u*v*w - (u**3+v**3+w**3)
        deta = x*(v*w-u**2)+y*(w*u-v**2)+z*(u*v-w**2)
        detb = x*(w*u-v**2)+y*(u*v-w**2)+z*(v*w-u**2)
        detc = x*(u*v-w**2)+y*(v*w-u**2)+z*(w*u-v**2)
        det, deta, detb, detc = -det, -deta, -detb, -detc

        if det > 0 and deta >= 0 and detb >= 0 and detc >= 0:
            d = gcd(det, gcd(deta, gcd(detb, detc)))
            det, deta, detb, detc = det//d, deta//d, detb//d, detc//d
            
            am_gm = deta*a**u*b**v*c**w + detb*a**v*b**w*c**u + detc*a**w*b**u*c**v - det*a**x*b**y*c**z
            
            return coeff(large)/det * CyclicSum(am_gm) + (coeff(large) + coeff(small)) * CyclicSum(a**x * b**y * c**z)

    return None
