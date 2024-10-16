from typing import List, Tuple, Dict, Callable, Optional
from math import gcd

import sympy as sp

from .utils import (
    CyclicSum, CyclicProduct, Coeff, SS, prove_univariate, radsimp
)
from .dense_symmetric import sos_struct_dense_symmetric
from .quadratic import sos_struct_quadratic
from .quartic import sos_struct_quartic

def sos_struct_sparse(coeff, real = True):
    """
    Solver to very sparse inequalities like AM-GM.

    The function does not use `recursion` for minimal dependency.

    This method does not present solution for a,b,c in R in prior, but in R+.
    Inequalities of degree 2 and 4 are skipped because they might be
    handled in R.
    """
    a, b, c = sp.symbols('a b c')
    if len(coeff) > 6:
        return None

    degree = coeff.degree()
    if degree < 5:
        if degree == 0:
            v = coeff((0,0,0))
            return v if v >= 0 else None
        elif degree == 1:
            v = coeff((1,0,0))
            return v * CyclicSum(a) if v >= 0 else None
        elif degree == 2:
            return sos_struct_quadratic(coeff)
        elif degree == 4:
            # quartic should be handled by _sos_struct_quartic
            # because it presents proof for real numbers
            return sos_struct_quartic(coeff, None)

    monoms = list(coeff.coeffs.keys())
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


    SPECIAL_AMGMS = {
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
    _amgm = SPECIAL_AMGMS.get((large, small))
    if _amgm is not None:
        return coeff(large) * _amgm + (coeff(large) + coeff(small)) * CyclicSum(a**small[0]*b**small[1]*c**small[2])

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



#####################################################################
#
#                          Heuristic method
#
#####################################################################

def _acc_dict(items: List[Tuple]) -> Dict:
    """
    Accumulate the coefficients in a dictionary.
    """
    d = {}
    for k, v in items:
        if k in d:
            d[k] += v
        else:
            d[k] = v
    d = {k: v for k,v in d.items() if v != 0}
    return d

def _separate_product_wrapper(recursion: Callable) -> Callable:
    """
    A wrapper of recursion function to avoid nested CyclicProduct(a).
    For instance, if we have CyclicProduct(a) * (CyclicProduct(a)*F(a,b,c) + G(a,b,c)),
    we had better expand it to CyclicProduct(a**2) * F(a,b,c) + CyclicProduct(a) * G(a,b,c).
    """
    a = sp.symbols('a')
    def _extract_cyclic_prod(x: sp.Expr) -> Tuple[int, sp.Expr]:
        """
        Given x, return (d, r) such that x = r * CyclicProduct(a**d).
        """
        if not (isinstance(x, sp.Expr) or x.is_Mul or x.is_Pow or isinstance(x, CyclicProduct)):
            return 0, x
        if isinstance(x, CyclicProduct):
            if x.args[0] == a:
                return 1, sp.S(1)
            elif x.args[0].is_Pow and x.args[0].base == a:
                return x.args[0].exp, sp.S(1)
        elif x.is_Pow:
            d, r = _extract_cyclic_prod(x.base)
            return d * x.exp, r**x.exp
        elif x.is_Mul:
            rs = []
            d = 0
            for arg in x.args:
                dd, r = _extract_cyclic_prod(arg)
                d += dd
                rs.append(r)
            return d, sp.Mul(*rs)
        return 0, x

    def _new_recursion(x: Coeff, **kwargs) -> Optional[sp.Expr]:
        x = recursion(x, **kwargs)
        if x is None:
            return x
        d, r = _extract_cyclic_prod(x)
        if r.is_Add:
            new_args = []
            for arg in r.args:
                d2, r2 = _extract_cyclic_prod(arg)
                new_args.append(CyclicProduct(a**(d + d2)) * r2)
            return sp.Add(*new_args)
        return x

    return _new_recursion

class Pnrms:
    """
    Represent s(a^n(b^r-c^r)) * s(a^m(b^s-c^s)).
    """
    @classmethod
    def coeff(cls, n, r, m, s, v = 1) -> Coeff:
        v = sp.S(v)
        return Coeff(_acc_dict([
            ((r + s, m + n, 0), v), ((r + s, 0, m + n), v), ((m + n, r + s, 0), v), ((m + n, 0, r + s), v),
            ((0, r + s, m + n), v), ((0, m + n, r + s), v), ((m + r, n + s, 0), -v), ((m + r, 0, n + s), -v),
            ((n + s, m + r, 0), -v), ((n + s, 0, m + r), -v), ((0, m + r, n + s), -v), ((0, n + s, m + r), -v),
            ((r, m, n + s), v), ((r, n + s, m), v), ((s, n, m + r), v), ((s, m + r, n), v),
            ((m, r, n + s), v), ((m, n + s, r), v), ((n, s, m + r), v), ((n, m + r, s), v),
            ((m + r, s, n), v), ((m + r, n, s), v), ((n + s, r, m), v), ((n + s, m, r), v),
            ((r, s, m + n), -v), ((r, m + n, s), -v), ((s, r, m + n), -v), ((s, m + n, r), -v),
            ((m, n, r + s), -v), ((m, r + s, n), -v), ((n, m, r + s), -v), ((n, r + s, m), -v),
            ((r + s, m, n), -v), ((r + s, n, m), -v), ((m + n, r, s), -v), ((m + n, s, r), -v),
        ]), is_rational = True if isinstance(v, sp.Rational) else False)

    @classmethod
    def as_expr(cls, n, r, m, s, v = 1) -> sp.Expr:
        a, b, c = sp.symbols('a b c')
        if n == r or m == s or r == 0 or s == 0:
            return sp.S(0)
        sol = None
        if n == m and r == s:
            sol = CyclicSum(a**n * (b**r - c**r))**2
        f1, f2 = cls._half_side(n, r), cls._half_side(m, s)
        if f1 is not None and f2 is not None:
            sol = CyclicProduct((a-b)**2) * f1 * f2
        else:
            sol = CyclicSum(a**n * (b**r - c**r)) * CyclicSum(a**m * (b**s - c**s))
        return v * sol

    @classmethod
    def _half_side(cls, n, r) -> sp.Expr:
        """
        Return s(a^n(b^r-c^r)) / [(b-a)(c-b)(a-c)]
        """
        a, b, c = sp.symbols('a b c')
        if n + r <= 12:
            RESULT = {
                (2, 1): sp.S(1),
                (3, 1): CyclicSum(a),
                (4, 1): CyclicSum(a**2+b*c),
                (5, 1): CyclicProduct(a) + CyclicSum(a) * CyclicSum(a**2),
                (6, 1): CyclicSum(a**2*(a+b)*(a+c)) + CyclicSum(a**2*b**2),
                (7, 1): CyclicSum(a**3*(a+b)*(a+c)) + CyclicSum(a) * CyclicSum(a**2*b**2),
                (8, 1): CyclicSum(a**2*(2*a**2+b**2+c**2)*(a+b)*(a+c))/2 + CyclicProduct(a**2+b**2)/2, # s(a2(2a2+b2+c2)(a+b)(a+c))/2+1/2(b2+c2)(a2+b2)(a2+c2)
                (9, 1): CyclicSum(a**3*(2*a**2+b**2+c**2)*(a+b)*(a+c))/2 + CyclicSum(a) * CyclicProduct(a**2+b**2)/2, # s(a3(2a2+b2+c2)(a+b)(a+c))/2+1/2s(a)(b2+c2)(a2+b2)(a2+c2)
                (3, 2): CyclicSum(a*b),
                (4, 2): CyclicProduct(a+b),
                (5, 2): CyclicSum(a*b) * CyclicSum(a**2) + CyclicSum(a**2*(b+c)**2) / 2,
                (6, 2): CyclicProduct(a+b) * CyclicSum(a**2),
                (7, 2): CyclicProduct(a+b) * (CyclicProduct(a) + CyclicSum(a**3)) + CyclicSum(a**3*b**3),
                (8, 2): CyclicProduct(a+b) * CyclicSum(a**4 + b**2*c**2),
                (4, 3): CyclicSum(a**2*(b+c)**2) / 2,
                (5, 3): CyclicSum(a) * CyclicSum(a**2*b**2) + CyclicSum((a+b)**2) * CyclicProduct(a) / 2,
                (6, 3): CyclicProduct(a**2+a*b+b**2),
                (7, 3): CyclicSum(a**3*(b**2+b*c+c**2)*(a+b)*(a+c)) + 2 * CyclicSum(a) * CyclicProduct(a**2),
                (8, 3): CyclicSum(a**4*(b**2+c**2)*(a+b)*(a+c)) + CyclicSum(a**2*b*(a+c)) * CyclicSum(a**2*c*(a+b)),
                (5, 4): CyclicSum(a**2*b**2) * CyclicSum(a*b) + CyclicProduct(a**2),
                (6, 4): CyclicProduct(a+b) * CyclicSum(a**2*b**2),
                (7, 4): 2*CyclicProduct(a**2)*CyclicSum(a**2) + CyclicSum(a*b)*(CyclicSum(a**3*b**3) + CyclicSum(a**2)*CyclicSum(a**2*b**2)), # 2p(a2)s(a2)+s(ab)(s(a3b3)+s(a2)s(a2b2)),
                (8, 4): CyclicProduct(a+b) * CyclicProduct(a**2+b**2),
                (6, 5): CyclicSum(b**3*c**3*(a+b)*(a+c)) + CyclicSum(a**2) * CyclicProduct(a**2)
            }
            return RESULT.get((n, r), None)


class Hnmr:
    """
    Represent s(a^n*(b^m+c^m)*(a^r-b^r)*(a^r-c^r)) when n >= 0,
    and s(b^(-n)*c^(-n)*(b^m+c^m))*(a^r-b^r)*(a^r-c^r)) when n < 0.

    We require m >= 0 and r >= 0 always.

    We have recursion identity:
    H(n,m,r) = P(2r,r,n,m-r) + (a^r*b^r*c^r) * H(n-r,m-2r,r)

    Also, we have inverse substituion (n,m,r) -> (m-n-r,m,r).
    When n <= 0 or n + r >= m, we have H(n,m,r) >= 0.
    """
    @classmethod
    def coeff(cls, n, m, r, v = 1) -> Coeff:
        v = sp.S(v)
        if n >= 0:
            coeffs = [
                ((m, n + 2*r, 0), v), ((m, 0, n + 2*r), v), ((n + 2*r, m, 0), v), ((n + 2*r, 0, m), v),
                ((0, m, n + 2*r), v), ((0, n + 2*r, m), v), ((m + r, n + r, 0), -v), ((m + r, 0, n + r), -v),
                ((n + r, m + r, 0), -v), ((n + r, 0, m + r), -v), ((0, m + r, n + r), -v), ((0, n + r, m + r), -v),
                ((r, n, m + r), v), ((r, m + r, n), v), ((n, r, m + r), v), ((n, m + r, r), v),
                ((m + r, r, n), v), ((m + r, n, r), v), ((r, m, n + r), -v), ((r, n + r, m), -v),
                ((m, r, n + r), -v), ((m, n + r, r), -v), ((n + r, r, m), -v), ((n + r, m, r), -v)
            ]
        else:
            n = -n
            coeffs = [
                ((n + r, m + n + r, 0), v), ((n + r, 0, m + n + r), v), ((m + n + r, n + r, 0), v), ((m + n + r, 0, n + r), v),
                ((0, n + r, m + n + r), v), ((0, m + n + r, n + r), v), ((n, 2*r, m + n), v), ((n, m + n, 2*r), v),
                ((2*r, n, m + n), v), ((2*r, m + n, n), v), ((m + n, n, 2*r), v), ((m + n, 2*r, n), v),
                ((r, n, m + n + r), -v), ((r, n + r, m + n), -v), ((r, m + n, n + r), -v), ((r, m + n + r, n), -v),
                ((n, r, m + n + r), -v), ((n, m + n + r, r), -v), ((n + r, r, m + n), -v), ((n + r, m + n, r), -v),
                ((m + n, r, n + r), -v), ((m + n, n + r, r), -v), ((m + n + r, r, n), -v), ((m + n + r, n, r), -v)
            ]
        return Coeff(_acc_dict(coeffs), is_rational = True if isinstance(v, sp.Rational) else False)

    @classmethod
    def as_expr(cls, n, m, r, v = 1) -> sp.Expr:
        a, b, c = sp.symbols('a b c')
        if r == 0:
            return sp.S(0)
        sol = None
        if n >= 0:
            if n == m:
                sol = CyclicSum(a**n*b**n*(a**r-b**r)**2)
            elif n == m - r:
                if n >= r:
                    sol = CyclicSum(a**(n-r)*b**(n-r)*(a**r-b**r)**2) * CyclicProduct(a**r)
                else: # if n < r:
                    sol = CyclicSum(c**(r-n)*(a**r-b**r)**2) * CyclicProduct(a**n)
            elif m == 2*r and n >= r:
                sol = Pnrms.as_expr(2*r, r, n, m - r) + CyclicProduct(a**r) * cls.as_expr(n - r, m - 2*r, r)
            elif m == r:
                if n % r == 0 and (n // r) <= 8:
                    RESULT = {
                        2: CyclicSum(a*(b-c)**2*(b+c-a)**2),
                        4: CyclicSum(a*(b-c)**2*(b+c-a)**4) + 2 * CyclicSum(a) * CyclicProduct((a-b)**2),
                        5: CyclicSum(a*b*(a-b)**2*(a**2+b**2-c**2)**2) + CyclicProduct((a-b)**2) * CyclicSum((a-b)**2) / 2,
                        6: CyclicSum(a*(b-c)**2*(b+c-a)**2*(b**2+c**2-a**2)**2) + CyclicProduct((a-b)**2) * (3*CyclicSum(a*(b-c)**2) + 22*CyclicProduct(a)),
                        7: CyclicSum(a*b*(a-b)**2*(a**3 + 2*a*b*c - 2*a*c**2 + b**3 - 2*b*c**2 + c**3)**2) + CyclicProduct((a-b)**2) * (CyclicSum(a*b)*CyclicSum(a**2) + CyclicSum((a**2-b**2)**2)/2),
                        8: CyclicSum(a*(b-c)**2*(b+c-a)**2*(b**3+c**3-a**3)**2) + CyclicProduct((a-b)**2) * (CyclicSum(a*b*(a+b))*CyclicSum(a)**2 + CyclicSum((a-b)**2) * CyclicProduct(a))
                    }
                    sol = RESULT.get(n // r, None)
                    if sol is not None:
                        sol = sol.xreplace({a:a**r, b:b**r, c:c**r})
                elif n == 1 and r >= 2:
                    sol = Pnrms.as_expr(r+1, 1, r, r-1) + CyclicProduct(a) * CyclicSum(c**(r-2)*(a**r-b**r)**2)
                else:
                    d = sp.gcd(n, r)
                    RESULT = {
                        (1, 2): CyclicSum(a) * CyclicProduct((a-b)**2) + CyclicProduct(a) * CyclicSum((a-b)**2),
                        (3, 2): CyclicProduct((a-b)**2) * (CyclicSum(a)*CyclicSum(a**2) + CyclicProduct(a+b)) + CyclicProduct(a) * CyclicSum((a**2-b**2)**2*(a+b-c)**2), # (s(a)s(a2)+p(a+b))p(a-b)2+p(a)s((a2-b2)2(a+b-c)2)
                        (5, 2): CyclicProduct((a-b)**2) * (CyclicSum(a)**2*CyclicSum(a**3) + CyclicSum(a*b)*CyclicProduct(a+b)) + CyclicProduct(a) * CyclicSum((a**2-b**2)**2*(a**2+b**2-c**2)**2), # p(a-b)2(s(a3)s(a)2+p(a+b)s(ab))+s((a2-b2)2(a2+b2-c2)2)p(a)
                    }
                    sol = RESULT.get((n // d, r // d), None)
                    if sol is not None:
                        sol = sol.xreplace({a:a**d, b:b**d, c:c**d})
                if sol is None:
                    if n >= m + r:
                        numerator = CyclicSum(a**n*(b**m+c**m)*(a**r-b**r)**2*(a**r-c**r)**2) + Pnrms.as_expr(2*r, r, n, m + r)
                        denominator = CyclicSum((a**r - b**r)**2) / 2
                        sol = numerator / denominator
                    else:
                        if n >= r:
                            numerator = CyclicSum(a**(n-r)*(b**m+c**m)*(a**r-b**r)**2*(a**r-c**r)**2) * CyclicProduct(a**r)
                        else:
                            numerator = CyclicSum(b**(r-n)*c**(r-n)*(b**m+c**m)*(a**r-b**r)**2*(a**r-c**r)**2) * CyclicProduct(a**n)
                        numerator += Pnrms.as_expr(2*r, r, n + m + r, m + r)
                        denominator = CyclicSum(a**(2*r)*(b**r - c**r)**2) / 2
                        sol = numerator / denominator
            if sol is None:
                sol = CyclicSum(a**n*(b**m+c**m)*(a**r-b**r)*(a**r-c**r))
        else:
            n = -n
            if n == r:
                sol = CyclicSum(c**(m+2*r)*(a**r-b**r)**2)
            if sol is None:
                sol = CyclicSum(b**n*c**n*(b**m+c**m)*(a**r-b**r)*(a**r-c**r))
        return v * sol



def sos_struct_heuristic(coeff, real=True):
    """
    Solve high-degree but sparse inequalities by heuristic method.
    It subtracts some structures from the inequality and calls
    the recursion function to solve the problem.

    WARNING: Only call this function when degree > 6. And make sure that
    coeff.clear_zero() to remove zero terms on the border.
    
    Examples
    -------
    s(ab(a-b)2(a4-3a3b+2a2b2+3b4))

    s(c8(a-b)2(a4-3a3b+2a2b2+3b4))
    """
    degree = coeff.degree()
    # assert degree > 6, "Degree must be greater than 6 in heuristic method."
    if degree <= 6:
        return None
    if True:
        solution = sos_struct_dense_symmetric(coeff, real = real)
        if solution is not None:
            return solution
    recursion = SS.structsos.ternary._structural_sos_3vars_cyclic
    recursion = _separate_product_wrapper(recursion)

    if coeff((degree, 0, 0)):
        # not implemented
        return None

    monoms = list(coeff.coeffs.items())
    if monoms[0][1] < 0 or monoms[-1][1] < 0 or monoms[-1][0][0] != 0:
        return None

    border1, border2 = [], []
    for (i,j,k), v in monoms[::-1]:
        if i != 0:
            break
        if v != 0:
            border1.append((j, v))
    border1 = border1[::-1]
    i0 = monoms[0][0][0]
    for (i,j,k), v in monoms:
        if i != i0:
            break
        if v != 0:
            border2.append((j, v))

    if len(border1) == 1 and border1[0][0] * 2 == degree:
        return None

    # print('Coeff =' , coeff.as_poly())
    # print('Border1 =', border1)
    # print('Border2 =', border2)

    for border in (border1, border2):
        if len(border) * 3 == len(coeff):
            # all coefficients are on the border
            x = sp.Symbol('x')
            a, b, c = sp.symbols('a b c')
            border_poly = sp.Poly.from_dict(dict(border), gens = (x,))
            border_proof = prove_univariate(border_poly)
            if border_proof is not None:
                if border is border1:
                    border_proof = border_proof.subs(x, a / b).together() * b**degree
                else:
                    border_proof = border_proof.subs(x, b / c).together() * a**i0 * c**(degree - i0)
                border_proof = CyclicSum(border_proof)
            return border_proof

    c0 = border1[0][1]
    c0_ = border1[-1][1]
    if c0 < 0 or c0_ < 0:
        return None
    if border1[0][0] + border1[-1][0] == degree and c0 == c0_ and i0 == border1[0][0]:
        if len(border1) <= 4 and len(border2) <= 4:
            # symmetric hexagon
            gap11, gap12, gap21, gap22 = -1, -1, -1, -1
            if len(border1) >= 3 and border1[1][0] + border1[-2][0] == degree:
                gap11 = border1[0][0] - border1[1][0]
                gap12 = border1[0][0] - border1[-2][0]
            elif len(border1) <= 2:
                gap11 = 0
                gap12 = 0

            if len(border2) >= 3 and border2[1][0] + border2[-2][0] == degree - i0:
                gap21 = border2[0][0] - border2[1][0]
                gap22 = border2[0][0] - border2[-2][0]
            elif len(border2) <= 2:
                gap21 = 0
                gap22 = 0

            # print('Symmetric Hexagon Gap =', (gap11, gap12, gap21, gap22))
            if gap11 != -1 and gap21 != -1:
                if gap11 != 0 and gap21 != 0:
                    r_, s_ = gap21, gap22
                    for n_, m_ in ((r_ + gap11, s_ + gap12), (r_ + gap12, s_ + gap11)):
                        # print('>> s(a%d(b%d-c%d))s(a%d(b%d-c%d))' % (n_, r_, r_, m_, s_, s_))

                        solution = recursion(coeff - Pnrms.coeff(n_, r_, m_, s_, c0), real = False)
                        if solution is not None:
                            return solution + Pnrms.as_expr(n_, r_, m_, s_, c0)
                elif gap11 != 0 and gap21 == 0:
                    for r_ in (gap11, gap12):
                        n_ = border1[0][0] - 2*r_
                        m_ = degree - border1[0][0]
                        if n_ >= 0 and m_ >= 0 and m_ <= n_ + r_:
                            # print('>> s(a%d(b%d+c%d)(a%d-b%d)(a%d-c%d))' % (n_, m_, m_, r_, r_, r_, r_))

                            solution = recursion(coeff - Hnmr.coeff(n_, m_, r_, c0 if m_ else c0/2), real = False)
                            if solution is not None:
                                return solution + Hnmr.as_expr(n_, m_, r_, c0 if m_ else c0/2)

                        if m_ > r_ and n_ + r_ > m_:
                            # print('>> s(a%d(b%d-c%d))s(a%d(b%d-c%d))' % (2*r_, r_, r_, n_, m_-r_, m_-r_))

                            solution = recursion(coeff - Pnrms.coeff(2*r_, r_, n_, m_-r_, c0), real = False)
                            if solution is not None:
                                return solution + Pnrms.as_expr(2*r_, r_, n_, m_-r_, c0)

                elif gap11 == 0 and gap21 != 0:
                    for r_ in (gap21, gap22):
                        n_ = degree - border1[0][0] - r_
                        m_ = 2 * border1[0][0] - degree
                        if n_ >= 0 and m_ >= 0:
                            # print('>> s(b%dc%d(b%d+c%d)(a%d-b%d)(a%d-c%d))' % (n_, n_, m_, m_, r_, r_, r_, r_))

                            solution = recursion(coeff - Hnmr.coeff(-n_, m_, r_, c0 if m_ else c0/2), real = False)
                            if solution is not None:
                                return solution + Hnmr.as_expr(-n_, m_, r_, c0 if m_ else c0/2)

                        # if m_ >= r_:
                        #     print('>> s(a%d(b%d-c%d))s(a%d(b%d-c%d))' % (2*r_, r_, r_, n_, m_-r_, m_-r_))

                        #     solution = recursion(coeff - Pnrms.coeff(2*r_, r_, n_, m_-r_, c0), real = False)
                        #     if solution is not None:
                        #         return solution + Pnrms.coeff(2*r_, r_, n_, m_-r_, c0)


    return None
