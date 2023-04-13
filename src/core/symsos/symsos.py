import sympy as sp

from ..pqrsos.pqrsos import pqr_cyc
from ...utils.text_process import PreprocessText as pl

def symprove(poly, only_numer = False):
    """
    x = s(a(a-b)(a-c))
    y = s(a(b-c)2)
    z = s(a3)p(a-b)2/p(a)
    """
    a,b,c,x,y,z,w,p,q,r = sp.symbols('a b c x y z w p q r')
    if isinstance(poly, str):
        poly = pl(poly)
    poly = pqr_cyc(poly.as_poly(a,b,c))
    if poly[1] != 0:
        return None

    poly = sp.together(poly[0].subs({
        q: y * (4*x**2+5*x*y+y**2+z) / w / p,
        r: x * y**2 / w
    }))

    fracs = [sp.S(1), sp.S(1)]
    for i in range(2 - only_numer):
        frac = sp.fraction(poly)[i]
        if i == 0:
            frac = frac.subs(p, sp.cancel((z*(x+4*y)+4*(x+y)**3) / w)**(sp.S(1)/3))
            frac = frac.subs(w, y**2 - 4*x*y + 4*x**2 + z)
            frac = frac.as_poly(z)
        elif i == 1:
            # denominator is in the form of p**r * w**s
            # w = (abc(s(a(b-c)2)-2s(a(a-b)(a-c)))2+s(a3)p(a-b)2) / (abc)
            # w = s(a(b-c)2)2s(a(a-b)(a-c)) / abc
            deg_p = frac.as_poly(p).degree()
            deg_w = frac.as_poly(w).degree()
            frac = (y**2 - 4*x*y + 4*x**2 + z) ** deg_w
        fracs[i] = frac
    
    if only_numer:
        return fracs[0]

    fracs = list(sp.fraction(sp.cancel((fracs[0] / fracs[1]))))

    for i in range(2 - only_numer):
        frac = fracs[i].as_poly(z)
        fracs[i] = sum(coeff.factor() * z**deg for coeff, (deg, ) in zip(frac.coeffs(), frac.monoms()))
    return fracs[0] / fracs[1] / p**deg_p


def _symprove_verify(poly):
    a,b,c,x,y,z = sp.symbols('a b c x y z')

    org = None
    if isinstance(poly, str):
        org = pl(poly).as_poly(a,b,c)
        poly = symprove(org, only_numer = False) 

    fracs = []
    poly = sp.fraction(poly)

    from tqdm import tqdm
    for i in range(2):
        frac = poly[i].as_poly(z)
        g = pl('s(a)3p(a-b)2')
        n = frac.degree()
        frac = sum(coeff * g**deg * (a*b*c) ** (n-deg) for coeff, (deg, ) in zip(frac.coeffs(), frac.monoms()))
        
        px, py, pp = pl('s(a(a-b)(a-c))'), pl('s(a(b-c)2)'), pl('s(a)')
        
        # from functools import lru_cache
        # @lru_cache
        # def pow(s: int, n: int):
        #     if s == 0:
        #         return px ** n
        #     elif s == 1:
        #         return py ** n
        #     return pp ** n

        px, py, pp = px.as_expr(), py.as_expr(), pp.as_expr()
        def fastsub(subpoly):
            return subpoly.subs({
                x: px,
                y: py,
                p: pp
            }).as_poly(a,b,c)

            # # slow version:
            # subpoly = subpoly.as_poly(x, y, p)
            # return sum(coeff * pow(0, d0) * pow(1, d1) * pow(2, d2) 
            #             for coeff, (d0, d1, d2) in zip(subpoly.coeffs(), subpoly.monoms()))

        frac = sum(fastsub(coeff) * a**d0 * b**d1 * c**d2 
                        for coeff, (d0, d1, d2) in tqdm(zip(frac.coeffs(), frac.monoms()), total = len(frac.coeffs())))
        frac = frac.as_poly(a,b,c)
        fracs.append(frac)

    if org is not None:
        return fracs[0] - fracs[1] * org
    return fracs[0] / fracs[1]
