"""
Experimemntal File. DO NOT USE.
"""

from itertools import product

import sympy as sp
import numpy as np

class FastPositiveChecker():
    def __init__(self):
        self.poly = None
        # self.points = [] # should use a numpy matrix to represent quadratic form
        # (a, ka, 1)   where k = slope, WLOG. a >= ka (and a >= 1), i.e. k <= 1
        self.slopes = (0, 1, sp.Rational(1,2), sp.Rational(1,3), sp.Rational(2,3), sp.Rational(1,5), sp.Rational(4,5))
        self.sloped_poly = []
        self.args = None
    
    def setPoly(self, poly):
        poly = poly.subs('c', 1)
        self.poly = poly
        self.sloped_poly = [poly.subs('b', slope) for slope in self.slopes]
    
    def check(self, args = None):
        self.args = args
        poly_subs = self.poly.subs(args) if args is not None else self.poly
        if poly_subs(sp.Rational(1,3),sp.Rational(1,2)) < 0:
            return 32

        is_zeroroot = poly_subs(0,0) == 0
        for i in range(len(self.slopes)):
            sloped_poly = self.sloped_poly[i].subs(args) if args is not None else self.sloped_poly[i]
            count_roots = sp.polys.polytools.count_roots(sloped_poly, 0)
            
            if count_roots > is_zeroroot + 1:
                return 32 - i
            elif count_roots == is_zeroroot + 1:
                is_oneroot = sloped_poly(1) == 0
                if (not is_oneroot) or sloped_poly(sp.Rational(1,2)) < 0 or sloped_poly(2) < 0:
                    return 32 - i

        v = self.strict_check(poly_subs)
        if not v:
            return 16
        return 0
    
    def strict_check(self, poly):
        da = poly.diff('a')
        db = poly.diff('b')
        da2 = da.diff('a')
        dab = da.diff('b')
        db2 = db.diff('b')
        for a , b in product(np.linspace(0.1,0.9,num=10),repeat=2):
            outside = False
            for iter in range(20): # by experiment, 20 is oftentimes more than enough
                # x =[a',b'] <- x - inv(nabla)^-1 @ grad
                lasta = a
                lastb = b
                da_  = da(a,b)
                db_  = db(a,b)
                da2_ = da2(a,b)
                dab_ = dab(a,b)
                db2_ = db2(a,b)
                det_ = da2_ * db2_ - dab_ * dab_
                if det_ <= -1e-6: # not locally convex
                    break
                elif det_ == 0: # not invertible
                    break
                else:
                    a , b = a - (db2_ * da_ - dab_ * db_) / det_ , b - (-dab_ * da_ + da2_ * db_) / det_
                    if a < 0 or b < 0:
                        outside = True
                    if abs(a - lasta) < 1e-9 and abs(b - lastb) < 1e-9:
                        # stop updating
                        break

            if not outside:
                if poly(a, b) < -1e-6:
                    return False
        return True

    def argnames(self):
        args = []
        for arg in self.poly.args[1:]:
            arg_str = str(arg)
            if arg_str != 'a' and arg_str != 'b':
                args.append(arg_str)
        return args


def search_positive(poly_str: str):
    poly = preprocess_text(poly_str)
    
    fpc = FastPositiveChecker()
    fpc.setPoly(poly)
    argnames = fpc.argnames()

    search_range = []
    if len(argnames) is None:
        return poly_str
    elif len(argnames) == 1:
        search_range = [(i,) for i in range(-6, 7)] + [(sp.Rational(i, 2),) for i in range(-9, 11, 2)]
    elif len(argnames) == 2:
        search_range = product(range(-6, 7), repeat = 2) # 169
    elif len(argnames) == 3:
        search_range = product(range(-2, 3), repeat = 3) # 125
    else:
        search_range = product(range(-1, 2), repeat = len(argnames)) # 3^n
    
    def _grid_search(poly_str, search_range):
        best_choice = None
        for args in search_range:
            v = fpc.check(dict(zip(argnames, args)))
            if best_choice is None or best_choice[0] > v:
                best_choice = (v, args)
            if v == 0:
                for argname, arg in zip(argnames, args):
                    poly_str = poly_str.replace(argname, f'({arg})')
                return poly_str , best_choice
        return None , best_choice
    
    result , best_choice = _grid_search(poly_str, search_range)
    if result is not None:
        return result
    if best_choice[0] <= 30: # promising
        if len(argnames) == 1:
            search_range = [(0,)] + [(sp.Rational(i, 4),) for i in range(-3, 5, 2)] + [(sp.Rational(i, 8),) for i in range(-7, 9, 2)]
        else:
            if len(argnames) == 2:
                search_range = [sp.Rational(i, 2) for i in (-1, 1)] + [sp.Rational(i, 3) for i in (-2, -1, 1, 2)] + [sp.Rational(i, 4) for i in range(-3, 5, 2)]
            elif len(argnames) == 3:
                search_range = [sp.Rational(i, 2) for i in (-1, 1)] + [sp.Rational(i, 3) for i in (-2, -1, 1, 2)]
            else:
                search_range = [sp.Rational(i, 2) for i in (-1, 1)]

            search_range = [0] + search_range
            search_range = list(product(search_range, repeat = len(argnames)))

        for i in range(len(search_range)):
            biased_args = tuple(a+b for a, b in zip(best_choice[1], search_range[i]))
            search_range[i] = biased_args
            
        result , best_choice = _grid_search(poly_str, search_range)

    return result







if __name__ == '__main__':
    # s = '3*m*(m+n-(-2*x*y))-(p-(-2*x+y^2))^2-(q-(x^2-2*y))^2-(p-(-2*x+y^2))*(q-(x^2-2*y))'
    # a,m,p,n,q = 1,1,-2,-5,-2#25,50,-261,146,1
    # m,p,n,q = [sp.Rational(i,a) for i in (m,p,n,q)]
    # s = s.replace('m',m.__str__()).replace('n',n.__str__()).replace('p',p.__str__()).replace('q',q.__str__())
    # s = '-(%s)'%s
    # poly = sp.polys.polytools.Poly(s)
    # print(s)

    # poly = sp.polys.polytools.Poly('115911-(x^3*y+y^3*(6-x-y)+(6-x-y)^3*x-64*x*y*(6-x-y))')
    # a , b = optimize_determinant(poly, soft=True)
    # print(a, b, poly(a,b))


    # txt = '3s(a2)3-(s(a3+2a2b))2-2s(a(a-b)(a-c))2-s(ac(2a2-2ab+(b2-ab)+x(ac-ab)+y(bc-ab))2)'
    # txt = 's(a2c(a-b)(a+c-4b))s(a2+3ab)-s(c(a3-ab2+x(ab2-abc)-3(a2b-abc)+3/2(a2c-abc)-(bc2-abc)-y(b2c-abc))2)'
    txt = 's(a5c2+a4b3-7a4bc2+2a4c3+2a3b3c+a3b2c2)-s(a(a2c-abc-(ac2-abc)+x(b2c-abc)-y(ab2-abc))2) '
    txt ='(6s(a)s(c3b)-abc(37s(a2)-19s(ab)))s(a2+16ab)-6s(c(a(a2-bc+4(ac-bc)-x(ab-bc)+(y)(b2-bc))-3(bc2-abc))2)'
    txt = '9(27s(a2)3-(s(4a3+5a2b))2)-99s((a3-2a2b-a2c+2abc)2)-9s(ab(2(a2-ab)+(b2-ab)+(ac-ab)+x(bc-ab))2)-189s(ac(a2-ab-(ac-ab)+1/8(b2-ab)+1(bc-ab))2)-154s(a2b-abc-2(ab2-abc))2-110(s(a2b-ab2))2'
    txt = 's(a2(a-b)(a2-3bc))-s(a(a-b)(a-c))s(a2-ab)-s(c(z(a2-ab)-(b2-ab)+x(ac-ab)+y(bc-ab))2)'
    txt = '(s(a12c5+a11b2c4+a10b6c-9a10b3c4+2a10b2c5+a10bc6+2a10c7+a9b8+2a9b4c4-7a9b2c6+2a8b8c+a8b7c2+2a8b6c3-8a8b4c5+2a8b3c6+2a8b2c7+a7b6c4+2a7b4c6+a6b6c5)-s(a5b8(a2+b2-2c2)2)-s(a4b7c2(a-b)4)-s(a3b7cs(ab(ab-ac)(ab-bc)))-s(a4b4(ab-ac)2c(a-c)4)-2s(a4b8c(a2-bc+(bc-ac)+2(c2-ac))2))/p(a2)-6s(ab6(a2-ac-1/2(ab-ac)+x(bc-ac)-y(c2-ac))2)'
    
    # IMPORTANT, CANNOT SOLVE YET
    txt = 's(6a4c-31a3bc+6a3c2+19a2b2c)s(a2)-6s(a((b3-abc)-(ac2-abc)+2(bc2-abc)-3(b2c-abc)+x/8(ab2-abc)+y/8(a2c-abc))2)'
    
    txt = '2(s(ab)2s(a)-2abcs(a)2-3abcs(ab))(3s(ab)s(a)-19abc)-3p(a-b)(s(ab)2s(a)+4abcs(a)2+3abcs(ab)) -6s(ab(0(ab2-abc)-(a2b-abc)+0(bc2-abc)+0(b2c-abc)+0(ac2-abc)+0(a2c-abc))2)-3s(ab((ab2-abc)-(a2b-abc)-(bc2-abc)-2(b2c-abc)+x(ac2-abc)-y(a2c-abc))2)'
    txt = 's(18a4b4+9a5b2c+11a4b3c-66a3b4c+2a2b5c+10a4b2c2+16a3b3c2)-s(ab(3(b2c-abc)-1(a2c-abc)+(-4)(bc2-abc)+(5)(ac2-abc))2)-1s(ab((a2c-abc)+x(bc2-abc)+y(ac2-abc))2)'
    txt = 's(a3b3+7a4bc-29a3b2c+12a3bc2+9a2b2c2)s(ab)-7s(ab((b2c-abc)-(a2c-abc)+x(bc2-abc)+y(ac2-abc))2)'
    txt = 's(a5b+7a5c-7a4b2-5a4bc-19a4c2+18a3b3+2a3b2c+8a3bc2-5a2b2c2)-1/55s(ab((b2-ac)-(a2-ac)+x(c2-ac)-y(bc-ac)+(ab-ac))2)'
    txt = '(s(a6c2-6a4b2c2+5a3b3c2)-s((a3c-a2bc)2))/a/b/c*s(a2-ab)-2s(c(a3-a2c+(bc2-abc)-x(a2b-abc)+y(ab2-abc))2)'
    
    txt = 's(ab)(s(a)(s(a4c)s(a)-6abcs(ab)s(a)+45a2b2c2)-s(c(a3-a2c-(a2b-abc))2))-s(a2c(0(a2b-abc)+2a2c-2b2c+4(b2c-abc)+0(a2b-abc)-2(ab2-abc)-x(bc2-abc)+y(b3-abc))2)'
    txt = 's(6a6b2c+6a5b4-35a5b2c2+12a5bc3-35a4b4c+63a4b3c2+18a4b2c3-35a3b3c3)-6s(bc2(a2c-b2c-(12/5+x)(a2b-abc)+y(ab2-abc))2)'
    # txt = 's(a2b(a-b)(a-b-c))s(a2-ab)-s(b(a3-ab2+x(a2c-abc)+y(ac2-abc))2)'
    # txt = 's(a5c2+a4b3-7a4bc2+2a4c3+2a3b3c+a3b2c2)-s(a(a2c-abc-(ac2-abc)+x(b2c-abc)-y(ab2-abc))2) '
    
    print(search_positive(txt))

    