from math import gcd

import sympy as sp

from ...utils.text_process import cycle_expansion
from ...utils.root_guess import rationalize
from .peeling import _merge_sos_results, FastPositiveChecker

def _sos_struct_quintic(poly, degree, coeff, recurrsion):
    multipliers, y, names = [], None, None

    if coeff((5,0,0)) == 0:
        if coeff((4,1,0)) == 0 and coeff((1,4,0)) >= 0:
            if coeff((3,2,0)) > 0 and coeff((2,3,0)) ** 2 <= 4 * coeff((1,4,0)) * coeff((3,2,0)):
                # https://tieba.baidu.com/p/6472739202
                u, x_, y_, z_, = sp.symbols('u'), coeff((1,4,0)) / coeff((3,2,0)), coeff((2,3,0)) / coeff((3,2,0)), coeff((3,1,1)) / coeff((3,2,0))
                u_, y__, z__, w__ = None, y_, z_, coeff((2,2,1)) / coeff((3,2,0)) + (x_ + y_ + z_ + 1)
                eq = (u**5*x_**2 - u**4*x_**2 - 2*u**3*x_ + u**2*(-x_**2 - x_*y_ + x_) + u*(-x_*y_ - 4*x_ - y_ + 1) - x_ - y_ - 2).as_poly(u)
                for root in sp.polys.roots(eq, cubics = False, quartics = False).keys():
                    if isinstance(root, sp.Rational) and root > .999:
                        u_ = root
                        v_ = (u_**3*x_ - u_**2*x_ + u_*x_ - u_ + x_ + 1) / (u_*x_ + 1)
                        if u_ >= (v_ - 1)**2 / 4 + 1:
                            z__ = (-2*u_**2*v_ + u_**2 + u_*v_ - v_**2)/(u_**3 - u_**2 - u_*v_ + u_ + 1)
                            break
                        u_ = None
                else:
                    for root in sp.polys.nroots(eq):
                        if root.is_real and root > .999:
                            u_ = root
                            v_ = (u_**3*x_ - u_**2*x_ + u_*x_ - u_ + x_ + 1) / (u_*x_ + 1)
                            if u_ >= (v_ - 1)**2 / 4 + 1:
                                break
                            u_ = None
                    if u_ is not None:
                        # approximate a rational number
                        direction = (u_**2*x_ - 1)*(3*u_**4*x_**2 + 2*u_**3*x_**2 + 4*u_**3*x_ - 3*u_**2*x_**2 + 3*u_**2*x_ - 6*u_*x_ - x_**2 + x_ - 3)
                        direction = -1 if direction > 0 else 1
                        u_numer = u_
                        for tol in (.3, .1, 3e-2, 3e-3, 3e-4, 3e-5, 3e-7, 3e-9):
                            u_ = sp.Rational(*rationalize(u_numer + direction * tol * 3, rounding = tol))
                            v_ = (u_**3*x_ - u_**2*x_ + u_*x_ - u_ + x_ + 1) / (u_*x_ + 1)
                            if u_ >= (v_ - 1)**2 / 4 + 1:
                                y__ = (u_**5*x_**2 - u_**4*x_**2 - 2*u_**3*x_ - u_**2*x_**2 + u_**2*x_ - 4*u_*x_ + u_ - x_ - 2)/((u_ + 1)*(u_*x_ + 1))
                                if y__ <= y_:
                                    z__ = (-2*u_**2*v_ + u_**2 + u_*v_ - v_**2)/(u_**3 - u_**2 - u_*v_ + u_ + 1)
                                    if z__ <= z_:
                                        break 
                            u_ = None


                if u_ is not None and isinstance(u_, sp.Rational):
                    # now both u_, v_ are rational
                    u, v = u_, v_
                    r = (u.q * v.q / sp.gcd(u.q, v.q)) # cancel the denominator is good
                    # r = 1
                    r2 = r ** 2

                    multipliers = [f'{r}*a*a+{r*(u+v+1)}*b*c']
                    names = [f'a*({r2*(-u*v + u + 2)}*a^2*b+{r2*(-u*v + u - v + 1)}*a*b^2+{r2*(-v-1)}*b^3+{r2*(-2*u + v**2 + 3)}*a^2*c'
                                + f'+{r2*(-4*u**2 + 4*u*v + 2*u - v**2 - 3*v)}*a*b*c+{r2*(2*u**2 - u*v + 3*u + v**2 + 2*v - 3)}*b^2*c'
                                + f'+{r2*(2*u**2 + u*v - 3*u - v - 1)}*a*c^2+{r2*(-2*u*v - 2*u - v**2 + 4*v - 1)}*b*c^2)^2',

                            f'a*({r*u}*a^2*b+{r*(u-1)}*a*b^2+{-r}*b^3+{r*(-v-1)}*a^2*c+{r*(-2*u+v)}*a*b*c+{r*(-u+v+1)}*b^2*c+{r*(u+1)}*a*c^2+{r*(1-v)}*b*c^2)^2',
                            f'a*b*c*(a*a-b*b+{u}*(a*b-a*c)+{v}*(b*c-a*b))^2',
                            f'({y_ - y__}*a+{(z_ - z__)/2}*c)*a*b*(b-c)^2*({r}*(a*a+b*b+c*c)+{r*(u+v+1)}*(a*b+b*c+c*a))',
                            f'a^2*b^2*c*({r}*(a*a+b*b+c*c)+{r*(u+v+1)}*(a*b+b*c+c*a))']

                    denom = (u**3 - u**2 - u*v + u + 1)
                    y = [1 / denom / 4 / r2 / r,
                        (4*u - v*v + 2*v - 5) / denom / 4 / r,
                        (u + v + 2) * (4*u + v - 4) / denom / 2 * r,
                        1 if y_ != y__ or z_ != z__ else 0,
                        w__]
                    print(y, names)

                    y = [_ * coeff((3,2,0)) for _ in y]


                if False:
                    # deprecated
                    if sp.ntheory.primetest.is_square(t.p) and sp.ntheory.primetest.is_square(t.q):
                        t, p_, q_ = coeff((3,2,0)), sp.sqrt(t.p), sp.sqrt(t.q)
                        x_ = p_ / q_
                        if coeff((2,3,0)) == -t * p_ / q_ * 2:
                            v = sp.symbols('v')
                            for root in sp.polys.roots(x_ * (v**3 - 3*v*v + 7*v - 13) + 4*(v + 1)).keys():
                                if isinstance(root, sp.Rational) and root >= -1:
                                    v = root
                                    break
                            else:
                                v = None
                            
                            if v is not None:
                                y_ = 4*(v**2-4*v+7)*(2*v**3-3*v**2+6*v-1)/(v**3-3*v**2+7*v-13)**2
                                diff = coeff((3,1,1)) / t - (-2*x_*x_ + 2*x_ - y_)
                                if diff >= 0:
                                    diff2 = coeff((2,2,1)) / t + diff - (x_*x_ + y_ - 1)
                                    if diff2 >= 0:
                                        if diff >= y_:
                                            # use trivial method
                                            diff -= y_
                                            y = [sp.S(1), diff / 2, diff2]
                                            y = [_ * t for _ in y]
                                            names = [f'a*b*b*(a-{x_}*b+{x_-1}*c)^2', 'a*b*c*(a-b)^2', 'a^2*b^2*c']
                                        else:
                                            u = (v*v - 2*v + 5) / 4
                                            multipliers = [f'(a*a+{(v*v + 2*v + 9)/4}*b*c)']
                                            y = [4/(v**3 - 3*v*v + 7*v - 13)**2, 8*(v*v - v + 1)*(v*v + 2*v + 13)/(v**3 - 3*v*v + 7*v - 13)**2,
                                                diff, diff2]
                                            y = [_ * t for _ in y]
                                            # names = [f'a*(({v**2-2*v+9}*b-{v**2-1}*c)*(a*a-b*b+{u}*(a*b-a*c)+{v}*(b*c-a*b))'
                                            #                     + f'+({2*(v+1)}*a+{v**2-4*v+7}*b)*(b*b-c*c+{u}*(b*c-a*b)+{v}*(c*a-b*c)))^2']
                                            names = [f'a*({-2*u*v-2*u+v**2-2*v+9}*a^2*b+{2*u*v+2*u-v**3+2*v**2-7*v+2}*a*b^2+{-2*v-2}*b^3'
                                                            + f'+{v**2+2*v+1}*a^2*c+{-2*u*v**2+4*u*v-6*u+2*v**3-6*v**2+4*v}*a*b*c'
                                                            + f'+{u*v**2-4*u*v+7*u+3*v**2+2*v-1}*b^2*c+{u*v**2-u-2*v-2}*a*c^2+{-v**3-v**2+5*v-7}*b*c^2)^2']
                                            names += [f'a*b*c*(a*a-b*b+{u}*(a*b-a*c)+{v}*(b*c-a*b))^2',
                                                        f'a*b*c*(a*a+{(v*v + 2*v + 9)/4}*b*c)*(a*a+b*b+c*c-a*b-b*c-c*a)',
                                                        f'a*b*c*(a*a+{(v*v + 2*v + 9)/4}*b*c)*(a*b+b*c+c*a)']

            elif coeff((3,2,0)) == 0 and coeff((2,3,0)) >= 0 and coeff((1,4,0)) > 0:
                multipliers, y, names = _sos_struct_quintic_uncentered(coeff)


        elif coeff((1,4,0)) == 0 and coeff((4,1,0)) >= 0:
            # reflect the polynomial
            def new_coeff(c):
                return coeff((c[0], c[2], c[1]))
            multipliers, y, names = _sos_struct_quintic(None, 5, new_coeff, recurrsion)
            if y is not None:
                names = [_.translate({98: 99, 99: 98}) for _ in names]

        if y is None:
            # try hexagon
            multipliers = ['a*b']
            poly2 = poly * sp.polys.polytools.Poly('a*b+b*c+c*a')
            multipliers , y , names = _merge_sos_results(multipliers, y, names, recurrsion(poly2, 7))
    else:
        a = coeff((5,0,0))
        if a > 0:
            # try Schur to hexagon
            b = coeff((4,1,0))
            if b >= -2 * a:
                fpc = FastPositiveChecker()
                # name = '(a^2+b^2+c^2-a*b-b*c-c*a)*a*(a-b)*(a-c)'
                name = 'a*(a-b)^2*(a-c)^2'
                poly2 = poly - a * sp.sympify(cycle_expansion(name))
                fpc.setPoly(poly2)
                if fpc.check() == 0:
                    y = [a]
                    names = [name]
                    multipliers , y , names = _merge_sos_results(multipliers, y, names, recurrsion(poly2, 5))
                if y is None and b >= -a:
                    name = 'a^3*(a-b)*(a-c)'
                    poly2 = poly - a * sp.sympify(cycle_expansion(name))
                    fpc.setPoly(poly2)
                    if fpc.check() == 0:
                        y = [a]
                        names = [name]
                        multipliers , y , names = _merge_sos_results(multipliers, y, names, recurrsion(poly2, 5))
                    
    return multipliers, y, names


def _sos_struct_quintic_uncentered(coeff):
    """
    Give the solution to s(ab4+?a2b3-?a3bc+?a2b2c) >= 0, 
    which might have equality not at (1,1,1) but elsewhere.

    Example
    -------
    s(2ab4+5a2b3-17a3bc+13a2b2c)

    s(ab4+347/30a2b3-3833/230a3bc+475/69a2b2c)

    s(ab4+0a2b3-19/4a3bc+9/2a2b2c)

    s(2ab4+4a2b3-13a3bc+7a2b2c)
    """

    multipliers, y, names = [], None, None
    u, v = sp.symbols('u v')
    t = coeff((1,4,0))
    r1, r2, r3, u_, v_ = coeff((2,3,0)) / t, coeff((3,1,1)) / t, coeff((2,2,1)) / t, None, None
    r1_, r2_ = r1, r2
    
    if coeff((2,3,0)) + coeff((1,4,0)) + coeff((3,1,1)) + coeff((2,2,1)) == 0:
        if coeff((2,3,0)) == 0:
            return _sos_struct_quintic_windmill_special(coeff)

        # v = (u^3 - u^2 + u + 1) / u
        eq = (u**4 - u**3 - (1 + r1) * u - r1).as_poly(u)

        for root in sp.polys.roots(eq, cubics = False, quartics = False).keys():
            if isinstance(root, sp.Rational) and root > 0:
                u_ = root
                if u_ ** 3 - u_ ** 2 - 1 >= 0:
                    v_ = (u_**3 - u_**2 + u_ + 1) / u_
                    r1_ = r1
                    r2_ = -(3*u_**4 - 2*u_**3 + 3*u_ + 1)/(u_*(u_ + 1))
                    break
                u_ = None
        else:
            for root in sp.polys.nroots(eq):
                if root.is_real and root > 1:
                    u_ = root
                    break
            if u_ is not None:
                # approximate a rational number
                u_numer = u_
                for tol in (.3, .1, 3e-2, 3e-3, 3e-4, 3e-5, 3e-7, 3e-9):
                    u_ = sp.Rational(*rationalize(u_numer - tol * 3, rounding = tol))
                    if u_ > 0:
                        r1_ = u_*(u_**3 - u_**2 - 1)/(u_ + 1) 
                        if 0 <= r1_ <= r1:
                            r2_ = -(3*u_**4 - 2*u_**3 + 3*u_ + 1)/(u_*(u_ + 1))# + 2*(r1 - r1_)
                            if r2_ <= r2:
                                v_ = (u_**3 - u_**2 + u_ + 1) / u_
                                break


    elif coeff((2,3,0)) + coeff((1,4,0)) + coeff((3,1,1)) + coeff((2,2,1)) >= 0:
        eq_coeffs = [
            r1*(r1*(28*r1 - 20*r2 - 44) + r2*(4*r2 + 8) + 52),
            r1*(r1*(r1*(-4*r1 + 32*r2 - 128) + r2*(24 - 20*r2) + 92) + r2*(4*r2**2 + 20) - 264) + r2*(4*r2 + 8) + 52,
            r1*(r1*(r1*(r1*(80 - 4*r2) + r2*(11*r2 - 124) + 308) + r2*(r2*(45 - 5*r2) - 8) - 56) \
                + r2*(r2*(r2*(r2 - 6) - 7) - 132) + 464) + r2*(4*r2**2 + 40) - 220,
            r1*(r1*(r1*(r1*(-16*r1 + r2*(36 - r2) - 176) + r2*(r2*(r2 - 36) + 192) - 352) + r2*(r2*(13*r2 - 33) - 24) + 40) \
                + r2*(r2*(6 - 2*r2**2) + 168) - 328) + r2*(r2*(r2*(r2 - 6) + 13) - 156) + 400,
            r1*(r1*(r1*(r1*(r1*(r1 - 3*r2 + 33) + r2*(4*r2 - 56) + 176) + r2*(r2*(36 - 2*r2) - 141) + 189) \
                + r2*(r2*(-9*r2 - 8) + 52) - 108) + r2*(r2*(r2*(r2 + 9) - 8) - 39) - 11) + r2*(r2*(r2*(5 - 2*r2) - 36) + 214) - 361,
            r1*(r1*(r1*(r1*(r1*(-2*r1 + 4*r2 - 20) + r2*(27 - 3*r2) - 48) + r2*(r2*(r2 - 10) + 18) + 16) \
                + r2*(r2*(r2 + 25) - 67) + 128) + r2*(r2*(16 - 7*r2) - 66) + 108) + r2*(r2*(r2*(r2 - 2) + 35) - 105) + 142
        ]


        eq = sum(eq_coeffs[i] * v ** (5 - i) for i in range(6)).as_poly(v)

        for root in sp.polys.roots(eq, cubics = False, quartics = False).keys():
            if isinstance(root, sp.Rational) and root >= 2:
                v_ = root
                eq2 = u**4 + (r1 - v_ - 3)*u**3 + (r1*v_ - 3*r1 + v_**2 + 6)*u**2 \
                        + (-3*r1*v_ + 4*r1 - 2*v_ - 5)*u - r1*v_**2 + 3*r1*v_ - 2*r1 - v_ + 4
                eq2 = eq2.as_poly(u)
                for root2 in sp.polys.roots(eq2, cubics = False, quartics = False).keys():
                    if isinstance(root2, sp.Rational) and root2 > 0:
                        u_ = root2
                        r1_ = r1
                        r2_ = r2
                        break
                else:
                    v_ = None
                    continue

                break
        else:
            for root in sp.polys.nroots(eq):
                if root.is_real and root >= 2:
                    v_ = root
                    break
            if False and v_ is not None:
                # approximate a rational number
                v_numer = v_
                for tol in (.3, .1, 3e-2, 3e-3, 3e-4, 3e-5, 3e-7, 3e-9):
                    v_ = sp.Rational(*rationalize(v_numer - tol * 3, rounding = tol))
                    if v_ >= 2:
                        r1_ = u_*(u_**3 - u_**2 - 1)/(u_ + 1) 
                        if r1_ <= r1:
                            r2_ = -(3*u_**4 - 2*u_**3 + 3*u_ + 1)/(u_*(u_ + 1))# + 2*(r1 - r1_)
                            if r2_ <= r2:
                                v_ = (u_**3 - u_**2 + u_ + 1) / u_
                                break

    if u_ is not None and v_ is not None:
        u, v = u_, v_
        factor1 = (u**4 + 3*u**3*v - 2*u**3 - u**2*v**2 - 5*u**2*v - u**2 - 2*u*v**2 + 5*u*v + 5*u - v**2 + 4*v - 6)
        factor2 = (u**3 + u**2*v - u**2 - 3*u*v - u - v + 4)
        w = (u + 1) * (2*u**2 - u*v - 3*u - v + 4) / ((u**2 - u + 1) * factor2)
        d1 = u**2*v**2*w**2 + 2*u**2*v*w - 2*u*v*w**2 - 2*u*w - 2*u + w**2
        a_ = -(u + 1) * (u*v - 1) * (2*u**2 - u*v - 3*u - v + 4) / factor1
        phi = r1_ * (1/(u + a_)**2 - 1) + d1
        rho = (u**2 - u*v - u + v**2 - v + 3)/(u + v - 1)
        z = a_ / (u*v - 1)

        # cancel the denominator is good
        m1 = z.as_numer_denom()[1]
        m2 = (u.as_numer_denom()[1]**2 - u.as_numer_denom()[1] + 1, v.as_numer_denom()[1])
        m2 = (m2[0] * m2[1]) / sp.gcd(m2[0], m2[1])
        
        multipliers = [f'a^2+{phi}*a*b']
        
        y = [
            sp.S(1) / m2**2,
            r1_ * factor1**2 / (((u**2 - u + 1) * factor2 * m1)**2),
            (u + v - 1) * (u**3 - u*u - u*v + u + 1) * (phi + 1) / (u*v - 1) / (u*u - 2*u - v + 2) / 3
        ]
        y.append((r2_ + phi + 2*(u**3*w - u + v + w) - 3 * y[-1]) / 2)
        y.append(1 if (r1 != r1_ or r2 != r2_) else 0)

        y = [_ * t for _ in y]

        names = [
            f'a*({m2*(-u*v*w + w)}*a^2*c + {m2*(u*u*v*w - u*w - u)}*a*b^2 + {m2*(u**3*w - u*u*v*w - u*u*w + u*v*w + u*w - v*w + v)}*a*b*c'
           +f'+ {m2}*b^3 + {m2*(-u**3*w + u - v - w)}*b^2*c + {m2*(u**2*w + v*w - 1)}*b*c^2)^2',
            
            f'a*({m1}*a^2*c + {m1*(u*v*z - z)}*a*b^2 + {m1*(u**2*z - u*z + u - v**2*z + v*z - v)}*a*b*c + {m1*(-u*v*z - u + z)}*a*c^2 +'
           +f'{m1*(-u**2*z - v*z - 1)}*b^2*c + {m1*(u*z + v**2*z + v)}*b*c^2)^2',

            f'a*b*c*(a^2+b^2+c^2-{rho}*a*b-{rho}*b*c-{rho}*c*a)^2',

            f'a*b*c*(a^2-b^2-{u}*a*c+{v}*b*c+{u-v}*a*b)^2',

            f'a*c*(a-b)^2*({r1-r1_}*c+{(r2-r2_)/2}*b)*(a^2+b^2+c^2+{phi}*a*b+{phi}*b*c+{phi}*c*a)',
        ]


    return multipliers, y, names


def _sos_struct_quintic_windmill_special(coeff):
    """
    Give the solution to s(ab4-a2b2c) >= wabcs(a2-ab)
    here optimal w = 3.5814121796 is the root of x^3-8x^2+39x-83

    Idea: x, y, z are coeffs to be determined, and apply quartic discriminant on:
    s(ab4-a2b2c-wabc(a2-ab))s(a2+(z*z+2)ab) - s(c(a3-abc-(x)(a2b-abc)+(y)(ab2-abc)-(z)(bc2-abc)+(a2c-abc))2)

    Optimal solution:
    x = 3.72931121531208
    y = 2.07904149945832
    z = 1.68232780382802
    """
    t = coeff((1,4,0))
    w =  - coeff((3,1,1)) / t
    if w > 3 and w**3 - 8*w*w + 39*w - 83 > 0:
        return [], None, None
    elif w <= 2:
        # very trivial in this case
        multipliers = []
        y = [t, (2 - w) * t / 2]
        names = ['a^2*c*(a-b)^2', 'a*b*c*(a-b)^2']

        return multipliers, y, names


    # compute the quartic discriminant
    def det_(x, y, z):
        p_ = -w*z**2 - w - x**2 - 2*y + 2*z
        q_ = -w*z**2 - w + 2*x*z - y**2 - 2*y*z + 2*y + 3*z**2 - 6*z + 5
        return p_, q_, (-3*w + 6*x + 3*z**2 + 6)*(w*z**2 - w + 2*x*y + 2*y*z + 2*y + 4) - (p_**2 + p_*q_ + q_**2)
    
    # candidate selections of (x,y,z) such that det >= 0
    candidates = [
        (sp.S(3), sp.S(2), sp.S(1)),                       # w = 3.5433               > 7/2
        (sp.S(26)/7, sp.S(2), sp.S(12)/7),                 # w = 3.580565...
        (sp.S(26)/7, sp.S(79)/38, sp.S(5)/3),              # w = 3.5813989766...      > 25/43
        (sp.S(2163)/580, sp.S(1736)/835, sp.S(1647)/979)   # w = 3.58141217960666...
    ]

    for x_, y_, z_ in candidates:
        p_, q_, det__ = det_(x_, y_, z_)
        if det__ < 0:
            continue

        multipliers = [f'a^2+{z_**2+2}*a*b']

        m_ = (-w + 2*x_ + z_**2 + 2) 
        y = [
            sp.S(1), 
            m_ / 2,
            det__ / 6 / m_
        ]
        y = [_ * t for _ in y]

        names = [
            f'c*(a^3-{x_}*a^2*b+{y_}*a*b^2-{z_}*b*c^2+a^2*c+{x_-y_+z_-2}*a*b*c)^2',
            f'a*b*c*(a^2-b^2+{-(p_+q_*2)/3/m_}*(a*b-a*c)+{-(2*p_+q_)/3/m_}*(b*c-a*b))^2',
            f'a^3*b*c*(b-c)^2'
        ]

        return multipliers, y, names
    
    return [], None, None