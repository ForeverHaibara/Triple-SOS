from math import gcd

import sympy as sp

from ...utils.text_process import cycle_expansion
from ...utils.root_guess import rationalize, rationalize_bound
from .peeling import _merge_sos_results, FastPositiveChecker

def _sos_struct_quintic(poly, degree, coeff, recurrsion):
    multipliers, y, names = [], None, None

    if coeff((5,0,0)) == 0:
        if coeff((4,1,0)) == 0 or coeff((1,4,0)) == 0:
            multipliers, y, names = _sos_struct_quintic_windmill(coeff)

        if y is not None:
            return multipliers, y, names

        multipliers, y, names = _sos_struct_quintic_hexagon(coeff, poly, recurrsion)

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


def _sos_struct_quintic_windmill(coeff):
    """
    Give solution to s(a3b2+?a2b3+?ab4+?a3bc+?a2b2c) >= 0,

    Examples
    -------
    s(a2c(4a-3b-c)2)-20abcs(a2-ab) 

    s(a)2s(a2b)-9abcs(a2) 

    4s(c(a-b)2(a-c)2)-s(c2(11a-15b-c)(a-b)2)

    s(4a4c+a3b2-23a3bc+4a3c2+14a2b2c)

    (s(c(4b2-c2)(a-b)2)-11s(c2(b-a)3))

    2s(a4b-3a3b2+ 4a2b3-2a2b2c)-17/2abcs(a2-ab)

    s(c(2a2-ab-ac)2)-4abcs(a2-ab)

    s(ab2(2(a-b)-5(b-c))2)-59abcs(a2-ab)

    s(ac2(3(a-b)+5(b-c))2)-25abcs(a2-ab)

    Reference
    -------
    [1] https://tieba.baidu.com/p/6472739202
    """
    multipliers, y, names = [], None, None
    if coeff((5,0,0)) != 0 or (coeff((4,1,0)) != 0 and coeff((1,4,0)) != 0):
        return [], None, None
    

    if coeff((4,1,0)) != 0:
        # reflect the polynomial so that coeff((4,1,0)) == 0
        def new_coeff(c):
            return coeff((c[0], c[2], c[1]))
        multipliers, y, names = _sos_struct_quintic_windmill(new_coeff)
        if y is not None:
            names = [_.translate({98: 99, 99: 98}) for _ in names]
        return multipliers, y, names
    
    # now we assume coeff((4,1,0)) == 0
    if coeff((1,4,0)) < 0:
        return [], None, None
    elif coeff((1,4,0)) == 0:
        # coeff((4,1,0)) == coeff((1,4,0)) == 0
        # now if we change variables a -> 1/a, b -> 1/b, c -> 1/c, it will be quartic
        # so we can see that coeff((3,0,2)) * x^2 + coeff((3,1,1)) * x + coeff((3,2,0)) >= 0
        # e.g. s(a3b2+4a2b3-5a2b2c)-4abcs(a2-ab)
        const_ = coeff((3,2,0)) + coeff((2,3,0)) + coeff((3,1,1)) + coeff((2,2,1))
        u_ = None
        if coeff((3,2,0)) < 0 or coeff((2,3,0)) < 0 or const_ < 0:
            y = None
        elif coeff((3,1,1)) >= 0:
            y = [sp.S(0), coeff((3,2,0)), coeff((2,3,0)), coeff((3,1,1)) / 2, const_]
        elif coeff((3,1,1)) ** 2 > 4 * coeff((3,2,0)) * coeff((2,3,0)):
            y = None
        else:
            u_ = coeff((3,1,1)) / (-2 * coeff((3,2,0)))
            y = [coeff((3,2,0)), sp.S(0), coeff((2,3,0)) - coeff((3,2,0)) * u_ ** 2, sp.S(0), const_]
        
        if y is not None:
            names = [
                f'b*({u_}*a*b-{u_-1}*a*c-b*c)^2' if u_ is not None else 'a^3*b*c',
                'a^2*c*(b-c)^2',
                'a^2*b*(b-c)^2',
                'a*b*c*(b-c)^2',
                'a^2*b^2*c'
            ]
        return multipliers, y, names

    if coeff((3,2,0)) == 0: 
        if coeff((2,3,0)) >= 0:
            multipliers, y, names = _sos_struct_quintic_uncentered(coeff)
        return multipliers, y, names
    elif coeff((3,2,0)) < 0:
        return [], None, None


    u, x_, y_, z_, = sp.symbols('u'), coeff((1,4,0)) / coeff((3,2,0)), coeff((2,3,0)) / coeff((3,2,0)), coeff((3,1,1)) / coeff((3,2,0))
    u_, y__, z__, w__ = None, y_, z_, coeff((2,2,1)) / coeff((3,2,0)) + (x_ + y_ + z_ + 1)

    # if coeff((2,3,0)) >= 0 and 1 + x_ + z_ >= 0:
    #     # Easy case 1. try very trivial case
    #     y = [
    #         sp.S(1),
    #         x_ - u_ ** 2,
    #         z_ - (-2 * u_**2 + 2 * u_),
    #         w__
    #     ]
    #     if any(_ < 0 for _ in y):
    #         y = None
    #     else:
    #         y = [_ * coeff((3,2,0)) for _ in y]
    #         names = ['a*c^2*(b-c)^2', 'a*b^2*(b-c)^2', 'a*b*c*(b-c)^2', 'a^2*b^2*c']
    #         return multipliers, y,  names

    if True:
        # Easy case 2. try another case where we do not need to updegree
        # idea: use s(a*b^2*(a-ub+(u-1)c)^2)
        u_ = y_ / (-2)
        y = [
            sp.S(1),
            x_ - u_ ** 2,
            (z_ - (-2 * u_**2 + 2 * u_)) / 2,
            w__
        ]
        if any(_ < 0 for _ in y):
            y = None
        else:
            y = [_ * coeff((3,2,0)) for _ in y]
            names = [f'a*b^2*(a-{u_}*b+{u_-1}*c)^2', 'a*b^2*(b-c)^2', 'a*b*c*(b-c)^2', 'a^2*b^2*c']
            return multipliers, y,  names

    

    # now we formally start
    # Case A.
    if coeff((2,3,0)) ** 2 <= 4 * coeff((1,4,0)) * coeff((3,2,0)):
        if w__ < 0:
            return [], None, None

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
            # Case 1. add a small perturbation so that we can obtain rational u and v
            if not ((y_ < 0) and (y_ ** 2 == 4 * x_)):
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
                    for u_ in rationalize_bound(u_numer, direction = direction, compulsory = True):
                        v_ = (u_**3*x_ - u_**2*x_ + u_*x_ - u_ + x_ + 1) / (u_*x_ + 1)
                        if u_ >= (v_ - 1)**2 / 4 + 1:
                            y__ = (u_**5*x_**2 - u_**4*x_**2 - 2*u_**3*x_ - u_**2*x_**2 + u_**2*x_ - 4*u_*x_ + u_ - x_ - 2)/((u_ + 1)*(u_*x_ + 1))
                            if y__ <= y_:
                                z__ = (-2*u_**2*v_ + u_**2 + u_*v_ - v_**2)/(u_**3 - u_**2 - u_*v_ + u_ + 1)
                                if z__ <= z_:
                                    break 
                        u_ = None

            else:
                # Case 2. when y_ < 0 and y_^2 = 4x_, then there is a root on the border
                # then perturbation has no chance
                # we take the limit of our sum of squares to the numerical, exact solution
                # e.g. s(c(2a2-ab-ac)2)-4abcs(a2-ab)
                
                # first solve v, which is a cubic equation
                # r = -4(v+1) / (v**3 - 3*v**2 + 7*v - 13)
                v = sp.symbols('v')
                r = -y_ / 2
                eq = ((v**3 - 3*v**2 + 7*v - 13) + 4*(v+1) / r).as_poly(v)
                # we can see that eq is strictly increasing and has a real root > -1
                for root in sp.polys.nroots(eq):
                    if root.is_real:
                        v_ = root
                        break
                
                # the true g is given by
                # g = b*(a*a + (1-r)*a*b - r*b*b + 2*(r-1)*a*c) + z0*a*c*(r*a - c - (r-1)*b) + b*c*(z1*b+z2*c-(z1+z2)*a)
                # with z0, z1, z2 given below
                z0  = (v_ +1)/2
                z1 = (3-v_)/2 + r*(2*v_-1)
                z2 = (1-v_)*(3*v_-1)*r/4 - 1
                
                w_ = coeff((3,1,1)) / coeff((3,2,0))
                # we have det >= 0 as long as w < -1 (in non-trivial case)
                def compute_discriminant(z0, z1, z2):
                    m_ = r**2*z0**2 + 2*r**2 - 2*r*z0 + 2*r*z1 + w_
                    n_ = -r**2*z0**2 + 2*r**2 - 2*r*z0**2 - 2*r*z0*z2 + 8*r*z0 - 2*r*z1 - 2*r*z2 - 8*r - w_*z0**2 + 2*z0**2 - 2*z0*z1 - 2*z0*z2 - 6*z0 - 2*z1*z2 + 2*z2 + 4
                    p_ = 2*r**2*z0 + 2*r*z2 - 2*r + w_*z0**2 + w_ + z0**2 + 2*z0*z2 - 2*z0 - z1**2 + 2*z1 + 2*z2 + 5
                    q_ = 3*r**2*z0**2 - 6*r**2*z0 + 5*r**2 - 4*r*z0**2 + 2*r*z0*z1 + 2*r*z0*z2 + 6*r*z0 - 2*r*z2 - 6*r + w_*z0**2 + w_ + 2*z0 - 2*z1 - z2**2 - 1
                    det_ = 3 * m_ * (m_ + n_) - (p_ ** 2 + q_ ** 2 + p_ * q_)
                    return m_, n_, p_, q_, det_
                
                det_, m_ = -1, -1
                for rounding in (.5, .2, .1, 1e-2, 1e-3, 1e-5, 1e-8):
                    z0_, z1_, z2_ = [sp.Rational(*rationalize(_, rounding = rounding, reliable = False)) for _ in (z0, z1, z2)]
                    m_, n_, p_, q_, det_ = compute_discriminant(z0_, z1_, z2_)
                    if det_ >= 0 and m_ > 0:
                        break

                if det_ >= 0 and m_ > 0:
                    u_ = 1 + (v_ - 1)**2/4
                    multipliers = [f'a*a+{z0_**2 + 2}*b*c']
                    y = [sp.S(1), m_ / 2, det_ / 6, w__]
                    y = [_ * coeff((3,2,0)) for _ in y]
                    names = [
                        f'a*(a^2*b+{1-r}*a*b^2-{r}*b^3+{r*z0_}*a^2*c-{z0_}*a*c^2+{z1_}*b^2*c+{z2_}*b*c^2-{(2-z0_)*(1-r)+z1_+z2_}*a*b*c)^2',
                        f'a*b*c*(a^2-b^2+{-(p_ + 2*q_)/3/m_}*(a*b-a*c)+{-(q_ + 2*p_)/3/m_}*(b*c-a*b))^2',
                        f'a^3*b*c*(b-c)^2',
                        f'a^2*b^2*c*(a*a+b*b+c*c+{z0_**2 + 2}*(a*b+b*c+c*a))'
                    ]

                return multipliers, y, names

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
                sp.S(1) if y_ != y__ or z_ != z__ else sp.S(0),
                w__]

            y = [_ * coeff((3,2,0)) for _ in y]

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

    s(a2c(a-b)(a+c-4b))
    """

    multipliers, y, names = [], None, None
    u, v = sp.symbols('u v')
    t = coeff((1,4,0))
    r1, r2, r3, u_, v_ = coeff((2,3,0)) / t, coeff((3,1,1)) / t, coeff((2,2,1)) / t, None, None
    r1_, r2_ = r1, r2

    if r2 + 2 * r1 >= 0:
        multipliers = []
        y = [coeff((2,3,0)), coeff((1,4,0)), coeff((3,1,1)) / 2 + coeff((1,4,0)),
            coeff((2,3,0)) + coeff((1,4,0)) + coeff((3,1,1)) + coeff((2,2,1))]
        if any(_ < 0 for _ in y):
            y = None
        else:
            names = ['a^2*b*(b-c)^2', 'a*b^2*(b-c)^2', 'a*b*c*(b-c)^2', 'a^2*b^2*c']
            return multipliers, y, names
    
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
                for u_ in rationalize_bound(u_numer, direction = -1, compulsory = True):
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
    here optimal w = 3.581412179607289955451719993913205662648 is the root of x**3-8*x**2+39*x-83

    Idea: x, y, z are coeffs to be determined, and apply quartic discriminant on:
    s(ab4-a2b2c-wabc(a2-ab))s(a2+(z*z+2)ab) - s(c(a3-abc-(x)(a2b-abc)+(y)(ab2-abc)-(z)(bc2-abc)+(a2c-abc))2)

    Optimal solution:
    x = 3.729311215312077309477934958844193027565   rootof(x**3-3*x**2+19*x-81)
    y = 2.079041499458323407339677415370865580968   rootof(x**3+6*x**2+x-37)
    z = 1.682327803828019327369483739711048256891   rootof(x**3-3*x**2+4*x-3)

    Code:
    p_ = -w*z**2 - w - x**2 - 2*y + 2*z
    q_ = -w*z**2 - w + 2*x*z - y**2 - 2*y*z + 2*y + 3*z**2 - 6*z + 5
    det_ = (-3*w + 6*x + 3*z**2 + 6)*(w*z**2 - w + 2*x*y + 2*y*z + 2*y + 4) - (p_**2 + p_*q_ + q_**2)
    det_ = det_.subs(w,3.581412179607289955451719993913205662648)
    print(sp.nsolve((det_.diff(x), det_.diff(y), det_.diff(z)), (x,y,z), (sp.S(2163)/580, sp.S(1736)/835, sp.S(1647)/979), prec = 40))
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
        det_ = (-3*w + 6*x + 3*z**2 + 6)*(w*z**2 - w + 2*x*y + 2*y*z + 2*y + 4) - (p_**2 + p_*q_ + q_**2)
        return p_, q_, det_
    
    # candidate selections of (x,y,z) such that det >= 0
    candidates = [
        (sp.S(3), sp.S(2), sp.S(1)),                       # w = 3.5433               > 7/2
        (sp.S(26)/7, sp.S(2), sp.S(12)/7),                 # w = 3.580565...
        (sp.S(26)/7, sp.S(79)/38, sp.S(5)/3),              # w = 3.5813989766...      > 25/43
        (sp.S(2163)/580, sp.S(1736)/835, sp.S(1647)/979),  # w = 3.58141217960666...
        (sp.S(2520773882)/675935511, sp.S(1053041451)/506503334, sp.S(947713664)/563334721) # > w - 3.5e-37
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


def _sos_struct_quintic_hexagon(coeff, poly, recurrsion):
    """
    Try solving quintics without s(a5).

    Examples
    -------
    s(c(a-b)2(a+b-3c)2)

    s(a4b+a4c+6a3b2+a2b3-9a3bc)-10abcs(a2-ab)

    s(a4b+a4c+5a3b2+3a2b3-10a2b2c)-20s(a3bc-a2b2c)

    s((23(b-a)+31c)(a2-b2+(ab-ac)+2(bc-ab))2)
    """
    if coeff((5,0,0)) != 0 or coeff((4,1,0)) <= 0 or coeff((1,4,0)) <= 0:
        return [], None, None
    
    multipliers, y, names = [], None, None

    if coeff((4,1,0)) == coeff((1,4,0)):
        t = coeff((4,1,0))

        p, q, z, w = coeff((3,2,0)) / t, coeff((2,3,0)) / t, coeff((3,1,1)) / t, coeff((2,2,1)) / t
        # u^2-2v <= p   --->  v <= (u^2-p)/2
        # v^2-2u <= q   --->  (u^2-p)^2/4 - 2u <= q
        u = sp.symbols('u')
        eq = ((u*u - p)**2 / 4 - 2*u - q).as_poly(u)
        u_, v_ = None, None
        for root in sp.polys.roots(eq, cubics = False, quartics = False):
            if isinstance(root, sp.Rational):
                u_ = root
                v_ = (u_ ** 2 - p) / 2
                if -2 * u_ * v_ <= z:
                    break
                u_, v_ = None, None
        
        if u_ is None:
            try:
                for root in sp.polys.nroots(eq):
                    if root.is_real and - root * (root**2 - p) <= z:
                        direction = 1 if eq.diff()(root) <= 0 else -1
                        for u_ in rationalize_bound(root, direction, compulsory = True):
                            if eq(u_) <= 0:
                                v_ = (u_ ** 2 - p) / 2
                                if -2 * u_ * v_ <= z:
                                    break
                        else:
                            u_, v_ = None, None
            except: pass

        if u_ is not None:
            y = [
                1,
                q - (v_ **2 - 2 * u_),
                (z + 2 * u_ * v_) / 2,
                2 + p + q + z + w
            ]
            if any(_ < 0 for _ in y):
                y = []
            else:
                y = [_ * t for _ in y]
                names = [
                    f'c*(a^2-b^2+{u_}*(a*b-a*c)+{v_}*(b*c-a*b))^2',
                    'a^2*b*(b-c)^2',
                    'a*b*c*(b-c)^2',
                    'a^2*b^2*c'
                ]
                return multipliers, y, names
            

    if y is None:
        # try updegree
        multipliers = ['a*b']
        poly2 = poly * sp.polys.polytools.Poly('a*b+b*c+c*a')
        multipliers , y , names = _merge_sos_results(multipliers, y, names, recurrsion(poly2, 7))

    return multipliers, y, names
