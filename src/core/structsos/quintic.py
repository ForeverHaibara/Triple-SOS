from itertools import zip_longest, product

import sympy as sp
import numpy as np

from .quintic_symmetric import sos_struct_quintic_symmetric
from ...utils.roots.roots import Root
from ...utils.basis_generator import invarraylize
from .utils import (
    CyclicSum, CyclicProduct, Coeff,
    sum_y_exprs, nroots, rationalize, rationalize_bound, cancel_denominator, radsimp,
    prove_univariate
)

a, b, c = sp.symbols('a b c')

def _verify_border_nonnegative(border):
    """Verify whether a polynomial >= 0 over R+."""
    if border.degree() == 0:
        return True
    if border.LC() < 0:
        return False
    factor_list = border.factor_list()[1]
    for factor, multiplicity in factor_list:
        if multiplicity % 2 == 1:
            if factor.degree() == 1 and factor.coeff_monomial((0,)) == 0:
                continue
            if sp.polys.count_roots(factor, 0, None):
                return False
    return True

def sos_struct_quintic(poly, coeff, recurrsion):
    """
    Solve quintic inequalities.
    """

    # first try symmetric solution
    if coeff.is_rational and coeff((4,1,0)) == coeff((1,4,0)) and coeff((3,2,0)) == coeff((2,3,0)):
        return sos_struct_quintic_symmetric(poly, coeff, recurrsion)

    if coeff((5,0,0)) == 0:
        return _sos_struct_quintic_hexagon(coeff)
    else:
        return _sos_struct_quintic_full(coeff, poly)
    return None


def _sos_struct_quintic_full(coeff, poly):
    """
    Try to solve quintic with nonzero a^5 coefficient by subtracting something.

    Let f(a,b,c) = a^2-b^2 + u(ab-ac) + v(bc-ab) to be the quadratic form. We consider
    F(a,b,c) = \sum a(f(a,b,c) + xf(b,c,a))^2 + k\sum cf(a,b,c)^2 >= 0.
    This is tight when uv >= 1 and u,v >= 0 because there exists nontrivial equality cases.

    Expand F, we obtain
    F(a,b,c) = \sum (a + c(x^2+k))f(a,b,c)^2 + 2x\sum af(a,b,c)f(b,c,a).
    Now we take y = x^2 + k >= x^2. 
    We solve u,v,x,y from the coefficient of a^5,a^4b,a^3b^2,a^2b^3,ab^4.
    
    Examples
    -------
    s(a5+a4b-3a3b2-9a3bc+4a3c2+6a2b2c)

    s(a5-3a4b+3a3b2-6a3bc+3a3c2+2a2b2c)

    s(2a5-5a4b+a4c+5a3b2-19a3bc+6a3c2+10a2b2c)

    s(a2(a-b)(a2+2b2-2ac+2c2))-5/4abcs(a2-ab)

    s(a2(a-b)(a2-3bc))

    s(10a5-19a4b+4a3b2+5a2b2c)

    s(38a5-73a4b+16a3b2+19a2b2c)

    s(a2(a-b)(a2+ab-5bc))

    s(a2(a-b)(a2+b2-3ac+3c2))-1/3abcs(a2-ab)

    s(2a5-5a4b-7a4c+5a3b2+14a3bc+6a3c2-15a2b2c)

    s((23a-5b-c)(a-b)2(a+b-3c)2)

    s(36a5-39a4b-164a4c-122a3b2+311a3bc+253a3c2-275a2b2c)

    s(53361a5-354459a4b-107547a4c+678010a3b2+1034825a3bc-254726a3c2-1049464a2b2c)

    s(144400a5-261516a4b-1447320a4c-1613008a3b2+5235581a3bc+3966569a3c2-6024706a2b2c)

    s(34105600a5-71885436a4b-328258080a4c-328208596a3b2+1188005225a3bc+862751801a3c2-1356510514a2b2c)

    s(13660416a5-74323095a4b-26634720a4c+120979510a3b2+194638463a3bc-42269615a3c2-186050959a2b2c)
    """

    if coeff((5,0,0)) <= 0:
        return None

    #########################################################
    #
    #                     Method 1
    #
    #########################################################

    if True:
        # Method 1: Try subtracting s(a^2-ab)s(a^3 + xa^2b + yab^2 - (x+y+1)abc)
        # so that the rest falls into the hexagon case with equal coefficients of a^4b and ab^4.
        # To ensure coeff((4,1,0)) == coeff((1,4,0)), we can assume x = x0 - t, y = y0 - t
        coeff500 = coeff((5,0,0))
        x0, y0 = coeff((4,1,0)) / coeff500 + 1, coeff((1,4,0)) / coeff500 + 1
        # the remaining coefficient of a^3b^2 and a^2b^3 are given by
        p, q = coeff((3,2,0)) / coeff500 - (1-x0+y0), coeff((2,3,0)) / coeff500 - (1-y0+x0)
        z0 = coeff((3,1,1)) / coeff500 + (4+4*(x0+y0))
        x0, y0, p, q, z0 = radsimp([x0, y0, p, q, z0])
        print(p, q, z0)

        if p >= 0 and q >= 0 and (4*p*q >= z0**2 or z0 >= 0):
            t0 = sp.S(0)
        else:
            # Plug in the discriminant, the discriminant is linear of t, so we can solve the exact result.
            denom = radsimp((2*p + 2*q + z0)*(4*p**2 - 4*p*q - 2*p*z0 + 4*q**2 - 2*q*z0 + z0**2))
            if denom <= 0 or 4*p*q - z0**2 == 0:
                t0 = None
            else:
                t0 = (-4*p*q + z0**2)**2/(8*denom)

        if t0 is not None: # and t0 >= 0
            t0 = radsimp(t0)
            x_, y_ = x0 - t0, y0 - t0
            if (x_ >= 0 and y_ >= 0) or radsimp(-4*x_**3 + x_**2*y_**2 + 18*x_*y_ - 4*y_**3 - 27) <= 0:
                u_, v_ = 4*(-2*p**2 + q*z0)/(4*p*q - z0**2), 4*(p*z0 - 2*q**2)/(4*p*q - z0**2)
                u_, v_ = radsimp([u_, v_])

                if x_ >= 0 and y_ >= 0:
                    y = [sp.Rational(1,2)]
                    exprs = [CyclicSum((a-b)**2) * CyclicSum(a**3 + x_*a**2*b + y_*a*b**2 - (x_+y_+1)*CyclicProduct(a))]
                else:
                    # find t such that ((t**4 - 4*t**3 + 6*t**2 - 2*t - 1)/(t-1)**2, (2*t**3 - 6*t**2 + 6*t - 1)/(t-1)**2) <= (x,y)
                    t = sp.symbols('t')
                    eq = ((2*t**3 - 6*t**2 + 6*t - 1) - y_ * (t-1)**2).as_poly(t)
                    
                    t_ = None
                    if not coeff.is_rational:
                        # first check whether there exists multiplicative roots
                        eq_x = (t**4 - 4*t**3 + 6*t**2 - 2*t - 1 - x_ * (t-1)**2).as_poly(t)
                        eq_gcd = sp.polys.gcd(eq, eq_x)
                        if eq_gcd.degree() == 1:
                            t_ = radsimp(- eq_gcd.coeff_monomial((0,)) / eq_gcd.coeff_monomial((1,)))

                    _compute_x = lambda t: (t**4 - 4*t**3 + 6*t**2 - 2*t - 1)/(t-1)**2
                    _compute_y = lambda t: (2*t**3 - 6*t**2 + 6*t - 1)/(t-1)**2
                    if t_ is None:
                        for t_ in nroots(eq, method = 'factor', real = True):
                            if t_ < 1 and _compute_x(t_) <= x_:
                                if not isinstance(t_, sp.Rational):
                                    for t__ in rationalize_bound(t_, direction = -1, compulsory = True):
                                        if _compute_x(t__) <= x_ and _compute_y(t__) <= y_:
                                            t_ = t__
                                            break
                                break
                            else:
                                t_ = None
                    if t_ is not None:
                        y = [radsimp((t_ - 1)**(-2)), sp.S(1)]
                        x_, y_ = radsimp([x_ - _compute_x(t_), y_ - _compute_y(t_)])
                        one_minus_t_ = radsimp(1 - t_)
                        exprs = [
                            CyclicSum(a*(one_minus_t_*c-a+t_*b)**2*(one_minus_t_*a-b+t_*c)**2),
                            CyclicSum(a*(b-c)**2*(x_ * (a-c)**2 + y_*(a-b)**2))
                        ]

                if y is not None:
                    if p >= 0 and q >= 0 and (4*p*q >= z0**2 or z0 >= 0):
                        if z0 >= 0:
                            y.extend([sp.S(1), z0/2])
                            exprs.extend([
                                CyclicSum((q*a + p*b)*c**2*(a-b)**2),
                                CyclicProduct(a) * CyclicSum((a-b)**2)
                            ])
                        else:
                            w = radsimp(-z0/(2*p)) if p != 0 else sp.S(0)
                            y.extend([
                                p,
                                radsimp(q - p*w**2),
                            ])
                            exprs.extend([
                                CyclicSum(a*(a*b-w*a*c+radsimp(w-1)*b*c)**2),
                                CyclicSum(a*c**2*(a-b)**2),
                            ])
                    else:
                        y.append(t0)
                        exprs.append(CyclicSum(c*(a**2-b**2+u_*(a*b-a*c)+v_*(b*c-a*b))**2))
                    if all(_ >= 0 for _ in y):
                        y = [_ * coeff500 for _ in y]
                        y.append(poly(1,1,1) / 3)
                        exprs.append(CyclicSum(a**2*b**2*c))
                        return sum_y_exprs(y, exprs)

            # return None

    if not coeff.is_rational:
        return None

    #########################################################
    #
    #                     Method 2
    #
    #########################################################

    def _solve_uvxy(coeff):
        """
        Given quintic polynomial F(a,b,c), we can enhance it by finding maximum t such that
        F(a,b,c) - tabcs(a^2-ab) >= 0.
        At this maximum t, there exists nontrivial equality cases, and we assume
        that the equality satisfies a^2-b^2+u(ab-ac)+v(bc-ab) = 0 and its cyclic permutations.

        Now we represent F in the form of 
        F(a,b,c) = \sum (a + cy)f(a,b,c)^2 + 2x\sum af(a,b,c)f(b,c,a).

        Return u, v, x, y.
        """
        u, v = sp.symbols('u v')
        m, p, q, g, h = [coeff(_) for _ in [(5,0,0),(4,1,0),(1,4,0),(3,2,0),(2,3,0)]]
        p, q, g, h = p/m, q/m, g/m, h/m
        x1 = -p + q + 4*u - 2*v - 1
        x2 = 2*(u + v - 1)
        y1 = (2*u**2 - 4*u*v + 2*u - 2*v - 4)*x1 + 4*u**2*v - 4*u**2 + 2*u*v**2 - 4*u*v + 8*u - 2*v**3 + 6*v**2 - 4 + (g - h)*x2
        y3 = (u**2 + 2*u - v**2 - 2*v)
        y2 = y3 * x2

        eq1 = ((-p + 2*u - 2*v)*y2 - 2*u*x1*y3 + y1)
        eq2 = (-2*u**2 + 2*u*v + 2*u + 2)*x1*y3 + (u**2 - 2*v)*y1 + (- g + u**2 - 2*u*v + v**2 - 2)*y2
        eq2_ = eq2
        # print(sp.latex(eq1.subs({u:sp.symbols('x'), v:sp.symbols('y')})).replace(' ',''))
        # print(sp.latex(eq2.subs({u:sp.symbols('x'), v:sp.symbols('y')})).replace(' ',''))
        # print(eq1, '\n', eq2)
        eq1 = eq1.expand()
        eq2 = eq2.expand()

        # numerical solver: unstable
        # try:
        #     u, v = (sp.nsolve((eq1, eq2), (u, v), (2, 2)))
        # except:
        #     return None

        if True:
            u_, v_ = None, None
            v_eq = sp.polys.resultant(eq1, eq2, u).as_poly(v)
            method = 'factor' # if is_rational else 'numpy'
            roots = sorted(nroots(v_eq, method = method, real = True, nonnegative = True))#[::-1]
            for v_ in roots:
                u_eq = eq1.subs(v, v_).as_poly(u)
                roots_u = nroots(u_eq, method = method, real = True, nonnegative = True)
                for u_ in roots_u:
                    if abs(u_ - v_) > 1e-5 and abs(u_ + v_ - 1) > 1e-5 and abs(eq2_.subs({u:u_, v:v_})) < 1e-3:
                        u, v = u_, v_
                        break
                else:
                    u_ = None
                    continue
                break

        if u_ is None:
            return None

        x = (-p + q + 4*u - 2*v - 1)/(2*(u + v - 1))
        y = (g - h + 2*u**2*x - 4*u*v*x + 2*u*v + 2*u*x - 2*u - v**2 - 2*v*x + 2*v - 4*x + 2)/(u**2 + 2*u - v**2 - 2*v)
        # print('u v x y =', u, v, x, y)
        if not isinstance(x, (sp.Float, sp.Rational)) or not isinstance(y, (sp.Float, sp.Rational)):
            # this might happen when x = inf or y = inf
            return None

        # z = 2*u**2*x - 2*u**2 + 2*u*v*x - 2*u*v*y + 2*u*v - 2*v**2*x
        # print(u, v, x, y, z)
        # if z < coeff((3,1,1)) / m - 1e-5:
        #     return None
        return u, v, x, y


    sol = _solve_uvxy(coeff)
    if sol is None:
        return None

    u_, v_, x_, y_ = sol
    if y_ >= x_**2:
        m, p, q, g, h, z, w = [coeff(_) for _ in [(5,0,0),(4,1,0),(1,4,0),(3,2,0),(2,3,0),(3,1,1),(2,2,1)]]
        p, q, g, h, z, w = p/m, q/m, g/m, h/m, z/m, w/m

        lastu, lastv = None, None

        for u, v in zip_longest(
            rationalize_bound(u_, direction = 0), rationalize_bound(v_, direction = 0), fillvalue = None
        ):
            if u is None:
                u = lastu
            if v is None:
                v = lastv
            lastu, lastv = u, v

            x = (-p + q + 4*u - 2*v - 1)/(2*(u + v - 1))
            y = x**2
            if p + 2*(u*x-u+v) >= y:
                _new_coeffs = {
                    (5,0,0): sp.S(0),
                    (4,1,0): (p - (-2*u*x + 2*u - 2*v + y)) * m,
                    (1,4,0): (p - (-2*u*x + 2*u - 2*v + y)) * m,
                    (3,2,0): (g - (-2*u**2*x + u**2*y + u**2 + 2*u*v*x - 2*u*v + 2*u*x + v**2 - 2*v*y + 2*x - 2)) * m,
                    (2,3,0): (h - (u**2 - 2*u*v*x + 4*u*x - 2*u*y - 2*u + v**2*y - 2*v*x + 2*v - 2*x)) * m,
                    (3,1,1): (z - (2*u**2*x - 2*u**2 + 2*u*v*x - 2*u*v*y + 2*u*v - 2*v**2*x)) * m,
                    (2,2,1): (w - (-u**2*y - 2*u*v*x + 2*u*v*y - 4*u*x + 2*u*y + 2*u + 2*v**2*x - v**2*y - v**2 + 2*v*y + 2*x - 2*y)) * m,
                }
                solution = _sos_struct_quintic_hexagon(Coeff(_new_coeffs))
                if solution is not None:
                    xu, xv = x*u, x*v
                    solution = m * CyclicSum(
                        a*(a**2-b**2+(u-v)*a*b-u*a*c+v*b*c + x*b**2-x*c**2 + (xu-xv)*b*c - xu*b*a + xv*c*a)**2
                    ) + solution
                    return solution
        return None



    #########################################################
    #
    #                     Method 3
    #
    #########################################################

    # now we handle the ultimate case
    def _prove_quintic_from_border(border):
        """
        Let f(a,b,c) be a septic (degree-7) cyclic polynomial without a^7 term.
        Then B(x) = f(x,1,0) / x is quintic. Assume that B(x) >= 0 (x >= 0).
        Given such B, we find a sum of square form g(a,b,c), such that 
        \sum abg(a,b,c) = f(a,b,c) when c = 0. That is, the border of \sum g
        coincides with the border of f.

        This is used to cancel out the border of a septic polynomial, so that 
        it remains to prove a quartic * abc.
        """
        if border.LC() < 0:
            return None
        proof = prove_univariate(border, return_raw = True)
        if proof is None:
            return None
        s_args = []
        a, b, c = sp.symbols('a b c')
        for multiplier, coeffs, polys in proof:
            if multiplier == 1:
                for coeff, poly in zip(coeffs, polys):
                    bias = poly(1) # ensure f(1,1,1) = 0
                    poly = sum(v*a**i*b**(2-i) for (i,), v in poly.terms()) - bias*a*c
                    s_args.append(coeff * b * poly**2)
            elif multiplier == border.gen:
                for coeff, poly in zip(coeffs, polys):
                    bias = poly(1)
                    poly = sum(v*a**i*b**(2-i) for (i,), v in poly.terms()) - bias*b*c
                    s_args.append(coeff * a * poly**2)
        return s_args # sp.Add(*s_args)
    
    def _compute_ker(poly, uv = None):
        """
        Given quintic F(a,b,c). We denote B(x) = F(x,1,0) >= 0 be its border.
        Assume F(a,b,c) * \sum(a^2 + zab) = \sum a*V(a,b,c)^2 + abc\sum R(a,b,c).
        We can apply the quartic theorem on R(a,b,c). Now we will illustrate how to find V and z.

        Let f(x) = V(x,1,0) and g(x) = V(1,0,x). Take b = 1 and c = 0 in the identity
        we yield xf(x)^2 + g(x)^2 = B(x)(x^2 + zx + 1).

        Now we take x to be the five roots of B(x), then for each root we must have
        sqrt(-x) * f(x) \pm g(x) = 0
        Also we may assume V(1,0,0) = 1 when the leading coeff F is 1. We also require V(1,1,1) = 0
        to cover the equality at the center. Moreover, when F has tight nontrivial equality case
        (a0, b0, c0) and its permutations satisfying that \sum (a^2-b^2+u(ab-ac)+v(bc-ab))^2 == 0,
        we need V(a0,b0,c0) = 0 and its permutations.

        Together, there are 10 equations and 10 unknowns. And we can solve for V by solving a linear system,
        NUMERICALLY. Lastly, z is given by
        z = lim(x->0) (xf(x)^2 + g(x)^2 - B(x)) / (xB(x)) = f(0)^2 + 2g'(0) - B'(0)


        In most ideal cases, V and z are rational and everything is done. However, when V and z are
        not rational, we first assume that the R(a,b,c) > 0 is strict when (a,b,c) != (1,1,1). Then
        we can add a little more:
        F(a,b,c) * \sum (a^2 + (z+dz)ab) = \sum a* #V(a,b,c)^2 + \sum abU(a,b,c) + abc\sum #R(a,b,c).
        Here #V is a perturbation of V. We select dz > 0 such that the border is slightly larger than
        the original. Then U(a,b,c) is small and we use it to handle the remaining part of the border.
        Lastly we apply quartic theorem on the perturbed #R.
        """
        def _extract_quartic(poly):
            """Given septic polynomial F(a,b,c), return the quartic polynomial with coefficients
            being the a^5bc, a^4b^2c, a^3b^3c, a^2b^4c, ab^5c coefficients of F."""
            return sp.Poly([poly.coeff_monomial((5-i,i+1,1)) for i in range(5)], a)

        a, b, c, z = sp.symbols('a b c z')
        # standardize
        coeff500 = poly.coeff_monomial((5,0,0))
        poly = (poly / coeff500).as_poly(a,b,c)

        if uv is None:
            u_, v_, __, ___ = _solve_uvxy(poly)
        else:
            u_, v_ = uv

        border = poly.subs({b:1,c:0}).as_poly(a)
        border_std = ((a*a+z*a+1).as_poly(a) * border - a**7 - 1).div(a.as_poly(a))[0]
        border_std_coeff5 = border_std.coeff_monomial((5,))
        quartic_mul_a2 = _extract_quartic((a*a+b*b).as_poly(a,b,c) * poly)
        quartic_mul_ab = _extract_quartic((a*b+b*c+c*a).as_poly(a,b,c) * poly)

        def _get_quartic_remain(mul, params, return_mpnq = False):
            """
            Params is the coefficient of V, which is a length-10 vector.
            Compute _extract_quartic(f(a,b,c) * sum(z^2 + zab) - sum a*V(a,b,c)^2).
            When setting return_mpnq = True, return the coefficient of 
            a^5bc, a^4b^2c, a^3b^3c, a^2b^4c (denoted by m, p, n, q).
            """
            _, x1, x2, x3, x4, x5, x6, x7, x8, x9 = params
            quartic_sub = (2*x1*x2 + 2*x4 + 2*x6*x7 + 2*x8*x9)*(a**4+1) + (2*x1*x4 + 2*x2*x3 + 2*x4*x9 + 2*x5*x8 + 2*x6*x8 + x7**2 + 2*x7)*a**3 + (2*x1*x7 + 2*x1*x9 + 2*x2*x6 + 2*x2*x8 + 2*x3*x4 + 2*x4*x5 + 2*x6*x9 + 2*x7*x8)*a**2 + (2*x1*x5 + 2*x2*x4 + 2*x3*x7 + 2*x4*x6 + 2*x7*x9 + x8**2 + 2*x8)*a
            quartic_sub = quartic_sub.as_poly(a)
            quartic_rem = quartic_mul_a2 + quartic_mul_ab * mul - quartic_sub
            if return_mpnq:
                return [quartic_rem.coeff_monomial((4-i,)) for i in range(4)]
            return quartic_rem


        # construct the linear system, which is 10*10
        eqs = [np.array([1.] + [0] * 9), np.array([1.] * 10)]

        uvroot = Root.from_uv((u_, v_))
        a0, b0, c0 = uvroot.root
        as_vec = lambda *x: np.array(Root(x).as_vec(3, numer=False)).astype(np.float64).flatten()
        eqs.extend([as_vec(a0,b0,c0), as_vec(b0,c0,a0), as_vec(c0,a0,b0)])

        criterion = lambda m,p,n,q: (m>0 and (3*m*(m+n)-p**2-p*q-q**2)>=0) or (m==0 and p>=0 and 4*p*q>=n**2)

        def _solve_flip_sgn_eqs(eqs, border, _get_quartic_remain = _get_quartic_remain, mpnq_criterion = criterion):
            """
            Add in the constraints of sqrt(-x) * f(x) \pm g(x) = 0.
            Note that there are 2 choices of sign for each equation, we need to try
            both of them.

            Return None if no solution is found. Return params of V and z otherwise.         
            """
            eqs = np.array(eqs, dtype = np.float64)
            target = np.array([1] + [0]*9, dtype=eqs.dtype)
            borderdiff = border.all_coeffs()[-2]
            as_vec_float = lambda *x: np.array(Root(x).as_vec(3, numer=False)).astype(np.float64).reshape((1,10))
            as_vec_complex = lambda *x: np.array(Root(x).as_vec(3, numer=False)).astype(np.complex128).reshape((1,10))

            eqs_choices = []
            for r in nroots(border):
                if r.is_real:
                    if r > 0:
                        return None
                    else: # r <= 0
                        eqs_choices.append(
                            [(sgn * float(sp.sqrt(-r).n(20)) * as_vec_float(r,1,0) + as_vec_float(1,0,r)) for sgn in (1,-1)]
                        )
                elif sp.im(r) > 0:
                    # conjugate pair must have equal selection of sign and we handle them at the same time
                    # hint: to ensure the solution is real, we can add in the real and imag part of a complex equation
                    # to the real equations
                    tmp = []
                    for sgn in (1,-1):
                        eq_complex = sgn * complex(sp.sqrt(-r).n(20)) * as_vec_complex(r,1,0) + as_vec_complex(1,0,r)
                        eq_real = np.vstack([np.real(eq_complex), np.imag(eq_complex)])
                        tmp.append(eq_real)
                    eqs_choices.append(tmp)

            for comb in product(range(2), repeat = len(eqs_choices)):
                # select the combination of signs
                eqs_ = np.vstack([eqs] + [eqs_choices[i][comb[i]] for i in range(len(eqs_choices))])
                params_ = np.linalg.solve(eqs_, target)
                mul_ = params_[6]**2 + params_[2]*2 - borderdiff
                if mul_ >= -1:
                    # we require mul >= -1 and det >= 0 to apply the quartic theorem
                    m_, p_, n_, q_ = _get_quartic_remain(mul_, params_, return_mpnq = True)
                    if mpnq_criterion(m_, p_, n_, q_):
                        print('Expected mpnq =', (m_, p_, n_, q_), '\nExpected det =', (3*m_*(m_+n_) - (p_**2+p_*q_+q_**2)).n(20), '\nExpected mul =', mul_)
                        return params_, mul_
            return None
        
        solved_params = _solve_flip_sgn_eqs(eqs, border)
        if solved_params is None:
            return None
        params_, mul_ = solved_params

        # print(f's(a2+{mul_}ab)' + poly_get_factor_form(poly))
        # ker = invarraylize(params_, cyc = False)
        # print(f's(a({poly_get_factor_form(ker)})2)')

        for rounding in (1, 144, 144**2, 144**3, 144**5):
            params = [sp.S(round(_ * rounding)) / rounding for _ in params_[:-1]]
            params.append(-sum(params)) # be sure that the sum is zero
            _, x1, x2, x3, x4, x5, x6, x7, x8, x9 = params
            border_sub = (2*x1 + x9**2)*a**5 + (x1**2 + 2*x3 + 2*x5*x9)*a**4 + (2*x1*x3 + 2*x2*x9 + x5**2 + 2*x6)*a**3 + (2*x1*x6 + 2*x2*x5 + x3**2 + 2*x9)*a**2 + (x2**2 + 2*x3*x6 + 2*x5)*a + 2*x2 + x6**2
            border_sub = border_sub.as_poly(a)

            # compute a rationalization of mul_
            z_min = sp.solve(border_std_coeff5 - (2*x1+x9**2), z)[0]
            det_z = sp.polys.discriminant(border_std - border_sub, a).as_poly(z)
            # print(border_sub, border_std)

            lower_bounds = nroots(det_z, method = 'numpy', real = True)
            lb = None
            for lb_ in lower_bounds:
                if lb_ >= z_min and (lb is None or lb_ < lb):
                    lb = lb_
            else:
                lb = max(mul_, z_min)
            lb = sp.S(round(lb * rounding + .5)) / rounding
            border_rem_tmp = border_std.subs(z, lb) - border_sub
            if _verify_border_nonnegative(border_rem_tmp):
                border_rem = border_rem_tmp
                mul = lb
            else:
                mul = None
                # for index, ((left, right), __) in enumerate(det_z.intervals()[::-1]):
                if True:
                    (left, right), _ = det_z.intervals()[-1]
                    left, right = det_z.refine_root(left, right, eps = 1/rounding)#/24)
                    for val in (left, right):
                        if val >= z_min:
                            border_rem_tmp = border_std.subs(z, val) - border_sub
                            # print('COUNT =', sp.polys.count_roots(border_rem_tmp, 0, None), 'LC =', border_rem_tmp.LC(), '\t', '(l,r) =', (left,right), 'index =', index)#border_rem_tmp.as_expr().n(5))
                            if _verify_border_nonnegative(border_rem_tmp):
                                border_rem = border_rem_tmp
                                mul = val
                                break
                    # if left < z_min:
                    #     break
                if mul is None:
                    continue

            quartic_rem = _get_quartic_remain(mul, params, return_mpnq = False)
            if quartic_rem.LC() < 0:
                continue

            border_proof = _prove_quintic_from_border(border_rem)
            if border_proof is None:
                continue
            quartic_sub2 = _extract_quartic((a*b).as_poly(a,b,c) * sp.Add(*border_proof).as_poly(a,b,c))
            quartic_rem -= quartic_sub2
            m_, p_, n_, q_ = [quartic_rem.coeff_monomial((4-i,)) for i in range(4)]

            # print('Det =', (3*m_*(m_+n_) - (p_**2+p_*q_+q_**2)).n(20), quartic_rem.as_expr().n(10), 'Mul =', mul.n(20), mul_)
            if criterion(m_, p_, n_, q_) and m_ > 0:
                # print('Solution Found.')
                # print(poly_get_factor_form(invarraylize(sp.Matrix(params), cyc = False)), mul)
                # print(border_proof)
                multiplier = CyclicSum(a**2 + mul*b*c)
                sol_main = CyclicSum(a * invarraylize(sp.Matrix(params), cyc = False).as_expr()**2)

                border_proof_split_a, border_proof_split_b = [], []
                for arg in border_proof: #.args:
                    if a in arg.args:
                        border_proof_split_a.append(arg)
                    else:
                        border_proof_split_b.append(arg)
                border_proof_split_a = sp.Add(*border_proof_split_a)
                border_proof_split_b = sp.Add(*border_proof_split_b)
                border_proof = CyclicSum(a*b*border_proof_split_a) + CyclicSum(a*b*border_proof_split_b)
                
                rest = 0
                if m_ >= 0:
                    det = 3*m_*(m_+n_) - (p_**2+p_*q_+q_**2)
                    rest = m_ / 2 * CyclicProduct(a) * CyclicSum((a**2-b**2+(p_+2*q_)/3/m_*(a*c-a*b)-(q_+2*p_)/3/m_*(b*c-a*b))**2)
                    rest = rest + det / 6 / m_ * CyclicProduct(a) * CyclicSum(a**2*(b-c)**2)
                elif m_ == 0:
                    # rest = 
                    0
                rest += poly(1,1,1) * (1 + mul) * CyclicSum(a**3*b**2*c**2)
                return (coeff500 * sol_main + coeff500 * border_proof + coeff500 * rest) / multiplier



    if not (isinstance(u_, sp.Rational) and isinstance(v_, sp.Rational)):
        return _compute_ker(poly, (u_, v_))

    return None



def _sos_struct_quintic_windmill(coeff):
    """
    Give solution to s(a3b2+?a2b3+?ab4+?a3bc+?a2b2c) >= 0,

    Theorem:
    When u >= (v - 1)^2/4 + 1
    s((b-a+(2u-1)c)(a^2-b^2+u(ab-ac)+v(bc-ab))^2) >= 0

    Examples
    -------
    s(a2c(4a-3b-c)2)-20abcs(a2-ab)

    s(a)2s(a2b)-9abcs(a2) 

    4s(c(a-b)2(a-c)2)-s(c2(11a-15b-c)(a-b)2)

    s(4a4c+a3b2-23a3bc+4a3c2+14a2b2c)

    (s(c(4b2-c2)(a-b)2)-11s(c2(b-a)3))

    2s(a4b-3a3b2+ 4a2b3-2a2b2c)-17/2abcs(a2-ab)

    s(a4b-3a3b2-8a3bc+7a3c2+3a2b2c)

    s(c(2a2-ab-ac)2)-4abcs(a2-ab)

    s(ab2(2(a-b)-5(b-c))2)-59abcs(a2-ab)

    s(ac2(3(a-b)+5(b-c))2)-25abcs(a2-ab)

    s(6a4c+10a3b2-67a3bc+19a3c2+32a2b2c)

    s(a4c+a3b2-13a3bc+6a3c2+5a2b2c)

    s(29a4c+711a3b2-4100a3bc+3599a3c2-239a2b2c)

    s(8a4c+2a3b2-425a3bc+613a3c2-198a2b2c)

    s(9a4c+a3b2-44a3bc+6a3c2+28a2b2c)

    s(2a4b-6a3b2-5a3bc+8a3c2+a2b2c)

    s(c(a-b)2(2a2+3ab+10ac+2c2))-abcs(a2-ab)

    s((b-a+3c)(a2-b2+2(ab-ac)+31/9(bc-ab))2)
    
    s((b-a+207/100c)((a2-b2+307/200(ab-ac)+247/100(bc-ab)))2)

    Reference
    -------
    [1] https://tieba.baidu.com/p/6472739202
    """
    solution = None
    if coeff((5,0,0)) != 0 or (coeff((4,1,0)) != 0 and coeff((1,4,0)) != 0):
        return None

    if not coeff.is_rational:
        return None

    if coeff((4,1,0)) != 0:
        # reflect the polynomial so that coeff((4,1,0)) == 0
        solution = _sos_struct_quintic_windmill(coeff.reflect())
        if solution is not None:
            solution = solution.xreplace({b:c, c:b})
        return solution
    
    # now we assume coeff((4,1,0)) == 0
    if coeff((1,4,0)) < 0:
        return None
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
        
        if y is not None and all(_ >= 0 for _ in y):
            exprs = [
                CyclicSum(b*(u_*a*b-(u_-1)*a*c-b*c)**2 if u_ is not None else a**3*b*c),
                CyclicSum(a**2*c*(b-c)**2),
                CyclicSum(a**2*b*(b-c)**2),
                CyclicSum((b-c)**2) * CyclicProduct(a),
                CyclicSum(a*b) * CyclicProduct(a)
            ]
            return sum_y_exprs(y, exprs)
        return None

    if coeff((3,2,0)) == 0:
        if coeff((2,3,0)) >= 0:
            solution = _sos_struct_quintic_uncentered(coeff)
        return solution
    elif coeff((3,2,0)) < 0:
        return None


    u, x_, y_, z_, = sp.symbols('u'), coeff((1,4,0)) / coeff((3,2,0)), coeff((2,3,0)) / coeff((3,2,0)), coeff((3,1,1)) / coeff((3,2,0))
    u_, y__, z__, w__ = None, y_, z_, coeff((2,2,1)) / coeff((3,2,0)) + (x_ + y_ + z_ + 1)
    
    if w__ < 0:
        return solution

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
    #         exprs = ['a*c^2*(b-c)^2', 'a*b^2*(b-c)^2', 'a*b*c*(b-c)^2', 'a^2*b^2*c']
    #         return multipliers, y,  exprs



    if True:
        # Easy case 2. in the form of s(c(a-b)2(xa2+yac+zc2+uab+vbc)) + ...s(a2b2c)
        # such that y^2 <= 4xz where x = coeff((1,4,0)) is fixed

        if coeff((2,3,0)) >= 0:
            # A. possibly simple and nice
            t = min(coeff((3,2,0)), coeff((2,3,0)))
            y = [
                coeff((1,4,0)), t,
                coeff((3,1,1)) / 2 + coeff((1,4,0)) + t,
                abs(coeff((3,2,0)) - coeff((2,3,0))),
            ]
            if all(_ >= 0 for _ in y):
                r = 1 / cancel_denominator(y)
                y = [_ * r for _ in y]
                character = b if coeff((3,2,0)) >= coeff((2,3,0)) else a
                exprs = [
                    CyclicSum(c*(a-b)**2*(y[0]*a**2 + y[1]*c**2 + y[2]*a*b + y[3]*character*c)),
                    CyclicSum(a*b) * CyclicProduct(a)
                ]
                y = [1 / r, w__ * coeff((3,2,0))]
                return sum_y_exprs(y, exprs)
        
        # B. y^2 <= 4xz
        # x = coeff((1,4,0))
        # y + z = coeff((2,3,0))
        # v + z = coeff((3,2,0)) => z <= coeff((3,2,0)) => y >= coeff((2,3,0)) - coeff((3,2,0)
        # -2(x + z) + 2u = coeff((3,1,1)) => z >= - x - coeff((3,1,1)) / 2
        #      => y <= coeff((2,3,0)) + x + coeff((3,1,1)) / 2
        # Problem: whether y >= ... has a solution such that
        # y^2 <= 4x(coeff((2,3,0)) - y)
        
        bound_l = coeff((2,3,0)) - coeff((3,2,0))
        bound_r = coeff((2,3,0)) + coeff((1,4,0)) + coeff((3,1,1)) / 2
        if bound_l <= bound_r:
            bound = -2 * coeff((1,4,0)) # symmetric axis
            if bound_l <= bound <= bound_r:
                # symmetric axis inbetween
                pass
            elif bound <= bound_l:
                bound = bound_l
            else:
                bound = bound_r

            y = [
                coeff((1,4,0)),
                bound,
                coeff((2,3,0)) - bound,
                coeff((3,1,1)) / 2 + coeff((1,4,0)) + (coeff((2,3,0)) - bound),
                coeff((3,2,0)) - (coeff((2,3,0)) - bound)
            ]
            
            if not (y[1] < 0 and y[1]**2 > 4 * y[0] * y[2]):
                r1 = 1 / cancel_denominator([sp.S(1), -y[1] / y[0] / 2])
                
                tmpcoeffs = [y[2] - y[1]**2 / 4 / y[0], y[3], y[4]]
                r2 = 1 / cancel_denominator(tmpcoeffs)

                exprs = [
                    CyclicSum(c*(a-b)**2*(r1*a-(-y[1]/y[0]/2*r1)*c)**2),
                    CyclicSum(c*(a-b)**2*(r2*tmpcoeffs[0]*c**2 + r2*y[3]*a*b + r2*y[4]*b*c)),
                    CyclicSum(a*b) * CyclicProduct(a)
                ]
                y = [y[0] / r1**2, 1 / r2, w__ * coeff((3,2,0))]

                if all(_ >= 0 for _ in y):
                    return sum_y_exprs(y, exprs)
                    


    if True:
        # Easy case 3. try another case where we do not need to updegree
        # idea: use s(a*b^2*(a-ub+(u-1)c)^2)
        u_ = y_ / (-2)
        y = [
            sp.S(1),
            x_ - u_ ** 2,
            (z_ - (-2 * u_**2 + 2 * u_)) / 2 + (x_ - u_ ** 2),
            w__
        ]
        if all(_ >= 0 for _ in y):
            y = [_ * coeff((3,2,0)) for _ in y]
            exprs = [
                CyclicSum(a*b**2*(a-u_*b+(u_-1)*c)**2),
                CyclicSum(a*b**2*(b-c)**2),
                CyclicSum((b-c)**2) * CyclicProduct(a),
                CyclicSum(a*b) * CyclicProduct(a)
            ]
            return sum_y_exprs(y, exprs)

    if True:
        # Easy case 4. use s(c*(a-c)^2*(a-t*b)^2)
        if x_ >= 1:
            t = (z_ + 2*(x_ - 1)) / (-4)
            if t**2 - 2 <= y_:
                y = [
                    sp.S(1),
                    y_ - (t**2 - 2),
                    x_ - 1,
                    w__
                ]
                if all(_ >= 0 for _ in y):
                    y = [_ * coeff((3,2,0)) for _ in y]
                    exprs = [
                        CyclicSum(c*(a-c)**2*(a-t*b)**2),
                        CyclicSum(a*c**2*(a-b)**2),
                        CyclicSum(a**2*c*(a-b)**2),
                        CyclicSum(a*b) * CyclicProduct(a)
                    ]
                    return sum_y_exprs(y, exprs)
        elif x_ < 1:
            # two cases:
            # 1. x_s(c(a-c)2(a-tb)2) + ?s(a3(b-c)2) + ??s(bc2(a-b)2)
            # 2. x_s(c(a-c)2(a-tb)2) + ?s(a3(b-c)2) + ??s(a2c(a-b)2)
            # t = z_ / (-4) / x_
            # if x_*(t**2 - 2) <= y_:
            #     y = [
            #         x_,
            #         1 - x_,
            #         y_ - (t**2 - 2) * x_,
            #         w__
            #     ]
            #     if all(_ >= 0 for _ in y):
            #         y = [_ * coeff((3,2,0)) for _ in y]
            #         exprs = [
            #             CyclicSum(c*(a-c)**2*(a-t*b)**2),
            #             CyclicSum(b*c**2*(a-b)**2),
            #             CyclicSum(a**2*c*(a-b)**2),
            #             CyclicSum(a*b) * CyclicProduct(a)
            #         ]
            #         return sum_y_exprs(y, exprs)

            # -z/2 - 2tx >= 0
            # 1 + 2tx - x + z/2 >= 0
            # thus, (x - z/2 - 1)/(2x) <= t <= -z/(4x)
            # in this bound, we should make t as close to 1 as possible
            # to minimize t^2 - 2t - 2
            bound_l = (x_ - z_ / 2 - 1) / (2*x_)
            bound_r = -z_ / (4*x_)
            if bound_l <= bound_r:
                if bound_l <= 1 <= bound_r:
                    t = sp.S(1)
                elif abs(bound_l - 1) < abs(bound_r - 1):
                    t = bound_l
                else:
                    t = bound_r

                # the sum of following equals to the polynomial
                y = [
                    x_,
                    -z_ / 2 - 2*t*x_,
                    1 + 2*t*x_ - x_ + z_ / 2,
                    y_ - (t**2 - 2*t - 2)*x_ + z_ / 2,
                    w__
                ]
                # print('(l,r) =', (bound_l, bound_r), '  t =', t,'\ny =', y)
                if all(_ >= 0 for _ in y):
                    y = [_ * coeff((3,2,0)) for _ in y]
                    exprs = [
                        CyclicSum(c*(a-c)**2*(a-t*b)**2),
                        CyclicSum(a**3*(b-c)**2),
                        CyclicSum(b*c**2*(a-b)**2),
                        CyclicSum(a*c**2*(a-b)**2),
                        CyclicSum(a*b) * CyclicProduct(a)
                    ]
                    return sum_y_exprs(y, exprs)


    if True:
        # Try subtracting s(c(a2-ab-u(ac-ab)+v(bc-ab))2)
        # and the rest does not have s(ab^4) term.
        # In this case, we must have coeff((2,3,0))*coeff((3,2,0))*4 >= coeff((3,1,1))**2,
        # which is equivalent to det <= 0 where det =
        # 8*u^3-8*u^2*v+(4*y+4)*u^2+8*u*v^2+4*z*u*v+(-8*x-4*z-8)*u+(4*x+4)*v^2+(4*z+8)*v-4*x*y+z^2+4*z+4

        # It is a quadratic function of v, so WLOG v = symmetric axis = (2*u**2 - u*z - z - 2)/(2*(2*u + x + 1))
        # substitute back, the discriminant is factorizable and we require
        # det_u = 12*u**2 + (8*x + 8*y + 4*z + 16)*u + 4*x*y + 4*y - z**2 - 4*z - 4 >= 0

        # meanwhile coeff((3,2,0)) >= 0 requires u^2 <= x
        x_, y_, z_ = coeff((3,2,0)) / coeff((1,4,0)), coeff((2,3,0)) / coeff((1,4,0)), coeff((3,1,1)) / coeff((1,4,0))

        eq1 = 12*x_ + 4*x_*y_ + 4*y_ - (z_ + 2)**2
        sym1 = 8*x_ + 8*y_ + 4*z_ + 16
        if eq1 >= 0 or eq1**2 <= sym1**2 * x_:
            # det_u >= 0
            u_ = sp.sqrt(x_)
            if not isinstance(u_, sp.Rational):
                eq1 -= 12*x_
                det_u = lambda u: 12*u**2 + sym1*u + eq1
                for u__ in rationalize_bound(u_.n(20), direction = -1, compulsory = True):
                    if det_u(u__) >= 0 and u__**2 <= x_: # prevent u__ < 0 but u__^2 >= x_
                        u_ = u__
                        break
                else:
                    u_ = None
            if u_ is not None:
                v_ = (2*u_**2 - u_*z_ - z_ - 2)/(2*(2*u_ + x_ + 1))
                print(u_, v_, sym1, eq1)
                _new_coeffs = {
                    (3,2,0): x_ - u_**2,
                    (2,3,0): y_ + 2*u_ - v_**2,
                    (3,1,1): 2*u_*v_ - 2*u_ + 2*v_ + z_ + 2,
                    (2,2,1): coeff((2,2,1)) / coeff((1,4,0)) + ((u_ - v_)**2 - 2*v_ - 1)
                }
                solution = _sos_struct_quintic_windmill(Coeff(_new_coeffs))
                if solution is not None:
                    return solution * coeff((1,4,0)) + coeff((1,4,0)) * CyclicSum(c*(a**2+(u_-v_-1)*a*b-u_*a*c+v_*b*c)**2)




    u, x_, y_, z_, = sp.symbols('u'), coeff((1,4,0)) / coeff((3,2,0)), coeff((2,3,0)) / coeff((3,2,0)), coeff((3,1,1)) / coeff((3,2,0))
    u_, y__, z__, w__ = None, y_, z_, coeff((2,2,1)) / coeff((3,2,0)) + (x_ + y_ + z_ + 1)


    # now we have to higher the degree
    if True:
        # first try whether we can handle neat cases
        # Theorem 1.1:
        # f(a,b,c) = s(a(ac+b^2-2bc)^2(a-b)^2) = s(a^2-ab)s(a^4c-3a^3bc+2a^2b^2c) >= 0
        # g(a,b,c) = s(a(ab-2bc+c^2)^2(a-b)^2) = s(a^2-ab)s(a^3b^2-a^2b^2c) >= 0
        # h(a,b,c) = s(a(ac+b^2-2bc)(ab-2bc+c^2)(a-b)^2) = -s(a^2-ab)s(a^3bc-a^3c^2)
        # Then,
        # c1f(a,b,c) + c2h(a,b,c) + c3g(a,b,c) >= 0 holds when c2^2 <= 4c1c3

        # Case A.
        c1, c2, c3, c4 = coeff((1,4,0)), None, coeff((3,2,0)), sp.S(0)

        # Constraints on c2:
        # c2 <= coeff((2,3,0))
        # -3*c1 - c2 <= coeff((3,1,1))
        # Thus,
        # -3*c1 - coeff((3,1,1)) <= c2 <= coeff((2,3,0))
        # Also, c2^2 <= 4c1c3

        lb_, ub_ = -3*c1 - coeff((3,1,1)), coeff((2,3,0))
        if lb_ <= ub_:
            if ub_ < 0:
                c2 = ub_
            elif lb_ > 0:
                c2 = lb_
            else:
                c2 = sp.S(0)

        if c2 is not None and c2**2 <= 4*c1*c3:
            pp = (c3 - c2**2/(4*c1)) * b + (coeff((2,3,0)) - c2) * a
            pp = sp.together(pp).as_coeff_Mul()
            y = [
                2 * c1,
                pp[0],
                (coeff((3,1,1)) + 3*c1 + c2) / 2,
                w__ * coeff((3,2,0))
            ]
            if all(_ >= 0 for _ in y):
                multiplier = CyclicSum((a-b)**2)
                w = c2/(2*c1)
                exprs = [
                    CyclicSum(a*(a-b)**2*(w*a*b-(2+2*w)*b*c+w*c**2+a*c+b**2)**2),
                    CyclicSum(c**2*(a-b)**2*pp[1]) * multiplier,
                    CyclicProduct(a) * CyclicSum((a-b)**2)**2,
                    CyclicProduct(a) * CyclicSum(a*b) * multiplier
                ]
                return sum_y_exprs(y, exprs) / multiplier
        

        # Case B.
        # Assume we subtract c1(f(a,b,c) + wg(a,b,c))^2, then the remaining does not contain s(ab^4) term.
        # In this case, we must have coeff((2,3,0))*coeff((3,2,0))*4 >= coeff((3,1,1))**2
        w = sp.symbols('w')
        eq = (-(coeff((3,2,0)) - c1*w**2) * (coeff((2,3,0)) - 2*c1*w) * 4 + (coeff((3,1,1)) + c1*(3 + 2*w))**2).as_poly(w)
        eq2 = (coeff((2,3,0)) - 2*c1*w).as_poly(w)
        for (w_, interval_end), _ in (eq * eq2).intervals():
            if eq2(w_) >= 0 and eq(w_) <= 0:
                w = w_
                break
    
        if isinstance(w, sp.Rational):
            multiplier = CyclicSum((a-b)**2)
            _new_coeffs = {
                (3,2,0): coeff((3,2,0)) - c1*w**2,
                (2,3,0): coeff((2,3,0)) - 2*c1*w,
                (3,1,1): coeff((3,1,1)) + c1*(3 + 2*w),
                (2,2,1): coeff((2,2,1)) - c1*(2 - w**2),
            }

            solution = _sos_struct_quintic_windmill(Coeff(_new_coeffs))
            if solution is not None:
                return (2*c1*CyclicSum(a*(a-b)**2*(w*a*b-(2+2*w)*b*c+w*c**2+a*c+b**2)**2) + solution * multiplier) / multiplier



    # now we formally start
    # Case A.
    # u >= (v - 1)^2 / 4 + 1
    
    def _compute_yz(u_, v_):
        denom = u_**3 - u_**2 - u_*v_ + u_ + 1
        y__ = (-2*u_**2 + u_*v_**2 - u_*v_ + 2*u_ - v_ - 1)/denom
        z__ = (-2*u_**2*v_ + u_**2 + u_*v_ - v_**2)/denom
        return y__, z__


    eq = (u**5*x_**2 - u**4*x_**2 - 2*u**3*x_ + u**2*(-x_**2 - x_*y_ + x_) + u*(-x_*y_ - 4*x_ - y_ + 1) - x_ - y_ - 2).as_poly(u)
    criterion = lambda u, v: u >= (v-1)**2/4+1 or (2*u**2-u*v**2+u*v-2*u+v+1 <= 0 and u**3-u**2-u*v+u+1 >= 0)

    for root in sp.polys.roots(eq, cubics = False, quartics = False).keys():
        # first try rational u, v
        if isinstance(root, sp.Rational) and root > .999:
            u_ = root
            v_ = (u_**3*x_ - u_**2*x_ + u_*x_ - u_ + x_ + 1) / (u_*x_ + 1)
            if criterion(u_, v_):
                y__, z__ = _compute_yz(u_, v_)
                break
            u_ = None
    else:
        # Normal case: add a small perturbation so that we can obtain rational u and v
        if not ((y_ < 0) and (y_ ** 2 == 4 * x_)):
            for root in sp.polys.nroots(eq):
                if root.is_real and root > .999:
                    u_ = root
                    v_ = (u_**3*x_ - u_**2*x_ + u_*x_ - u_ + x_ + 1) / (u_*x_ + 1)
                    if criterion(u_, v_):
                        break
                    u_ = None

            if u_ is not None:
                # approximate a rational number
                direction = (u_**2*x_ - 1)*(3*u_**4*x_**2 + 2*u_**3*x_**2 + 4*u_**3*x_ - 3*u_**2*x_**2 + 3*u_**2*x_ - 6*u_*x_ - x_**2 + x_ - 3)
                direction = -1 if direction > 0 else 1
                u_numer = u_
                for u_ in rationalize_bound(u_numer, direction = direction, compulsory = True):                
                    # despite (v = the following formula) cancels out a^3b^2 and ab^4 terms perfectly
                    # the v is oftentimes too complicated
                    v_ = (u_**3*x_ - u_**2*x_ + u_*x_ - u_ + x_ + 1) / (u_*x_ + 1)
                    if criterion(u_, v_):
                        y__, z__ = _compute_yz(u_, v_)
                        if y__ <= y_ and z__ <= z_:
                            # re-rationalize v such that v is simpler
                            for v__ in rationalize_bound(u_.q * v_.n(20), direction = -1, compulsory = True):
                                # trick: keep the denominator of v and u aligned
                                v__ /= u_.q
                                # print(u_, v_.n(20), v__.n(20), v__)

                                if v__ <= v_ and criterion(u_, v__):
                                    y__, z__ = _compute_yz(u_, v__)
                                    x__ = (u_ + v__ - 1) / (u_**3 - u_**2 - u_*v__ + u_ + 1)
                                    if y__ <= y_ and (z_ - z__) / 2 + x_ - x__ >= 0:
                                        # use the simpler solution
                                        v_ = v__
                                        break
                            else:
                                # restore the complicated solution
                                # it is not expected to reach here
                                y__, z__ = _compute_yz(u_, v_)              

                            break
                    u_ = None

        else:
            # Case Special. when y_ < 0 and y_^2 = 4x_, then there is a root on the border
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
                z0_, z1_, z2_ = [rationalize(_, rounding = rounding, reliable = False) for _ in (z0, z1, z2)]
                m_, n_, p_, q_, det_ = compute_discriminant(z0_, z1_, z2_)
                if det_ >= 0 and m_ > 0:
                    break

            if det_ >= 0 and m_ > 0:
                u_ = 1 + (v_ - 1)**2/4
                multiplier = CyclicSum(a**2 + (z0_**2 + 2)*b*c)
                y = [sp.S(1), m_ / 2, det_ / 6 / m_, w__]
                y = [_ * coeff((3,2,0)) for _ in y]
                exprs = [
                    CyclicSum(a*(a**2*b+(1-r)*a*b**2-r*b**3+(r*z0_)*a**2*c-z0_*a*c**2+z1_*b**2*c+z2_*b*c**2-((2-z0_)*(1-r)+z1_+z2_)*a*b*c)**2),
                    CyclicProduct(a) * CyclicSum((a**2-b**2-(p_ + 2*q_)/3/m_*(a*b-a*c)-(q_ + 2*p_)/3/m_*(b*c-a*b))**2),
                    CyclicProduct(a) * CyclicSum(a**2*(b-c)**2),
                    CyclicProduct(a) * CyclicSum(a*b) * CyclicSum(a**2+(z0_**2 + 2)*a*b)
                ]

                return sum_y_exprs(y, exprs) / multiplier

            u_ = None


    # Now we call the solver to solve the problem at (u, v).
    main_solution = _sos_struct_quintic_windmill_uv(u_, v_)

    if main_solution is not None:
        multiplier, main_solution = main_solution

        denom = u_**3 - u_**2 - u_*v_ + u_ + 1
        u, v = u_, v_
        _new_coeffs = {
            (2,3,0): y_ + (2*u**2-u*v**2+u*v-2*u+v+1) / denom,
            (1,4,0): x_ - (u_ + v_ - 1) / denom,
            (3,1,1): z_ + (2*u**2*v-u**2-u*v+v**2) / denom,
            (2,2,1): coeff((2,2,1)) / coeff((3,2,0)) - (-u**3 + 2*u**2*v + 2*u**2 - u*v**2 + u*v - 4*u + v**2 + 1) / denom
        }

        rest_solution = _sos_struct_quintic_windmill(Coeff(_new_coeffs))
        if rest_solution is not None:
            ker = coeff((3,2,0)) / 2 / denom
            return (ker * main_solution + coeff((3,2,0)) * multiplier * rest_solution) / multiplier

    return None


def _sos_struct_quintic_windmill_uv(u, v):
    """
    Given (u,v), solve inequality
    f(a,b,c) = s((b-a+(2u-1)c)(a^2-b^2+u(ab-ac)+v(bc-ab))^2) >= 0.

    Return the multiplier g and the result f*g.

    Reference
    -------
    [1] https://tieba.baidu.com/p/6472739202
    """
    if u is None or not isinstance(u, sp.Rational) or u + v < 1:
        return None

    if u > 1 and 2*u**2 - u*v - 2*u + 3 >= 0 and u**2 + u*v - u - v**2 + v - 1 >= 0:
        # Theorem 2.0
        # When 2u^2-uv-2u+3 >= 0 and u^2+uv-u-v^2+v-1 >= 0, the inequality holds.
        # Because it yields better solution than Theorem 1, we use it first.

        # This is because
        # (u - 1)*(u + v - 1) * F1 + (2*u**2 - u*v - 2*u + 3) * F2 + (-u**2 + u*v + 3*u - v**2 + 3*v - 5) * F3
        # = (2u - v + 1)/2 * s(a^2-ab) * s((b-a+(2u-1)c)(a^2-b^2+u(ab-ac)+v(bc-ab))^2)
        # where
        # F1 = s(a(b-c)^2(a^2-b^2+u(ab-ac)+v(bc-ab))^2)
        # F2 = s(a(a-b)^2(b^2-c^2+u(bc-ba)+v(ca-bc))^2)
        # F3 = s(a(a-b)(b-c)(a^2-b^2+u(ab-ac)+v(bc-ab))(b^2-c^2+u(bc-ba)+v(ca-bc)))

        c1 = (u - 1)*(u + v - 1)
        c2 = (2*u**2 - u*v - 2*u + 3)
        c3 = (-u**2 + u*v + 3*u - v**2 + 3*v - 5)
        
        multiplier = CyclicSum((a-b)**2)
        denom2 = ((2*u - v + 1) / 4)

        y = [
            c1 / denom2,
            (c2 - c3**2 / (4*c1)) / denom2
        ]

        p1 = (b-c)*(a**2-b**2+u*(a*b-a*c)+v*(b*c-a*b))
        p2 = (a-b)*(b**2-c**2+u*(b*c-a*b)+v*(c*a-b*c))
        exprs = [
            CyclicSum(a * ((p1 + c3/(2*c1)*p2).expand())**2),
            CyclicSum(a * p2**2)
        ]

        return multiplier, sum_y_exprs(y, exprs)


    if u >= (v - 1)**2/4 + 1:
        # Theorem 1 (Main Theorem)
        # When u >= (v - 1)^2/4 + 1, the inequality holds.

        multiplier = CyclicSum(a**2 + (u+v+1)*b*c)

        p1 = (-u*v + u + 2)*a**2*b + (-2*u + v**2 + 3)*a**2*c + (-u*v + u - v + 1)*a*b**2 \
            + (-4*u**2 + 4*u*v + 2*u - v**2 - 3*v)*a*b*c + (2*u**2 + u*v - 3*u - v - 1)*a*c**2 + (-v - 1)*b**3 \
            + (2*u**2 - u*v + 3*u + v**2 + 2*v - 3)*b**2*c + (-2*u*v - 2*u - v**2 + 4*v - 1)*b*c**2
        p2 = u*a**2*b + (-v - 1)*a**2*c + (u - 1)*a*b**2 + (-2*u + v)*a*b*c + (u + 1)*a*c**2 - b**3 + (-u + v + 1)*b**2*c + (1 - v)*b*c**2

        exprs = [
            CyclicSum(a * p1**2),
            CyclicSum(a * p2**2),
            CyclicProduct(a) * CyclicSum((a*a-b*b+u*(a*b-a*c)+v*(b*c-a*b))**2)
        ]

        y = [
            sp.S(1) / 2,
            (4*u - v*v + 2*v - 5) / 2,
            (u + v + 2) * (4*u + v - 4)
        ]

        return multiplier, sum_y_exprs(y, exprs)
    
    
    if (3*u**2 + 2*u*v - 4*u - v**2 - 8) >= 0:
        # Theorem 2.1
        # When 3u^2 + 2uv - 4u - v^2 - 8 >= 0, the inequality holds.

        multiplier = CyclicSum(a*b)

        denom = u**3 - u**2 - u*v + u + 1
        denom2 = 2 * (u+1)**2 * denom
        g = -(u**2 - u*v + 2*u + 2)/(2*u**3 + u**2*v - u*v**2 - u + v + 2)

        p1 = a**2*c*(-u*v + 1) + a*b**2*(u**2*v - u) + a*b*c*(u**3 - u**2*v - u**2 + u*v + u - v) + b**2*c*(-u**3 - 1) + b*c**2*(u**2 + v)
        p2 = a**2*c + a*b**2*(g*u*v - g) + a*b*c*(g*u**2 - g*u - g*v**2 + g*v + u - v) + a*c**2*(-g*u*v + g - u) + b**2*c*(-g*u**2 - g*v - 1) + b*c**2*(g*u + g*v**2 + v)

        exprs = [
            CyclicSum(a * p1**2),
            CyclicSum(a * p2**2),
            CyclicProduct(a) * CyclicSum((a*a-b*b+u*(a*b-a*c)+v*(b*c-a*b))**2)
        ]

        y = [
            (3*u**2 + 2*u*v - 4*u - v**2 - 8) / denom2,
            (2*u**3 + u**2*v - u*v**2 - u + v + 2)**2 / denom2,
            u + v - 1
        ]

        return multiplier, sum_y_exprs(y, exprs)

    if u*u - u*v + 2*u + 2 < 0:
        # Theorem 2.2 (very ugly)
        # When (u,v) is in the following region, the inequality holds:
        # u^2-uv+2u+2 < 0
        # 3*u^7-u^6*v-2*u^6-u^5*v^2+u^5*v-5*u^5+3*u^4*v^2-4*u^4*v-u^3*v^2+u^3*v+2*u^3+2*u^2*v^2+8*u^2*v-10*u^2+u*v^2-3*u*v-9*u-v <= 0
        # 6*u^6-2*u^5*v+11*u^5-2*u^4*v^2-3*u^4*v+11*u^4+u^3*v^2-5*u^3*v+13*u^3-2*u^2*v^2-12*u^2*v+18*u^2-2*u*v^2+20*u+2*v+6 <= 0

        # Although the region is complicated, it handles almost all cases with very large v.

        w = (3*u**5 - u**4*v + 4*u**4 - u**3*v**2 + u**3 - 2*u*v + 2*u + 2)/(u**2*(u**2 - u*v + 2*u + 2))
        if w >= -1:
            multiplier = CyclicSum(a**2 + w*b*c)

            p3 = a**2*c*(-u*v + 1) + a*b*c*(u**3 - u**2*v - u**2 + u*v**2 + u*v + u - 2*v) + b**3*(u*v - 1) + b**2*c*(-u**3 + u**2*v - u*v**2 - u + v - 1) + b*c**2*(u**2 - u*v + v + 1)
            p4 = a**2*c*(-u*v + 1) + a*b**2*(u**2*v - u) + a*b*c*(u**3 - u**2*v - u**2 + u*v + u - v) + b**2*c*(-u**3 - 1) + b*c**2*(u**2 + v)

            exprs = [
                CyclicSum(a*(u*a**2*b+(-v-1)*a**2*c+(u-1)*a*b**2+(-2*u+v)*a*b*c+(u+1)*a*c**2-b**3+(-u+v+1)*b**2*c+(1-v)*b*c**2)**2),
                CyclicSum(a*c**2*(a**2-b**2+u*(a*b-a*c)+v*(b*c-a*b))**2),
                CyclicSum(a * p3**2),
                CyclicSum(a * p4**2),
                CyclicProduct(a) * CyclicSum((a*a-b*b+u*(a*b-a*c)+v*(b*c-a*b))**2)
            ]

            y = [
                2 * (u**3 - u**2 - u*v + u + 1) / u**2,
                2 * (u**5*w - 3*u**5 - u**4*w + u**4 + u**3*v**2 - u**3*v*w + 2*u**3*v + u**3*w - 2*u**3 - u**2*v + u**2*w - 3*u**2 + u*v - u - 1) / u**4,
                2 * (u + 1) / (u**2 * (u*v - 1)),
                2 * (3*u**4*v - u**4 - u**3*v**2 + 3*u**3*v - 6*u**3 - u**2*v**3 + u**2*v**2 + 3*u**2*v - 6*u**2 + 2*u*v**2 - u*v - 3*u - v)/(u**3*(u*v - 1)*(u**2 - u*v + 2*u + 2)),
                (6*u**6 - 2*u**5*v + 11*u**5 - 2*u**4*v**2 - 3*u**4*v + 11*u**4 + u**3*v**2 - 5*u**3*v + 13*u**3 - 2*u**2*v**2 - 12*u**2*v + 18*u**2 - 2*u*v**2 + 20*u + 2*v + 6)/(u**2*(u**2 - u*v + 2*u + 2)),
            ]

            if all(_ >= 0 for _ in y):
                return multiplier, sum_y_exprs(y, exprs)



def _sos_struct_quintic_uncentered(coeff):
    """
    Give the solution to s(ab4+?a2b3-?a3bc+?a2b2c) >= 0, 
    which might have equality not at (1,1,1) but elsewhere.

    Examples
    -------
    s(2ab4+5a2b3-17a3bc+13a2b2c)

    s(ab4+347/30a2b3-3833/230a3bc+475/69a2b2c)

    s(ab4+0a2b3-19/4a3bc+9/2a2b2c)

    s(2ab4+4a2b3-13a3bc+7a2b2c)

    s(a2c(a-b)(a+c-4b))

    s(a2c(a-5/2b)2)-abcs(8a2-131/4ab)

    Reference
    -------
    [1] https://artofproblemsolving.com/community/u426077h2242759p21856167

    [2] https://artofproblemsolving.com/community/u861323h3019177p27134161

    [3] https://tieba.baidu.com/p/7710460926
    """

    u, v = sp.symbols('u v')
    t = coeff((1,4,0))

    r1, r2, r3, u_, v_ = coeff((2,3,0)) / t, coeff((3,1,1)) / t, coeff((2,2,1)) / t, None, None
    r1_, r2_ = r1, r2
    rem = coeff((2,3,0)) + coeff((1,4,0)) + coeff((3,1,1)) + coeff((2,2,1))
    if rem < 0:
        return None

    if True:
        y = [coeff((2,3,0)), coeff((1,4,0)), coeff((3,1,1)) / 2 + coeff((1,4,0)), rem]
        if all(_ >= 0 for _ in y):
            exprs = [
                CyclicSum(a**2*b*(b-c)**2),
                CyclicSum(a*b**2*(b-c)**2),
                CyclicSum((b-c)**2) * CyclicProduct(a),
                CyclicSum(a*b) * CyclicProduct(a)
            ]
            return sum_y_exprs(y, exprs)

    if t <= 0:
        # the case t == 0 should already be handled above
        return None

    if True:
        # Theorem 1.
        # f(a,b,c) = s(a(ac+b^2-2bc)^2(a-b)^2) = s(a^2-ab)s(a^4c-3a^3bc+2a^2b^2c) >= 0
        # g(a,b,c) = s(a(ac+b^2-2bc)^2(a-c)^2) = s(a^2-ab)s(a^3c^2-a^2b^2c) >= 0
        # h(a,b,c) = s(a(ac+b^2-2bc)^2(a-b)(a-c)) = s(a^2-ab)^2 * abc
        # This implies that f(a,b,c) - 2xh(a,b,c) + x^2g(a,b,c) >= 0.
        # This gives very beautiful solution, if valid.
        x = max(sp.S(0), (r2 + 3) / (-2))
        y = [
            sp.S(2),
            r1 - x**2,
            (r2 - (-3 - 2*x)) / 2,
            rem / t
        ]
        if all(_ >= 0 for _ in y):
            multiplier = CyclicSum((a-b)**2)
            y = [_ * t for _ in y]
            exprs = [
                CyclicSum(a*(a*c+b**2-2*b*c)**2*((1-x)*a - b + x*c)**2),
                CyclicSum(a*c**2*(a-b)**2) * multiplier,
                CyclicProduct(a) * CyclicSum((a-b)**2)**2,
                CyclicSum(a*b) * CyclicProduct(a) * multiplier
            ]
            return sum_y_exprs(y, exprs) / multiplier


    if rem == 0:
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


    elif rem >= 0:

        if r1 == 0 and r2 == -13:
            # The only exception where u, v are irrational but coefficients are rational
            # is s(a2c(a-5/2b)2)-abcs(8a2-131/4ab)
            # in this case u = 1/2, v = (17+3*sqrt(13))/4
            # So we use the direct result:
            # (s(a2c(a-5/2b)2)-abcs(8a2-131/4ab))s(ab(a-b+3c)2) = 
            # s(a(a^3c-8a^2bc-a^2c^2+10ab^2c-4abc^2-b^2c^2+3bc^3)^2)
            # + 1/2p(a)s((9a2c-2ab2-26abc-8ac2+b3-7b2c+5bc2+c3)2) + 3/2p(a)s(a2b-4a2c-6abc)2
            if r3 >= 39:
                multiplier = CyclicSum(a*b*(a-b+3*c)**2)
                y = [
                    t,
                    t / 2,
                    t * 3 / 2,
                    (r3 - 39) * t
                ]
                exprs = [
                    CyclicSum(a*c**2*(a**3 - 8*a**2*b - a**2*c + 10*a*b**2 - 4*a*b*c - b**2*c + 3*b*c**2)**2),
                    CyclicProduct(a) * CyclicSum((9*a**2*c - 2*a*b**2 - 26*a*b*c - 8*a*c**2 + b**3 - 7*b**2*c + 5*b*c**2 + c**3)**2),
                    CyclicProduct(a) * CyclicSum(a**2*b - 4*a**2*c - 6*a*b*c)**2,
                    CyclicSum(a*b) * CyclicProduct(a) * multiplier
                ]
                return sum_y_exprs(y, exprs) / multiplier
            return None


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
            for root in nroots(eq, real = True):
                if root >= 2:
                    v_ = root
                    break

            # TODO: handle normal cases

            if False and v_ is not None:
                # approximate a rational number
                v_numer = v_
                for tol in (.3, .1, 3e-2, 3e-3, 3e-4, 3e-5, 3e-7, 3e-9):
                    v_ = rationalize(v_numer - tol * 3, rounding = tol)
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

        multiplier = CyclicSum(a**2 + phi*b*c)
        
        y = [
            sp.S(1) / m2**2,
            r1_ * factor1**2 / (((u**2 - u + 1) * factor2 * m1)**2),
            (u + v - 1) * (u**3 - u*u - u*v + u + 1) * (phi + 1) / (u*v - 1) / (u*u - 2*u - v + 2)
        ]
        y.append((r2_ + phi + 2*(u**3*w - u + v + w) - y[-1]) / 2)
        y.append(1 if (r1 != r1_ or r2 != r2_) else 0)

        y = [_ * t for _ in y]

        exprs = [
            CyclicSum(a*(m2*(-u*v*w+w)*a**2*c + m2*(u**2*v*w-u*w-u)*a*b**2 + m2*(u**3*w-u**2*v*w-u**2*w+u*v*w+u*w-v*w+v)*a*b*c\
                +m2*b**3 + m2*(-u**3*w+u-v-w)*b**2*c + m2*(u**2*w+v*w-1)*b*c**2)**2),
            CyclicSum(a*(m1*a**2*c + m1*(u*v*z-z)*a*b**2 + m1*(u**2*z-u*z+u-v**2*z+v*z-v)*a*b*c + m1*(-u*v*z-u+z)*a*c**2\
                +m1*(-u**2*z-v*z-1)*b**2*c + m1*(u*z+v**2*z+v)*b*c**2)**2),
            CyclicProduct(a) * CyclicSum(a**2 - rho*a*b)**2,
            CyclicProduct(a) * CyclicSum((a**2 - b**2 - u*a*c + v*b*c + (u-v)*a*b)**2),
            CyclicSum(a*c*(a-b)**2*((r1-r1_)*c + (r2-r2_)/2*b)) * multiplier
        ]

        return sum_y_exprs(y, exprs) / multiplier

    return None


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

    Reference
    -------
    [1] https://tieba.baidu.com/p/7710460926
    """
    t = coeff((1,4,0))
    w =  - coeff((3,1,1)) / t
    if w > 3 and w**3 - 8*w*w + 39*w - 83 > 0:
        return None
    elif w <= 2:
        # very trivial in this case
        return t * CyclicSum(a**2*c*(a-b)**2) + (2 - w) * t / 2 * CyclicSum((a-b)**2) * CyclicProduct(a)

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

        multiplier = CyclicSum(a**2 + (z_**2 + 2)*b*c)

        m_ = (-w + 2*x_ + z_**2 + 2)
        y = [
            sp.S(1),
            m_ / 2,
            det__ / 6 / m_
        ]
        y = [_ * t for _ in y]

        exprs = [
            CyclicSum(c*(a**3 - x_*a**2*b + y_*a*b**2 - z_*b*c**2 + a**2*c + (x_-y_+z_-2)*a*b*c)**2),
            CyclicProduct(a) * CyclicSum((a**2 - b**2 + (-(p_+q_*2)/3/m_)*(a*b-a*c) + (-(2*p_+q_)/3/m_)*(b*c-a*b))**2),
            CyclicProduct(a) * CyclicSum(a**2*(b-c)**2)
        ]

        return sum_y_exprs(y, exprs) / multiplier
    
    return None


def _sos_struct_quintic_hexagon(coeff):
    """
    Try solving quintics without s(a^5).

    Theorem 1.
    When Det = 64*p^3-16*p^2*q^2+8*p*q*z^2+32*p*q*z-256*p*q+64*q^3-z^4-24*z^3-192*z^2-512*z >= 0,
    we have f(a,b,c) = s(a^4b + pa^3b^2 + qa^2b^3 + ab^4 + za^3bc - (p+q+z+2)a^2b^2c) >= 0.

    Proof: Take w = z + 8, u = 4(2p^2-qw)/(w^2-4pq), v = 4(2q^2-pw)/(w^2-4pq),
    and t = (4pq-w^2)^2/(8(2p+2q+w)(3(p-q)^2+(w-p-q)^2)),
    then 1 - t = Det/... >= 0, and we have
    f(a,b,c) = t\sum c(a^2-b^2+u(ab-ac)+v(bc-ab))^2 + (1 - t)\sum c(a-b)^4 >= 0.

    Examples
    -------
    s(c(a-b)2(a+b-3c)2)

    s(a4b+a4c+6a3b2+a2b3-9a3bc)-10abcs(a2-ab)

    s(4a4b+4a4c-8a3b2-29a3bc+8a3c2+21a2b2c)

    s(a4b+a4c+5a3b2+3a2b3-10a2b2c)-20s(a3bc-a2b2c)

    s(4a4b+a4c+9a3b2-36a3bc+2a3c2+20a2b2c)

    s(4a4b+a4c-3a3b2-16a3bc+2a3c2+12a2b2c)
    
    s(a4b-3a3b2+6a2b3+3ab4-7a3bc)

    s(a4b+7a4c-3a3b2-17a3bc-a3c2+13a2b2c)

    2s(a2(a-b)(a2-3bc))-s((a+b-c)(a-b)2(a+b-3/2c)2)

    s((23(b-a)+31c)(a2-b2+(ab-ac)+2(bc-ab))2)

    Reference
    -------
    [1] https://artofproblemsolving.com/community/u426077h2246130p17263853
    """
    if coeff((5,0,0)) != 0 or coeff((4,1,0)) < 0 or coeff((1,4,0)) < 0:
        return None
    if coeff((4,1,0)) == 0 or coeff((1,4,0)) == 0:
        return _sos_struct_quintic_windmill(coeff)

    rem = radsimp(sum(coeff(_) for _ in [(4,1,0),(3,2,0),(2,3,0),(1,4,0),(3,1,1),(2,2,1)]))
    if rem < 0:
        return None


    if coeff((4,1,0)) == coeff((1,4,0)):
        # Theorem 1.
        # In this case, there must exist u, v such that
        # f / coeff((4,1,0)) >= s(c(a^2-b^2+u(ab-ac)+v(bc-ab))^2)
        coeff410 = coeff((4,1,0))

        p, q, z = coeff((3,2,0)) / coeff410, coeff((2,3,0)) / coeff410, coeff((3,1,1)) / coeff410
        p, q, z = radsimp([p, q, z])

        det = radsimp(64*p**3 - 16*p**2*q**2 + 8*p*q*z**2 + 32*p*q*z - 256*p*q + 64*q**3 - z**4 - 24*z**3 - 192*z**2 - 512*z)
        if det >= 0:
            if p != q or w != p + q:
                # Actually the case p == q will be handled in quintic_symmetric, so we ignore it here.
                w = z + 8
                denom = radsimp(1 / (8*(2*p + 2*q + w)*(3*(p-q)**2 + (w-p-q)**2)))
                y = radsimp([denom * coeff410, det * denom * coeff410, rem])
                c1, c2, c3 = radsimp([w**2-4*p*q, 4*(2*p**2-q*w), 4*(2*q**2-p*w)])
                exprs = [
                    CyclicSum(c*(c1*(a**2-b**2) + c2*(a*b-a*c) + c3*(b*c-a*b))**2),
                    CyclicSum(c*(a-b)**4),
                    CyclicSum(a**2*b**2*c)
                ]
                return sum_y_exprs(y, exprs)

        if z >= -2:
            det = radsimp(-16*(p**2*q**2 + 18*p*q - 4*p**3 - 4*q**3 - 27))
            if det >= 0 and (p != 3 or q != 3):
                u, v = (2*p**2 - 6*q) / (9 - p*q), (2*q**2 - 6*p) / (9 - p*q)
                t = (9 - p*q)**2 / (p + q + 3) / (3*(p - q)**2 + (6 - p - q)**2)
                u, v, t = radsimp([u, v, t])

                y = radsimp([
                    (z + 2) * coeff410 / 2,
                    t * coeff410,
                    (1 - t) * coeff410,
                    rem
                ])
                exprs = [
                    CyclicProduct(a) * CyclicSum((a-b)**2),
                    CyclicSum(c*(a**2 - b**2 + u*(a*b-a*c) + v*(b*c-a*b))**2),
                    CyclicSum(c*(a-b)**4),
                    CyclicSum(a**3*b*c)
                ]
                return sum_y_exprs(y, exprs)

        if not coeff.is_rational:
            return None

        # backup plan:
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
                rem / coeff410
            ]
            if all(_ >= 0 for _ in y):
                y = [_ * coeff410 for _ in y]
                exprs = [
                    CyclicSum(c*(a**2 - b**2 + u_*(a*b-a*c) + v_*(b*c-a*b))**2),
                    CyclicSum(a**2*b*(b-c)**2),
                    CyclicProduct(a) * CyclicSum((b-c)**2),
                    CyclicProduct(a) * CyclicSum(a*b)
                ]
                return sum_y_exprs(y, exprs)

        # The case must have been solved above.
        # If not, the problem is false.
        return None


    if True:
        # solve trivial case:
        # Try to represent f(a,b,c) in the form
        # xs(c(a^2-b^2+u(ab-ac)+v(bc-ab))^2) + zs(c(a-b)^4) - ys(c(a^2-b^2+u(ab-ac)+v(bc-ab))(a-b)^2)
        # Then, when y^2 <= 4xz, the result is positive semi-definite.
        y_ = (coeff((4,1,0)) - coeff((1,4,0))) / 2
        s_ = (coeff((4,1,0)) + coeff((1,4,0))) / 2 # s = x + z
        x_ = s_ / 2 + sp.sqrt(coeff((4,1,0)) * coeff((1,4,0))) / 2
        g, h = coeff((3,2,0)), coeff((2,3,0))
        if not isinstance(x_, sp.Rational):
            x_ = x_.n(20)

        # u^2x - 2vx - vy = g => v = (u^2x - g) / (2x + y)
        # v^2x - 2ux + uy = h


        def _compute_uvcoef(u_, x_, y_, s_, g):
            # coef is the coefficient of a^3bc
            v_ = (u_**2*x_ - g) / (2*x_ + y_)
            coef = -2*u_*v_*x_ - 2*u_*y_ + 2*v_*y_ - 8*(s_ - x_)
            return u_, v_, coef

        def _solve_uvcoef(x_, y_, g, h, s_):
            u = sp.symbols('u')
            u_, v_ = None, None
            eq = (x_**3*u**4 - 2*g*x_**2*u**2 + (-8*x_**3 - 4*x_**2*y_ + 2*x_*y_**2 + y_**3)*u + g**2*x_ - 4*h*x_**2 - 4*h*x_*y_ - h*y_**2).as_poly(u)

            if isinstance(x_, sp.Rational):
                roots = sp.polys.roots(eq, cubics = False, quartics = False)
                for root in roots:
                    if isinstance(root, sp.Rational):
                        u_, v_, coef = _compute_uvcoef(root, x_, y_, s_, g)
                        if coef <= coeff((3,1,1)):
                            return u_, v_, coef
            
            roots = nroots(eq, real = True, nonnegative = True)
            for root in roots:
                u_, v_, coef = _compute_uvcoef(root, x_, y_, s_, g)
                if coef <= coeff((3,1,1)):
                    return u_, v_, coef
            return None

        params = _solve_uvcoef(x_, y_, g, h, s_)
        if params is not None and not isinstance(x_, sp.Rational):
            # perturb x to be slightly smaller
            for x__ in rationalize_bound(x_, direction = -1, compulsory = True):
                if x__ < s_ / 2:
                    continue
                params = _solve_uvcoef(x__, y_, g, h, s_)
                if params is not None:
                    x_ = x__
                    break
                params = None

        if params is not None:
            # now that x is rational
            u_, v_, coef = params
            
            if not isinstance(u_, sp.Rational):
                # for u__ in rationalize_bound(u_, )
                direction = -1 if (4*u_*x_**2*(-g + u_**2*x_)/(2*x_ + y_)**2 - 2*x_ + y_) > 0 else 1
                for u__ in rationalize_bound(u_, direction = direction, compulsory = True):
                    u__, v_, coef = _compute_uvcoef(u__, x_, y_, s_, g)
                    if v_**2*x_ - 2*u_*x_ + u_*y_ <= h and coef <= coeff((3,1,1)):
                        u_ = u__
                        break

            # print(x_, params, u_, v_, (u_**2*x_ - 2*v_*x_ - v_*y_), (v_**2*x_ - 2*u_*x_ + u_*y_))
            if isinstance(u_, sp.Rational):
                z_ = s_ - x_
                y = [
                    x_,
                    z_ - y_**2 / 4 / x_,
                    g - (u_**2*x_ - 2*v_*x_ - v_*y_),
                    h - (v_**2*x_ - 2*u_*x_ + u_*y_),
                    (coeff((3,1,1)) - (-2*u_*v_*x_ - 2*u_*y_ + 2*v_*y_ - 8*(s_ - x_))) / 2,
                    rem
                ]
                if all(_ >= 0 for _ in y):
                    exprs = [
                        CyclicSum(c*((1-y_/(2*x_))*a**2-(1+y_/(2*x_))*b**2+(y_/x_+u_-v_)*a*b + v_*b*c -u_*a*c)**2),
                        CyclicSum(c*(a-b)**4),
                        CyclicSum(b*c**2*(a-b)**2),
                        CyclicSum(a*c**2*(a-b)**2),
                        CyclicProduct(a) * CyclicSum((a-b)**2),
                        CyclicProduct(a) * CyclicSum(a*b)
                    ]
                    return sum_y_exprs(y, exprs)

            

    return None
