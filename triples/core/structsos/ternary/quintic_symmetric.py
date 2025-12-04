import sympy as sp

from .quartic import sos_struct_quartic
from .utils import (
    Coeff, CommonExpr,
    sum_y_exprs, nroots, rationalize, rationalize_bound
)

def sos_struct_quintic_symmetric(coeff: Coeff, real = True):
    """
    The function solves symmetric quintic problems with s(a^5) term in an
    incomplete attempt.

    For symmetric quintics with equality at (1,1,1), they are determined by their
    symmetric axis, F(a,1,1). Without loss of generality we assume
    F(a,1,1) = a^3 + xa^2 + ya + z
    and F(a,b,c) = s((a^3 + xa^2(b+c)/2 + yabc + zbc(b+c)/2)(a-b)(a-c)).

    The border of F is equivalent to F(a,1,0)/a^2 = 2*u**2 + (x - 4)*u + (z - 2*x) where u = a + 1/a.
    F(a,1,0) >= 0 on R+ iff D = (x+4)**2 - 8*z <= 0 or (x >= -4 and z >= 0).

    Theorem: For t <= 3, we have
    F(a,b,c) = s((a+b-c)(a-b)^2(a+b-tc)^2) >= 0.
    Proof: F(a,b,c)s((a-b)^2) = s(c(2c^3-2abc+(t-1)(a^2c+b^2c-a^2b-ab^2)-(t+1)(ac^2+bc^2-2abc))^2)
                                 + (3-t)(t+1)s(a)p(a-b)^2 >= 0

    For t >= 3, (things become much more complicated) let t = (3u^2+2u+3)/(4u) where u >= 3,
    we have
    F(a,b,c) = s((a+b - (u^2+6u-9)/(2u^2)c)(a-b)^2(a+b-tc)^2)/2 >= 0.
    Proof: Denote
    g(a,b,c) = ((3-u)/(4u)(a+b) - c)(a^2+b^2+c^2 - (t+1)/2c(a+b) + (t-2)ab) - 9(u-1)^2(u+1)(u+3)/(32u^3)ab(2c-a-b).
    Then, F(a,b,c) * s(a^2 - (7u^2+30u-9)/(16u^2)ab) = sum cg(a,b,c)^2 + 27(u-3)(u^2-u+2)/(32u^3)*sum (a-b)^2(a+b-tc)^2 >= 0.


    Examples
    --------
    2(s(a5+2ab(a3+b3)+4a2b2(a+b)-13a3bc)-262/10abcs(a2-ab))

    s(a5-1/2ab(a3+b3)+5a2b2(a+b)-10a2b2c-22abc(a2-ab))

    s((a+b+10c)(a-b)2(a+b-5c)2)+s((a+b)(a-b)2(a+b-3c)2)

    s(a5+4a4b+4a4c-39a3bc+30a2b2c)

    s(100a5-90a4b-90a4c-6a3b2+31a3bc-6a3c2+61a2b2c)

    s(a5-a2b2c-2ab(a3+b3-2abc)+a2b2(a+b-2c)+4abc(a2-ab))

    s(3(a+b-4/5c)(a-b)2(a+b-3/2c)2)

    s((a+b-5/4c)(a-b)2(a+b-3/2c)2)+s((a+b-2/3c)(a-b)2(a+b-5/3c)2)

    s((a+b-c)(a-b)2(a+b-3/2c)2)+s((a+b-1/2c)(a-b)2(a+b-c)2)

    s((2a2-11ab+2b2)2(a+b)-4a5-94a3bc+396abc(a2-ab))

    s((a+b-31/32c)(a-b)2(a+b-59/16c)2)

    s(625a5-2725a4b-2725a4c+2389a3b2+11132a3bc+2389a3c2-11085a2b2c)

    s((a+b-7/8c)(a-b)2(a+b-5c)2)     (HINT: s(c(a+b-2c)2(a2-10ab+5ac+b2+5bc-2c2)2))

    s(a3)s(a)2+s(ab)p(a+b)*(27+18sqrt(3))/8-(3/8+(27+18sqrt(3))/24)p(a+b)s(a)2

    s((a+b-(1+sqrt(15))/5c)(a-b)2(a+b-(4sqrt(15)/5+1/2)c)2)

    s(a(b-c)2(b+c-a)2)+s(a3(a-b-c)2)

    References
    ----------
    [1] https://math.stackexchange.com/questions/3809195/proving-q-fraca3b3c3abbccak-cdot-fracabbccaa
    """
    if not (coeff((4,1,0)) == coeff((1,4,0)) and coeff((3,2,0)) == coeff((2,3,0))) or coeff((5,0,0)) < 0:
        return None

    if coeff((5,0,0)) == 0:
        return _sos_struct_quintic_symmetric_hexagon(coeff)

    # if not coeff.is_rational:
    #     # algorithm down here uses poly.intervals() and does not support irrational coefficients
    #     return None

    # first determine how much abcs(a2-ab) can we subtract from the poly

    m, z, u, v = coeff((5,0,0)), coeff((4,1,0)), coeff((3,2,0)), coeff((3,1,1))
    z, u, v = [z / m, u / m, v / m]
    rem = (1 + v + (z + u)*2) * m + coeff((2,2,1))
    if rem < 0:
        return None

    a, b, c = coeff.gens
    CyclicSum, CyclicProduct = coeff.cyclic_sum, coeff.cyclic_product

    if True:
        # Try subtracting some s(a(a2-xab-xac-yb2-yc2+(2x+2y-1)bc)2) so that
        # the rest is positive. See criterion at _sos_struct_quintic_symmetric_hexagon.
        def _criterion(x):
            denom = 2*(-4*u - v + 4*x**2 - 8*x - 8*z - 4)
            if denom == 0:
                return None
            y = (-(x - 1)*(-2*u - v + 4*x**2 - 12*x - 8*z - 2) / denom)
            z2 = 2*x - y**2 + z
            if z2 < 0:
                return None
            u2 = u - x**2 - 2*x*y + 2*y
            v2 = v - 2*x**2 + 8*x*y - 4*x + 8*y**2 - 8*y + 2
            if (z2 > 0 and ((2*u2 + v2)**2 + 8*v2*z2) <= 0) or (z2 == 0 and u2 >= 0 and 2*u2 + v2 >= 0):
                return y
            return None

        x = sp.symbols('x')
        # We shall have det1 <= 0
        det1 = (16*x**4 - 32*x**3 + (-16*u - 8*v - 32*z)*x**2 + (-16*u + 8*v - 16)*x + 4*u**2 + 4*u*v + 8*u + v**2 + 8*v*z + 4*v + 4).as_poly(x)
        det2 = (-4*u - v + 6*x**2 - 12*x - 8*z - 2).as_poly(x)

        y_ = None
        # first check whether there exists common roots
        det_gcd = sp.gcd(det1, det1.diff())
        if det_gcd.degree() == 1:
            x_ = -det_gcd.coeff_monomial((0,)) / det_gcd.coeff_monomial((1,))
            y_ = _criterion(x_)

        if y_ is None:
            if coeff.is_rational:
                # only rational polynomials are supported by intervals()
                det = det1 * det2
                intervals = sp.polys.polytools.intervals(det)
                if len(intervals):
                    for interval in intervals[:-1]:
                        x_ = interval[0][1]
                        y_ = _criterion(x_)
                        if y_ is not None:
                            break
                    else:
                        y_ = None
            else: # not coeff.is_rational
                for x_ in nroots(det1, method = 'sympy', real = True, nonnegative = True):
                    y_ = _criterion(x_)
                    if y_ is not None:
                        y_ = None
                        direction = 1 if det1.diff()(x_) <= 0 else -1
                        for x__ in rationalize_bound(x_, direction = direction, compulsory = True):
                            y__ = _criterion(x__)
                            if y__ is not None:
                                x_, y_ = x__, y__
                                break
                    if y_ is not None:
                        break
                    y_ = None

        if y_ is not None:
            u2 = u - x_**2 - 2*x_*y_ + 2*y_
            v2 = v - 2*x_**2 + 8*x_*y_ - 4*x_ + 8*y_**2 - 8*y_ + 2
            _new_coeffs = {
                (4,1,0): m*(2*x_-y_**2+z), (1,4,0): m*(2*x_-y_**2+z),
                (3,2,0): m*u2, (2,3,0): m*u2,
                (3,1,1): m*v2,
                (2,2,1): (coeff((2,2,1)) + m*(4*x_**2 - 4*x_*y_ - 6*y_**2 + 4*y_ - 1))
            }
            solution = _sos_struct_quintic_symmetric_hexagon(Coeff(_new_coeffs, is_rational = coeff.is_rational))
            if solution is not None:
                solution = solution + m * CyclicSum(
                    a*(a**2 - x_*a*b - x_*a*c - y_*b**2 - y_*c**2 + (2*x_ + 2*y_ - 1)*b*c)**2
                )
                return solution

    # real start below
    if z >= -1:
        if u + 1 + z >= 0 and v + 3 + 4*z + 2*u >= 0:
            # everything is trivial if (u,v) is on the upper-right side of (-1-z, -1-2z), the
            # leftmost point of the cubic where t = 1.
            y = [
                m / 2 if 2*z + 1 >= 0 else sp.S(0),

                # when 2z + 1 < 0, use Schur: s((a+b-c)(a-b)2(a+b-c)2) = s(a3(a-b)(a-c))
                m if 2*z + 1 < 0 else sp.S(0),
                m * (z+1) if 2*z + 1 < 0 else sp.S(0),

                (u + 1 + z) * m,
                (v + 3 + 4*z + 2*u) * m / 2,
                rem
            ]
            exprs = [
                CyclicSum((a+b+(2*z+1)*c)*(a-b)**2*(a+b-c)**2),
                CommonExpr.schur(5),
                CyclicSum(c*(a-b)**2*(a+b-c)**2),
                CyclicSum(a**3*(b-c)**2),
                CyclicProduct(a) * CyclicSum((a-b)**2),
                CyclicProduct(a) * CyclicSum(a*b)
            ]
            return sum_y_exprs(y, exprs)

        if True:
            # very trivial case:
            # linear combination of s(a3(b-c)2), abcs(a2-ab), s((a+b+(2z-1)/3c))(a-b)2(a+b+(2z-1)/3c)2)
            c1 = coeff((3,2,0)) - m * (z + 1)*(4*z**2 + 8*z - 23)/27
            y = [
                m / 2,
                c1,
                (coeff((3,1,1)) + 2 * c1 + m * (2*z - 1)**3/27) / 2,
                rem
            ]
            if all(_ >= 0 for _ in y):
                # the third equation is equivalent to
                # -6*u - 3*v + 4*z**2 - 4*z - 5 <= 0
                exprs = [
                    CyclicSum((a+b+(2*z-1)/3*c)*(a-b)**2*(a+b+(2*z-1)/3*c)**2),
                    CyclicSum(a**3*(b-c)**2),
                    CyclicProduct(a) * CyclicSum((a-b)**2),
                    CyclicProduct(a) * CyclicSum(a*b)
                ]
                return sum_y_exprs(y, exprs)

        # assume s((a+b+(2z-1+2t)c)(a-b)2(a+b-tc)2)/2 <= poly / coeff((5,0,0))
        # this requires (u,v) is above the curve (t^3+(z-2)t^2+(1-2z)t-1, -2t^3+(1-2z)t^2)
        # the curve is a singular cubic with node t = (1-2z)/3
        # hence, when (u,v) is above the curve, it is often
        # a linear combination with node and another point, we can do secant method

        # theorem: 2t + 2z + 1 > 0 iff the point is below x = 0

        # 1. when z >= 1/2, the node is below x = 0. Both intersections must below the
        # x = 0. Thus 2t + 2z + 1 >= 0 automatically holds.

        # 2. when -1 <= z <= 1/2, (2z - 1 + 2t) in [-1,0] where t = (1-2z)/3, so the node
        # is positive but shall higher degree. Also, dv/dt < 0 when t > (1-2z)/3, so we can
        # assume the second intersection has t > (1-2z)/3. It is also positive.

        t1 = (1 - 2*z) / 3
        t2 = -(2*u*z - u + v*z + 4*v + 2*z - 1)/(-6*u - 3*v + 4*z**2 - 4*z - 5)
        y1 = -t1**2*(2*t1 + 2*z - 1)
        y2 = -t2**2*(2*t2 + 2*z - 1)

        if y1 != y2:
            # weight of linear combination
            w1 = (v - y2) / (y1 - y2)
            w2 = 1 - w1
            # print('(w1, w2) =', (w1, w2), '\n(y1, y2) =', (y1, y2), '\n(t1, t2) =', (t1, t2))
            if 0 <= w1 <= 1:
                if (w1 == 0 and y2 <= 0) or (y1 <= 0 and y2 <= 0):
                    y = [
                        w1 * m / 2,
                        w2 * m / 2,
                        rem
                    ]
                    exprs = [
                        CyclicSum((a + b + (2*z - 1 + 2*t1)*c)*(a - b)**2*(a + b - t1*c)**2),
                        CyclicSum((a + b + (2*z - 1 + 2*t2)*c)*(a - b)**2*(a + b - t2*c)**2),
                        CyclicProduct(a) * CyclicSum(a*b)
                    ]
                    return sum_y_exprs(y, exprs)
                else:
                    # higher degree
                    # we must have t1 = (1 - 2z) / 3 <= 1
                    multiplier = CyclicSum((a - b)**2)/2
                    if y2 <= 0: # 2z - 1 + 2t_2 >= 0
                        # note that:
                        # s((a+b-c)(a-b)2(a+b-tc)2)s(a2-ab)
                        # = s(a(a-b)(a-c))s((b-c)2(b+c-ta)2) + 2(2-t)(t+1)s(a)p(a-b)2 >= 0
                        y = [
                            w1 * m / 2,
                            w1 * m * (2 - t1) * (t1 + 1),
                            w1 * m * (z + t1) / 2,
                            w2 * m / 4,
                            rem / 2
                        ]
                        exprs = [
                            CyclicSum(a*(a-b)*(a-c)) * CyclicSum((b-c)**2*(b+c-t1*a)**2),
                            CyclicSum(a) * CyclicProduct((a-b)**2),
                            CyclicSum((a-b)**2) * CyclicSum(c*(a - b)**2*(a + b - t1*c)**2),
                            CyclicSum((a-b)**2) * CyclicSum((a + b + (2*z - 1 + 2*t2)*c)*(a - b)**2*(a + b - t2*c)**2),
                            CyclicSum((a-b)**2) * CyclicProduct(a) * CyclicSum(a*b)
                        ]
                        return sum_y_exprs(y, exprs) / multiplier
                    else: # 2z - 1 + 2t_2 < 0, in this case t < 2
                        y = [
                            w1 * m / 2,
                            w2 * m / 2,
                            (w1 * (2 - t1) * (t1 + 1) + w2 * (2 - t2) * (t2 + 1)) * m,
                            w1 * m * (z + t1) / 2,
                            w2 * m * (z + t2) / 2,
                            rem / 2
                        ]
                        exprs = [
                            CyclicSum(a*(a-b)*(a-c)) * CyclicSum((b-c)**2*(b+c-t1*a)**2),
                            CyclicSum(a*(a-b)*(a-c)) * CyclicSum((b-c)**2*(b+c-t2*a)**2),
                            CyclicSum(a) * CyclicProduct((a-b)**2),
                            CyclicSum((a-b)**2) * CyclicSum(c*(a - b)**2*(a + b - t1*c)**2),
                            CyclicSum((a-b)**2) * CyclicSum(c*(a - b)**2*(a + b - t2*c)**2),
                            CyclicSum((a-b)**2) * CyclicProduct(a) * CyclicSum(a*b)
                        ]
                        return sum_y_exprs(y, exprs) / multiplier



        # we assert the problem cannot be solved otherwise
        # but the inequalit might be true if it does not have root (1,1,1)
        return None

    elif -3 <= z <= -1:
        # things are getting weird
        # the node of the cubic is not positive now and we cannot apply the secant trick
        0
        # however, we can use barycentric coordinates as we have three points on the cubic
        # t1 = -z, t2 = ..., t3 = \infty
        # so that the weighted sum of s((a+b+(2z-1+2t)c)(a-b)2(a+b-tc)2)/2 for these three points
        # equals to the polynomial

        if True:
            # trivial case, where (u,v) is over the asymptotic line from (-1-z, z^2)
            # which is a linear combination of s((a+b-c)(a-b)2(a+b+zc)2), s(a3(b-c)2) and abcs(a2-ab)
            y = [
                m / 4,
                (3 + z) * (1 - z) * m / 4,
                (u + z + 1) * m / 2,
                (v - z**2 + 2*(u + z + 1)) * m / 4,
                rem / 2
            ]
            if all(_ >= 0 for _ in y):
                multiplier = CyclicSum((a - b)**2)/2
                exprs = [
                    CyclicSum(c*(a-c)**2*(b-c)**2*((z+1)*(a+b) + 2*c)**2),
                    CyclicSum(a) * CyclicProduct((a-b)**2),
                    CyclicSum((a-b)**2) * CyclicSum(a**3*(b-c)**2),
                    CyclicProduct(a) * CyclicSum((a-b)**2)**2,
                    CyclicSum((a-b)**2) * CyclicProduct(a) * CyclicSum(a*b),
                ]

                return sum_y_exprs(y, exprs) / multiplier

        t1 = -z
        t2 = -(2*u*z - u + v*z + 4*v + 2*z - 1)/(-6*u - 3*v + 4*z**2 - 4*z - 5)

        u1 = -z - 1
        u2 = (t2**3 + t2**2*z - 2*t2**2 - 2*t2*z + t2 - 1)
        u3 = 1
        v1 = (z**2)
        v2 = (-t2**2*(2*t2 + 2*z - 1))
        v3 = -2

        # w1 + w2 = 1
        # w1u1 + w2u2 + w3u3 = u
        # w1v1 + w2v2 + w3v3 = v
        w1 = (u*v3 - u2*v3 - u3*v + u3*v2)/(u1*v3 - u2*v3 - u3*v1 + u3*v2)
        w2 = 1 - w1
        w3 = u - w1*u1 - w2*u2

        y = [
            w1 * m / 4,
            w2 * m / 4,
            (w1 * (3 + z) * (1 - z) + w2 * (3 - t2) * (1 + t2)) * m / 4,
            w2 * (z + t2) * m / 2,
            w3 * m / 2,
            rem / 2
        ]
        if all(_ >= 0 for _ in y):
            multiplier = CyclicSum((a - b)**2)/2
            exprs = [
                CyclicSum(c*(a-c)**2*(b-c)**2*((1-t1)*(a+b) + 2*c)**2),
                CyclicSum(c*(a-c)**2*(b-c)**2*((1-t2)*(a+b) + 2*c)**2),
                CyclicSum(a) * CyclicProduct((a-b)**2),
                CyclicSum((a-b)**2) * CyclicSum(c*(a-b)**2*(a+b-t2*c)**2),
                CyclicSum((a-b)**2) * CyclicSum(a**3*(b-c)**2),
                CyclicProduct(a) * CyclicSum((a-b)**2) * CyclicSum(a*b)
            ]
            return sum_y_exprs(y, exprs) / multiplier

        return None

    else: # if z <= -3
        # In this case there might be roots on the border.
        du = u - (z**2 + 2*z + 5)/4 # ensure the border is positive
        if du < 0:
            return None

        if True:
            # a trivial but beautiful case
            y = [
                m,
                m * du / 2,
                m * (v - (z**2 - 6*z - 9)/2 + 2*du) / 4,
                rem / 2
            ]
            if y[-2] >= 0:
                multiplier = CyclicSum((a - b)**2)/2
                exprs = [
                    CyclicSum(c*(a**2*c+b**2*c+c**3-(1-z)/2*a*c**2-(1-z)/2*b*c**2-(2*z+3)*a*b*c+(z+1)/2*a**2*b+(z+1)/2*a*b**2)**2),
                    CyclicSum(c**3*(a-b)**2) * CyclicSum((a-b)**2),
                    CyclicSum((a-b)**2)**2 * CyclicProduct(a),
                    CyclicSum((a-b)**2) * CyclicProduct(a) * CyclicSum(a*b)
                ]
                return sum_y_exprs(y, exprs) / multiplier


        # Now we formally start.
        # Note that (u,v) should be over the following curve: (t >= 3)
        # x = (27*t**6 + 36*t**5*z - 18*t**5 - 48*t**4*z + 69*t**4 + 24*t**3*z - 92*t**3 - 48*t**2*z + 69*t**2 + 36*t*z - 18*t + 27)/(64*t**3)
        # y = (-27*t**6 - 36*t**5*z - 36*t**5 - 48*t**4*z - 93*t**4 - 88*t**3*z - 72*t**3 - 48*t**2*z - 93*t**2 - 36*t*z - 36*t - 27)/(32*t**3)
        # Then,
        # dx/dt = 3*(t - 1)*(t + 1)*(3*t**2 - 2*t + 3)*(9*t**2 + 8*t*z + 2*t + 9)/(64*t**4)
        # dy/dt = -3*(t - 1)*(t + 1)*(3*t**2 + 2*t + 3)*(9*t**2 + 8*t*z + 2*t + 9)/(32*t**4)
        # We see that dy/dx < 0. So we can solve a point on the curve that is below (u,v).

        t, t_ = sp.symbols('t'), None
        eqt1 = (27*t**6 + 36*t**5*z - 18*t**5 - 48*t**4*z + 69*t**4 + 24*t**3*z - 92*t**3 - 48*t**2*z + 69*t**2 + 36*t*z - 18*t + 27) - u*(64*t**3)
        eqt2 = (-27*t**6 - 36*t**5*z - 36*t**5 - 48*t**4*z - 93*t**4 - 88*t**3*z - 72*t**3 - 48*t**2*z - 93*t**2 - 36*t*z - 36*t - 27) - v*(32*t**3)
        eqt1, eqt2 = eqt1.as_poly(t), eqt2.as_poly(t)

        for t_ in nroots(eqt1, method = 'factor', real = True, nonnegative = True):
            if t_ >= 3 and eqt2(t_) <= 1e-8:
                break
        else:
            return None
        u_ = t_

        if du == 0:
            # Case a. There is a root on the border. We shall preserve the root.
            r = (1 - z)/2
            def _compute_mpnq_discriminant(x, y):
                m_ = -2*r**2 - 8*r*y**2 - 4*r*y + v - 4*x*y + 4*x + 9*y**2 + 6*y + 3
                p_ = 2*r**2*y**2 + r**2 - 4*r*x - 4*r*y**2 - 20*r*y + 6*r + v*y**2 + 2*v*y - 2*v - x**2 + 10*x*y - 4*x + 3*y**2 + 24*y - 9
                n_ = -2*r**2*y**2 + 12*r**2*y - 12*r*x*y + 4*r*x + 8*r*y**2 - 8*r*y - 12*r - v*y**2 - 2*v*y + 3*v - 2*x**2 + 4*x*y + 8*x - 9*y**2 - 6*y + 15
                det = 3*m_*(m_ + n_) - 3*p_**2
                return (m_, p_, n_), det
            def _verify_xy(x, y):
                if -2 < y < 0:
                    return False
                (m_, p_, n_), det = _compute_mpnq_discriminant(x, y)
                if (m_ > 0 and det >= 0) or (m_ == 0 and 2*p_ + n_ >= 0):
                    return True
                return False

            # We shall find x, y such that m >= 0 and det >= 0 and y(y+2) >= 0.
            # We first compute the optimal x, y through the exact theorem.
            w0, w1 = [(3-u_)/(4*u_), 9*(u_-1)**2*(u_+1)*(u_+3)/(32*u_**3)]
            y0, x0 = -w0, -r*w0 - w1

            if _verify_xy(x0, y0):
                if not isinstance(u_, sp.Rational):
                    x0, y0 = x0.n(20), y0.n(20)
                    for rounding in (.5, .1, 1e-2, 1e-3, 1e-5, 1e-8, 1e-12):
                        x_, y_ = rationalize(x0, rounding=rounding, reliable = False), rationalize(y0, rounding=rounding, reliable=False)
                        if _verify_xy(x_, y_):
                            x0, y0 = x_, y_
                            break

                multiplier = CommonExpr.quadratic(1, y0**2 + 2*y0 - 1)
                func_g = (c**2 - r*b*c + b**2 - (2-r)*a*b)*(c+y0*b) + (c**2 - r*a*c + a**2 - (2-r)*a*b)*(c+y0*a) - c**3 + a*b*c + x0*(a**2*b + a*b**2 - 2*a*b*c)
                func_g = func_g.expand()
                main_solution = m * CyclicSum(c * func_g**2)

                m_, p_, n_ = _compute_mpnq_discriminant(x0, y0)[0]
                m_, p_, n_ = [m*m_, m*p_, m*n_]
                quartic = {
                    (4,0,0): m_, (3,1,0): p_, (2,2,0): n_, (1,3,0): p_, (2,1,1): -m_-p_*2-n_
                }
                quartic_solution = sos_struct_quartic(Coeff(quartic), None)
                solution = main_solution + (quartic_solution + rem * CyclicSum(a*b) * multiplier) * CyclicProduct(a)
                return solution / multiplier

        if du == 0 and coeff.is_rational:
            # this cannot happen
            return None

        if True:
            # Case b. If there is root on the symmetric axis. We shall preserve the root.
            sym = (a**3 + (2*z + 2)*a**2 + (2*u + v + 4*z + 3)*a + 2*u + 2*z + 2).as_poly(a)
            sym_diff = sym.diff(a)
            sym_gcd = sp.polys.gcd(sym, sym_diff)

            if sym_gcd.degree() == 1:
                root = -sym_gcd.coeff_monomial((0,)) / sym_gcd.coeff_monomial((1,))
                t = root + 1
                w = 2*z - 1 + 2*t
                extra_w = sp.S(0)

                if du == 0 and not coeff.is_rational:
                    # This might happen when the exact u is irrational
                    # and we can solve it from (3u^2+2u+3) - 4ut = 0
                    u_ = (2*t + 2*sp.sqrtdenest(sp.sqrt(t**2 - t - 2)) - 1)/3
                    if u_ < 3:
                        return None

                w_lower_bound = (-(u_**2 + 6*u_ - 9)/(2*u_**2))
                if w < w_lower_bound:
                    return None

                # F(a,b,c) = s((a+b+(2z-1+2t)c)(a-b)^2(a+b-tc)^2)/2

                def _compute_coeffs(x, y, w):
                    # u: the multiplier is s(a^2 + u*b*c)
                    # v: coefficient of p(a-b)^2s(a)
                    # q: coefficient of s(c(a-b)^2(tab - (t-1)c(a+b))^2)
                    u = (4*t**2*x**2 - 4*t**2*x + t**2 - 8*t*x**2 - 8*t*x*y + 12*t*x + 4*t*y - 4*t - 2*w + 4*x**2 + 8*x*y - 16*x + 4*y**2 + 2)/4
                    v = (2*t**2*w + 4*t**2*x**2 - 8*t**2*x*y + 4*t**2*y - 4*t*u - 4*t*w + 8*t*x**2 - 28*t*x + 8*t*y**2 + 12*t*y + 10*t + 2*u*w + 2*u - 16*x**2 + 12*x + 4*y + 3)/4
                    q = -(-2*t**2*u*w - 2*t**2*u - 2*t**2*w + 4*t**2*x**2 - 16*t**2*x*y - 4*t**2*x + 4*t**2*y**2 + 12*t**2*y - t**2 + 4*t*u*w + 4*t*w + 8*t*x**2 + 24*t*x*y \
                            - 24*t*x - 8*t*y + 14*t + 4*u - 4*v - 2*w - 16*x**2 - 16*x*y + 20*x + 12*y + 1)/(4*(t - 1)**2)
                    ind = (8*t**2*w*x**2 - 8*t**2*w*x + 2*t**2*w - 4*t**2*x**2 + 4*t**2*x - t**2 - 16*t*w*x**2 - 16*t*w*x*y + 24*t*w*x + 8*t*w*y \
                            - 8*t*w + 56*t*x**2 - 40*t*x*y - 36*t*x + 20*t*y + 4*t - 4*w**2 + 8*w*x**2 + 16*w*x*y - 32*w*x + 8*w*y**2 - 2*w - 68*x**2 + 24*x*y + 28*y**2 + 16*y + 10)/4
                    return u, v, q, ind

                def _verify_xy(x, y, w):
                    u, v, q, ind = _compute_coeffs(x, y, w)
                    if u >= -1 and v >= 0 and q >= 0 and ind >= 0:
                        return True

                # We first compute the optimal x, y through the exact theorem.
                w_ = (3*u_**3 + u_**2 + 9*u_ - 9)/(4*u_**2)
                w0, w1 = [(3-u_)/(4*u_), 9*(u_-1)**2*(u_+1)*(u_+3)/(32*u_**3)]
                x0 = (t**2 - 2*t - 2*w0*w_ - 2*w1 + 1)/(2*t**2 - 6*t + 4)
                y0 = (-2*t*w0 - 2*w0*w_ + 4*w0 - 2*w1 + 1)/(2*t - 4)

                # print('(x,y) =', (x0, y0), ' coeffs =', _compute_coeffs(x0, y0, w))
                if not _verify_xy(x0, y0, w):
                    # It is expected that _verify_xy(x0, y0) is valid when w -> w_lower_bound,
                    # we can subtract more w to do the approximation
                    for w__ in rationalize_bound(w_lower_bound, direction = 1, compulsory = True):
                        if w__ <= w and _verify_xy(x0, y0, w__):
                            extra_w = w - w__
                            w = w__
                            break
                    else:
                        return None


                if (not isinstance(u_, sp.Rational)) and (coeff.is_rational or du != 0):
                    x0, y0 = x0.n(20), y0.n(20)
                    for rounding in (.5, .1, 1e-2, 1e-3, 1e-5, 1e-8, 1e-12):
                        x_, y_ = rationalize(x0, rounding=rounding, reliable = False), rationalize(y0, rounding=rounding, reliable=False)
                        if _verify_xy(x_, y_, w):
                            x0, y0 = x_, y_
                            break

                c1, c2, c3, ind = _compute_coeffs(x0, y0, w)
                multiplier = CommonExpr.quadratic(1, c1)
                func_g = x0*((b+c-t*a)*(c-b)**2+(c+a-t*b)*(a-c)**2) + (sp.Rational(1,2)-x0)*(b+c-t*a)*(c+a-t*b)*(2*c-a-b) + y0*(a+b-t*c)*(a-b)**2
                func_g = func_g.expand()

                y = [
                    m,
                    m * c2,
                    m * c3,
                    m * extra_w / 2,
                    m * ind / 2,
                    rem
                ]
                exprs = [
                    CyclicSum(c * func_g**2),
                    CyclicSum(a) * CyclicProduct((a-b)**2),
                    CyclicSum(c*(a-b)**2*(t*a*b - (t-1)*a*c - (t-1)*b*c)**2),
                    multiplier * CyclicSum(c*(a-b)**2*(a+b-t*c)**2),
                    CyclicProduct(a) * CyclicSum((a-b)**2*(a+b-t*c)**2),
                    multiplier * CyclicProduct(a) * CyclicSum(a*b)
                ]
                return sum_y_exprs(y, exprs) / multiplier


        if True:
            # Case c. If it is strictly > 0 other than the centroid. Then it is easy to make a perturbation.

            if not isinstance(t_, sp.Rational):
                direction = 1 if 9*t_**2 + 8*t_*z + 2*t_ + 9 <= 0 else -1
                for t__ in rationalize_bound(t_, direction = direction, compulsory = True):
                    if t__ >= 3 and eqt1(t__) <= 0 and eqt2(t__) <= 0:
                        t_ = t__
                        break

            u_ = t_
            t_ = (3*u_**2 + 2*u_ + 3) / (4*u_)
            w_ = (3*u_**3 + u_**2 + 9*u_ - 9)/(4*u_**2)
            sub = (u_**2 + 6*u_ - 9)/(2*u_**2)
            extra = 2*w_ + 2*z
            # F(a,b,c) >= s((a+b + (extra - sub)c)(a-b)^2(a+b-tc)^2) / 2

            if extra >= sub:
                m_inv_u = -m / (64*u_**3)
                y = [
                    m / 2,
                    m_inv_u * eqt1(u_),
                    m_inv_u * eqt2(u_),
                    rem
                ]
                if any (_ < 0 for _ in y):
                    return None
                exprs = [
                    CyclicSum((a + b + (extra - sub)*c)*(a-b)**2*(a+b-t_*c)**2),
                    CyclicSum(a**2*(b+c)*(b-c)**2),
                    CyclicProduct(a) * CyclicSum((a-b)**2),
                    CyclicProduct(a) * CyclicSum(a*b)
                ]
                return sum_y_exprs(y, exprs)

            elif extra >= 0:
                w0, w1 = [(3-u_)/(4*u_), 9*(u_-1)**2*(u_+1)*(u_+3)/(32*u_**3)]
                func_g = ((w0*(a+b) - c)*(a**2+b**2+c**2 - (w_+1)/2*c*(a+b) + (w_-2)*a*b) - w1*a*b*(2*c-a-b)).expand()
                multiplier = CommonExpr.quadratic(1, -(7*u_**2+30*u_-9)/(16*u_**2))
                m_inv_u = -m / (64*u_**3)

                pw1, pw2 = m_inv_u * eqt1(u_), m_inv_u * eqt2(u_)
                if pw1 < 0 or pw2 < 0:
                    return None

                p1 = sp.Add(*[
                    (extra * m / 2) * CyclicSum(c*(a-b)**2*(a+b-t_*c)**2),
                    pw1 * CyclicSum(a**2*(b+c)*(b-c)**2),
                    pw2 * CyclicProduct(a) * CyclicSum((a-b)**2),
                    rem * CyclicProduct(a) * CyclicSum(a*b)
                ]).together().as_coeff_Mul()

                y = [
                    m,
                    27*(u_ - 3)*(u_**2 - u_ + 2)/u_**3/32 * m,
                    p1[0]
                ]

                exprs = [
                    CyclicSum(c * func_g**2),
                    CyclicProduct(a) * CyclicSum((a-b)**2*(a+b-t_*c)**2),
                    multiplier * p1[1]
                ]
                return sum_y_exprs(y, exprs) / multiplier

    return None



def _sos_struct_quintic_symmetric_hexagon(coeff: Coeff):
    """
    Prove symmetric quintic without s(a5). It should also handle
    the case when (1,1,1) is not a root.

    Theorem: Suppose s(ab(a3+b3) + ua2b2(a+b) + va3bc + za2b2c) >= 0.
    1. When (u,v) is on the upper-right of parametric parabola (t^2-2t, -2t^2),
    which is also (2x+y)^2 = -8y, then we can use the fact that any point on the parabola
    is equivalent to s(c(a-b)2(a+b-tc)2).

    2. When (u,v) is on the upper-right of parametric parabola
    (24t(t-1)/(5t+1)^2, -48(t^2+4t+1)/(5t+1)^2),
    which is also (2x+y)^2 = 8(36x-7y-48), then we can use the fact that any point on the parabola
    is equivalent to s(a(b-c)2((7t-1)a2+(5t+1)(b2+c2-ab-ac)+(5t-11)bc)2).

    Examples
    -------
    s(2c(a-b)2(a+b-3c)2)

    s(c(a-b)2(a+b-3c)2)+s(c(a-b)2(a+b-4c)2)

    s(a(a+b)(a+c))s(ab)-p(a)(12s(ab)+(9+4sqrt(2))s(a2-ab))
    """
    m = coeff((4,1,0))
    if m < 0:
        return None

    rem = (m + coeff((3,2,0))) * 2 + coeff((3,1,1)) + coeff((2,2,1))
    if rem < 0:
        return None

    a, b, c = coeff.gens
    CyclicSum, CyclicProduct = coeff.cyclic_sum, coeff.cyclic_product

    if True:
        # very trivial as a combination of s(a(b-c)4), s(a3(b-c)2), abcs(a2-ab)
        # this corresponds to the region
        # u >= 0 and 8 + 2u + v >= 0
        y = [
            m,
            coeff((3,2,0)),
            m * 4 + coeff((3,2,0)) + coeff((3,1,1)) / 2,
            rem
        ]
        if all(_ >= 0 for _ in y):
            exprs = [
                CyclicSum(a*(b-c)**4),
                CyclicSum(a**3*(b-c)**2),
                CyclicProduct(a) * CyclicSum((a-b)**2),
                CyclicProduct(a) * CyclicSum(a*b)
            ]
            return sum_y_exprs(y, exprs)

    if m == 0:
        return None

    u, v = [coeff((3,2,0)) / m, coeff((3,1,1)) / m]
    if u >= -1 and v >= -2:
        # is strictly larger than the point (-1,-2) where t = -1
        y = [
            m,
            (u + 1) * m,
            (v + 2*u + 4) * m / 2,
            rem
        ]
        exprs = [
            CyclicSum(a*(b-c)**2*(b+c-a)**2),
            CyclicSum(a**3*(b-c)**2),
            CyclicProduct(a) * CyclicSum((a-b)**2),
            CyclicProduct(a) * CyclicSum(a*b)
        ]
        return sum_y_exprs(y, exprs)


    if (2*u + v)**2 + 8*v <= 0:
        # linear combination of t1 = 2 and t2 = ...? on parabola (t^2-2t, -2t^2)
        k = (v + 8) / u
        t2 = -4/(k + 2)
        x2 = t2**2 - 2*t2

        w2 = u / x2 # (u - 0) / (x2 - 0)
        w1 = 1 - w2
        if x2 != 0 and 0 <= w1 <= 1:
            y = [
                w1 * m,
                w2 * m,
                rem
            ]
            exprs = [
                CyclicSum(a*(b-c)**4), # s(a(b-c)2(b+c-2a)2) == s(a(b-c)4)
                CyclicSum(a*(b-c)**2*(b+c-t2*a)**2),
                CyclicProduct(a) * CyclicSum(a*b)
            ]
            return sum_y_exprs(y, exprs)

    return None
    if (2*u + v)**2 - 8*(36*u - 7*v - 48) <= 0:
        k = (v + 8) / u
        t2 = 5/(3*k - 19)
        x2 = 24*t2*(t2-1)/(5*t2+1)**2

        w2 = u / x2 # (u - 0) / (x2 - 0)
        w1 = 1 - w2
        if 0 <= w1 <= 1:
            multiplier = CyclicSum((a - b)**2)/2
            y = [
                w1 * m / 2,
                w2 * m,
                rem / 4
            ]
            exprs = [
                CyclicSum(a*(b-c)**4) * CyclicSum((a-b)**2),
                CyclicSum(a*(b-c)**2*((7*t2-1)*a**2 + (5*t2+1)*(b**2+c**2-a*b-a*c) + (5*t2-11)*b*c)**2),
                CyclicProduct(a) * CyclicSum((a-b)**2)**2
            ]
            return sum_y_exprs(y, exprs) / multiplier

    # idea:
    # if there is no root on symmetric axis, then there is no root at all -> subtract anything
    # has root on symmetric axis -> subtract s(c(a-b)2(a+b-xc)2) until another root appears

    return None
