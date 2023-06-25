import sympy as sp

from .utils import CyclicSum, CyclicProduct, _sum_y_exprs

a, b, c = sp.symbols('a b c')

def sos_struct_quintic_symmetric(poly, coeff, recurrsion):
    """
    
    Examples
    --------
    2(s(a5+2ab(a3+b3)+4a2b2(a+b)-13a3bc)-262/10abcs(a2-ab))

    s(a5-1/2ab(a3+b3)+5a2b2(a+b)-10a2b2c-22abc(a2-ab))

    s((a+b+10c)(a-b)2(a+b-5c)2)+s((a+b)(a-b)2(a+b-3c)2)

    s(a5+4a4b+4a4c-39a3bc+30a2b2c)

    s(100a5-90a4b-90a4c-6a3b2+31a3bc-6a3c2+61a2b2c)

    s(a5-a2b2c-2ab(a3+b3-2abc)+a2b2(a+b-2c)+4abc(a2-ab))
    """
    if not (coeff((4,1,0)) == coeff((1,4,0)) and coeff((3,2,0)) == coeff((2,3,0))):
        return None

    if coeff((5,0,0)) <= 0:
        # this case is degenerated and shall be handled in the non-symmetric case
        # TODO: add support for this case to present pretty solution
        return _sos_struct_quintic_symmetric_degenerate(coeff)

    # first determine how much abcs(a2-ab) can we subtract from the poly
    
    m, z, u, v = coeff((5,0,0)), coeff((4,1,0)), coeff((3,2,0)), coeff((3,1,1))
    z, u, v = z / m, u / m, v / m
    rem = (1 + v + (z + u)*2) * m + coeff((2,2,1))
    if rem < 0:
        return None

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
                CyclicSum(a**3*(a-b)*(a-c)),
                CyclicSum(c*(a-b)**2*(a+b-c)**2),
                CyclicSum(a**3*(b-c)**2),
                CyclicProduct(a) * CyclicSum((a-b)**2),
                CyclicProduct(a) * CyclicSum(a*b)
            ]
            return _sum_y_exprs(y, exprs)

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
                return _sum_y_exprs(y, exprs)

        # assume s((a+b+(2z-1+2t)c)(a-b)2(a+b-tc)2)/2 <= poly / coeff((5,0,0))
        # this requires (u,v) is above the curve (t^3+(z-2)t^2+(1-2z)t-1, -2t^3+(1-2z)t^2)
        # the curve is a singular cubic with node t = (1-2z)/3
        # hence, when (u,v) is above the curve, it is often
        # a linear combination with node and another point, we can do secant method

        # theorem: 2t + 2z + 1 > 0 iff the point is below x = 0

        # 1. when z >= 1/2, the node is below x = 0. Both intersections must below the 
        # x = 0. Thus 2t + 2z + 1 >= 0 automatically holds.

        # 2. when -1 <= z <= 1/2, (2z - 1 + 2t) \in [-1,0] where t = (1-2z)/3, so the node
        # is positive but shall higher degree. Also, dv/dt < 0 when t > (1-2z)/3, so we can 
        # assume the second intersection has t > (1-2z)/3. It is also positive.

        t1 = (1 - 2*z) / 3
        t2 = -(2*u*z - u + v*z + 4*v + 2*z - 1)/(-6*u - 3*v + 4*z**2 - 4*z - 5)
        y1 = -t1**2*(2*t1 + 2*z - 1)
        y2 = -t2**2*(2*t2 + 2*z - 1)

        # weight of linear combination
        w1 = (v - y2) / (y1 - y2)
        w2 = 1 - w1
        # print(w1, w2, y1, y2, t1, t2)
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
                return _sum_y_exprs(y, exprs)
            else:
                # higher degree
                # we must have t1 = (1 - 2z) / 3 <= 1
                multiplier = CyclicSum(a**2 - b*c)
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
                    return _sum_y_exprs(y, exprs) / multiplier
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
                    return _sum_y_exprs(y, exprs) / multiplier


        
        # we assert the problem cannot be solved otherwise
        # but the inequalit might be true if it does not have root (1,1,1)
        return None

    elif -3 <= z <= -1:
        # things are getting weird
        # the node of the cubic is not positive now and we cannot apply the secant trick
        0
        # however, we can use barycentric coordinates as we have three points on the cubic
        # t1 = -z, t2 = ..., t3 = \infty

        if True:
            # trivial case, where (u,v) is over the asymptotic line from (-1-z, z^2)
            # which is a linear combination of s((a+b-c)(a-b)2(a+b+zc)2), s(a3(b-c)2) and abcs(a2-ab)
            y = [
                m / 4,
                (3 + z) * (1 - z) * m / 4,
                (u + z + 1) * m / 2,
                (v - z**2 + 2*(u + z + 1)) * m / 4,
                rem
            ]
            if all(_ >= 0 for _ in y):
                multiplier = CyclicSum(a**2 - b*c)
                exprs = [
                    CyclicSum(c*(a-c)**2*(b-c)**2*((z+1)*(a+b) + 2*c)**2),
                    CyclicSum(a) * CyclicProduct((a-b)**2),
                    CyclicSum((a-b)**2) * CyclicSum(a**3*(b-c)**2),
                    CyclicProduct(a) * CyclicSum((a-b)**2)**2,
                    CyclicSum((a-b)**2) * CyclicProduct(a) * CyclicSum(a*b),
                ]

                return _sum_y_exprs(y, exprs) / multiplier

        t1 = -z
        t2 = -(2*u*z - u + v*z + 4*v + 2*z - 1)/(-6*u - 3*v + 4*z**2 - 4*z - 5)

        u1 = -z - 1
        u2 = t2**3 + t2**2*z - 2*t2**2 - 2*t2*z + t2 - 1
        u3 = sp.S(1)
        v1 = z**2
        v2 = -t2**2*(2*t2 + 2*z - 1)
        v3 = sp.S(-2)

        # w1 + w2 = 1
        # w1u1 + w2u2 + w3u3 = u
        # w1v1 + w2v2 + w3v3 = v
        w1 = (u*v3 - u2*v3 - u3*v + u3*v2)/(u1*v3 - u2*v3 - u3*v1 + u3*v2)
        w2 = 1 - w1
        w3 = (u - w1*u1 - w2*u2)

        y = [
            w1 * m / 4,
            w2 * m / 4,
            (w1 * (3 + z) * (1 - z) + w2 * (3 - t2) * (1 + t2)) * m / 4,
            w3 * m / 2,
            rem
        ]
        if all(_ >= 0 for _ in y):
            multiplier = CyclicSum(a*a - b*c),
            exprs = [
                CyclicSum(c*(a-c)**2*(b-c)**2*((z+1)*(a+b) + 2*c)**2),
                CyclicSum(c*(a-c)**2*(b-c)**2*((1-t2)*(a+b) + 2*c)**2),
                CyclicSum(a) * CyclicProduct((a-b)**2),
                CyclicSum((a-b)**2) * CyclicSum(a**3*(b-c)**2),
                CyclicProduct(a) * CyclicSum((a-b)**2) * CyclicSum(a*b)
            ]
            return _sum_y_exprs(y, exprs) / multiplier
        
        return None


    return None



def _sos_struct_quintic_symmetric_degenerate(coeff):
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
    is equivalent to s(c(a-b)2((7t-1)a2+(5t+1)(b2+c2-ab-ac)+(5t-11)bc)2).
    """
    m = coeff((4,1,0))
    if m < 0:
        return None

    rem = (coeff((4,1,0)) + coeff((3,2,0))) * 2 + coeff((3,1,1)) + coeff((2,2,1))
    if rem < 0:
        return None

    if True:
        # very trivial as a combination of s(a(b-c)4), s(a3(b-c)2), abcs(a2-ab)
        y = [
            coeff((4,1,0)),
            coeff((3,2,0)),
            coeff((4,1,0)) * 4 + coeff((3,2,0)) + coeff((3,1,1)) / 2,
            rem
        ]
        if all(_ >= 0 for _ in y):
            exprs = [
                CyclicSum(a*(b-c)**4),
                CyclicSum(a**3*(b-c)**2),
                CyclicProduct(a) * CyclicSum((a-b)**2),
                CyclicProduct(a) * CyclicSum(a*b)
            ]
            return _sum_y_exprs(y, exprs)

    if m == 0:
        return None
    if rem == 0:
        return None
    return None
    u, v = coeff((3,2,0)) / m, coeff((3,1,1)) / m
    if (2*u + v)**2 + 8*v <= 0:
        # linear combination of t1 = -2 and t2 = ...? on parabola (t^2-2t, -2t^2)
        k = (v + 8) / u
        t2 = -4/(k + 2)
        x2 = t2**2 - 2*t2

        w2 = u / x2 # (u - 0) / (x2 - 0)
        w1 = 1 - w2
        if 0 <= w1 <= 1:
            y = [
                w1 * m,
                w2 * m,
                rem / 2
            ]
            exprs = [
                CyclicSum(a*(b-c)**4), # s(a(b-c)2(b+c-2a)2) == s(a(b-c)4)
                CyclicSum(a*(b-c)**2*(a+b-t2*c)**2),
                CyclicProduct(a) * CyclicSum((a-b)**2)
            ]
            return _sum_y_exprs(y, exprs)

    if (2*u + v)**2 - 8*(36*u - 7*v - 48) <= 0:
        k = (v + 8) / u
        t2 = 5/(3*k - 19)
        x2 = 24*t2*(t2-1)/(5*t2+1)**2

        w2 = u / x2 # (u - 0) / (x2 - 0)
        w1 = 1 - w2
        if 0 <= w1 <= 1:
            multiplier = CyclicSum(a*a - b*c)
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
            return _sum_y_exprs(y, exprs) / multiplier

    # idea:
    # if there is no root on symmetric axis, then there is no root at all -> subtract anything
    # has root on symmetric axis -> subtract s(c(a-b)2(a+b-xc)2) until another root appears

    return None