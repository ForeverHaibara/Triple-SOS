import sympy as sp
from sympy import Poly, Symbol, Rational, Add
from sympy import MutableDenseMatrix as Matrix

from .sextic_symmetric import _restructure_quartic_polynomial

from .utils import (
    Coeff, CommonExpr, DomainExpr,
    quadratic_weighting, sum_y_exprs, rationalize_func
)

def _solve_inverse_quartic(coeff: Coeff, m, p, n, r):
    """
    Solve a symmetric inverse quartic expression fast without callbacks. It only involves
    monoms inside the triangle (a^4b^4, a^4c^4, b^4c^4). Hence it is equivalent to a
    quartic with respect to ab, bc and ca.

    Formally, it solves the problem:
    s(a^4b^4 + p(a^4b^3c+a^4bc^3) + qa^4b^2c^2 + ra^3b^3c^2) >= 0.
    """
    a, b, c = coeff.gens
    CyclicSum, CyclicProduct = coeff.cyclic_sum, coeff.cyclic_product
    if m >= 0 and m + 2*p + n + r >= 0:
        if m != 0 and (n - ((p / m)**2 - 1) * m) >= 0:
            y = [
                m / 2,
                (n - ((p / m)**2 - 1) * m) / 2 if m != 0 else sp.S(0),
                m + 2*p + n + r
            ]
            exprs = [
                CyclicSum(a**2*(b - c)**2*(a*b + a*c + (p / m) *b*c)**2),
                CyclicProduct(a**2) * CyclicSum((a - b)**2),
                CyclicSum(a**3*b**3*c**2),
            ]
            return sum_y_exprs(y, exprs)

        elif p + m >= 0 and (n + 2*(p + m)) >= 0:
            y = [
                m / 2,
                p + m,
                (n + 2*(p + m)) / 2,
                m + 2*p + n + r
            ]
            exprs = [
                CyclicSum(a**2*(b-c)**2*(a*b + a*c - b*c)**2),
                CyclicProduct(a) * CyclicSum(a**3*(b - c)**2),
                CyclicProduct(a**2) * CyclicSum((a - b)**2),
                CyclicSum(a**3*b**3*c**2),
            ]
            return sum_y_exprs(y, exprs)


def sos_struct_octic_symmetric(coeff, real=True):
    if not all(coeff((i,j,k)) == coeff((j,i,k)) for (i,j,k) in ((7,1,0),(6,2,0),(5,3,0),(5,2,1),(4,3,1))):
        return

    if not coeff.is_rational:
        return

    if coeff((8,0,0)) == 0 and coeff((7,1,0)) == 0:
        if coeff((6,2,0)) == 0 and coeff((5,3,0)) == 0:
            return _sos_struct_octic_symmetric_hexagram(coeff)
        return _sos_struct_octic_symmetric_hexagon(coeff)

    if coeff((8,0,0)) != 0:
        return _sos_struct_octic_symmetric_quadratic_form(coeff.as_poly(), coeff)

def _sos_struct_octic_symmetric_hexagon_sdp(coeff: Coeff):
    """
    Solve symmetric hexagons for real numbers by subtracting r * s((a-b)^2(ab(a+b)+xc(a^2+b^2)+..)^2)
    so that the remaining part is a quadratic form with respect to
    s(a^3b-a^2bc), s(a^3c-a^2bc) and s(a^2b^2-a^2bc).

    In particular, since the polynomial is symmetric, we assume s(a^3b-a^2bc) and s(a^3c-a^2bc) are
    equivalent and the 3*3 matrix should have the following form:
    [[M00, M01, M02]
     [M01, M00, M02]
     [M02, M02, M22]]
    Matrix being positive semidefinite requires all principal submatrix determinants >= 0.
    We require M00 >= 0, M22 >= 0, M00^2 - M01^2 >= 0, M22*(M00+M01) - 2*M02^2 >= 0.
    The first and the third can be reduced to M00 - M01 >= 0 and M00 + M01 >= 0.
    The second automatically holds as long as M00 + M01 > 0 STRICTLY with the fourth.

    See similar methods in _sos_struct_sextic_full_sdp.

    TODO: 1. Handle w1 == 0. 2. Handle c620 +- 2c611 == 0.

    Examples
    --------
    (s(a2(b2-c2)2)-3/8p(a-b)2)s(a2)+s(a4(b-c)2)s(a2)/8

    s(4a4b2-7a4bc+4a4c2+8a3b3-12a3b2c-12a3bc2+15a2b2c2+a4(b-c)2)s(a2-ab)

    (85/336p(a-b)2+s(bc(a-b)(a-c)(a+b)(a+c))-16/15s(a2bc(b-c)2))s(a2-ab)

    s(a3(bc(a+b+c)((a-2b)(a-2c)-bc)+a(a-b-c)(a-3b-3c)(b-c)2))

    s(a2(a-(b+c))2((b-c)2+bc)(a-b)(a-c))
    """
    c620, c530, c440, c611, c521, c431, c422 = [coeff(_) for _ in ((6,2,0),(5,3,0),(4,4,0),(6,1,1),(5,2,1),(4,3,1),(4,2,2))]
    if (not coeff.is_rational) or c620 <= 0 or coeff.poly111() != 0:
        return None

    w1 = c521 + c611 + 2*c620
    w2 = c431 + 2*c440 + 3*c530 - c611
    w4 = c422 + 4*c431 + 5*c440 + 8*c521 + 12*c530 + 8*c611 + 18*c620
    if w4 < 0:
        # w4 = 1/2 * (df^2)/(da^2) at a = b = c = 1
        return None

    a, b, c = coeff.gens
    CyclicSum, CyclicProduct = coeff.cyclic_sum, coeff.cyclic_product

    def polylize(_M, gen = Symbol('t')):
        # n = len(_M) - 1
        # return lambda t: sum(_M[i] * t**(n-i) for i in range(n+1))
        return Poly.from_list(_M, gen)

    def stack_quad_form(M00t, M01t, M02t, M22t):
        return sp.Matrix([
            [M00t, M01t, M02t],
            [M01t, M00t, M02t],
            [M02t, M02t, M22t]
        ])

    def _compute_quad_form_sol(quad_form):
        """
        Solve v' * M * v where v = [s(a^3b-a^2bc), s(a^3c-a^2bc), s(a^2b^2-a^2bc)]
        while M is in the following form.
        [[M00, M01, M02]
        [M01, M00, M02]
        [M02, M02, M22]]
        """
        s1 = (quad_form[0,0] - quad_form[0,1])/2 * CyclicSum(a)**2 * CyclicProduct((a-b)**2)

        def mapping(_vec):
            x, y = _vec
            # return s(x(a3b+ab3-2a2bc)+y(a2b2-a2bc))^2
            if x == 1 and y == 2:
                return CyclicSum(a)**2 * CyclicSum(a*(b-c)**2)**2
            elif x == 1 and y == -2:
                return CyclicSum(a*b*(a-b)**2)**2
            elif x == 0 and y == 1:
                return CyclicSum(a**2*(b-c)**2)**2 / 4
            p1 = a**3*b + a*b**3 - a**2*b*c - a*b**2*c
            p2 = a**2*b**2 - a*b*c**2
            return CyclicSum((x*p1 + y*p2).expand().together())**2

        s2 = quadratic_weighting(coeff,
            (quad_form[0,0] + quad_form[0,1])/2,
            quad_form[0,2] * 2,
            quad_form[2,2],
            mapping = mapping
        )
        if s2 is None: return None

        return s1 + s2

    def _sol_to_result(sol):
        if sol is None:
            return None
        u210, u102, u201, u111, r, quad_form = sol
        quad_form_sol = _compute_quad_form_sol(quad_form)
        if r >= 0 and quad_form_sol is not None:
            ker = (a-b)*(u102*c**2*(b+a) + u210*(a*b*(a+b)-c**3) + u201*c*(a**2+b**2+c**2) + u111*a*b*c).expand().together()
            return r * CyclicSum(ker**2) + quad_form_sol


    def _nondegenerated_hessian():
        """
        This function solves the parameters when w4 > 0 strictly.
        Note that w4 = 1/2 * (df^2)/(da^2) at a = b = c = 1.
        In this case, the polynomial "ker" has only one degree of freedom, and can be parametrized by
        a single variable t.
        We solve for t such that the symmetric matrix M is PSD.
        """
        t = Symbol('t')
        _M00 = [
            81*c620*w1**2*w2**2*w4 - w1**4*w4**2 - 15*w1**3*w2**2*w4 + w1**3*w2*w4**2 - 117*w1**2*w2**4 - 9*w1**2*w2**3*w4 - 63*w1*w2**5 + 18*w1*w2**4*w4 + w1*w2**3*w4**2 - 9*w2**6 + 6*w2**5*w4 - w2**4*w4**2,
            -w1*(-162*c620*w1*w2**2*w4 + 4*w1**3*w4**2 + 36*w1**2*w2**2*w4 - 3*w1**2*w2*w4**2 + 45*w1*w2**4 + 9*w1*w2**3*w4 + 9*w2**5 - w2**3*w4**2),
            -3*w1**2*(-27*c620*w2**2*w4 + 2*w1**2*w4**2 + 9*w1*w2**2*w4 - w1*w2*w4**2 + 3*w2**4),
            -w1**3*w4*(4*w1*w4 + 6*w2**2 - w2*w4),
            -w1**4*w4**2
        ]

        _M01 = [
            (81*c611*w1**2*w2**2*w4 + w1**4*w4**2 + 42*w1**3*w2**2*w4 + 2*w1**3*w2*w4**2 + 198*w1**2*w2**4 - 6*w1**2*w2**2*w4**2 + 90*w1*w2**5 - 36*w1*w2**4*w4 + 2*w1*w2**3*w4**2 + 9*w2**6 - 6*w2**5*w4 + w2**4*w4**2)/2,
            w1*(81*c611*w1*w2**2*w4 + 2*w1**3*w4**2 + 45*w1**2*w2**2*w4 + 3*w1**2*w2*w4**2 + 63*w1*w2**4 + 9*w1*w2**3*w4 - 6*w1*w2**2*w4**2 + 18*w2**5 - 9*w2**4*w4 + w2**3*w4**2),
            3*w1**2*(27*c611*w2**2*w4 + 2*w1**2*w4**2 + 18*w1*w2**2*w4 + 2*w1*w2*w4**2 + 3*w2**4 + 6*w2**3*w4 - 2*w2**2*w4**2)/2,
            w1**3*w4*(2*w1*w4 + 3*w2**2 + w2*w4),
            w1**4*w4**2/2
        ]

        _M02 = [
            (81*c530*w1**2*w2**2*w4 + 2*w1**4*w4**2 + 48*w1**3*w2**2*w4 + w1**3*w2*w4**2 + 126*w1**2*w2**4 - 6*w1**2*w2**2*w4**2 + 99*w1*w2**5 - 36*w1*w2**4*w4 + w1*w2**3*w4**2 + 18*w2**6 - 12*w2**5*w4 + 2*w2**4*w4**2)/2,
            w1*(162*c530*w1*w2**2*w4 + 8*w1**3*w4**2 + 99*w1**2*w2**2*w4 + 3*w1**2*w2*w4**2 - 45*w1*w2**4 + 9*w1*w2**3*w4 - 12*w1*w2**2*w4**2 - 9*w2**5 + w2**3*w4**2)/2,
            3*w1**2*(27*c530*w2**2*w4 + 4*w1**2*w4**2 + 18*w1*w2**2*w4 + w1*w2*w4**2 - 3*w2**4 + 3*w2**3*w4 - 2*w2**2*w4**2)/2,
            w1**3*w4*(8*w1*w4 + 3*w2**2 + w2*w4)/2,
            w1**4*w4**2
        ]

        _M22 = [
            81*c440*w1**2*w2**2*w4 - 81*c611*w1**2*w2**2*w4 - 4*w1**4*w4**2 - 78*w1**3*w2**2*w4 - 2*w1**3*w2*w4**2 - 441*w1**2*w2**4 - 36*w1**2*w2**3*w4 + 12*w1**2*w2**2*w4**2 - 252*w1*w2**5 + 90*w1*w2**4*w4 - 2*w1*w2**3*w4**2 - 36*w2**6 + 24*w2**5*w4 - 4*w2**4*w4**2,
            -2*w1*(-81*c440*w1*w2**2*w4 + 81*c611*w1*w2**2*w4 + 8*w1**3*w4**2 + 72*w1**2*w2**2*w4 + 3*w1**2*w2*w4**2 + 63*w1*w2**4 + 45*w1*w2**3*w4 - 12*w1*w2**2*w4**2 + 18*w2**5 - 9*w2**4*w4 + w2**3*w4**2),
            -3*w1**2*(-27*c440*w2**2*w4 + 27*c611*w2**2*w4 + 8*w1**2*w4**2 + 18*w1*w2**2*w4 + 2*w1*w2*w4**2 + 3*w2**4 + 18*w2**3*w4 - 4*w2**2*w4**2),
            -2*w1**3*w4*(8*w1*w4 - 6*w2**2 + w2*w4),
            -4*w1**4*w4**2
        ]

        M00t, M01t, M02t, M22t = [polylize(_, t) for _ in (_M00, _M01, _M02, _M22)]
        det = (M22t * (M00t + M01t) - 2*M02t**2).div((t*(t+1)).as_poly(t))[0]


        def _is_valid(t):
            return sp.sign(det(t)) * sp.sign(t) * sp.sign(t+1) >= 0 and M00t(t) >= abs(M01t(t))

        # det is a 6-degree polynomial with respect to t
        t = rationalize_func(det.diff(), _is_valid)
        if t is None:
            return None

        # now we have a valid t
        if t != -1:
            p4 = (w1*w4 + 3*w2**2 - w2*w4)*t + w1*w4
            reg = (81*t**2*w1**2*w2**2*(t + 1)**2*w4)
            M00t, M01t, M02t, M22t = [f(t)/reg for f in (M00t, M01t, M02t, M22t)]
            u111_ = -(5*t**2*w1**2*w4 - 12*t**2*w1*w2**2 + 3*t**2*w1*w2*w4 - 3*t**2*w2**3 + t**2*w2**2*w4 + 10*t*w1**2*w4 - 3*t*w1*w2**2 + 3*t*w1*w2*w4 + 5*w1**2*w4)

            if p4 != 0:
                x_ = -w2/w1/(t+1)*(p4 + 9*t*w1*w2)/p4
                r = p4**2/(162*t**2*w2**2*w4)
                u210 = sp.S(1)
                u201 = x_ * t
                u102 = 2 + w2/w1 + x_
                u111 = u111_/(w1*(t + 1)*p4)
            else:
                x_ = -w2/w1/(t+1)*(p4 + 9*t*w1*w2)
                r = x_**2/(162*t**2*w2**2*w4)
                u210 = sp.S(0)
                u201 = t
                u102 = sp.S(1)
                u111 = u111_/(-w2*(p4 + 9*t*w1*w2))
        else:
            # take limit t -> -1
            x_ = -(3*w1 + w2)/(3*w1)
            r = w1**2/(6*(3*w1 + w2))
            M00t = -(-81*c620*w1 - 27*c620*w2 + 9*w1**2 + 3*w1*w2 + w2**2)/(27*(3*w1 + w2))
            M01t = -(-81*c611*w1 - 27*c611*w2 + 18*w1**2 + 6*w1*w2 - w2**2)/(54*(3*w1 + w2))
            M02t = -(-81*c530*w1 - 27*c530*w2 + 9*w1**2 + 12*w1*w2 + 4*w2**2)/(54*(3*w1 + w2))
            M22t = (81*c440*w1 + 27*c440*w2 - 81*c611*w1 - 27*c611*w2 + 18*w1**2 - 12*w1*w2 - 7*w2**2)/(27*(3*w1 + w2))
            u111 = -(-9*w1*w2 - 2*w2**2)/(3*w1*w2)

            u210 = sp.S(1)
            u201 = x_ * t
            u102 = 2 + w2/w1 + x_

        quad_form = stack_quad_form(M00t, M01t, M02t, M22t)
        # print('PARAMS =', t, u102, u201, u111, r, quad_form)
        return u210, u102, u201, u111, r, quad_form


    def _nondegenerated_hessian_degen_w1():
        """
        Special case when w1 == 0. In this case, we can parametrize the problem by:
        u201 = t
        u102 = -(t + 1)*(3*t*w2 - t*w4 + 2*w4)/(3*t*w2 - t*w4 - w4)
        u111 = -(3*t**2*w2 - t**2*w4 - 3*t*w2 + 3*t*w4 - 5*w4)/(3*t*w2 - t*w4 - w4)
        r = (3*t*w2 - t*w4 - w4)**2 / (162*t**2*w4)
        """
        t = Symbol('t')
        _M00 = [
            -2*(3*w2 - w4)**2,
            2*(3*w2 - w4)*(3*w2 + w4),
            -18*(-9*c620*w4 + w2**2),
            2*w4*(6*w2 - w4),
            -2*w4**2
        ]

        _M01 = [
            (3*w2 - w4)**2,
            -2*(3*w2 - w4)*(6*w2 - w4),
            3*(27*c611*w4 + 3*w2**2 + 6*w2*w4 - 2*w4**2),
            -2*w4*(3*w2 + w4),
            w4**2
        ]

        _M02 = [
            2*(3*w2 - w4)**2,
            (3*w2 - w4)*(3*w2 + w4),
            -3*(-27*c530*w4 + 3*w2**2 - 3*w2*w4 + 2*w4**2),
            -w4*(3*w2 + w4),
            2*w4**2
        ]

        _M22 = [
            -8*(3*w2 - w4)**2,
            4*(3*w2 - w4)*(6*w2 - w4),
            -6*(-27*c440*w4 + 27*c611*w4 + 3*w2**2 + 18*w2*w4 - 4*w4**2),
            -4*w4*(6*w2 - w4),
            -8*w4**2
        ]

        M00t, M01t, M02t, M22t = [polylize(_, t) for _ in (_M00, _M01, _M02, _M22)]
        det = (M22t * (M00t + M01t) - 2*M02t**2).div((t).as_poly(t))[0]


        def _is_valid(t):
            return t != 0 and sp.sign(det(t)) * sp.sign(t) >= 0 and M00t(t) >= abs(M01t(t))

        t = rationalize_func(det.diff(), _is_valid)
        if t is None:
            return None

        u210 = sp.S(1)
        u201 = t
        u102 = -(t + 1)*(3*t*w2 - t*w4 + 2*w4)/(3*t*w2 - t*w4 - w4)
        u111 = -(3*t**2*w2 - t**2*w4 - 3*t*w2 + 3*t*w4 - 5*w4)/(3*t*w2 - t*w4 - w4)
        reg = 162*t**2*w4
        r = (3*t*w2 - t*w4 - w4)**2 / reg
        M00t, M01t, M02t, M22t = [f(t)/reg for f in (M00t, M01t, M02t, M22t)]
        quad_form = stack_quad_form(M00t, M01t, M02t, M22t)
        return u210, u102, u201, u111, r, quad_form



    def _degenerated_hessian():
        """
        For case when w4 == 0, we also require w2 == 0. In this case the hessian of poly at (1,1,1) is zero matrix.
        This is often attained when poly = (sextic polynomial) * s(a^2-ab), so solving the case here
        means solving a handful of sextic inequalities.

        In this case, we have two degrees of freedom on "ker": u102 and u201.
        We require that
        eq1 := M00 - M01 >= 0
        eq2 := M00 + M01 >= 0
        det := M22 * (M00 + M01) - 2*M02**2 >= 0
        M22 >= 0

        In addition, we require
        w1 / (u201*(u102 + u201 + 1)) = -6r <= 0, where r is the coefficient of Cyclic(ker^2).

        Now we change the variable that u102 = x / (-w1) - y - 1, u201 = y.
        The constraints then converts to.
        """
        def _compute_params(x, y):
            u210 = sp.S(1)
            u201 = y
            u102 = x / (-w1) - y - 1

            u111 = -2*u102 - 3*u201 - 1
            reg = (3*u201*(u102 + u201 + 1))
            r = -w1/(2*reg)

            M00t = (3*c620*u102*u201 + 3*c620*u201**2 + 3*c620*u201 + u201**2*w1 - u201*w1 + w1)/reg
            M01t = (3*c611*u102*u201/2 + 3*c611*u201**2/2 + 3*c611*u201/2 - u201**2*w1/2 + 2*u201*w1 - w1/2)/reg
            M02t = (3*c530*u102*u201/2 + 3*c530*u201**2/2 + 3*c530*u201/2 + u102*u201*w1 - u102*w1/2)/reg
            M22t = (3*c440*u102*u201 + 3*c440*u201**2 + 3*c440*u201 - 3*c611*u102*u201 - 3*c611*u201**2 - 3*c611*u201 + u102**2*w1 + 3*u201**2*w1 - 6*u201*w1)/reg
            quad_form = stack_quad_form(M00t, M01t, M02t, M22t)
            if quad_form.is_positive_semidefinite:
                return u210, u102, u201, u111, r, quad_form

        y = Symbol('y')

        # The following w5 = -discriminant(poly(a,1,1) / (a-1)^4) / 4
        # so we must have w5 >= 0
        w5 = c440*c611 + 2*c440*c620 - c521**2 - 2*c521*c530 - 4*c521*c611 - 8*c521*c620 - c530**2 - 2*c530*c611 - 4*c530*c620 - 4*c611**2 - 14*c611*c620 - 12*c620**2
        if w5 < 0:
            return None


        def _strict_psd():
            _func_z_sym = [
                (2*c530*w1 + 4*c611*w1 + 8*c620*w1 + 2*w1**2 + 3*w5)/(2*(c611 + 2*c620)),
                w1*(c530 - c611 - 2*c620 + w1)/(c611 + 2*c620),
                0
            ]

            _func_z_det = [
                w5*(12*c530*w1 + 24*c611*w1 + 48*c620*w1 + 8*w1**2 + 9*w5)/(4*(c611 + 2*c620)**2),
                w1*w5*(3*c530 - 3*c611 - 6*c620 + w1)/(c611 + 2*c620)**2,
                -w1**2*w5/(c611 + 2*c620)**2,
                0,
                0
            ]
            func_z_sym = polylize(_func_z_sym, y)
            func_z_sym_lb = func_z_sym - (w1**2 / (2*c620 - c611) * (y-1)**2).as_poly(y)
            func_z_det = polylize(_func_z_det, y)
            _func_z_det_det = _func_z_det[1]**2 - 4*_func_z_det[0]*_func_z_det[2]

            # print('RHS =', sp.latex((func_z_sym.as_expr() + sp.sqrt(func_z_det.as_expr())).subs(y,Symbol('x'))))
            # print('LHS =', sp.latex((w1**2 / (2*c620 - c611) * (y-1)**2).subs(y,Symbol('x'))))

            # Require F(y) = func_z_sym_lb + sqrt(det) >= 0
            # Also, det >= 0 is a necessary condition
            if _func_z_det[0] < 0 and _func_z_det_det < 0:
                return None

            def _is_valid(y):
                u, v = func_z_sym_lb(y), func_z_det(y)
                return y != 0 and v >= 0 and (u >= 0 or u**2 <= v)

            y_ = None
            if func_z_det.degree() == 4 and func_z_det.LC() > 0:
                if func_z_sym_lb.LC() >= 0 or func_z_sym_lb.LC()**2 < func_z_det.LC():
                    # func_z_sym_lb.LC() + sp.sqrt(func_z_det.LC()) > 0
                    # let y -> oo
                    y_ = 1
                    for _ in range(100):
                        if _is_valid(y_):
                            break
                        y_ *= 2
                    else:
                        y_ = None

            if y_ is None:
                func_y_diff = func_z_sym_lb.diff(y)**2 * func_z_det * 4 - func_z_det.diff(y)**2
                y_ = rationalize_func(func_y_diff, _is_valid)

            if y_ is None and func_z_det.degree() == 4:
                # finally: check the boundary func_z_det >= 0
                a0, b0, c0, _, __ = func_z_det.all_coeffs()
                y_ = rationalize_func(polylize([a0, b0, c0], y), _is_valid, direction = 1)

            if y_ is not None:
                z_ = max(
                    w1**2 * (y_ - 1)**2 / (2*c620 - c611),
                    w1**2 * (y_ + 1)**2 / 3 / (2*c620 + c611),
                    func_z_sym(y_)
                )
                x_ = z_ / y_
                return _compute_params(x_, y_)

        def _degenerated_w5():
            """
            When w5 == 0, there is a root on the symmetric axis of poly.
            In this case, to ensure M22 * (M00 + M01) - 2*M02**2 >= 0,
            the two parameters x and y must satisfy a linear constraint:
            x = w1*(c530*y + c530 + 2*c611*y - c611 + 4*c620*y - 2*c620 + w1*y + w1)/(c611 + 2*c620)

            We try to find y such that eq2 >= 0 and eq1 >= 0.
            """
            _eq1 = [
                -3*w1*(c530*c611 - 2*c530*c620 + 2*c611**2 + 2*c611*w1 - 8*c620**2),
                3*w1*(-c530*c611 + 2*c530*c620 + c611**2 + c611*w1 - 4*c620**2 + 6*c620*w1),
                -3*w1**2*((c611 + 2*c620))
            ]
            _eq2 = [
                w1*(3*c530 + 6*c611 + 12*c620 + 2*w1)/2,
                w1*(3*c530 - 3*c611 - 6*c620 + w1)/2,
                -w1**2/2
            ]
            eq1, eq2 = polylize(_eq1, y), polylize(_eq2, y)

            def _is_valid(y):
                return eq1(y) >= 0 and eq2(y) >= 0

            y_ = rationalize_func((eq1 * eq2).diff(y), _is_valid)
            if y_ is not None:
                x_ = w1*(c530*y_ + c530 + 2*c611*y_ - c611 + 4*c620*y_ - 2*c620 + w1*y_ + w1)/(c611 + 2*c620)
                return _compute_params(x_, y_)

        def _degenerated_M00_M01():
            """
            The case when M00 == M01.
            """
            0

        sol = None
        if w5 > 0:
            sol = _strict_psd()
        elif w5 == 0:
            sol = _degenerated_w5()

        if sol is None:
            sol = _degenerated_M00_M01()
        return sol


    def _degenerated_hessian_degen_w1():
        """
        Special case when w1 == w2 == w4 == 0.
        In this case, we use u111 = -2*u102 - 1, u201 = 0, and there are two degrees of freedom: r and u102.

        Note that w5 = c440*c611 + 2*c440*c620 - c530**2 - c611**2 - 2*c611*c620 >= 0,
        because w5 = -discriminant(poly(a,1,1) / (a-1)^4) / 4.
        """
        w5 = c440*c611 + 2*c440*c620 - c530**2 - c611**2 - 2*c611*c620
        if w5 < 0:
            return None

        if c611 + 2*c620 == 0:
            # r = 0 is a must by the constraint M00 + M01 >= 0 and r >= 0.
            r = sp.S(0)
            u111, u102, u201 = sp.S(0), sp.S(0), sp.S(0)
            M00t = c620
            M01t = -c620
            M02t = c530/2
            M22t = c440 + 2*c620
        elif c611 + c620 >= 0:
            r = (2*c620 - c611)/6
            M00t = (c611 + c620)/3
            M01t = (c611 + c620)/3
            M02t = 2*c530*(c611 + c620)/(3*(c611 + 2*c620))
            M22t = (4*c530**2*(c611 + c620) + 3*w5*(c611 + 2*c620))/(3*(c611 + 2*c620)**2)
        else:
            r = (2*c620 + c611)/2
            M00t = -(c611 + c620)
            M01t = -M00t
            M02t = sp.S(0)
            M22t = w5 / (c611 + 2*c620)

        u210 = sp.S(1)
        if c611 + 2*c620 != 0:
            u102 = -c530/(c611 + 2*c620)
            u201 = sp.S(0)
            u111 = -2*u102 - 1
        quad_form = stack_quad_form(M00t, M01t, M02t, M22t)
        return u210, u102, u201, u111, r, quad_form

    if w4 > 0:
        if w1 != 0:
            return _sol_to_result(_nondegenerated_hessian())
        else:
            return _sol_to_result(_nondegenerated_hessian_degen_w1())
    elif w4 == 0 and w2 == 0:
        if w1 != 0:
            return _sol_to_result(_degenerated_hessian())
        else:
            return _sol_to_result(_degenerated_hessian_degen_w1())


def _sos_struct_octic_symmetric_hexagon(coeff: Coeff):
    """
    Try to solve symmetric octic hexagon, without terms a^8, a^7b and a^7c.

    For octics and structural method, the core is not to handle very complicated cases.
    Instead, we explore the art of sum of squares by using simple tricks.
    """
    c1, c2, c3, c4 = [coeff(_) for _ in ((6,2,0),(5,3,0),(6,1,1),(5,2,1))]
    if c1 < 0 or 2*c1 + c3 < 0:
        return None

    solution = _sos_struct_octic_symmetric_hexagon_sdp(coeff)
    if solution is not None:
        return solution

    a, b, c = coeff.gens
    CyclicSum, CyclicProduct = coeff.cyclic_sum, coeff.cyclic_product

    if True:
        # Case 1. use s(a(b-c)2)2s(xa2+yab)+p(a-b)2s(za2+wab)
        x_ = c1/2 + c3/4
        y_ = c1 + c2/4 + c3/2 + c4/4
        z_ = c1/2 - c3/4
        w_ = -c1 + 3*c2/4 - 3*c3/2 - c4/4
        # print(x_, y_, z_, w_)
        if x_ >= 0 and z_ >= 0 and x_ + y_ >= 0 and z_ + w_ >= 0:
            m_ = coeff((4,4,0)) - (-2*w_ + 2*x_ + 2*y_ + 2*z_)
            p_ = coeff((4,3,1)) - (w_ - 8*x_ - 7*y_)
            n_ = coeff((4,2,2)) - (2*w_ + 44*x_ - 18*y_ - 4*z_)
            r_ = coeff((3,3,2)) - (-2*w_ - 18*x_ + 22*y_ + 2*z_)
            solution = _solve_inverse_quartic(coeff, m_, p_, n_, r_)
            if solution is not None:
                return Add(
                    solution,
                    CyclicSum(a*(b-c)**2)**2 * CyclicSum(x_*a**2 + y_*b*c),
                    CyclicProduct((a-b)**2) * CyclicSum(z_*a**2 + w_*b*c)
                )

        if True:
            # Case 2.
            # use xs((a-b)2((a2b+a2c+ab2-ac2+b2c-bc2)+y(ac2+bc2-2abc))2)+p(a-b)2s(za2+wab)
            # this enables nontrivial equality cases on the symmetric axis
            x_ = (2*c1 + c3)/8
            y_ = -2*(2*c1 + c2 + c3 + c4)/(2*c1 + c3) if 2*c1 + c3 != 0 else sp.S(0)
            z_ = (2*c1 - c3)/4
            w_ = (10*c1 + 6*c2 + c3 + 2*c4)/4
            if x_ >= 0 and z_ >= 0 and z_ + w_ >= 0:
                m_ = coeff((4,4,0)) - (-2*w_ + 2*x_*y_**2 - 4*x_*y_ + 2*z_)
                p_ = coeff((4,3,1)) - (w_ - 4*x_*y_**2 + 6*x_*y_ + 2*x_)
                n_ = coeff((4,2,2)) - (2*w_ + 6*x_*y_**2 + 20*x_*y_ + 4*x_ - 4*z_)
                r_ = coeff((3,3,2)) - (-2*w_ - 20*x_*y_ + 2*z_)
                solution = _solve_inverse_quartic(coeff, m_, p_, n_, r_)
                if solution is not None:
                    return Add(
                        solution,
                        x_ * CyclicSum((a-b)**2 * (a**2*b+a**2*c+a*b**2+(y_-1)*a*c**2+b**2*c+(y_-1)*b*c**2-2*y_*a*b*c)**2),
                        CyclicProduct((a-b)**2) * CyclicSum(z_*a**2 + w_*b*c)
                    )


    if coeff((6,2,0)) == 0 and coeff((5,3,0)) == 0:
        return _sos_struct_octic_symmetric_hexagram(coeff)

    return None


def _sos_struct_octic_symmetric_hexagram(coeff: Coeff):
    """
    Solve octic symmetric hexagram, where all terms are inside the triangle (a^6bc,...) and (a^4b^4,...).

    The idea is to write the problem to s(bc(xa^4 + ya^3(b+c) + za^2(b^2+c^2) + wa^2bc + uabc(b+c) + vb^2c^2)(a-b)(a-c)).
    Then, we use the following lemma: if f(a,b,c) and g(a,b,c) are both symmetric polynomials with respect to b,c.
    Then, sum f(a,b,c)(a-b)(a-c) * sum g(a,b,c)(a-b)(a-c) - sum f(a,b,c)g(a,b,c)(a-b)(a-c)
    must be a multiple of p(a-b)2.
    A common choice of g is g(a,b,c) = 1.

    TODO: Restructure the function. It is too messy.

    Examples
    ---------
    s(bc(a2+1/2a(b+c)-bc)2(a-b)(a-c))

    s((a-b)2(a+b-3c)2)s(a2b2)+2s(a2(b-c)2(ab+ac-3/2bc)2)-p(a-b)2s(2a2-2ab)

    s(2a6bc-3a5b2c-3a5bc2+a4b4+3a4b2c2)

    s(bc(2a4+a3b+a3c+a2b2+9a2bc+a2c2-3ab2c-3abc2+b2c2)(a-b)(a-c))

    24s((a+b-c)(a-b)2(a+b-3c)2)p(a)+s(a2b2(ab-ac)(ab-bc))

    256p(a)s((64a+(b+c))(a+b-59/16c)(a+c-59/16b)(a-b)(a-c))+s(a2b2(ab-bc)(ab-ca))

    s(bc(a-b)(a-c)(a-2b)(a-2c)(a-3b)(a-3c))

    s(bc(a-b)(a-c)(a2-2a(b+c)+5bc)(a-2b)(a-2c))

    s(a4)s(a4)-3abcs(a5)-s((a2-bc)4)
    """
    x_ = coeff((6,1,1))
    v_ = coeff((4,4,0))
    rem = sum(coeff((i,j,k)) * (1 if i==j or j==k else 2) for i,j,k in ((6,1,1),(5,2,1),(4,3,1),(4,4,0),(4,2,2),(3,3,2)))
    if x_ <= 0 or v_ < 0 or rem < 0:
        return None

    y_ = coeff((5,2,1)) + x_
    u_ = coeff((4,3,1)) + v_ + y_
    balance = coeff((4,2,2)) - x_ + 2*u_ + 2*y_

    a, b, c = coeff.gens
    CyclicSum, CyclicProduct = coeff.cyclic_sum, coeff.cyclic_product

    # 2z + w = balance

    # now we ensure f(a,b,c) = (xa^4 + ya^3(b+c) + za^2(b^2+c^2) + wa^2bc + uabc(b+c) + vb^2c^2) >= 0
    # treat f as an quadratic form with respect to a^2, a(b+c) and bc, we shall have:

    # DEPRECATED: f = x(a^2 + r1*a(b+c) + (u/y)*bc)^2 + (v - u^2x/(y^2))b^2c^2 + (balance - y^2/x - 2ux/y)a^2bc + (z - y^2/(4x))(a(b-c))^2

    # let t be a parameter
    # f = x(a^2 + r1*a(b+c) + t*bc)^2 + (v - xt^2)(bc - ha(b+c))^2 + <rest>
    # rest = (w1 + z)a^2(b-c)^2 + (w2 + balance) * a^2bc
    # where r1 = y/(2x), h = (u-y*t)/(t^2*x-v)/2
    # w1 = (2*t*u*x*y - u**2*x - v*y**2)/(4*x*(v - t**2*x))
    # w2 = (2*t**3*x**3 + 2*t*u*x*y - 2*t*v*x**2 - u**2*x - v*y**2)/(x*(v - t**2*x))
    # to minimize f, we shall asume z = -w1 and w = balance + 2w1
    # we require w2 + balance >= 0, v >= xt^2

    t = sp.symbols('t')
    det = ((2*t**3*x_**3 + 2*t*u_*x_*y_ - 2*t*v_*x_**2 - u_**2*x_ - v_*y_**2) + balance * (x_*(v_ - t**2*x_))).as_poly(t)
    bound = (v_ - x_ * t**2).as_poly(t)
    # det2 = (-4*x_**2*(u_ + 2*v_ + x_)*t**2 + 2*u_*x_*y_*t + (-u_**2*x_ + 4*u_*v_*x_ + 8*v_**2*x_ + 4*v_*x_**2 - v_*y_**2)).as_poly(t)
    # print(det,'\n', det2, '\n', bound)

    for (t_, interval_end), _ in sp.polys.intervals(det * bound):
        bound_ = bound(t_)
        if bound_ > 0 and det(t_) >= 0: # and det2(t_) >= 0:
            t = t_
            h = (u_ - y_*t)/(t**2*x_ - v_)/2
            c2 = v_ - x_*t**2
            c3 = (2*t**3*x_**3 + 2*t*u_*x_*y_ - 2*t*v_*x_**2 - u_**2*x_ - v_*y_**2)/(x_*(v_ - t**2*x_)) + balance
            c4 = sp.S(0)
            break
        if bound_ == 0:
            # more special, v = xt^2
            # f = x(a^2 + r1*a(b+c) + t*bc)^2 + (u - ty)abc(b+c)
            #    + (z - y^2/(4x))a^2(b-c)^2 + (balance - (2*t*x**2 + y**2)/x) * a^2bc
            # WLOG z = y^2/(4x)
            h = sp.S(0)
            c2 = sp.S(0)
            c3 = balance - (2*t_*x_**2 + y_**2) / x_
            c4 = u_ - t_ * y_

            # r1 = y_ / (2*x_)
            # r2 = t_
            # degrade_a2bc = x_ * (-r1**2 + 2*r1*r2 + r2**2 + 1) + c4
            if c3 >= 0 and c4 >= 0: # and v_ + degrade_a2bc >= 0:
                t = t_
                break

    else:
        return None

    r1 = y_ / (2*x_)
    r2 = t

    if True:
        # take g(a,b,c) = 1 in the lemma
        degrade_a2b2 = v_
        degrade_a2bc = x_ * (-r1**2 + 2*r1*r2 + r2**2 + 1) + c2 * (-h**2 - 2*h + 1) + c4
        # print(degrade_a2b2, degrade_a2bc, (2*x_)*(a**2 + r1*a*b + r1*a*c + r2*b*c)**2 + c2*2*(b*c - h*a*b - h*a*c)**2 + c3*2*a**2*b*c + c4*a*b*c*(b+c))

        if degrade_a2b2 + degrade_a2bc >= 0:
            multiplier = CyclicSum((a-b)**2)
            # p1 == f(a,b,c)
            p1 = sp.together((2*x_)*(a**2 + r1*a*b + r1*a*c + r2*b*c)**2 + c2*2*(b*c - h*a*b - h*a*c)**2 + c3*2*a**2*b*c + c4*2*a*b*c*(b+c)).as_coeff_Mul()
            p2 = sp.together(degrade_a2b2 * CyclicSum(a**2*(b-c)**2) + 2*(degrade_a2bc + degrade_a2b2) * CyclicSum(a**2*b*c)).as_coeff_Mul()

            y = [
                p1[0],
                p2[0],
                rem
            ]
            exprs = [
                CyclicSum(b*c* p1[1] * (a-b)**2*(a-c)**2),
                CyclicProduct((a-b)**2) * p2[1],
                CyclicProduct(a**2) * CyclicSum(a*b) * multiplier,
            ]
            return sum_y_exprs(y, exprs) / multiplier

    if True:
        degrade_a3 = x_
        degrade_a2b = x_ * (2*(r1 - r2) + 1)
        degrade_abc = x_ * 3*((r1 - r2)**2 + 1) + c2 * 3*(h+1)**2 + c3 + c4

        if degrade_a2b + degrade_a3 >= 0 and degrade_a3*3 + degrade_a2b*6 + degrade_abc >= 0:
            # g(a,b,c) = bc
            multiplier = CyclicSum(a**2*(b-c)**2)

            p1 = (a**2 + r1*a*b + r1*a*c + r2*b*c).as_coeff_Mul()
            p2 = (b*c-h*a*b-h*a*c).as_coeff_Mul()
            p_fin = sp.together(degrade_a3 * CommonExpr.schur(3, (a,b,c))
                                + (degrade_a2b + degrade_a3) * CyclicSum(a*(b-c)**2)
                                + (degrade_a3*3 + degrade_a2b*6 + degrade_abc) * CyclicProduct(a)).as_coeff_Mul()

            y = [
                x_ * 2 * p1[0],
                c2 * 2 * p2[0],
                c3 * 2,
                c4 * 2,
                2 * p_fin[0],
                rem
            ]
            exprs = [
                CyclicSum(b*c * p1[1] * (a-b)*(a-c))**2,
                CyclicSum(b*c * p2[1] * (a-b)*(a-c))**2 ,
                CyclicProduct(a**2) * CyclicSum(b*c*(a-b)**2*(a-c)**2),
                CyclicProduct(a) * CyclicSum(a**5*(b-c)**4),
                CyclicProduct(a) * CyclicProduct((a-b)**2) * p_fin[1],
                CyclicProduct(a**2) * CyclicSum(a*b) * multiplier,
            ]
            return sum_y_exprs(y, exprs) / multiplier




def _sos_struct_octic_symmetric_quadratic_form(poly, coeff: Coeff):
    """
    Let F0 = s(a2(s(a2+ab)-bc)2(a-b)(a-c)).
    Then we have
    F_{x,y} = F0 - 2s(a2(s(a2+ab)-bc)(a-b)(a-c))f(a,b,c) + s(a2(a-b)(a-c))f(a,b,c)^2 >= 0

    See proof at class _octic_sym_axis.
    Such F_{x,y}, G_{x,y} has the property that the symmetric axis is a multiple of a^2 * (a-1)^2 * (...)^2.
    For more general septic symmetric polynomials, we can first decompose its symmetric axis
    into several F_{x,y} and then combine them together.

    For a more primary case, see `_sos_struct_sextic_symmetric_quadratic_form`.

    Examples
    ---------
    s(a2(a-b)(a-c))s(a2-ab)2-p(a-b)2s(3/2a2)

    s(a6(a-b)(a-c))-p(a-b)2(1/2s(a2)+1/6s(a)2)
    """
    return

    a, b, c = coeff.gens
    CyclicSum, CyclicProduct = coeff.cyclic_sum, coeff.cyclic_product

    # We require multiplicity 2 at (1,1,0) along the symmetric axis.
    sym = poly.subs({b:1,c:1}).div(Poly([1,-2,1,0,0], a))
    if not sym[1].is_zero:
        return None

    sym_axis = _restructure_quartic_polynomial(sym[0])
    if sym_axis is None:
        return None
    t, coeff0, x, y, rem_coeff, rem_ratio = sym_axis

    # ker_coeff is the remaining coefficient of (a-b)^2(b-c)^2(c-a)^2*s(a^2) and (a-b)^2(b-c)^2(c-a)^2*s(ab)
    # of Poly - (t*s(a2-ab)s(a3-a2b-a2c+abc)2 + coeff0 * F(x,y) + rem * s(a^2(a-b)(a-c))s(a^2+rab)^2)
    ker_coeff1 = poly.coeff_monomial((6,2,0)) - (2*t + coeff0 * (2*x**2 - 2*x*y - 2*x + y**2 + 1))
    ker_coeff2 = poly.coeff_monomial((5,3,0)) - (3*t + coeff0 * (-3*x**2 + 2*x*y + 4*x - y**2 - 2))
    if rem_ratio is sp.oo:
        # degenerates to s(a^2(a-b)(a-c))s(ab)^2
        ker_coeff1 -= rem_coeff
        ker_coeff2 += rem_coeff
    else:
        ker_coeff1 -= rem_coeff*(rem_ratio**2 - 2*rem_ratio + 2)
        ker_coeff2 -= rem_coeff*(-rem_ratio**2 + 2*rem_ratio - 3)
    ker_coeff = (ker_coeff1, ker_coeff2 + 2*ker_coeff1)

    # print('Coeff =', coeff0, 'ker =', ker_coeff)
    # print('  (x,y) =', (x, y), 'ker_std =', ker_coeff / coeff0)

    return _octic_sym_axis.solve(
        coeff0, x, y, ker_coeff, t, rem_coeff, rem_ratio
    )


class _octic_sym_axis(DomainExpr):
    """
    Let F0 = s(a^2(s(a^2+ab)-bc)^2(a-b)(a-c)) and f(a,b,c) = s(xa^2 + yab).
    Define
    F_{x,y}(a,b,c) = F0 - 2s(a^2(s(a^2+ab)-bc)(a-b)(a-c))f(a,b,c) + s(a^2(a-b)(a-c))f(a,b,c)^2.

    Then F_{x,y} >= 0 because
    F_{x,y} * s(a^2(a-b)(a-c)) = (s(a^2(s(a^2+ab)-bc)(a-b)(a-c)) - s(a^2(a-b)(a-c))f(a,b,c))^2 + 3p(a^2)p(a-b)^2

    The class provides different methods to solve F_{x,y}(a,b,c) >= 0. There are also
    two types of solvers.
    """

    def rem_poly(self, rem_coeff, rem_ratio):
        a, b, c = self.gens
        CyclicSum, CyclicProduct = self.cyclic_sum, self.cyclic_product
        return rem_coeff * (CyclicSum(a**2 + rem_ratio*a*b)**2 if not rem_ratio is sp.oo else CyclicSum(a*b)**2)

    def _wrap_F(self, f_type, f_solver):
        if f_type == 0:
            def _F(self, x, y, coeff0, ker_coeff, t_coeff, rem_coeff, rem_ratio):
                solution, flg = f_solver(x, y) #, ker_coeff/coeff0)
                if solution is not None:
                    a, b, c = self.gens
                    CyclicSum, CyclicProduct = self.cyclic_sum, self.cyclic_product
                    solution = Add(
                        coeff0 * solution,
                        t_coeff/2 * CyclicProduct((a-b)**2) * CyclicSum(a*(a-b)*(a-c))**2,
                        Rational(1,2) * CyclicSum((b-c)**2*(b+c-a)**2) * self.rem_poly(rem_coeff, rem_ratio)
                    )
                return solution, flg
        return _F

    def solve(self, coeff0, x, y, ker_coeff, t_coeff, rem_coeff, rem_ratio):
        SOLVERS = [
            # type, func
            # (0, self._F_regular),
        ]
        solutions = []

        for (solver_type, solver) in SOLVERS:
            f = self._wrap_F(solver_type, solver)
            solution, flg = f(x, y, coeff0, ker_coeff, t_coeff, rem_coeff, rem_ratio)
            # print(solver, solution, flg)
            if flg == 0:
                return solution
            elif flg == 1:
               solutions.append(solution)

        if len(solutions) > 0:
            return solutions[0]
