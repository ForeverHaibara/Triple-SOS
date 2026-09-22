from typing import TYPE_CHECKING

import sympy as sp
from sympy import Add, Integer, Poly, Rational, Symbol, sign
from sympy import MutableDenseMatrix as Matrix

# from .sextic_symmetric import _restructure_quartic_polynomial
from .quartic import structsos_quartic_param
from .utils import CommonExpr
from ..utils import (
    DomainExpr,
    intervals,
    quadratic_weighting,
    rationalize_func,
    sum_y_exprs,
)

if TYPE_CHECKING:
    from ....utils.expressions import Coeff


def _solve_inverse_quartic(coeff: 'Coeff', m, p, n, r):
    """
    Solve a symmetric inverse quartic expression fast without callbacks. It only involves
    monoms inside the triangle `(a^4b^4, a^4c^4, b^4c^4)`. Hence it is equivalent to a
    quartic with respect to `ab`, `bc` and `ca`.

    Formally, it solves the problem:
    `s(a^4b^4 + p(a^4b^3c+a^4bc^3) + qa^4b^2c^2 + ra^3b^3c^2) >= 0`.
    """
    a, b, c = coeff.gens
    CyclicSum, CyclicProduct = coeff.cyclic_sum, coeff.cyclic_product
    if m >= 0 and m + 2*p + n + r >= 0:
        if m != 0 and (n - ((p / m)**2 - 1) * m) >= 0:
            y = [
                m / 2,
                (n - ((p / m)**2 - 1) * m) / 2 if m != 0 else Integer(0),
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


def _sqrt_f6(f):
    """
    Compute the squareroot of a degree 6 polynomial.
    """
    if f.degree() != 6:
        return None

    _, A5, A4, A3, A2, A1, A0 = f.monic().rep.to_list()

    u = A5 / 2
    v = (A4 - u**2) / 2
    w = (A3 - 2*u*v) / 2
    return (u, v, w) if (A2, A1, A0) == (v*v + 2*u*w, 2*v*w, w**2) else None


def _poly_from_list(values, gen):
    """Build a univariate polynomial from coefficients in descending order."""
    return Poly.from_list(values, gen)


def _build_quad_form3(m00, m01, m02, m22):
    """Build the symmetric 3-by-3 matrix from parameters."""
    return Matrix([
        [m00, m01, m02],
        [m01, m00, m02],
        [m02, m02, m22],
    ])


def structsos_octic_symmetric(coeff, real=True):
    if not all(coeff((i,j,k)) == coeff((j,i,k)) for (i,j,k) in ((7,1,0),(6,2,0),(5,3,0),(5,2,1),(4,3,1))):
        return

    solution = _structsos_octic_symmetric_sqr_axis(coeff)
    if solution is not None:
        return solution

    if not coeff.is_rational:
        return

    if coeff((8,0,0)) == 0 and coeff((7,1,0)) == 0:
        if coeff((6,2,0)) == 0 and coeff((5,3,0)) == 0:
            return _structsos_octic_symmetric_hexagram(coeff)
        return _structsos_octic_symmetric_hexagon(coeff)

    if coeff((8,0,0)) != 0:
        return _structsos_octic_symmetric_quadratic_form(coeff.as_poly(), coeff)


def _octic_symmetric_hexagon_quad_form_solution(coeff, quad_form):
    """Convert an octic symmetric quadratic form into a structural SOS."""
    a, b, c = coeff.gens
    cyclic_sum, cyclic_product = coeff.cyclic_sum, coeff.cyclic_product
    first = (quad_form[0, 0] - quad_form[0, 1]) / 2 \
        * cyclic_sum(a)**2 * cyclic_product((a - b)**2)

    def mapping(vector):
        x, y = vector
        if x == 1 and y == 2:
            return cyclic_sum(a)**2 * cyclic_sum(a * (b - c)**2)**2
        if x == 1 and y == -2:
            return cyclic_sum(a * b * (a - b)**2)**2
        if x == 0 and y == 1:
            return cyclic_sum(a**2 * (b - c)**2)**2 / 4
        p1 = a**3*b + a*b**3 - a**2*b*c - a*b**2*c
        p2 = a**2*b**2 - a*b*c**2
        return cyclic_sum((x*p1 + y*p2).expand().together())**2

    second = quadratic_weighting(
        coeff,
        (quad_form[0, 0] + quad_form[0, 1]) / 2,
        quad_form[0, 2] * 2,
        quad_form[2, 2],
        mapping=mapping,
    )
    if second is None:
        return None
    return first + second


def _structsos_octic_symmetric_hexagon_sdp(coeff: 'Coeff'):
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

    See similar methods in _structsos_sextic_full_sdp.

    TODO: 1. Handle w1 == 0. 2. Handle c620 +- 2c611 == 0.

    Examples
    --------
    => (s(a2(b2-c2)2)-3/8p(a-b)2)s(a2)+s(a4(b-c)2)s(a2)/8

    => s(4a4b2-7a4bc+4a4c2+8a3b3-12a3b2c-12a3bc2+15a2b2c2+a4(b-c)2)s(a2-ab)

    => (85/336p(a-b)2+s(bc(a-b)(a-c)(a+b)(a+c))-16/15s(a2bc(b-c)2))s(a2-ab)

    => s(a3(bc(a+b+c)((a-2b)(a-2c)-bc)+a(a-b-c)(a-3b-3c)(b-c)2))

    => s(a2(a-(b+c))2((b-c)2+bc)(a-b)(a-c))
    """
    c620, c530, c440, c611, c521, c431, c422 = [
        coeff(_) for _ in ((6,2,0),(5,3,0),(4,4,0),(6,1,1),(5,2,1),(4,3,1),(4,2,2))]
    if (not coeff.is_rational) or c620 <= 0 or coeff.poly111() != 0:
        return None

    w1 = c521 + c611 + 2*c620
    w2 = c431 + 2*c440 + 3*c530 - c611
    w4 = c422 + 4*c431 + 5*c440 + 8*c521 + 12*c530 + 8*c611 + 18*c620
    if w4 < 0:
        # w4 = 1/2 * (df^2)/(da^2) at a = b = c = 1
        return None

    a, b, c = coeff.gens
    CyclicSum = coeff.cyclic_sum

    def _sol_to_result(sol):
        if sol is None:
            return None
        u210, u102, u201, u111, r, quad_form = sol
        quad_form_sol = _octic_symmetric_hexagon_quad_form_solution(coeff, quad_form)
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

        M00t, M01t, M02t, M22t = [_poly_from_list(_, t) for _ in (_M00, _M01, _M02, _M22)]
        det = (M22t * (M00t + M01t) - 2*M02t**2).div((t*(t+1)).as_poly(t))[0]


        def _is_valid(t):
            return sign(det(t)) * sign(t) * sign(t+1) >= 0 and M00t(t) >= abs(M01t(t))

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
                u210 = Integer(1)
                u201 = x_ * t
                u102 = 2 + w2/w1 + x_
                u111 = u111_/(w1*(t + 1)*p4)
            else:
                x_ = -w2/w1/(t+1)*(p4 + 9*t*w1*w2)
                r = x_**2/(162*t**2*w2**2*w4)
                u210 = Integer(0)
                u201 = t
                u102 = Integer(1)
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

            u210 = Integer(1)
            u201 = x_ * t
            u102 = 2 + w2/w1 + x_

        quad_form = _build_quad_form3(M00t, M01t, M02t, M22t)
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

        M00t, M01t, M02t, M22t = [_poly_from_list(_, t) for _ in (_M00, _M01, _M02, _M22)]
        det = (M22t * (M00t + M01t) - 2*M02t**2).div((t).as_poly(t))[0]


        def _is_valid(t):
            return t != 0 and sign(det(t)) * sign(t) >= 0 and M00t(t) >= abs(M01t(t))

        t = rationalize_func(det.diff(), _is_valid)
        if t is None:
            return None

        u210 = Integer(1)
        u201 = t
        u102 = -(t + 1)*(3*t*w2 - t*w4 + 2*w4)/(3*t*w2 - t*w4 - w4)
        u111 = -(3*t**2*w2 - t**2*w4 - 3*t*w2 + 3*t*w4 - 5*w4)/(3*t*w2 - t*w4 - w4)
        reg = 162*t**2*w4
        r = (3*t*w2 - t*w4 - w4)**2 / reg
        M00t, M01t, M02t, M22t = [f(t)/reg for f in (M00t, M01t, M02t, M22t)]
        quad_form = _build_quad_form3(M00t, M01t, M02t, M22t)
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
            u210 = Integer(1)
            u201 = y
            u102 = x / (-w1) - y - 1

            u111 = -2*u102 - 3*u201 - 1
            reg = (3*u201*(u102 + u201 + 1))
            r = -w1/(2*reg)

            M00t = (3*c620*u102*u201 + 3*c620*u201**2 + 3*c620*u201 + u201**2*w1 - u201*w1 + w1)/reg
            M01t = (3*c611*u102*u201/2 + 3*c611*u201**2/2 + 3*c611*u201/2 - u201**2*w1/2 + 2*u201*w1 - w1/2)/reg
            M02t = (3*c530*u102*u201/2 + 3*c530*u201**2/2 + 3*c530*u201/2 + u102*u201*w1 - u102*w1/2)/reg
            M22t = (3*c440*u102*u201 + 3*c440*u201**2 + 3*c440*u201 - 3*c611*u102*u201 - 3*c611*u201**2 - 3*c611*u201 + u102**2*w1 + 3*u201**2*w1 - 6*u201*w1)/reg
            quad_form = _build_quad_form3(M00t, M01t, M02t, M22t)
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
            func_z_sym = _poly_from_list(_func_z_sym, y)
            func_z_sym_lb = func_z_sym - (w1**2 / (2*c620 - c611) * (y-1)**2).as_poly(y)
            func_z_det = _poly_from_list(_func_z_det, y)
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
                y_ = rationalize_func(_poly_from_list([a0, b0, c0], y), _is_valid, direction = 1)

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
            eq1, eq2 = _poly_from_list(_eq1, y), _poly_from_list(_eq2, y)

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
            r = Integer(0)
            u111, u102, u201 = Integer(0), Integer(0), Integer(0)
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
            M02t = Integer(0)
            M22t = w5 / (c611 + 2*c620)

        u210 = Integer(1)
        if c611 + 2*c620 != 0:
            u102 = -c530/(c611 + 2*c620)
            u201 = Integer(0)
            u111 = -2*u102 - 1
        quad_form = _build_quad_form3(M00t, M01t, M02t, M22t)
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


def _structsos_octic_symmetric_hexagon(coeff: 'Coeff'):
    """
    Try to solve symmetric octic hexagon, without terms a^8, a^7b and a^7c.

    For octics and structural method, the core is not to handle very complicated cases.
    Instead, we explore the art of sum of squares by using simple tricks.
    """
    c1, c2, c3, c4 = [coeff(_) for _ in ((6,2,0),(5,3,0),(6,1,1),(5,2,1))]
    if c1 < 0 or 2*c1 + c3 < 0:
        return None

    solution = _structsos_octic_symmetric_hexagon_sdp(coeff)
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
            y_ = -2*(2*c1 + c2 + c3 + c4)/(2*c1 + c3) if 2*c1 + c3 != 0 else Integer(0)
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
        return _structsos_octic_symmetric_hexagram(coeff)

    return None


def _structsos_octic_symmetric_hexagram(coeff: 'Coeff'):
    """
    Solve octic symmetric hexagram, where all terms are inside the triangle (a^6bc,...) and (a^4b^4,...).

    The idea is to write the problem to s(bc(xa^4 + ya^3(b+c) + za^2(b^2+c^2) + wa^2bc + uabc(b+c) + vb^2c^2)(a-b)(a-c)).
    Then, we use the following lemma: if f(a,b,c) and g(a,b,c) are both symmetric polynomials with respect to b,c.
    Then, sum f(a,b,c)(a-b)(a-c) * sum g(a,b,c)(a-b)(a-c) - sum f(a,b,c)g(a,b,c)(a-b)(a-c)
    must be a multiple of p(a-b)2.
    A common choice of g is g(a,b,c) = 1.

    TODO: Restructure the function. It is too messy.

    Examples
    --------
    => s(bc(a2+1/2a(b+c)-bc)2(a-b)(a-c))

    => s((a-b)2(a+b-3c)2)s(a2b2)+2s(a2(b-c)2(ab+ac-3/2bc)2)-p(a-b)2s(2a2-2ab)

    => s(2a6bc-3a5b2c-3a5bc2+a4b4+3a4b2c2)

    => s(bc(2a4+a3b+a3c+a2b2+9a2bc+a2c2-3ab2c-3abc2+b2c2)(a-b)(a-c))

    => 24s((a+b-c)(a-b)2(a+b-3c)2)p(a)+s(a2b2(ab-ac)(ab-bc))

    => 256p(a)s((64a+(b+c))(a+b-59/16c)(a+c-59/16b)(a-b)(a-c))+s(a2b2(ab-bc)(ab-ca)) # doctest:+SKIP

    => s(bc(a-b)(a-c)(a-2b)(a-2c)(a-3b)(a-3c))

    => s(bc(a-b)(a-c)(a2-2a(b+c)+5bc)(a-2b)(a-2c))

    => s(a4)s(a4)-3abcs(a5)-s((a2-bc)4)
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
            c4 = Integer(0)
            break
        if bound_ == 0:
            # more special, v = xt^2
            # f = x(a^2 + r1*a(b+c) + t*bc)^2 + (u - ty)abc(b+c)
            #    + (z - y^2/(4x))a^2(b-c)^2 + (balance - (2*t*x**2 + y**2)/x) * a^2bc
            # WLOG z = y^2/(4x)
            h = Integer(0)
            c2 = Integer(0)
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


def _structsos_octic_symmetric_sqr_axis(coeff: 'Coeff'):
    """
    Solve the symmetric octic inequality
    ```
    F(a,b,c) = s((a**3-u/2*a**2*(b+c)+v*a*b*c-w/2*b*c*(b+c))**2*(a-b)*(a-c)) + p(a-b)**2*(x*s(a**2-a*b)+y*s(a)**2)
    ```
    The symmetric axis of `F` is `(a**3 - u*a**2 + v*a - w)**2*(a - 1)**2`.

    Examples
    --------
    :: ineqs = []

    => 6s(a6(a-b)(a-c))-p(a-b)2(3s(a2)+s(a)2)

    => s(a4(sqrt(4/3)a-b-c)2(a-b)(a-c))

    => (3s(a4(a-b-c)2(a-b)(a-c))+p(a-b)2s(a)2)+4p(a-b)2s(14a2-17ab)

    => s(a2(3a-2b-2c)4(a-b)(a-c))

    => s(a8-4a7b-4a7c+33a6b2-44a6bc+33a6c2-96a5b3+84a5b2c+84a5bc2-96a5c3+132a4b4-36a4b3c-183a4b2c2-36a4bc3+132a3b3c2)

    => s((a-b)(a-c)(a-2b)(a-2c)(a-3b)(a-3c)(a-4b)(a-4c))

    => (4(s((a3-5/4a2(b+c)+abc)2(a-b)(a-c))+p(a-b)2(-17/48s(a2-ab)+7/24s(a)2)))
    """
    if coeff((8,0,0)) <= 0:
        return
    a, b, c = coeff.gens
    poly = coeff.as_poly()
    axis = poly.eval((1,1))
    axis, rem = axis.div(coeff.from_list([1, -2, 1], (c,)).as_poly())
    if not rem.is_zero:
        return
    _sqrt = _sqrt_f6(axis)
    if _sqrt is None:
        return
    u, v, w = _sqrt
    u, w = -u, -w
    lc = axis.rep.LC()
    if u - v + w - 1 == 0:
        # TODO: has (a-1)**4
        return

    margin = poly.eval((0, 1))
    x = (margin.rep.eval(-1) / lc - (u + 2)**2)/12
    y = (margin.rep.eval(2) / lc - (7*u**2 - 62*u + 18*w**2 + 12*x + 127))/36

    u, v, w, x, y, lc = [coeff.wrap(i) for i in [u, v, w, x, y, lc]]

    # print(f'octic symmetric sqr axis: (u, v, w, x, y) = {(u, v, w, x, y)}')

    x_para_l = -(u + 2)**2/12
    y_para_l = (2*u**2 - 2*u*w + w**2 - 8*w - 4)/12
    if x < x_para_l:
        # equivalent to F(0,1,-1) < 0
        return


    CyclicSum, CyclicProduct = coeff.cyclic_sum, coeff.cyclic_product

    def _get_solution_parabola(y):
        if u - w - 2 == 0:
            # u = w + 2, but x == x_para_l
            assert y == (u - 4)**2/12
            return CyclicSum((a-b)**2)*CyclicSum(
                (2*a**3-u*a**2*b-u*a**2*c+2*(u+v)/3*a*b*c).together())**2/2

        dt = {
            (0, 0, 3): -4*u*w - 4*u + 2*w**2 + 20*w + 24*y,
            (0, 1, 2): 6*u**2 - 12*u*w - 12*u + 9*w**2 - 36*y + 12,
            (0, 2, 1): -6*u**2 + 4*u*w + 4*u + w**2 + 16*w + 12*y + 12,
            (0, 3, 0): 12*u - 12*w - 24,
            (1, 1, 1): 12*u*v + 4*u*w - 8*u - 12*v*w - 24*v - 8*w**2 + 16*w + 48*y,
            (1, 2, 0): -6*u**2 + 8*u*w + 20*u - w**2 - 16*w - 12*y - 12,
            (2, 0, 1): -6*u**2 + 4*u*w + 4*u + w**2 + 16*w + 12*y + 12,
            (3, 0, 0): 12*u - 12*w - 24
        }
        for m, k in dt.copy().items():
            dt[(m[1], m[0], m[2])] = k
        p1 = coeff.from_dict(dt)
        return CyclicSum(p1.as_poly().expr.together()**2*(a-b)**2) / (288*(u - w - 2)**2)

    if y >= y_para_l:
        x2 = lc/2 * (x - x_para_l)
        y2 = lc *(y - y_para_l)
        return lc * _get_solution_parabola(y_para_l) \
            + x2 * CyclicProduct((a-b)**2)*CyclicSum((a-b)**2)\
            + y2 * CyclicProduct((a-b)**2)*CyclicSum(a)**2

    if u - w - 2 != 0:
        x_para = (-w**2 + 4*w + 12*y - 4)*(-4*u**2 + 4*u*w - w**2 + 12*w + 12*y + 12)/(48*(u - w - 2)**2)
        if x >= x_para:
            x2 = lc/2 * (x - x_para)
            return lc * _get_solution_parabola(y) \
                + x2 * CyclicProduct((a-b)**2)*CyclicSum((a-b)**2)


    def _get_solution_cubic(z):
        return _solve_octic_symmetric_sqr_axis_cubic(coeff, u, v, w, z)


    eq0 = coeff.from_list([2, v + w - 4], (a,)).as_poly()
    x_cubic = eq0 * coeff.from_list([1, 4 - v - w], (a,)).as_poly()**2
    x_cubic = x_cubic.mul_ground(2/(u - v + w - 1)/27).add_ground((u + 2*v + w + 2)*(7*u + 2*v + w + 14)/108)
    y_cubic = coeff.from_list([4, 0, 2*u**2 + 4*u*v - v**2 - 6*v*w - 6*w**2 + 12*w - 36], (a,)).as_poly()
    y_cubic = y_cubic.mul_ground(-coeff.domain.one/36)

    eq1 = x_cubic.add_ground(-x)
    eq2 = y_cubic.add_ground(-y)

    for z in intervals([eq0, eq1, eq2], coeff.domain):
        x1 = coeff.wrap(x_cubic.rep.eval(z))
        y1 = coeff.wrap(y_cubic.rep.eval(z))
        if x >= x1 and y >= y1:
            cb = _get_solution_cubic(z)
            if cb is not None:
                return lc * cb\
                + (lc * (x - x1))/2 * CyclicProduct((a-b)**2)*CyclicSum((a-b)**2)\
                + (lc * (y - y1)) * CyclicProduct((a-b)**2)*CyclicSum(a)**2

    if u - w - 2 != 0:
        y_para_t = coeff.from_list([-4*(u - w - 2)**2,
            2*u**3 + 2*u**2*v + 4*u**2*w - 18*u**2 - 6*u*v*w - 16*u*v - 9*u*w**2 + 10*u*w \
                + 60*u + 3*v*w**2 + 24*v*w + 20*v + 5*w**3 - 9*w**2 - 44*w - 60], (a,)
            ).as_poly().mul_ground(1/(u - v + w - 1)/12)
        # x_para_t = (-w**2 + 4*w + 12*y - 4)*(-4*u**2 + 4*u*w - w**2 + 12*w + 12*y + 12)/(48*(u - w - 2)**2)
        y_para_t12 = y_para_t.mul_ground(12)
        x_para_t = y_para_t12.add_ground(-(w - 2)**2) * y_para_t12.add_ground(
            -4*u**2 + 4*u*w - w**2 + 12*w + 12).mul_ground(1/(u - w - 2)**2/48)

        area = x_para_t * y_cubic + x_cubic.mul_ground(y) + y_para_t.mul_ground(x)\
                - x_para_t.mul_ground(y) - y_para_t * x_cubic - y_cubic.mul_ground(x)

        for z in intervals([eq0, eq1, area], area.domain):
            xa = x_para_t.rep.eval(z)
            ya = y_para_t.rep.eval(z)
            xb = x_cubic.rep.eval(z)
            yb = y_cubic.rep.eval(z)
            xa, xb, ya, yb = [coeff.wrap(i) for i in [xa, xb, ya, yb]]
            if xa != xb and (x - xa) * (x - xb) <= 0:
                weight = (xb - x) / (xb - xa)
                y_comb = weight*ya + (1 - weight)*yb

                if y >= y_comb:
                    cb = _get_solution_cubic(z)

                    if cb is not None:
                        return (lc * weight) * _get_solution_parabola(ya)\
                            + (lc * (1 - weight)) * cb\
                            + (lc * (y - y_comb)) * CyclicProduct((a-b)**2)*CyclicSum(a)**2
    return


def _solve_octic_symmetric_sqr_axis_cubic(coeff: 'Coeff', u, v, w, z):
    """
    Solve the symmetric octic inequality
    ```
    F(a,b,c) = s((a**3-u/2*a**2*(b+c)+v*a*b*c-w/2*b*c*(b+c))**2*(a-b)*(a-c)) + p(a-b)**2*(x*s(a**2-a*b)+y*s(a)**2)
    ```
    where `(x, y)` lies on a cubic curve parametrized by
    ```
    x = (u + 2*v + w + 2)*(7*u + 2*v + w + 14)/108
        + 2*(-v - w + z + 4)**2*(v + w + 2*z - 4)/(27*u - 27*v + 27*w - 27)
    y = -(2*u**2 + 4*u*v - v**2 - 6*v*w - 6*w**2 + 12*w + 4*z**2 - 36)/36,
    ```
    """
    D = u - v + w - 1
    if D == 0:
        return

    a, b, c = coeff.gens
    CyclicSum, CyclicProduct = coeff.cyclic_sum, coeff.cyclic_product

    if (v + w + 2*z - 4)/(27*D) >= 0:
        dt = {
            (0, 0, 4): 2*u + 2*v + 6*w - 4*z,
            (0, 1, 3): -4*u - 7*v + 3*w + 2*z - 6,
            (0, 2, 2): 10*u + 4*v + 6*w + 4*z + 6,
            (0, 3, 1): -8*u + v + 3*w - 2*z - 18,
            (0, 4, 0): 18,
            (1, 1, 2): 10*u - 14*v - 12*w + 4*z + 24,
            (1, 2, 1): 4*u + 4*v - 6*w - 8*z - 18,
            (1, 3, 0): -10*u - v - 3*w + 2*z,
            (2, 2, 0): 4*u + 10*v + 4*z + 12,
        }
        for m, k in dt.copy().items():
            dt[(m[1], m[0], m[2])] = k
        p1 = coeff.from_dict(dt)
        p2 = CyclicSum(((u+v+3*w-2*z-9)*a**2 + (2*u-4*v+2*z+6)*b*c).together())**2
        return (CyclicSum(p1.as_poly().expr.together()**2*(a-b)**2) / 324\
            + (v + w + 2*z - 4)/(27*(u - v + w - 1)) * CyclicProduct((a-b)**2)*p2)/CyclicSum((a-b)**2)

    # general case: lift 4 degrees and
    # there exists a completely-symmetric solution

    A = 4 - v - w + z
    if A == 0:
        return

    C = 2*u - v + 3*w + 2*z - 6
    U = 9*(w - 4)

    E = 4*A*C - 2*C*D + D*U
    if E == 0:
        return
    R = A*C - 2*C*D + D*U
    S = 5*A*C - 18*A*D + 2*C*D - D*U
    m1, m0 = -R/E, -S/E

    T0 = 15*A*C - 8*A*D - 9*C*D - 2*D**2 + 5*D*U
    Tl = 3*A*C - 4*A*D - 3*C*D + 2*D**2 + D*U
    Tm = 12*A*C - 10*A*D - 9*C*D + 2*D**2 + 4*D*U

    Hl = (
        A**2*C*(3*C - 8*D)
        + A*D*(-6*C**2 + 20*C*D + 2*C*U - 108*C - 8*D*U + 216*D)
        + D**2*(-8*C*D + 2*C*U + 4*D*U - U**2)
    )
    Hm = 2*(
        A**2*C*(6*C - 22*D)
        + A*D*(9*C**2 + 10*C*D - 8*C*U - 54*C + 2*D*U + 108*D)
        + D**2*(-6*C**2 - 4*C*D + 7*C*U + 2*D*U - 2*U**2)
    )
    Hc = (
        A**2*(15*C**2 - 280*C*D + 396*D**2)
        + A*D*(54*C**2 + 52*C*D - 38*C*U + 108*C
               - 144*D**2 + 32*D*U - 216*D)
        + D**2*(-24*C**2 + 8*C*D + 22*C*U - 4*D*U - 5*U**2)
    )

    # With K = 108*A**2, q, g and h have affine numerators in l.
    Q1, Q0 = 1 + 4*m1, 5 + 4*m0
    G1, G0 = -2*(Tl + Tm*m1), -2*(T0 + Tm*m0)
    H1, H0 = -(Hl + Hm*m1), -(Hc + Hm*m0)

    # 4*h*q - g**2 = (aa*l**2 + bb*l + cc)/K**2.
    aa = 4*H1*Q1 - G1**2
    if aa == 0:
        return
    bb = 4*(H1*Q0 + H0*Q1) - 2*G1*G0
    l = -bb/(2*aa)

    # Use the computed extremal l; do not recompute the intermediate blocks.
    K = 108*A**2
    m = m1*l + m0
    N1 = (H1*l + H0)/K
    N2 = (G1*l + G0)/K
    N3 = (Q1*l + Q0)/K

    I = (-u - 2*w + z + 5)*(u + 2*w + 2*z - 5)
    p1 = [
        6*D,
        D*(-3*u - 6),
        -3*D**2 + D*(8*u + 7*w + z - 10) + I,
        D*(2*u - 8*w - 2*z + 20) - 2*I,
        6*D**2 + D*(-10*u - 14*w - 2*z + 20) - 2*I,
        D*(u + 5*w + 2*z - 14) + 2*I,
        D*(-12*u - 6*w - 6*z + 42) - 6*I
    ]
    p2 = [
        coeff.domain.zero,
        3*D,
        D*(-u + w + z - 13) + I,
        D*(-4*u - 2*w - 2*z + 14) - 2*I,
        D*(2*u - 8*w - 2*z + 20) - 2*I,
        -3*D**2 + D*(7*u + 11*w + 2*z - 20) + 2*I,
        18*D**2 + D*(-30*u - 42*w - 6*z + 78) - 6*I
    ]

    def comb_p1_p2(line):
        c1, c2 = line
        monoms = [(6, 0, 0), (5, 1, 0), (4, 2, 0), (4, 1, 1), (3, 3, 0), (3, 2, 1), (2, 2, 2)]
        dt = dict(zip(monoms, [i*c1 + j*c2 for i, j in zip(p1, p2)]))
        for (i,j,k), val in dt.copy().items():
            if j != k and i != j:
                dt[(i,k,j)] = val
        dt[(2,2,2)] = dt[(2,2,2)]/3
        dt = {(i-2, j, k): val for (i,j,k), val in dt.items()}
        return CyclicSum(a**2*coeff.from_dict(dt).as_poly().as_expr().together())**2


    conv = coeff.convert
    e1 = coeff.from_dict({(1,0,0): 1, (0,1,0): 1, (0,0,1): 1}).as_poly()
    e2 = coeff.from_dict({(1,1,0): 1, (0,1,1): 1, (1,0,1): 1}).as_poly()
    e3 = coeff.from_dict({(1,1,1): 1}).as_poly()
    p3 = (e1**3).mul_ground(conv(u + v + 3*w - 2*z - 9)) + e1*e2.mul_ground(conv(6*A))

    p40 = D**3 + 3*(-3*w + z - 6)*D**2 + (-u - 2*w + z + 5)*(3*(u - 4*w + 1)*D - 2*I)
    p41 = 54*A**2*(v + w + 2*z - 4)
    p4 = (e1**3).mul_ground(conv(p40)) + e3.mul_ground(conv(p41))

    def comb_p3_p4(line):
        poly = (p3.mul_ground(conv(line[0])) + p4.mul_ground(conv(line[1])))
        dt = poly.rep.to_dict()
        dt = {
            (2,0,0): dt.get((3,0,0), 0),
            (1,1,0): dt.get((2,1,0), 0),
            (1,0,1): dt.get((2,0,1), 0),
            (0,1,1): dt.get((1,1,1), poly.domain.zero)/3
        }
        poly = coeff.from_dict(dt).as_poly()
        return CyclicProduct((a-b)**2) * CyclicSum(a*poly.as_expr())**2

    # print('params =', (u,v,w,z), (1,2*(m+1),2*m+2+l), (N1,N2,N3))

    L = 1/D**2/36
    part1 = quadratic_weighting(coeff, L, 2*(m + 1)*L, (2*m + 2 + l)*L, mapping=comb_p1_p2)
    if part1 is None:
        return
    part2 = quadratic_weighting(coeff, N1*L, N2*L, N3*L, mapping=comb_p3_p4)
    if part2 is None:
        return

    mul = structsos_quartic_param(coeff, 1, m, l, m, -(2*m + l + 1))
    if mul is None:
        return None

    return (part1 + part2)/mul


def _structsos_octic_symmetric_quadratic_form(poly, coeff: 'Coeff'):
    """
    Let F0 = s(a2(s(a2+ab)-bc)2(a-b)(a-c)).
    Then we have
    `F_{x,y} = F0 - 2s(a2(s(a2+ab)-bc)(a-b)(a-c))f(a,b,c) + s(a2(a-b)(a-c))f(a,b,c)^2 >= 0`

    See proof at class _octic_sym_axis.
    Such F_{x,y}, G_{x,y} has the property that the symmetric axis is a multiple of a^2 * (a-1)^2 * (...)^2.
    For more general septic symmetric polynomials, we can first decompose its symmetric axis
    into several F_{x,y} and then combine them together.

    For a more primary case, see `_structsos_sextic_symmetric_quadratic_form`.

    Examples
    --------
    => s(a2(a-b)(a-c))s(a2-ab)2-p(a-b)2s(3/2a2) # doctest:+SKIP
    """
    return

    # a, b, c = coeff.gens
    # CyclicSum = coeff.cyclic_sum

    # # We require multiplicity 2 at (1,1,0) along the symmetric axis.
    # sym = poly.subs({b:1,c:1}).div(Poly([1,-2,1,0,0], a))
    # if not sym[1].is_zero:
    #     return None

    # sym_axis = _restructure_quartic_polynomial(sym[0])
    # if sym_axis is None:
    #     return None
    # t, coeff0, x, y, rem_coeff, rem_ratio = sym_axis

    # # ker_coeff is the remaining coefficient of (a-b)^2(b-c)^2(c-a)^2*s(a^2) and (a-b)^2(b-c)^2(c-a)^2*s(ab)
    # # of Poly - (t*s(a2-ab)s(a3-a2b-a2c+abc)2 + coeff0 * F(x,y) + rem * s(a^2(a-b)(a-c))s(a^2+rab)^2)
    # ker_coeff1 = poly.coeff_monomial((6,2,0)) - (2*t + coeff0 * (2*x**2 - 2*x*y - 2*x + y**2 + 1))
    # ker_coeff2 = poly.coeff_monomial((5,3,0)) - (3*t + coeff0 * (-3*x**2 + 2*x*y + 4*x - y**2 - 2))
    # if rem_ratio is sp.oo:
    #     # degenerates to s(a^2(a-b)(a-c))s(ab)^2
    #     ker_coeff1 -= rem_coeff
    #     ker_coeff2 += rem_coeff
    # else:
    #     ker_coeff1 -= rem_coeff*(rem_ratio**2 - 2*rem_ratio + 2)
    #     ker_coeff2 -= rem_coeff*(-rem_ratio**2 + 2*rem_ratio - 3)
    # ker_coeff = (ker_coeff1, ker_coeff2 + 2*ker_coeff1)

    # # print('Coeff =', coeff0, 'ker =', ker_coeff)
    # # print('  (x,y) =', (x, y), 'ker_std =', ker_coeff / coeff0)

    # return _octic_sym_axis.solve(
    #     coeff0, x, y, ker_coeff, t, rem_coeff, rem_ratio
    # )


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
        a, b, _ = self.gens
        CyclicSum = self.cyclic_sum
        return rem_coeff * (CyclicSum(a**2 + rem_ratio*a*b)**2 if rem_ratio is not sp.oo else CyclicSum(a*b)**2)

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
