import sympy as sp

from .utils import (
    CyclicSum, CyclicProduct,
    sum_y_exprs, radsimp, rationalize_func, quadratic_weighting
)
from ..sparse import sos_struct_quadratic

a, b, c = sp.symbols('a b c')

def sos_struct_cubic(coeff, real = True):
    """
    Solve cyclic cubic polynomials.

    This function supports irrational coefficients.
    """
    if coeff((2,1,0)) == coeff((1,2,0)):
        return _sos_struct_cubic_symmetric(coeff)
    if coeff((3,0,0)) == 0:
        return _sos_struct_cubic_degenerate(coeff)

    if not coeff.is_rational:
        return _sos_struct_cubic_nontrivial_irrational(coeff)

    solution = _sos_struct_cubic_parabola(coeff)
    if solution is not None:
        return solution

    return _sos_struct_cubic_nontrivial(coeff)


def _sos_struct_cubic_symmetric(coeff):
    """
    Cubic symmetric inequality can be handled with Schur.
    """
    # if not coeff((2,1,0)) == coeff((1,2,0)):
    #     return None
    y = radsimp([
        coeff((3,0,0)),
        coeff((3,0,0)) + coeff((2,1,0)),
        3 * (coeff((3,0,0)) + coeff((2,1,0)) * 2) + coeff((1,1,1))
    ])
    if all(_ >= 0 for _ in y):
        exprs = [
            CyclicSum(a*(a-b)*(a-c)),
            CyclicSum(a*(b-c)**2),
            CyclicProduct(a)
        ]
        return sum_y_exprs(y, exprs)
    return None


def _sos_struct_cubic_degenerate(coeff):
    # if coeff((3,0,0)) != 0:
    #     return None

    p, q, r = coeff((2,1,0)), coeff((1,2,0)), coeff((1,1,1))
    rem = 3 * (p + q) + r
    if p < 0 or q < 0 or rem < 0:
        return None

    if p == q:
        return CyclicSum(a*(b-c)**2) * p + rem * CyclicProduct(a)

    return p * CyclicSum(a**2*b - CyclicProduct(a)) + q * CyclicSum(a**2*c - CyclicProduct(a))\
             + rem * CyclicProduct(a)


def _sos_struct_cubic_parabola(coeff):
    """
    Although we can always multiply s(a) to convert the problem to a quartic one,
    sometimes the cubic inequality does not need to higher the degree.

    Apart from Schur, one of the cases is s(a(a+tb-(t+1)c)2), where the coefficient of s(a2b) and s(a2c)
    are (t^2+4t+1, t^2-2t-2), respectively. This is a parabola (x-y-9)^2 = 36(y+3).
    We will test whether the inequality is a linear (convex) combination of two points on the parabola.
    """
    m, p, q, r = coeff((3,0,0)), coeff((2,1,0)), coeff((1,2,0)), coeff((1,1,1))
    rem = 3 * (p + q + m) + r
    if m < 0 or rem < 0:
        return None

    p, q = p / m, q / m

    if p == q + 3:
        # then the line connecting (p,q) and (1,-2) is parallel to the symmetric axis of the parabola
        if p >= 1:
            return m * CyclicSum(a*(a-c)**2) + m*(p-1) * CyclicSum(a*(b-c)**2) + rem * CyclicProduct(a)

    elif p + 2*q == -3:
        # this is the tangent of parabola at (1,-2)
        if p == 1:
            return m * CyclicSum(a*(a-c)**2) + rem * CyclicProduct(a)

    elif p == 1:
        if -2 <= q <= 22:
            w1 = (q - 22) / (-24) * m
            w2 = m - w1
            return w1 * CyclicSum(a*(a-c)**2) + w2 * CyclicSum(a*(a-4*b+3*c)**2) + rem * CyclicProduct(a)

    else:
        x2 = (13*p**2 + 22*p*q + 18*p + q**2 - 18*q - 27)/(p - q - 3)**2
        # y2 = (q + 2)/(p - 1)*(x2 - 1) - 2

        w1 = (p - x2) / (1 - x2) * m
        w2 = m - w1

        if not 0 <= w1 <= m:
            return None

        t = 2*(p + 2*q + 3)/(p - q - 3)
        return w1 * CyclicSum(a*(a-c)**2) + w2 * CyclicSum(a*(a+t*b-(t+1)*c)**2) + rem * CyclicProduct(a)

    return None


def _sos_struct_cubic_nontrivial(coeff):
    """
    Solve nontrivial cyclic cubic polynomial by multiplying s(a).

    See further details in the theorem of quartic.

    Theorem:
    If and only if p,q >= 0 or p^2q^2 + 18pq - 4p^3 - 4q^3 - 27 <= 0, the inequality
    f(a,b,c) = s(a^3 + p*a^2*b + q*a*b^2 - (p+q+1)*a*b*c) >= 0 is true for all a,b,c >= 0.

    The curve -4*x**3 + x**2*y**2 + 18*x*y - 4*y**3 - 27 = 0 can be parametrized by
    x = -(2*t**3 - 1)/t**2
    y = (t**3 - 2)/t
    Using the parametrization, f_t(t,1,0) = 0. And we have the solution that
    s(a^2-ab)*f_t(a,b,c) = 1/t^2 * s(a(ta-(t-1)b-c)^2(tb-(t-1)c-a)^2) >= 0

    However, cubic polynomials can be solved by lifting at most 1 degree. So we will
    not use the solution above in prior. We convert it to a quartic instead.

    Examples
    ---------
    s(4a3-15a2b+12ab2-abc)

    s(a3+2a2b-3a2c)

    s(a3-26/10a2b+ab2+6/10abc)
    """
    coeff3, p, q, z = coeff((3,0,0)), coeff((2,1,0)), coeff((1,2,0)), coeff((1,1,1))
    if coeff3 <= 0:
        return None
    p, q, z = p / coeff3, q / coeff3, z / coeff3

    if not ((p >= 0 and q >= 0) or p**2*q**2 + 18*p*q - 4*p**3 - 4*q**3 - 27 <= 0):
        return None

    p2, n2, q2 = p + 1, p + q, q + 1
    u = sp.symbols('u')
    equ = (2*u**4 + p2*u**3 - q2*u - 2).as_poly(u)

    def _compute_params(u):
        t = ((2*q2+p2)*u**2 + 6*u + 2*p2+q2) / 2 / (u**4 + u**2 + 1)
        p_, n_, q_ = p2 - t, n2 + 2*t*u, q2 - t*u**2
        return t, p_, n_, q_
    def _check_valid(u):
        if u < 0:
            return False
        t, p_, n_, q_ = _compute_params(u)
        if 3*(1 + n_) >= p_**2 + p_*q_ + q_**2:
            return True
        return False

    u = rationalize_func(equ, _check_valid)

    t, p_, n_, q_ = _compute_params(u)
    # print(u, t, 3*(1+n_)**2 - (p_**2+p_*q_+q_**2))
    
    y = [
        coeff3 / 2,
        -(p_**2 + p_*q_ + q_**2 - 3*(1 + n_))/6 * coeff3,
        t * coeff3,
        (3*(1 + p + q) + z) * coeff3
    ]
    if y[-1] < 0:
        return None
    exprs = [
        CyclicSum((a**2-b**2+(p_+2*q_)/3*a*c - (2*p_+q_)/3*b*c + (p_- q_)/3*a*b)**2),
        CyclicSum(a**2*(b-c)**2),
        CyclicSum(a*b*(a-c - u*(b-c)).expand()**2),
        CyclicSum(a**2*b*c)
    ]
    return sum_y_exprs(y, exprs) / CyclicSum(a)

    # poly2 = poly * (a + b + c).as_poly(a,b,c)
    # solution = recursion(poly2)
    # if solution is not None:
    #     return solution / CyclicSum(a)


def _sos_struct_cubic_nontrivial_irrational(coeff):
    """
    Use ultimate theorem for cubic to handle general cases, including irrational coefficients.

    Theorem:
    If and only if p,q >= 0 or p^2q^2 + 18pq - 4p^3 - 4q^3 - 27 <= 0, the inequality
    f(a,b,c) = s(a^3 + p*a^2*b + q*a*b^2 - (p+q+1)*a*b*c) >= 0 is true for all a,b,c >= 0.

    The former is rather simple. The latter is more complicated. We have that
    f(a,b,c) * s(ab) = (p + q + 3) p(a)s(a^2-ab) + ts(c(a^2-b^2+u(ab-ac)+v(bc-ab))^2) + (1-t)s(c(a-b)^4) >= 0.
    where
    D = -16(p^2q^2 + 18pq - 4p^3 - 4q^3 - 27)
    u, v = (2*p**2 - 6*q) / (9 - p*q), (2*q**2 - 6*p) / (9 - p*q)
    t = (9 - p*q)**2 / (p + q + 3) / (3*(p - q)**2 + (6 - p - q)**2)

    Examples
    -------
    s(a)3-27abc-2sqrt(3)s((a-b)3)

    s((2a+b)(a-sqrt(2)b)2-2a3+(6sqrt(2)-7)abc)

    s(a3-a2b)+(sqrt(13+16sqrt(2))-1)/2s(ab2-a2b)

    References
    -------
    [1] http://kuing.infinityfreeapp.com/forum.php?mod=viewthread&tid=10631&extra=
    """
    m, p, q = coeff((3,0,0)), coeff((2,1,0)), coeff((1,2,0))
    rem = radsimp(3 * (m + p + q) + coeff((1,1,1)))
    if m < 0 or rem < 0:
        return None
    if p >= 0 and q >= 0:
        y = radsimp([m / 2, p, q, rem])
        exprs = [
            CyclicSum(a) * CyclicSum((b-c)**2),
            CyclicSum(a**2*b - CyclicProduct(a)),
            CyclicSum(a**2*c - CyclicProduct(a)),
            CyclicProduct(a)
        ]
        return sum_y_exprs(y, exprs)

    p, q = p / m, q / m
    det = radsimp(-16*(p**2*q**2 + 18*p*q - 4*p**3 - 4*q**3 - 27))
    if det < 0:
        return None
    u, v = (2*p**2 - 6*q) / (9 - p*q), (2*q**2 - 6*p) / (9 - p*q)
    t = (9 - p*q)**2 / (p + q + 3) / (3*(p - q)**2 + (6 - p - q)**2)
    u, v, t = radsimp([u, v, t])

    y = radsimp([
        (p + q + 3) * m / 2,
        t * m,
        (1 - t) * m,
        rem
    ])
    exprs = [
        CyclicProduct(a) * CyclicSum((a-b)**2),
        CyclicSum(c*(a**2 - b**2 + u*(a*b-a*c) + v*(b*c-a*b))**2),
        CyclicSum(c*(a-b)**4),
        CyclicSum(a**2*b**2*c)
    ]
    return sum_y_exprs(y, exprs) / CyclicSum(a*b)


#####################################################################
#
#                              Acyclic
#
#####################################################################

def sos_struct_acyclic_cubic(coeff, real = True):
    """
    Solve acyclic cubic polynomials.

    Nonnegative cubic polynomials will have at most one interior zero. If there are
    two, we can consider the set 0 <= f(a,b,c) < eps where eps is a sufficiently small
    positive number. Then the set has two connected components, and a line intersecting
    the two gives four intersections, a contradiction to degree 3. Tightest cubic
    polynomials can have one zero in the interior and three zeros on the borders.
    """
    solution = _sos_struct_acyclic_cubic_symmetric(coeff)
    if solution is not None:
        return solution

    solution = _sos_struct_acyclic_cubic_hexagon(coeff)
    if solution is not None:
        return solution



def _sos_struct_acyclic_cubic_hexagon(coeff):
    """
    Solve acyclic cubics in the form of
    ?a^2b+?ab^2+?ac^2+?bc^2+?a^2c+?b^2c >= ??abc.

    Examples
    ---------
    (25a2b+14a2c+20ab2-82abc+5ac2+19b2c+10bc2-2abc)

    (5a2b+a2c+5ab2-10abc+ac2+b2c+bc2)
    """
    if any(coeff(_) != 0 for _ in ((3,0,0), (0,3,0), (0,0,3))):
        return None
    corners = [coeff(_) for _ in ((1,2,0),(1,0,2),(0,1,2),(2,1,0),(2,0,1),(0,2,1))]
    if any(_ < 0 for _ in corners):
        return None

    corners = [(corners[2*i], corners[2*i+1]) for i in range(3)]
    center = coeff((1,1,1))

    gap = center/2 + sum(sp.sqrt(x * y) for x, y in corners)
    if gap >= 0:
        # very easy case
        zs = [None, None, None]

        def _get_check_valid_func(z0):
            def _check_valid(z):
                return z0 <= z <= 0 and z - z0 <= gap / 3
            return _check_valid

        for i, (x, y) in enumerate(corners):
            z0 = -2 * sp.sqrt(x * y)
            if isinstance(z0, sp.Rational):
                zs[i] = z0
            else:
                _check_valid_func = _get_check_valid_func(z0)
                eqz = sp.Poly([1, 0, -4*x*y], sp.Symbol('z'))
                zs[i] = rationalize_func(eqz, _check_valid_func, validation_initial = lambda z: z <= 0, direction = -1)
            if zs[i] is None:
                return None

        combs = [(a,b,c), (b,c,a), (c,a,b)]
        exprs = [quadratic_weighting(x, z, y, a=b0, b=c0)*a0 for (x,y), z, (a0, b0, c0) in zip(corners, zs, combs)]
        return sum(exprs) + (center - sum(zs))*(a*b*c)


def _sos_struct_acyclic_cubic_symmetric(coeff):
    """
    Solve acyclic cubic polynomials that are symmetric with respect to two variables.

    WLOG we assume it is symmetric with respect to a and b.
    Note that cubic polys have at most one interior zero, and in this case it should lies
    on the symmetric axis. Then we can always subtract some c*(a-b)^2 so that the poly
    has two roots on the two borders a = 0 or b = 0.
    Also, we can always subtract some a*b*c so that the poly has one interior root. To
    conclude, it suffices to consider polys with two roots on the border and one interior root
    on the symmetric axis.

    Theorem: let z >= 0 and (-8*t*w+4*w*z-9)/w >= 0. Define
    w0 = 2*(5*t**2 - 2*t*z + 2*z**2)/3
    y0 = 4*(-2*t + z)**3/27
    y1 = -3*(-2*t*w + w*z - 3)**2/w**3
    sym_c = (-2*t*w + w*z - 9)/(3*w)
    F(a,b,c) = ((t*a-c)**2*(z*a+c)+(t*b-c)**2*(z*b+c)-c**3+(-t**2*z+y0+y1)*a*b*(a+b)+(w0+w*y1)*a*b*c)
    Then F(a,b,c) >= 0.

    Equality cases at (1,0,t),(0,1,t) and (1,1,sym_c).
    Moreover, if -8*t*w+4*w*z-9 = 0, then there exists an extra equality case at (1,1,0).

    Proof:
    (a+b)*F(a,b,c) = z*(t*a+t*b-c)**2*(a-b)**2 + (-8*t*w+4*w*z-9)*4/(3*w)*a*b*(sym_c/2 *(a+b) - c)**2
                    + c*(a*(t*a+(sym_c-t)*b-c)**2 + b*(t*b+(sym_c-t)*a-c)**2) >= 0.

    TODO: Solve the inequality without lifting the degree if possible.

    Examples
    ---------
    9s(a3)-3s(a)s(ab)-5s(a)(a-c)2

    a(a-b)(a-c)+4/27(b+c)(b-c)2

    (2a3+2a2b-8a2c+2ab2+21abc+4ac2+2b3-8b2c+4bc2+9c3)

    (sqrt(2)b-c)2(b+c)+(sqrt(2)a-c)2(a+c)-c3+4abc

    (a+b+2c)((a+b-(sqrt(3)+1)/2c)2) + 4(a-b)2c
    """
    a, b, c = None, None, None
    if all(coeff((i,j,k)) == coeff((j,i,k)) for (i,j,k) in ((3,0,0),(2,1,0),(2,0,1),(1,0,2))):
        a, b, c = sp.symbols("a b c")
        x0, x1, x2, x3, x4, x5 = [coeff(_) for _ in ((0,0,3),(1,0,2),(2,0,1),(3,0,0),(2,1,0),(1,1,1))]
    if all(coeff((i,j,k)) == coeff((i,k,j)) for (i,j,k) in ((0,3,0),(0,2,1),(1,2,0),(2,1,0))):
        a, b, c = sp.symbols("b c a")
        x0, x1, x2, x3, x4, x5 = [coeff(_) for _ in ((3,0,0),(2,1,0),(1,2,0),(0,3,0),(0,2,1),(1,1,1))]
    if all(coeff((i,j,k)) == coeff((k,j,i)) for (i,j,k) in ((0,0,3),(1,0,2),(0,1,2),(0,2,1))):
        a, b, c = sp.symbols("c a b")
        x0, x1, x2, x3, x4, x5 = [coeff(_) for _ in ((0,3,0),(0,2,1),(0,1,2),(0,0,3),(1,0,2),(1,1,1))]


    if a is None:
        # it is not symmetric
        return
    if x0 < 0:
        return None

    if x0 == 0:
        # This is a degeneated case, the symmetric axis and the border are now quadratic.
        if x1 < 0 or x3 < 0:
            return None
        if x3 == 0:
            if not (x4 >= 0 and x2 >= 0 and 4*sp.sqrt(x4*x1) + 2*x2 + x5 >= 0):
                return None
            p1 = x2 * c*(a-b)**2
            p2 = quadratic_weighting(x4, x5/2 + x2, x1, a=a, b=c) * b
            p3 = quadratic_weighting(x4, x5/2 + x2, x1, a=b, b=c) * a
            return p1 + p2 + p3
        if x2 < 0 and 4*x1*x3 < x2**2:
            return None
        # The following x40 makes the symmetric axis have a multiplicity root (a,b,c) = (1,1,sym_c)
        # so either we require x4 >= x40, or the symmetric axis degree 1 term >= 0.
        x40 = radsimp(-(16*x1*x3 - 4*x2**2 - 4*x2*x5 - x5**2)/(16*x1))
        r = radsimp(x2/x3/2) if 4*x1*x3 >= x2**2 else sp.S(0)
        ker = radsimp(x1 - x3*r**2)
        if x1 > 0 and x4 >= x40:
            sym_c = radsimp((-2*x2 - x5)/(4*x1))
            w = radsimp(1 + r*sym_c)
            p1 = x3*a*(a + r*c - w*b)**2 + x3*b*(b + r*c - w*a)**2
            p2 = (x4 - x40)*a*b*(a+b) + (x2 - 2*r*x3)*c*(a-b)**2
            p3 = ker*a*(-sym_c*b + c)**2 + ker*b*(-sym_c*a + c)**2
            return p1 + p2 + p3
        elif 2*x2 + x5 >= 0 and x3 + x4 >= 0:
            p1 = x3*a*(a - b + r*c)**2 + x3*b*(b - a + r*c)**2 + (x3 + x4)*a*b*(a+b)
            p2 = (x2 - 2*r*x3)*c*(a-b)**2 + (2*x2 + x5)*a*b*c
            p3 = ker*c**2*(a + b)
            return p1 + p2 + p3
        return None



    def _determine_tz(x0, x1, x2, x3):
        """
        Compute t,z such that
        x0*(ta-c)^2(za+c) = x0*c^3 + x1*c^2*a + x2'*c*a^2 + x3'*a^3
        with x2' <= x2 and x3' <= x3. Also, z = x1/x0 + 2*t.
        """
        t = sp.Symbol('t')
        eqt1 = sp.Poly([3*x0, 2*x1, x2], t)
        eqt2 = sp.Poly([-2*x0, -x1, 0, x3], t)
        def _check_valid(t):
            return x1/x0 + 2*t >= 0 and eqt2(t) >= 0

        t_ = None
        if x2 >= 0 and _check_valid(0):
            t_ = sp.S(0)
        elif eqt1(-x1/x0/2) >= 0 and eqt2(-x1/x0/2) >= 0:
            t_ = radsimp(-x1/x0/2)
        else:
            # sometimes it is tight for irrational coefficients, we detect it first
            eqgcd = sp.gcd(eqt1, eqt2)
            if eqgcd.degree() == 1:
                t_ = radsimp(-eqgcd.coeff_monomial((0,)) / eqgcd.coeff_monomial((1,)))
                if x1/x0 + 2*t_ < 0:
                    t_ = None

        if t_ is None:
            t_ = rationalize_func(eqt1, _check_valid, direction = 1)
        if t_ is not None:
            return radsimp([t_, x1/x0 + 2*t_])

    def _determine_w(x0, x4, x5, t, z):
        """
        Determine w such that
        w0 = 2*(5*t**2 - 2*t*z + 2*z**2)/3
        y0 = 4*(-2*t + z)**3/27
        y1 = -3*(-2*t*w + w*z - 3)**2/w**3
        (-t**2*z+y0+y1) <= x4/x0 and (w0+w*y1) <= x5/x0.
        """
        w = sp.Symbol('w')
        w0 = radsimp(2*(5*t**2 - 2*t*z + 2*z**2)/3)
        y0 = radsimp(4*(-2*t + z)**3/27)
        eqw1 = (radsimp(x5/x0 - w0)*w**2 + 3*(-2*t*w + w*z - 3)**2).as_poly(w)
        eqw2 = (radsimp(x4/x0 + t**2*z - y0)*w**3 + 3*(-2*t*w + w*z - 3)**2).as_poly(w)
        eqw3 = ((-8*t*w + 4*w*z - 9)).as_poly(w)
        def _check_valid(w):
            return w != 0 and eqw3(w) * w >= 0 and eqw2(w) * w >= 0
        if z != 2*t:
            w_ = radsimp(9 / (z - 2*t) / 4)
            if eqw1(w_) >= 0 and _check_valid(w_):
                return w_
        if True:
            eqgcd = sp.gcd(eqw1, eqw2)
            if eqgcd.degree() == 1:
                w_ = radsimp(-eqgcd.coeff_monomial((0,)) / eqgcd.coeff_monomial((1,)))
                if _check_valid(w_):
                    return w_

        return rationalize_func(eqw1, _check_valid, direction = 1)

    def _solve_tzw(t, z, w):
        sym_c = radsimp((-2*t*w + w*z - 9)/(3*w))
        p1 = z*(t*a+t*b-c)**2*(a-b)**2 + radsimp((-8*t*w+4*w*z-9)*4/(3*w))*a*b*(sym_c/2 *(a+b) - c)**2
        p2 = c*(a*(t*a+(sym_c-t)*b-c)**2 + b*(t*b+(sym_c-t)*a-c)**2)
        return (p1 + p2)/(a + b)

    tz = _determine_tz(x0, x1, x2, x3)
    # print('tz =' , tz)
    if tz is None:
        return None

    t, z = tz
    y = radsimp([
        x3 - x0*t**2*z,
        x2 - x0*(t**2 - 2*t*z),
    ])
    x42 = x4 + y[0]
    x52 = x5 + 2*y[1]

    w = _determine_w(x0, x42, x52, t, z)
    # print('w =', w)
    if w is None:
        return None


    w0 = radsimp(2*(5*t**2 - 2*t*z + 2*z**2)/3)
    y0 = radsimp(4*(-2*t + z)**3/27)
    y1 = radsimp(-3*(-2*t*w + w*z - 3)**2/w**3)
    y += radsimp([
        x42 - x0*(-t**2*z + y0 + y1),
        x52 - x0*(w0 + w * y1)
    ])
    if all(_ >= 0 for _ in y):
        p1 = (y[0]*(a+b) + y[1]*c).together() * (a-b)**2
        p2 = (y[2]*(a+b) + y[3]*c).together() * a*b
        p3 = x0 * _solve_tzw(t, z, w)
        return p1 + p2 + p3

    return None



#####################################################################
#
#                          Nonhomogeneous
#
#####################################################################

def _sos_struct_nonhom_cubic_symmetric(coeff, real = False):
    """
    Solve nonhomogeneous 3-var symmetric cubic polynomials in
    the form of
    c100*(a+b+c) + c000 + c1*(a^2+b^2+c^2-a*b-b*c-c*a) + c2*(a^2+b^2+c^2-2*(a*b+b*c+c*a)) + a*b*c >= 0.

    Theorem:
    s(a)2(s(a)+3abc+2s(a2-2ab))
    = s((b-c)2(b+c-a)2)+2s(ab(a-b)2)+3s(a)s(ab(c-1)2)+s(a)s((a-b)2)/2

    s(a)2(1+2abc+s(a2-2ab))
    = s((b-c)2(b+c-a)2)/2+s((a-b)2)/2+s(ab(a-b)2)+3s(ab(c-1)2)+2p(a)s(a-1)2

    We always represent the polynomial as 
          t * (3*v**2/4*(a+b+c) + v*(a**2+b**2+c**2-2*(a*b+b*c+c*a)) + a*b*c)
    + (1-t) * (4*v**3 + v*(a**2+b**2+c**2-2*(a*b+b*c+c*a)) + a*b*c) >= 0

    Examples
    ---------
    4abc+9s(a2)-14s(ab)+4s(a)+4

    29/3s(a2)+13/3s(a)-58/3s(bc)+1+15abc

    1+2abc+s(a2-2ab)

    s(2a2-4ab)+3abc+s(a)

    s(a2-ab)-2s(a)+4+2abc
    """
    if coeff((3,0,0)) or coeff((2,1,0)):
        return None

    c100, c000, c200, c110, c111 = [coeff(_) for _ in [(1,0,0), (0,0,0), (2,0,0), (1,1,0), (1,1,1)]]
    if c111 == 0:
        return sos_struct_quadratic(coeff)
    elif c111 < 0 or c000 < 0:
        return None
    # normalize
    c100, c000, c200, c110 = radsimp([_/c111 for _ in [c100, c000, c200, c110]])

    c1 = 2*c200 + c110
    c2 = -c200 - c110
    if c1 < 0 or c2 < 0:
        return None

    def get_sol1(v):
        # solve 3*v**2/4*(a+b+c) + v*(a**2+b**2+c**2-2*(a*b+b*c+c*a)) + a*b*c >= 0
        # (8*a*b*v*(a - b)**2 + 2*a*b*(2*c - 3*v)**2*(a + b + c) + 3*v**2*(a - b)**2*(a + b + c) + 4*v*(b - c)**2*(-a + b + c)**2)
        return sp.Add(*[
            radsimp(4*v) * CyclicSum((b-c)**2*(b+c-a)**2),
            radsimp(3*v**2) * CyclicSum(a) * CyclicSum((a-b)**2),
            2 * CyclicSum(a) * CyclicSum(a*b*(2*c-3*v)**2),
            radsimp(8*v) * CyclicSum(a*b*(a-b)**2),
        ]) / (8 * CyclicSum(a)**2)


    def get_sol2(v):
        # solve 4*v**3 + v*(a**2+b**2+c**2-2*(a*b+b*c+c*a)) + a*b*c >= 0
        return sp.Add(*[
            v * CyclicSum((b-c)**2*(b+c-a)**2),
            radsimp(4*v**3) * CyclicSum((a-b)**2),
            radsimp(6*v) * CyclicSum(a*b*(c-2*v)**2),
            radsimp(2*v) * CyclicSum(a*b*(a-b)**2),
            2 * CyclicProduct(a) * CyclicSum(a-2*v)**2
        ]) / (2 * CyclicSum(a)**2)

    
    # find x, y, t such that
    # normalized poly >= get_sol1(x) * t + get_sol2(y) * (1-t)
    # 3*x**2/4 * t <= c100, 4*y**3 * (1-t) <= c000, x*t + y*(1-t) = c2
    # Cancelling y, t by x, we require x such that 4*x*(3*c2*x - 4*c100)**3/3/(3*x**2 - 4*c100)**2 <= c000
    # and 3x^2/4 >= c100 (so that t <= 1).

    def get_const(x):
        if x == 0: return c000 - 4*c2**3 # t = 0, y = c2
        if x == c2: return c000 # t = 1, y = 0
        return radsimp(c000 - 4*x*(3*c2*x - 4*c100)**3/3/(3*x**2 - 4*c100)**2)
    
    def check_x(x):
        if x == 0: return get_const(x) >= 0 # t = 0, y = c2
        if x is None or (not x.is_finite) or (not 3*x**2/4 >= c100):
            return False
        r = get_const(x)
        return r.is_finite and r >= 0

    for x in [0, radsimp(4*c100/3/c2), None]:
        if check_x(x):
            break
    if x is None:
        det = radsimp(c000**2 + 6*c000*c100*c2 - 4*c000*c2**3 + 4*c100**3 - 3*c100**2*c2**2)
        if det < 0:
            return None
        # find x near 2*(c2 + sqrt(c2**2 - c100))/3
        if c2**2 - c100 < 0:
            return None

        x = rationalize_func(sp.Poly([9, -12*c2, 4*c100], sp.Symbol('x')), validation=check_x,
                            validation_initial=lambda x: x >= 2*c2/3, direction = 1)
    # print(x, get_const(x), 't =', radsimp(c100 / (3*x**2/4)))
    if x is None:
        return None
    res = get_const(x)
    if res < 0:
        return None
    t1 = radsimp(c100 / (3*x**2/4)) if x != 0 else 0
    t2 = 1 - t1
    if t1 < 0 or t2 < 0:
        return None
    y = (c2 - x*t1)/t2 if t2 != 0 else 0
    
    return sp.Add(
        radsimp(c111 * t1) * get_sol1(x),
        radsimp(c111 * t2) * get_sol2(y),
        radsimp(c111 * res),
        radsimp(c111 * c1/2) * CyclicSum((a-b)**2)
    )