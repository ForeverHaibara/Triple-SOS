import sympy as sp

from .utils import (
    CyclicSum, CyclicProduct,
    sum_y_exprs, nroots, rationalize_bound, radsimp
)

a, b, c = sp.symbols('a b c')

def sos_struct_cubic(poly, coeff, recurrsion):
    """
    Solve cyclic cubic polynomials.

    This function only uses `coeff`. The `poly` and `recurrsion` is not used for minimium dependency.
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
    Solve nontrivial cyclic cubic polynomial by multiplying s(a). We avoid the use of recurrsion.

    See further details in the theorem of quartic.

    Examples
    -------
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
        t, p_, n_, q_ = _compute_params(u)
        if 3*(1 + n_) >= p_**2 + p_*q_ + q_**2:
            return True
        return False

    for u_ in nroots(equ, method = 'factor', real = True, nonnegative = True):
        if _check_valid(u_):
            u = u_
            break
    else:
        return None

    if not isinstance(u, sp.Rational):
        for u_ in rationalize_bound(u, direction = 0, compulsory = True):
            if u_ >= 0 and _check_valid(u_):
                u = u_
                break
        else:
            return None

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
    # solution = recurrsion(poly2)
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
        CyclicSum(a**3*b*c)
    ]
    return sum_y_exprs(y, exprs) / CyclicSum(a*b)