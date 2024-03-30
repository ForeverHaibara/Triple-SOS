from functools import partial

import sympy as sp

from .quartic import sos_struct_quartic
from .utils import (
    CyclicSum, CyclicProduct, Coeff, 
    sum_y_exprs, nroots, rationalize_bound, rationalize_func, quadratic_weighting, radsimp,
    prove_univariate
)

#####################################################################
#
#                              Symmetric
#
#####################################################################

a, b, c = sp.symbols('a b c')

def _wrap_c1_c2(c1, c2):
    """Return CyclicSum(c1*a^2 + c2*b*c) while clearing the denominators."""
    if c1 == 2 and c2 == -2:
        return CyclicSum((a-b)**2)
    if c2 == 2*c1:
        return c1 * CyclicSum(a)**2
    p = c1*a**2 + c2*b*c
    p = p.together().as_coeff_Mul()
    return p[0] * CyclicSum(p[1])

def _merge_quadratic_params(params):
    """
    If F = sum(pi * s(xi*a^2 + yi*a*b) for i in range(1,n+1)),
    then it can be represented as a whole:
    F = p0 * s(x0*a^2 + y0*a*b).

    Parameters
    -----------
    params : list
        A list of (pi, xi, yi) for i in range(1,n+1).
    
    Returns
    ---------
    p0, x0, y0, m, p, n, t, rem_coeff, rem_ratio:
        p0, x0, y0 are the coefficients of the merged quadratic form.

        m, p, n satisfy that
            sum(pi * s(xi*a^2 + yi*a*b)**2 for i in range(1,n+1)) - p0 * s(x0*a^2 + y0*a*b)**2
            = s(ma^4 + pa^3b + pab^3 + na^2b^2 - (m+2p+n)a^2bc)

        t, rem_coeff, rem_ratio satisfy that
            s(ma^4 + pa^3b + pab^3 + na^2b^2 - (m+2p+n)a^2bc) = t*s(a^2-ab)^2 + rem_coeff * s(a^2+rem_ratio*ab)^2
    """
    if len(params) == 0:
        return tuple([0] * 9)

    # merge these f(a,b,c) and standardize
    merged_params = sum([p1 for p1,p2,p3 in params]), sum([p1*p2 for p1,p2,p3 in params]), sum([p1*p3 for p1,p2,p3 in params])
    x_, y_ = merged_params[1] / merged_params[0], merged_params[2] / merged_params[0]

    # Now we have F(a,b,c) = merged_params[0] * (F0 - 2s(a4-a2bc)f(a,b,c) + s(a2-ab)f(a,b,c)^2) + s(a2-ab)g(a,b,c) + (..)*p(a-b)^2.
    # where f is the merged quadratic form and the g is the remaining part.
    # Assume g = s(ma^4 + pa^3b + pab^3 + na^2b^2 + ..a^2bc)
    # We can represent g = ts(a^2-ab)^2 + (m-t)s(a^2+rab)^2 >= 0
    m_, p_, n_ = sum([p1*p2**2 for p1,p2,p3 in params]), sum([2*p1*p2*p3 for p1,p2,p3 in params]), sum([p1*p3**2 for p1,p2,p3 in params])
    m_, p_, n_ = m_ - merged_params[0] * x_**2, p_ - 2 * merged_params[0] * x_ * y_, (n_ + 2*m_) - merged_params[0] * (2*x_**2 + y_**2)
    # print('Params =', params, '\nMerged Params =', merged_params, '(m,p,n) =', (m_, p_, n_))

    if not (n_ == 3*m_ and p_ == -2*m_):
        t_ = (-2*m_**2 + m_*n_ - p_**2/4)/(n_ + p_ - m_)
        if t_ < 0: # actually this will not happen
            return None
        if m_ != t_:
            # rem_coeff * s(a^2 + rem_ratio * ab)^2
            rem_coeff, rem_ratio = m_ - t_, (p_ + 2*t_) / (m_ - t_) / 2
        else:
            # it degenerates to rem_coeff * s(ab)^2
            rem_coeff = n_ - 3*t_
            if rem_coeff < 0: # this will not happen
                return None
    else:
        rem_coeff, rem_ratio, t_ = sp.S(0), sp.S(0), m_
    return merged_params[0], x_, y_, m_, p_, n_, t_, rem_coeff, rem_ratio


def _sos_struct_sextic_hexagon_symmetric_sdp(coeff):
    """
    Solve symmetric hexagons on real numbers without raising the degree.
    The idea is to subtract some CyclicSum(f(a,b,c)**2) so that the remainder
    term is a quadratic combination of s(a^2b-abc), s(a^2c-abc), (abc).

    In particular, we can WLOG assume that f is in the form of
    f(a,b,c) = (a-b) * (c^2-ab + (u+2)(ab+bc+ca))
    where u is a parameter to be determined.

    When the original polynomial has root at (1,1,1), or the coefficient of a^4bc equals to a^3b^3,
    then there is no degree of freedom and we can solve the parameter u directly.

    When poly(1,1,1) > 0. The quadratic form can be represented as
    [[M00, M01, M02]
     [M01, M00, M02]
     [M02, M02, l22]]
    where l22 = poly(1,1,1) > 0. To ensure det >= 0, we find u that maximizes the Schur complement:
    u = argmax {l22 - [M02, M02] * [M00, M01; M01, M00]^-1 * [M02, M02]}

    See similar methods at _sos_struct_sextic_hexagon_sdp and _sos_struct_octic_symmetric_hexagon_sdp.
    """
    c420, c411, c330, c321, c222 = [coeff(_) for _ in [(4,2,0), (4,1,1), (3,3,0), (3,2,1), (2,2,2)]]
    if c420 <= 0:
        if all(_ == 0 for _ in (c420, c411, c330, c321)) and c222 >= 0:
            return c222 * CyclicProduct(a)**2
        return None

    # l22 = poly(1,1,1)
    l22 = c222 + 6*c321 + 3*c330 + 3*c411 + 6*c420

    if not (l22 >= 0 and abs(c411) <= 2*c420 and abs(c330) <= 2*c420):
        return None

    w1 = (c411 + c330)/2
    w2 = (c411 - c330)/2

    def _compute_quad_form(u):
        M00 = radsimp(-(-3*c420*u**2 - 6*c420*u + 2*u**2*w2 + 6*u*w2 + 6*w2)/(3*u*(u + 2)))
        M01 = radsimp((3*u**2*w1 - u**2*w2 + 6*u*w1 - 6*u*w2 - 6*w2)/(6*u*(u + 2)))
        M02 = radsimp((c321*u + 4*c420*u + 3*u*w1 - 3*u*w2 - 6*w2)/(2*u))
        return M00, M01, M02

    def _is_valid(u):
        if u == 0 or u == -2:
            return False
        M00, M01, M02 = _compute_quad_form(u)
        return M00 >= abs(M01) and l22*(M00 + M01) - 2*M02**2 >= 0 and -(c330 - c411)*(u*(u + 2)) > 0

    def _compute_quad_form_sol(M00, M01, M02):
        """
        Solve v' * M * v where v = [s(a^2b-abc), s(a^2c-abc), abc]
        while M is in the following form.
        [[M00, M01, M02]
        [M01, M00, M02]
        [M02, M02, l22]]
        """
        if M00 < M01: return
        s1 = (M00 - M01)/2 * CyclicProduct((a-b)**2)

        def mapping(x, y):
            # return s(x(a2b+a2c-2abc)+y/3(abc))^2
            if x == 1 and y == 0:
                return CyclicSum(a*(b-c)**2)**2
            elif x == 0 and y == 1:
                return CyclicProduct(a**2)
            p = x*(a**2*b + a**2*c - 2*a*b*c) + y/3*a*b*c
            p = p.expand().together() if coeff.is_rational else p
            return CyclicSum(p)**2

        s2 = quadratic_weighting(
            (M00 + M01)/2,
            M02 * 2,
            l22,
            mapping = mapping
        )
        if s2 is None: return None

        return s1 + s2

    def _compute_sol(u):
        if u is None or u == 0 or u == -2:
            return
        quad_form_sol = _compute_quad_form_sol(*_compute_quad_form(u))
        r = radsimp(-(c330 - c411)/(6*u*(u + 2)))
        if r >= 0 and quad_form_sol is not None:
            # rs(((a-b)(c2+(u+2)s(ab)-ab))2) + quad_form_sol
            p = (c**2 + (u + 2)*(a*b+b*c+c*a) - a*b)
            p = p.expand().together() if coeff.is_rational else p
            return r * CyclicSum((a-b)**2 * p**2) + quad_form_sol

    u = None
    if c411 == c330:
        # there is no degree of freedom
        u = sp.S(1)

    else:
        denom = c321 + 4*c420 + 3*w1 - 3*w2
        if denom != 0:
            # when l22 == 0, there is no degree of freedom
            u = radsimp(6*w2/denom)

        if l22 > 0 and not _is_valid(u):
            eq_u = sp.Poly.from_list(
                [-2*c321 + 10*c420 + 3*w1 - 9*w2, -9*c321 + 36*c420 + 9*w1 - 45*w2, -9*c321 + 36*c420 + 9*w1 - 81*w2, -54*w2],
                sp.Symbol('u')
            )
            u = rationalize_func(eq_u, _is_valid)

    return _compute_sol(u)


def _sos_struct_sextic_hexagon_symmetric(coeff, real = False):
    """
    Solve symmetric hexagons without a^6, a^5b terms.
    Although we can subtract p(a-b)^2 to make the polynomial a positive hexagram on R+,
    we sometimes want to solve the hexagon on R.

    Consider the following hexagon:
    F(a,b,c) = s(a^2b^2(a+b)^2 + xa^4bc + ya^3bc(b+c) - ...a^2b^2c^2)
    It has root (1,-1,0) over R.
    
    Theorem 1:
    When t not in (-2,1), the following inequality holds for all real numbers a, b, c:
    f(a,b,c) = t^2/4 * p(a-b)^2 + s(bc(a-b)(a-c)(a-tb)(a-tc)) >= 0

    Proof: let
    lambda = (t**2 + 4*t - 8)**2/(4*(t - 2)**2*(5*t**2 - 4*t + 8 + (4*t - 16)*sqrt(t**2 + t - 2)))
    z = (2*t**2 - 2*t + 2*(t - 2)*sqrt(t**2 + t - 2))/(t**2 + 4*t - 8)
    Then we have,
    f(a,b,c) * s((a-b)^2) = lambda * s((a-b)^2*((t-2)ab(a+b) - (t-2z)c(a^2+b^2-c^2) - t(1-z)c^2(a+b+c) + (2t+4-3tz-2z)abc)^2)

    With the theorem, we can see that
    F_{x,y}(a,b,c) >= 0 holds for a,b,c in R if (x,y) = (4/t^2 - 2, -4/t^2 - 4/t - 2) and t not in (-2,1).
    It forms a parametric curve (parabola) (x + y + 2)^2 + 4y + 4 = 0 with constraint y <= -3x - 4.
    Here t = -4/(x + y + 4).
    As a result, any point (x, y) lies in this region is positive and is a linear combination of (2,-10) and (x2,y2).


    Examples
    ----------
    s(3a2b+ab2-4abc)2+s(a2b+3ab2-4abc)2    (real)

    s((b-c)2(a2-3(ab+ac)+2bc)2)/14-8/14abcs(a2b+ab2-2abc)    (real)

    s(a2(b-c)4)-1/2p(a-b)2      (real)

    s(a2(b2-c2)2)-3/8p(a-b)2    (real, root = (-1,-1,1))

    3p(a2+ab+b2)-s(a)2s(ab)2    (real)

    p(a2+ab+b2)-3s(ab)s(a2b2)   (real)

    s(a)s(ab)p(a+b)-6s(a(a+b)(a+c))p(a)-3abcs((a+b-c)(a-c)(b-c))   (real)

    8s(a2)3-9p(a+b)s(a3)-2s((b-c)2(b+c-1/8a))2    (real)

    s(bc(a-b)(a-c)(a-3b)(a-3c)) +1/4s(a2b+ab2-2abc)2+5p(a-b)2+4abcs(a(b-c)2)    (real)

    s(bc(a-b)(a-c)(a-3b)(a-3c))+9/4p(a-b)2    (real)

    s(bc(a-b)(a-c)(a-9/8b)(a-9/8c))+81/256p(a-b)2    (real)

    s(4a4b2-7a4bc+4a4c2+8a3b3-12a3b2c-12a3bc2+15a2b2c2)   (real)

    s(bc(b+c)2(a-b)(a-c))+s(bc(a-b)(a-c)(a-2b)(a-2c))+2s(a4(b-c)2)  (real)

    p(a2+ab+b2)+12a2b2c2-3p(a+b)2/5    (real, uncentered)
    
    References
    ----------
    [1] Vasile, Mathematical Inequalities Volume 1 - Symmetric Polynomial Inequalities. p.23

    [2] https://artofproblemsolving.com/community/u426077h3036593p28226075
    """
    if coeff((4,2,0)) <= 0:
        if coeff((4,2,0)) == 0:
            return _sos_struct_sextic_hexagram_symmetric(coeff)
        return None

    rem = radsimp((coeff((4,2,0)) + coeff((3,2,1))) * 6 + (coeff((3,3,0)) + coeff((4,1,1))) * 3 + coeff((2,2,2)))
    if rem < 0:
        return None
    
    # although subtracting p(a-b)2 always succeeds,
    # we can handle cases for real numbers and cases where raising the degree is not necessary
    solution = _sos_struct_sextic_hexagon_symmetric_sdp(coeff)
    if solution is not None:
        return solution


    c420, c330, c411, c321 = [coeff(_) for _ in [(4,2,0), (3,3,0), (4,1,1), (3,2,1)]]
    if real and abs(c330) <= 2*c420 and abs(c411) <= 2*c420:
        # Perform sum of squares for real numbers.
        # First we subtract coeffp * p(a-b)^2, so that the polynomial has root at (1,-1,0) or (1,-1,oo).
        if c330 >= c411:
            # will have root at (1,-1,0)
            type = 0
            coeffp = radsimp((c420 * 2 - c330) / 4)
            new_c420 = radsimp(c420 - coeffp)
            new_c411 = radsimp((c411 + coeffp * 2) / new_c420)
        else:
            # will have root at (1,-1,oo)
            type = 1
            coeffp = radsimp((c420 * 2 - c411) / 4)
            new_c420 = radsimp(c420 - coeffp)
            new_c411 = radsimp((c330 + coeffp * 2) / new_c420) # we borrow c411 to store c330
        new_c321 = radsimp((c321 - 2 * coeffp) / new_c420)


        def _linear_comb(x, y):
            """
            Any point (x, y) lies in the region (x + y + 2)^2 + 4y + 4 <= 0 and y <= -3x - 4
            can be expressed as a linear combination of (2,-10) and (x2,y2).
            Returns w, t so that t = -4/(x2 + y2 + 4) while (x, y) = w * (2,-10) + (1-w) * (x2, y2).
            """
            if not ((x + y + 2)**2 + 4*y + 4 <= 0 and y <= -3*x - 4):
                return None
            if x == 2 and y == 10:
                return sp.S(1), sp.S(1)

            r = radsimp(4*(x - 2)*(3*x + 2*y + 14)/(x + y + 8)**2)
            x2 = radsimp(r + 2)
            y2 = radsimp(-10 + (y + 10)/(x - 2)*r)
            t = radsimp(-4 / (x2 + y2 + 4)) if x2 + y2 + 4 != 0 else sp.oo

            w = radsimp((x2 - x) / (x2 - 2) if x2 != -1 else (y2 - y) / (y2 + 10))
            return w, t

        _comb = _linear_comb(new_c411, new_c321)
        if _comb is not None:
            w, t = _comb
            if type == 1:
                t = 1/t

            def _get_solution(t):
                """
                When t <= -2 or t >= 1, solve
                s((a-b)^2) * (s(bc(a-b)(a-c)(a-tb)(a-tc)) + t^2/4p(a-b)^2) * (4/t^2) >= 0.
                When -1/2 <= t < 1, solve
                s((a-b)^2) * (s(bc(a-b)(a-c)(a-tb)(a-tc)) + 1/4p(a-b)^2) * 4 >= 0.
                """
                # print('t =', t)
                if t == 1:
                    return CyclicSum(a*(b-c)**2)**2 * CyclicSum((a-b)**2)
                elif t is sp.oo:
                    # s((a-b)2(a2b-a2c+ab2+2abc-ac2-b2c-bc2)2)
                    return CyclicSum((a-b)**2*(a**2*b-a**2*c+a*b**2+2*a*b*c-a*c**2-b**2*c-b*c**2)**2)
                elif t == 2:
                    # s((a-b)2(a2b+ab2-5abc+ac2+bc2+c3)2)
                    return CyclicSum((a-b)**2*(a**2*b+a*b**2-5*a*b*c+a*c**2+b*c**2+c**3)**2)
                elif t == 0:
                    # s((a-b)4(ab+ac+bc-c2)2)
                    return CyclicSum((a-b)**4*(a*b+a*c+b*c-c**2)**2)
                elif t == -2:
                    # s(a)2s(a2b2-a2bc) * 2s(a2-ab)
                    return CyclicSum(a)**2 * CyclicSum(a**2*(b-c)**2) * CyclicSum((a-b)**2)/2
                elif t == -sp.S(1)/2:
                    # s(ab)2s(a2-ab) * 2s(a2-ab)
                    return CyclicSum(a*b)**2 * CyclicSum((a-b)**2)**2/2

                _expand = (lambda p: p.expand().together()) if coeff.is_rational else (lambda p: p)
                if t <= -2 or t >= 1:
                    ker = sp.sqrt(t**2 + t - 2)
                    w = radsimp(t**2 + 4*t - 8)
                    if w != 0 and isinstance(ker, sp.Rational):
                        l = radsimp(w**2/(4*(t - 2)**2*(5*t**2 - 4*t + ker*(4*t - 16) + 8)))
                        z = radsimp((2*t**2 - 2*t + ker*(2*t - 4))/w)
                        p = (a*b*c*(-3*t*z + 2*t - 2*z + 4) + a*b*(a + b)*(t - 2) - c**2*t*(1 - z)*(a + b + c) - c*(t - 2*z)*(a**2 + b**2 - c**2)).expand().together()
                        return radsimp(l * 4 / t**2) * CyclicSum((a-b)**2*p**2)

                    if w >= 0:
                        w1, w2 = radsimp(1/t**2), radsimp(w/t**2)
                        p1 = _expand(2*(a**2+b**2-c**2) - (2+3*t)*a*b + t*c*(a+b+c))
                        return w1 * CyclicSum(c**2 * (a-b)**2 * p1**2) + w2 * CyclicSum(a)**2 * CyclicProduct((a-b)**2)
                    else:
                        w1, w2 = radsimp(1/t**2/9), radsimp(4*(t-1)*(t+2)/3/t**2)
                        # p0 = (t(a+b-2c)s(a(b-c)2)+(2-2t)c(a3-2a2b-2ab2+4abc-ac2+b3-bc2))
                        # p0 = _expand(t*(a + b - 2*c)*CyclicSum(a*(b-c)**2) + (2 - 2*t)*c*(a**3 - 2*a**2*b - 2*a*b**2 + 4*a*b*c - a*c**2 + b**3 - b*c**2))
                        p1 = _expand(((t - 4)*a**2*b + (-t - 2)*a**2*c + (t - 4)*a*b**2 + (8*t + 10)*a*b*c - 3*t*a*c**2 + (-t - 2)*b**2*c - 3*t*b*c**2 + (2 - 2*t)*c**3))
                        return w1 * CyclicSum((a-b)**2 * p1**2) + w2 * CyclicSum(a)**2 * CyclicProduct((a-b)**2)

                elif -1 <= 2*t <= 2:
                    p1 = CyclicSum((a-b)*(a-c)*(2*t*b*c-a*b-a*c))**2
                    if 2*t <= 1:
                        return (1 - 2*t) * p1 + (1 + 2*t) * CyclicSum((a-b)**2*(a-c)**2*(2*t*b*c-a*b-a*c)**2)
                    else:
                        w1, w2 = radsimp(4*(1-t)/3), radsimp((2*t+1)/3)
                        p2 =  _expand(-a**2*b - a**2*c - a*b**2 + 2*a*b*c + a*c**2 - b**2*c + b*c**2 + t*(4*a*b*c - 2*a*c**2 - 2*b*c**2))
                        return w1 * p1 + w2 * CyclicSum((a-b)**2 * p2**2)

            main_solution = radsimp(w*new_c420) * _get_solution(1) + radsimp((1 - w)*new_c420) * _get_solution(t)
            rem_solution = coeffp * CyclicProduct((a-b)**2) + rem * CyclicProduct(a**2)
            return main_solution / CyclicSum((a-b)**2) + rem_solution


    if True:
        # subtract p(a-b)^2
        t = coeff((4,2,0))
        new_coeffs_ = {
            (3,3,0): coeff((3,3,0)) + 2 * t,
            (4,1,1): coeff((4,1,1)) + 2 * t,
            (3,2,1): coeff((3,2,1)) - 2 * t,
            (2,3,1): coeff((2,3,1)) - 2 * t,
            (2,2,2): coeff((2,2,2)) + 6 * t
        }

        solution = _sos_struct_sextic_hexagram_symmetric(Coeff(new_coeffs_, is_rational=coeff.is_rational))
        if solution is not None:
            return solution + coeff((4,2,0)) * CyclicProduct((a-b)**2)

    return None


def _sos_struct_sextic_hexagram_symmetric(coeff):
    """
    Solve s(a3b3+xa4bc+ya3b2c+ya2b3c+wa2b2c2) >= 0

    Theorem 1: For real number u, 
        f(a,b,c) = s(a4bc+(u-1)^2*a3b3-(u^2-u+1)*a2b2c(a+b)+u^2*a2b2c2) >= 0
    Because
        f(a,b,c) * 2s(a) = abcs((b-c)^2(b+c-ua)^2)+2s(a(b-c)^2((1-u)(ab+ac)+bcu)^2) >= 0
    As a consequence, if (x, y) lies in the parametric curve ((u-1)^2, -(u^2-u+1)),
    which is parabola x >= (1+x+y)^2
    then it is positive.

    Examples
    -------    
    s(a4bc+4a3b3-7a3b2c-7a3bc2+9a2b2c2)
    
    s(a3b3+2a4bc- 44/10(a3b2c+a3bc2-2a2b2c2)-3a2b2c2)

    s(21a4bc+7a3b3-40a3b2c-40a3bc2+52a2b2c2)

    s(ab(a-c)2(b-c)2)+3s(ab(ab-bc)(ab-ca))

    s(ab)3+abcs(a)3+64a2b2c2-12abcs(a)s(ab)

    7s(ab)3+8abcs(a)3+392a2b2c2-84abcs(a)s(ab)
    
    References
    -------
    [1] https://tieba.baidu.com/p/8039371307 
    """
    if coeff((3,3,0)) < 0 or coeff((4,1,1)) < 0:
        return None
    
    # first try trivial cases
    if True:
        # For s(a3b3+xa4bc+ya3b2c+ya2b3c+wa2b2c2) with 1+x+2y+w = 0,
        # it covers the case: y + min(x,1) + x + 1 >= 0

        x_ = min(coeff((3,3,0)), coeff((4,1,1)))
        y = radsimp([
            x_,
            coeff((3,3,0)) - x_,
            coeff((4,1,1)) - x_,
            coeff((3,2,1)) + x_ + (coeff((3,3,0))) + (coeff((4,1,1))),
            (coeff((3,3,0)) + coeff((4,1,1)) + coeff((3,2,1)) * 2) * 3 + coeff((2,2,2))
        ])
        if all(_ >= 0 for _ in y):
            exprs = [
                CyclicSum(a*b*(a-c)**2*(b-c)**2),
                CyclicSum(b**2*c**2*(a-b)*(a-c)),
                CyclicSum(a*(a-b)*(a-c)) * CyclicProduct(a),
                CyclicSum(a*(b-c)**2) * CyclicProduct(a),
                CyclicProduct(a**2)
            ]
            return sum_y_exprs(y, exprs)
    
    if coeff((4,1,1)) != 0:
        x_ = coeff((3,3,0)) / coeff((4,1,1))
        y_ = coeff((3,2,1)) / coeff((4,1,1))
        w_ = coeff((2,2,2)) / coeff((4,1,1))
        x_, y_, w_ = radsimp([x_, y_, w_])
        if x_ >= radsimp((1 + x_ + y_)**2):
            # apply theorem 1
            # use vieta jumping, a point inside the parabola is a linear combination
            # of u = 1 and u = (y + 1) / (x + y + 1)
            u_ = radsimp((y_ + 1) / (x_ + y_ + 1))

            # weights of linear combination
            w2 = x_ / (u_ - 1)**2
            w1 = 1 - w2
            w1, w2 = radsimp([w1, w2])

            # NOTE: the case x + y + 1 == 0 has been handled in the trivial case

            # abcs((b-c)2(b+c-ua)2)+2s(a(b-c)2((1-u)(ab+ac)+bcu)2)
            r = u_.as_numer_denom()[1] # cancel the denominator is good

            y = radsimp([
                w1 / 2,
                w1,
                w2 / r**2 / 2,
                w2 / r**2,
                ((coeff((3,3,0)) + coeff((4,1,1)) + coeff((3,2,1)) * 2) + coeff((2,2,2)) / 3) / coeff((4,1,1)) * 3
            ])
            if any(_ < 0 for _ in y):
                y = None
            else:
                multiplier = CyclicSum(a)
                y = [radsimp(_ * coeff((4,1,1))) for _ in y]
                exprs = [
                    CyclicSum((b-c)**2*(b+c-a)**2) * CyclicProduct(a),
                    CyclicSum(b*c*(b-c)**2) * CyclicProduct(a),
                    CyclicSum((b-c)**2*(r*b+r*c-r*u_*a)**2) * CyclicProduct(a),
                    CyclicSum(a*(b-c)**2*(r*(1-u_)*a*b+r*(1-u_)*a*c+r*u_*b*c)**2),
                    CyclicSum(a) * CyclicProduct(a**2)
                ]
                # print(y, exprs)
                return sum_y_exprs(y, exprs) / multiplier



    if coeff((3,3,0)) != 0:
        # https://tieba.baidu.com/p/8039371307
        x_, y_, z_ = coeff((4,1,1)) / coeff((3,3,0)), -coeff((3,2,1)) / coeff((3,3,0)), coeff((2,2,2)) / coeff((3,3,0))
        x_, y_, z_ = radsimp([x_, y_, z_])
        z0 = radsimp(x_**2 + x_*y_ + y_**2/3 - y_ + (y_ + 3)**3/(27*x_))
        if x_ > 0 and 3 * x_ + y_ + 3 >= 0 and z_ >= z0:
            ker = 324 * x_ * (27*x_**3 + 27*x_**2*y_ + 81*x_**2 + 9*x_*y_**2 - 189*x_*y_ + 81*x_ + y_**3 + 9*y_**2 + 27*y_ + 27)
            ker = radsimp(ker)
            if ker > 0:
                w1 = radsimp(-(9*x_**2 + 6*x_*y_ - 306*x_ + y_**2 + 6*y_ + 9) / ker)
                if w1 > 0:
                    w2 = radsimp(1 / ker)
                    phi2 = radsimp(36*x_**2 + 15*x_*y_ - 117*x_ + y_**2 + 6*y_ + 9)
                    phi1 = radsimp(9*x_**2 + 6*x_*y_ - 117*x_ + y_**2 + 15*y_ + 36)

                    multiplier = CyclicSum(a) * CyclicSum(a*b)
                    y = [w1, w2, z_ - z0]
                    y = [radsimp(_ * coeff((3,3,0))) for _ in y]

                    c11, c12, c13, c14, c15, c16, c17, c18 = radsimp([
                        -9*x_**2-3*x_*y_+18*x_, -9*x_**2+9*x_+y_**2-9, 9*x_**2+3*x_*y_-3*y_-9, -18*x_+3*y_+9,
                        -9*x_**2-3*x_*y_+18*x_, 9*x_**2+3*x_*y_-3*y_-9, 9*x_**2-9*x_-y_**2+9, -18*x_+3*y_+9
                    ])
                    c21, c22, c23, c24, c25, c26, c27, c28 = radsimp([
                        -3*phi1*x_, -3*phi1*x_+phi1*y_+3*phi1-3*phi2, -3*phi1*x_+phi1*y_+3*phi1+3*phi2*x_+phi2*y_-3*phi2,
                        -3*phi2, -3*phi1*x_, -3*phi1*x_+phi1*y_+3*phi1+3*phi2*x_+phi2*y_-3*phi2, -3*phi1*x_+3*phi2*x_+phi2*y_-3*phi2, -3*phi2
                    ])
                    exprs = [
                        CyclicSum(c*(c11*a**3*b + c12*a**2*b**2 + c13*a**2*b*c + c14*a**2*c**2 + c15*a*b**3 + c16*a*b**2*c + c17*a*b*c**2 + c18*b**2*c**2)**2),
                        CyclicSum(c*(c21*a**3*b + c22*a**2*b**2 + c23*a**2*b*c + c24*a**2*c**2 + c25*a*b**3 + c26*a*b**2*c + c27*a*b*c**2 + c28*b**2*c**2)**2),
                        multiplier * CyclicProduct(a**2)
                    ]
    
                    return sum_y_exprs(y, exprs) / multiplier
    return None


def _sos_struct_sextic_tree(coeff):
    """
    Solve s(a6 + ua3b3 + va4bc - 3(1+u+v)a2b2c2) >= 0

    Theorem:
    If the inequality holds for all a,b,c >= 0, then there must exist x >= 1
    such that
        f(a,b,c) = s(a2+xab) * s((a-b)^2*(a+b-xc)^2) / 2
                    + (u - (x^3 - 3*x)) * s(a3b3 - a2b2c2) 
                        + (v + 3*x*(x-1)) * s(a4bc - a2b2c2)

    where (u - (x^3 - 3*x)) >= 0 and (v + 3*x*(x-1)) >= 0. Actually, x (>= 1)
    can be the root of (x^3 - 3*x - u).
    
    We can see that the inequality holds for real numbers when -1 <= x <= 2.
    Further, if (u,v) falls inside the (closed) parametric curve (x^3-3x,-3x(x-1)) where -1<=x<=2, 
    which is 27*u^2+27*u*v+54*u+v^3+18*v^2+54*v = 0, a strophoid,
    then the inequality is a linear combination of two positive ones.

    Examples
    -------
    s(2a6-36a4bc+36a3b3-2a2b2c2)

    s(a6+4a3b3-7a4bc+2a2b2c2)
    """

    t = coeff((6,0,0))
    rem = radsimp(coeff((2,2,2)) + (coeff((6,0,0))+coeff((4,1,1))+coeff((3,3,0))) * 3)
    if rem < 0 or t < 0:
        return None
    if t == 0 and coeff((3,3,0)) >= 0 and coeff((4,1,1)) >= 0:
        return sp.Add(
            coeff((3,3,0))/2 * CyclicSum(a*b) * CyclicSum(a**2*(b-c)**2),
            coeff((4,1,1))/2 * CyclicProduct(a) * CyclicSum((a-b)**2) * CyclicSum(a),
            rem * CyclicProduct(a**2)
        )

    # t != 0 by assumption
    u, v = radsimp(coeff((3,3,0))/t), radsimp(coeff((4,1,1))/t)
    if u < -2:
        return None


    if v != -6 and u != 2:
        # try sum of squares with real numbers first
        # if (u,v) falls inside the parametric curve (x^3-3x,-3x(x-1)) where -1<=x<=2,
        # then it is a rational linear combination of (t^3-3t, -3t(t-1)) and (2, -6)
        # with t = -(3u + v) / (v + 6)
        # note: (2, -6) is the singular node of the strophoid
        t__ = radsimp(-(3*u + v) / (v + 6))
        if -1 <= t__ <= 2:
            x = t__**3 - 3*t__
            w1 = radsimp((27*u**2 + 27*u*v + 54*u + v**3 + 18*v**2 + 54*v)/(27*(u - 2)*(u + v + 4)))
            w2 = 1 - w1
            q, p = t__.as_numer_denom()
            if 0 <= w1 <= 1:
                y = radsimp([w1 * t / 2, w2 * t / 2 / p**3, rem])
                exprs = [
                    CyclicSum(a)**2 * CyclicSum((b-c)**4),
                    CyclicSum(p*a**2 + q*b*c) * CyclicSum((a-b)**2*(p*a+p*b-q*c)**2),
                    CyclicProduct(a**2),
                ]
                return sum_y_exprs(y, exprs)
    
    x = sp.symbols('x')
    equ = (x**3 - 3*x - u).as_poly(x)
    r = None
    if not coeff.is_rational:
        # first check whether there is exact solution
        eqv = (3*x**2 - 3*x + v).as_poly(x)
        eq_gcd = sp.gcd(equ, eqv)
        if eq_gcd.degree() == 1:
            r = radsimp(-eq_gcd.coeff_monomial((0,)) / eq_gcd.coeff_monomial((1,)))
            if r < -1:
                r = None

    if r is None:
        def _is_valid(r):
            return r >= 0 and 3*r*(r-1) + v >= 0
        r = rationalize_func(equ, _is_valid, direction = -1)

    if r is not None:            
        # now r is rational
        y = radsimp([t/2, t*(u-(r**3-3*r))/2, t*(v+3*r*(r-1))/2, rem])
        exprs = [
            CyclicSum(a**2 + r*b*c) * CyclicSum((a-b)**2*(a+b-r*c)**2),
            CyclicSum(a*b) * CyclicSum(a**2*(b-c)**2),
            CyclicSum(a) * CyclicSum((a-b)**2) * CyclicProduct(a),
            CyclicProduct(a**2),
        ]
        if r == 2:
            exprs[0] = CyclicSum(a)**2 * CyclicSum((a-b)**2*(a+b-2*c)**2)

        return sum_y_exprs(y, exprs)

    return None



def _sos_struct_sextic_iran96(coeff, real = False):
    """
    Solve s(a5b+ab5-x(a4b2+a2b4)+ya3b3-za4bc+w(a3b2c+a2b3c)+..a2b2c2) >= 0

    Observe that g(a,b,c) = s(bc(b-c)^2(b+c-(u+1)a)^2) >= 4((a-b)*(b-c)*(c-a))^2,
    which is because
        g(a,b,c)s(a) = s(c(a-b)2((a+b-c)2-uab)2) + abcs((b-c)2(b+c-(u+1)a)2) >= 0

    Also note that h(a,b,c) = s(bc(a-b)(a-c)(a-ub)(a-uc)) >= 0

    In general, we can show that f(a,b,c) = g(a,b,c) + (t-2)^2/u^2 * h(a,b,c) >= 2*t * ((a-b)*(b-c)*(c-a))^2

    If we let x = (t*(u+1) - u^2 - 2) / u, then
        f(a,b,c)s(a) = s(a(b-c)^2(a^2+b^2+c^2-tab-tac+xbc)^2) + (t-2*u-2)^2/(2u^2) * s(abc(b-c)^2(b+c-(u+1)a)^2)

    The structure is named after Iran-96, but the original Iran-96 as a sextic is very very weak.

    Examples
    -------
    4s(ab)s((a+b)^2(a+c)^2)-9p((a+b)^2)

    s(a2(a2-b2)(a2-c2))-s(a4(a-b)(a-c))+5p(a-b)2

    (s(a2bc(a2-b2+4(bc-ac)))+s(ac(3c+13/7b)(a-b)(3(a+b)c-4ab)))+9(s(ab(a2-b2+4(bc-ac))2)-6p(a-b)2)
    
    s(ab(a4+b4)-6(a4b2+a2b4)+11a3b3+13abca(a-b)(a-c)-3(a3b2c+a2b3c)+5a2b2c2)

    (s(ab(a-b)4)-8abcs(a3-2a2b-2a2c+3abc))-p(a-b)2+1/4s(a3b3-a2b2c2)

    s(2a6-a4(b2+c2))-27p(a-b)2-2s(a3-abc-7/5(a2b+ab2-2abc))2

    s(4a6-a3b3-3a2b2c2)-63p(a-b)2-4s(a3-abc-3/2(a2b+ab2-2abc))2

    s(ab(a-b)2(a+b-c)2)-4p(a-b)2
    
    (s(a(a+b)(a+c)(a+b+c)2)+12abcs(ab)-2p(a+b)s(a)2)s(a)-s(a(a-b)(a-c))2-16p(a-b)2

    s(a4(a-b)(a-c))-5p(a-b)2+s(a3-abc-3(a2b+ab2-2abc))2-10p(a-b)2-2s(a3-abc-9/4(a2b+ab2-2abc))2

    s(ab)s((a-b)2(a+b-5c)2)+2s(ab(a-c)2(b-c)2)+s(ab(a-b)4)-18p(a-b)2-2/3s(ab(a-c)2(b-c)2)

    s(a(b+c)(b+c-2a)4)

    s(a(b+c)(b+c-2a)2(b-c)2)

    s(12a5b+12a5c+72a4b2-212a4bc+72a4c2-167a3b3+200a3b2c+200a3bc2-189a2b2c2)

    729p(a2)+288s(b2c2)s(a)2+21s(a3)s(a)3-14s(a)6-7s(a3-abc-3(a2b+ab2-2abc))2

    s(21a6-20a5b-20a5c+825a4b2-1667a4bc+825a4c2-1640a3b3+1679a3b2c+1679a3bc2-1682a2b2c2)-21s(a(a-b)(a-c))2

    s(ab(a2+b2-2c2-3(2ab-ac-bc))2)

    (s(22a6-36a5b-36ab5+657a4b2+657a2b4-28a3b3-540abc(a2b+ab2)-420a4bc+792/3a2b2c2)-22s(a(a-b)(a-c))2)

    s((a3+b3-2a2b-2ab2+c2(a+b)-3(c2a+c2b-c(a2+b2)))2)-s((b-c)2(b+c-a))2/2

    3s(a/3)6-s(ab)s(a/3)4-(69+11sqrt(33))/648p(a-b)2-s(a3-abc-(sqrt(33)/4 + 7/4-1)(a2b+ab2-2abc))2/243

    s(bc(b-c)2(b+c-(1+sqrt(2))a)2)+7/2s(bc(a-b)(a-c)(a-sqrt(2)b)(a-sqrt(2)c))-(2sqrt(7)+4)p(a-b)2

    References
    ----------
    [1] https://tieba.baidu.com/p/8205000150

    [2] https://artofproblemsolving.com/community/c6t29440f6h3146163_inspired_by_my_own_results
    """
    if not (coeff((6,0,0)) == 0 and coeff((5,1,0)) == coeff((1,5,0)) and coeff((4,2,0)) == coeff((2,4,0)) and\
        coeff((3,2,1)) == coeff((2,3,1)) and coeff((5,1,0)) >= 0):
        return None

    m, p, q, w, z = coeff((5,1,0)), coeff((4,2,0)), coeff((3,3,0)), coeff((4,1,1)), coeff((3,2,1))
    rem = radsimp(coeff((2,2,2)) + 3*((m + p + z) * 2 + q + w))

    if m < 0 or rem < 0:
        return None
    elif m == 0:
        # only perform sum of squares for real numbers when explicitly inquired
        return _sos_struct_sextic_hexagon_symmetric(coeff, real = real)
    
    if w >= 0 and w + z >= 0:
        # Easy case 1, really trivial
        y = [
            m,
            p + m * 4,
            (q - m * 6 + 2 * (p + m * 4)) / 2,
            w,
            z + w,
            rem
        ]

        if all(_ >= 0 for _ in y):
            exprs = [
                CyclicSum(a*b*(a-b)**4),
                CyclicSum(a**2*b**2*(a-b)**2),
                CyclicSum(a*b) * CyclicSum(a**2*(b-c)**2),
                CyclicSum(a*(a-b)*(a-c)) * CyclicProduct(a),
                CyclicSum(a*(b-c)**2) * CyclicProduct(a),
                CyclicProduct(a**2)
            ]
            return sum_y_exprs(y, exprs)


    if p >= -4 * m and q + 2 * (p + m) >= 0:
        # Easy case 2, subtract enough p(a-b)2 and s(ab(a-c)2(b-c)2)
        # e.g. s(a(b+c)(b+c-2a)4)
        y = [
            m,
            (p + 4 * m),
            q + 2 * (p + m),
            w + 2 * (p + 4 * m) - (q + 2 * (p + m)),
            4*m + 4*p + 2*q + w + z,
            rem
        ]
        
        if all(_ >= 0 for _ in y):
            exprs = [
                CyclicSum(a*b*(a-b)**4),
                CyclicProduct((a-b)**2),
                CyclicSum(a*b*(a-c)**2*(b-c)**2),
                CyclicSum(a*(a-b)*(a-c)) * CyclicProduct(a),
                CyclicSum(a*(b-c)**2) * CyclicProduct(a),
                CyclicProduct(a**2)
            ]
            return sum_y_exprs(y, exprs)


    if p >= 0 and q + 2 * m + 2 * p >= 0:
        # Easy case 3, subtract s(ab(a-b)2(a+b-xc)2) such that the coeffs of 
        # a^4bc and a^3b^3 are equal
        
        if True:
            x_ = radsimp((q - w) / (4 * m) + sp.Rational(1,2))
            y = radsimp([
                m,
                p,
                q + 2 * m + 2 * p,
                z - m*x_*(x_+2) - 2*p + 3*(q + 2*m + 2*p),
                rem
            ])
            if all(_ >= 0 for _ in y):
                exprs = [
                    CyclicSum(a*b*(a-b)**2*(a+b-x_*c)**2),
                    CyclicProduct((a-b)**2),
                    CyclicSum(a*b*(a-c)**2*(b-c)**2),
                    CyclicSum(a*(b-c)**2) * CyclicProduct(a),
                    CyclicProduct(a**2)
                ]
                return sum_y_exprs(y, exprs)


        # Easy case 4, when we do not need to higher the degree

        # find some u such that
        # w' = w + 2 * p + 4 * u * m >= 0
        # q' = 2*m + 2*p + q
        # 2*q'+w' + min(2*q', w') + (z - 2 * p - (u*u + 2*u) * m) >= 0
        # which is equivalent to
        # u >= -(w + 2p) / 4m
        # u^2 - 2u <= (w + z) / m
        u_ = -(w + 2*p)/4/m
        q2 = 2*(m + p) + q
        if u_ < 1:
            u_ = sp.S(1)
        w2 = w + 2 * p + 4 * u_ * m
        
        if 2*q2 + w2 + min(2*q2, w2) + (z - 2*p - (u_**2 + 2*u_)*m) < 0:
            u_ = None

        if u_ is not None:
            y = [
                m,
                p,
                min(w2, q2),
                q2 - min(w2, q2),
                w2 - min(w2, q2),
                radsimp(z - u_ * (u_ + 2) * m - 2*p + (w2 + q2 - 2 * min(w2, q2))),
                rem
            ]

            if all(_ >= 0 for _ in y):
                exprs = [
                    CyclicSum(a*b*(a-b)**2*(a+b-u_*c)**2),
                    CyclicProduct((a-b)**2),
                    CyclicSum(a*b*(a-c)**2*(b-c)**2),
                    CyclicSum(b**2*c**2*(a-b)*(a-c)),
                    CyclicSum(a*(a-b)*(a-c)) * CyclicProduct(a),
                    CyclicSum(a*(b-c)**2) * CyclicProduct(a),
                    CyclicProduct(a**2)
                ]
                return sum_y_exprs(y, exprs)


    if True:
        # Case 5. the border is tight.
        y_hex = radsimp(q - p**2/4/m - 2*m)
        if y_hex >= 0:
            x_ = p / m / 4
            y = [
                m,
                y_hex,
                radsimp(w - m*(4 - 4*x_) - y_hex),
                radsimp(z + w + m*(3*x_**2 + 2*x_) + 2*y_hex),
                rem
            ]
            if all(_ >= 0 for _ in y):
                exprs = [
                    CyclicSum(a*b*(a**2+b**2-2*c**2+2*x_*a*b-x_*a*c-x_*b*c)**2),
                    CyclicSum(a*b*(a-c)**2*(b-c)**2),
                    CyclicSum(a*(a-b)*(a-c)) * CyclicProduct(a),
                    CyclicSum(a*(b-c)**2) * CyclicProduct(a),
                    CyclicProduct(a**2)
                ]
                return sum_y_exprs(y, exprs)

    if coeff.is_rational:
        # Easy case 6, when we can extract some s(ab) * quartic
        # e.g. s(ab)s(a4-4a3b-4ab3+7a2bc+15(a2b2-a2bc))
        #      s(ab)s(a2(a-b)(a-c))

        # Idea: subtract some up(a-b)2 and vs(bc(a-b)2(a-c)2), so that:
        # 3m(m+q-v+2u) >= 3(p-u)^2               => v <= m+q-(p-u)^2/m+2u
        # -m - 2*p + 4*u - v + w >= 0            => v <= 4u+w-m-2p
        # -p + u + 2*v + w + z >= 0              => v >= (-u+p-w-z)/2
        u_ = -(-2*m - 5*p + 3*w + z)/9
        v_ = (-u_ + p - w - z)/2
        if u_ >= 0 and v_ >= 0:
            pass
        elif u_ >= 0 and v_ < 0:
            u_ = (2*p+m-w) / 4
        elif u_ < 0:
            u_ = sp.S(0)
            v_ = max(sp.S(0), (-u_+p-w-z)/2)

        tmp = radsimp(m + q - (p-u_)**2/m + 2*u_)
        if v_ <= tmp:
            pass
        elif m + p >= u_: # symmetric axis of the parabola >= u_
            u_ = m + p
            v_ = max(sp.S(0), (-u_+p-w-z)/2)

        if u_ >= 0 and 0 <= v_ <= tmp and v_ <= 4*u_+w-m-2*p and v_ >= (-u_+p-w-z)/2:
            y = [
                m / 2,
                (tmp - v_)/2,
                u_,
                v_,
                -m - 2*p + 4*u_ - v_ + w,
                -p + u_ + 2*v_ + w + z,
                rem
            ]

            if all(_ >= 0 for _ in y):
                exprs = [
                    CyclicSum(a*b) * CyclicSum((a-b)**2*(a+b-(-p+u_)/m*c)**2),
                    CyclicSum(a*b) * CyclicSum(a**2*(b-c)**2),
                    CyclicProduct((a-b)**2),
                    CyclicSum(a*b*(a-c)**2*(b-c)**2),
                    CyclicSum(a*(a-b)*(a-c)) * CyclicProduct(a),
                    CyclicSum(a*(b-c)**2) * CyclicProduct(a),
                    CyclicProduct(a**2)
                ]
                return sum_y_exprs(y, exprs)


    # real start below
    # process:
    p, q, w, z = radsimp([p / m, q / m, w / m, z / m])

    if not ((p <= -4 and q >= p*p/4 + 2) or (p >= -4 and q >= -2*(p + 1))):
        # checking the border yields that:
        # when p <= -4, we must have q >= p*p / 4 + 2
        # when p >= -4, we must have q >= -2*(p + 1)

        # both two cases should imply 2*p + q + 2 >= 0
        return None
    
    # First, we peek whether there are nontrivial roots in the interior with a == b.
    # f(a,a,1)/(a-1)^2 = sym(a)
    u = sp.symbols('u')
    root, u_ = None, None
    sym = ((2*p + q + 2)*a**3 + (4*p + 2*q + 2*w + 2*z + 6)*a**2 + (2*p + w + 4)*a + 2).as_poly(a)

    # sym should be nonnegative when a >= 0
    # sym_roots_count = sp.polys.count_roots(sym, 0, None)
    # if sym_roots_count > 1:
    #     return None
    # elif sym_roots_count == 1:
    #     # yes there are nontrivial roots
    #     root = list(filter(lambda x: x >= 0, sp.polys.roots(sym).keys()))[0]
    #     if root != 1:
    #         u_ = radsimp(1 / root + 1)
    #     else:
    #         root = None
    sym_diff = sym.diff(a)
    sym_gcd = sp.gcd(sym, sym_diff)
    if sym_gcd.degree() == 1:
        root = radsimp(-sym_gcd.coeff_monomial((0,)) / sym_gcd.coeff_monomial((1,)))
        if root != 1:
            u_ = radsimp(1 / root + 1)
        else:
            root = None

    if u_ is not None:
        # The polynomial must be in the form of 
        # c1 * s((2a(b+c)-bc)(b-c)^2(b+c-ua)^2) + (1 - c1) * s(bc(b-c)^2(b+c-ua)^2) + rp(a-b)^2.
        # Note that SOS theorem states that
        # s((2a(b+c)-bc)(b-c)^2(b+c-ua)^2) = s(bc((a-b)(a+b-uc)-(c-a)(c+a-ub))2) >= 0
        c1 = radsimp((2*p + q + 2)/(4*(u_ - 1)**2))
        r = radsimp((p*u_**2 + p + q*u_ + 2*u_)/(u_ - 1)**2)
        if 0 <= c1 <= 1 and r >= 0:
            y = radsimp([
                m * c1,
                m * (1 - c1),
                m * r,
                rem
            ])
            exprs = [
                CyclicSum(b*c*((a-b)*(a+b-u_*c) - (c-a)*(c+a-u_*b)).expand()**2),
                CyclicSum(b*c*(b-c)**2*(b+c-u_*a)**2),
                CyclicProduct((a-b)**2),
                CyclicProduct(a**2)
            ]
            return sum_y_exprs(y, exprs)


    # Second, determine t by coefficient at (4,2,0) and (3,3,0)
    # this is done by subtracting as much ((a-b)*(b-c)*(c-a))^2 as possible
    # until there are zeros at the border

    # subtract some ((a-b)*(b-c)*(c-a))^2
    # p - r == 2*t,   q + 2*r == t*t + 2
    # r = p + 4 + 2 * sqrt(2*p + q + 2)

    # Case A. r is irrational, instead we subtract some hexagrams
    r = radsimp(p + 4 + sp.sqrtdenest(2 * sp.sqrt(2 * p + q + 2)))
    y_hex = 0
    if coeff.is_rational and not isinstance(r, sp.Rational):
        # make a perturbation on q so that 2*p + q' + 2 is a square
        
        if u_ is None:
            # Case A.A there are no nontrivial roots, then we can make any slight perturbation
            # here we use s(ab(a-c)2(b-c)2)
            dw = 1
            dz = -3
        else:
            # Case A.B there exists nontrivial roots, then we make a slight perturbation
            # using the hexagram generated by the root
            dw = radsimp(1 / (u_ - 1)**2)
            dz = radsimp((-u_**2 + u_ - 1) / (u_ - 1)**2)

        numer_r = sp.sqrt(2 * p + q + 2).n(20)
        for numer_r2 in rationalize_bound(numer_r, direction = -1, compulsory = True):
            if numer_r2 >= 0 and p + 4 + 2 * numer_r2 >= 0:
                q2 = numer_r2 ** 2 - 2 * p - 2
                y_hex = q - q2
                w2 = w - dw * y_hex
                z2 = z - dz * y_hex
                if y_hex >= 0:
                    sym = ((2*p + q2 + 2)*a**3 + (4*p + 2*q2 + 2*w2 + 2*z2 + 6)*a**2 + (2*p + w2 + 4)*a + 2).as_poly(a)
                    if sym.LC() >= 0 and sp.polys.count_roots(sym, 0, None) <= 1:
                        q = q2
                        break
        else:
            return None

        w -= dw * y_hex
        z -= dz * y_hex
        r = radsimp(p + 4 + 2 * sp.sqrt(2 * p + q + 2))

    # Case B. now 2*p + q + 2 is a square and r is rational
    t = - (p - r) / 2
    w -= -2 * r
    z -= 2 * r
    
    # Third, determine u by the coefficient at (4,1,1), which is w
    coeff_z = lambda u__: -(t**2*u__**2 - t**2*u__ + t**2 - 4*t*u__ - u__**4 + 7*u__**2 - 6*u__ + 4)/(u__ - 1)**2
    if u_ is None:
        if t == 2:
            u_ = 2 - w / 4
            if u_ != 1 and z < coeff_z(u_):
                u_ = 1
        else:
            equ = (-4*u**3 + (4*t - w + 8)*u**2 + (-8*t + 2*w - 4)*u + t**2 - w + 4).as_poly(u)
            for u__ in sp.polys.roots(equ, cubics = False):
                if isinstance(u__, sp.Rational):
                    if coeff_z(u__) <= z:
                        u_ = u__
                        break
                
            # find a rational approximation
            if u_ is None:
                for u__ in sp.polys.nroots(equ)[::-1]:
                    if u__.is_real:
                        if coeff_z(u__) <= z:
                            direction = ((t**2 - 4*t + 2*u__**3 - 6*u__**2 + 6*u__ + 2)/(u__ - 1))
                            direction = 1 if direction > 0 else -1
                            for u_ in rationalize_bound(u__, direction = direction, compulsory = True):
                                if u_ != 1 and coeff_z(u_) <= z and (u_ - u__) * direction > 0:
                                    break
                                u_ = None
            
    # print('W Z R Y U T =', w, z, r, y_hex, u_, t)
    if u_ is None:
        return None

    if u_ != 1:
        # rather u_ in the introduction, we use u_ + 1 here as u_
        phi = radsimp((t * u_ - u_**2 + 2*u_ - 3) / (u_ - 1))

        multiplier = CyclicSum(a)
        y = radsimp([
            y_hex if root is None else sp.S(0),
            r,
            rem / m,
            sp.S(1),
            sp.S(0) if root is None else 1 / (u_ - 1)**2 * y_hex
        ])
        # print(r, t, u_, phi, y)

        pw1 = (w - (t**2 + 4*t*u_**2 - 8*t*u_ - 4*u_**3 + 8*u_**2 - 4*u_ + 4)/(u_ - 1)**2) / 2
        pw2 = z - coeff_z(u_)
        pw3 = (t - 2*u_) ** 2 / 2 / (u_ - 1)**2 + (sp.S(0) if root is None else y_hex / 2 / (u_ - 1)**2)
        pw1, pw2, pw3 = radsimp([m * pw1, m * pw2, m * pw3])

        if any(_ < 0 for _ in y) or any(_ < 0 for _ in [pw1, pw2, pw3]):
            return None

        p1 = pw1 * CyclicSum(a)**2 * CyclicSum((a-b)**2) +\
            pw2 * CyclicSum(a) * CyclicSum(a*(b-c)**2) +\
            pw3 * CyclicSum((b-c)**2*(b+c-u_*a)**2)
        p1 = p1.as_coeff_Mul()

        y = radsimp([_ * m for _ in y])
        exprs = [
            CyclicSum(a) * CyclicSum(a*b*(a-c)**2*(b-c)**2),
            CyclicSum(a) * CyclicProduct((a-b)**2),
            CyclicSum(a) * CyclicProduct(a**2),
            CyclicSum(a*(b-c)**2*(a**2+b**2+c**2-t*a*b-t*a*c+phi*b*c)**2),
            CyclicSum(c*(a-b)**2*(u_*a*b-(u_-1)*a*c-(u_-1)*b*c)**2)
        ]
        return (sum_y_exprs(y, exprs) + p1[0] * p1[1] * CyclicProduct(a)) / multiplier

    elif u_ == 1:
        # very special case, it must be t == 2
        # f(a,b,c) = (s(ab(a-b)2(a+b-c)2)-4p(a-b)2)
        # then f(a,b,c)s(a) = s(a(b-c)2(b+c-a)4) + 2abcs((b-c)2(b+c-a)2)
        
        multiplier = CyclicSum(a)
        y = radsimp([
            y_hex if root is None else sp.S(0),
            r,
            rem / m,
            sp.S(1),
        ])

        pw1 = w - 4
        pw2 = z + w + 1
        pw3 = sp.S(2)

        if any(_ < 0 for _ in y) or pw1 < 0 or pw2 < 0:
            return None

        pw1, pw2, pw3 = radsimp([m * pw1, m * pw2, m * pw3])
        p1 = pw1  * CyclicSum(a) * CyclicSum(a*(a-b)*(a-c))\
            + pw2 * CyclicSum(a) * CyclicSum(a*(b-c)**2)\
            + pw3 * CyclicSum((b-c)**2*(b+c-a)**2)
        p1 = p1.as_coeff_Mul()

        y = radsimp([_ * m for _ in y])
        exprs = [
            CyclicSum(a) * CyclicSum(a*b*(a-c)**2*(b-c)**2),
            CyclicSum(a) * CyclicProduct((a-b)**2),
            CyclicSum(a) * CyclicProduct(a**2),
            CyclicSum(a*(b-c)**2*(b+c-a)**4),
        ]
        return (sum_y_exprs(y, exprs) + p1[0] * p1[1] * CyclicProduct(a)) / multiplier

    return None


def _sos_struct_sextic_symmetric_quadratic_form(poly, coeff):
    """
    Theorem:
    Let F0 = s(a^6+a^5b+a^5c+a^4bc-2a^3b^2c-2a^3bc^2) and f(a,b,c) = s(xa^2 + yab).

    Then we have
    F_{x,y}(a,b,c) = F0 - 2s(a^4-a^2bc)f(a,b,c) + s(a^2-ab)f(a,b,c)^2 >= 0
    because
    F(a,b,c) * s(a^2-ab) = (s(a^2-ab)f(a,b,c) - s(a^4-a^2bc))^2 + s(ab)p(a-b)^2 >= 0.

    So we try to write the original polynomial in such quadratic form. Note that for such F,
    it has three multiplicative roots on the symmetric axis b=c=1, one of which is the centroid a=b=c=1.
    The other two roots determine the coefficient x and y.

    For normal polynomials without three multiplicative roots, we first write the symmetric axis in the form
    of (a-1)^2 * ((a^2+..a+..)^2 + (..a+..)^2 + ...). Now we apply the theorem to each of them, and then
    merge their f(a,b,c) accordingly.


    Moreover, there exists u, v such that u+v = (2-2*y)/(x-1), uv = (2*x+y-2)/(x-1) so that
    F(a,b,c) = (x-1)^2s((a-b)(a-c)(a-ub)(a-uc)(a-vb)(a-vc)) + (x^2-xy+y^2-y)p(a-b)^2

    Examples
    --------
    72p(a+b-2c)2+s(a2-ab)s(11a2-14ab)2

    (s((b-c)2(7a2-b2-c2)2)-112p(a-b)2)

    s((a-b)2(-a2-b2+2c2+2(ab-c2)-3s(ab)+2s(a2))2)

    s(a5)s(a/3)+19abc(s(ab)s(a/3)-3s(a/3)3)+3abc(abc-2s(a/3)3)

    s((a-b)(a-c)(a-2b)(a-2c)(a-18b)(a-18c))-53p(a-b)2

    s((a-b)(a-c)(a-7b)(a-7c)(a-3b)(a-3c))-73p(a-b)2

    s((b2+c2-5a(b+c))2(b-c)2)-22p(a-b)2

    s(a6+6a5b+6a5c-93a4b2+3a4bc-93a4c2+236a3b3+87a3b2c+87a3bc2-240a2b2c2)

    s(a6-21a5b-21a5c-525a4b2+1731a4bc-525a4c2+11090a3b3-13710a3b2c-13710a3bc2+15690a2b2c2)

    s(a2(a-b)(a-c)(a-5b)(a-5c))+s(a2(a-b)(a-c)(a-3b)(a-3c))+15p(a-b)2
    
    s(a2(a-b)(a-c)(3a-2b)(3a-2c))+15p(a-b)2        (real)

    s(56a6-41a5b-56a4b2+82a3b3-56a2b4-83a3b2c-83a2b3c-41ab5+98a2b2c2+124a4bc)      (real)

    p(a2+s(a/6)2)-125/8p(a)s(a/6)3                 (real)

    s(36a6-84a5b-84a5c+87a4b2+130a4bc+87a4c2-77a3b3-55a3b2c-55a3bc2+15a2b2c2)       (real)

    (s(a2(a-b)(a-c)(a-3b)(a-3c))+p(a-b)2)+s(a2-2ab)2s(a2-ab)/4+s((a-b)(a-c)(a-2b)(a-2c)(a-4b)(a-4c))-6p(a-b)2   (real)
    
    References
    -------
    [1] https://artofproblemsolving.com/community/c6t243f6h3013463_symmetric_inequality

    [2] https://tieba.baidu.com/p/8261574122

    TODO:
    1. Remove the use of prove_univariate to support irrational coeffs also.
    """
    a, b, c = sp.symbols('a b c')
    sym0 = poly.subs({b:1,c:1}).div((a**2-2*a+1).as_poly(a))
    if not sym0[1].is_zero:
        return None

    # write the symmetric axis in sum-of-squares form
    sym = prove_univariate(sym0[0], return_raw = True)
    if sym is None or len(sym[1][1]) > 0: # this is not positive over R
        return None
    # print(sym)

    def _solve_from_sym(sym):
        # given symmetric axis with three roots, we determine the exact coefficient f(a,b,c)
        # (x,y) are the parameters of f(a,b,c). While coeff stands for the scaling factor.
        w, v, u = [sym.coeff_monomial((i,)) for i in range(3)]
        x, y = (2*u + v - 2*w)/(4*u + v - 2*w), (4*u - 2*w)/(4*u + v - 2*w)
        coeff = v / (2*y - 2) if y != 1 else (w / (2*x + y - 2) if x != sp.Rational(1,2) else 2*u)
        return x, y, coeff

    params = []
    for coeff0, sym_part in zip(sym[0][1], sym[0][2]):
        x_, y_, coeff1 = _solve_from_sym(sym_part)
        if coeff1 is sp.nan:
            return None
        # part_poly = coeff0 * coeff1**2 * pl(f's(a6+a5b+a5c+a4bc-2a3b2c-2a3bc2)-2s(a4-a2bc)s({x_}a2+{y_}ab)+s(a2-ab)s({x_}a2+{y_}ab)2')
        # print((x_, y_), coeff0 * coeff1**2, poly_get_factor_form(part_poly))
        params.append((coeff0 * coeff1**2, x_, y_))

    coeff0, x, y, m, p, n, t_coeff, rem_coeff, rem_ratio = _merge_quadratic_params(params)

    # ker_coeff is the remaining coefficient of (a-b)^2(b-c)^2(c-a)^2
    ker_coeff = (poly.coeff_monomial((4,2,0)) - coeff0 * (3*x**2 - 2*x*y - 2*x + y**2) - (n - p + m))

    # each t_ exchanges for 27/4p(a-b)^2 because s(a^2-ab)^3 = 1/4 * p(2a-b-c)^2 + 27/4 * p(a-b)^2
    ker_coeff += 27 * t_coeff / 4

    # print('Coeff =', coeff0, 'ker =', ker_coeff)
    # print('  (x,y) =', (x, y), 'ker_std =', ker_coeff / coeff0)
    # print('  (m,p,n,t) = ', (m, p, n, t_coeff))

    return _sextic_sym_axis.solve(
        coeff0, x, y, ker_coeff, t_coeff, rem_coeff, rem_ratio
    )


class _sextic_sym_axis:
    """    
    Let F0 = s(a^6+a^5b+a^5c+a^4bc-2a^3b^2c-2a^3bc^2) and f(a,b,c) = s(xa^2 + yab).
    Define
    F_{x,y}(a,b,c) = F0 - 2s(a^4-a^2bc)f(a,b,c) + s(a^2-ab)f(a,b,c)^2.

    The class provides different methods to solve F_{x,y}(a,b,c) >= 0. There are also
    two types of solvers.

    * Type0:
    F(x, y, ker_coeff):
        Solve F_{x,y} + ker_coeff * p(a-b)^2 >= 0.
        Return solution, flg. If flg == 0, solution is on R. If flg == 1, solution is on R+. If flg == 2, the solver fails.

    * Type1:
    F(x, y):
        Solve F_{x,y} >= 0.
        Return p1, c1, c2, multiplier such that
        F_{x,y} * s(mutiplier[0]*a^2 + multiplier[1]*a*b) = p1 + s(c1*a^2 + c2*a*b) * p(a-b)^2
    """
    @staticmethod
    def _F_square(x, y, ker_coeff):
        """
        When x + y == 5/3 and x != 1,
        F_{x,y} = (x-1)^2/4 * s((b-c)^2(b+c-za))^2 + 3(x-1)(9x-5)/4 * p(a-b)^2
        """
        if x + y == sp.S(5)/3 and x != 1:
            z = 2*(3*x - 2) / 3 / (x - 1)
            ker_coeff2 = -3*(x - 1)*(9*x - 5)/4
            if ker_coeff >= ker_coeff2:
                p0 = 2*a**3 - (z + 1)*b*c**2 - (z + 1)*b**2*c + 2*z*a*b*c
                p0 = p0.together().as_coeff_Mul()
                p1 = sp.Add(
                    (x - 1)**2/4 * p0[0]**2 * CyclicSum(p0[1])**2,
                    (ker_coeff - ker_coeff2) * CyclicProduct((a-b)**2)
                )
                return p1, 0
        return None, 2

    @staticmethod
    def _F_trivial(x, y, ker_coeff):
        """
        F(a,b,c) + p(a-b)^2/3 = s((a-b)^2((3*x-3)*a^2+(3*y-4)*a*b+(3*y-2)*a*c+(3*x-3)*b^2+(3*y-2)*b*c+(3*x-1)*c^2)^2)/18
        """
        if ker_coeff >= sp.Rational(1,3):
            # Case that we do not need to higher the degree because
            # F(a,b,c) + p(a-b)^2/3 = s((a-b)^2((3*x-3)*a^2+(3*y-4)*a*b+(3*y-2)*a*c+(3*x-3)*b^2+(3*y-2)*b*c+(3*x-1)*c^2)^2)/18
            p1 = (3*x - 3)*a**2 + (3*y - 4)*a*b + (3*y - 2)*a*c + (3*x - 3)*b**2 + (3*y - 2)*b*c + (3*x - 1)*c**2
            if x < 1: p1 = -p1
            solution = sp.Add(
                sp.Rational(1,18) * CyclicSum((a-b)**2*p1**2),
                (ker_coeff - sp.Rational(1,3)) * CyclicProduct((a-b)**2)
            )
            return solution, 0
        return None, 2


    @staticmethod
    def _F_regular(x, y):
        """
        Return p1, c1, c2, multiplier such that
        F(x,y) * s(mutiplier[0]*a^2 + multiplier[1]*a*b) = p1 + s(c1*a^2 + c2*a*b) * p(a-b)^2
        """
        p1 = 2 * (CyclicSum(a**2 - b*c)*CyclicSum(x*a**2 + y*a*b) - CyclicSum(a**4 - a**2*b*c))**2
        c1, c2 = sp.S(0), sp.S(2)
        return p1, c1, c2, (2, -2)

    @staticmethod
    def _F_sos2(x, y, z_type = 0):
        """
        F(a,b,c) * 2s(a2-ab) = 1/9 * s(h(a,b,c)^2) + p(a-b)^2 * s(c1*a^2 + c2*a*b)
        where h(a,b,c), c1, c2 are defined as below.
        c12 = [
            ((72*x**2 - 18*x*y - 96*x - 9*y**2 + 30*y + 20)/6, (36*x**2 + 72*x*y - 120*x - 45*y**2 + 24*y + 40)/6),
            (3*(x - 1)*(9*x - 5)/2, (27*x**2 + 54*x*y - 90*x - 42*y + 55)/2),
        ]

        Return p1, c1, c2, multiplier such that
        F(x,y) * s(mutiplier[0]*a^2 + multiplier[1]*a*b) = p1 + s(c1*a^2 + c2*a*b) * p(a-b)^2
        """
        z = [-(2*x + y - 2)/(2*(x - 1)), (x + 2*y - 3)/(2*(x - 1))][z_type]
        def _compute_h_c1_c2(z):
            """
            The following h, c1, c2 satisfy that
            F(a,b,c) * 2s(a2-ab) = 1/9 * s(h(a,b,c)^2) + p(a-b)^2 * s(c1*a^2 + c2*a*b)
            for arbitrary z. 
            Therefore, we can choose z such that c1 >= 0 and c1 + c2 >= 0.
            This is often done by selecting the symmetric axis of the parabola.
            """
            # h = -a**4 - a**3*b*z + a**3*c*z + 2*a**2*b**2*z + 6*a**2*b**2 - a**2*b*c*z - 2*a**2*b*c - a**2*c**2*z - 3*a**2*c**2 - a*b**3*z - a*b**2*c*z - 2*a*b**2*c + 2*a*b*c**2*z + 4*a*b*c**2 - b**4 + b**3*c*z - b**2*c**2*z - 3*b**2*c**2 + 2*c**4\
            #     + x*(a**4 + a**3*b*z - a**3*c*z - 2*a**3*c - 2*a**2*b**2*z - 6*a**2*b**2 + a**2*b*c*z + 6*a**2*b*c + a**2*c**2*z + 3*a**2*c**2 + a*b**3*z + a*b**2*c*z + 6*a*b**2*c - 2*a*b*c**2*z - 12*a*b*c**2 + 2*a*c**3 + b**4 - b**3*c*z - 2*b**3*c + b**2*c**2*z + 3*b**2*c**2 + 2*b*c**3 - 2*c**4)\
            #     + y*(2*a**3*c - 2*a**2*b**2 - 2*a**2*b*c + a**2*c**2 - 2*a*b**2*c + 4*a*b*c**2 - 2*a*c**3 + 2*b**3*c + b**2*c**2 - 2*b*c**3)
            h = (a-b)*(
                -3*a**3 + a**2*b*z - 3*a**2*b - a**2*c*z + a*b**2*z - 3*a*b**2 - 4*a*b*c*z - 6*a*b*c + 3*a*c**2*z + 9*a*c**2 - 3*b**3 - b**2*c*z + 3*b*c**2*z + 9*b*c**2 - 2*c**3*z\
                + x*(3*a**3 - a**2*b*z - a**2*b + a**2*c*z - 2*a**2*c - a*b**2*z - a*b**2 + 4*a*b*c*z + 16*a*b*c - 3*a*c**2*z - 9*a*c**2 + 3*b**3 + b**2*c*z - 2*b**2*c - 3*b*c**2*z - 9*b*c**2 + 2*c**3*z + 2*c**3)\
                + y*(4*a**2*b + 2*a**2*c + 4*a*b**2 - 4*a*b*c - 3*a*c**2 + 2*b**2*c - 3*b*c**2 - 2*c**3)
            ).expand().together()
            c1 = 20*x**2 - x*y - 30*x - y**2 + 3*y + z**2*(-x**2 + 2*x - 1) + z*(x**2 + 2*x*y - 4*x - 2*y + 3) + 9
            c2 = 16*x**2 + 28*x*y - 48*x - 8*y**2 - 6*y + z**2*(x**2 - 2*x + 1) + z*(8*x**2 + 7*x*y - 20*x - 7*y + 12) + 21
            c1 = 2 * c1 / 3
            c2 = 2 * c2 / 3
            return h, c1, c2
        func_h, c1, c2 = _compute_h_c1_c2(z)
        p1 = sp.Rational(1,9) * CyclicSum(func_h**2)
        return p1, c1, c2, (2, -2)


    @staticmethod
    def _F_alternative_find_z(x, y, z_type = 0, ker_coeff = 0):
        """
        Return proper parameter z for the function _F_alternative.
        """
        if z_type == 1:
            return -x - 1
        z = sp.symbols('z')
        supplement = (ker_coeff * (x + y - 1)**2, ker_coeff * (2*x + 2*y - 4)*(x + y - 1))

        c1 = (-3*x - 3*y + 5)*z**2 + (-6*x**2 + 4*x + 6*y**2 - 16*y + 10)*z + 6*x**2*y - 7*x**2 + 3*x*y**2 - 14*x*y + 10*x - 3*y**3 + 11*y**2 - 12*y + 5 + supplement[0]
        c2 = (-6*x*y - 6*y**2 + 10*y)*z - 2*x**2 + 6*x*y**2 - 8*x*y + 2*x + 6*y**3 - 19*y**2 + 14*y + supplement[1]
        # find z such that 2*c1 >= c2 and c1 + c2 >= 0
        c1, c2 = c1.as_poly(z), c2.as_poly(z)
        f1, f2 = 2*c1 - c2, c1 + c2
        f_gcd = sp.gcd(f1, f2)
        if f_gcd.degree() == 1:
            return -f_gcd.coeff_monomial((0,)) / f_gcd.coeff_monomial((1,))
        for (z1, z2), _ in sp.polys.intervals(f1 * f2):
            for z_ in (z1, z2):
                if f1(z_) >= 0 and f2(z_) >= 0:
                    return z_
        return -x + 3*y/2 - 1

    @staticmethod
    def _F_alternative(x, y, z = 0):
        """
        F(a,b,c) * s(a^2 + (2*x + 2*y - 4)/(x + y - 1)*a*b)
            = 1/(x + y - 1)^2/9 * s(a^2-ab) * s(a*(3*a^2*(x-1)*(x+y-1)+3*a*(b+c)*(x^2+2*x*y-4*x+y^2-3*y+3)+b*c*(3*x*y+3*y^2-9*y+4)))^2
            + (3x + 3y - 5)/(x + y - 1)^2 * s(a^2(b-c)^2(a^2*(x-y+1)+a*(b+c)*(2*y-2)+(b^2+c^2)*(x-1)+z(a-c)(a-b))^2)
            + s(c1*a^2 + c2*a*b) * p(a-b)^2
        where c1 = (-3*x-3*y+5)*z^2+(-6*x^2+4*x+6*y^2-16*y+10)*z+6*x^2*y-7*x^2+3*x*y^2-14*x*y+10*x-3*y^3+11*y^2-12*y+5
                c2 = (-6*x*y-6*y^2+10*y)*z-2*x^2+6*x*y^2-8*x*y+2*x+6*y^3-19*y^2+14*y

        Return p1, c1, c2, multiplier such that
        F(x,y) * s(mutiplier[0]*a^2 + multiplier[1]*a*b) = p1 + s(c1*a^2 + c2*a*b) * p(a-b)^2
        """
        w = x + y - 1
        if w < sp.S(2)/3: # 3x + 3y - 5 < 0
            return None

        def _compute_h_c1_c2(z):
            h1 = 3*a**2*(x - 1)*w + 3*a*(b + c)*(x**2 + 2*x*y - 4*x + y**2 - 3*y + 3) + b*c*(3*x*y + 3*y**2 - 9*y + 4)
            h2 = a**2*(x - y + 1) + a*(b + c)*(2*y - 2) + z*(a - b)*(a - c) + (b**2 + c**2)*(x - 1)
            h1 = h1.expand().together()
            h2 = h2.expand().together()
            c1 = (-3*x - 3*y + 5)*z**2 + (-6*x**2 + 4*x + 6*y**2 - 16*y + 10)*z + 6*x**2*y - 7*x**2 + 3*x*y**2 - 14*x*y + 10*x - 3*y**3 + 11*y**2 - 12*y + 5
            c2 = (-6*x*y - 6*y**2 + 10*y)*z - 2*x**2 + 6*x*y**2 - 8*x*y + 2*x + 6*y**3 - 19*y**2 + 14*y
            c1, c2 = c1 / w**2, c2 / w**2
            return h1, h2, c1, c2
        func_h1, func_h2, c1, c2 = _compute_h_c1_c2(z)
        p1 = 1/w**2/18 * CyclicSum((a-b)**2) * CyclicSum(a * func_h1)**2\
            + (3*x + 3*y - 5)/w**2 * CyclicSum(a**2 * (b-c)**2 * func_h2**2)
        return p1, c1, c2, (1, (2*x + 2*y - 4)/w)


    @staticmethod
    def rem_poly(rem_coeff, rem_ratio):
        return rem_coeff * (CyclicSum(a**2 + rem_ratio*a*b)**2 if not rem_ratio is sp.oo else CyclicSum(a*b)**2)

    @staticmethod
    def _rem_regular(t_coeff, rem_coeff, rem_ratio, multiplier):
        """
        Write t/4 * s((a-b)^2)p(a+b-2c)^2 + 2 * s(a^2-ab)^2 * rem_poly in the form of (p2 + s(c1*a^2 + c2*a*b) * p(a-b)^2)
        """
        rem_poly = _sextic_sym_axis.rem_poly(rem_coeff, rem_ratio)
        if multiplier == (2, -2):
            p2 = t_coeff/4 * CyclicSum((a-b)**2) * CyclicProduct((a+b-2*c)**2) + 2 * CyclicSum(a**2-a*b)**2 * rem_poly
        else:
            p0 = _wrap_c1_c2(multiplier[0], multiplier[1])
            p2 = t_coeff/4 * p0 * CyclicProduct((a+b-2*c)**2) + CyclicSum((a-b)**2)/2 * p0 * rem_poly
        return p2, sp.S(0), sp.S(0)

    @staticmethod
    def _rem_sos(t_coeff, rem_coeff, rem_ratio, multiplier):
        """
        Solve t/4 * s((a-b)^2)p(a+b-2c)^2 + 2 * s(a^2-ab)^2 * rem_poly >= 0.
        The inequality is tried to solve on R in advance.

        Theorem:
        G(a,b,c) = s(a2-ab)2s(ua2+vab)2 + p(a-b)2((3v2-24u2+6uv)/4s(a)2+9(2u-v)2/4s(ab)) >= 0.
        Proof:
        G(a,b,c) = 1/(8u^2) * s((a-b)2(a+b-2c)2(2u2(a2-ab+ac+b2+bc)+uv(3ab+ac+bc+c2))2)
        """
        if multiplier != (2, -2):
            return None
        if rem_ratio is sp.oo:
            u2, uv, v2 = sp.S(0), sp.S(0), rem_coeff * 2
        else:
            u2, uv, v2 = rem_coeff * 2, rem_coeff * rem_ratio * 2, rem_coeff * rem_ratio**2 * 2
        func_h = (2*u2*(a**2-a*b+a*c+b**2+b*c) + uv*(3*a*b+a*c+b*c+c**2)).expand().together()
        p21 = CyclicSum((a-b)**2*(a+b-2*c)**2*func_h**2)

        p2 = t_coeff/4 * CyclicSum((a-b)**2) * CyclicProduct((a+b-2*c)**2) + 1/(8*u2) * p21
        c1 = -(3*v2 - 24*u2 + 6*uv)/4
        c2 = -9*(4*u2 - 4*uv + v2)/4 + 2*c1
        return p2, c1, c2

    @staticmethod
    def _merge_remainder_terms(p1, c11, c12, p2, c21, c22, ker_coeff, multiplier = (2,-2)):
        """
        Merge p1 + s(c11*a^2 + c12*a*b) * p(a-b)^2 and p2 + s(c21*a^2 + c22*a*b) * p(a-b)^2
        and ker_coeff * s(multiplier[0]*a^2 + multiplier[1]*a*b) * p(a-b)^2.

        Consider the quadratic form s(xa^2 + yab). It is positive on R when {x+y>=0, 2x>=y}, on R+ when {x+y>=0}.
        Both two regions are convex. Thus we only need to check whether the vertices of the segment fall
        in the region.

        Returns
        ---------
        sol: sp.Expr
            The solution.
        flg: int
            When flg == 0, it is solved on R. When flg == 1, it is solved on R+. When flg == 2, it is not solved.
        """
        if multiplier[0] + multiplier[1] < 0:
            return None, 2

        c1 = c11 + c21 + multiplier[0]*ker_coeff
        c2 = c12 + c22 + multiplier[1]*ker_coeff

        multiplier_func = _wrap_c1_c2(multiplier[0], multiplier[1])

        if c1 >= 0 and c1 + c2 >= 0:
            sol = p1 + p2 + _wrap_c1_c2(c1, c2) * CyclicProduct((a-b)**2)
            flg = 0 if 2*c1 >= c2 and 2*multiplier[0] >= multiplier[1] else 1
            return sol / multiplier_func, flg

        return None, 2


    @staticmethod
    def _wrap_F(f_type, f_solver):
        cls = _sextic_sym_axis
        if f_type == 0:
            def _F(x, y, coeff0, ker_coeff, t_coeff, rem_coeff, rem_ratio):
                solution, flg = f_solver(x, y, ker_coeff/coeff0)
                if solution is not None:
                    solution = sp.Add(
                        coeff0 * solution,
                        t_coeff/4 * CyclicProduct((a+b-2*c)**2),
                        sp.Rational(1,2) * CyclicSum((a-b)**2) * cls.rem_poly(rem_coeff, rem_ratio)
                    )
                return solution, flg

        else:
            def _F(x, y, coeff0, ker_coeff, t_coeff, rem_coeff, rem_ratio):
                solutions = []

                f_sol = f_solver(x, y)
                if f_sol is None:
                    return None, 2
                p1, c1, c2, multiplier = f_sol
                p1, c1, c2 = p1 * coeff0, c1 * coeff0, c2 * coeff0
                REM_SOLVERS = [
                    cls._rem_regular, cls._rem_sos
                ]
                for rem_sol in REM_SOLVERS:
                    rem_sol = rem_sol(t_coeff, rem_coeff, rem_ratio, multiplier)
                    if rem_sol is None:
                        continue
                    p2, c21, c22 = rem_sol
                    solution, flg  = cls._merge_remainder_terms(p1, c1, c2, p2, c21, c22, ker_coeff, multiplier)
                    if flg == 0:
                        return solution, 0
                    elif flg == 1:
                        solutions.append(solution)

                if len(solutions) > 0:
                    return solutions[0], 1
                return None, 2

        return _F

    @staticmethod
    def _F_tighter_bound_border(x, y, ker_coeff):
        """
        Enhanced version of proving F_{x,y} >= 0. For
        R(a,b,c) = s((a-b)(a-c)(a-ub)(a-uc)(a-vb)(a-vc)) - wp(a-b)^2.
        Denote suv = u+v, puv = u*v,
        D1 = -2*m**3 + (suv+7)*m**2 - 4*(suv+1)*m + puv**2 + 4*(suv-1)
        D2 = 4*m**3 - (4*suv+7)*m**2 + ((suv+4)**2+2*puv-20)*m + (puv-2)**2-2*suv**2
        phi = -puv**2 * D2 / (m-2) / (m+puv-2)**2 / (suv-2*m-1) - m
        final_coeff = D1*D2 / (m-2)**2 / (m+puv-2)**2 / (suv-2*m-1)
        
        If w = (m**3 - (suv+1)*m**2 + (suv+puv-3)*m + (puv-1)**2+2*suv+1) / (m - 2),
        Then we have identity
        R(a,b,c) * s(a^2 + phi*b*c) = s((a-b)^2g(a,b,c)^2)/2 + (phi - (m-2)*(suv-2*m)/(m+puv-2)) * s(ab(a-b)^2h(a,b,c)^2)
            + final_coeff * (puv**2 * s(ab)p(a-b)^2 + (2-m)s(bc(a-b)(a-c)(a-ub)(a-uc)(a-vb)(a-vc)))

        Explicit forms of function g and h are omitted here. Please refer to the code.
            
        In particular, if puv == 0, WLOG v = 0, suv = u. Let m -> 2, w = 5 - 3*u, when u <= 4
        R(a,b,c) * s(a^2-u/4*ab) = s(a(a-b)(a-c)(2a-ub-uc))^2/4 + (1-u/4)s(ab(a-b)^2((a-b)^2+(2-u)(a+b)c-(3-u)c^2)^2)
            + (1-u/4)/2 * s((a(b-c)((b+c-a)(b+c-(u+1)a)-u(a-c)(a-b)))^2)

        References
        -----------
        [1] https://tieba.baidu.com/p/8261574122
        """
        suv, puv = (2 - 2*y)/(x - 1), (2*x + y - 2)/(x - 1)
        a, b, c, m = sp.symbols('a b c m')
        det1 = (-2*m**3 + (suv + 7)*m**2 - 4*(suv + 1)*m + puv**2 + 4*(suv - 1)).as_poly(m)
        def _compute_params(suv, puv, m):
            det1 = -2*m**3 + (suv + 7)*m**2 - 4*(suv + 1)*m + puv**2 + 4*(suv - 1)
            det2 = 4*m**3 - (4*suv + 7)*m**2 + ((suv + 4)**2 + 2*puv - 20)*m + (puv-2)**2 - 2*suv**2
            denom = (m + puv - 2)**2 * (suv - 2*m - 1)
            phi = -puv**2 * det2 / (m-2) / denom - m
            final_coeff = det1 * det2 / (m-2)**2 / denom
            w = (m**3 - (suv + 1)*m**2 + (suv + puv - 3)*m + (puv - 1)**2 + 2*suv + 1) / (m - 2)
            ker_coeff = -(w*(x - 1)**2 + (x**2 - x*y + y**2 - y))
            return suv, puv, m, det1, det2, phi, final_coeff, ker_coeff
        def _validate_params(params):
            suv, puv, m, det1, det2, phi, final_coeff, ker_coeff = params
            return phi >= -1 and phi >= ((m - 2)*(suv - 2*m)/(m + puv - 2))

        if puv == 0 and suv <= 4:
            # degenerated case
            def _solve_degen(x, y, u, ker_coeff):
                extra = x**2 - x*y + y**2 - y - (x - 1)**2 * (3*u - 5) + ker_coeff
                if extra < 0:
                    return None, 2
                func_g = ((a-b)**2 + (2-u)*(a+b)*c - (3-u)*c**2).expand().together()
                func_h = ((b+c-a)*(b+c-(u+1)*a) - u*(a-c)*(a-b)).expand().together()
                r = (x-1)**2
                solution = r/4 * CyclicSum(a*(a-b)*(a-c)*(2*a-u*b-u*c).together())**2 \
                    + r*(1-u/4) * CyclicSum(a*b*(a-b)**2*func_g**2) + r*(1-u/4)/2 * CyclicSum(a**2*(b-c)**2*func_h**2)
                p1 = (a**2 - u/4*b*c).together().as_coeff_Mul()
                solution = solution / (p1[0] * CyclicSum(p1[1])) + extra * CyclicProduct((a-b)**2)
                return solution, 1
            return _solve_degen(x, y, suv, ker_coeff)


        for root in nroots(det1, method = 'factor', real = True):
            if isinstance(root, sp.Rational) and (root == 2 or m + puv == 2 or suv - 2*m - 1 == 0):
                return None, 2
            if not root.is_real:
                continue
            params = _compute_params(suv, puv, root)
            # print(params)
            if _validate_params(params) and params[-1] <= ker_coeff:
                break
        else:
            return None, 2


        def _solve_border(*args, with_tail = True, with_frac = False):
            if len(args) == 3:
                args = _compute_params(*args)
            suv, puv, m, det1, det2, phi, final_coeff, ker_coeff = args
            l1 = (-m**2 + (puv+1)*m - puv*(suv+1)+2)/(m+puv-2)
            l2 = ((1-2*puv)*m**2 + (puv*(suv+1)-1)*m + puv-2)/(m+puv-2)
            l3 = (2*(puv+suv)*m**2 - (puv*(suv-1)+(suv+2)**2-2)*m + (puv-2)**2+2*suv**2)/(m+puv-2)
            l4 = ((2*puv-1)*m - puv*(suv+1)+2)/(m+puv-2)
            g = (a**2-m*a*b+b**2)*(a+b)+l1*c*(a**2+b**2)+l2*c**2*(a+b)+l3*a*b*c+l4*c**3
            h = a**2-m*a*b+b**2+(puv+suv-1-m)*c**2-(suv-m)*c*(a+b)
            g, h = g.expand(), h.expand()
            # print('Phi =', phi, 'final_coeff =', final_coeff, 'w =', _compute_w(suv, puv, m))
            solution = CyclicSum((a-b)**2*g**2) / 2 + (phi - (m-2)*(suv-2*m)/(m+puv-2)) * CyclicSum(a*b*(a-b)**2*h**2)
            if with_tail:
                solution += final_coeff * (puv**2 * CyclicSum(a*b)*CyclicProduct((a-b)**2) +
                    (2-m)* CyclicSum(b*c*(a-b)*(a-c)*(a**2-suv*a*b+puv*b**2)*(a**2-suv*a*c+puv*c**2)))
            if with_frac:
                solution = solution / CyclicSum(a**2 + phi*b*c)
            return solution
        
        if isinstance(params[2], sp.Rational):
            solution = sp.Add(
                (x-1)**2 * _solve_border(*params, with_tail = False, with_frac = True),
                (ker_coeff - params[-1]) * CyclicProduct((a-b)**2)
            )
            return solution, 1

        for ((m1, m2), mul) in det1.intervals():
            if m1 <= params[2] <= m2:
                break

        # When the exact m is irrational, the inequality is a linear combination of m1,m2
        for rounding in (None, 1e-1, 1e-2, 1e-3, 1e-4, 1e-8, 1e-12):
            if rounding is not None:
                m1, m2 = det1.refine_root(m1, m2, eps = rounding)
            if m1 <= 2 <= m2:
                continue
            params1, params2 = _compute_params(suv, puv, m1), _compute_params(suv, puv, m2)
            if not _validate_params(params1) or not _validate_params(params2):
                continue
            final_coeff1, final_coeff2 = params1[-2], params2[-2]
            if (final_coeff1 < 0 and final_coeff2 < 0) or (final_coeff1 > 0 and final_coeff2 > 0):
                continue
            w1, w2 = final_coeff1 * (2-m1), final_coeff2 * (2-m2)
            # The weight of linear combination: w1*a + w2*b = 0 => a:b = -w2:w1
            w1, w2 = (-w2/(w1 - w2), w1/(w1 - w2)) if final_coeff1 != 0 else (1, 0)
            phiw = params1[-3] * w1 + params2[-3] * w2
            ker_coeffw1 = params1[-1] * w1 + params2[-1] * w2
            ker_coeffw2 = params1[-3] * (ker_coeff - params1[-1]) * w1  + params2[-3] * (ker_coeff - params2[-1]) * w2
            final_coeffw = params1[-2] * w1 + params2[-2] * w2
            ker_coeff_tmp = ker_coeffw2 + final_coeffw * puv**2 * (x-1)**2
            if phiw >= -1 and ker_coeff >= ker_coeffw1 and (ker_coeff - ker_coeffw1) + ker_coeff_tmp >= 0:
                kwargs = {'with_tail': False, 'with_frac': False}
                p1 = _solve_border(*params1, **kwargs)
                p2 = _solve_border(*params2, **kwargs)
                solution = w1*(x-1)**2 * p1 + w2*(x-1)**2 * p2
                quad_form = ((ker_coeff - ker_coeffw1) * a**2 + ker_coeff_tmp * b*c).together().as_coeff_Mul()
                solution += quad_form[0] * CyclicSum(quad_form[1]) * CyclicProduct((a-b)**2)
                solution = solution / CyclicSum(a**2 + phiw*b*c)
                return solution, 1
        return None, 2

    @staticmethod
    def solve(coeff0, x, y, ker_coeff, t_coeff, rem_coeff, rem_ratio):
        """
        Let F0 = s(a^6+a^5b+a^5c+a^4bc-2a^3b^2c-2a^3bc^2) and f(a,b,c) = s(xa^2 + yab).
        Define
        F_{x,y}(a,b,c) = F0 - 2s(a^4-a^2bc)f(a,b,c) + s(a^2-ab)f(a,b,c)^2.

        Solve the inequality:
        Poly = F_{x,y} * coeff0 + ker_coeff * p(a-b)^2 + t_coeff * p(a+b-2c)^2 + rem_coeff * s(a^2-ab) * s(a^2 + rem_ratio*a*b)^2 >= 0.

        If the polynomial has a solution for a,b,c on R rather R+, it is returned in prior.
        """
        cls = _sextic_sym_axis
        SOLVERS = [
            # type, func
            # first two do not need to lift the degree
            (0, cls._F_square),
            (0, cls._F_trivial),

            (1, cls._F_regular),
            (1, partial(cls._F_sos2, z_type = 0)),
            (1, partial(cls._F_sos2, z_type = 1)),
            (1, partial(cls._F_alternative, z = cls._F_alternative_find_z(x, y, 0, ker_coeff/coeff0))),
            (1, partial(cls._F_alternative, z = cls._F_alternative_find_z(x, y, 1, ker_coeff/coeff0))),
            (0, cls._F_tighter_bound_border)
        ]
        solutions = []

        for (solver_type, solver) in SOLVERS:
            f = cls._wrap_F(solver_type, solver)
            solution, flg = f(x, y, coeff0, ker_coeff, t_coeff, rem_coeff, rem_ratio)
            # print(solver, solution, flg)
            if flg == 0:
                return solution
            elif flg == 1:
               solutions.append(solution)

        if len(solutions) > 0:
            return solutions[0]


def sos_struct_sextic_symmetric_ultimate(coeff, recurrsion, real = True):
    """
    Solve symmetric sextics.
    
    1. First we assume there exist nontrivial roots. Three cases:
        A. On the border, e.g. (.618, 0, 1), (1.618, 0, 1)
        B. On the symmetric axis, e.g. (0.5, 0.5, 1)
        C. Nontrivial interior, e.g. (.25, .5, 1), (.5, .25, 1)
    
    Case A can subtract some s(a3-abc + x(ab2+a2b-2abc))2 to Case (A+B).
    Case B can subtract some p(a-b)2 to Case (A+B) or Case (B+C).
    Case C can subtract some s(a2-xab)2s(a2-ab) to Case (B+C).

    2. To summarize, we can move to Case (A+B) or Case (B+C).
    For Case (A+B), we subtract s(a3-abc + x(ab2+a2b-2abc))2 to cancel s(a6)
        and call function iran96. 
    Note: a special case is when the border's root is (1,0,1), we shall handle more carefully.
    For Case (B+C), we can multiplicate s(a2+xab)


    Examples
    --------
    Trivial.
    s(5a2-ab)s(a)4-72(p(a2+b2)+11/2p(a2))


    s(a6-a2b2c2)+s(a3b3-a4bc)-12s(a4b2+a4c2-2a2b2c2)+22s(a3b3-a2b2c2)+14s(a2b+ab2-2abc)abc-2p(a-b)2
    
    Case C.
    s(409a6-1293a5b-1293a5c+651a4b2+5331a4bc+651a4c2+818a3b3-5190a3b2c-5190a3bc2+5106a2b2c2)

    s(38a6-148a5b-148a5c+225a4b2+392a4bc+225a4c2-210a3b3-320a3b2c-320a3bc2+266a2b2c2)

    s(414a6-1470a5b-1470a5c+979a4b2+5864a4bc+979a4c2+644a3b3-5584a3b2c-5584a3bc2+5228a2b2c2)
    """

    coeff6 = coeff((6,0,0))
    if not (coeff((5,1,0)) == coeff((1,5,0)) and coeff((4,2,0)) == coeff((2,4,0)) and coeff((3,2,1)) == coeff((2,3,1))):
        # asymmetric
        return None
    elif coeff6 == 0:
        # degenerated
        return _sos_struct_sextic_iran96(coeff, real = real)
    elif coeff6 < 0:
        return None

    if coeff((5,1,0)) == 0 and coeff((4,2,0)) == 0 and coeff((3,2,1)) == 0 and coeff6 != 0:
        return _sos_struct_sextic_tree(coeff)

    x0, x1, x2, x3, x4, x5 = radsimp([coeff(_) for _ in [(6,0,0),(5,1,0),(4,2,0),(3,3,0),(4,1,1),(3,2,1)]])
    rem = radsimp(3*(x0 + x3 + x4) + 6*(x1 + x2 + x5) + coeff((2,2,2)))

    poly = None

    # try trivial cases
    if True:
        # write in the form of 
        # s(a2-ab)s(m(a^4-a^2bc)+p(a^3b+ab^3-2a^2bc)+n(a^2b^2-a^2bc) + ua^2bc) + vp(a)s(a(b-c)^2) + wp(a-b)^2
        if rem != 0:
            # do not try
            return None

        m = x0
        p = x0 + x1
        n = -2*x0 + 2*x2 + x3
        u = 8*x0 + 6*x1 - x3 + x4
        v = -6*x0 - 2*x1 + 4*x2 + 3*x3 + x5
        w = 2*x0 + x1 - x2 - x3

        if 3*u + 2*v < 0:
            # this implies that the value on the symmetric axis is negative around (1,1,1)
            return None

        if v == 0:
            if w == 0:
                # is a multiple of s(a^2-ab) -> degenerates to quartic
                poly_div_quad = (
                    m * (a**4 + b**4 + c**4) +
                    p * (a**3*(b+c) + b**3*(c+a) + c**3*(a+b)) +
                    n * (a**2*b**2 + b**2*c**2 + c**2*a**2) +
                    (u - m - 2*p - n) * (a**2*b*c + b**2*c*a + c**2*a*b)
                ).as_poly(a,b,c) 
                solution = sos_struct_quartic(Coeff(poly_div_quad), recurrsion)
                if solution is not None:
                    solution = sp.Rational(1,2) * CyclicSum((a-b)**2) * solution
                    return solution
                return None

            if True:
                # this case will lead to failure in _sos_struct_sextic_symmetric_quadratic_form
                t_ = radsimp(p / (-2*m))
                n_ = n - m * (t_**2 + 2)
                u_ = u - 3*m*(1 - t_)**2
                y = radsimp([
                    m / 2,
                    n_ / 4 - u_ / 12,
                    u_ / 6,
                    w + 3 * n_ / 4 - u_ / 4,
                ])
                if all(_ >= 0 for _ in y):
                    exprs = [
                        CyclicSum((a-b)**2) * CyclicSum(a*(a - t_*b))**2,
                        CyclicSum(a*(b-c)**2)**2,
                        CyclicSum((a-b)**2) * CyclicSum(a*b)**2,
                        CyclicProduct((a-b)**2)
                    ]
                    return sum_y_exprs(y, exprs)


        # try neat cases
        # note that this might also handle cases for real numbers
        poly = coeff.as_poly()
        if coeff.is_rational:
            try:
                solution = _sos_struct_sextic_symmetric_quadratic_form(poly, coeff)
                if solution is not None:
                    return solution
            except:
                pass

        if m + p >= 0:
            # s(a2-ab)s(a2(a-b)(a-c)) = s(a(a-b)(a-c))^2 + 3p(a-b)^2
            # s(a2-ab)s(ab(a-b)2) = s(ab(a-b)^4) + p(a-b)^2
            # s(a2-ab)s(a2b2-a2bc) = s(ab(a-c)^2(b-c)^2) + p(a-b)^2
            if u >= 0 and u + v >= 0:
                y = radsimp([
                    m,
                    m + p,
                    n + 2*(m + p),
                    1,
                    3*m + 3*(m + p) + n + w
                ])
                if all(_ >= 0 for _ in y):
                    p1 = u*CyclicSum(a*(a-b)*(a-c)) + (u+v)*CyclicSum(a*(b-c)**2)
                    p1 = p1.together().as_coeff_Mul()
                    y[-2] = p1[0]
                    exprs = [
                        CyclicSum(a*(a-b)*(a-c))**2,
                        CyclicSum(a*b*(a-b)**4),
                        CyclicSum(a*b*(a-c)**2*(b-c)**2),
                        CyclicProduct(a) * p1[1],
                        CyclicProduct((a-b)**2)
                    ]
                    return sum_y_exprs(y, exprs)

    # roots detection
    u, v, w, x, z = sp.symbols('u v w x z')

    # Detect Roots
    roots = [[], [], []]

    # Case A. border
    # rather record the true pair of roots (x and 1/x), we compute x + 1/x to avoid radicals
    eq = sp.polys.Poly([x0, x1, x2 - 3*x0, x3 - 2*x1], x) # this shall be the equation of x + 1/x.
    eqdiff = eq.diff(x)
    eq_gcd = sp.polys.gcd(eq, eqdiff)
    if 0 < eq_gcd.degree() <= 3:
        for r in sp.polys.roots(eq_gcd, cubics = False):
            if r.is_real and r >= 2:
                roots[0].append(r)

    # Case B. symmetric axis
    eq = poly.subs({b:1, c:1}).as_poly(a).div(sp.polys.Poly([1,-2,1], a))
    if not eq[1].is_zero:
        return None
    eq = eq[0]
    eqdiff = eq.diff(a)
    eq_gcd = sp.polys.gcd(eq, eqdiff)
    if 0 < eq_gcd.degree() <= 2:
        for r in sp.polys.roots(eq_gcd, cubics = False):
            if r.is_real and r >= 0:
                roots[1].append(r)
    elif eq_gcd.degree() == 3 and eq.degree() == 4:
        # this means that there exists roots with multiplicity 4
        eq_gcd = eq.div(eq_gcd)[0]
        r = radsimp(-eq_gcd.coeff_monomial((0,)) / eq_gcd.coeff_monomial((1,)))
        if r >= 0:
            roots[1].append(r)
            roots[1].append(r)

    # Case C.
    # TO BE IMPLEMENTED
  
    # print('Roots Info = ', roots)
    sum_of_roots = sum((len(_) > 0) for _ in roots)

    if sum_of_roots == 1:
        return _sos_struct_sextic_symmetric_ultimate_1root(coeff, poly, recurrsion, roots, real = real)
    elif sum_of_roots == 2:
        return _sos_struct_sextic_symmetric_ultimate_2roots(coeff, poly, recurrsion, roots)

    return None

def _sos_struct_sextic_symmetric_ultimate_1root(coeff, poly, recurrsion, roots, real = True):
    """
    Examples
    -------
    Case A.
        s(a2)3-27(abc)2-27p((a-b)2)
        
        s(a2/3)3-a2b2c2-p(a-b)2

        s(4a6-a3b3-3a2b2c2)-63p(a-b)2

        4s(a4(a-b)(a-c))+s(a(a-b)(a-c))2

        3s(a/3)6-s(ab)s(a/3)4-(69+11sqrt(33))/648p(a-b)2

    Case B.
        s((b2+c2+5bc-a2/2)(b-c)2(b+c-4a)2)

        s(a)/3s(a5)+(21+9sqrt(5))/2abc(abc-s(a)3/27)-abcs(a)3/9

    Reference
    -------
    [1] https://artofproblemsolving.com/community/c6t29440f6h3147050_zhihu_and_kuing
    """
    coeff6 = coeff((6,0,0))
    if len(roots[0]): 
        # border
        # be careful that we use r + 1/r == roots[0][0]
        if len(roots[0]) == 1 and roots[0][0] != 2:
            # Case A.
            # subtract some s(a3-abc-x(a2b+ab2-2abc))2
            x_ = roots[0][0] - 1
            # if not isinstance(x_, sp.Rational):
            #     return None

            # 1. try subtracting all the s(a6)
            # e.g. s(a2/3)3-a2b2c2-p(a-b)2
            if coeff((5,1,0)) >= -2 * x_:
                poly2 = poly - ((a**3+b**3+c**3-3*a*b*c-x_*(a*a*(b+c)+b*b*(c+a)+c*c*(a+b)-6*a*b*c))**2).as_poly(a,b,c) * coeff6
                solution = _sos_struct_sextic_iran96(Coeff(poly2, is_rational = coeff.is_rational), real = real)
                if solution is not None:
                    if x_ == sp.Rational(3,2):
                        solution += coeff6 / 4 * CyclicProduct((a+b-2*c)**2)
                    elif x_ == 1:
                        solution += coeff6 * CyclicSum(a*(a-b)*(a-c))**2
                    else:
                        solution += coeff6 * CyclicSum(a**3-x_*a**2*(b+c)+(2*x_-1)*a*b*c)**2

                    return solution

            # until the symmetric axis is touched
            # # the subtractor = (a-1)^4 * (2(x-1)a - 1)^2 on the symmetric axis a == b and c == 1
            # sym = poly.subs(c,1).subs(b,a).factor() / (a - 1)**4 / (2*(x-1)*a - 1)**2

    elif len(roots[1]):
        # symmetric axis
        if len(roots[1]) == 1 and roots[1][0] != 0:
            x_ = roots[1][0] + 1
        else:
            return None
        if coeff.is_rational and not isinstance(x_, sp.Rational):
            return None
        # try SOS theorem
        x0, x1, x2, x3 = [coeff((6-i, i, 0)) for i in range(4)]

        denom = radsimp(1 / (2*x_**4 - 4*x_**2 + 2))
        z0 = x0/2
        z1 = (2*x0*x_**5 - 4*x0*x_**3 - 2*x0*x_ + 2*x1*x_**4 - 6*x1*x_**2 - 4*x2*x_ - x3*x_**2 - x3) * denom
        z2 = (4*x0*x_ + 2*x1*x_**2 + 2*x1 + 4*x2*x_ + x3*x_**2 + x3) * denom
        z3 = (-x0*x_**4 + 4*x0*x_**2 + x0 + 4*x1*x_ + 2*x2*x_**2 + 2*x2 + 2*x3*x_) * denom

        z0, z1, z2, z3 = radsimp([z0, z1, z2, z3])

        # Then the polynomial can be written in the form of
        # s((z0(a^2+b^2) + z1ab + z2c(a+b) + z3c^2)(a-b)^2(a+b-xc)^2).

        # First try sum-of-square for real numbers if available.
        # Suppose 2F(a,b,c) = 2\sum f(a,b,c)^2 
        #   = \sum (f(a,b,c)^2 + f(b,a,c)^2)
        #   = 2/3*(\sum f)^2 + 1/3*\sum (f(a,b,c) - f(b,c,a))^2 + 1/3*\sum (f(b,a,c) - f(a,c,b))^2
        #   = 2/3*(\sum f)^2 + 1/3*\sum (f(a,b,c)+f(b,a,c)-f(a,c,b)-f(b,c,a))^2
        #          + 1/3*\sum (f(a,b,c)-f(b,a,c)+f(a,c,b)-f(b,c,a))^2 
      
        # 1. Here the leading term \sum f has only two degrees of freedom:
        # \sum f ~ p(a-b) or s((b-c)^2(b+c-xa))

        # 2. The second term is symmetric w.r.t. a,b,c and also covers equalities at (x-1,1,1) 
        # and its permutations. Also, s((a-b) * cubic) == 0 by assumption. The form should be
        # s((a3+b3-2c3+u(a2b+ab2-a2c-b2c)+v(a2b+ab2-ac2-bc2))2) where u = -vx - x^2 + x - 1.

        # 3. The last can be divided by (a-b)^2. So it is \sum (a-b)^2 * quadratic^2.
        # The quadratic polynomial must be symmetric w.r.t. a,b and also cover equalities at
        # (x-1,1,1). Also, s((a-b) * quadratic) == 0 by assumption. The form should be
        # s((a-b)2(a2+b2-c(a+b)+ucs(a)+vs(ab))2) where u = (-2*v*x + v - x**2 + 3*x - 2)/(x + 1).

        # Note commonly-used identities:
        # s((a-c)(b-c)(a-b)^2(a+b-xc)^2) = (x+1)^2 * p(a-b)^2
        # print('(z0, z1, z2, z3) =', (z0, z1, z2, z3))
        p1 = None
        if z3 > 0 or (z2 == 0 and z3 == 0):
            if z3 > 0:
                ratio = radsimp(-z2 / (2*z3))
                r1 = radsimp(z0 - z2**2/4/z3)
                r2 = z1 + 2*r1 - 2*(z0 - r1)
            else:
                r1, r2 = z0, z1 + 2*z0
            if r1 >= 0 and r2 >= 0:
                p1 = z3*(c-ratio*a-ratio*b)**2 if z3 > 0 else sp.S(0)
                if r2 > 4*r1:
                    p1 += r1*(a-b)**2 + r2*a*b
                else:
                    p1 += (r1 - r2/4) * (a-b)**2 + r2/4 * (a+b)**2
        elif 2*z0 + z1 >= 0 and z2 >= 0 and z3 >= 0:
            p1 = z0*(a-b)**2 + (2*z0+z1)*a*b + z2*c*(a+b) + z3*c**2

        if p1 is not None:
            p1 = p1.together().as_coeff_Mul()
            return p1[0] * CyclicSum(p1[1] * (a-b)**2 * (a+b-x_*c)**2)

        if 2*z0 + z3 >= 0 and 2*z0 + z3 + z1 + 2*z2 >= 0:
            # Apply SOS theorem
            quartic = [
                ((4, 0, 0), z0**2 + 2*z0*z3),
                ((3, 1, 0), z0*z1 + 3*z0*z2 + z1*z3 + z2*z3),
                ((1, 3, 0), z0*z1 + 3*z0*z2 + z1*z3 + z2*z3),
                ((2, 2, 0), 3*z0**2 + 2*z0*z3 + 2*z1*z2 + z2**2 + z3**2),
                ((2, 1, 1), 2*z0*z1 + 2*z0*z2 + z1**2 + 2*z1*z2 + 3*z2**2 + 2*z2*z3)
            ]
            quartic = [(_[0], radsimp(_[1])) for _ in quartic]
            is_rational = all(isinstance(_[1], sp.Rational) for _ in quartic)

            quartic_solution = sos_struct_quartic(Coeff(dict(quartic), is_rational = is_rational), None)
            if quartic_solution is not None:
                p0 = (2*(2*z0 + z3)*a**2 + 2*(z1 + 2*z2)*b*c).together().as_coeff_Mul()
                multiplier = p0[0] * CyclicSum(p0[1])
                p1 = quartic_solution * CyclicSum((a-b)**2*(a+b-x_*c)**2)
                func = lambda a,b,c: (z0*(a**2+b**2) + z1*a*b + z2*c*(a+b) + z3*c**2)*(a-b)*(a+b-x_*c)
                p2 = CyclicSum((func(b,c,a) - func(c,a,b)).expand()**2)
                return (p1 + p2) / multiplier


    return None


def _sos_struct_sextic_symmetric_ultimate_2roots(coeff, poly, recurrsion, roots):
    """

    Examples
    --------
    Case (A+B)

    s((a-b-c)4a-abc(3a-b-c)2)s(a)-(s(ab(a2-b2+3(ab-ac)+3(bc-ab))2)-4p(a-b)2)

    s(4a6-6(a5b+a5c)-12(a4b2+a4c2)+37a4bc+28a3b3-31(a3b2c+a3bc2)+29a2b2c2)

    s(a4(a-b)(a-c)) - 5p(a-b)2

    
    Reference
    -------
    [1] Vasile, Mathematical Inequalities Volume 1 - Symmetric Polynomial Inequalities. 3.78
    
    [2] https://artofproblemsolving.com/community/c6t243f6h3013463_symmetric_inequality

    [3] https://tieba.baidu.com/p/8261574122
    """
    coeff6 = coeff((6,0,0))

    
    if len(roots[2]) == 0:
        diffpoly = None
        # Case (A + B)
        if not (len(roots[0]) == 1 and roots[0][0] == 2):
            # just try
            x_ = roots[0][0] - 1
            if not isinstance(x_, sp.Rational):
                return None
            solution = CyclicSum(a**3-x_*a**2*(b+c)+(2*x_-1)*a*b*c)**2
            diffpoly = solution.doit().as_poly(a,b,c)
            solution *= coeff6

        elif 1 <= len(roots[1]) <= 2: # roots[0][0] == 2:
            # find the nontrivial root (other than 0) on the symmetric axis
            if len(roots[1]) == 2 and (roots[1][0] == 0 or roots[1][1] == 0):
                x = roots[1][0] + roots[1][1] # note that one of the roots is 0
            elif len(roots[1]) == 1:
                x = roots[1][0]
            if x > 4:
                return None

            # Theorem: when x <= 4
            # f(a,b,c) = s(a6-(x+1)(a5b+a5c)+(4x-5)(a4b2+a4c2)+(x2-4x+11)a4bc
            #               -2(3x-5)a3b3+(-x2+5x-10)(a3b2c+a3bc2)+(x2-6x+10)a2b2c2)
            #          = s(a2(a-b)(a-c)(a-xb)(a-xc)) + (3x-5)p(a-b)^2 >= 0
            # Proof: when 1 <= x <= 4,
            # we have f(a,b,c)s(a2+(x-2)ab) = (4-x)(x-1)/2s(a2(b-c)2((a-b-c)2-xbc)2)
            #               + s(a2-ab)((2x-x2)abc-p(a+b-c))2
            #        when x < 1,
            # we have f(a,b,c)s(a2b2-a2bc) = 1/2abcs(a(a-b)(a-c))s((b-c)2(b+c-(x+1)a)2)
            #               + p(a-b)2(s(a2(a-b)(a-c)) + (2-x)s(ab(a-b)2) + 6(1-x)s(a2bc))

            get_diffpoly = lambda x: (a**6+(-x-1)*a**5*b+(-x-1)*a**5*c+(4*x-5)*a**4*b**2+(x**2-4*x+11)*a**4*b*c\
                +(4*x-5)*a**4*c**2+(10-6*x)*a**3*b**3+(-x**2+5*x-10)*a**3*b**2*c+(-x**2+5*x-10)*a**3*b*c**2\
                +(10-6*x)*a**3*c**3+(4*x-5)*a**2*b**4+(-x**2+5*x-10)*a**2*b**3*c+(3*x**2-18*x+30)*a**2*b**2*c**2\
                +(-x**2+5*x-10)*a**2*b*c**3+(4*x-5)*a**2*c**4+(-x-1)*a*b**5+(x**2-4*x+11)*a*b**4*c+(-x**2+5*x-10)*a*b**3*c**2\
                +(-x**2+5*x-10)*a*b**2*c**3+(x**2-4*x+11)*a*b*c**4+(-x-1)*a*c**5+b**6+(-x-1)*b**5*c+(4*x-5)*b**4*c**2\
                +(10-6*x)*b**3*c**3+(4*x-5)*b**2*c**4+(-x-1)*b*c**5+c**6).as_poly(a,b,c)

            if x == 4:
                # easy case, no need to higher the degree
                solution = coeff6 / 2 * CyclicSum((a-b)**2) * CyclicSum(a**2-2*b*c)**2
                diffpoly = solution.doit().as_poly(a,b,c)
            elif x == 1:
                solution = coeff6 * CyclicSum(a*(a-b)*(a-c)) ** 2
                diffpoly = solution.doit().as_poly(a,b,c)
            elif x > 1:
                diffpoly = get_diffpoly(x)
                multiplier = CyclicSum(a**2 + (x-2)*b*c)
                solution = (4-x)*(x-1) / 2 * coeff6 * CyclicSum(a**2*(b-c)**2*(a**2+b**2+c**2-2*a*b-2*a*c+(2-x)*b*c)**2) \
                    + coeff6 / 2 * CyclicSum((b-c)**2) * (((2*x-x**2)*a*b*c-CyclicProduct(a+b-c)))**2
                solution = solution / multiplier
            elif x < 1:
                if x <= 0:
                    # x < 0 is no stronger than x == 0
                    # x == 0 corresponds to the case s(a4(a-b)(a-c)) - 5p(a-b)2
                    diffpoly = ((a**4*(a-b)*(a-c)+b**4*(b-c)*(b-a)+c**4*(c-a)*(c-b)) \
                                    - 5 * ((a-b)*(b-c)*(c-a))**2).as_poly(a,b,c)
                    x = 0
                else:
                    diffpoly = get_diffpoly(x)

                multiplier = CyclicSum(a**2*(b-c)**2)
                pp = coeff6*CyclicSum((a-b)**2*(a+b-c)**2) + 2*(2 - x)*coeff6*CyclicSum(a*b*(a-b)**2) + 12*(1-x)*coeff6*CyclicSum(a**2*b*c)
                pp = sp.together(pp).as_coeff_Mul()
                y = [coeff6, pp[0]]
                exprs = [
                    CyclicSum(a*(a-b)*(a-c)) * CyclicProduct(a) * CyclicSum((b-c)**2*(b+c-(x+1)*a)**2),
                    CyclicProduct((a-b)**2) * pp[1]
                ]
                solution = sum_y_exprs(y, exprs) / multiplier


        if diffpoly is not None:
            new_poly = poly - coeff6 * diffpoly
            rest_solution = _sos_struct_sextic_iran96(Coeff(new_poly, is_rational = coeff.is_rational))
            if rest_solution is not None:
                f1, f2 = sp.fraction(sp.together(solution + rest_solution))
                f1 = sp.collect(f1, CyclicProduct((a-b)**2))
                return f1 / f2

    elif roots[0] is None:
        # Case (B + C)
        pass

    return None