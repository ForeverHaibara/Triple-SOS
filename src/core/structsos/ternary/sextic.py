import sympy as sp

from .cubic import sos_struct_cubic
from .sextic_symmetric import (
    _sos_struct_sextic_hexagon_symmetric,
    _sos_struct_sextic_hexagram_symmetric,
    sos_struct_sextic_symmetric_ultimate
)
from .utils import (
    CyclicSum, CyclicProduct, Coeff,
    sum_y_exprs, nroots, rationalize_bound, rationalize_func, radsimp,
    quadratic_weighting, reflect_expression, inverse_substitution, congruence,
    zip_longest
)

a, b, c = sp.symbols('a b c')

def sos_struct_sextic(coeff, recurrsion, real = True):
    if coeff((5,1,0)) == coeff((1,5,0)) and coeff((4,2,0)) == coeff((2,4,0)) and coeff((3,2,1)) == coeff((3,1,2)):
        return sos_struct_sextic_symmetric_ultimate(coeff, recurrsion, real = real)

    if coeff((6,0,0)) == 0 and coeff((5,1,0)) == 0 and coeff((5,0,1)) == 0:
        return _sos_struct_sextic_hexagon(coeff, recurrsion, real = real)

    if coeff((6,0,0)) != 0:
        return _sos_struct_sextic_full_sdp(coeff)

    return None


def _sos_struct_sextic_hexagram(coeff):
    """
    Solve s(a3b3+xa4bc+ya3b2c+za2b3c+wa2b2c2) >= 0. The structure is known as hexagram.
    Typically we multiply s(a) to solve it.

    Theorem 1:
    F(a,b,c) = s(c((uv-1)(a^2c-b^2c)-(u^2+v)(a^2b-abc)+(v^2+u)(ab^2-abc))^2)
        + 1/2*(u^2-uv+v^2+u+v+1)abcs((a^2-b^2+u(ab-ac)+v(bc-ab))^2) >= 0
    And F(a,b,c) is a multiple of s(a).

    Theorem 2:
    When k >= max_rootof(16*k**3 + 567*k**2 - 3402*k - 18225) \approx 8.10864,
    s(a2c)s(b2c) + ka2b2c2 >= ws(a2c)abc

    Examples
    -------
    s(a3b3+7a4bc-29a3b2c+12a3bc2+9a2b2c2)

    9s(a3b3)+4s(a4bc)-11abcp(a-b)-37s(a3b2c)+72a2b2c2

    s(11a4bc+11a3b3+153a3bc2-153a3b2c-22a2b2c2)

    s(bc(a2-bc+2(ab-ac)+3(bc-ab))2)

    s(ab(ab-4ac+5bc-2c2)2)

    s(3a4bc+2a3b3+a3bc2-6a2b2c2)

    (s(a2c)s(b2c)+25/2p(a2)-27/4s(a2c)p(a))

    s(a2c)s(b2c)+9p(a2)-15*4^(-2/3)p(a)s(a2b)

    (s(a2c)s(b2c)+(5sqrt(5)-2)p(a2)-6s(a2c)p(a))

    (s(b2c)s(a2c)+25/3p(b2)-4*3^(1/3)s(b2c)p(b))

    (s(a2c)s(b2c)+200/9p(a2)-19*18^(1/3)/6s(a2c)p(a))

    Reference
    ----------
    [1] https://www.zhihu.com/question/619911891
    """
    if coeff((3,3,0)) < 0 or coeff((4,1,1)) < 0:
        return None
    if coeff((3,3,0)) == 0:
        # degenerates to cubic
        new_coeffs_ = {(3,0,0): coeff((4,1,1)), (2,1,0): coeff((3,2,1)), (1,2,0): coeff((2,3,1)), (1,1,1): coeff((2,2,2))}
        solution = sos_struct_cubic(Coeff(new_coeffs_), None, None)
        if solution is not None:
            return solution * CyclicProduct(a)
        return None
    if coeff((4,1,1)) == 0:
        # degenerates to inverse cubic
        new_coeffs_ = {(3,0,0): coeff((3,3,0)), (2,1,0): coeff((2,3,1)), (1,2,0): coeff((3,2,1)), (1,1,1): coeff((2,2,2))}
        solution = sos_struct_cubic(Coeff(new_coeffs_), None, None)
        if solution is not None:
            solution = inverse_substitution(solution, factor_degree = 0)
        return solution


    if coeff((3,2,1)) == coeff((2,3,1)):
        # call symmetric solution in priority
        solution = _sos_struct_sextic_hexagram_symmetric(coeff)
        if solution is not None:
            return solution
    
    if True:
        # First try trivial cases.
        # Idea: Observe s(ab(ab+vc^2-(1+v)/2c(a+b) + xc(a-b))^2) >= 0
        # Particularly, when x = 0 the inequality is symmetric. We determine x by the asymmetric part.
        c1, c2 = coeff((3,3,0)), coeff((4,1,1))
        diff = (coeff((3,2,1)) - coeff((2,3,1))) / c1

        def _compute_x_rest(v):
            # after normalizing coeff((3,3,0)) == 1, we subtract
            # s(ab(ab+vc^2-(1+v)/2c(a+b) + xc(a-b))^2) + (c2/c1 - v^2)p(a)s(a(a-b)(a-c))
            # choose x such that the remaining part coeff((3,2,1)) == coeff((2,3,1)) are equal, denoted by 'rest'
            x = diff / (6*(v+1))
            rest = coeff((3,2,1)) / c1 - (x**2 + 3*(v+1)*x - 3*(v+1)**2/4) + (c2/c1 - v**2)
            return x, rest
        def _check_valid(v):
            if not (v >= 0 and c1*v**2 <= c2):
                return False
            x, rest = _compute_x_rest(v)
            if rest >= 0:
                return True
            return False

        eq = sp.Poly.from_list([c1, 0, -c2], sp.Symbol('u'))
        v = rationalize_func(eq, _check_valid, validation_initial = lambda x: x >= 0, direction = -1)
            
        if v is not None:
            x, rest = _compute_x_rest(v)
            p1 = ((c2 - c1*v**2) * CyclicSum(a*(a-b)*(a-c)) + (rest*c1) * CyclicSum(a*(b-c)**2)).together().as_coeff_Mul()
            y = [
                c1,
                p1[0],
                sum(coeff(i) for i in [(3,3,0),(4,1,1),(3,2,1),(2,3,1)])*3 + coeff((2,2,2))
            ]
            if y[-1] < 0:
                return None
            exprs = [
                CyclicSum(a*b*(a*b+v*c**2-(1+v)/2*a*c-(1+v)/2*b*c+x*a*c-x*b*c)**2),
                CyclicProduct(a) * p1[1],
                CyclicProduct(a**2)
            ]
            return sum_y_exprs(y, exprs)


    rem = sum(coeff(i) for i in [(3,3,0),(4,1,1),(3,2,1),(2,3,1)]) * 3 + coeff((2,2,2))
    if rem > 0 and coeff((4,1,1)) == coeff((3,3,0)):
        # Sometimes we have this type of problem, corresponding theorem 2.
        # Simplest idea: s(ab(ab+c2+uac+vbc)2) >= 0.
        x_ = (coeff((3,2,1)) / coeff((3,3,0)))
        y_ = (coeff((3,1,2)) / coeff((3,3,0)))
        # 4u+v^2 = x => u = (x-v^2)/4
        t, u_, v_ = sp.symbols('t'), None, None
        center0 = radsimp(coeff((2,2,2)) / coeff((3,3,0)))
        rem0 = radsimp(rem / coeff((3,3,0)))

        # let t = u - v, keep coeff((3,2,1)) == coeff((3,1,2)) after subtraction, make eqt <= 0
        eqt = (t**4 + 48*t**2 + 32*t*y_ + (-2*t**2 - 16*t)*(x_ + y_) + (x_ - y_)**2).as_poly(t)
        if not coeff.is_rational:
            # first check the exact solution
            # eqv = (4*t + (x_ - t**2)**2/16 - y_).as_poly(t)  # here we borrow the symbol t but it is actually v
            eqv2 = (6*(x_-t**2)/4*t + 6 - center0).as_poly(t)
            # eq_gcd = sp.gcd(eqv, eqv2)

            # NOTE: direct gcd does not work because there may exist root that does not lie in the domain of eqv.
            # Instead we compute the resultant directly.

            z_ = center0
            det = radsimp(-20736*x_**3 + 1296*x_**2*y_**2 - 72*x_*y_*(z_**2 - 60*z_ - 4284) - 20736*y_**3 + (z_ - 102)**3*(z_ - 6))
            # det = sp.polys.resultant(sp.polys.resultant(4*u+v**2-x,4*v+u**2-y,u), 6*(x-v**2)/4*v+6-z, v).factor()
            if det == 0:
                # v is computed by symbolic gcd
                v_ = radsimp(24*(6*x_**2 + y_*z_ - 102*y_)/(36*x_*y_ - z_**2 + 204*z_ - 10404))
                u_ = radsimp((x_ - v_**2)/4)
                t_ = radsimp(u_ - v_)

        if v_ is None:
            def _compute_uv(t):
                v_ = radsimp((t*(4 - t) - x_ + y_)/(2*t))
                return t + v_, v_

            def _is_valid_t(t):
                u, v = _compute_uv(t)
                return (u + v + 2)**2 * 3 <= rem0

            t_ = rationalize_func(eqt, _is_valid_t, direction = -1)
            if t_ is not None:
                u_, v_ = _compute_uv(t_)

        if v_ is not None:
            y = radsimp([
                coeff((3,3,0)),
                - eqt(t_) / 4 / t_**2 * coeff((3,3,0)),
                coeff((3,3,0)) * (rem0 - (u_ + v_ + 2)**2 * 3),
            ])
            exprs = [
                CyclicSum(a*b*(a*b + c**2 + u_*a*c + v_*b*c)**2),
                CyclicProduct(a) * CyclicSum(a*(b-c)**2),
                CyclicProduct(a**2)
            ]
            return sum_y_exprs(y, exprs)

        # Theorem 2.
        if coeff((3,2,1)) == 0 or coeff((3,1,2)) == 0:
            reflect = False if coeff((3,2,1)) == 0 else True
            coeff33 = coeff((3,3,0))
            x_ = radsimp(coeff((2,2,2)) / coeff33 - 3)
            y_ = radsimp(-(coeff((3,2,1)) + coeff((3,1,2))) / coeff33)
            y3 = radsimp(y_**3)
            y3_bound = radsimp(27*x_**2/32 + 135*x_/8 - 27*(x_ - 8)*sp.sqrtdenest(sp.sqrt(radsimp(x_**2 - 8*x_)))/32 - sp.S(27)/4)
            if y3 == y3_bound: # y3 <= y3_bound
                y3, y3_copy, y_, y_copy = y3_bound, y3, y3_bound**sp.Rational(1,3), y_

                r = (81*x_**5 + 2025*x_**4 - 21*x_**3*y3 + 17658*x_**3 - 2406*x_**2*y3 + 62370*x_**2 - 16*x_*y_**6 - 10773*x_*y3 + 79461*x_ + 864*y_**6 - 18468*y3 + 32805)\
                    /(y_*(27*x_**5 + 891*x_**4 - 16*x_**3*y3 + 9558*x_**3 - 1112*x_**2*y3 + 41742*x_**2 - 6336*x_*y3 + 78975*x_ + 384*y_**6 - 10044*y3 + 45927))
                r = radsimp(r)
                z0 = radsimp(r**2*(x_ - 3*y_ + 9)/(4*(r - 1)**4))
                z = min(sp.S(1), z0)

                solution = radsimp(z / r**2) * CyclicSum(c*(-a + b*r)**2*(a*(b+c)*r - a*b - b*c)**2)
                quartic_part = sp.S(0)
                if z != 1:
                    quartic_part = radsimp(1 / r**2 / 2) * CyclicSum((-a + b*r)**2*(a*r + b - c*(r+1))**2)
                    solution += radsimp((1 - z) / r**2) * CyclicSum(c*(-a + b*r)**2*(a*(b-c)*r + a*b - b*c)**2)
                else:
                    uncentered = radsimp(4 * (z0 - z) / 3)
                    p1 = (r*a**2 - radsimp(r**2-r+1)*b*c).together().as_coeff_Mul()
                    if uncentered <= 1:
                        quartic_part = radsimp((1 - uncentered) / r**2 / 2) * CyclicSum((-a + b*r)**2*(a*r + b - c*(r+1))**2) \
                            + radsimp(uncentered / r**2 * p1[0]**2) * (CyclicSum(p1[1]))**2
                    else:
                        solution = None

                if solution is not None:
                    quartic_part += (y_ - y_copy) * CyclicSum(a*b**2) * CyclicSum(a)
                    solution = (solution + CyclicProduct(a) * quartic_part) / CyclicSum(a)
                    if reflect:
                        solution = reflect_expression(solution)
                    return coeff33 * solution



    # hexagram (star)
    if coeff((4,1,1)) > 0:
        # first we solve the exact (numerical) solution (u, v) by the following two equations:
        # (u**4 + u**3 - 3*u**2*v - 2*u*v**3 + 3*u*v**2 + 2*u - 2*v**3 - 3*v - 1)/(u*v - 1)**2  = coeff((3,2,1)) / coeff((4,1,1))
        # (-2*u**3*v - 2*u**3 + 3*u**2*v - 3*u*v**2 - 3*u + v**4 + v**3 + 2*v - 1)/(u*v - 1)**2 = coeff((3,1,2)) / coeff((4,1,1))

        x_ = (coeff((3,2,1)) / coeff((3,3,0))).n(20)
        y_ = (coeff((3,1,2)) / coeff((3,3,0))).n(20)
        coeffs = [
            4*x_**3 - x_**2*y_**2 - 18*x_*y_ + 4*y_**3 + 27,
            4*(2*x_**3 + 6*x_**2*y_ + 18*x_**2 - x_*y_**3 - 3*x_*y_**2 + 9*x_*y_ - 4*y_**3 - 9*y_**2 - 27*y_ - 27),
            4*x_**3 + 30*x_**2*y_ + 9*x_**2 + 75*x_*y_**2 + 207*x_*y_ + 189*x_ - 4*y_**4 + 25*y_**3 + 171*y_**2 + 378*y_ + 270,
            -16*x_**3 + 4*x_**2*y_**2 + 12*x_**2*y_ + 18*x_**2 - 21*x_*y_**2 - 198*x_*y_ - 297*x_ + 5*y_**3 - 180*y_**2 - 594*y_ - 432,
            -32*x_**3 - 72*x_**2*y_ + 72*x_**2 + 10*x_*y_**3 + 24*x_*y_**2 + 423*x_*y_ + 459*x_ + 7*y_**3 + 378*y_**2 + 783*y_ + 513,
            -16*x_**3 - 72*x_**2*y_ + 36*x_**2 - 123*x_*y_**2 - 234*x_*y_ - 324*x_ + 4*y_**4 - 52*y_**3 - 162*y_**2 - 567*y_ - 432,
            16*x_**3 - 4*x_**2*y_**2 - 24*x_**2*y_ + 72*x_**2 - 3*x_*y_**2 + 198*x_*y_ + 189*x_ - 17*y_**3 + 198*y_**2 + 378*y_ + 270,
            32*x_**3 + 48*x_**2*y_ + 108*x_**2 - 4*x_*y_**3 + 18*x_*y_**2 + 90*x_*y_ - 27*x_ - 13*y_**3 + 36*y_**2 - 81*y_ - 108,
            16*x_**3 + 24*x_**2*y_ + 36*x_*y_**2 + 18*x_*y_ - 27*x_ - y_**4 + 13*y_**3 + 45*y_**2 + 27*y_ + 27
        ]
        u, v = sp.symbols('u v')
        eqv = sum(coeff * v**i for coeff, i in zip(coeffs, range(8, -1, -1))).as_poly(v)

        coeffsu1 = [
            4*x_**2 - x_*y_**2 - 3*y_,
            8*x_**2 + 16*x_*y_ + 36*x_ - 2*y_**3 - 3*y_**2 + 3*y_ - 9,
            4*x_**2 + 6*x_*y_ + 21*x_ + 2*y_**3 + 22*y_**2 + 75*y_ + 72,
            8*x_**2 - 2*x_*y_**2 + 8*x_*y_ + 18*x_ - 5*y_**3 - 32*y_**2 - 84*y_ - 90,
            4*x_**2 + 3*x_*y_**2 + 32*x_*y_ + 72*x_ + 4*y_**3 + 51*y_**2 + 195*y_ + 144,
            -16*x_**2 - 36*x_*y_ + 6*x_ - y_**3 - 55*y_**2 - 93*y_ - 99,
            4*x_**2 - 4*x_*y_**2 - 14*x_*y_ + 45*x_ + y_**3 + 2*y_**2 + 66*y_ + 63,
            32*x_**2 + 52*x_*y_ + 90*x_ - 3*y_**3 + 18*y_**2 + 30*y_ - 9,
            16*x_**2 + 12*x_*y_ + 12*x_ + 13*y_**2 - 6*y_ - 9,
            -4*x_*y_ - 2*y_**2 - 15*y_ - 9
        ]
        coeffsu2 = [
            -2*(4*x_*y_ - y_**3 - 9),
            12*x_**2 - 3*x_*y_**2 + 4*x_*y_ + 12*x_ - 5*y_**3 - 19*y_**2 - 54*y_ - 36,
            12*x_**2 + 3*x_*y_**2 + 36*x_*y_ + 24*x_ + 5*y_**3 + 62*y_**2 + 120*y_ + 90,
            -12*x_**2 - 46*x_*y_ - 45*x_ - y_**3 - 66*y_**2 - 180*y_ - 126,
            12*x_**2 - 6*x_*y_**2 - 22*x_*y_ + 15*x_ + y_**3 + 13*y_**2 + 54*y_ + 90,
            48*x_**2 + 72*x_*y_ + 66*x_ - 4*y_**3 + 28*y_**2 - 48*y_ - 63,
            2*(12*x_**2 + 8*x_*y_ - 9*x_ + 12*y_**2 - 9*y_ + 9),
            -2*(4*x_*y_ + 12*x_ + 2*y_**2 + 27*y_ + 18),
            -12*x_ + 2*y_**2 - 6*y_ + 9
        ]
        def compute_u(v):
            frac1, frac2 = 0, 0
            for i in coeffsu1:
                frac1 *= v
                frac1 += i
            for i in coeffsu2:
                frac2 *= v
                frac2 += i
            return frac1 / frac2

        u_, v_ = None, None
        for root in nroots(eqv, method = 'factor', real = True, nonnegative = True):
            u_ = compute_u(root)
            if u_ * root > 1:
                v_ = root
                break
        if v_ is not None:
            # now that we have obtained u and v
            # we are sure that f(a,b,c) * s(a) >= coeff((3,3,0)) * s(c(a2c-b2c-w(a2b-abc)+z(ab2-abc))2)
            # where w = (u^2+v)/(uv-1), z = (v^2+u)/(uv-1)
            # so we can subtract the right hand side and apply the quartic theorem
            m_, r_ = coeff((4,1,1)), coeff((3,3,0))
            p_, q_ = coeff((3,2,1)) + m_, coeff((2,3,1)) + m_
            n_ = coeff((3,2,1)) + coeff((2,3,1)) + r_
            def get_discriminant(m2, p2, n2, q2):
                m3, p3, n3, q3 = m_ - m2, p_ - p2, n_ - n2, q_ - q2
                det = 3 * m3 * (m3 + n3) - (p3**2 + p3 * q3 + q3**2)
                return det, (m3, p3, n3, q3)
            def get_discriminant_uv(u, v):
                w, z = (u*u + v) / (u*v - 1), (v*v + u) / (u*v - 1)
                m2, p2, n2, q2 = 0, r_*(w*w - 2*z), -r_*2*w*z, r_*(z*z - 2*w)
                return get_discriminant(m2, p2, n2, q2)


            det_ = None
            # print('(u, v) =', (u_, v_))
            # print('det =', get_discriminant_uv(u_, v_))
            if get_discriminant_uv(u_, v_)[0] > -sp.S(10)**(-16):
                # first check that the result is good

                # do rational approximation for both u and v
                for u, v in zip(
                    rationalize_bound(u_, direction = 0, compulsory = True),
                    rationalize_bound(v_, direction = 0, compulsory = True)
                ):
                    det_, (m3, p3, n3, q3) = get_discriminant_uv(u, v)
                    if isinstance(det_, sp.Rational) and det_ >= 0:
                        break
                    det_ = None
                
            if det_ is not None:
                w, z = (u*u + v) / (u*v - 1), (v*v + u) / (u*v - 1)
                p3, n3, q3 = p3 / m3, n3 / m3, q3 / m3
                y = [
                        r_,
                        m3 / 2,
                        (3*(1 + n3) - (p3**2 + p3*q3 + q3**2)) / 6 * m3,
                        rem
                ]
                if all(_ >= 0 for _ in y):
                    multiplier = CyclicSum(a)
                    exprs = [
                        CyclicSum(c*(a**2*c-b**2*c-w*a**2*b+z*a*b**2+(w-z)*a*b*c)**2),
                        CyclicSum((a**2-b**2+(p3+2*q3)/3*(a*c-a*b)-(2*p3+q3)/3*(b*c-a*b))**2) * CyclicProduct(a),
                        CyclicSum(a**2*(b-c)**2) * CyclicProduct(a),
                        CyclicSum(a) * CyclicProduct(a**2)
                    ]
                    return sum_y_exprs(y, exprs) / multiplier


    # Final trial (the code is not expected to reach here)
    # poly = poly * sp.polys.polytools.Poly('a+b+c')
    # multipliers, y, exprs = _merge_sos_results(['a'], y, exprs, recurrsion(poly, 7))

    return None


def _sos_struct_sextic_hexagon(coeff, recurrsion, real = True):
    """
    Solve hexagon s(a4b2+xa2b4+ya3b3+za4bc+wa3b2c+ua2b3c+...a2b2c2)

    Examples
    -------
    s(2a4b2-5a3b3+4a2b4-a4bc)-8/5s(a(b-c)2)p(a)

    2s(a3b3+7a4bc-29a3b2c+12a3bc2+9a2b2c2)+s((2ab2-ca2-a2b-bc2+c2a)2)

    3s(3a4b2-7a4bc+9a4c2-10a3b3+20a3b2c-15a3bc2)-2(s(2a2b-3ab2)+3abc)2

    2s((2ab2-ca2-a2b-bc2+c2a)2)-(s(2a2b-3ab2)+3abc)2


    Reference
    -------
    [1] https://artofproblemsolving.com/community/u426077h1892902p16370027
    """
    # hexagon
    if not (coeff((6,0,0))==0 and coeff((5,1,0))==0 and coeff((5,0,1))==0):
        return None

    if coeff((4,2,0)) == coeff((4,0,2)) and coeff((3,2,1)) == coeff((2,3,1)):
        # symmetric
        # this case must be handled before (?)
        return _sos_struct_sextic_hexagon_symmetric(coeff)

    if coeff((4,2,0)) == 0 or coeff((4,0,2)) == 0:
        if coeff((4,2,0)) == 0 and coeff((4,0,2)) == 0:
            return _sos_struct_sextic_hexagram(coeff)
        if coeff((3,3,0)) == 0 and coeff((4,1,1)) == 0:
            return _sos_struct_sextic_rotated_tree(coeff)

    c420, c240, c330, c411 = [coeff(_) for _ in ((4,2,0),(2,4,0),(3,3,0),(4,1,1))]
    # The hexagon might be positive over a,b,c in R.
    can_be_real = c420 > 0 and c240 > 0 and 4 * c420 * c240 >= max(c330**2, c411**2)

    solvers = []
    if can_be_real:
        if real:
            solvers = [
                _sos_struct_sextic_hexagon_sdp,
                _sos_struct_sextic_hexagon_sdp2,
                _sos_struct_sextic_hexagon_to_hexagram
            ]
        else:
            solvers = [
                _sos_struct_sextic_hexagon_sdp,
                _sos_struct_sextic_hexagon_to_hexagram,
                _sos_struct_sextic_hexagon_sdp2,
            ]
    else:
        solvers = [_sos_struct_sextic_hexagon_to_hexagram]

    for solver in solvers:
        solution = solver(coeff)
        if solution is not None:
            return solution


def _sos_struct_sextic_rotated_tree(coeff):
    """
    Solve s(a2b4+xa3b2c-ya3bc2+za2b2c2) >= 0. Note that this structure is a rotated
    version of symmetric tree.

    Examples
    -------
    s(a2b4-6a3bc2+2a3b2c+3a2b2c2)

    s(a4b2-18a3b2c+18a3bc2-1a2b2c2)

    s(20a4b2-26a3b2c-29a3bc2+35a2b2c2)

    s(a2b4+a3b2c-a3bc2-a2b2c2)

    s(a4b2-14a3b2c+14a3bc2-1a2b2c2)

    s(2a2b4-8a3bc2+a3b2c+5a2b2c2)

    s(a4b2-p(a2))-(p(a-b)-s(a(b-c)2))2/8
    """
    if coeff((4,2,0)) != 0:
        # reflect the polynomial so that coeff((4,2,0)) == 0
        def new_coeff(c):
            return coeff((c[0], c[2], c[1]))
        solution = _sos_struct_sextic_rotated_tree(new_coeff)
        if solution is not None:
            solution = reflect_expression(solution)
        return solution
    
    u = coeff((3,2,1)) / coeff((2,4,0))
    if coeff((2,4,0)) <= 0 or coeff((4,2,0)) != 0 or u < -2:
        return None
    v = coeff((3,1,2)) / coeff((2,4,0))

    if u >= 2 and v >= -6:
        y = [sp.S(1), u - 2, v + 6,
            (coeff((2,4,0)) + coeff((3,1,2)) + coeff((3,2,1))) * 3 + coeff((2,2,2))
        ]
        y = [_ * coeff((2,4,0)) for _ in y[:-1]] + [y[-1]]
        exprs = [
            CyclicSum(a**2*c-a*b*c)**2,
            CyclicProduct(a) * CyclicSum(a**2*b - CyclicProduct(a)),
            CyclicProduct(a) * CyclicSum(a**2*c - CyclicProduct(a)),
            CyclicProduct(a**2)
        ]
        return sum_y_exprs(y, exprs)
    
    if u >= 0 and v >= -2:
        y = [sp.S(1), u, v + 2,
            (coeff((2,4,0)) + coeff((3,1,2)) + coeff((3,2,1))) * 3 + coeff((2,2,2))
        ]
        y = [_ * coeff((2,4,0)) for _ in y[:-1]] + [y[-1]]
        exprs = [
            CyclicSum(a**2*c**2*(a-b)**2),
            CyclicProduct(a) * CyclicSum(a**2*b - CyclicProduct(a)),
            CyclicProduct(a) * CyclicSum(a**2*c - CyclicProduct(a)),
            CyclicProduct(a**2)
        ]
        return sum_y_exprs(y, exprs)

    if True:
        # simple case, we only need to multiply s(a)
        y_ = sp.sqrt(u + 2)
        if not isinstance(y_, sp.Rational):
            y_ = y_.n(16)

        x_ = 1 - (y_ + 2)*(-u - v + 4*y_ + 4)/(2*(u + 4*y_ + 6))
        q1 = v - (x_ - 1)**2 + 2 + 4*y_
        if q1 >= 0:
            if not isinstance(y_, sp.Rational):
                for y_ in rationalize_bound(y_, direction = -1, compulsory = True):
                    x_ = 1 - (y_ + 2)*(-u - v + 4*y_ + 4)/(2*(u + 4*y_ + 6))
                    q1 = v - (x_ - 1)**2 + 2 + 4*y_
                    if q1 >= 0:
                        p1 = u + 2 - y_**2
                        n1 = 2 / (y_ + 2) * (x_ - 1) * p1
                        # after subtracting s(a(a2c-ab2+x(bc2-abc)+y(b2c-abc))2),
                        # the remaining is abcs(p1*a3b+n1*a2b2+q1*ab3+...a2bc)
                        if 4*p1*q1 >= n1**2:
                            # ensure the remainings are positive
                            # note: when y -> sqrt(u^2+2), we have 4pq - n^2 > 0
                            break
                else:
                    y_ = None
            else:
                p1 = u + 2 - y_**2
                n1 = 2 / (y_ + 2) * (x_ - 1) * p1
                if not (4*p1*q1 >= n1**2):
                    y_ = None
            
            if y_ is not None:
                multiplier = CyclicSum(a)
                y = [
                    sp.S(1),
                    p1,
                    q1 - n1**2/4/p1 if p1 != 0 else q1,
                    sp.S(0) if p1 != 0 else n1 / 2,
                    3 * (coeff((2,4,0)) + coeff((3,1,2)) + coeff((3,2,1))) + coeff((2,2,2))
                ]
                y = [_ * coeff((2,4,0)) for _ in y[:-1]] + [y[-1]]
                exprs = [
                    CyclicSum(a*(a**2*c-a*b**2+x_*b*c**2+y_*b**2*c-(x_+y_)*a*b*c)**2),
                    CyclicProduct(a) * CyclicSum(a*b*(a-(-n1/2/p1)*b+(-n1/2/p1-1)*c)**2),
                    CyclicProduct(a) * CyclicSum(a*b*(b-c)**2),
                    CyclicProduct(a) * CyclicSum(a**2*(b-c)**2),
                    CyclicProduct(a**2) * CyclicSum(a)
                ]
                return sum_y_exprs(y, exprs) / multiplier


    r, x = sp.symbols('r'), None
    eq = (r**3 - 3*r - u).as_poly(r)

    def _is_valid_x(x):
        return x >= 1 and v + 3*x*(x - 1) >= 0
    x = rationalize_func(eq, _is_valid_x, direction = -1)

    # print(x, u, v, eq.as_expr().factor())
    if x is None:
        return None
    
    z = x * (x + 1) / 3

    multiplier = CyclicSum(a*b**2 + z*a*b*c) * CyclicSum(a)
    y = [
        1,
        (x + 1) / 2,
        x + 1,
        v + 3*x*(x - 1),
        u - (x**3 - 3*x),
        (coeff((2,4,0)) + coeff((3,1,2)) + coeff((3,2,1))) * 3 + coeff((2,2,2))
    ]
    y = [_ * coeff((2,4,0)) for _ in y[:-1]] + [y[-1]]
    exprs = [
        CyclicSum(a) * CyclicSum(a**2*c*(a**2*c-x*a*b**2-x*b*c**2+(x*x-1)*b**2*c-x*(x-2)*a*b*c)**2),
        CyclicProduct(a**2) * CyclicSum(((x-1)*a-b)**2*(a+(x-1)*b-x*c)**2),
        CyclicProduct(a) * CyclicSum(c*((x-1)*a-b)**2*(x*a*b-a*c-(x-1)*b*c)**2),
        CyclicProduct(a) * CyclicSum(a) * CyclicSum(a**2*c - CyclicProduct(a)) * CyclicSum(a*b**2 + z*a*b*c),
        CyclicProduct(a) * CyclicSum(a) * CyclicSum(a**2*b - CyclicProduct(a)) * CyclicSum(a*b**2 + z*a*b*c),
        CyclicProduct(a**2) * CyclicSum(a) * CyclicSum(a*b**2 + z*a*b*c)
    ]

    return sum_y_exprs(y, exprs) / multiplier


def _sos_struct_sextic_hexagon_full(coeff):
    """
    Still in development

    Examples
    -------
    s(4a5b+9a5c-53a4bc+10a4c2-19a3b3+47a3b2c+52a3bc2-50a2b2c2)
    """
    if coeff((6,0,0)) != 0:
        return None

    coeff51 = coeff((5,1,0))
    if coeff51 == 0 and coeff((1,5,0)) != 0:
        # reflect the polynomial
        return None

    m = sp.sqrt(coeff((1,5,0)) / coeff51)
    if not isinstance(m, sp.Rational):
        return None

    p, n, q = coeff((4,2,0)) / coeff51, coeff((3,3,0)) / coeff51, coeff((2,4,0)) / coeff51

    # (q - t) / (p - t) = -2m
    t = (2*m*p + q)/(2*m + 1)
    z = (p - t) / (-2)
    if t < 0 or n + 2 * t < z**2 - 2*m:
        return None

    # Case A. solve it directly through a cubic equation

    # Case B. optimize discriminant >= 0

    return None



def _sos_struct_sextic_hexagon_sdp(coeff):
    """
    This function is a special case of _sos_struct_sextic_full_sdp
    when coefficients of s(a^6), s(a^5b), s(a^5c) are zero.

    We subtract some s((a2b-ac2-x(a2b-b2c)+y(a2c-ab2))^2) so that
    the rest of the polynomial is a quadratic form with respect to
    s(a^2b-abc) and s(ab^2-abc). See further details at _sos_struct_sextic_full.

    The case is degenerated and actually there is no degree of freedom.

    Examples
    ----------
    s(15a4b2-17a4bc+7a4c2-16a3b3-7a3b2c+21a3bc2-3a2b2c2)
    """
    c42, c33, c24 = [coeff((4-i,i+2,0)) for i in range(3)]
    c411, c321, c231 = [coeff((4-i,i+1,1)) for i in range(3)]

    poly111 = coeff.poly111()
    if (not coeff.is_rational) or poly111 < 0:
        return None

    p1 = 2*c231**2 + 14*c231*c24 + 5*c231*c321 + 18*c231*c33 + 9*c231*c411 + 22*c231*c42 + 20*c24**2 + 22*c24*c321 + 72*c24*c33 + 36*c24*c411 + 104*c24*c42 + 2*c321**2 + 18*c321*c33 + 9*c321*c411 + 14*c321*c42 + 27*c33**2 + 54*c33*c411 + 72*c33*c42 + 36*c411*c42 + 20*c42**2
    p3 = (c231 + 4*c24 + c321 + 3*c33 + 3*c411 + 4*c42)
    p2 = 3*p3*(2*c231 + 10*c24 + c321 + 6*c33 + 3*c411 + 2*c42) - p1

    if p1 == 0 or p2 == 0 or p3 == 0:
        return None

    r = p1**2/p2/54/p3
    x = -9*(c33 - c411)*p3/p1 + 1
    y = -p2/p1

    quad_form = [
        c42 - 2*r*(x**2 - x + 1),
        c24 - 2*r*y**2,
        c33 - 2*r*y*(x-2)
    ]

    if quad_form[0] >= 0 and quad_form[0] * quad_form[1] * 4 >= quad_form[2]**2:
        mapping = lambda x, y: CyclicSum((x*a**2*b + y*a*b**2 - (x+y)*a*b*c).together())**2
        quad_form_sol = quadratic_weighting(quad_form[0], quad_form[2], quad_form[1], mapping = mapping)

        # rs((a2b-ac2-x(a2b-b2c)+y(a2c-ab2))^2)
        solution = r * CyclicSum((a**2*b-a*c**2-x*(a**2*b-b**2*c)+y*(a**2*c-a*b**2)).expand().together()**2)
        return quad_form_sol + solution + poly111 * CyclicProduct(a**2)


def _sos_struct_sextic_hexagon_sdp2(coeff):
    """
    This function is a generalization of _sos_struct_sextic_hexagon_sdp
    that performs sum-of-square on the octic f_2(a,b,c) = f(a,b,c) * s(a^2-ab).

    """
    c420, c330, c240, c411, c321, c312 = [coeff(_) for _ in ((4,2,0),(3,3,0),(2,4,0),(4,1,1),(3,2,1),(3,1,2))]
    if (not coeff.is_rational) or coeff.poly111() != 0:
        return None

    x, y, z = sp.symbols('x y z')

    w1 = c240 + 2*c312 + c321 + 2*c420
    w2 = -c240 + c312 - c321 + c420
    w3 = 2*c240 - 2*c312 - c321 - 3*c330 - 3*c411 + c420
    w5 = c240**2 + c240*c312 + 2*c240*c321 + c240*c420 + c312**2 + c312*c321 + 2*c312*c420 + c321**2 + c321*c420 + c420**2
    
    def stack_quad_form(x, y, z, only_lower = True, div3x = True):
        M = sp.Matrix.zeros(3)
        w0 = y**2 + y + 1
        M[0,0] = (3*c420*x - z*w0)
        M[1,0] = (3*c411*x + z*(-x*y + x + w0))/2
        M[1,1] = (3*c240*x + z*(-x**2 + 2*x*y + x - w0))
        M[2,0] = (w1 - w2 + x*(3*c330 - 3*c420) + y*(w1) + z*(2*x*y + x - w0))/2
        M[2,1] = (w2 + x*(w2 + 3*c321 + 3*c330) + y*(w2 - w1) + z*(x**2 - w0))/2
        M[2,2] = (-w5 + z**2*(-x**2 + x*y - x - w0) + z*(w1 + x*w3 + y*w2))/z

        if not only_lower:
            M[0,1] = M[1,0]
            M[0,2] = M[2,0]
            M[1,2] = M[2,1]
        if div3x:
            M /= 3*x
        return M

    def _get_detur3():
        x, y, z = sp.symbols('x y z')
        M = stack_quad_form(x, y, z, div3x = False)
        eq1z = ((M[0,0]*M[1,1]-M[1,0]**2)*(M[2,2]*z)).as_poly(z)
        eq2z = (2*M[1,0]*M[2,0]*M[2,1] - M[1,1]*M[2,0]**2 - M[0,0]*M[2,1]**2).as_poly(z)
        # return eq1z + z * eq2z
        return sp.Poly.from_list([
            eq1z.coeff_monomial((3,)) + eq2z.coeff_monomial((2,)),
            eq1z.coeff_monomial((2,)) + eq2z.coeff_monomial((1,)),
            eq1z.coeff_monomial((1,)) + eq2z.coeff_monomial((0,)),
            eq1z.coeff_monomial((0,))
        ], z)

    # detur3 = det(M) * 27x^3 * z >= 0
    # To make the quad-form PSD, we let detur3, a cubic poly of z with params (x,y), positive.
    # This is done by optimizing the discriminant.
    detur3 = _get_detur3()

    if False:
        # too slow by computing the exact extrema from the resultant
        from time import time
        time0 = time()
        discriminant = sp.polys.discriminant(detur3)
        discriminant_diffx = sp.diff(discriminant, x)
        discriminant_diffy = sp.diff(discriminant, y)
        discriminant_gcd = sp.gcd(discriminant, discriminant_diffx).factor()
        time1 = time()
        print('Time:', time1 - time0)
        print('GCD =', discriminant_gcd)

    # Alternative: 
    if True:
        # first detect whether
        0

def _sos_struct_sextic_hexagon_to_hexagram(coeff):
    """
    Solve hexagons by subtracting some s(c1(a^2b-abc) - c2(ab^2-abc))^2
    so that the remaing part is a hexagram.

    Sometimes the c1, c2 might be irrational, so we use alternative:
    subtracting c1s(a^2b-abc)^2 + c2s(ab^2-abc)^2 - c3s((a^2b-abc)(ab^2-abc))
    with 4c1c2 >= c3^2 by the quadratic constraint.
    """

    c1, c2 = coeff((4,2,0)), coeff((4,0,2))
    c30, z0, x0, y0, w0 = coeff((3,3,0)), coeff((4,1,1)), coeff((3,2,1)), coeff((2,3,1)), coeff((2,2,2))
    if c1 < 0 or c2 < 0 or (c30 < 0 and c30**2 > 4*c1*c2) or (z0 < 0 and z0**2 > 4*c1*c2):
        return None
    x0, y0, w0 = x0 + 6*c1 - 2*c2, y0 + 6*c2 - 2*c1, w0 - 9*(c1 + c2)

    def _compute_hexagram_discriminant(z, x, y, coeff33 = 1):
        z, x, y = z / coeff33, x / coeff33, y / coeff33
        s1, s2 = x + y, x * y
        d, e = (s1**2 + 9*s1 - s2 + 27), s1**2 - s2
        args = [
            e * d**3,
            -27 * e * d**2,
            - e * d * (4*s1**2 - 45*s1 - 4*s2 - 378),
            (8*s1**5 - 207*s1**4 - 16*s1**3*s2 - 3051*s1**3 + 234*s1**2*s2 - 9963*s1**2 + 8*s1*s2**2 + 2970*s1*s2 - 729*s1 - 27*s2**2 + 9477*s2)/3,
            (10*s1**4 + 351*s1**3 - 20*s1**2*s2 + 1890*s1**2 - 351*s1*s2 + 486*s1 + 10*s2**2 - 1728*s2)/3,
            -(16*s1**3 + 207*s1**2 - 16*s1*s2 + 189*s1 - 171*s2)/3,
            (28*s1**2 + 111*s1 - 12*s2 + 9)/9,
            -(8*s1 + 3)/9,
            sp.Rational(1,9)
        ]
        u = x + y + 3*z + 3
        discriminant = sum(u**i*args[i] for i in range(len(args)))
        return discriminant

    def _compute_subtracted_params(c3, return_func = False):
        params = z0 + c3, x0 - 3*c3, y0 - 3*c3, c30 + c3, w0 + 12*c3
        # print('c3 =', c3, 'params =', params)
        if return_func:
            coeffs_ = dict(zip([(4,1,1),(3,2,1),(2,3,1),(3,3,0),(2,2,2)], params))
            coeffs_[(3,1,2)] = coeffs_[(2,3,1)]
            return Coeff(coeffs_)
        return params
    
    def _check_valid(c3):
        if not (c3 >= 0 and 4*c1*c2 >= c3**2):
            return False
        params = _compute_subtracted_params(c3)
        if params[0] < 0 or params[3] < 0:
            return False
        if params[1] >= 0 and params[2] >= 0:
            return True
        if params[0] == 0 or params[3] == 0:
            if params[0] == 0 and params[3] == 0:
                return False
        reg = max(params[0], params[3])
        # degenerates to cubic
        x, y = params[1] / reg, params[2] / reg
        if x**2*y**2 + 18*x*y - 4*x**3 - 4*y**3 - 27 <= 0:
            return True
        if params[0] == 0 or params[3] == 0:
            return False
        det = _compute_hexagram_discriminant(*params[:-1])
        if det <= 0:
            return True
        return False


    if x0 == -3*c30 and y0 == -3*c30:
        # Degenerates to c1*s(a2b-abc)^2 + c2*s(ab2-abc)^2 + c3s(a2b-abc)s(ab2-abc) + 0
        c3 = -c30
    else:
        eq = sp.Poly.from_list([1, 0, -4*c1*c2], sp.Symbol('u'))
        c3 = rationalize_func(eq, _check_valid, validation_initial = lambda x: x >= 0, direction = -1)
        
    if c3 is not None:
        remain_solution = _sos_struct_sextic_hexagram(_compute_subtracted_params(c3, return_func = True))
        if remain_solution is not None:
            mapping = lambda x, y: CyclicSum((x*a**2*b + y*a*b**2 - (x+y)*a*b*c).together())**2
            main_solution = quadratic_weighting(c1, -c3, c2, mapping = mapping)
            return main_solution + remain_solution


def _sos_struct_sextic_full_sdp(coeff):
    """
    Heuristically solve full sextics with the method of unknown coefficients.
    Idea: Let f(a,b,c) = a^3-b^3+ua^2b+vb^2c-(u+v)ac^2+xab^2+ya^2c-(x+y)bc^2.
    Then we note that \sum f(a,b,c) == 0.
    We subtract some \sum f(a,b,c)^2 from the original polynomial to ensure that
    the rest part is a quadratic form with respect to s(a^3-abc), s(a^2b-abc), s(ab^2-abc).


    Examples
    ----------
    s(2a6+a4bc-3a5b)

    16s(a2)3-9s((a3+3b2c)2)

    s((a3+3b2c)2)-16abcs(a3)

    s((a2-bc)(4b2+4c2+a2)(4c2+4a2+b2))

    243s(a(a+b)5)-32s(a)6

    s((a-b)2(a2-2ab-ac+b2)2)
    
    References
    ----------
    [1] https://artofproblemsolving.com/community/u585449h2736959p29529763

    [2] https://artofproblemsolving.com/community/c6t243f6h2747973_inequality

    [3] https://artofproblemsolving.com/community/u426077h3222992p29498828

    [4] https://artofproblemsolving.com/community/q1h101901p575510
    """
    c60, c51, c42, c33, c24, c15 = [coeff((6-i,i,0)) for i in range(6)]
    c411, c321, c231 = [coeff((4-i,i+1,1)) for i in range(3)]

    if (not coeff.is_rational) or c60 < 0 or coeff.poly111() != 0: 
        return None

    x, y = sp.symbols('x y')

    # compute the coefficients of the quadratic form
    # with respect to s(a^3-abc), s(a^2b-abc), s(ab^2-abc)
    # quad_form = [
    #     [c60, c51/2, c15/2],
    #     [c51/2, c42-c15, c33/2-c60],
    #     [c15/2, c33/2-c60, c24-c51]
    # ]
    
    def _compute_quad_form_sol(quad_form):
        q, ss = congruence(quad_form)
        quad_form_sol = [
            (q[0,0]*(a**3-a*b*c)+q[0,1]*(a**2*b-a*b*c)+q[0,2]*(a*b**2-a*b*c)),
            (q[1,1]*(a**2*b-a*b*c)+q[1,2]*(a*b**2-a*b*c)),
            (q[2,2]*(a*b**2-a*b*c))
        ]
        quad_form_sol = map(lambda x: CyclicSum(x.expand().together())**2, quad_form_sol)
        return sum_y_exprs(ss, quad_form_sol)

    # rest form is what the original polynomial subtracts the quadratic form,
    # corresponding to the coefficients of a^4bc, a^3b^2c, a^2b^3c
    rest_form = [
        c411 + 3*c15 - c33 + 3*c51 + 8*c60, 
        c321 + -6*c15 - 2*c24 + 3*c33 + 6*c42 + 4*c51 - 6*c60, 
        c231 + 4*c15 + 6*c24 + 3*c33 - 2*c42 - 6*c51 - 6*c60
    ]

    if all(_ == 0 for _ in rest_form):
        quad_form = sp.Matrix([
            [c60, c51/2, c15/2],
            [c51/2, c42-c15, c33/2-c60],
            [c15/2, c33/2-c60, c24-c51]
        ])
        quad_form_sol = _compute_quad_form_sol(quad_form)
        return quad_form_sol


    if rest_form[0] != 0:
        r1, r2 = rest_form[1] / rest_form[0], rest_form[2] / rest_form[0]
        denom = ((r1 + 2*r2 + sp.Rational(9,2))**2 + sp.Rational(27,4))

        # We shall subtract r/z * \sum f(a,b,c)^2 from the original polynomial to make rest_form == 0
        # where z = x^2 + x(y-3) + y^2 + 3 is a variable
        r = rest_form[0] / (54*(r1 + r2 + 3)/denom)

        # the coefficients u,v in f can be written as functions of x, y
        u = (2*r1**2 + 5*r1*r2 + 9*r1 + 2*r2**2 + 9*r2)*x + (2*r1**2 + 5*r1*r2 + 18*r1 + 2*r2**2 + 18*r2 + 27)*y + (- 4*r1**2 - 13*r1*r2 - 18*r1 - 10*r2**2 - 36*r2 - 27)
        v = (9*r1 + 9*r2 + 27)*x + (-2*r1**2 - 5*r1*r2 - 9*r1 - 2*r2**2 - 9*r2)*y + (- r1**2 - r1*r2 - 18*r1 + 2*r2**2 - 9*r2 - 27)
        u, v = u / denom, v / denom

    elif rest_form[1] != 0:
        r1 = rest_form[2] / rest_form[1]
        denom = 2*r1 + 1

        r = rest_form[1] * denom**2/(54*(r1 + 1))
        u = (r1 + 2)*x + (r1 + 2)*y + (-5*r1 - 4)
        v = (-r1 - 2)*y + (r1 - 1)
        u, v = u / denom, v / denom

    else:
        # not implemented
        return None

    z = x**2 + x*(y - 3) + y**2 + 3

    # find x, y such that the following quad_form is positive semidefinite
    # (the upper triangular part is omitted)
    quad_form = sp.Matrix([
        [c60*z - 2*r, 0, 0],
        [c51/2*z - r*(u-v), (c42-c15)*z - r*(2*u**2+2*u*v+2*v**2+6*x), 0],
        [c15/2*z - r*(y-x), (c33/2-c60)*z - r*(2*u*x - 2*u*y - 2*v*x - 4*v*y - 6)/2, (c24-c51)*z - r*(-6*u + 2*x**2 + 2*x*y + 2*y**2)]
    ]).expand()

    def quad_form_det(x0, y0, n = 3):
        q = quad_form[:n,:n].subs({x:x0, y:y0})
        if n == 3:
            # it is a six-degree polynomial with respect to x0, y0
            return q[0,0]*q[1,1]*q[2,2] + 2*q[1,0]*q[2,1]*q[2,0] - q[0,0]*q[2,1]**2 - q[2,2]*q[1,0]**2 - q[1,1]*q[2,0]**2
        elif n == 2:
            return q[0,0]*q[1,1]-q[1,0]**2
        elif n == 1:
            return q[0,0]

    # optimize the determinant of the quad_form by taking partial derivatives
    success = False
    for n_ in (3,):
        det = quad_form_det(x,y,n_)
        res = sp.resultant(det.diff(x), det.diff(y), x)
        for y_ in nroots(res.as_poly(y), method='factor', real=True):
            det2 = det.subs(y,y_).as_poly(x)
            for x_ in nroots(det2.diff(x), method='factor', real=True):
                if det2(x_) >= 0 and quad_form_det(x_,y_,1) >= 0 and quad_form_det(x_,y_,2) >= 0:
                    # print((x_, y_))
                    success = True
                    break
            if success:
                break

    if not success:
        return None            

    if not isinstance(x_, sp.Rational) or not isinstance(y_, sp.Rational):
        # make a perturbation so that the quad_form is positive semidefinite
        x_, y_ = x_.n(15), y_.n(15)
        for x__, y__ in zip_longest(
            rationalize_bound(x_, direction = 0, compulsory = True),
            rationalize_bound(y_, direction = 0, compulsory = True),
        ):
            if all(quad_form_det(x__, y__, i) >= 0 for i in range(1,4)):
                x_, y_ = x__, y__
                break
        else:
            return None

    # symmetrize as Hermitian matrix
    quad_form = quad_form.subs({x:x_, y:y_})
    z_ = z.subs({x:x_, y:y_})
    quad_form = ((quad_form + quad_form.T) - sp.diag(*quad_form.diagonal())) / z_

    quad_form_sol = _compute_quad_form_sol(quad_form)

    u_, v_ = u.subs({x:x_, y:y_}), v.subs({x:x_, y:y_})
    main_sol = CyclicSum((a**3 + a**2*b*u_ + a**2*c*y_ + a*b**2*x_ + a*c**2*(-u_ - v_) - b**3 + b**2*c*v_ + b*c**2*(-x_ - y_)).together()**2)
    return quad_form_sol + r/z_ * main_sol