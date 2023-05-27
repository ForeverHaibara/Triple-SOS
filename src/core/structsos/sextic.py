import sympy as sp
from sympy.solvers.diophantine.diophantine import diop_DN

from .sextic_symmetric import (
    _sos_struct_sextic_hexagon_symmetric,
    _sos_struct_sextic_hexagram_symmetric,
    sos_struct_sextic_symmetric_ultimate
)
from .utils import CyclicSum, CyclicProduct, _sum_y_exprs
from ...utils.text_process import cycle_expansion
from ...utils.roots.rationalize import rationalize_bound, cancel_denominator


a, b, c = sp.symbols('a b c')


def sos_struct_sextic(poly, coeff, recurrsion):
    solution = sos_struct_sextic_symmetric_ultimate(poly, coeff, recurrsion)
    if solution is not None:
        return solution

    if coeff((6,0,0))==0 and coeff((5,1,0))==0 and coeff((5,0,1))==0:
        return _sos_struct_sextic_hexagon(coeff, poly, recurrsion)

    return None


def _sos_struct_sextic_inverse_triangle(coeff, poly):
    """
    Try to solve s(xa3b3 + ya3b2c + za2b3c + wa2b2c2) >= 0
    without updegree.
    """
    pass



def _sos_struct_sextic_hexagram(coeff, poly, recurrsion):
    """
    Solve s(a3b3+xa4bc+ya3b2c+za2b3c+wa2b2c2) >= 0

    Examples
    -------
    s(a3b3+7a4bc-29a3b2c+12a3bc2+9a2b2c2)

    9s(a3b3)+4s(a4bc)-11abcp(a-b)-37s(a3b2c)+72a2b2c2

    s(11a4bc+11a3b3+153a3bc2-153a3b2c-22a2b2c2)

    s(bc(a2-bc+2(ab-ac)+3(bc-ab))2)
    """
    if coeff((3,3,0)) < 0 or coeff((4,1,1)) < 0:
        return None

    if coeff((3,2,1)) == coeff((2,3,1)):
        # call symmetric solution in priority
        solution = _sos_struct_sextic_hexagram_symmetric(coeff)
        if solution is not None:
            return solution
    
    # simple cases where we do not need to updegree
    if coeff((4,1,1)) != 0 and coeff((3,3,0)) != 0:
        # try subtracting s(bc(a-ub)^2(a-c)^2)
        m = coeff((4,1,1))
        u = (coeff((3,3,0)) / m) ** .5

        def verify(c1, c2, c3):
            # test whether quadratic function c1*x^2 + c2*x + c3 >= 0
            # or c1*x^2 + c3*x + c2 >= 0
            if c2 >= 0 and c3 >= 0:
                return True
            if c2 < 0 and c2**2 <= 4*c1*c3:
                return True
            if c3 < 0 and c3**2 <= 4*c1*c2:
                return True
            return False
        
        success = 0
        if coeff((3,2,1)) - (-2*u**2 - 2*u + 1) * m >= 0 and\
            coeff((3,1,2)) - (u**2 - 2*u - 2) * m >= 0:
            # s(bc(a-ub)^2(a-c)^2)
            success = -1
        elif coeff((3,1,2)) - (-2*u**2 - 2*u + 1) * m >= 0 and\
            coeff((3,2,1)) - (u**2 - 2*u - 2) * m >= 0:
            # symmetric, s(bc(a-b)^2(a-uc)^2)
            success = -2
        
        if success < 0:
            # find a rational approximation
            for u_ in rationalize_bound(u, direction = -1):
                if verify(
                    coeff((3, 3, 0)) - u_**2 * m,
                    coeff((3, 3+success, -success)) - (-2*u_**2 - 2*u_ + 1) * m,
                    coeff((3, -success, 3+success)) - (u_**2 - 2*u_ - 2) * m
                ):
                    u = u_
                    success = -success
                    break

        if success > 0:
            y = [m]
            rem = {(3,3,0): coeff((3,3,0)) - u**2 * m,
                    (3, 3+success, -success): coeff((3, 3+success, -success)) - (-2*u**2 - 2*u + 1) * m,
                    (3, -success, 3+success): coeff((3, -success, 3+success)) - (u**2 - 2*u - 2) * m
                }
            if success == 1:
                exprs = [f'b*c*(a-{u}*b)^2*(a-c)^2']
            elif success == 2:
                exprs = [f'b*c*(a-b)^2*(a-{u}*c)^2']
            



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
        for root in sp.polys.nroots(eqv, n = 20):
            if root.is_real and root > 0:
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
                    if det_ >= 0:
                        break
                    det_ = None
                
            if det_ is not None:
                w, z = (u*u + v) / (u*v - 1), (v*v + u) / (u*v - 1)
                p3, n3, q3 = p3 / m3, n3 / m3, q3 / m3
                y = [
                        r_,
                        m3 / 2,
                        (3*(1 + n3) - (p3**2 + p3*q3 + q3**2)) / 6 * m3,
                        sum(coeff(i) for i in [(3,3,0),(4,1,1),(3,2,1),(2,3,1)]) * 3 + coeff((2,2,2))
                ]
                if all(_ >= 0 for _ in y):
                    multiplier = CyclicSum(a)
                    exprs = [
                        CyclicSum(c*(a**2*c-b**2*c-w*a**2*b+z*a*b**2+(w-z)*a*b*c)**2),
                        CyclicSum((a**2-b**2+(p3+2*q3)/3*(a*c-a*b)-(2*p3+q3)/3*(b*c-a*b))**2) * CyclicProduct(a),
                        CyclicSum(a**2*(b-c)**2) * CyclicProduct(a),
                        CyclicSum(a) * CyclicProduct(a**2)
                    ]
                    return _sum_y_exprs(y, exprs) / multiplier


    # Final trial (the code is not expected to reach here)
    # poly = poly * sp.polys.polytools.Poly('a+b+c')
    # multipliers, y, exprs = _merge_sos_results(['a'], y, exprs, recurrsion(poly, 7))

    return None


def _sos_struct_sextic_hexagon(coeff, poly, recurrsion):
    """
    Solve hexagon s(a4b2+xa2b4+ya3b3+za4bc+wa3b2c+ua2b3c+...a2b2c2)

    Examples
    -------
    2s(a3b3+7a4bc-29a3b2c+12a3bc2+9a2b2c2)+s((2ab2-ca2-a2b-bc2+c2a)2)
    3s(3a4b2-7a4bc+9a4c2-10a3b3+20a3b2c-15a3bc2)-2(s(2a2b-3ab2)+3abc)2
    2s((2ab2-ca2-a2b-bc2+c2a)2)-(s(2a2b-3ab2)+3abc)2


    Reference
    -------
    [1] https://artofproblemsolving.com/community/u426077h1892902p16370027
    """
    # hexagon
    if coeff((6,0,0))==0 and coeff((5,1,0))==0 and coeff((5,0,1))==0:
        if coeff((4,2,0)) != coeff((4,0,2)):
            # first try whether we can cancel these two corners in one step
            # CASE 1.
            if coeff((4,2,0)) == 0 or coeff((4,0,2)) == 0:
                if coeff((3,3,0)) == 0 and coeff((4,1,1)) == 0:
                    return _sos_struct_sextic_rotated_tree(coeff)

                if coeff((4,0,2)) == 0:
                    solution = coeff((4,2,0)) * CyclicSum(a**2*b-a*b*c)**2
                else:
                    solution = coeff((4,0,2)) * CyclicSum(a*b**2-a*b*c)**2

                poly2 = poly - solution.doit().as_poly(a,b,c)
                v = 0 # we do not need to check the positivity, just try
                if v != 0:
                    y, exprs = None, None
                else: # positive
                    new_solution = recurrsion(poly2)
                    if new_solution is not None:
                        return solution + new_solution

            # CASE 2.
            else:
                # Temporarily deprecated.
                # The current method is not complete.
                return 
                a , b = (coeff((4,2,0)) / coeff((4,0,2))).as_numer_denom()
                if sp.ntheory.primetest.is_square(a) and sp.ntheory.primetest.is_square(b):
                    y = [coeff((4,2,0)) / a / 3]
                    exprs = [f'({sp.sqrt(a)}*(a*a*b+b*b*c+c*c*a-3*a*b*c)-{sp.sqrt(b)}*(a*b*b+b*c*c+c*a*a-3*a*b*c))^2']
                
                    poly2 = poly - 3 * y[0] * sp.sympify(exprs[0])
                    v = 0 # we do not need to check the positivity, just try
                    if v != 0:
                        y, exprs = None, None
                    else: # positive
                        multipliers , y , exprs = _merge_sos_results(multipliers, y, exprs, recurrsion(poly2, 6))

                # CASE 3.
                # a = t(p^2+1) , b = t(q^2+1)
                # => b^2p^2 + b^2 = ab(q^2 + 1) => (bp)^2 - ab q^2 = b(a-b)
                #                            or => (aq)^2 - ab p^2 = a(b-a)
                # here we should require b is squarefree and not too large
                elif a < 30 and b < 30: # not too large for solving Pell's equation
                    pairs = []
                    for pair in diop_DN(a*b, b*(a-b)):
                        if pair[0] % b == 0:
                            pairs.append((abs(pair[0] // b), abs(pair[1])))
                    for pair in diop_DN(a*b, a*(b-a)):
                        if pair[0] % a == 0:
                            pairs.append((abs(pair[1]), abs(pair[0] // a)))
                    pairs = set(pairs)
                    
                    for p , q in pairs:
                        p , q = abs(p) , abs(q)
                        t = coeff((4,2,0)) / (p*p + 1)
                        if coeff((3,3,0)) + t * 2 * p * q < 0 or coeff((4,1,1)) + t * 2 * (p + q) < 0:
                            # negative vertex, skip it
                            continue
                        y = [t]
                        exprs = [f'(a*a*c-b*b*c-{p}*(a*a*b-a*b*c)+{q}*(a*b*b-a*b*c))^2']
                        poly2 = poly - t * sp.sympify(cycle_expansion(exprs[0]))
                        multipliers , y , exprs = _merge_sos_results(multipliers, y, exprs, recurrsion(poly2, 6))
                        if y is not None:
                            break


        else:# coeff((4,2,0)) == coeff((4,0,2)):
            if coeff((4,2,0)) == 0:
                return _sos_struct_sextic_hexagram(coeff, poly, recurrsion)
            elif coeff((3,2,1)) == coeff((2,3,1)):
                # symmetric
                return _sos_struct_sextic_hexagon_symmetric(coeff)
            else:
                solution = coeff((4,2,0)) * CyclicProduct((a-b)**2)

                poly2 = poly - solution.doit().as_poly(a,b,c)
                v = 0 # we do not need to check the positivity, just try
                if v != 0:
                    y, exprs = None, None
                else: # positive
                    new_solution = recurrsion(poly2)
                    if new_solution is not None:
                        return solution + new_solution
                
    return None



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
    """
    if coeff((4,2,0)) != 0:
        # reflect the polynomial so that coeff((4,2,0)) == 0
        def new_coeff(c):
            return coeff((c[0], c[2], c[1]))
        solution = _sos_struct_sextic_rotated_tree(new_coeff)
        if solution is not None:
            solution = solution.xreplace({b:c, c:b})
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
        return _sum_y_exprs(y, exprs)
    
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
        return _sum_y_exprs(y, exprs)

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
                    (coeff((2,4,0)) + coeff((3,1,2)) + coeff((3,2,1))) + coeff((2,2,2)) / 3
                ]
                y = [_ * coeff((2,4,0)) for _ in y[:-1]] + [y[-1]]
                exprs = [
                    CyclicSum(a*(a**2*c-a*b**2+x_*b*c**2+y_*b**2*c-(x_+y_)*a*b*c))**2,
                    CyclicProduct(a) * CyclicSum(a*b*(a-(-n1/2/p1)*b+(-n1/2/p1-1)*c)**2),
                    CyclicProduct(a) * CyclicSum(a*b*(b-c)**2),
                    CyclicProduct(a) * CyclicSum(a**2*(b-c)**2),
                    CyclicProduct(a**2) * CyclicSum(a)
                ]
                return _sum_y_exprs(y, exprs) / multiplier


    r, x = sp.symbols('r'), None
    eq = (r**3 - 3*r - u).as_poly(r)
    for root in sp.polys.roots(eq, cubics = False, quartics = False):
        if isinstance(root, sp.Rational) and root >= 1:
            x = root
            break
    
    if x is None:
        for root in sp.polys.nroots(eq):
            if root.is_real and root >= 1:
                for x_ in rationalize_bound(root, direction = -1, compulsory = True):
                    if x_**3 - 3*x_ <= u and v + 3*x_*(x_ - 1) >= 0:
                        x = x_
                        break
    
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

    return _sum_y_exprs(y, exprs) / multiplier