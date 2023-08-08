import sympy as sp

from .utils import CyclicSum, CyclicProduct, _sum_y_exprs, _make_coeffs
from ...utils.roots.rationalize import rationalize_bound, cancel_denominator


#####################################################################
#
#                              Symmetric
#
#####################################################################

a, b, c = sp.symbols('a b c')

def _sos_struct_sextic_hexagon_symmetric(coeff, real = False):
    """
    Solve symmetric hexagons.

    Theorem 1:
    When x not in (-2,1), the following inequality holds for all real numbers a, b, c:
    f(a,b,c) = x^2/4 * p(a-b)^2 + s(bc(a-b)(a-c)(a-xb)(a-xc)) >= 0

    Proof: let
    lambda = (x**2 + 4*x - 8)**2/(4*(x - 2)**2*(5*x**2 - 4*x + 8 + (4*x - 16)*sqrt(x**2 + x - 2)))
    z = (2*x**2 - 2*x + 2*(x - 2)*sqrt(x**2 + x - 2))/(x**2 + 4*x - 8)
    Then we have,
    f(a,b,c) * s((a-b)^2) = lambda * s((a-b)^2*((x-2)ab(a+b) - (x-2z)c(a^2+b^2-c^2) - x(1-z)c^2(a+b+c) + (2x+4-3xz-2z)abc)^2)


    Examples
    --------
    s(3a2b+ab2-4abc)2+s(a2b+3ab2-4abc)2    (real)

    s((b-c)2(a2-3(ab+ac)+2bc)2)/14-8/14abcs(a2b+ab2-2abc)    (real)

    s(a2(b-c)4)-1/2p(a-b)2      (real)

    s(a2(b2-c2)2)-3/8p(a-b)2    (real, root = (-1,-1,1))

    3p(a2+ab+b2)-s(a)2s(ab)2    (real)

    p(a2+ab+b2)-3s(ab)s(a2b2)   (real)

    s(a)s(ab)p(a+b)-6s(a(a+b)(a+c))p(a)-3abcs((a+b-c)(a-c)(b-c))   (real)

    s(bc(a-b)(a-c)(a-3b)(a-3c))+9/4p(a-b)2

    s(bc(a-b)(a-c)(a-3b)(a-3c)) +1/4s(a2b+ab2-2abc)2+5p(a-b)2+4abcs(a(b-c)2)
    
    s(bc(a-b)(a-c)(a-9/8b)(a-9/8c))+81/256p(a-b)2

    p(a2+ab+b2)+12a2b2c2-3p(a+b)2/5    (real, uncentered)
    
    Reference
    ---------
    [1] Vasile, Mathematical Inequalities Volume 1 - Symmetric Polynomial Inequalities. p.23

    [2] https://artofproblemsolving.com/community/u426077h3036593p28226075
    """    
    if coeff((4,2,0)) <= 0:
        if coeff((4,2,0)) == 0:
            return _sos_struct_sextic_hexagram_symmetric(coeff)
        return None

    rem = (coeff((4,2,0)) + coeff((3,2,1))) * 6 + (coeff((3,3,0)) + coeff((4,1,1))) * 3 + coeff((2,2,2))
    if rem < 0:
        return None
    
    # although subtracting p(a-b)2 always succeeds,
    # we can handle cases for real numbers and cases where updegree is not necessary
    if True:
        # if 3*(coeff((4,2,0))*2 + coeff((3,3,0)) + coeff((4,1,1)) + coeff((3,2,1))*2) + coeff((2,2,2)) == 0:
        # has root at (1,1,1)
        x_, y_, z_ = coeff((3,3,0)) / coeff((4,2,0)), coeff((4,1,1)) / coeff((4,2,0)), -coeff((3,2,1)) / coeff((4,2,0))
        
        if x_ < -2 or y_ < -2:
            return None
            
        if x_ == y_ and x_ <= 2:
            # try us(a2b+ab2-2abc)2 + (1-u)p(a-b)2
            # where 2u - 2(1-u) = x  =>  u = (x + 2) / 4
            y = [
                (x_ + 2) / 4,
                (2 - x_) / 4,
                -z_ + 10 * (x_ + 2) / 4 - 2 * (2 - x_) / 4
            ]
            if y[-1] >= 0:
                y = [_ * coeff((4,2,0)) for _ in y] + [rem]
                exprs = [
                    CyclicSum(a*(b-c)**2)**2,
                    CyclicProduct((a-b)**2),
                    CyclicProduct(a) * CyclicSum(a*(b-c)**2),
                    CyclicProduct(a**2)
                ]

                return _sum_y_exprs(y, exprs)
                

        if x_ <= 2 and y_ <= 2 and (x_ != -2 or y_ != -2):
            u_ = sp.sqrt(x_ + 2)
            v_ = sp.sqrt(y_ + 2)
            if u_**2 + v_**2 + u_*v_ - 2 == z_:
                if u_ != 0:
                    r = v_ / u_
                    # (x - w) / (1 - w) = (-2*r*r-2*r+1)/(r*r+r+1)
                    t = (-2*r*r-2*r+1)/(r*r+r+1)
                    w = (t - x_) / (t + 2)
                    if 0 <= w <= 1:
                        x__, y__ = 1-3/(r+2), -(2*r+1)/(r+2)
                        x__q, y__q = x__.as_numer_denom()[1], y__.as_numer_denom()[1]
                        rr = x__q * y__q / sp.gcd(x__q, y__q)
                        y = [
                            w,
                            (1 - w) / (x__**2 + y__**2 + 1) / rr**2
                        ]
                        y = [_ * coeff((4,2,0)) for _ in y] + [rem]
                        exprs = [
                            CyclicProduct((a-b)**2),
                            CyclicSum((b-c)**2*(rr*a**2-rr*x__*a*b-rr*x__*a*c+rr*y__*b*c)**2),
                            CyclicProduct(a**2)
                        ]
                        return _sum_y_exprs(y, exprs)
        
    if real and abs(coeff((3,3,0))) <= 2 * (coeff((4,2,0))) and abs(coeff((4,1,1))) <= 2 * (coeff((4,2,0))):
        # perform sum of squares for real numbers
        
        # case 1. we can subtract enough p(a-b)2 till the bound on the border
        # and use Theorem 1.
        coeff42, u, v = coeff((4,2,0)), coeff((3,3,0)), coeff((4,1,1))
        if u >= v and (u != coeff42 * (-2) or v != coeff42 * (-2)):
            # subtracting p(a-b)2 will first lead to s(a4b2+2a3b3+a2b4)
            # assume coeffp = coefficient of p(a-b)2
            # (coeff42 - coeffp) * (2) = u + 2 * coeffp
            coeffp = (coeff42 * 2 - u) / 4
            coeff42 -= coeffp
            u, v = sp.S(2), (v + coeffp * 2) / coeff42

            # now we need to ensure (v + 2) is a square, and make a perturbation otherwise
            # assume coeffsym = coefficient of s(a2b+ab2-2abc)2
            # (v - 2 * coeffsym / coeff42) / (1 - coeffsym / coeff42) + 2 = x*x <= 4
            # print(v, v.n(20), coeffp)
            coeffsym, x = sp.S(0), sp.sqrt(v + 2)
            if not isinstance(x, sp.Rational):
                for x_ in rationalize_bound(x.n(20), direction = -1, compulsory = True):
                    if abs(x_) == 2:
                        continue

                    coeffsym = ((v + 2 - x_**2) / (4 - x_**2)) * coeff42
                    coeff321_std = (-x_**2 - 2*x_ - 2)
                    coeff321 = (coeff((3,2,1)) - 2 * coeffp + 10 * coeffsym) / (coeff42 - coeffsym)
                    if 0 < x_ <= 1 and 0 <= coeff321 - coeff321_std <= 4 * x_:
                        x = x_
                        break
                    elif 1 < x_ <= 2 and coeff321 == coeff321_std:
                        x = x_
                        break
                else:
                    x_ = None
            coeff42 -= coeffsym

            if x is not None and isinstance(x, sp.Rational):
                if x == 0:
                    # must be the case s(bc(b+c)2(a-b)(a-c))
                    # Solution1: s((a-b)2(a2b-a2c+ab2-10abc+3ac2-b2c+3bc2+4c3)2)
                    # Solution2: s((a-b)2(a2b-a2c+ab2+2abc-ac2-b2c-bc2)2)
                    if (coeff((3,2,1)) - 2 * coeffp + 10 * coeffsym) == -2 * coeff42:
                        y = [coeffp, coeffsym, coeff42, rem]
                        multiplier = CyclicSum((a-b)**2)
                        exprs = [
                            CyclicProduct((a-b)**2) * CyclicSum((a-b)**2),
                            CyclicSum(a*(b-c)**2)**2 * CyclicSum((a-b)**2),
                            CyclicSum((a-b)**2 * (a**2*b - a**2*c + a*b**2 + 2*a*b*c - a*c**2 - b**2*c - b*c**2)**2),
                            CyclicProduct(a**2) * multiplier
                        ]
                        if all(_ >= 0 for _ in y):
                            return _sum_y_exprs(y, exprs) / multiplier
                
                else:
                    x = 2 / x

                    # weight of linear combination of x and -x
                    w2 = ((coeff((3,2,1)) - 2 * coeffp + 10 * coeffsym) / coeff42 - (-2*(x**2 + 2*x + 2)/x**2)) / (8 / x)
                    w1 = 1 - w2
                    if w1 == 0 or w2 == 0:
                        # low rank case
                        if (w1 == 0 and x >= 2) or (w2 == 0 and x >= 1):
                            if w1 == 0:
                                x = -x
                            multiplier = CyclicSum((a-b)**2)
                            tt = sp.sqrt(x*x + x - 2)
                            if isinstance(tt, sp.Rational):
                                y = [coeffp, coeffsym]
                                exprs = [
                                    CyclicProduct((a-b)**2) * CyclicSum((a-b)**2),
                                    CyclicSum(a*(b-c)**2)**2 * CyclicSum((a-b)**2)
                                ]
                                if x != 2:
                                    z = 2*(x**2 - x + (x-2)*tt) / (x*x + 4*x - 8)
                                    r2 = 1 / cancel_denominator([x-2, 2*z-x, -3*x*z+2*x-2*z+4, x*(1-z)])

                                    y.append((5*x**2 - 4*x + 8 - 4*(x - 4)*tt) / 9 / (x - 2)**2 * coeff42 / r2**2 / x**2)
                                    exprs.append(
                                        CyclicSum((a-b)**2*(r2*(x-2)*a**2*b+r2*(-x+2*z)*a**2*c+r2*(x-2)*a*b**2+r2*(-3*x*z+2*x-2*z+4)*a*b*c
                                            +r2*(x*z-x)*a*c**2+r2*(-x+2*z)*b**2*c+r2*(x*z-x)*b*c**2+r2*(x*z-2*z)*c**3)**2)
                                    )
                                elif x == 2:
                                    y.append(sp.S(4) * coeff42 / x**2)
                                    exprs.append(
                                        CyclicSum((a-b)**2*(a**2*b+a*b**2-5*a*b*c+a*c**2+b*c**2+c**3)**2)
                                    )
                                return _sum_y_exprs(y, exprs) / multiplier


                    if x >= 2:
                        multiplier = CyclicSum((a-b)**2)

                        r1 = 1 / cancel_denominator([2, 2+3*x, x])

                        y = [
                            coeffp,
                            coeffsym,
                            w1 / 4 / r1**2,
                            (x**2 + 4*x - 8) / 4 * w1,
                        ]

                        exprs = [
                            CyclicProduct((a-b)**2) * CyclicSum((a-b)**2),
                            CyclicSum(a*(b-c)**2)**2 * CyclicSum((a-b)**2),
                            CyclicSum(c**2*(a-b)**2*(2*r1*a**2+2*r1*b**2-2*r1*c**2-(2+3*x)*r1*a*b+x*r1*a*c+x*r1*b*c+x*r1*c**2)**2),
                            CyclicSum(a)**2 * CyclicProduct((a-b)**2),
                        ]

                        if w2 != 0 and y[-1] + (x**2 - 4*x - 8) / 8 * w2 >= 0:
                            y[-1] += (x**2 - 4*x - 8) / 4 * w2
                            y.append(w2 / 4 / r1**2)
                            exprs.append(
                                CyclicSum(c**2*(a-b)**2*(2*r1*a**2+2*r1*b**2-2*r1*c**2+(3*x-2)*r1*a*b-x*r1*a*c-x*r1*b*c-x*r1*c**2)**2)
                            )

                        elif w2 != 0: # in this case we must have x**2 - 4*x - 8 <= 0:
                            r2 = 1 / cancel_denominator([x+2, x-2, 6-5*x, 2*x])
                            y += [
                                (x - 2)*(5*x + 6)/(4*(x + 2)*(x + 10)) * w2 / r1**2,
                                -(x**2 - 4*x - 8)/(4*(x + 2)*(x + 10)) * w2 / r2**2
                            ]
                            exprs += [
                                CyclicSum(c**2*(a-b)**2*(2*r1*a**2+2*r1*b**2-2*r1*c**2+(3*x-2)*r1*a*b-x*r1*a*c-x*r1*b*c-x*r1*c**2)**2),
                                CyclicSum((a-b)**2*(-r2*(x+2)*a**2*b+r2*(x-2)*a**2*c-r2*(x+2)*a*b**2+r2*(-5*x+6)*a*b*c
                                    +r2*2*x*a*c**2+r2*(x-2)*b**2*c+r2*2*x*b*c**2+r2*(x+2)*c**3)**2)
                            ]


                        if all(_ >= 0 for _ in y):
                            for i in range(2, len(y)):
                                y[i] *= coeff42 * 4 / x**2
                            y.append(rem)
                            exprs.append(CyclicProduct(a**2) * multiplier)
                            return _sum_y_exprs(y, exprs) / multiplier


        elif u < v:
            # subtracting p(a-b)2 will first lead to s(a4b2+2a4bc+a4c2)
            # assume coeffp = coefficient of p(a-b)2
            # (coeff42 - coeffp) * (2) = v + 2 * coeffp
            coeffp = (coeff42 * 2 - v) / 4
            coeff42 -= coeffp
            u, v = (u + coeffp * 2) / coeff42, sp.S(2)
            
            
            # now we need to ensure (u + 2) is a square, and make a perturbation otherwise
            # assume coeffsym = coefficient of s(a2b+ab2-2abc)2
            # (u - 2 * coeffsym / coeff42) / (1 - coeffsym / coeff42) + 2 = x*x <= 4
            # print(v, v.n(20), coeffp)
            coeffsym, x = sp.S(0), sp.sqrt(u + 2)
            if not isinstance(x, sp.Rational):
                for x_ in rationalize_bound(x.n(20), direction = -1, compulsory = True):
                    if abs(x_) == 2:
                        continue

                    coeffsym = ((u + 2 - x_**2) / (4 - x_**2)) * coeff42
                    coeff321_std = (-x_**2 - 2*x_ - 2)
                    coeff321 = (coeff((3,2,1)) - 2 * coeffp + 10 * coeffsym) / (coeff42 - coeffsym)
                    if 0 < x_ <= 1 and 0 <= coeff321 - coeff321_std <= 4 * x_:
                        x = x_
                        break
                    elif 1 < x_ <= 2 and coeff321 == coeff321_std:
                        x = x_
                        break
                else:
                    x_ = None

            coeff42 -= coeffsym

            if x is not None and isinstance(x, sp.Rational):
                x = x / 2
                # Theorem 1.5:
                # When x \in [-1/2, 1],
                # f(a,b,c) = s(bc(a-b)(a-c)(a-xb)(a-xc)) + 1/4p(a-b)^2 >= 0 for all real numbers a,b,c
                # because
                # 4f(a,b,c)s((a-b)(a-c)) = (1-2x)/2 * s((a-b)(a-c)(2xbc-ab-ac))^2 + (1+2x)/2 * s((a-b)^2(a-c)^2(2xbc-ab-ac)^2)
                #                       >= (2-2x)/3 * s((a-b)(a-c)(2xbc-ab-ac))^2 >= 0
                # The second line is due to the fact u^2+v^2+w^2 >= (u+v+w)^2 / 3.
                if x == 0:
                    # must be the case s(bc(4a2+(b-c)2)(a-b)(a-c))
                    # Solution:
                    # s(bc(4a2+(b-c)2)(a-b)(a-c))s((a-b)^2) = s((a-b)4(ab+ac+bc-c2)2)
                    
                    if (coeff((3,2,1)) - 2 * coeffp + 10 * coeffsym) == -2 * coeff42:
                        y = [coeffp, coeffsym, coeff42, rem]
                        multiplier = CyclicSum((a-b)**2)
                        exprs = [
                            CyclicProduct((a-b)**2) * CyclicSum((a-b)**2),
                            CyclicSum(a*(b-c)**2)**2 * CyclicSum((a-b)**2),
                            CyclicSum((a-b)**4 * (a*b+a*c+b*c-c**2)**2),
                            CyclicProduct(a**2) * multiplier
                        ]
                        if all(_ >= 0 for _ in y):
                            return _sum_y_exprs(y, exprs) / multiplier

                else:
                    # linear combination of x and -x
                    w2 = ((coeff((3,2,1)) - 2 * coeffp + 10 * coeffsym) / coeff42 - (-4*x**2-4*x-2)) / (8*x)
                    w1 = 1 - w2
                    if -sp.S(1)/2 <= x <= 1 and 0 <= w1 <= 1:
                        multiplier = CyclicSum((a-b)**2)
                        y = [
                            coeffp,
                            coeffsym
                        ]
                        exprs = [
                            CyclicProduct((a-b)**2) * CyclicSum((a-b)**2),
                            CyclicSum(a*(b-c)**2)**2 * CyclicSum((a-b)**2)
                        ]
                        def _extend_solution(y, exprs, weight, x):
                            if x <= sp.S(1)/2:
                                y.extend([
                                    weight * (1 - 2*x) / 2,
                                    weight * (1 + 2*x) / 2
                                ])
                                exprs.extend([
                                    CyclicSum((2*x*b*c-a*b-a*c)*(a-b)*(a-c))**2,
                                    CyclicSum((2*x*b*c-a*b-a*c)**2*(a-b)**2*(a-c)**2),
                                ])
                            elif x <= 1:
                                y.extend([
                                    weight * (2 - 2*x) / 3,
                                    weight * (1 + 2*x) / 6
                                ])
                                exprs.extend([
                                    CyclicSum((2*x*b*c-a*b-a*c)*(a-b)*(a-c))**2,
                                    CyclicSum((a-b)**2 * (a**2*b + a**2*c + a*b**2 + (-4*x - 2)*a*b*c + (2*x - 1)*a*c**2 + b**2*c + (2*x - 1)*b*c**2)**2),
                                ])
                        
                        _extend_solution(y, exprs, w1, x)
                        _extend_solution(y, exprs, w2, -x)

                        if all(_ >= 0 for _ in y):
                            for i in range(2, len(y)):
                                y[i] *= coeff42 * 2
                            y.append(rem)
                            exprs.append(CyclicProduct(a**2) * multiplier)
                            return _sum_y_exprs(y, exprs) / multiplier

    if rem > 0:
        if coeff((3,3,0)) == coeff((4,1,1)):
            # try c1p(a-b)2 + c2s(a2b-xabc)2 + c2s(ab2-xabc)2 + c3p(a2)
            c1 = coeff((3,3,0)) / (-2)
            c2 = coeff((4,2,0)) - c1
            x_ = (2 - (coeff((3,2,1)) - 2*c1) / c2) / 6
            y = [
                c1,
                c2,
                c2,
                rem - 18*c2*(x_ - 1)**2
            ]
            if all(_ >= 0 for _ in y):
                exprs = [
                    CyclicProduct((a-b)**2),
                    (a**2*b + b**2*c + c**2*a - (3*x_)*a*b*c)**2,
                    (a**2*c + b**2*a + c**2*b - (3*x_)*a*b*c)**2,
                    CyclicProduct(a**2)
                ]
                return _sum_y_exprs(y, exprs)



    if True:
        # subtract p(a-b)2
        def new_coeff(x):
            t = coeff((4,2,0))
            coeffs = {
                (3,3,0): coeff((3,3,0)) + 2 * t,
                (4,1,1): coeff((4,1,1)) + 2 * t,
                (3,2,1): coeff((3,2,1)) - 2 * t,
                (2,3,1): coeff((2,3,1)) - 2 * t,
                (2,2,2): coeff((2,2,2)) + 6 * t
            }
            v = coeffs.get(x)
            return v if v is not None else 0

        solution = _sos_struct_sextic_hexagram_symmetric(new_coeff)
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
        y = [
            x_,
            coeff((3,3,0)) - x_,
            coeff((4,1,1)) - x_,
            coeff((3,2,1)) + x_ + (coeff((3,3,0))) + (coeff((4,1,1))),
            (coeff((3,3,0)) + coeff((4,1,1)) + coeff((3,2,1)) * 2) * 3 + coeff((2,2,2))
        ]
        if all(_ >= 0 for _ in y):
            exprs = [
                CyclicSum(b*c*(a-b)**2*(a-c)**2),
                CyclicSum(b**2*c**2*(a-b)*(a-c)),
                CyclicSum(a*(a-b)*(a-c)) * CyclicProduct(a),
                CyclicSum(a*(b-c)**2) * CyclicProduct(a),
                CyclicProduct(a**2)
            ]
            return _sum_y_exprs(y, exprs)
    
    if coeff((4,1,1)) != 0:
        x_ = coeff((3,3,0)) / coeff((4,1,1))
        y_ = coeff((3,2,1)) / coeff((4,1,1))
        w_ = coeff((2,2,2)) / coeff((4,1,1))
        if False:
            u_ = sp.sqrt(x_) + 1
            y_schur = sp.S(0)
            if not isinstance(u_, sp.Rational):
                # subtract a small perturbation
                u_ = u_.n(15)
                if y_ >= -(u_**2 - u_ + 1):
                    for u_ in rationalize_bound(u_, direction = -1, compulsory = True):
                        y_schur = x_ - (u_ - 1)**2
                        y2 = y_ + y_schur
                        if y_schur >= 0 and u_ > 1 and y2 >= -(u_**2 - u_ + 1):
                            x_ -= y_schur
                            y_ = y2
                            break
                        u_ = None

            # now that u is Rational
            if isinstance(u_, sp.Rational) and u_ != 1:
                # abcs((b-c)2(b+c-ua)2)+2s(a(b-c)2((1-u)(ab+ac)+bcu)2)
                r = u_.as_numer_denom()[1] # cancel the denominator is good
                y = [
                    sp.S(1) / r / r,
                    sp.S(2) / r / r,
                    y_schur,
                    y_ + (u_**2 - u_ + 1),
                    ((coeff((3,3,0)) + coeff((4,1,1)) + coeff((3,2,1)) * 2) + coeff((2,2,2)) / 3) / coeff((4,1,1)) * 3
                ]
                if any(_ < 0 for _ in y):
                    y = None
                else:
                    multipliers = ['a']
                    y = [_ * coeff((4,1,1)) for _ in y]
                    exprs = [
                        f'a*b*c*(b-c)^2*({r}*b+{r}*c-{r*u_}*a)^2',
                        f'a*(b-c)^2*({r*(1-u_)}*a*b+{r*(1-u_)}*a*c+{r*u_}*b*c)^2',
                        'b^2*c^2*(a-b)*(a-c)*(a+b+c)',
                        'a^2*b*c*(b-c)^2*(a+b+c)',
                        'a^3*b^2*c^2'
                    ]
                    None
        if x_ >= (1 + x_ + y_)**2:
            # apply theorem 1
            # use vieta jumping, a point inside the parabola is a linear combination
            # of u = 1 and u = (y + 1) / (x + y + 1)
            u_ = (y_ + 1) / (x_ + y_ + 1)

            # weights of linear combination
            w2 = x_ / (u_ - 1)**2
            w1 = 1 - w2

            # NOTE: the case x + y + 1 == 0 has been handled in the trivial case

            # abcs((b-c)2(b+c-ua)2)+2s(a(b-c)2((1-u)(ab+ac)+bcu)2)
            r = u_.as_numer_denom()[1] # cancel the denominator is good

            y = [
                w1 / 2,
                w1,
                w2 / r**2 / 2,
                w2 / r**2,
                ((coeff((3,3,0)) + coeff((4,1,1)) + coeff((3,2,1)) * 2) + coeff((2,2,2)) / 3) / coeff((4,1,1)) * 3
            ]
            if any(_ < 0 for _ in y):
                y = None
            else:
                multiplier = CyclicSum(a)
                y = [_ * coeff((4,1,1)) for _ in y]
                exprs = [
                    CyclicSum((b-c)**2*(b+c-a)**2) * CyclicProduct(a),
                    CyclicSum(b*c*(b-c)**2) * CyclicProduct(a),
                    CyclicSum((b-c)**2*(r*b+r*c-r*u_*a)**2) * CyclicProduct(a),
                    CyclicSum(a*(b-c)**2*(r*(1-u_)*a*b+r*(1-u_)*a*c+r*u_*b*c)**2),
                    CyclicSum(a) * CyclicProduct(a**2)
                ]
                # print(y, exprs)
                return _sum_y_exprs(y, exprs) / multiplier



    if coeff((3,3,0)) != 0:
        # https://tieba.baidu.com/p/8039371307
        x_, y_, z_ = coeff((4,1,1)) / coeff((3,3,0)), -coeff((3,2,1)) / coeff((3,3,0)), coeff((2,2,2)) / coeff((3,3,0))
        z0 = x_**2 + x_*y_ + y_**2/3 - y_ + (y_ + 3)**3/(27*x_)
        if x_ > 0 and 3 * x_ + y_ + 3 >= 0 and z_ >= z0:
            ker = 324 * x_ * (27*x_**3 + 27*x_**2*y_ + 81*x_**2 + 9*x_*y_**2 - 189*x_*y_ + 81*x_ + y_**3 + 9*y_**2 + 27*y_ + 27)
            if ker > 0:
                w1 = -(9*x_**2 + 6*x_*y_ - 306*x_ + y_**2 + 6*y_ + 9) / ker
                if w1 > 0:
                    w2 = 1 / ker
                    phi2 = (36*x_**2 + 15*x_*y_ - 117*x_ + y_**2 + 6*y_ + 9)
                    phi1 = (9*x_**2 + 6*x_*y_ - 117*x_ + y_**2 + 15*y_ + 36)

                    multiplier = CyclicSum(a) * CyclicSum(a*b)
                    y = [w1, w2, 3*(z_ - z0)]
                    y = [_ * coeff((3,3,0)) for _ in y]

                    exprs = [
                        CyclicSum(c*((-9*x_**2-3*x_*y_+18*x_)*a**3*b+(-9*x_**2+9*x_+y_**2-9)*a**2*b**2+(9*x_**2+3*x_*y_-3*y_-9)*a**2*b*c
                                    +(-18*x_+3*y_+9)*a**2*c**2+(-9*x_**2-3*x_*y_+18*x_)*a*b**3+(9*x_**2+3*x_*y_-3*y_-9)*a*b**2*c
                                    +(9*x_**2-9*x_-y_**2+9)*a*b*c**2+(-18*x_+3*y_+9)*b**2*c**2)**2),
                        CyclicSum(c*((-3*phi1*x_)*a**3*b+(-3*phi1*x_+phi1*y_+3*phi1-3*phi2)*a**2*b**2+(-3*phi1*x_+phi1*y_+3*phi1+3*phi2*x_+phi2*y_-3*phi2)*a**2*b*c
                                    +(-3*phi2)*a**2*c**2+(-3*phi1*x_)*a*b**3+(-3*phi1*x_+phi1*y_+3*phi1+3*phi2*x_+phi2*y_-3*phi2)*a*b**2*c
                                    +(-3*phi1*x_+3*phi2*x_+phi2*y_-3*phi2)*a*b*c**2+(-3*phi2)*b**2*c**2)**2),
                        multiplier * CyclicProduct(a**2)
                    ]
    
                    return _sum_y_exprs(y, exprs) / multiplier
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
    # t != 0 by assumption
    u = coeff((3,3,0))/t
    v = coeff((4,1,1))/t
    if u < -2:
        return None

    if v != -6 and u != 2:
        # try sum of squares with real numbers first
        # if (u,v) falls inside the parametric curve (x^3-3x,-3x(x-1)) where -1<=x<=2,
        # then it is a rational linear combination of (t^3-3t, -3t(t-1)) and (2, -6)
        # with t = -(3u + v) / (v + 6)
        # note: (2, -6) is the singular node of the strophoid
        t__ = -(3*u + v) / (v + 6)
        if -1 <= t__ <= 2:
            x = t__**3 - 3*t__
            w1 = (27*u**2 + 27*u*v + 54*u + v**3 + 18*v**2 + 54*v)/(27*(u - 2)*(u + v + 4))
            w2 = 1 - w1
            q, p = t__.as_numer_denom()
            if 0 <= w1 <= 1:
                y = [w1 * t / 2, w2 * t / 2 / p**3,
                    (coeff((2,2,2))/3+coeff((6,0,0))+coeff((4,1,1))+coeff((3,3,0))) * 3]
                exprs = [
                    CyclicSum(a)**2 * CyclicSum((b-c)**4),
                    CyclicSum(p*a**2 + q*b*c) * CyclicSum((a-b)**2*(p*a+p*b-q*c)**2),
                    CyclicProduct(a**2),
                ]
                return _sum_y_exprs(y, exprs)
    
    x = sp.symbols('x')
    roots = sp.polys.roots(x**3 - 3*x - u).keys()
    for r in roots:
        numer_r = complex(r)
        if abs(numer_r.imag) < 1e-12 and numer_r.real >= 1:
            numer_r = numer_r.real
            if not isinstance(r, sp.Rational):
                for r in rationalize_bound(numer_r, direction = -1, compulsory = True):
                    if r**3-3*r <= u and 3*r*(r-1)+v >= 0:
                        break
                else:
                    break
            elif 3*r*(r-1)+v < 0:
                break
            
            # now r is rational
            y = [t/2, t*(u-(r*r*r-3*r))/2, t*(v+3*r*(r-1))/2,
                (coeff((2,2,2))/3+coeff((6,0,0))+coeff((4,1,1))+coeff((3,3,0))) * 3]
            exprs = [
                CyclicSum(a**2 + r*b*c) * CyclicSum((a-b)**2*(a+b-r*c)**2),
                CyclicSum(a*b) * CyclicSum(a**2*(b-c)**2),
                CyclicSum(a) * CyclicSum((a-b)**2) * CyclicProduct(a),
                CyclicProduct(a**2),
            ]
            if r == 2:
                exprs[0] = '(a-b)^2*(a+b-2*c)^2*(a+b+c)^2'
                exprs[0] = CyclicSum(a)**2 * CyclicSum((a-b)**2*(a+b-2*c)**2)

            return _sum_y_exprs(y, exprs)

    return None



def _sos_struct_sextic_iran96(coeff, real = False):
    """
    Give a solution to s(a5b+ab5-x(a4b2+a2b4)+ya3b3-za4bc+w(a3b2c+a2b3c)+..a2b2c2) >= 0

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
    """
    if not (coeff((6,0,0)) == 0 and coeff((5,1,0)) == coeff((1,5,0)) and coeff((4,2,0)) == coeff((2,4,0)) and\
        coeff((3,2,1)) == coeff((2,3,1)) and coeff((5,1,0)) >= 0):
        return None

    m, p, q, w, z = coeff((5,1,0)), coeff((4,2,0)), coeff((3,3,0)), coeff((4,1,1)), coeff((3,2,1))

    if m < 0:
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
            coeff((2,2,2)) + 3 *((m + p + z) * 2 + q + w)
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
            return _sum_y_exprs(y, exprs)


    if p >= -4 * m and q + 2 * (p + m) >= 0:
        # Easy case 2, subtract enough p(a-b)2 and s(ab(a-c)2(b-c)2)
        # e.g. s(a(b+c)(b+c-2a)4)
        y = [
            m,
            (p + 4 * m),
            q + 2 * (p + m),
            w + 2 * (p + 4 * m) - (q + 2 * (p + m)),
            4*m + 4*p + 2*q + w + z,
            coeff((2,2,2)) + 3*((m + p + z) * 2 + q + w)
        ]
        
        if all(_ >= 0 for _ in y):
            exprs = [
                CyclicSum(a*b*(a-b)**4),
                CyclicProduct((a-b)**2),
                CyclicSum(b*c*(a-b)**2*(a-c)**2),
                CyclicSum(a*(a-b)*(a-c)) * CyclicProduct(a),
                CyclicSum(a*(b-c)**2) * CyclicProduct(a),
                CyclicProduct(a**2)
            ]
            return _sum_y_exprs(y, exprs)


    if p >= 0 and q + 2 * m + 2 * p >= 0:
        # Easy case 2, when we do not need to updegree

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
                z - u_ * (u_ + 2) * m - 2*p + (w2 + q2 - 2 * min(w2, q2)),
                coeff((2,2,2)) + 3 *((m + p + z) * 2 + q + w)
            ]

            if all(_ >= 0 for _ in y):
                exprs = [
                    CyclicSum(a*b*(a-b)**2*(a+b-u_*c)**2),
                    CyclicProduct((a-b)**2),
                    CyclicSum(b*c*(a-b)**2*(a-c)**2),
                    CyclicSum(b**2*c**2*(a-b)*(a-c)),
                    CyclicSum(a*(a-b)*(a-c)) * CyclicProduct(a),
                    CyclicSum(a*(b-c)**2) * CyclicProduct(a),
                    CyclicProduct(a**2)
                ]
                return _sum_y_exprs(y, exprs)


    if True:
        # Easy case 4, when we can extract some s(ab) * quartic
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
        
        if v_ <= m+q-(p-u_)**2/m+2*u_:
            pass
        elif m + p >= u_: # symmetric axis of the parabola >= u_
            u_ = m + p
            v_ = max(sp.S(0), (-u_+p-w-z)/2)

        if u_ >= 0 and 0 <= v_ <= m+q-(p-u_)**2/m+2*u_ and v_ <= 4*u_+w-m-2*p and v_ >= (-u_+p-w-z)/2:
            y = [
                m / 2,
                (m+q-(p-u_)**2/m+2*u_ - v_)/2,
                u_,
                v_,
                -m - 2*p + 4*u_ - v_ + w,
                -p + u_ + 2*v_ + w + z,
                coeff((2,2,2)) + 3*((m + p + z) * 2 + q + w)
            ]

            if all(_ >= 0 for _ in y):
                exprs = [
                    CyclicSum(a*b) * CyclicSum((a-b)**2*(a+b-(-p+u_)/m*c)**2),
                    CyclicSum(a*b) * CyclicSum(a**2*(b-c)**2),
                    CyclicProduct((a-b)**2),
                    CyclicSum(b*c*(a-b)**2*(a-c)**2),
                    CyclicSum(a*(a-b)*(a-c)) * CyclicProduct(a),
                    CyclicSum(a*(b-c)**2) * CyclicProduct(a),
                    CyclicProduct(a**2)
                ]
                return _sum_y_exprs(y, exprs)


    # real start below
    # process:
    p, q, w, z = p / m, q / m, w / m, z / m

    if not ((p <= -4 and q >= p*p/4 + 2) or (p >= -4 and q >= -2*(p + 1))):
        # checking the border yields that:
        # when p <= -4, we must have q >= p*p / 4 + 2
        # when p >= -4, we must have q >= -2*(p + 1)

        # both two cases should imply 2*p + q + 2 >= 0
        return None
    
    # First, we peek whether there are nontrivial roots in the interior with a == b.
    u = sp.symbols('u')
    sym = ((2*p + q + 2)*a**3 + (4*p + 2*q + 2*w + 2*z + 6)*a**2 + (2*p + w + 4)*a + 2).as_poly(a)
    # sym should be nonnegative when a >= 0
    sym_roots_count = sp.polys.count_roots(sym, 0, None)
    root, u_ = None, None
    if sym_roots_count > 1:
        return None
    elif sym_roots_count == 1:
        # yes there are nontrivial roots
        root = list(filter(lambda x: x >= 0, sp.polys.roots(sym).keys()))[0]
        if root != 1:
            u_ = 1 / root + 1
        else:
            root = None

    # Second, determine t by coefficient at (4,2,0) and (3,3,0)
    # this is done by subtracting as much ((a-b)*(b-c)*(c-a))^2 as possible
    # until there are zeros at the border

    # subtract some ((a-b)*(b-c)*(c-a))^2
    # p - r == 2*t,   q + 2*r == t*t + 2
    # r = p + 4 + 2 * sqrt(2*p + q + 2)

    # Case A. r is irrational, instead we subtract some hexagrams
    r = p + 4 + 2 * sp.sqrt(2 * p + q + 2)
    y_hex = 0
    if not isinstance(r, sp.Rational):
        # make a perturbation on q so that 2*p + q' + 2 is a square
        
        if u_ is None:
            # Case A.A there are no nontrivial roots, then we can make any slight perturbation
            # here we use s(ab(a-c)2(b-c)2)
            dw = 1
            dz = -3
        else:
            # Case A.B there exists nontrivial roots, then we make a slight perturbation
            # using the hexagram generated by the root
            dw = 1 / (u_ - 1)**2
            dz = (-u_**2 + u_ - 1) / (u_ - 1)**2
        
        numer_r = sp.sqrt(2 * p + q + 2).n(20)
        for numer_r2 in rationalize_bound(numer_r, direction = -1, compulsory = True):
            if numer_r2 >= 0 and p + 4 + 2 * numer_r2 >= 0:
                q2 = numer_r2 ** 2 - 2 * p - 2
                y_hex = q - q2
                w2 = w - dw * y_hex
                z2 = z - dz * y_hex
                if y_hex >= 0:
                    sym = ((2*p + q2 + 2)*a**3 + (4*p + 2*q2 + 2*w2 + 2*z2 + 6)*a**2 + (2*p + w2 + 4)*a + 2).as_poly(a)
                    if sp.polys.count_roots(sym, 0, None) <= 1:
                        q = q2
                        break
        else:
            return None

        w -= dw * y_hex
        z -= dz * y_hex
        r = p + 4 + 2 * sp.sqrt(2 * p + q + 2)

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
            
    # print('W Z R Y U T=', w, z, r, y_hex, u_, t)
    if u_ is None:
        return None

    if u_ != 1:
        # rather u_ in the introduction, we use u_ + 1 here as u_
        phi = (t * u_ - u_**2 + 2*u_ - 3) / (u_ - 1)

        multiplier = CyclicSum(a)
        y = [
            y_hex if root is None else sp.S(0),
            r,
            (w - (t**2 + 4*t*u_**2 - 8*t*u_ - 4*u_**3 + 8*u_**2 - 4*u_ + 4)/(u_ - 1)**2) / 2,
            z - coeff_z(u_),
            (coeff((2,2,2)) / 3 + (coeff((5,1,0)) + coeff((4,2,0)) + coeff((3,2,1))) * 2 \
                + coeff((3,3,0)) + coeff((4,1,1))) / m * 3,
            sp.S(1),
            (t - 2*u_) ** 2 / 2 / (u_ - 1)**2 + (sp.S(0) if root is None else y_hex / 2 / (u_ - 1)**2),
            sp.S(0) if root is None else 1 / (u_ - 1)**2 * y_hex
        ]
        # print(r, t, u_, phi, y)

        if any(_ < 0 for _ in y):
            return None

        y = [_ * m for _ in y]
        exprs = [
            CyclicSum(a) * CyclicSum(a*b*(a-c)**2*(b-c)**2),
            CyclicSum(a) * CyclicProduct((a-b)**2),
            CyclicSum(a)**2 * CyclicSum((a-b)**2) * CyclicProduct(a),
            CyclicSum(a) * CyclicSum(a*(b-c)**2) * CyclicProduct(a),
            CyclicSum(a) * CyclicProduct(a**2),
            CyclicSum(a*(b-c)**2*(a**2+b**2+c**2-t*a*b-t*a*c+phi*b*c)**2),
            CyclicSum((b-c)**2*(b+c-u_*a)**2) * CyclicProduct(a),
            CyclicSum(c*(a-b)**2*(u_*a*b-(u_-1)*a*c-(u_-1)*b*c)**2)
        ]
        return _sum_y_exprs(y, exprs) / multiplier

    elif u_ == 1:
        # very special case, it must be t == 2
        # f(a,b,c) = (s(ab(a-b)2(a+b-c)2)-4p(a-b)2)
        # then f(a,b,c)s(a) = s(a(b-c)2(b+c-a)4) + 2abcs((b-c)2(b+c-a)2)
        
        multiplier = CyclicSum(a)
        y = [
            y_hex if root is None else sp.S(0),
            r,
            w - 4,
            z + w + 1,
            (coeff((2,2,2)) / 3 + (coeff((5,1,0)) + coeff((4,2,0)) + coeff((3,2,1))) * 2 \
                + coeff((3,3,0)) + coeff((4,1,1))) / m * 3,
            sp.S(1),
            sp.S(2)
        ]

        if any(_ < 0 for _ in y):
            return None

        y = [_ * m for _ in y]
        exprs = [
            CyclicSum(a) * CyclicSum(a*b*(a-c)**2*(b-c)**2),
            CyclicSum(a) * CyclicProduct((a-b)**2),
            CyclicSum(a) * CyclicSum(a*(a-b)*(a-c)) * CyclicProduct(a),
            CyclicSum(a) * CyclicSum(a*(b-c)**2) * CyclicProduct(a),
            CyclicSum(a) * CyclicProduct(a**2),
            CyclicSum(a*(b-c)**2*(b+c-a)**4),
            CyclicSum((b-c)**2*(b+c-a)**2) * CyclicProduct(a)
        ]
        return _sum_y_exprs(y, exprs) / multiplier

    return None



def sos_struct_sextic_symmetric_ultimate(poly, coeff, recurrsion):
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
        return _sos_struct_sextic_iran96(coeff, real = True)
    elif coeff6 < 0:
        return None

    if coeff((5,1,0)) == 0 and coeff((4,2,0)) == 0 and coeff((3,2,1)) == 0 and coeff6 != 0:
        return _sos_struct_sextic_tree(coeff)


    u, v, w, x, z = sp.symbols('u v w x z')

    # Detect Roots
    roots = [None, None, None]

    # Case A.
    eq = coeff6 * (x**6 + 1) + coeff((5,1,0)) * (x**5 + x) + coeff((4,2,0)) * (x**4 + x*x) + coeff((3,3,0)) * x**3
    eq = eq.as_poly(x)
    eqdiff = eq.diff(x)
    eqdiff2 = eqdiff.diff(x)
    for r in sp.polys.roots(eq, cubics = False, quartics = False):
        if r.is_real and 0 < r <= 1:
            if isinstance(r, sp.Rational) and eqdiff(r) == 0 and eqdiff2(r) >= 0:
                roots[0] = r
                break
            elif abs(eqdiff(r.n(20))) * 10**12 < coeff6 and eqdiff2(r.n(20)) >= 0:
                roots[0] = r
                break
            else:
                return None
            
    # Case B.
    eq = poly.subs('c', 1).subs('b', a).as_poly(a)
    eqdiff = eq.diff(a)
    eqdiff2 = eqdiff.diff(a)
    for r in sp.polys.roots(eq, cubics = False, quartics = False):
        if r.is_real and r > 0 and r != 1:
            if isinstance(r, sp.Rational) and eqdiff(r) == 0 and eqdiff2(r) >= 0:
                roots[1] = r
                break
            else:
                # not locally convex
                return None
    if roots[1] is None:
        if eq(1) == 0 and eqdiff(1) == 0:
            if eq.degree() == 2:
                roots[1] = sp.oo
            elif eqdiff2(1) == 0:
                roots[1] = 1
        if roots[0] is None and roots[1] is None:
            for r in sp.polys.roots(eq, cubics = False, quartics = False):
                if r.is_real and r < 0:
                    if isinstance(r, sp.Rational) and eqdiff(r) == 0 and eqdiff2(r) >= 0:
                        roots[1] = r
                        break

    # Case C.
  
    print('Roots Info = ', roots)
    sum_of_roots = sum(_ is not None for _ in roots)

    if sum_of_roots == 1:
        return _sos_struct_sextic_symmetric_ultimate_1root(coeff, poly, recurrsion, roots)
    elif sum_of_roots == 2:
        return _sos_struct_sextic_symmetric_ultimate_2roots(coeff, poly, recurrsion, roots)

    return None

def _sos_struct_sextic_symmetric_ultimate_1root(coeff, poly, recurrsion, roots):
    """
    Examples
    -------
    Case A.
        s(a2)3-27(abc)2-27p((a-b)2)
        
        s(a2/3)3-a2b2c2-p(a-b)2

        s(4a6-a3b3-3a2b2c2)-63p(a-b)2

        4s(a4(a-b)(a-c))+s(a(a-b)(a-c))2
    """
    coeff6 = coeff((6,0,0))
    if roots[0] is not None:
        # Case A.
        # subtract some s(a3-abc-x(a2b+ab2-2abc))2
        x_ = sp.simplify(roots[0] + 1/roots[0]) - 1
        if not isinstance(x_, sp.Rational):
            return None

        # 1. try subtracting all the s(a6)
        # e.g. s(a2/3)3-a2b2c2-p(a-b)2
        if coeff((5,1,0)) >= -2 * x_:
            poly2 = poly - ((a**3+b**3+c**3-3*a*b*c-x_*(a*a*(b+c)+b*b*(c+a)+c*c*(a+b)-6*a*b*c))**2).as_poly(a,b,c) * coeff6
            coeff2 = _make_coeffs(poly2)
            solution = _sos_struct_sextic_iran96(coeff2)
            if solution is not None:
                if x_ == sp.S(3)/2:
                    solution += coeff6 / 4 * CyclicProduct((a+b-2*c)**2)
                elif x_ == 1:
                    solution += coeff6 * CyclicSum(a*(a-b)*(a-c))**2
                else:
                    solution += coeff6 * CyclicSum(a**3-x_*a**2*(b+c)+(2*x_-1)*a*b*c)**2

                return solution

        # until the symmetric axis is touched
        # # the subtractor = (a-1)^4 * (2(x-1)a - 1)^2 on the symmetric axis a == b and c == 1
        # sym = poly.subs(c,1).subs(b,a).factor() / (a - 1)**4 / (2*(x-1)*a - 1)**2


    return None


def _sos_struct_sextic_symmetric_ultimate_2roots(coeff, poly, recurrsion, roots):
    """

    Examples
    --------
    Case (A+B)

    s((a-b-c)4a-abc(3a-b-c)2)s(a)-(s(ab(a2-b2+3(ab-ac)+3(bc-ab))2)-4p(a-b)2)

    s(4a6-6(a5b+a5c)-12(a4b2+a4c2)+37a4bc+28a3b3-31(a3b2c+a3bc2)+29a2b2c2)

    s(a4(a-b)(a-c)) - 5p(a-b)2

    s((b-c)2(7a2-b2-c2)2)-112p(a-b)2
    
    s((a-b)(a-c)(a-2b)(a-2c)(a-18b)(a-18c))-53p(a-b)2

    s((b2+c2-5a(b+c))2(b-c)2)-22p(a-b)2 

    s(a6+6a5b+6a5c-93a4b2+3a4bc-93a4c2+236a3b3+87a3b2c+87a3bc2-240a2b2c2)

    s(a6-21a5b-21a5c-525a4b2+1731a4bc-525a4c2+11090a3b3-13710a3b2c-13710a3bc2+15690a2b2c2)
    
    Reference
    -------
    [1] Vasile, Mathematical Inequalities Volume 1 - Symmetric Polynomial Inequalities. 3.78
    
    [2] https://artofproblemsolving.com/community/c6t243f6h3013463_symmetric_inequality

    [3] https://tieba.baidu.com/p/8261574122
    """
    coeff6 = coeff((6,0,0))

    
    if roots[2] is None:
        # Case (A + B)
        if roots[0] != 1:
            x_ = sp.simplify(roots[0] + 1/roots[0]) - 1
            if not isinstance(x_, sp.Rational):
                return None
            solution = CyclicSum(a**3-x_*a**2*(b+c)+(2*x_-1)*a*b*c)**2
            diffpoly = solution.doit().as_poly(a,b,c)
            solution *= coeff6

        elif roots[0] == 1:
            x = 1 / roots[1]
            if x > 4:
                return None

            # Theorem: when x <= 4
            # f(a,b,c) = s(a6-(x+1)(a5b+a5c)+(4x-5)(a4b2+a4c2)+(x2-4x+11)a4bc
            #               -2(3x-5)a3b3+(-x2+5x-10)(a3b2c+a3bc2)+(x2-6x+10)a2b2c2) >= 0
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
                # easy case, no need to updegree
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
                solution = _sum_y_exprs(y, exprs) / multiplier

        new_poly = poly - coeff6 * diffpoly
        new_coeff = _make_coeffs(new_poly)
        rest_solution = _sos_struct_sextic_iran96(new_coeff)
        if rest_solution is not None:
            f1, f2 = sp.fraction(sp.together(solution + rest_solution))
            f1 = sp.collect(f1, CyclicProduct((a-b)**2))
            return f1 / f2

    elif roots[0] is None:
        # Case (B + C)
        pass

    return None