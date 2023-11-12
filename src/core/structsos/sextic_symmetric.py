import sympy as sp

from .quartic import sos_struct_quartic
from .utils import (
    CyclicSum, CyclicProduct, Coeff, 
    sum_y_exprs, nroots, rationalize_bound, cancel_denominator, radsimp,
    prove_univariate
)

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

    8s(a2)3-9p(a+b)s(a3)-2s((b-c)2(b+c-1/8a))2    (real)

    s(bc(a-b)(a-c)(a-3b)(a-3c)) +1/4s(a2b+ab2-2abc)2+5p(a-b)2+4abcs(a(b-c)2)    (real)

    s(bc(a-b)(a-c)(a-3b)(a-3c))+9/4p(a-b)2    (real)

    s(bc(a-b)(a-c)(a-9/8b)(a-9/8c))+81/256p(a-b)2    (real)

    s(4a4b2-7a4bc+4a4c2+8a3b3-12a3b2c-12a3bc2+15a2b2c2)   (real)

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

    rem = radsimp((coeff((4,2,0)) + coeff((3,2,1))) * 6 + (coeff((3,3,0)) + coeff((4,1,1))) * 3 + coeff((2,2,2)))
    if rem < 0:
        return None
    
    # although subtracting p(a-b)2 always succeeds,
    # we can handle cases for real numbers and cases where highering the degree is not necessary
    if True:
        # if 3*(coeff((4,2,0))*2 + coeff((3,3,0)) + coeff((4,1,1)) + coeff((3,2,1))*2) + coeff((2,2,2)) == 0:
        # has root at (1,1,1)
        x_, y_, z_ = coeff((3,3,0)) / coeff((4,2,0)), coeff((4,1,1)) / coeff((4,2,0)), -coeff((3,2,1)) / coeff((4,2,0))
        
        if x_ < -2 or y_ < -2:
            return None
            
        if abs(x_ - y_) < 2:
            # Case 1. Try linear combination of s(a(b-c)^2)^2, p(a-b)^2, p(a)s(a(b-c)^2)
            # and s(a^4(b-c)^2) or s(a^2b^2(a-b)^2)
            # the last s(a^4(b-c)^2) or s(a^2b^2(a-b)^2) balance the coeffs of a^3b^3 and a^4bc
            # where 2u - 2(1-u) = x  =>  u = (x + 2) / 4
            reg = radsimp(1 - abs(x_ - y_) / 2)
            w_ = radsimp(max(x_, y_) / reg)
            y = radsimp([
                1 - reg,
                (w_ + 2) / 4 * reg,
                (2 - w_) / 4 * reg,
                -z_ + (10 * (w_ + 2) / 4 - 2 * (2 - w_) / 4) * reg,
            ])
            if all(_ >= 0 for _ in y):
                # small tip: note the identity s(a2(b2-c2)2) = (s(a(b-c)2)2+p(a-b)2+8p(a)s(a(b-c)2))/2
                if min(y[1], y[2]) * 8 >= y[3]:
                    exchange = y[3] / 8
                    y[1] -= exchange
                    y[2] -= exchange
                    y[3] = 0
                    y.append(exchange * 2)
                else:
                    y = None
                    # y.append(sp.S(0))
                
                if y is not None:
                    y = [radsimp(_ * coeff((4,2,0))) for _ in y] + [rem]
                    exprs = [
                        CyclicSum(a**4*(b-c)**2) if x_ > y_ else CyclicSum(a**2*b**2*(a-b)**2),
                        CyclicSum(a*(b-c)**2)**2,
                        CyclicProduct((a-b)**2),
                        CyclicProduct(a) * CyclicSum(a*(b-c)**2),
                        CyclicSum(a**2*(b**2-c**2)**2),
                        CyclicProduct(a**2)
                    ]

                    return sum_y_exprs(y, exprs)
                
        if x_ == (z_ - 4)/3:
            # Case 2: linear combination of s(a(b-c)^2)^2, s(a^2-ab)s(ab)^2, p(a-b)^2
            y = radsimp([
                -y_/12 + z_/9 + sp.Rational(1,18),
                y_/6 - z_/18 + sp.Rational(2,9),
                sp.Rational(1,2) - y_ / 4
            ])
            if all(_ >= 0 for _ in y):
                y = [radsimp(_ * coeff((4,2,0))) for _ in y] + [rem]
                exprs = [
                    CyclicSum(a*(b-c)**2)**2,
                    CyclicSum((a-b)**2) * CyclicSum(a*b)**2,
                    CyclicProduct((a-b)**2),
                    CyclicProduct(a)**2
                ]

                return sum_y_exprs(y, exprs)

        if x_ <= 2 and y_ <= 2 and (x_ != -2 or y_ != -2):
            u_ = sp.sqrt(x_ + 2)
            v_ = sp.sqrt(y_ + 2)
            if u_**2 + v_**2 + u_*v_ - 2 == z_:
                if u_ != 0:
                    r = radsimp(v_ / u_)
                    # (x - w) / (1 - w) = (-2*r*r-2*r+1)/(r*r+r+1)
                    t = radsimp((-2*r*r-2*r+1)/(r*r+r+1))
                    w = radsimp((t - x_) / (t + 2))
                    if 0 <= w <= 1:
                        x__, y__ = radsimp(1-3/(r+2)), radsimp(-(2*r+1)/(r+2))
                        x__q, y__q = x__.as_numer_denom()[1], y__.as_numer_denom()[1]
                        rr = radsimp(x__q * y__q / sp.gcd(x__q, y__q))
                        y = [
                            w,
                            (1 - w) / (x__**2 + y__**2 + 1) / rr**2
                        ]
                        y = [radsimp(_ * coeff((4,2,0))) for _ in y] + [rem]
                        rrx, rry = radsimp(rr * x__), radsimp(rr * y__)
                        exprs = [
                            CyclicProduct((a-b)**2),
                            CyclicSum((b-c)**2*(rr*a**2-rrx*a*b-rrx*a*c+rry*b*c)**2),
                            CyclicProduct(a**2)
                        ]
                        return sum_y_exprs(y, exprs)
        
    if real and abs(coeff((3,3,0))) <= 2 * (coeff((4,2,0))) and abs(coeff((4,1,1))) <= 2 * (coeff((4,2,0))):
        # perform sum of squares for real numbers
        
        # case 1. we can subtract enough p(a-b)2 till the bound on the border
        # and use Theorem 1.
        coeff42, u, v = coeff((4,2,0)), coeff((3,3,0)), coeff((4,1,1))
        if u >= v and (u != coeff42 * (-2) or v != coeff42 * (-2)):
            # subtracting p(a-b)2 will first lead to s(a4b2+2a3b3+a2b4)
            # assume coeffp = coefficient of p(a-b)2
            # (coeff42 - coeffp) * (2) = u + 2 * coeffp
            coeffp = radsimp((coeff42 * 2 - u) / 4)
            coeff42 = radsimp(coeff42 - coeffp)
            u, v = sp.S(2), radsimp((v + coeffp * 2) / coeff42)

            # now we need to ensure (v + 2) is a square, and make a perturbation otherwise
            # assume coeffsym = coefficient of s(a2b+ab2-2abc)2
            # (v - 2 * coeffsym / coeff42) / (1 - coeffsym / coeff42) + 2 = x*x <= 4
            # print(v, v.n(20), coeffp)
            coeffsym, x = sp.S(0), sp.sqrt(v + 2)
            need_rationalize = True if not isinstance(x, sp.Rational) else False

            if not coeff.is_rational:
                # first check whether there is exact solution
                coeff321_std = radsimp(-x**2 - 2*x - 2)
                coeff321 = radsimp((coeff((3,2,1)) - 2 * coeffp) / (coeff42))
                diff = radsimp(coeff321 - coeff321_std)
                if 0 < x <= 1 and (diff == 0 or diff == 4 * x):
                    need_rationalize = False
                elif 1 < x <= 2 and coeff321 == coeff321_std:
                    need_rationalize = False

            if need_rationalize:
                for x_ in rationalize_bound(x.n(20), direction = -1, compulsory = True):
                    if abs(x_) == 2:
                        continue

                    coeffsym = radsimp(((v + 2 - x_**2) / (4 - x_**2)) * coeff42)
                    coeff321_std = radsimp(-x_**2 - 2*x_ - 2)
                    coeff321 = radsimp((coeff((3,2,1)) - 2 * coeffp + 10 * coeffsym) / (coeff42 - coeffsym))
                    if 0 < x_ <= 1 and 0 <= coeff321 - coeff321_std <= 4 * x_:
                        x = x_
                        break
                    elif 1 < x_ <= 2 and coeff321 == coeff321_std:
                        x = x_
                        break
                else:
                    x , x_ = None, None
            coeff42 -= coeffsym

            if x is not None:
                if x == 0:
                    # must be the case s(bc(b+c)2(a-b)(a-c))
                    # Solution1: s((a-b)2(a2b-a2c+ab2-10abc+3ac2-b2c+3bc2+4c3)2)
                    # Solution2: s((a-b)2(a2b-a2c+ab2+2abc-ac2-b2c-bc2)2)
                    if radsimp(coeff((3,2,1)) - 2 * coeffp + 10 * coeffsym) == -2 * coeff42:
                        y = radsimp([coeffp, coeffsym, coeff42, rem])
                        multiplier = CyclicSum((a-b)**2)
                        exprs = [
                            CyclicProduct((a-b)**2) * CyclicSum((a-b)**2),
                            CyclicSum(a*(b-c)**2)**2 * CyclicSum((a-b)**2),
                            CyclicSum((a-b)**2 * (a**2*b - a**2*c + a*b**2 + 2*a*b*c - a*c**2 - b**2*c - b*c**2)**2),
                            CyclicProduct(a**2) * multiplier
                        ]
                        if all(_ >= 0 for _ in y):
                            return sum_y_exprs(y, exprs) / multiplier
                
                else:
                    x = 2 / x

                    # weight of linear combination of x and -x
                    w2 = radsimp(((coeff((3,2,1)) - 2 * coeffp + 10 * coeffsym) / coeff42 - (-2*(x**2 + 2*x + 2)/x**2)) / (8 / x))
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
                                    z = radsimp(2*(x**2 - x + (x-2)*tt) / (x*x + 4*x - 8))
                                    r2 = radsimp(1 / cancel_denominator(radsimp([x-2, 2*z-x, -3*x*z+2*x-2*z+4, x*(1-z)])))

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
                                return sum_y_exprs(radsimp(y), exprs) / multiplier


                    if x >= 2:
                        multiplier = CyclicSum((a-b)**2)

                        r1 = radsimp(1 / cancel_denominator(radsimp([2, 2+3*x, x])))

                        y = [
                            coeffp,
                            coeffsym,
                            w1 / 4 / r1**2,
                            (x**2 + 4*x - 8) / 4 * w1,
                        ]

                        xr1 = radsimp(x * r1)
                        exprs = [
                            CyclicProduct((a-b)**2) * CyclicSum((a-b)**2),
                            CyclicSum(a*(b-c)**2)**2 * CyclicSum((a-b)**2),
                            CyclicSum(c**2*(a-b)**2*(2*r1*a**2+2*r1*b**2-2*r1*c**2-radsimp((2+3*x)*r1)*a*b+xr1*a*c+xr1*b*c+xr1*c**2)**2),
                            CyclicSum(a)**2 * CyclicProduct((a-b)**2),
                        ]
                        extra = radsimp((x**2 - 4*x - 8) / 8 * w2)
                        if w2 != 0 and y[-1] + extra >= 0:
                            y[-1] += extra * 2
                            y.append(radsimp(w2 / 4 / r1**2))
                            exprs.append(
                                CyclicSum(c**2*(a-b)**2*(2*r1*a**2+2*r1*b**2-2*r1*c**2+radsimp((3*x-2)*r1)*a*b-xr1*a*c-xr1*b*c-xr1*c**2)**2)
                            )

                        elif w2 != 0: # in this case we must have x**2 - 4*x - 8 <= 0:
                            r2 = radsimp(1 / cancel_denominator(radsimp([x+2, x-2, 6-5*x, 2*x])))
                            y += [
                                (x - 2)*(5*x + 6)/(4*(x + 2)*(x + 10)) * w2 / r1**2,
                                -(x**2 - 4*x - 8)/(4*(x + 2)*(x + 10)) * w2 / r2**2
                            ]
                            xr1 = radsimp(x * r1)
                            r2p2, r2m2, r2d2 = radsimp(r2*(x+2)), radsimp(r2*(x-2)), radsimp(r2*2*x)
                            exprs += [
                                CyclicSum(c**2*(a-b)**2*(2*r1*a**2 + 2*r1*b**2 - 2*r1*c**2 + radsimp((3*x-2)*r1)*a*b
                                    - xr1*a*c - xr1*b*c - xr1*c**2)**2),
                                CyclicSum((a-b)**2*(-r2p2*a**2*b + r2m2*a**2*c - r2p2*a*b**2 + radsimp(r2*(-5*x+6))*a*b*c
                                    + r2d2*a*c**2 + r2m2*b**2*c + r2d2*b*c**2 + r2p2*c**3)**2)
                            ]


                        if all(_ >= 0 for _ in y):
                            mul = radsimp(coeff42 * 4 / x**2)
                            for i in range(2, len(y)):
                                y[i] *= mul
                            y.append(rem)
                            exprs.append(CyclicProduct(a**2) * multiplier)
                            return sum_y_exprs(radsimp(y), exprs) / multiplier


        elif u < v:
            # subtracting p(a-b)2 will first lead to s(a4b2+2a4bc+a4c2)
            # assume coeffp = coefficient of p(a-b)2
            # (coeff42 - coeffp) * (2) = v + 2 * coeffp
            coeffp = radsimp((coeff42 * 2 - v) / 4)
            coeff42 = radsimp(coeff42 - coeffp)
            u, v = radsimp((u + coeffp * 2) / coeff42), sp.S(2)
            
            
            # now we need to ensure (u + 2) is a square, and make a perturbation otherwise
            # assume coeffsym = coefficient of s(a2b+ab2-2abc)2
            # (u - 2 * coeffsym / coeff42) / (1 - coeffsym / coeff42) + 2 = x*x <= 4
            # print(v, v.n(20), coeffp)
            coeffsym, x = sp.S(0), sp.sqrt(u + 2)
            if not isinstance(x, sp.Rational):
                for x_ in rationalize_bound(x.n(20), direction = -1, compulsory = True):
                    if abs(x_) == 2:
                        continue

                    coeffsym = radsimp(((u + 2 - x_**2) / (4 - x_**2)) * coeff42)
                    coeff321_std = radsimp(-x_**2 - 2*x_ - 2)
                    coeff321 = radsimp((coeff((3,2,1)) - 2 * coeffp + 10 * coeffsym) / (coeff42 - coeffsym))
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
                    
                    if radsimp(coeff((3,2,1)) - 2 * coeffp + 10 * coeffsym) == -2 * coeff42:
                        y = radsimp([coeffp, coeffsym, coeff42, rem])
                        multiplier = CyclicSum((a-b)**2)
                        exprs = [
                            CyclicProduct((a-b)**2) * CyclicSum((a-b)**2),
                            CyclicSum(a*(b-c)**2)**2 * CyclicSum((a-b)**2),
                            CyclicSum((a-b)**4 * (a*b+a*c+b*c-c**2)**2),
                            CyclicProduct(a**2) * multiplier
                        ]
                        if all(_ >= 0 for _ in y):
                            return sum_y_exprs(y, exprs) / multiplier

                else:
                    # linear combination of x and -x
                    w2 = radsimp(((coeff((3,2,1)) - 2 * coeffp + 10 * coeffsym) / coeff42 - (-4*x**2-4*x-2)) / (8*x))
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
                            return sum_y_exprs(radsimp(y), exprs) / multiplier

    if rem > 0:
        if coeff((3,3,0)) == coeff((4,1,1)):
            # try c1p(a-b)2 + c2s(a2b-xabc)2 + c2s(ab2-xabc)2 + c3p(a2)
            c1 = coeff((3,3,0)) / (-2)
            c2 = coeff((4,2,0)) - c1
            x_ = radsimp((2 - (coeff((3,2,1)) - 2*c1) / c2) / 6)
            y = [
                c1,
                c2,
                c2,
                radsimp(rem - 18*c2*(x_ - 1)**2)
            ]
            if all(_ >= 0 for _ in y):
                exprs = [
                    CyclicProduct((a-b)**2),
                    (a**2*b + b**2*c + c**2*a - (3*x_)*a*b*c)**2,
                    (a**2*c + b**2*a + c**2*b - (3*x_)*a*b*c)**2,
                    CyclicProduct(a**2)
                ]
                return sum_y_exprs(y, exprs)



    if True:
        # subtract p(a-b)2
        t = coeff((4,2,0))
        new_coeffs_ = {
            (3,3,0): coeff((3,3,0)) + 2 * t,
            (4,1,1): coeff((4,1,1)) + 2 * t,
            (3,2,1): coeff((3,2,1)) - 2 * t,
            (2,3,1): coeff((2,3,1)) - 2 * t,
            (2,2,2): coeff((2,2,2)) + 6 * t
        }

        solution = _sos_struct_sextic_hexagram_symmetric(Coeff(new_coeffs_))
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
        for r in nroots(equ, method = 'factor', real = True, nonnegative = True):
            if 3*r*(r-1) + v > 1e-14:
                if not isinstance(r, sp.Rational):
                    numer_r = r
                    for r in rationalize_bound(numer_r, direction = -1, compulsory = True):
                        if r**3-3*r <= u and 3*r*(r-1)+v >= 0:
                            break
            if 3*r*(r-1) + v >= 0:
                break
        else:
            r = None
    
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
    F(a,b,c) = F0 - 2s(a^4-a^2bc)f(a,b,c) + s(a^2-ab)f(a,b,c)^2 >= 0
    because
    F(a,b,c) * s(a^2-ab) = (s(a^2-ab)f(a,b,c) - s(a^4-a^2bc))^2 + s(ab)p(a-b)^2 >= 0.

    So we try to write the original polynomial in such quadratic form. Note that for such F,
    it has three multiplicative roots on the symmetric axis b=c=1, one of which is the centroid a=b=c=1.
    The other two roots determine the coefficient x and y.

    For normal polynomial without three multiplicative roots, we first write the symmetric axis in the form
    of (a-1)^2 * ((a^2+..a+..)^2 + (..a+..)^2 + ...). Now we apply the theorem to each of them, and then
    merge their f(a,b,c) according.


    Moreover, there exists u, v such that u+v = (2-2*y)/(x-1), uv = (2*x+y-2)/(x-1) so that
    F(a,b,c) = (x-1)^2s((a-b)(a-c)(a-ub)(a-uc)(a-vb)(a-vc)) + (x^2-xy+y^2-y)p(a-b)^2

    Examples
    --------
    72p(a+b-2c)2+s(a2-ab)s(11a2-14ab)2

    (s((a-b)(a-c)(a-2b)(a-2c)(a-18b)(a-18c))-53p(a-b)2)

    (s((b-c)2(7a2-b2-c2)2)-112p(a-b)2)

    s((a-b)2(-a2-b2+2c2+2(ab-c2)-3s(ab)+2s(a2))2)

    s(a5)s(a/3)+19abc(s(ab)s(a/3)-3s(a/3)3)+3abc(abc-2s(a/3)3)

    s((a-b)(a-c)(a-2b)(a-2c)(a-18b)(a-18c))-53p(a-b)2

    s((b2+c2-5a(b+c))2(b-c)2)-22p(a-b)2 

    s(a6+6a5b+6a5c-93a4b2+3a4bc-93a4c2+236a3b3+87a3b2c+87a3bc2-240a2b2c2)

    s(a6-21a5b-21a5c-525a4b2+1731a4bc-525a4c2+11090a3b3-13710a3b2c-13710a3bc2+15690a2b2c2)

    References
    -------
    [1] https://artofproblemsolving.com/community/c6t243f6h3013463_symmetric_inequality

    [2] https://tieba.baidu.com/p/8261574122
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
        coeff = v/(2*y-2) if y != 1 else w / (2*x + y - 2)
        return x, y, coeff

    params = []
    for coeff0, sym_part in zip(sym[0][1], sym[0][2]):
        x_, y_, coeff1 = _solve_from_sym(sym_part)
        if coeff1 is sp.nan:
            return None
        # part_poly = coeff0 * coeff1**2 * pl(f's(a6+a5b+a5c+a4bc-2a3b2c-2a3bc2)-2s(a4-a2bc)s({x_}a2+{y_}ab)+s(a2-ab)s({x_}a2+{y_}ab)2')
        # print((x_, y_), coeff0 * coeff1**2, poly_get_factor_form(part_poly))
        params.append((coeff0 * coeff1**2, x_, y_))

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

    if not (m_ == 0 and p_ == 0 and n_ == 0):
        t_ = (-2*m_**2 + m_*n_ - p_**2/4)/(n_ + p_ - m_)
        if t_ < 0: # actually this will not happen
            return None
        if m_ != t_:
            # rem_coeff * s(a^2 + rem_ratio * ab)^2
            rem_ratio = (p_ + 2*t_) / (m_ - t_) / 2
            rem_poly = (m_ - t_) * CyclicSum(a**2 + rem_ratio*a*b)**2
        else:
            # it degenerates to rem_coeff * s(ab)^2
            if n_ < 3*t_: # this will not happen
                return None
            rem_poly = (n_ - 3*t_) * CyclicSum(a*b)**2
    else:
        t_, rem_poly = sp.S(0), sp.S(0)

    # ker_coeff is the remaining coefficient of (a-b)^2(b-c)^2(c-a)^2
    ker_coeff = (poly.coeff_monomial((4,2,0)) - merged_params[0] * (3*x_**2 - 2*x_*y_ - 2*x_ + y_**2) - (n_ - p_ + m_))

    # each t_ exhanges 27/4p(a-b)^2 because s(a^2-ab)^3 = 1/4 * p(2a-b-c)^2 + 27/4 * p(a-b)^2
    ker_coeff += 27 * t_ / 4

    print('Coeff =', merged_params[0], 'ker =', ker_coeff)
    print('  (x,y) =', (x_, y_), 'ker_std =', ker_coeff / merged_params[0])
    print('  (m,p,n,t) = ', (m_, p_, n_, t_))

    if x_ + y_ == sp.S(5)/3 and x_ != 1:
        # F_{x,y} = (x-1)^2/4 * s((b-c)^2(b+c-za))^2 + 3(x-1)(9x-5)/4 * p(a-b)^2
        z_ = 2*(3*x_ - 2) / 3 / (x_ - 1)
        ker_coeff2 = -3*(x_ - 1)*(9*x_ - 5) / 4 * merged_params[0]
        # p0 = (b-c)**2*(b+c-z_*a)
        p0 = 2*a**3 - (z_ + 1)*b*c**2 - (z_ + 1)*b**2*c + 2*z_*a*b*c
        p0 = p0.together().as_coeff_Mul()
        if ker_coeff >= ker_coeff2:
            solution = [
                merged_params[0] * (x_ - 1)**2/4 * p0[0]**2 * CyclicSum(p0[1])**2,
                t_/4 * CyclicProduct((a+b-2*c)**2),
                sp.Rational(1,2) * CyclicSum((a-b)**2) * rem_poly,
                (ker_coeff - ker_coeff2) * CyclicProduct((a-b)**2)
            ]
            return sp.Add(*solution)

    if ker_coeff >= merged_params[0] / 3:
        # Case that we do not need to higher the degree because
        # F(a,b,c) + p(a-b)^2/3 = s((a-b)^2((3*x-3)*a^2+(3*y-4)*a*b+(3*y-2)*a*c+(3*x-3)*b^2+(3*y-2)*b*c+(3*x-1)*c^2)^2)/18
        p1 = (3*x_ - 3)*a**2 + (3*y_ - 4)*a*b + (3*y_ - 2)*a*c + (3*x_ - 3)*b**2 + (3*y_ - 2)*b*c + (3*x_ - 1)*c**2
        if x_ < 1: p1 = -p1
        solution = [
            merged_params[0]/18 * CyclicSum((a-b)**2*p1**2),
            t_/4 * CyclicProduct((a+b-2*c)**2),
            sp.Rational(1,2) * CyclicSum((a-b)**2) * rem_poly,
            (ker_coeff - merged_params[0] / 3) * CyclicProduct((a-b)**2)
        ]
        return sp.Add(*solution)

    # Elapse if there is a root on the symmetric axis, in which case we apply SOS theorem directly.
    # This is because it is more beautiful and might handle sum-of-square for real numbers.
    sym_diff = sym0[0].diff(a)
    sym_gcd = sp.gcd(sym0[0], sym_diff)
    if sym_gcd.degree() == 1:
        root = radsimp(-sym_gcd.coeff_monomial((0,)) / sym_gcd.coeff_monomial((1,)))
        solution = _sos_struct_sextic_symmetric_ultimate_1root(coeff, poly, None, [[], [root], []])
        if solution is not None:
            return solution


    if ker_coeff >= 0:
        # Regular case
        p1 = ker_coeff*2*a**2 + (2*merged_params[0] - ker_coeff*2)*b*c
        p1 = p1.together().as_coeff_Mul()
        solution = [
            merged_params[0]*2 * (CyclicSum(a**2-b*c)*CyclicSum(x_*a**2+y_*a*b) - CyclicSum(a**4-a**2*b*c))**2,
            t_/4 * CyclicSum((a-b)**2) * CyclicProduct((a+b-2*c)**2),
            2 * CyclicSum(a**2-a*b)**2 * rem_poly,
            p1[0] * CyclicSum(p1[1]) * CyclicProduct((a-b)**2)
        ]
        return sp.Add(*solution) / CyclicSum((a-b)**2)



    def _tighter_bound_border(x, y, ker_coeff = 0):
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
        R(a,b,c) * s(a^2 + phi*b*c) = s((a-b)^2g(a,b,c)^2) + (phi - (m-2)*(suv-2*m)/(m+puv-2)) * s(ab(a-b)^2h(a,b,c)^2)
            + final_coeff * (puv**2 * s(ab)p(a-b)^2 + (2-m)s(bc(a-b)(a-c)(a-ub)(a-uc)(a-vb)(a-vc)))

        References
        --------
        [1] https://tieba.baidu.com/p/8261574122
        """
        suv, puv = (2-2*y)/(x-1), (2*x+y-2)/(x-1)
        a, b, c, m = sp.symbols('a b c m')
        det1 = (-2*m**3 + (suv+7)*m**2 - 4*(suv+1)*m + puv**2+4*(suv-1)).as_poly(m)
        def _compute_params(suv, puv, m):
            det1 = -2*m**3 + (suv+7)*m**2 - 4*(suv+1)*m + puv**2 + 4*(suv-1)
            det2 = 4*m**3 - (4*suv+7)*m**2 + ((suv+4)**2+2*puv-20)*m + (puv-2)**2-2*suv**2
            denom = (m+puv-2)**2 * (suv-2*m-1)
            phi = -puv**2 * det2 / (m-2) / denom - m
            final_coeff = det1 * det2 / (m-2)**2 / denom
            w = (m**3 - (suv+1)*m**2 + (suv+puv-3)*m + (puv-1)**2+2*suv+1) / (m - 2)
            ker_coeff = -(w*(x-1)**2 + (x**2-x*y+y**2-y))
            return suv, puv, m, det1, det2, phi, final_coeff, ker_coeff
        def _validate_params(params):
            suv, puv, m, det1, det2, phi, final_coeff, ker_coeff = params
            return phi >= -1 and phi >= ((m-2)*(suv-2*m)/(m+puv-2))
        for root in nroots(det1, method = 'factor', real = True):
            if isinstance(root, sp.Rational) and (root == 2 or m + puv == 2 or suv - 2*m - 1 == 0):
                return None
            if not root.is_real:
                continue
            params = _compute_params(suv, puv, root)
            # print(params)
            if _validate_params(params) and params[-1] <= ker_coeff:
                break
        else:
            return None


        def _solve_border(*args, with_tail = True, with_frac = False):
            if len(args) == 3:
                args = _compute_params(*args)
            suv, puv, m, det1, det2, phi, final_coeff, ker_coeff = args
            l1 = (-m**2+(puv+1)*m-puv*(suv+1)+2)/(m+puv-2)
            l2 = ((1-2*puv)*m**2+(puv*(suv+1)-1)*m+puv-2)/(m+puv-2)
            l3 = (2*(puv+suv)*m**2-(puv*(suv-1)+(suv+2)**2-2)*m+(puv-2)**2+2*suv**2)/(m+puv-2)
            l4 = ((2*puv-1)*m-puv*(suv+1)+2)/(m+puv-2)
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
            return (x-1)**2 * _solve_border(*params, with_tail = False, with_frac = True) + (ker_coeff - params[-1]) * CyclicProduct((a-b)**2)

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
                return solution

    main_solution = None
    if True:
        main_solution =  _tighter_bound_border(x_, y_, ker_coeff / merged_params[0])

    if main_solution is not None:
        return merged_params[0] * main_solution + t_/4 * CyclicProduct((a+b-2*c)**2) + sp.Rational(1,2) * CyclicSum((a-b)**2) * rem_poly
    


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
  
    print('Roots Info = ', roots)
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