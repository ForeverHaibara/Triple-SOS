import sympy as sp

from .utils import (
    CyclicSum, CyclicProduct,
    sum_y_exprs, inverse_substitution
)


def sos_struct_octic(poly, coeff, recurrsion):
    a, b, c = sp.symbols('a b c')
    if any((coeff((8,0,0)), coeff((7,1,0)), coeff((7,0,1)))):
        return None

    if not any((coeff((6,2,0)), coeff((2,6,0)), coeff((5,3,0)), coeff((3,5,0)), coeff((4,4,0)))):
        solution = recurrsion(poly.div((a*b*c).as_poly(a,b,c))[0])
        if solution is not None:
            return CyclicProduct(a) * solution
        return None
    
    if not coeff.is_rational:
        return None

    if coeff((6,2,0)) == coeff((2,6,0)) and coeff((5,3,0)) == coeff((3,5,0))\
         and coeff((5,2,1)) == coeff((2,5,1)) and coeff((4,3,1)) == coeff((3,4,1)):
        solution = _sos_struct_octic_symmetric_hexagon(coeff, poly, recurrsion)
        if solution is not None:
            return solution

    if coeff((6,2,0))==0 and coeff((6,1,1))==0 and coeff((6,0,2))==0:

        # equivalent to degree-7 hexagon when applying (a,b,c) -> (1/a,1/b,1/c)
        poly2 = coeff((0,3,5))*a**5*b**2 + coeff((1,2,5))*a**4*b**3 + coeff((2,1,5))*a**3*b**4 + coeff((3,0,5))*a**2*b**5\
                + coeff((3,3,2))*a**2*b**2*c**3+coeff((0,4,4))*a**5*b*c+coeff((1,3,4))*a**4*b**2*c+coeff((2,2,4))*a**3*b**3*c+coeff((3,1,4))*a**2*b**4*c
        poly2 = CyclicSum(poly2).doit().as_poly(a,b,c)
        solution = recurrsion(poly2)

        if solution is not None:
            # unrobust method handling fraction
            return inverse_substitution(solution, factor_degree = 2)


    return None


def _sos_struct_octic_symmetric_hexagon(coeff, poly, recurrsion):
    """
    Try to solve symmetric octic hexagon, without terms a^8, a^7b and a^7c.

    For octics and structural method, the core is not to handle very complicated cases.
    Instead, we explore the art of sum of squares by using simple tricks.
    """
    a, b, c = sp.symbols('a b c')
    c1, c2, c3, c4 = [coeff(_) for _ in ((6,2,0),(5,3,0),(6,1,1),(5,2,1))]
    if c1 < 0 or 2*c1 + c3 < 0:
        return None

    def _append_inverse_quartic(y, exprs, m_, p_, n_, r_):
        """
        Solve a symmetric inverse quartic expression fast without callback. It only involves
        monoms inside the triangle (a^4b^4, a^4c^4, b^4c^4). Hence it is equivalent to a
        quartic with respect to ab, bc and ca.

        It returns a solution with the original y, exprs added.
        """
        if m_ >= 0 and m_ + 2*p_ + n_ + r_ >= 0 and all(_ >= 0 for _ in y):
            if m_ != 0 and (n_ - ((p_ / m_)**2 - 1) * m_) >= 0:
                y.extend([
                    m_ / 2,
                    (n_ - ((p_ / m_)**2 - 1) * m_) / 2 if m_ != 0 else sp.S(0),
                    m_ + 2*p_ + n_ + r_
                ])
                exprs.extend([
                    CyclicSum(a**2*(b-c)**2*(a*b+a*c + (p_ / m_) *b*c)**2),
                    CyclicProduct(a**2) * CyclicSum((a-b)**2),
                    CyclicSum(a**3*b**3*c**2),
                ])
                return sum_y_exprs(y, exprs)

            elif p_ + m_ >= 0 and (n_ + 2*(p_ + m_)) >= 0:
                y.extend([
                    m_ / 2,
                    p_ + m_,
                    (n_ + 2*(p_ + m_)) / 2,
                    m_ + 2*p_ + n_ + r_
                ])
                exprs.extend([
                    CyclicSum(a**2*(b-c)**2*(a*b+a*c-b*c)**2),
                    CyclicProduct(a) * CyclicSum(a**3*(b-c)**2),
                    CyclicProduct(a**2) * CyclicSum((a-b)**2),
                    CyclicSum(a**3*b**3*c**2),
                ])
                return sum_y_exprs(y, exprs)
        return None


    if True:
        if True:
            # Case 1. use 
            # s(a(b-c)2)2s(xa2+yab)+p(a-b)2s(za2+wab)
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
                y = [
                    sp.S(1),
                    sp.S(1)
                ]
                exprs = [
                    CyclicSum(a*(b-c)**2)**2 * CyclicSum(x_*a**2 + y_*b*c),
                    CyclicProduct((a-b)**2) * CyclicSum(z_*a**2 + w_*b*c)
                ]
                solution = _append_inverse_quartic(y, exprs, m_, p_, n_, r_)
                if solution is not None:
                    return solution
        
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
                y = [
                    x_,
                    sp.S(1)
                ]
                exprs = [
                    CyclicSum((a-b)**2 * (a**2*b+a**2*c+a*b**2+(y_-1)*a*c**2+b**2*c+(y_-1)*b*c**2-2*y_*a*b*c)**2),
                    CyclicProduct((a-b)**2) * CyclicSum(z_*a**2 + w_*b*c)
                ]
                solution = _append_inverse_quartic(y, exprs, m_, p_, n_, r_)
                if solution is not None:
                    return solution


    if coeff((6,2,0)) == 0 and coeff((5,3,0)) == 0:
        return _sos_struct_octic_symmetric_hexagram(coeff)

    return None


def _sos_struct_octic_symmetric_hexagram(coeff):
    """
    Solve octic symmetric hexagram, where all terms are inside the triangle (a^6bc,...) and (a^4b^4,...).

    The idea is to write the problem to s(bc(xa^4 + ya^3(b+c) + za^2(b^2+c^2) + wa^2bc + uabc(b+c) + vb^2c^2)(a-b)(a-c)).
    Then, we use the following lemma: if f(a,b,c) and g(a,b,c) are both symmetric polynomials with respect to b,c.
    Then, \sum f(a,b,c)(a-b)(a-c) * \sum g(a,b,c)(a-b)(a-c) - \sum f(a,b,c)g(a,b,c)(a-b)(a-c)
    must be a multiple of p(a-b)2.
    A common choice of g is g(a,b,c) = 1.

    
    Examples
    -------
    s(bc(a2+1/2a(b+c)-bc)2(a-b)(a-c))

    s((a-b)2(a+b-3c)2)s(a2b2)+2s(a2(b-c)2(ab+ac-3/2bc)2)-p(a-b)2s(2a2-2ab)

    s(2a6bc-3a5b2c-3a5bc2+a4b4+3a4b2c2)

    s(bc(2a4+a3b+a3c+a2b2+9a2bc+a2c2-3ab2c-3abc2+b2c2)(a-b)(a-c))

    24s((a+b-c)(a-b)2(a+b-3c)2)p(a)+s(a2b2(ab-ac)(ab-bc))

    256p(a)s((64a+(b+c))(a+b-59/16c)(a+c-59/16b)(a-b)(a-c))+s(a2b2(ab-bc)(ab-ca))

    s(bc(a-b)(a-c)(a-2b)(a-2c)(a-3b)(a-3c))

    s(bc(a-b)(a-c)(a2-2a(b+c)+5bc)(a-2b)(a-2c))
    """
    a, b, c = sp.symbols('a b c')
    x_ = coeff((6,1,1))
    v_ = coeff((4,4,0))
    rem = sum(coeff((i,j,k)) * (1 if i==j or j==k else 2) for i,j,k in ((6,1,1),(5,2,1),(4,3,1),(4,4,0),(4,2,2),(3,3,2)))
    if x_ <= 0 or v_ < 0 or rem < 0:
        return None
    
    y_ = coeff((5,2,1)) + x_
    u_ = coeff((4,3,1)) + v_ + y_
    balance = coeff((4,2,2)) - x_ + 2*u_ + 2*y_

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
        print(degrade_a2b2, degrade_a2bc, (2*x_)*(a**2 + r1*a*b + r1*a*c + r2*b*c)**2 + c2*2*(b*c - h*a*b - h*a*c)**2 + c3*2*a**2*b*c + c4*a*b*c*(b+c))

        if degrade_a2b2 + degrade_a2bc >= 0:
            multiplier = CyclicSum((a-b)**2)
            # p1 == f(a,b,c)
            p1 = sp.together((2*x_)*(a**2 + r1*a*b + r1*a*c + r2*b*c)**2 + c2*2*(b*c - h*a*b - h*a*c)**2 + c3*2*a**2*b*c + c4*a*b*c*(b+c)).as_coeff_Mul()
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
            p_fin = sp.together(degrade_a3 * CyclicSum(a*(a-b)*(a-c)) 
                                + (degrade_a2b + degrade_a3) * CyclicSum(a*(b-c)**2) 
                                + (degrade_a3*3 + degrade_a2b*6 + degrade_abc) * CyclicProduct(a)).as_coeff_Mul()
            
            y = [
                x_ * 2 * p1[0],
                c2 * 2 * p2[0],
                c3 * 2,
                c4 * 2,
                sp.S(2) * p_fin[0],
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