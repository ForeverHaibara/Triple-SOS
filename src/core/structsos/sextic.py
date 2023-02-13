import sympy as sp
from sympy.solvers.diophantine.diophantine import diop_DN

from ...utils.text_process import cycle_expansion
from ...utils.root_guess import rationalize, rationalize_bound
from .peeling import _merge_sos_results, _make_coeffs_helper

def _sos_struct_sextic(poly, degree, coeff, recurrsion):
    multipliers, y, names = _sos_struct_sextic_symmetric_ultimate(coeff, poly, recurrsion)
    if y is not None:
        return multipliers, y, names

    if coeff((6,0,0))==0 and coeff((5,1,0))==0 and coeff((5,0,1))==0:
        multipliers, y, names = _sos_struct_sextic_hexagon(coeff, poly, recurrsion)

    return multipliers, y, names


def _sos_struct_sextic_hexagram(coeff, poly, recurrsion):
    """
    Solve s(a3b3+xa4bc+ya3b2c+za2b3c+wa2b2c2) >= 0

    Examples
    -------
    s(a3b3+7a4bc-29a3b2c+12a3bc2+9a2b2c2)

    9s(a3b3)+4s(a4bc)-11abcp(a-b)-37s(a3b2c)+72a2b2c2
    """
    multipliers, y, names = [], None, None

    if coeff((3,2,1)) == coeff((2,3,1)):
        multipliers, y, names = _sos_struct_sextic_hexagram_symmetric(coeff)
    
    if y is None:
        # hexagram (star)
        poly = poly * sp.polys.polytools.Poly('a+b+c')
        multipliers, y, names = _merge_sos_results(['a'], y, names, recurrsion(poly, 7))
    
    return multipliers, y, names


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
    multipliers, y, names = [], None, None
    # hexagon
    if coeff((6,0,0))==0 and coeff((5,1,0))==0 and coeff((5,0,1))==0:
        if coeff((4,2,0)) != coeff((4,0,2)):
            # first try whether we can cancel these two corners in one step 
            # CASE 1. 
            if coeff((4,2,0)) == 0 or coeff((4,0,2)) == 0:
                if coeff((3,3,0)) == 0 and coeff((4,1,1)) == 0:
                    return _sos_struct_sextic_rotated_tree(coeff)

                if coeff((4,0,2)) == 0:
                    y = [coeff((4,2,0))]
                    names = ['(a*a*b+b*b*c+c*c*a-3*a*b*c)^2']
                else:
                    y = [coeff((4,0,2))]
                    names = ['(a*b*b+b*c*c+c*a*a-3*a*b*c)^2']                 

                poly2 = poly - y[0] * sp.sympify(names[0])
                v = 0 # we do not need to check the positivity, just try
                if v != 0:
                    y, names = None, None 
                else: # positive
                    multipliers , y , names = _merge_sos_results(multipliers, y, names, recurrsion(poly2, 6))

            # CASE 2. 
            else:
                a , b = (coeff((4,2,0)) / coeff((4,0,2))).as_numer_denom()
                if sp.ntheory.primetest.is_square(a) and sp.ntheory.primetest.is_square(b):
                    y = [coeff((4,2,0)) / a / 3] 
                    names = [f'({sp.sqrt(a)}*(a*a*b+b*b*c+c*c*a-3*a*b*c)-{sp.sqrt(b)}*(a*b*b+b*c*c+c*a*a-3*a*b*c))^2']
                
                    poly2 = poly - 3 * y[0] * sp.sympify(names[0])
                    v = 0 # we do not need to check the positivity, just try
                    if v != 0:
                        y, names = None, None 
                    else: # positive
                        multipliers , y , names = _merge_sos_results(multipliers, y, names, recurrsion(poly2, 6))

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
                        names = [f'(a*a*c-b*b*c-{p}*(a*a*b-a*b*c)+{q}*(a*b*b-a*b*c))^2']
                        poly2 = poly - t * sp.sympify(cycle_expansion(names[0]))
                        multipliers , y , names = _merge_sos_results(multipliers, y, names, recurrsion(poly2, 6))
                        if y is not None:
                            break


        else:# coeff((4,2,0)) == coeff((4,0,2)):
            if coeff((4,2,0)) == 0:
                multipliers, y, names = _sos_struct_sextic_hexagram(coeff, poly, recurrsion)
            elif coeff((3,2,1)) == coeff((2,3,1)):
                # symmetric
                multipliers, y, names = _sos_struct_sextic_hexagon_symmetric(coeff)
            else:
                y = [coeff((4,2,0)) / 3] 
                names = [f'(a-b)^2*(b-c)^2*(c-a)^2']

                poly2 = poly - y[0] * 3 * sp.sympify(names[0])
                v = 0 # we do not need to check the positivity, just try
                if v != 0:
                    y, names = None, None 
                else: # positive
                    multipliers , y , names = _merge_sos_results(multipliers, y, names, recurrsion(poly2, 6))
        
    return multipliers, y, names



def _sos_struct_sextic_hexagon_full(coeff):
    """
    Examples
    -------
    s(4a5b+9a5c-53a4bc+10a4c2-19a3b3+47a3b2c+52a3bc2-50a2b2c2)
    """
    if coeff((6,0,0)) != 0:
        return [], None, None

    coeff51 = coeff((5,1,0))
    if coeff51 == 0 and coeff((1,5,0)) != 0:
        # reflect the polynomial
        return [], None, None

    m = sp.sqrt(coeff((1,5,0)) / coeff51)
    if not isinstance(m, sp.Rational):
        return [], None, None

    p, n, q = coeff((4,2,0)) / coeff51, coeff((3,3,0)) / coeff51, coeff((2,4,0)) / coeff51

    # (q - t) / (p - t) = -2m
    t = (2*m*p + q)/(2*m + 1)
    z = (p - t) / (-2)
    if t < 0 or n + 2 * t < z**2 - 2*m:
        return [], None, None

    # Case A. solve it directly through a cubic equation

    # Case B. optimize discriminant >= 0

    return [], None, None


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
        multipliers, y, names = _sos_struct_sextic_rotated_tree(new_coeff)
        if y is not None:
            multipliers = [_.translate({98: 99, 99: 98}) for _ in multipliers]
            names = [_.translate({98: 99, 99: 98}) for _ in names]
        return multipliers, y, names
    
    u = coeff((3,2,1)) / coeff((2,4,0))
    if coeff((2,4,0)) <= 0 or coeff((4,2,0)) != 0 or u < -2:
        return [], None, None
    v = coeff((3,1,2)) / coeff((2,4,0))

    multipliers, y, names = [], None, None
    if u >= 2 and v >= -6:
        y = [sp.S(1) / 3, u - 2, v + 6,
            (coeff((2,4,0)) + coeff((3,1,2)) + coeff((3,2,1))) + coeff((2,2,2)) / 3
        ]
        y = [_ * coeff((2,4,0)) for _ in y[:-1]] + [y[-1]]
        names = [
            '(a^2*c+b^2*a+c^2*b-3*a*b*c)^2',
            'a*b*c*(a^2*b-a*b*c)',
            'a*b*c*(a^2*c-a*b*c)',
            'a^2*b^2*c^2'
        ]
        return multipliers, y, names
    
    if u >= 0 and v >= -2:
        y = [sp.S(1), u, v + 2,
            (coeff((2,4,0)) + coeff((3,1,2)) + coeff((3,2,1))) + coeff((2,2,2)) / 3
        ]
        y = [_ * coeff((2,4,0)) for _ in y[:-1]] + [y[-1]]
        names = [
            'a^2*c^2*(a-b)^2',
            'a*b*c*(a^2*b-a*b*c)',
            'a*b*c*(a^2*c-a*b*c)',
            'a^2*b^2*c^2'
        ]
        return multipliers, y, names

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
                multipliers = ['a']
                y = [
                    sp.S(1),
                    p1,
                    q1 - n1**2/4/p1 if p1 != 0 else q1,
                    sp.S(0) if p1 != 0 else n1 / 2,
                    (coeff((2,4,0)) + coeff((3,1,2)) + coeff((3,2,1))) + coeff((2,2,2)) / 3
                ]
                y = [_ * coeff((2,4,0)) for _ in y[:-1]] + [y[-1]]
                names = [
                    f'a*(a^2*c-a*b^2+{x_}*b*c^2+{y_}*b^2*c-{x_+y_}*a*b*c)^2',
                    f'a^2*b^2*c*(a-{-n1/2/p1}*b+{-n1/2/p1-1}*c)^2',
                    'a^2*b^2*c*(b-c)^2',
                    'a^3*b*c*(b-c)^2',
                    'a^3*b^2*c^2'
                ]
                return multipliers, y, names


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
        return [], None, None
    
    z = x * (x + 1) / 3
    
    multipliers = [f'a*b*b+{z}*a*b*c', 'a']
    y = [
        1,
        (x + 1) / 2,
        x + 1,
        v + 3*x*(x - 1),
        u - (x**3 - 3*x),
        (coeff((2,4,0)) + coeff((3,1,2)) + coeff((3,2,1))) + coeff((2,2,2)) / 3
    ]
    y = [_ * coeff((2,4,0)) for _ in y[:-1]] + [y[-1]]
    names = [
        f'a^2*c*(a+b+c)*(a^2*c-{x}*a*b^2-{x}*b*c^2+{x*x-1}*b^2*c-{x*(x-2)}*a*b*c)^2',
        f'a^2*b^2*c^2*({x-1}*a-b)^2*(a+{x-1}*b-{x}*c)^2',
        f'a*b*c^2*({x-1}*a-b)^2*({x}*a*b-a*c-{x-1}*b*c)^2',
        f'a*b*c*(a+b+c)*(a*b^2+b*c^2+c*a^2+{3*z}*a*b*c)*(a^2*c-a*b*c)',
        f'a*b*c*(a+b+c)*(a*b^2+b*c^2+c*a^2+{3*z}*a*b*c)*(a^2*b-a*b*c)',
        f'a^3*b^2*c^2*(a*b^2+b*c^2+c*a^2+{3*z}*a*b*c)'
    ]

    return multipliers, y, names



#####################################################################
#
#                              Symmetric
#
#####################################################################

def _sos_struct_sextic_hexagon_symmetric(coeff, real = True):
    """
    Solve symmetric hexagons.

    Examples
    --------
    s(3a2b+ab2-4abc)2+s(a2b+3ab2-4abc)2    (real)

    s((b-c)2(a2-3(ab+ac)+2bc)2)/14-8/14abcs(a2b+ab2-2abc)    (real)

    s(a2(b-c)4)-1/2p(a-b)2      (real)

    s(a2(b2-c2)2)-3/8p(a-b)2    (real, root = (-1,-1,1))

    3p(a2+ab+b2)-s(a)2s(ab)2    (real)

    p(a2+ab+b2)-3s(ab)s(a2b2)   (real)

    p(a2+ab+b2)+12a2b2c2-3p(a+b)2/5    (real, uncentered)

    Reference
    ---------
    [1] Vasile, Mathematical Inequalities Volume 1 - Symmetric Polynomial Inequalities. p.23
    """
    multipliers, y, names = [], None, None
    
    if coeff((4,2,0)) <= 0:
        if coeff((4,2,0)) == 0:
            return _sos_struct_sextic_hexagram_symmetric(coeff)
        return [], None, None
    
    # although subtracting p(a-b)2 always succeeds,
    # we can handle cases for real numbers and cases where updegree is not necessary
    if 3*(coeff((4,2,0))*2 + coeff((3,3,0)) + coeff((4,1,1)) + coeff((3,2,1))*2) + coeff((2,2,2)) == 0:
        # has root at (1,1,1)
        x_, y_, z_ = coeff((3,3,0)) / coeff((4,2,0)), coeff((4,1,1)) / coeff((4,2,0)), -coeff((3,2,1)) / coeff((4,2,0))
        
        if x_ < -2 or y_ < -2:
            return [], None, None
            
        if x_ == y_ and x_ <= 2:
            # try us(a2b+ab2-2abc)2 + (1-u)p(a-b)2
            # where 2u - 2(1-u) = x  =>  u = (x + 2) / 4
            y = [
                (x_ + 2) / 12,
                (2 - x_) / 12,
                z_ + 10 * (x_ + 2) / 12 - 2 * (2 - x_) / 12
            ]
            if y[-1] < 0:
                y = None
            else:
                y = [_ * coeff((4,2,0)) for _ in y]
                names = [
                    '(a^2*b+a^2*c+b^2*c+b^2*a+c^2*a+c^2*b-6*a*b*c)^2',
                    '(a-b)^2*(b-c)^2*(c-a)^2',
                    'a^2*b*c*(b-c)^2'
                ]
                return multipliers, y, names
                

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
                            w / 3,
                            (1 - w) / (x__**2 + y__**2 + 1) / rr**2
                        ]
                        y = [_ * coeff((4,2,0)) for _ in y]
                        names = [
                            '(a-b)^2*(b-c)^2*(c-a)^2',
                            f'(b-c)^2*({rr}*a^2-{rr*x__}*a*b-{rr*x__}*a*c+{rr*y__}*b*c)^2',
                        ]
                        return multipliers, y, names
        

    if y is None:
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
        
        multipliers = []
        y = [coeff((4,2,0)) / 3]
        names = ['(a-b)^2*(b-c)^2*(c-a)^2']
        result = _sos_struct_sextic_hexagram_symmetric(new_coeff)
        multipliers, y, names = _merge_sos_results(multipliers, y, names, result)

    return multipliers, y, names


def _sos_struct_sextic_hexagram_symmetric(coeff):
    """
    Solve s(a3b3+xa4bc+ya3b2c+ya2b3c+wa2b2c2) >= 0

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
    multipliers, y, names = [], None, None
    if coeff((3,3,0)) < 0 or coeff((4,1,1)) < 0:
        return [], None, None
    
    # first try trivial cases
    if True:
        x_ = min(coeff((3,3,0)), coeff((4,1,1)))
        y = [
            x_, 
            coeff((3,3,0)) - x_, 
            coeff((4,1,1)) - x_, 
            coeff((3,2,1)) + x_ + (coeff((3,3,0))) + (coeff((4,1,1))),
            (coeff((3,3,0)) + coeff((4,1,1)) + coeff((3,2,1)) * 2) + coeff((2,2,2))  / 3
        ]
        if any(_ < 0 for _ in y):
            y = None
        else:
            names = [
                'b*c*(a-b)^2*(a-c)^2',
                'b^2*c^2*(a-b)*(a-c)',
                'a^2*b*c*(a-b)*(a-c)',
                'a^2*b*c*(b-c)^2',
                'a^2*b^2*c^2'
            ]
            return multipliers, y, names
    
    if coeff((4,1,1)) != 0:
        x_ = coeff((3,3,0)) / coeff((4,1,1))
        y_ = coeff((3,2,1)) / coeff((4,1,1))
        w_ = coeff((2,2,2)) / coeff((4,1,1))
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
                names = [
                    f'a*b*c*(b-c)^2*({r}*b+{r}*c-{r*u_}*a)^2',
                    f'a*(b-c)^2*({r*(1-u_)}*a*b+{r*(1-u_)}*a*c+{r*u_}*b*c)^2',
                    'b^2*c^2*(a-b)*(a-c)*(a+b+c)',
                    'a^2*b*c*(b-c)^2*(a+b+c)',
                    'a^3*b^2*c^2'
                ]
                None


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

                    multipliers = ['a', 'a*b']
                    y = [w1, w2, z_ - z0]
                    y = [_ * coeff((3,3,0)) for _ in y]
                    names = [f'c*({-9*x_**2-3*x_*y_+18*x_}*a^3*b+{-9*x_**2+9*x_+y_**2-9}*a^2*b^2+{9*x_**2+3*x_*y_-3*y_-9}*a^2*b*c'
                            +f'+{-18*x_+3*y_+9}*a^2*c^2+{-9*x_**2-3*x_*y_+18*x_}*a*b^3+{9*x_**2+3*x_*y_-3*y_-9}*a*b^2*c'
                            +f'+{9*x_**2-9*x_-y_**2+9}*a*b*c^2+{-18*x_+3*y_+9}*b^2*c^2)^2',

                                f'c*(+{-3*phi1*x_}*a^3*b+{-3*phi1*x_+phi1*y_+3*phi1-3*phi2}*a^2*b^2+{-3*phi1*x_+phi1*y_+3*phi1+3*phi2*x_+phi2*y_-3*phi2}*a^2*b*c'
                            +f'+{-3*phi2}*a^2*c^2+{-3*phi1*x_}*a*b^3+{-3*phi1*x_+phi1*y_+3*phi1+3*phi2*x_+phi2*y_-3*phi2}*a*b^2*c'
                            +f'+{-3*phi1*x_+3*phi2*x_+phi2*y_-3*phi2}*a*b*c^2+{-3*phi2}*b^2*c^2)^2']
                        
                    names += [f'a^3*b^3*c^2*(a+b+c)']
    
    return multipliers, y, names


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
    multipliers, y, names = [], None, None

    t = coeff((6,0,0))
    # t != 0 by assumption
    u = coeff((3,3,0))/t
    v = coeff((4,1,1))/t
    if u < -2:
        return [], None, None

    if v != -6:
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
                multipliers = []
                y = [w1 * t / 2, w2 * t / p**2,
                    coeff((2,2,2))/3+coeff((6,0,0))+coeff((4,1,1))+coeff((3,3,0))]
                names = [
                    '(a+b+c)^2*(b-c)^4',
                    f'({p}*a^2+{p}*b^2+{p}*c^2+{q}*(a*b+b*c+c*a))*(a-b)^2*({p}*a+{p}*b-{q}*c)^2',
                    'a^2*b^2*c^2',
                ]
                return multipliers, y, names
    
    x = sp.symbols('x')
    roots = sp.polys.roots(x**3 - 3*x - u).keys()
    for r in roots:
        numer_r = complex(r)
        if abs(numer_r.imag) < 1e-12 and numer_r.real >= 1:
            numer_r = numer_r.real 
            if not isinstance(r, sp.Rational):
                for r in rationalize_bound(numer_r, direction = -1, compulsory = True):
                    if r*r*r-3*r <= u and 3*r*(r-1)+v >= 0:
                        break 
                else:
                    break 
            elif 3*r*(r-1)+v < 0:
                break 
            
            # now r is rational 
            y = [t/2, t*(u-(r*r*r-3*r)), t*(v+3*r*(r-1)), 
                coeff((2,2,2))/3+coeff((6,0,0))+coeff((4,1,1))+coeff((3,3,0))]
            names = [f'(a^2+b^2+c^2+{r}*(a*b+b*c+c*a))*(a-b)^2*(a+b-{r}*c)^2',
                    'a^3*b^3-a^2*b^2*c^2',
                    'a^4*b*c-a^2*b^2*c^2',
                    'a^2*b^2*c^2']

    return multipliers, y, names



def _sos_struct_sextic_iran96(coeff):
    """
    Give a solution to s(a5b+ab5-x(a4b2+a2b4)+ya3b3-za4bc+w(a3b2c+a2b3c)+..a2b2c2) >= 0

    Observe that g(a,b,c) = s(ab(a2-b2 + x(bc-ac))^2) >= 4((a-b)*(b-c)*(c-a))^2,
    which is because
        g(a,b,c)s(a) = s(c(a-b)2((a+b-c)2-(x-1)ab)2) + 2abcs((a2-b2+x(bc-ac))2)

    Also note that h(a,b,c) = s(a2bc(a2-b2+u(bc-ac))+ac((u-1)c+xb)(a-b)((u-1)(a+b)c-uab)) >= 0
    where x = (u^2 - u + 1) / (2*u - 1)

    In general, we can show that f(a,b,c) = g(a,b,c) + (t-2)^2/(u-1)^2 * h(a,b,c) >= 2*t * ((a-b)*(b-c)*(c-a))^2

    If we let x = (t*u - u*u + 2*u - 3) / (u - 1), then
        f(a,b,c)s(a) = s(a(b-c)^2(a^2+b^2+c^2-tab-tac+xbc)^2) + (t-2*u)^2/(2(u-1)^2) * s(abc(a^2-b^2+u(bc-ac))^2)

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
    """
    if not (coeff((6,0,0)) == 0 and coeff((5,1,0)) == coeff((1,5,0)) and coeff((4,2,0)) == coeff((2,4,0)) and\
        coeff((3,2,1)) == coeff((2,3,1)) and coeff((5,1,0)) >= 0):
        return [], None, None


    multipliers, y, names = [], None, None

    m, p, q, w, z = coeff((5,1,0)), coeff((4,2,0)), coeff((3,3,0)), coeff((4,1,1)), coeff((3,2,1))

    if m < 0:
        return multipliers, y, names
    elif m == 0:
        return _sos_struct_sextic_hexagon_symmetric(coeff, real = False)
    
    if w >= 0 and w + z >= 0:
        # Easy case 1, really trivial
        y = [
            m,
            p + m * 4,
            q - m * 6 + 2 * (w + m * 4),
            w,
            z + w,
            (coeff((2,2,2)) / 3 + (m + p + z) * 2 + q + w)
        ]
        
        if any(_ < 0 for _ in y):
            multipliers, y, names = [], None, None
        else:
            names = [
                'a*b*(a-b)^4',
                'a^2*b^2*(a-b)^2',
                'a^3*b^3-a^2*b^2*c^2',
                'a^2*b*c*(a-b)*(a-c)',
                'a^2*b*c*(b-c)^2',
                'a^2*b^2*c^2'
            ]
            return multipliers, y, names


    if p >= -4 * m and q + 2 * (p + m) >= 0:
        # Easy case 2, subtract enough p(a-b)2 and s(ab(a-c)2(b-c)2)
        # e.g. s(a(b+c)(b+c-2a)4)
        y = [
            m,
            (p + 4 * m) / 3,
            q + 2 * (p + m),
            w + 2 * (p + 4 * m) - (q + 2 * (p + m)),
            z + w + 3 * (q + 2 * (p + m)),
            (coeff((2,2,2)) / 3 + (m + p + z) * 2 + q + w)
        ]
        
        if any(_ < 0 for _ in y):
            multipliers, y, names = [], None, None
        else:
            names = [
                'a*b*(a-b)^4',
                '(a-b)^2*(b-c)^2*(c-a)^2',
                'b*c*(a-b)^2*(a-c)^2',
                'a^2*b*c*(a-b)*(a-c)',
                'a^2*b*c*(b-c)^2',
                'a^2*b^2*c^2'
            ]
            return multipliers, y, names

    
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
                p / 3,
                min(w2, q2),
                q2 - min(w2, q2),
                w2 - min(w2, q2),
                z - u_ * (u_ - 2) * m - 2*p + (w2 + q2 - min(w2, q2)),
                (coeff((2,2,2)) / 3 + (m + p + z) * 2 + q + w)
            ]

            if any(_ < 0 for _ in y):
                multipliers, y, names = [], None, None
            else:
                names = [
                    f'a*b*(a-b)^2*(a+b-{u_}*c)^2',
                    '(a-b)^2*(b-c)^2*(c-a)^2',
                    'b*c*(a-b)^2*(a-c)^2',
                    'b^2*c^2*(a-b)*(a-c)',
                    'a^2*b*c*(a-b)*(a-c)',
                    'a^2*b*c*(b-c)^2',
                    'a^2*b^2*c^2'
                ]
                return multipliers, y, names

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
                u_ / 3,
                v_,
                -m - 2*p + 4*u_ - v_ + w,
                -p + u_ + 2*v_ + w + z,
                (coeff((2,2,2)) / 3 + (m + p + z) * 2 + q + w)
            ]

            if any(_ < 0 for _ in y):
                multipliers, y, names = [], None, None
            else:
                names = [
                    f'(a*b+b*c+c*a)*(a-b)^2*(a+b-{-(p-u_)/m}*c)^2',
                    '(a*b+b*c+c*a)*a^2*(b-c)^2',
                    '(a-b)^2*(b-c)^2*(c-a)^2',
                    'b*c*(a-b)^2*(a-c)^2',
                    'a^2*b*c*(a-b)*(a-c)',
                    'a^2*b*c*(b-c)^2',
                    'a^2*b^2*c^2'
                ]
                return multipliers, y, names

    # real start below
    # process:
    p, q, w, z = p / m, q / m, w / m, z / m

    if not ((p <= -4 and q >= p*p/4 + 2) or (p >= -4 and q >= -2*(p + 1))):
        # checking the border yields that:
        # when p <= -4, we must have q >= p*p / 4 + 2
        # when p >= -4, we must have q >= -2*(p + 1)

        # both two cases should imply 2*p + q + 2 >= 0
        return [], None, None
    
    # First, we peek whether there are nontrivial roots in the interior with a == b.
    a, u = sp.symbols('a u')
    sym = ((2*p + q + 2)*a**3 + (4*p + 2*q + 2*w + 2*z + 6)*a**2 + (2*p + w + 4)*a + 2).as_poly(a)
    # sym should be nonnegative when a >= 0
    sym_roots_count = sp.polys.count_roots(sym, 0, None)
    root, u_ = None, None
    if sym_roots_count > 1:
        return [], None, None
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
            return [], None, None

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
        return [], None, None

    if u_ != 1:
        phi = (t * u_ - u_**2 + 2*u_ - 3) / (u_ - 1)

        multipliers = ['a']
        y = [
            y_hex if root is None else sp.S(0),
            r,
            w - (t**2 + 4*t*u_**2 - 8*t*u_ - 4*u_**3 + 8*u_**2 - 4*u_ + 4)/(u_ - 1)**2,
            z - coeff_z(u_),
            (coeff((2,2,2)) / 3 + (coeff((5,1,0)) + coeff((4,2,0)) + coeff((3,2,1))) * 2 \
                + coeff((3,3,0)) + coeff((4,1,1))) / m,
            sp.S(1),
            (t - 2*u_) ** 2 / 2 / (u_ - 1)**2 + (sp.S(0) if root is None else y_hex / 2 / (u_ - 1)**2),
            sp.S(0) if root is None else 1 / (u_ - 1)**2 * y_hex
        ]
        # print(r, t, u_, phi, y)

        if any(_ < 0 for _ in y):
            return [], None, None

        y = [_ * m for _ in y]
        names = [
            'a*b*(a-c)^2*(b-c)^2*(a+b+c)',
            'a*(a-b)^2*(b-c)^2*(c-a)^2',
            'a*b*c*(a^3-a*b*c)*(a+b+c)',
            'a^2*b*c*(b-c)^2*(a+b+c)',
            'a^2*b^2*c^2*(a+b+c)',
            f'a*(b-c)^2*(a^2+b^2+c^2-{t}*a*b-{t}*a*c+{phi}*b*c)^2',
            f'a*b*c*(b-c)^2*(b+c-{u_}*a)^2',
            f'c*(a-b)^2*({u_}*a*b-{u_-1}*a*c-{u_-1}*b*c)^2'
        ]

    elif u_ == 1:
        # very special case, it must be t == 2
        # f(a,b,c) = (s(ab(a-b)2(a+b-c)2)-4p(a-b)2)
        # then f(a,b,c)s(a) = s(a(b-c)2(b+c-a)4) + 2abcs((b-c)2(b+c-a)2)
        
        multipliers = ['a']
        y = [
            y_hex if root is None else sp.S(0),
            r,
            w - 4,
            z + w + 1,
            (coeff((2,2,2)) / 3 + (coeff((5,1,0)) + coeff((4,2,0)) + coeff((3,2,1))) * 2 \
                + coeff((3,3,0)) + coeff((4,1,1))) / m,
            sp.S(1),
            sp.S(2)
        ]

        if any(_ < 0 for _ in y):
            return [], None, None

        y = [_ * m for _ in y]
        names = [
            'a*b*(a-c)^2*(b-c)^2*(a+b+c)',
            'a*(a-b)^2*(b-c)^2*(c-a)^2',
            'a^2*b*c*(a-b)*(a-c)*(a+b+c)',
            'a^2*b*c*(b-c)^2*(a+b+c)',
            'a^2*b^2*c^2*(a+b+c)',
            f'a*(b-c)^2*(b+c-a)^4',
            f'a*b*c*(b-c)^2*(b+c-a)^2'
        ]

    return multipliers, y, names



def _sos_struct_sextic_symmetric_ultimate(coeff, poly, recurrsion):
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
    s(38a6-148a5b-148a5c+225a4b2+392a4bc+225a4c2-210a3b3-320a3b2c-320a3bc2+266a2b2c2)
    """

    coeff6 = coeff((6,0,0))
    if not (coeff((5,1,0)) == coeff((1,5,0)) and coeff((4,2,0)) == coeff((2,4,0)) and coeff((3,2,1)) == coeff((2,3,1))):
        # asymmetric
        return [], None, None
    elif coeff6 == 0:
        # degenerated
        return _sos_struct_sextic_iran96(coeff)
    elif coeff6 < 0:
        return [], None, None

    if coeff((5,1,0)) == 0 and coeff((4,2,0)) == 0 and coeff((3,2,1)) == 0:
        return _sos_struct_sextic_tree(coeff)
    
    multipliers, y, names = [], None, None

    a, u, v, w, x, z = sp.symbols('a u v w x z')

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
                return [], None, None
            
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
                return [], None, None
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

    return [], None, None

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
    multipliers, y, names = [], None, None
    coeff6 = coeff((6,0,0))
    a, b, c = sp.symbols('a b c')
    if roots[0] is not None:
        # Case A.
        # subtract some s(a3-abc-x(a2b+ab2-2abc))2
        x_ = sp.simplify(roots[0] + 1/roots[0]) - 1
        if not isinstance(x_, sp.Rational):
            return [], None, None

        # 1. try subtracting all the s(a6)
        # e.g. s(a2/3)3-a2b2c2-p(a-b)2
        if coeff((5,1,0)) >= -2 * x_:
            poly2 = poly - ((a**3+b**3+c**3-3*a*b*c-x_*(a*a*(b+c)+b*b*(c+a)+c*c*(a+b)-6*a*b*c))**2).as_poly(a,b,c) * coeff6
            coeff2, _ = _make_coeffs_helper(poly2, 6)
            result = _sos_struct_sextic_iran96(coeff2)
            if result is not None and result[1] is not None:
                multipliers = []
                if x_ == sp.S(3)/2:
                    y = [coeff6 / 12]
                    names = ['(a+b-2*c)^2*(b+c-2*a)^2*(c+a-2*b)^2']
                elif x_ == 1:
                    y = [coeff6 / 3]
                    names = [f'(a*(a-b)*(a-c)+b*(b-c)*(b-a)+c*(c-a)*(c-b))^2']
                else:
                    y = [coeff6 / 3]
                    names = [f'(a^3+b^3+c^3-{x_}*a^2*b-{x_}*a^2*c-{x_}*b^2*c-{x_}*b^2*a-{x_}*c^2*a-{x_}*c^2*b+{6*x_-3}*a*b*c)^2']
                    
                result = _merge_sos_results(multipliers, y, names, result)
                return result

        # until the symmetric axis is touched
        # # the subtractor = (a-1)^4 * (2(x-1)a - 1)^2 on the symmetric axis a == b and c == 1
        # sym = poly.subs(c,1).subs(b,a).factor() / (a - 1)**4 / (2*(x-1)*a - 1)**2


    return multipliers, y, names


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

    Reference
    -------
    [1] Vasile, Mathematical Inequalities Volume 1 - Symmetric Polynomial Inequalities. 3.78
    
    [2] https://artofproblemsolving.com/community/c6t243f6h3013463_symmetric_inequality

    [3] https://tieba.baidu.com/p/8261574122
    """
    multipliers, y, names = [], None, None
    coeff6 = coeff((6,0,0))

    
    if roots[2] is None:
        # Case (A + B)
        a, b, c = sp.symbols('a b c')
        if roots[0] != 1:
            multipliers = []
            y = [coeff6 / 3]
            names = [f'(a^3+b^3+c^3-{(1+1/roots[0]/2)}*(a^2*b+a^2*c+b^2*c+b^2*a+c^2*a+c^2*b)+{(3+3/roots[0])}*a*b*c)^2']
            diffpoly = (a**3+b**3+c**3 - (1+1/roots[0]/2)*(a*a*(b+c)+b*b*(c+a)+c*c*(a+b)-6*a*b*c) - 3*a*b*c)**2
            diffpoly = diffpoly.as_poly(a,b,c)

        elif roots[0] == 1:
            x = 1 / roots[1]
            if x > 4:
                return [], None, None

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
                diffpoly = (a*a+b*b+c*c-a*b-b*c-c*a)*((a*a+b*b+c*c-2*a*b-2*b*c-2*c*a))**2
                diffpoly = diffpoly.as_poly(a,b,c)
                multipliers = []
                y = [coeff6 / 2]
                names = ['(a-b)^2*(a*a+b*b+c*c-2*a*b-2*b*c-2*c*a)^2']
            elif x == 1:
                diffpoly = (a*(a-b)*(a-c)+b*(b-c)*(b-a)+c*(c-a)*(c-b))**2
                diffpoly = diffpoly.as_poly(a,b,c)
                multipliers = []
                y = [coeff6 / 3]
                names = ['(a*(a-b)*(a-c)+b*(b-c)*(b-a)+c*(c-a)*(c-b))^2']
            elif x > 1:
                diffpoly = get_diffpoly(x)
                multipliers = [f'a^2+{x-2}*a*b']
                y = [(4-x)*(x-1)/2 * coeff6, coeff6 / 3]
                names = [
                    f'a^2*(b-c)^2*(a^2+b^2+c^2-2*a*b-2*a*c+{2-x}*b*c)^2',
                    f'(b-c)^2*({2*x-x*x}*a*b*c-(a+b-c)*(b+c-a)*(c+a-b))^2'
                ]
            elif x < 1:
                if x <= 0:
                    # x < 0 is no stronger than x == 0
                    # x == 0 corresponds to the case s(a4(a-b)(a-c)) - 5p(a-b)2
                    diffpoly = ((a**4*(a-b)*(a-c)+b**4*(b-c)*(b-a)+c**4*(c-a)*(c-b)) \
                                    - 5 * ((a-b)*(b-c)*(c-a))**2).as_poly(a,b,c)
                    x = 0
                else:
                    diffpoly = get_diffpoly(x)

                multipliers = ['a^2*(b-c)^2']
                y = [coeff6, coeff6, 2*(2 - x)*coeff6, 12*(1 - x)*coeff6]
                names = [
                    f'a*b*c*(a*(a-b)*(a-c)+b*(b-c)*(b-a)+c*(c-a)*(c-b))*(b-c)^2*(b+c-{x+1}*a)^2',
                    '(a-b)^4*(b-c)^2*(c-a)^2*(a+b-c)^2',
                    'a*b*(a-b)^4*(b-c)^2*(c-a)^2',
                    'a^2*b*c*(a-b)^2*(b-c)^2*(c-a)^2'
                ]

        new_poly = poly - coeff6 * diffpoly
        if not new_poly.is_zero:
            new_coeff, _ = _make_coeffs_helper(new_poly, 6)
            result = _sos_struct_sextic_iran96(new_coeff)
            result = _merge_sos_results(
                multipliers = multipliers,
                y = y,
                names = names,
                result = result,
                keepdeg = True
            )
        else:
            result = (multipliers, y, names)
        return result

    elif roots[0] is None:
        # Case (B + C)
        pass

    return multipliers, y, names