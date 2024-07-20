from typing import Tuple, List

import sympy as sp

from .utils import (
    CyclicSum, CyclicProduct, Coeff,
    congruence, sum_y_exprs, quadratic_weighting, radsimp,
    nroots, rationalize, rationalize_bound, rationalize_func,
    univariate_intervals, common_region_of_conics
)

a, b, c = sp.symbols('a b c')

def sos_struct_quartic(coeff, recurrsion = None, real = True):
    """
    Solve cyclic quartic problems.

    The function only uses `coeff`. The `poly` and `recurrsion` is not used for minimium dependency.

    Core theorem:
    If f(a,b,c) = s(a4 + pa3b + na2b2 + qab3 - (1+p+n+q)a2bc) >= 0 holds for all a,b,c >= 0, 
    then it must be one of the following two cases:

    A. If [3*(1+n) - (p*p+p*q+q*q)] >= 0, then
        f(a,b,c) = s((a2 - b2 + (p+2*q)/3*(a*c-a*b) - (2*p+q)/3*(b*c-a*b))^2)
                    + (3*(1+n)-(p*p+p*q+q*q))/6 * s(a^2*(b-c)^2)

    B. There exists a positive root of the quartic 2*u^4 + p*u^3 - q*u - 2 = 0,
        such that t = ((2*q+p)*u^2 + 6*u + (2*p+q)) / (2*(u^4 + u^2 + 1)) >= 0
        and the following polynomial
            g(a,b,c) = f(a,b,c) - t * s(ab(a-c - u(b-c))^2) 
        must satisfy the condition A.
        
    Examples
    -------
    s(a2)2-3s(a3b)

    4s(a)s(2a3-a2b-a2c)
    """
    solution = None

    m, p, n, q = coeff((4,0,0)), coeff((3,1,0)), coeff((2,2,0)), coeff((1,3,0))
    if m > 0:
        if m + p + n + q + coeff((2,1,1)) > 0:
            solution = _sos_struct_quartic_uncentered(coeff)
            if solution is not None:
                return solution


        solution = _sos_struct_quartic_core(coeff)
        if solution is not None:
            return solution

        else:
            if coeff.is_rational and p == -q:
                # handle some special case in advance
                solution = _sos_struct_quartic_quadratic_border(coeff)
                if solution is not None:
                    return solution

            return _sos_struct_quartic_biased(coeff)
        
    elif m == 0:
        return _sos_struct_quartic_degenerate(coeff)

    return solution


def _sos_struct_quartic_core(coeff):
    """
    Main theorem:
    f(a,b,c) = s(a4 + pa3b + na2b2 + qab3 - (1+p+n+q)a2bc)
    If [3*(1+n) - (p*p+p*q+q*q)] >= 0, then
        f(a,b,c) = s((a2 - b2 + (p+2*q)/3*(a*c-a*b) - (2*p+q)/3*(b*c-a*b))^2)
                    + (3*(1+n)-(p*p+p*q+q*q))/6 * s(a^2*(b-c)^2)
                >= 0

    Examples
    -------
    s(a2)2-3s(a3b)

    s(a2)2-3s(ab3)

    s((a-2b+c)2(5a-b-7c)2)

    s((a2-b2-ac+ab)2)
    """
    m, p, n, q = coeff((4,0,0)), coeff((3,1,0)), coeff((2,2,0)), coeff((1,3,0))
    det = radsimp(3*m*(m+n) - (p*p + p*q + q*q))
    if m < 0 or det < 0:
        return None
    elif m == 0: # and det >= 0, so it must be p,q == 0
        return _sos_struct_quartic_degenerate(coeff)

    y = radsimp([
        m/2,
        det/6/m,
        coeff((2,1,1)) + m + p + n + q
    ])

    if y[-1] < 0:
        return None

    c1, c2, c3 = (p+2*q)/m/3, (p-q)/m/3, -(2*p+q)/m/3
    c1, c2, c3 = radsimp([c1, c2, c3])
    exprs = [
        CyclicSum((a**2 - b**2 + c1*a*c + c2*a*b + c3*b*c)**2),
        CyclicSum(a**2*(b-c)**2),
        CyclicSum(a**2*b*c)
    ]

    return sum_y_exprs(y, exprs)


def _sos_struct_quartic_quadratic_border(coeff):
    """
    Give a solution to s(a4 - 2t(a3b - ab3) + (t^2 - 2)(a2b2 - a2bc) - a2bc) >= 0,
    which has nontrivial zeros (a, b, c) = ((sqrt(t*t + 4) - t)/2, 0, 1) 

    f(a,b,c)s(a) = s(a(b2+c2-a2-bc+t(ab-ac))2) + (t^2 + 3)/2*abcs((b-c)2)

    Examples
    --------
    s(a4-2a3b+2ab3-a2b2)

    s(a4+4a3b-4ab3+2a2b2-3a2bc)

    s(a4-3a3b+3ab3+1/4a2b2-5/4a2bc)

    s(a4 - 2sqrt(2)(a3b - ab3) - a2bc)
    """
    m, p, n, q = coeff((4,0,0)), coeff((3,1,0)), coeff((2,2,0)), coeff((1,3,0))

    t = q / 2 / m
    if p == -q and n == (t**2 - 2) * m:
        w = sp.sqrt(t**2 + 4)
        if not isinstance(w, sp.Rational):
            y = [m, (t**2 + 3) / 2 * m, (coeff((2,1,1)) + m + p + n + q)]
            if y[-1] >= 0:
                exprs = [
                    CyclicSum(a*(b**2+c**2-a**2-b*c+t*a*b-t*a*c)**2),
                    CyclicSum((b-c)**2) * CyclicProduct(a),
                    CyclicSum(a)**2 * CyclicProduct(a)
                ]
                return sum_y_exprs(y, exprs) / CyclicSum(a)
    
        # if it is rational, then fall back to normal mode
        # u = (w + t) / 2
        # r = (2*t*(u**2 - 1) + 6*u) / (2*(u**4 + u**2 + 1))

    return None


def _sos_struct_quartic_biased(coeff):
    """
    Core theorem:
    If f(a,b,c) = s(a4 + pa3b + na2b2 + qab3 - (1+p+n+q)a2bc) >= 0 holds for all a,b,c >= 0,
    and if [3*(1+n) - (p*p+p*q+q*q)] < 0, 
    then there exists a positive root of the quartic 2*u^4 + p*u^3 - q*u - 2 = 0,
    such that t = ((2*q+p)*u^2 + 6*u + (2*p+q)) / (2*(u^4 + u^2 + 1)) >= 0
    and the following polynomial
        g(a,b,c) = f(a,b,c) - t * s(ab(a-c - u(b-c))^2)
    must satisfy the main theorem.

    Examples
    --------
    s(a4-3a3b+3ab3+1/4a2b2-5/4a2bc)

    (9s(a3)+24abc-17s(a2c))s(a)

    s(a)s(a(a-b)(a-c))

    10s(a4-4a3b+2ab3+a2b2)+19s(a2b2-a2bc)

    s((2a+b)(a-sqrt(2)b)2(a+b)-2a4+(12sqrt(2)-16)a2bc)
    """
    m, p, n, q = coeff((4,0,0)), coeff((3,1,0)), coeff((2,2,0)), coeff((1,3,0))

    # try subtracting t*s(ab(a-c-u(b-c))2) and use the theorem
    # solve all extrema
    n, p, q = n / m, p / m, q / m
    n, p, q = radsimp([n, p, q])
    x, u_ = sp.symbols('x'), None
    eq = (2*x**4 + p*x**3 - q*x - 2).as_poly(x)
    
    # must satisfy that symmetric >= 0
    symmetric = lambda _x: radsimp(((2*q+p)*_x + 6)*_x + 2*p+q)
 
    # the discriminant after subtraction
    head = radsimp(p*p+p*q+q*q-3*n-3)
    def new_det(sym, root):
        return head - sym**2/(4*(root**2*(root**2 + 1) + 1))


    if True:
        # check whether there exists multiplicative roots on the border
        eq_diff = (x**4 + p*x**3 + n*x**2 + q*x + 1).as_poly(x)
        eq_gcd = sp.gcd(eq, eq_diff)
        if eq_gcd.degree() == 1:
            u_ = radsimp(-(eq_gcd.all_coeffs()[1] / eq_gcd.LC()))
            if u_ < 0:
                u_ = None
        elif eq_gcd.degree() == 2:
            c2, c1, c0 = radsimp(eq_gcd.all_coeffs())
            if c2 < 0:
                c2, c1, c0 = -c2, -c1, -c0
            delta = radsimp(c1**2 - 4*c2*c0)
            if delta >= 0:
                u_ = radsimp((-c1 + sp.sqrtdenest(sp.sqrt(delta))) / (2*c2))

    if u_ is None:
        def _is_valid(u):
            sym_axis = symmetric(u)
            return sym_axis >= 0 and new_det(sym_axis, u) <= 0
        u_ = rationalize_func(eq, _is_valid)

    if u_ is not None:
        y_ = radsimp((symmetric(u_) / (2*(u_**2*(u_**2 + 1) + 1)) * m))
        if y_ >= 0:
            solution = y_ * CyclicSum((a*b*(a-u_*b+(u_-1)*c)**2))
            
            def new_coeff(d):
                subs = {(4,0,0): 0, (3,1,0): 1, (2,2,0): -2*u_, (1,3,0): u_**2, (2,1,1): 2*u_-u_**2-1}
                return coeff(d) - y_ * subs[d]

            new_solution = _sos_struct_quartic_core(new_coeff)
            if new_solution is not None:
                return solution + new_solution

    return None


def _sos_struct_quartic_degenerate(coeff):
    """
    Given the solution to f(a,b,c) = s(pa3b + na2b2 + qab3 + ra2bc) >= 0.
    Note that the coefficient of a^4 is zero in this case.

    Since we must have p,q >= 0 and n >= -sqrt(4*p*q) and p + n + q + r >= 0.
        f(a,b,c) = s(ab(sqrt(p)(a-c) - sqrt(q)(b-c))^2) + (n+2sqrt(pq))s(a2b2-a2bc) + (p+n+q+r)s(a2bc)

    Examples
    -------
    2/5s(ab(3a-2b-c)2)

    s(2a3b+a3c-a2b2-2a2bc)

    s(a3b-14/5a2b2+2ab3-1/5a2bc)

    s(a2b2+3ab3-4a2bc)
    """
    m, p, n, q, r = coeff((4,0,0)), coeff((3,1,0)), coeff((2,2,0)), coeff((1,3,0)), coeff((2,1,1))

    if m == 0 and p == 0 and q == 0 and n >= 0 and n + r >= 0 and r <= 2*n:
        # in this special case the polynomial is positive for real numbers
        w1 = radsimp(2*n - r) / 3
        w2 = radsimp(n - w1)
        return w1 / 2 * CyclicSum(a**2*(b-c)**2) + w2 * CyclicSum(a*b)**2


    if m == 0 and p >= 0 and q >= 0 and p + n + q + r >= 0 and (n >= 0 or n**2 <= 4*p*q):
        rem = radsimp(p + n + q + r) * CyclicSum(a**2*b*c)
        if n >= 0:
            # very trivial in this case and we can only use AM-GM
            part = ((n/2) * a + q*b + p*c).together().as_coeff_Mul()
            solution = part[0] * CyclicSum(a*(b-c)**2*part[1])

        else:
            # if n < 0, we must have p > 0 and q > 0
            mapping = lambda x, y: CyclicSum(b*c*(x*(a-b) + y*(a-c))**2)
            solution = quadratic_weighting(p, n, q, mapping = mapping)

        return solution + rem

    return None


def _sos_struct_quartic_uncentered(coeff, recur = False):
    """
    Solve quartic problems which do not have zeros at (1,1,1) but elsewhere.

    Idea: subtract some z*(s(a^2-w*ab))^2 so that it has zero at (1,1,1).

    Note: sometimes the inequality holds for all real numbers (a, b, c),
    in this case we should present a solution that handles that.

    Examples
    -------
    (s(a2)2-3s(a3b))/3+s(2a2-3ab)2/6

    s(a4-3a3b-2ab3+4a2b2+2/9a2bc)

    s(2a2-3ab)2
    
    s(2a2-3ab)2+s(ab3-a2bc)         (real numbers)

    s(2a2-3ab)2+s(ab3-5/3a2bc)      (real numbers)

    s(2a2-3ab)2+s(ab3+81/5a2bc)     (real numbers)

    s(4a4-5a3b+7a2b2-9ab3+4a2bc)    (real numbers)

    115911s(a/6)4-s(a3b-32/3a2bc)   (real numbers with equality)

    s(a2)2-2s(a2bc)                 (real numbers)

    s(a2)2+6s(a2bc)-s(ab3)          (real numbers)

    s(a4-3a3b-2ab3+7/2a2b2+5a2bc)

    4s(a4)+11abcs(a)-8s(ab)s(a2-ab)

    s(a4-a3b-2ab3+11/10a2b2)+2abcs(a)

    s(a)3s(a)-s(a2)s(a)2/3

    (567+45sqrt(105))/32/81s(a)4-s(a3b)

    s(a2)2-16sqrt(2)/9s(bc(b2-c2))    (real numbers)

    References
    -------
    [1] https://tieba.baidu.com/p/8069929018

    [2] https://tieba.baidu.com/p/8241977884
    """
    m, p, n, q, r = coeff((4,0,0)), coeff((3,1,0)), coeff((2,2,0)), coeff((1,3,0)), coeff((2,1,1))
    if m > 0 and m + p + n + q + r > 0:
        p, n, q, r = p / m, n / m, q / m, r / m
        p, n, q, r = radsimp([p, n, q, r])
        s = radsimp(3*(1 + p + n + q + r))

        eq_coeffs = radsimp([
            -27*n + 9*p**2 + 9*p*q + 9*q**2 + 3*s - 27,
            81*n - 27*p**2 - 27*p*q + 6*p*s - 27*q**2 + 6*q*s - 3*s + 81,
            3*n*s - 81*n + 27*p**2 + 27*p*q - 6*p*s + 27*q**2 - 6*q*s + s**2 + 12*s - 81,
            -3*n*s + 27*n - 9*p**2 - 9*p*q - 9*q**2 + s**2 - 12*s + 27
        ])
        
        w, w_ = sp.symbols('w'), None

        # the equation is exactly the discriminant of the quartic after subtracting z*s(a2 - w*ab)2
        # find some w such that eq * (w - 1) <= 0
        # it would be best if we can let eq == 0
        eq = sum(eq_coeffs[i] * w ** (3 - i) for i in range(4)).as_poly(w)

        if coeff.is_rational:
            for root in sp.polys.roots(eq, cubics = False, quartics = False, quintics = False):
                if root.is_real and isinstance(root, sp.Rational) and root != 1:
                    if root != 1 and s <= 9 * (1 - root)**2:
                        w_ = root
                        break
        elif not coeff.is_rational:
            # check whether there exists multiplicitive root
            eq_diff = eq.diff(w)
            eq_gcd = sp.gcd(eq, eq_diff)
            if eq_gcd.degree() == 1:
                w_ = radsimp(-(eq_gcd.all_coeffs()[1] / eq_gcd.LC()))

        if w_ is None:
            # find rational approximation of w such that eq * (w - 1) <= 0
            # (eq is the discriminant)

            # note that: though w has no rational root, but when it takes the irrational root
            # we can see that the equality holds for all REAL numbers.
            # We must restore the solution to real numbers.

            if eq_coeffs[0] > 0:
                # the leading coefficient of the cubic equation is positive
                roots = nroots(eq, real = True, method = 'factor')

                # since they are the real roots of a cubic
                # and the cubic has no multiplicative roots (which will be detected as rationals otherwise)
                # we assert len(roots) == 1 or len(roots) == 3
                if len(roots) == 3:
                    # since eq(1) = 2*s^2 >= 0, one of the root must < 1
                    if roots[2] > 1:
                        roots = ((roots[2], -1), (roots[1], 1), (roots[0], 1))
                    else:
                        roots = ((roots[0], 1), (roots[1], -1), (roots[2], 1))
                else:
                    # len(roots) == 1:
                    # the root must < 1
                    roots = ((roots[0], 1), )

                for root, direction in roots:
                    if s <= 9 * (1 - root)**2:
                        # apply small perturbation
                        for w_ in rationalize_bound(root, direction = direction, compulsory = True):
                            if w_ != 1 and s <= 9 * (1 - w_)**2 and eq.subs(w, w_) * (w_ - 1) <= 0:
                                break
                            w_ = None
                        else:
                            continue
                        break

            elif eq_coeffs[0] < 0:
                # the leading coefficient of the cubic equation is negative
                # in this case it is trivial, we can choose any large enough w
                roots = nroots(eq, real = True, method = 'factor')
                root = max(roots) # it must be root > 1
                root = max(1 + sp.sqrt(s) / 3, root)

                w_ = sp.floor(root + 2)

            elif eq_coeffs[0] == 0:
                # in this case, 3*(1+n) - (p*p + p*q + q*q) = s / 3 >= 0
                if eq_coeffs[1] > 0:
                    # select small enough w
                    roots = nroots(eq, real = True, method = 'factor')
                    root = min(roots) if len(roots) else 0
                    root = min(1 - sp.sqrt(s) / 3, root)
                    w_ = sp.floor(root - 2)
                
                elif eq_coeffs[1] < 0:
                    # select large enough w
                    roots = nroots(eq, real = True, method = 'factor')
                    root = max(roots) if len(roots) else 2
                    root = max(1 + sp.sqrt(s) / 3, root)
                    w_ = sp.floor(root + 2)

                elif eq_coeffs[1] == 0:
                    # we can prove that in this case it must be p + q == -1
                    # (or p*p+p*q+q*q==3*(1+n) and s == 0, which has been excluded)

                    # in this case it must be coeffs[2] > 0

                    # in this case, actually w = \infty,
                    # so we need to subtract s(ab)2

                    # in this case, we must have
                    # r = 8*n-3*p*p-3*p+6
                    # (n+r) / 6 >= (3*n-p*p-p+2) / 3 >= 0
                    y_ = radsimp((3*n-p*p-p+2) / 3 * m)
                    solution = y_ * CyclicSum(a*b)**2

                    def new_coeff(d):
                        subs = {(4,0,0): 0, (3,1,0): 0, (1,3,0): 0, (2,2,0): 1, (2,1,1): 2}
                        return coeff(d) - y_ * subs[d]
                    
                    new_solution = _sos_struct_quartic_core(new_coeff)
                    if new_solution is not None:
                        return solution + new_solution

        if w_ is not None:
            y_ = radsimp(s / (1 - w_) ** 2 / 9 * m)
            solution = y_ * CyclicSum(a**2-w_*a*b)**2

            def new_coeff(d):
                subs = {(4,0,0): 1, (3,1,0): -2*w_, (2,2,0): w_**2+2, (1,3,0): -2*w_, (2,1,1): 2*w_*(w_-1)}
                return coeff(d) - y_ * subs[d]

            new_solution = _sos_struct_quartic_core(new_coeff)
            if new_solution is not None:
                return solution + new_solution
            
    
        if recur:
            return None

        # if we reach here, it means that the inequality does not hold for all real numbers
        # we can subtract as many s(a2bc) as possible
        if radsimp(3*(1 + n) - (p*p + p*q + q*q)) >= 0:
            # Case 1. 3m(m+n) - (p^2+pq+q^2) >= 0,
            # then it is directly handled by the core function
            return _sos_struct_quartic_core(coeff)
        else:
            # assume the inequality holds, then on the border it must >= 0
            if 2*p + q >= 0 or 2*q + p >= 0 or radsimp((2*p+q)*(2*q+p)) <= 9:
                # then it falls to the biased case
                return _sos_struct_quartic_biased(coeff)

            # now that we assume 2p + q < 0 and 2q + p < 0 and (2p+q)(2q+p) > 9
            # first compute minimum bound for n on the border
            # n >= -sup(x^2 + 1/x^2 + px + q/x) over x > 0
            if p == q:
                if p >= -4:
                    n_ = -2 - 2*p
                else:
                    n_ = p**2 / 4 + 2
            else:
                x = sp.symbols('x')
                eqx = (2*x**4 + p*x**3 - q*x - 2).as_poly(x) # the derivative of n
                eqn = lambda x: -(x*x + 1/x/x + p*x + q/x)
                extrema = []
                for root in sp.polys.roots(eqx, cubics = False, quartics = False):
                    if root.is_real and root > 0:
                        if isinstance(root, sp.Rational):
                            extrema.append((eqn(root), root))
                        else: # quadratic root
                            extrema.append((eqn(root.n(16)), root.n(16)))
                
                try:
                    for root in sp.polys.nroots(eqx):
                        if root.is_real and root > 0:
                            if any(abs(_[1] - root) < 1e-13 for _ in extrema):
                                # already found
                                continue
                            extrema.append((eqn(root), root))
                except: pass

                n_, _ = max(extrema)

            # then we compute x such that
            # s(ma4+pa3b+na2b2+qab3) + xs(a2bc) >= 0 is strict

            x_ = None
            if p == q:
                if p >= -4:
                    x_ = -3*(p + 1)
                else:
                    x_ = radsimp(9*(p + 2)**2 / 4)
            else:
                u = sp.symbols('u')
                det_coeffs = radsimp([
                    1,
                    -3*(p*q + 14*p + 14*q - 20),
                    9*(-6*n_**2 + n_*p**2 - n_*p*q + 26*n_*p + n_*q**2 + 26*n_*q - 152*n_ + 12*p**2*q + 57*p**2 + 12*p*q**2 + 63*p*q - 6*p + 57*q**2 - 6*q - 162),
                    -27*(8*n_**3 - 2*n_**2*p**2 - n_**2*p*q - 22*n_**2*p - 2*n_**2*q**2 - 22*n_**2*q - 212*n_**2 + 10*n_*p**3 + 24*n_*p**2*q + 84*n_*p**2 \
                        + 24*n_*p*q**2 + 18*n_*p*q - 252*n_*p + 10*n_*q**3 + 84*n_*q**2 - 252*n_*q - 560*n_ + p**4 + 34*p**3*q + 86*p**3 + 39*p**2*q**2 \
                        + 192*p**2*q + 114*p**2 + 34*p*q**3 + 192*p*q**2 + 51*p*q - 278*p + q**4 + 86*q**3 + 114*q**2 - 278*q - 388),
                    81*(n_ + 2*p + 2*q + 5)**3*(-3*n_ + p**2 + p*q + q**2 - 3)
                ])
                det = sum(det_coeffs[i] * u ** (4 - i) for i in range(5)).as_poly(u)

                if isinstance(n_, sp.Rational):
                    for root, mul in sp.polys.roots(det, cubics = False, quartics = False).items():
                        if mul == 2 and root.is_real and root > 0:
                            if isinstance(root, sp.Rational):
                                x_ = root
                            else:
                                x_ = root.n(16)
                            break
                    
                else:
                    # do not compute the root here because it is not numerically stable
                    # instead compute the root of derivative of discriminant
                    detdiff = det.diff()
                    roots = [(abs(det(root)), root) for root in sp.polys.nroots(detdiff) if root.is_real and root > 0]
                    # print(roots)
                    if len(roots) > 0:
                        x_ = min(roots)[1]
                    
            if x_ is not None:
                x_ = x_ / 3 - 1 - p - q - n_
                # print('n =', n_, 'x = ', x_)

                def _try_solve(x_):
                    new_coeffs_ = {(4,0,0): coeff((4,0,0)), (3,1,0): coeff((3,1,0)),
                                (2,2,0): coeff((2,2,0)), (1,3,0): coeff((1,3,0)), (2,1,1): x_ * m}
                    solution = _sos_struct_quartic_uncentered(Coeff(new_coeffs_, is_rational = coeff.is_rational), recur = True)
                    if solution is not None:
                        solution += radsimp((r - x_) * m) * CyclicSum(a**2*b*c)
                        return solution
                    return None


                # finally subtract enough s(a2bc) to xs(a2bc) (or near xs(a2bc))
                if isinstance(x_, sp.Float):
                    for x2 in rationalize_bound(x_, direction = 1, compulsory = True):
                        if x2 < r:
                            solution = _try_solve(x2)
                            if solution is not None:
                                return solution
                else:
                    return _try_solve(x_)

            return None

    return None




#####################################################################
#
#                              Acyclic
#
#####################################################################

def sos_struct_acyclic_quartic(coeff, recurrsion = None, real = True):
    """
    Solve acyclic quartic problems.
    """
    return _sos_struct_acyclic_quartic_symmetric(coeff)
    return _sos_struct_acyclic_quartic_real(coeff)


def _sos_struct_acyclic_quartic_symmetric(coeff, recurrsion = None, real = True):
    """
    Solve acyclic quartic polynomials that are symmetric with respect to two variables.
    If it is nonnegative over R, it must be sum of squares by Hilbert's 17th problem, we can write it in
    the form of: (assume f(a,b,c) = f(b,a,c) by symmetriciy)
    f(a,b,c) = p1' * M1 * p1 + (a-b)^2 * p2' * M2 * p2
    where p1 = [c**2, c*(a+b), a*b, (a-b)**2]', p2 = [a+b, c]

    and M1 = sp.Matrix([
        [c004, c103/2, c112/2 + c202 - 2*l, c202/2 - l/2 - r11/2],
        [c103/2, l, c211/2 + c301/2, c301/2 - r01],
        [c112/2 + c202 - 2*l, c211/2 + c301/2, c220 + 2*c310 + 2*c400, c310/2 + 2*c400 - 2*r00],
        [c202/2 - l/2 - r11/2, c301/2 - r01, c310/2 + 2*c400 - 2*r00, c400 - r00]
    ])
    and M2 = sp.Matrix([[r00, r01], [r01, r11]]).
    Here l, r00, r01, r11 are four variables. Select them properly so that M1 and M2 are PSD.

    Denote f(1,1,c) = w4*c**4 + w3*c**3 + w2*c**2 + w1*c + w0,
    then M1[:-1,:-1].det() * (-4) == leading_det = ... (please refer to the code).

    Choose l such that leading_det == 0 or slightly negative, assume
    vec = [(a-b*w3/w4)/4, b, 1/4].T and M[:,-1] = M[:-1,:-1] * vec.
    Here a and b are new parameters, they determine the values of r00, r01, r11.
    To make sure that M1 >= 0, we require det1 = M[-1,-1] - vec.T * M[:-1,:-1] * vec >= 0.
    Also, det2 = r00*r11 - r01**2 >= 0.
    The constraints det1 >= 0 and det2 >= 0 are both quadratic with respect to a and b.

    Examples
    ----------
    (b-c)2(b+c-3a)2+(a-c)2(a+c-3b)2+1/4(a+b-3/2c)2(a+b-3c)2

    2(c-a)2(c+a-4b)2+7/2(c-a)2(c+a-b)2+s(c2-5ca)2

    (3c4-2c3a-4c3b-c2a2+4c2ab-2ca3+4ca2b-2cab2+3a4-4a3b+b4)

    s(2a2-5ab)2+(a-b)2(a+b-3c)2

    (a4-6a3c+2a2b2+2a2bc+10a2c2+2ab2c-16abc2+b4-6b3c+10b2c2)

    (a-b)4 + c2(a-b)2

    (b2-2ba+c2-2ca)2 +(b2+c2-5a2)2 +2(bc-a2)2

    (ab+c2)(a+b-3c)2+s(a2-3ab)2/4

    (3339a4-5949a3b-2469a3c+9288a2b2-243a2bc+1159a2c2-5949ab3-243ab2c+278abc2-262ac3+3339b4-2469b3c+1159b2c2-262bc3+38c4)
    """
    if not coeff.is_rational:
        return

    monoms = [(4,0),(3,1),(2,2),(3,0),(2,1),(2,0),(1,0),(1,1),(0,0)]
    mappings = [(lambda i, j: (i,j,4-i-j)), (lambda i, j: (i,4-i-j,j)), (lambda i, j: (4-i-j,i,j))]
    symbol_orders = [sp.symbols("a b c"), sp.symbols("a c b"), sp.symbols("b c a")]
    for symbol_order, mapping in zip(symbol_orders, mappings):
        if all(coeff(mapping(i,j)) == coeff(mapping(j,i)) for i,j in ((4,0),(3,1),(3,0),(2,1),(2,0))):
            break
    else:
        return None

    c400, c310, c220, c301, c211, c202, c103, c112, c004 = [coeff(mapping(i, j)) for i, j in monoms]
    w4 = c004
    w3 = 2*c103
    w2 = c112 + 2*c202
    w1 = 2*c211 + 2*c301
    w0 = c220 + 2*c310 + 2*c400
    if w4 < 0:
        return # not implemented

    def _get_quad_forms(r00, r01, r11, l):
        M1 = sp.Matrix([
            [c004, c103/2, c112/2 + c202 - 2*l, c202/2 - l/2 - r11/2],
            [c103/2, l, c211/2 + c301/2, c301/2 - r01],
            [c112/2 + c202 - 2*l, c211/2 + c301/2, c220 + 2*c310 + 2*c400, c310/2 + 2*c400 - 2*r00],
            [c202/2 - l/2 - r11/2, c301/2 - r01, c310/2 + 2*c400 - 2*r00, c400 - r00]
        ])
        M2 = sp.Matrix([[r00, r01], [r01, r11]])
        return M1, M2

    def _get_solution(r00, r01, r11, l):
        M1, M2 = _get_quad_forms(r00, r01, r11, l)
        US1 = congruence(M1)
        US2 = congruence(M2)
        if US1 is None or US2 is None:
            return None
        a, b, c = symbol_order

        def _US_vec(US, vec):
            U, S = US
            return sp.Add(*[s*v.together()**2 for s, v in zip(S, U*vec)])
        solution = sp.Add(
            _US_vec(US1, sp.Matrix([c**2, c*a+c*b, a*b, a**2-2*a*b+b**2])),
            (a-b)**2 * _US_vec(US2, sp.Matrix([a+b, c]))
        )
        solution = solution.subs({
            (a**2 + 2*a*b + b**2)**2: (a + b)**4,
            (a**2 - 2*a*b + b**2)**2: (a - b)**4
        })
        return solution


    a, b, l = sp.symbols('a b l') # l == l11
    if w4 > 0:
        # find (a, b, l) such that leading_det == -4*M1[:-1,:-1].det() <= 0
        leading_det = sp.Poly([16, -8*w2, -4*w0*w4 + w1*w3 + w2**2, (w0*w3**2 + w1**2*w4 - w1*w2*w3)/4], l)

        if True:
            # first try r00 == r01 == r11 == 0
            det1 = sp.Poly([
                -(c220 - 2*c310 + 2*c400)/4,
                -(4*c112*c310 - 16*c112*c400 - 8*c202*c220 + 8*c202*c310 + 16*c202*c400 - c211**2 + 6*c211*c301 - 9*c301**2)/16,
                (8*c004*c220*c400 - 2*c004*c310**2 - 16*c004*c400**2 + c103*c211*c310 - 4*c103*c211*c400 - 2*c103*c220*c301 \
                    + c103*c301*c310 + 8*c103*c301*c400 - 2*c112**2*c400 + 2*c112*c202*c310 + c112*c211*c301 - 3*c112*c301**2 \
                    - 2*c202**2*c220 + 4*c202**2*c400 - c202*c211**2 + 4*c202*c211*c301 - 3*c202*c301**2)/8,
                _get_quad_forms(sp.S(0), sp.S(0), sp.S(0), sp.S(0))[0].det()
            ], l)
            for l1 in univariate_intervals([sp.Poly([1, 0], l), leading_det, det1]):
                if l1 >= 0 and leading_det(l1) <= 0 and det1(l1) >= 0:
                    solution = _get_solution(sp.S(0), sp.S(0), sp.S(0), l1)
                    if solution is not None:
                        return solution
            # TODO: consider r00 == r01 == 0 or r01 == r11 == 0 separately

        # find (a, b, l) such that det1 >= 0
        det1 = dict([
            ((2, 0, 0), -w4**2),
            ((0, 2, 1), -16*w4),
            ((0, 2, 0), w3**2),
            ((0, 0, 0), w4*(-4*c310 + w0))
        ])
        det1 = sp.Poly.from_dict(det1, (a, b, l))

        # find (a, b, l) such that det2 >= 0
        det2 = dict([
            ((2, 0, 1), -32*w4**3),
            ((2, 0, 0), w4**2*(8*w2*w4 - w3**2)),
            ((1, 1, 0), 2*w4*(8*w1*w4**2 - 4*w2*w3*w4 + w3**3)),
            ((1, 0, 1), -16*w4**2*(-4*c202 + w2)),
            ((1, 0, 0), 2*w4**2*(-8*c202*w2 + 8*c301*w3 - 16*c310*w4 - 64*c400*w4 + 8*w0*w4 - w1*w3 + 2*w2**2)),
            ((0, 2, 2), -256*w4**2),
            ((0, 2, 1), 32*w3**2*w4),
            ((0, 2, 0), -w3**4),
            ((0, 1, 1), -16*w4*(4*c202*w3 - 16*c301*w4 + 2*w1*w4 - w2*w3)),
            ((0, 1, 0), 2*w4*(-16*c202*w1*w4 + 8*c202*w2*w3 - 8*c301*w3**2 + 4*w1*w2*w4 + w1*w3**2 - 2*w2**2*w3)),
            ((0, 0, 0), w4**2*(64*c202*c310 + 256*c202*c400 - 32*c202*w0 - 64*c301**2 + 16*c301*w1 - 16*c310*w2 - 64*c400*w2 + 8*w0*w2 - w1**2))
        ])
        det2 = sp.Poly.from_dict(det2, (a, b, l))

        def _get_solution_from_ab(a, b, l):
            r11 = (-2*a*c004*w4 + 2*b*c004*w3 - 4*b*c103*w4 - c112*w4 + 2*c202*w4)/(4*w4)
            r01 = (-a*c103*w4 + b*c103*w3 - 8*b*l*w4 - c211*w4 + 3*c301*w4)/(8*w4)
            r00 = (-a*c112*w4 - 2*a*c202*w4 + 4*a*l*w4 + b*c112*w3 + 2*b*c202*w3 - 4*b*c211*w4 - 4*b*c301*w4 - 4*b*l*w3 - 2*c220*w4 + 12*c400*w4)/(16*w4)
            return _get_solution(r00, r01, r11, l)


        for l0 in nroots(leading_det, method='factor', real=True, nonnegative=True):
            if c004*l0 - c103**2/4 < 0: # M1[:2,:2].det()
                continue
            l1 = l0 if isinstance(l0, sp.Rational) else rationalize(l0, rounding=1e-15)
            # we need l11 to be rational to trigger common_region_of_conics
            f1, f2 = det1.subs(l, l1), det2.subs(l, l1)
            point = common_region_of_conics([f1, f2])
            # print(sp.latex(sp.GreaterThan(f1.subs({a:sp.Symbol('x'), b:sp.Symbol('y')}).as_expr(), 0)))
            # print(sp.latex(sp.GreaterThan(f2.subs({a:sp.Symbol('x'), b:sp.Symbol('y')}).as_expr(), 0)))
            # print(sp.latex(sp.GreaterThan(((-2*a*c004*w4 + 2*b*c004*w3 - 4*b*c103*w4 - c112*w4 + 2*c202*w4)/(4*w4)).subs({a:sp.Symbol('x'), b:sp.Symbol('y')}).as_expr(), 0)))
            # print(leading_det(l1))
            # print('\n')
            if point is not None:
                print(_get_solution_from_ab(*point, l1))
                if isinstance(l0, sp.Rational):
                    solution = _get_solution_from_ab(*point, l0)
                    if solution is not None:
                        return solution

                grad = leading_det.diff(l)(l0)
                for l1 in rationalize_bound(l0, direction=1 if grad < 0 else -1, compulsory=True):
                    if leading_det(l1) <= 0:
                        f1, f2 = det1.subs(l, l1), det2.subs(l, l1)
                        point = common_region_of_conics([f1, f2])
                        if point is not None:
                            solution = _get_solution_from_ab(*point, l1)
                            if solution is not None:
                                return solution

    else: # elif w4 == 0:
        if c103 != 0: # not positive semidefinite on R
            return None

        l1 = c112/4 + c202/2
        r11 = -c112/4 + c202/2
        if l1 < 0 or r11 < 0:
            return None

        leading_det = (c112 + 2*c202)*(c220 + 2*c310 + 2*c400) - (c211 + c301)**2
        if leading_det < 0:
            return None
        elif leading_det > 0:
            det1 = dict([
                ((2, 0), -c112 - 2*c202),
                ((1, 1), 2*(c211 + c301)),
                ((1, 0), -(c112*c220 - 6*c112*c400 + 2*c202*c220 - 12*c202*c400 - c211**2 + 2*c211*c301 + 3*c301**2)/4),
                ((0, 2), -c220 - 2*c310 - 2*c400),
                ((0, 1), -(c211*c310 + 4*c211*c400 - 2*c220*c301 - 3*c301*c310)/2),
                ((0, 0), (4*c112*c220*c400 - c112*c310**2 - 8*c112*c400**2 + 8*c202*c220*c400 - 2*c202*c310**2 - 16*c202*c400**2 - 4*c211**2*c400 + 4*c211*c301*c310 + 8*c211*c301*c400 - 4*c220*c301**2 - 4*c301**2*c310 + 4*c301**2*c400)/16)
            ])
            det1 = sp.Poly.from_dict(det1, (a, b))
    
            det2 = (r11*a - b**2).as_poly(a, b)

            point = common_region_of_conics([det1, det2])
            if point is not None:
                return _get_solution(*point, r11, l1)
        else: # leading_det == 0:
            det1 = sp.Poly([-c112 - 2*c202, c211 + c301, -c310], a).as_poly(a, b)
            det2 = sp.Poly([
                -(c112 + 2*c202)**2,
                c112*c211 + 5*c112*c301 - 2*c202*c211 + 6*c202*c301,
                -c112*c310 - 4*c112*c400 + 2*c202*c310 + 8*c202*c400 - 4*c301**2
            ], a).as_poly(a, b)
            point = common_region_of_conics([det1, det2])
            if point is not None:
                def _get_solution_from_a(a):
                    r00 = -(a*c211 + a*c301 - c310 - 4*c400)/4
                    r01 = -(a*c112 + 2*a*c202 - 2*c301)/4
                    return _get_solution(r00, r01, r11, l1)
                return _get_solution_from_a(point[0])



class _quadratic_minimization():
    """
    A helper class to solve quadratic minimization problems. Assume
    f(x,y) = a(y) * x^2 + b(y) * x + c(y) is a quadratic function with
    respect to x, and a, b, c are all functions of y. We want to minimize
    f(x,y).
    """
    def __init__(self, a, b, c, *args):
        self.a = a.as_poly(*args)
        self.b = b.as_poly(*args)
        self.c = c.as_poly(*args)
    def symmetric_axis(self, *args):
        a, b = self.a(*args), self.b(*args)
        return - b / (2*a)
    def extrema(self, *args):
        a, b, c = [i(*args) for i in (self.a, self.b, self.c)]
        return c - b**2 / (4*a)
    def _diff_of_extrema(self, *args):
        a, b, c = self.a, self.b, self.c
        return 4*c.diff(*args)*a**2 - 2*b*b.diff(*args)*a + b**2*a.diff(*args)
    def extrema_2d(self, *args):
        roots = nroots(self._diff_of_extrema(*args), method='factor', real = True)
        candidates = []
        for root in roots:
            v = self.extrema(root)
            if v is not sp.zoo:
                candidates.append((root, v))
        if len(candidates) == 0:
            return []
        minimum = min(_[1] for _ in candidates)
        roots = [((self.symmetric_axis(r), r), v) for r, v in candidates if v == minimum]
        return roots

def _sos_struct_acyclic_quartic_real(coeff):
    """
    Solve acyclic quartic problems over a,b,c in R.

    Hilbert's 17th problem, solved by Artin, answers the question that
    a 3-variable polynomial of degree 4 is nonnegative over R must be sum of squares.
    The SOS decomposition can be attained by solving SDP. 
    This function is an alternative algorithm to solve the SDP manually.

    We follow the following steps:
    1. Subtract enough c^4 so that the polynomial is nonnegative and has a root over R.
    2. Subtract enough c^2*(a - ?c)^2 so that the polynomial is nonnegative and has two roots.
    3. Subtract enough (...)^2 so that the polynomial is nonnegative and has three roots.

    When the polynomial has three roots over R and is still nonnegative, its sum of squares
    decomposition matrix (the SDP matrix) is defined uniquely by its coefficients. Then we
    can recover the SOS decomposition to the original polynomial. Actually, we can make a
    linear transformation that f(a,b,c) = g(x,y,z) such that g(1,0,0)=g(0,1,0)=g(0,0,1) are
    the three roots, and g is then a quadratic form of xy, yz and zx.

    Examples
    ----------
    (20a4-94a3b-12a3c+171a2b2+28a2bc+4a2c2-151ab3-11ab2c-8abc2+77b4-21b3c+7b2c2)
    """
    poly = coeff.as_poly()
    roots = _sos_struct_acyclic_quartic_reaL_findroots(coeff, poly)
    if roots is None:
        return None

    poly2 = poly
    for root, subtraction in roots:
        poly2 = poly2 - subtraction

    def get_base_mat(poly):
        u400, u310, u301, u220, u211, u202, u130, u121, u112, u103, u040, u031, u022, u013, u004 = [poly.coeff_monomial(_) for _ in
            ((4,0,0),(3,1,0),(3,0,1),(2,2,0),(2,1,1),(2,0,2),(1,3,0),(1,2,1),(1,1,2),(1,0,3),(0,4,0),(0,3,1),(0,2,2),(0,1,3),(0,0,4))
        ]
        M = [
            [u400, u220/2, u202/2, u310/2, u211/2, u301/2],
            [u220/2, u040, u022/2, u130/2, u031/2, u121/2],
            [u202/2, u022/2, u004, u112/2, u013/2, u103/2],
            [u310/2, u130/2, u112/2, 0, 0, 0],
            [u211/2, u031/2, u013/2, 0, 0, 0],
            [u301/2, u121/2, u103/2, 0, 0, 0]
        ]
        return sp.Matrix(M)

    def get_constraints(base_mat, root):
        a0, b0, c0 = root[0]
        vec = sp.Matrix([a0**2, b0**2, c0**2, a0*b0, b0*c0, c0*a0])
        target = -base_mat * vec
        A = [
            [1, 0, 0, 0, 0, 0],
            [a0**2/b0**2, 1, 0, 0, 0, 0],
            [0, b0**2/c0**2, 1, 0, 0, 0],
            [-2*a0/b0, 0, c0**2/(a0*b0), 1, 0, 0],
            [0, -2*b0/c0, -c0/b0, 0, 1, 0],
            [0, 0, -c0/a0, 0, 0, 1]
        ]

    def _as_abc2_vec(expr):
        expr = expr.as_poly(a,b,c)
        v = lambda d0,d1,d2: expr.coeff_monomial((d0,d1,d2))
        return sp.Matrix([v(2,0,0),v(0,2,0),v(0,0,2),v(1,1,0),v(0,1,1),v(1,0,1)])

    def solve_numer_params_from_roots(poly, roots):
        # first apply transform
        x, y = sp.symbols('x y')
        Rmat = sp.Matrix([list(r) for r, _, __ in roots]).T
        Rinv = Rmat.inv()
        trans = lambda x, y, z: list(Rmat * sp.Matrix([x, y, z]))
        invtrans = lambda a, b, c: list(Rinv * sp.Matrix([a, b, c]))
        poly2 = poly - sum([v*sub**2 for _, v, sub in roots])
        poly2 = poly2(*trans(x, y, 1)).as_poly(x, y)
        # write poly2 in sum-of-squares
        v = lambda d0, d1: poly2.coeff_monomial((d0, d1))
        Mxyz = sp.Matrix([
            [v(2,2), v(1,2)/2, v(2,1)/2],
            [v(1,2)/2, v(0,2), v(1,1)/2],
            [v(2,1)/2, v(1,1)/2, v(2,0)]
        ])
        # print(((poly2 - (sp.Matrix([x*y,y,x]).T * M * sp.Matrix([x*y,y,x]))[0,0])).coeffs())
        if not Mxyz.is_positive_semidefinite:
            return
        x0, y0, z0 = invtrans(a, b, c)
        Mtrans = sp.Matrix.vstack(*[_as_abc2_vec(_).T for _ in (x0*y0, y0*z0, z0*x0)])
        Mtrans_right = Mtrans[:,3:]
        Mabc_lower_right = Mtrans_right.T * Mxyz * Mtrans_right
        for root, value, sub in roots:
            subvec_right = _as_abc2_vec(sub)[3:,:]
            Mabc_lower_right += subvec_right * subvec_right.T * value
        params = []
        for i, j in ((0,0),(1,1),(2,2),(1,0),(0,2),(1,2)):
            params.append(Mabc_lower_right[i,j])
        return params

    def construct_mat_from_params(base_mat, params):
        l33, l44, l55, l34, l45, l35 = params
        addition = sp.Matrix([
            [0, -l33/2, -l55/2, 0, -l35, 0],
            [-l33/2, 0, -l44/2, 0, 0, -l34],
            [-l55/2, -l44/2, 0, -l45, 0, 0],
            [0, 0, -l45, l33, l34, l35],
            [-l35, 0, 0, l34, l44, l45],
            [0, -l34, 0, l35, l45, l55]
        ])
        return base_mat + addition

def _sos_struct_acyclic_quartic_reaL_findroots(
        coeff, poly = None
    ) -> List[Tuple[Tuple[sp.Float, sp.Float, sp.Float], sp.Float, sp.Expr]]:
    """
    Subtract some polynomials from the original polynomial so that the remaining polynomial
    has at least three roots over R.

    Returns
    ---------
    List of (root, value, subtraction) where root is the root,
    so that the original poly - sum(value[j]*subtraction[j]**2 for j in range(i+1)) vanishes at root[i].

    If the polynomial detects to be not nonnegative, or there are at most two roots found,
    then it returns None.
    """
    if poly is None:
        poly = coeff.as_poly()
    
    def find_trivial_root(coeff) -> List:
        if coeff((4,0,0)) == 0:
            if coeff((3,1,0)) == 0 and coeff((3,0,1)) == 0:
                return [((1,0,0), 0, sp.S(0))]
            return None # not positive on R
        elif coeff((0,4,0)) == 0:
            if coeff((1,3,0)) == 0 and coeff((0,3,1)) == 0:
                return [((0,1,0), 0, sp.S(0))]
            return None
        elif coeff((0,0,4)) == 0:
            if coeff((1,0,3)) == 0 and coeff((0,1,3)) == 0:
                return [((0,0,1), 0, sp.S(0))]
            return None
        return []

    def find_first_root(poly) -> List:
        """
        Solve maximum x such that poly >= x*c^4.
        By homogeneous property, we can assume c = 1 and solve for the extrema.
        """
        poly2 = poly.subs(c, 1)
        diff1 = poly2.diff(a)
        diff2 = poly2.diff(b)
        res = sp.polys.resultant(diff1, diff2, a)
        broots = nroots(res.as_poly(b), real = True, method = 'factor')
        candidates = []
        for b_ in broots:
            for a_ in nroots(diff1.subs(b,b_).as_poly(a), real=True, method='factor'):
                candidates.append(((a_, b_), poly2(a_,b_)))
        if len(candidates) == 0:
            return []
        minimum = min(_[1] for _ in candidates)
        if minimum < 0:
            return []
        roots = [((x[0][0], x[0][1], sp.S(1)), minimum, c**2) for x in candidates if x[1] == minimum]
        return roots

    def find_second_root(poly, root) -> List:
        """
        Solve maximum w such that poly - poly(root)*c^4 >= w * c^2*(root[2]*a - root[0]*c)^2
        We make the transformation that
        a = x + root[0]
        b = x*y + root[1]
        c = root[2]
        Then we w = sup{g(x,y)/x^2/root[2]^2}. However, we can show that g(x,y)/x^2 is a
        quadratic polynomial with respect to x.
        """
        x, y = sp.symbols('x y')
        root, value, sub1 = root
        trans = lambda x, y, z: (x + root[0], x*y + root[1], root[2])
        poly2 = poly - value * sub1**2
        poly2 = poly2(*trans(x, y, 1)).as_poly(x)

        # poly2 does not have x^0 and x^1 terms
        quad = _quadratic_minimization(
            poly2.coeff_monomial((4,)),
            poly2.coeff_monomial((3,)),
            poly2.coeff_monomial((2,)),
            y
        )
        roots = quad.extrema_2d(y)
        # transform back to a,b,c
        if len(roots) == 0:
            return []
        minimum = roots[0][1] / root[2]**2
        subtraction = c * (root[2]*a - root[0]*c)
        return [(trans(x, y, 1), minimum, subtraction) for (x, y), v in roots]

    
    def find_third_root(poly, root1, root2) -> List:
        """
        Define f = poly - ...*c^4 - - ...*c^2*(...*a -...*c)^2 >= 0
        with roots at root1 and root2. We can make a linear transformation that
        a = x + root2[0]*y + root1[0]*z
        b =     root2[1]*y + root1[1]*z
        c =     root2[2]*y + root1[2]*z
        Then g(0,0,1) = g(0,1,0) = 0. Also, if we let z = 1, then g is quadratic
        with respect to y. We can solve the maximum w such that g(x,y)/(y^2z^2) >= w.
        We can let u = 1/y, then g(x,y)/y^2 = h(x,u) is quadratic of u.

        The inverse transformation is that
        y = (b*root1[2] - c*root1[1]) / (root2[1]*root1[2] - root2[2]*root1[1])
        z = (c*root2[1] - b*root2[2]) / (root2[1]*root1[2] - root2[2]*root1[1])
        """
        root1, value1, sub1 = root1
        root2, value2, sub2 = root2
        x, y = sp.symbols('x y')
        poly2 = poly - value1*sub1**2 - value2*sub2**2
        trans = lambda x, y, z: (x + root2[0]*y + root1[0]*z, root2[1]*y + root1[1]*z, root2[2]*y + root1[2]*z)
        poly2 = poly2(*trans(x, y, 1)).as_poly(y)
        # note: implicitly change the variable to u = 1/y
        quad = _quadratic_minimization(
            poly2.coeff_monomial((0,)),
            poly2.coeff_monomial((1,)),
            poly2.coeff_monomial((2,)),
            x
        )
        roots = quad.extrema_2d(x)
        if len(roots) == 0:
            return []

        # transform back to a,b,c
        minimum = roots[0][1] / (root2[1]*root1[2] - root2[2]*root1[1])**4
        y_ = (b*root1[2] - c*root1[1])
        z_ = (c*root2[1] - b*root2[2])
        subtraction = y_ * z_

        roots = [(trans(x, 1/u, 1), minimum, subtraction) for (u, x), v in roots]
        return roots


    roots = find_trivial_root(coeff)
    if roots is None:
        return None
    for i, func in enumerate([find_first_root, find_second_root, find_third_root]):
        if len(roots) == i:
            roots += func(poly, *roots)
    if len(roots) < 3:
        return None