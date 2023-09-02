import sympy as sp

from .utils import (
    CyclicSum, CyclicProduct, Coeff,
    sum_y_exprs, nroots, rationalize_bound, quadratic_weighting, radsimp
)

a, b, c = sp.symbols('a b c')

def sos_struct_quartic(poly, coeff, recurrsion):
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

    """
    m, p, n, q = coeff((4,0,0)), coeff((3,1,0)), coeff((2,2,0)), coeff((1,3,0))

    t = q / 2 / m
    if p == -q and n == (t**2 - 2) * m:
        w = sp.sqrt(t*t + 4)
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
    symmetric = lambda _x: ((2*q+p)*_x + 6)*_x + 2*p+q
    
    # the discriminant after subtraction
    head = radsimp(p*p+p*q+q*q-3*n-3)
    def new_det(sym, root):
        return head - sym**2/(4*(root**2*(root**2 + 1) + 1))

    if coeff.is_rational:
        for root in sp.polys.roots(eq, cubics = False, quartics = False, quintics = False).keys():
            if root.is_real and root > 0:
                if isinstance(root, sp.Rational):
                    symmetric_axis = symmetric(root)
                    if symmetric_axis >= 0:
                        det = (new_det(symmetric_axis, root))

                        if det <= 0:
                            u_ = root
                            break
    else:
        # check whether there exists multiplicitive root on the border
        eq_diff = (x**4 + p*x**3 + n*x**2 + q*x + 1).as_poly(x)
        eq_gcd = sp.gcd(eq, eq_diff)
        if eq_gcd.degree() == 1:
            # TODO: if degree >= 2?
            u_ = radsimp(-(eq_gcd.all_coeffs()[1] / eq_gcd.LC()))

    if u_ is None:
        # find a rational approximation
        for numer_r in sp.polys.nroots(eq):
            if (not numer_r.is_real) or symmetric(numer_r) < 1e-6:
                continue

            for numer_r2 in rationalize_bound(numer_r, direction = 0, compulsory = True):
                symmetric_axis = symmetric(numer_r2)
                if symmetric_axis >= 0 and new_det(symmetric_axis, numer_r2) <= 0:
                    u_ = numer_r2
                    break
    
    if u_ is not None:
        y_ = radsimp((symmetric(u_) / (2*(u_**2*(u_**2 + 1) + 1)) * m))
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
            solution = quadratic_weighting(p, q, n, a-b, a-c, formal = True)
            solution = sum(radsimp(wi) * CyclicSum(b*c*xi**2) for wi, xi in solution)

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
                    x_ = 9*(p + 2)**2 / 4
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
            
                # finally subtract enough s(a2bc) to xs(a2bc) (or near xs(a2bc))
                for x2 in rationalize_bound(x_, direction = 1, compulsory = True):
                    if x2 < r:
                        new_coeffs_ = {(4,0,0): coeff((4,0,0)), (3,1,0): coeff((3,1,0)),
                                    (2,2,0): coeff((2,2,0)), (1,3,0): coeff((1,3,0)), (2,1,1): x2 * m}
                        solution = _sos_struct_quartic_uncentered(Coeff(new_coeffs_, is_rational = coeff.is_rational), recur = True)
                        if solution is not None:
                            solution += radsimp((r - x2) * m) * CyclicSum(a**2*b*c)
                            return solution

            return None

    return None