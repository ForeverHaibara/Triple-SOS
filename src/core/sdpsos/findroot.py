import sympy as sp
from sympy.polys.polyclasses import ANP

from ...utils.roots.roots import Root, RootAlgebraic, RootRational

def _sqrt_coeff_and_core(sqrpart):
    """
    Given a square root of a Rational, return the coefficient and the core, i.e.
    sqrpart = sqrpart_coeff * sqrt(core)

    >>> _sqrt_coeff_and_core(sp.sqrt(12))
    (2, 3)
    """
    if sqrpart == 0:
        return sp.S(0), sp.S(0)
    if isinstance(sqrpart, sp.Mul):
        core = sqrpart.args[0] if sqrpart.args[0].is_Pow else sqrpart.args[1]
        sqrpart_coeff = sqrpart.args[1] if sqrpart.args[0].is_Pow else sqrpart.args[0]
    else:
        core = sqrpart
        sqrpart_coeff = sp.S(1)
    core = core.args[0] # sqrtpart = ... * sqrt(core)
    return sqrpart_coeff, core


def _sqrt_of_sqrt(const, sqrpart, return_split = False):
    """
    Simplify sqrt(const + sqrpart)
    where const is Rational and sqrpart is a square root of a Rational.

    If return_split is True, return the two parts of the split square root.

    >>> _sqrt_of_sqrt(7, 4*sp.sqrt(3))
    2 + sqrt(3)

    >>> _sqrt_of_sqrt(7, 4*sp.sqrt(3), return_split = True)
    (2, sqrt(3))
    """
    if sqrpart == 0:
        sqrt = sp.sqrt(const)
        if isinstance(sqrt, sp.Rational):
            return sqrt if not return_split else sqrt, sp.S(0)
        return sp.sqrt(const) if not return_split else sp.S(0), sp.sqrt(const)
    sqrpart_coeff, core = _sqrt_coeff_and_core(sqrpart)

    # assume (a + b*sqrt(core))**2 = const + sqrpart_coeff * sqrt(core)
    # a^2 + b^2 * core = const
    # 2ab = sqrpart_coeff
    # a^4 - const * a^2 + sqrpart_coeff**2 * core / 4 = 0
    a_det_ = sp.sqrt(const**2 - sqrpart_coeff**2 * core)
    if not isinstance(a_det_, sp.Rational):
        return
    a = sp.sqrt((const - a_det_) / 2)
    if not isinstance(a, sp.Rational):
        a = sp.sqrt((const + a_det_) / 2)
        if not isinstance(a, sp.Rational):
            return
    b = sqrpart_coeff / (2 * a)
    if (b < 0 and a**2 < b**2 * core):
        a = -a
        b = -b
    if return_split:
        return a, b * sp.sqrt(core)
    return a + b * sp.sqrt(core)

def _quadratic_as_ANP(const, sqrpart):
    """
    Given a quardatic algebraic number x = const + sqrpart, where
    const is rational and sqrpart is a square root of a rational,
    return the ANP representation of x.

    >>> _quadratic_as_ANP(7, 4*sp.sqrt(3))
    ANP([4, 7], [1, 0, -3], QQ)
    """
    sqrpart_coeff, core = _sqrt_coeff_and_core(sqrpart)
    return ANP([(sqrpart_coeff), (const)], [1, 0, -int(core)], sp.QQ)

def _find_nearest_root(poly, v):
    """
    Find the nearest root of a polynomial to a given value.
    This helps select the closed-form root corresponding to a numerical value.
    """
    if poly.degree() == 1:
        c1, c0 = poly.all_coeffs()
        return sp.Rational(-c0, c1)
    v = v.n(20)
    best, best_dist = None, None
    for r in sp.polys.roots(poly):
        dist = abs(r.n(20) - v)
        if best is None or dist < best_dist:
            best, best_dist = r, dist
    return best

def _compute_uvxy(a, b):
    t = ((a*b-a)*(a-b)-(b*(a-1))**2)

    # basic quadratic form
    u = ((b*b-a*a)*(a-b) - (b-a*b)*(1-b*b))/t
    v = ((a*b-a)*(1-b*b) - (b*b-a*a)*(b-b*a))/t

    x = ((v-u)*(u*v+u+v-2)+u**3+1)/(1-u*v)
    y = ((v-u)*(u*v+u+v-2)-v**3-1)/(1-u*v)
    return u, v, x, y

def _findroot_resultant_root_pairs(poly, factors, prec = 20, tolerance = 1e-12):
    """
    Find the numerical root pairs of a polynomial using the resultant method.
    The polynomial should be of two variables. Factors are the minimal polynomial
    of the second variable.

    Suppose (a, b, 1) is a root of the original 3-var polynomial, then b is 
    a root in one of the `factors`. Take the cyclic permutation, (b/a, 1/a, 1) is also
    valid. Thus, variable a must be the inverse of one of the factors. There are exceptions
    where ab = 0. When both are zeros, it can be discarded and handled automatically through
    polynomial convex hull in SDP SOS. When b = 0, it does not matter actually. When a = 0,
    we have already know the value of a.

    As there are multiple factors, we need to pair them into (a, b) pairs.

    Parameters
    ----------
    poly : sympy.Poly
        The polynomial of two variables.
    factors: List[Tuple[sympy.Poly, int]]
        The minimal polynomials of the second variable. It is resultant.factor_list()[1].
    prec: int
        The precision of solving roots of each factor numerically.
    tolerance: float
        The tolerance of a pair (a,b) is recognized as a root pair.
    """
    all_roots = []
    all_roots_inv = []
    for i, factor_ in enumerate(factors):
        for r in sp.polys.nroots(factor_[0], n = prec):
            if r.is_real:
                all_roots.append((r, i))
                if r != 0:
                    all_roots_inv.append((1/r, i))
                else:
                    all_roots_inv.append((0, i))

    pairs = []
    for b_, j in all_roots:
        for a_, i in all_roots_inv:
            if abs(a_) > 1 or abs(b_) > 1:
                # we could assume |c| = max{|a|, |b|, |c|} = 1
                # as the polynomial is cyclic
                continue
            v = poly(a_, b_)
            if abs(v) < tolerance:
                pairs.append(((a_, b_), (i, j)))
    return pairs

def _findroot_resultant(poly):
    """
    Find root of a 2-var polynomial with respect to a, b using the method of 
    resultant. The polynomial should be irreducible.

    Parameters
    ----------
    poly : sympy.Poly
        The polynomial of two variables. It should be irreducible.

    Returns
    -------
    roots: List[RootAlgebraic]
        The roots of the polynomial.    
    """
    a, b, c = sp.symbols('a b c')
    grad = poly.diff(a)
    res = sp.resultant(poly, grad)
    factors = set(sp.polys.gcd(res, res.diff(b)).factor_list()[1])

    # roots on the border might not be local minima
    poly_border = poly.subs(a, 0)
    grad_border = poly_border.diff(b)
    factors_border = set(sp.polys.gcd(poly_border, grad_border).factor_list()[1])
    if len(factors_border):
        # conjugate factors, b = 0
        factors_border.add((sp.Poly([1,0], b), len(factors_border)))
    factors |= factors_border
    factors = list(factors)

    pairs = _findroot_resultant_root_pairs(poly, factors)
    roots = []
    for (a_, b_), (i, j) in pairs:
        if i == j:
            fx = factors[i][0]
            if fx.degree() == 1:
                c1, c0 = fx.all_coeffs()
                if abs(c1) == abs(c0):
                    roots.append(RootRational((-1 if c1 == c0 else 1, 1, 1)))
            elif fx.degree() == 3:
                c3, c2, c1, c0 = fx.all_coeffs()
                if c3 != -c0:
                    continue
                x, y = c2 / c3, c1 / c3
                denom = 2*(x**2 + x*y + 3*x + y**2 - 3*y + 9)
                det = sp.sqrt(4*x**3 + x**2*y**2 - 18*x*y - 4*y**3 - 27) / denom
                sym_u = (x**2*y - 2*x**2 + x*y + 3*x - 2*y**2 + 6*y + 9) / denom
                sym_v = (-2*x**2 - x*y**2 + x*y - 6*x - 2*y**2 - 3*y + 9) / denom
                # sym_u - sym_v = (x + y)(xy + 9)
                u1 = sym_u + (x + 3) * det
                v1 = sym_v + (3 - y) * det
                u2 = sym_u - (x + 3) * det
                v2 = sym_v - (3 - y) * det

                u__, v__, x__, y__ = _compute_uvxy(a_, b_)
                dist = lambda u, v: abs(u.n(20) - u__) + abs(v.n(20) - v__)
                if dist(u1, v1) < dist(u2, v2):
                    u, v = u1, v1
                else:
                    u, v = u2, v2
                roots.append(RootAlgebraic(u, v))
            elif fx.degree() == 6:
                c6, c5, c4, c3, c2, c1, c0 = fx.monic().all_coeffs()
                # here two (u,v) solutions are conjugate
                # so we only need one of them
                # the sextic is solvable, f(x) = (t^3+x1t^2+y1t-1)(t^3+x2t^2+y2t-1)
                if c0 != 1:
                    continue
                # x1 + x2 = c5
                # x1x2 = c4 + c1
                # x1y2 + y1x2 = c3 + 2
                # y1y2 = c2 + c5
                # y1 + y2 = -c1
                sym_x = c5 / 2
                sym_y = -c1 / 2
                diff = c3 + 2 - 2*sym_x*sym_y
                if diff**2 != 4*(sym_x**2 - c4 - c1)*(sym_y**2 - c2 - c5):
                    # x1y2 + x2y1 = 2(sym_xsym_y \pm det_xdet_y)
                    continue
                det_x = sp.sqrt(sym_x**2 - c4 - c1)
                det_y = sp.sqrt(sym_y**2 - c2 - c5)
                if diff > 0:
                    det_y = -det_y

                u__, v__, x__, y__ = _compute_uvxy(a_, b_)
                if sym_x > x__:
                    det_x, det_y = -det_x, -det_y

                sqrpart = (c1**2*c5/2 + c1*c3/2 + 6*c1 - 4*c4 + 4*c5**2)*det_x + (-4*c1**2 - c1*c5**2/2 + 4*c2 - c3*c5/2 - 6*c5)*det_y
                const = 7*c1**3/4 - c1**2*c4/4 + c1**2*c5**2/2 - 6*c1*c2 + 3*c1*c3*c5/4 - 3*c1*c5/2 - c2*c5**2/4 + c3**2/4 + 10*c3 - 6*c4*c5 + 7*c5**3/4 - 8
                
                # x, y = sym_x + det_x, sym_y + det_y
                # print(sp.simplify(4*x**3 + x**2*y**2 - 18*x*y - 4*y**3 - 27), '=', const + sqrpart)
                x, y = _quadratic_as_ANP(sym_x, det_x), _quadratic_as_ANP(sym_y, det_y)

                det = _sqrt_of_sqrt(const, sqrpart, return_split = True)
                # print('det =',det)
                if det is None:
                    continue
                mod = x.mod # core might be rational but x must be irrational
                f = lambda x: ANP([0, x], mod, sp.QQ)
                det = _quadratic_as_ANP(*det) if det[1] != 0 else f(det[0])

                inv_denom = f(sp.S(1)/2) / (x**2 + x*y + f(3)*x + y**2 - f(3)*y + f(9))
                det = det * inv_denom
                sym_u = ((x**2*y - f(2)*x**2 + x*y + f(3)*x - f(2)*y**2 + f(6)*y + f(9)) * inv_denom)
                sym_v = ((f(-2)*x**2 - x*y**2 + x*y - f(6)*x - f(2)*y**2 - f(3)*y + f(9)) * inv_denom)
                core = -mod[-1]

                to_sympy = lambda value_: value_.rep[0] * sp.sqrt(core) + value_.rep[1] \
                    if len(value_.rep) == 2 else sp.S(value_.rep[0])
                u1 = to_sympy(sym_u + (x + f(3)) * det)
                v1 = to_sympy(sym_v + (f(3) - y) * det)
                u2 = to_sympy(sym_u - (x + f(3)) * det)
                v2 = to_sympy(sym_v - (f(3) - y) * det)

                dist = lambda u, v: abs(u.n(20) - u__) + abs(v.n(20) - v__)
                if dist(u1, v1) < dist(u2, v2):
                    u, v = u1, v1
                else:
                    u, v = u2, v2

                roots.append(RootAlgebraic(u, v, K = sp.QQ.algebraic_field(sp.sqrt(core))))
                # print(u1, v1, u2, v2)
        elif i != j:
            fx, fy = factors[i][0], factors[j][0]
            # reverse fx
            fx = sp.Poly(fx.all_coeffs()[::-1], fy.gens[0])
            if fx.degree() == 0:
                fx = sp.Poly([1,0], fy.gens[0]) # cancel reverse
            if fx.degree() == 1 and fy.degree() == 1:
                a_ = -fx.all_coeffs()[1]/fx.all_coeffs()[0]
                b_ = -fy.all_coeffs()[1]/fy.all_coeffs()[0]
                roots.append(RootRational((a_, b_, 1)))
            elif fx.degree() < 3 and fy.degree() < 3:
                # construct uv by a2-b2+u(ab-ac)+v(bc-ab) = 0 and b2-c2+u(bc-ba)+v(ca-bc)=0
                # we can use sp.simplify
                cores = []
                roots_ = []
                for r_, poly_ in zip((a_, b_), (fx, fy)):
                    if poly_.degree() == 1:
                        c1, c0 = poly_.all_coeffs()
                        roots_.append((-c0/c1, 0))
                    elif poly_.degree() == 2:
                        c2, c1, c0 = poly_.monic().all_coeffs()
                        sym = -c1 / 2
                        det = sp.sqrt(c1**2 - 4*c0) / 2
                        if r_ > sym:
                            roots_.append((sym, det))
                        else:
                            roots_.append((sym, -det))
                        cores.append(_sqrt_coeff_and_core(det)[1])

                if len(cores) == 2 and cores[0] != cores[1]:
                    continue
                for i in range(len(roots_)):
                    if roots_[i][1] == 0:
                        roots_[i] = ANP([0, roots_[i][0]], [1, 0, -int(cores[0])], sp.QQ)
                    else:
                        roots_[i] = _quadratic_as_ANP(*roots_[i])
                
                a__, b__ = roots_
                denom = (a__**2*b__**2 - a__**2*b__ + a__**2 - a__*b__**2 - a__*b__ + b__**2)
                inv_denom = ANP([0, 1], [1, 0, -int(cores[0])], sp.QQ) / denom
                u = (a__**3 - a__**2*b__ + a__*b__**3 - a__*b__**2 - a__*b__ + b__) * inv_denom
                v = (a__**3*b__ - a__**2*b__ - a__*b__**2 - a__*b__ + a__ + b__**3) * inv_denom
                u = u.rep[0] * sp.sqrt(cores[0]) + u.rep[1]
                v = v.rep[0] * sp.sqrt(cores[0]) + v.rep[1]
                roots.append(RootAlgebraic(u, v, K = sp.QQ.algebraic_field(sp.sqrt(cores[0]))))

    # print(factors, roots)
    return roots


def findroot_resultant(poly):
    """
    Find the roots of a polynomial using the resultant method. This is
    essential in SDP SOS, because we need to construct exact subspace
    constrainted by roots of the polynomial to perform SDP on a full-rank manifold.

    The method does not guarantee to find all roots, but find at least one in 
    each Galois group. For instance, if (1,1,sqrt(2)) is a root, then (1,1,-sqrt(2))
    must be a root as well, due to the principle of minimal polynomial.
    Then it is possible for us to return (1,1,-sqrt(2)), although it is not
    positive. This seems to miss some roots, but it is not a problem in SDP SOS,
    because the spanned subspace algorithm includes the permutation of roots.

    TODO:
    1. Handle the case where the polynomial is cubic.
    2. Optimize the speed by removing redundant permuations.

    Parameters
    ----------
    poly : sympy.Poly
        The polynomial to find roots of.

    Returns
    -------
    roots : List[RootAlgebraic]
        A list of roots of the polynomial.
    """
    a, b, c = sp.symbols('a b c')
    poly = poly.subs(c,1) # -a-b).as_poly(a,b)
    parts = poly.factor_list()[1]
    roots = []
    for part in parts:
        poly, multiplicity = part
        if poly.degree() == 1:
            continue
        elif poly.degree() == 2:
            if poly(1,1) == 0:
                roots.append(RootRational((1,1,1)))
            continue
        else:
            roots.extend(_findroot_resultant(poly))


    # put the positive roots in front
    # roots_positive, roots_negative = [], []
    # for root in roots:
    #     if root.root[0] >= 0 and root.root[1] >= 0 and root.root[2] >= 0:
    #         roots_positive.append(root)
    #     else:
    #         roots_negative.append(root)
    # roots = roots_positive + roots_negative

    # remove duplicate roots
    roots_clear = []
    for root in roots:
        if any(r.root == root.root for r in roots_clear):
            continue
        roots_clear.append(root)
    return roots_clear
