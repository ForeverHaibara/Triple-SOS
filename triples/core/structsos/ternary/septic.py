from itertools import product

import sympy as sp
from sympy import Poly, Symbol, Rational, Add

from .quartic import sos_struct_quartic
from .septic_symmetric import sos_struct_septic_symmetric
from .utils import (
    Coeff, sum_y_exprs, nroots, rationalize, rationalize_bound, try_perturbations,
    zip_longest
)

_VERBOSE_OPTIMIZE_DISCRIMINANT = False

def optimize_discriminant(discriminant, soft = False, verbose = False):
    # TODO: DEPRECATE IT?
    x, y = discriminant.gens
    best_choice = (2147483647, 0, 0)
    for a, b in product(range(-5, 7, 2), repeat = 2): # integer
        v = discriminant(a, b)
        if v <= 0:
            best_choice = (v, a, b)
        elif v < best_choice[0]:
            best_choice = (v, a, b)

    v , a , b = best_choice
    if v > 0:
        for a, b in product(range(a-1, a+2), range(b-1, b+2)): # search a neighborhood
            v = discriminant(a, b)
            if v <= 0:
                best_choice = (v, a, b)
                break
            elif v < best_choice[0]:
                best_choice = (v, a, b)
    if verbose:
        print('Starting Search From', best_choice[1:], ' f =', best_choice[0])
    if v <= 0:
        return {x: a, y: b}

    def _compute_hessian(f, symbols):
        x, y = symbols
        f = f.as_poly(x, y)
        return f.diff(x), f.diff(y), f.diff(x, x), f.diff(x, y), f.diff(y, y)

    if v > 0:
        a = a * 1.0
        b = b * 1.0
        dervs = _compute_hessian(discriminant, (x, y))
        # x = [a',b'] <- x - inv(nabla)^-1 @ grad
        for i in range(20):
            lasta , lastb = a , b
            da_, db_, da2_, dab_, db2_ = [f(a,b) for f in dervs]
            det_ = da2_ * db2_ - dab_ * dab_
            if verbose:
                print('Step Position %s, f = %s, H = %s'%((a,b), discriminant(a,b).n(20), det_))
            if det_ == 0:
                break
            else:
                a , b = a - (db2_ * da_ - dab_ * db_) / det_ , b - (-dab_ * da_ + da2_ * db_) / det_
                if abs(lasta - a) < 1e-9 and abs(lastb - b) < 1e-9:
                    break
        v = discriminant(a, b)

    if v > 1e-6 and not soft:
        return None

    # iterative deepening
    rounding = 0.5
    for i in range(5):
        a_ = rationalize(a, rounding, reliable = False)
        b_ = rationalize(b, rounding, reliable = False)
        v = discriminant(a_, b_)
        if v <= 0:
            break
        rounding *= .1
    else:
        return {x: a_, y: b_} if soft else None

    return {x: a_, y: b_}

def sos_struct_septic(coeff, real = True):
    if coeff((7,0,0)) == 0 and coeff((6,1,0)) == 0 and coeff((6,0,1)) == 0:
        if coeff((5,2,0)) == 0 and coeff((5,0,2)) == 0:
            # star
            return _sos_struct_septic_star(coeff)
        else:
            # hexagon
            return _sos_struct_septic_hexagon(coeff)

    if all(coeff((i,j,k)) == coeff((j,i,k)) for (i,j,k) in ((6,1,0),(5,2,0),(4,3,0),(4,2,1))):
        return sos_struct_septic_symmetric(coeff, real = real)

    return None

def _quartic_det(m, p, n, q):
    return 3*m*(m+n) - (p**2+p*q+q**2)


def _fast_solve_quartic(coeff: Coeff, m, p, n, q, rem = 0, mul_abc = True):
    """
    Solve s(m*a^5bc + p*a^4b^2c + n*a^3b^3c + q*a^2b^4c + (rem-m-p-n-q)*a^3b^2c^2) >= 0.
    """
    if isinstance(rem, Poly):
        rem = rem(1,1,1) / 3
    elif isinstance(rem, Coeff):
        rem = rem.poly111() / 3
    if rem < 0:
        return None

    a, b, c = coeff.gens
    CyclicSum, CyclicProduct = coeff.cyclic_sum, coeff.cyclic_product

    if m > 0:
        det = _quartic_det(m, p, n, q)
        if det >= 0:
            # most common case, use inplace computation
            solution = Add(*[
                m / 2 * CyclicSum((a**2-b**2+(p+2*q)/3/m*(a*c-a*b)-(q+2*p)/3/m*(b*c-a*b))**2),
                (det / 6 / m) * CyclicSum(a**2*(b-c)**2),
                rem * CyclicSum(a**2*b*c)
            ]).as_coeff_Mul()
            return solution[0] * CyclicProduct(a) * solution[1]

    coeffs_ = {
        (4,0,0): m, (3,1,0): p, (2,2,0): n, (1,3,0): q, (2,1,1): (rem - m - p - n - q)
    }
    solution = sos_struct_quartic(coeff.from_dict(coeffs_), None)
    if mul_abc and solution is not None:
        solution = solution * CyclicProduct(a)
    return solution


def _sos_struct_septic_star(coeff: Coeff):
    """
    Solve s(ua4b3 + va3b4) + abcf(a,b,c) >= 0 where f is degree 4.

    Idea: Subtract something and apply the quadratic theorem, which is a
    discriminant minimization theorem.

    Examples
    -------
    s(ab(a-c)2(b-c)2)s(a)

    s((a-b)2(a-c)2(b+c))s(ab)-s(a)p(a-b)2

    s(a3b3+7a4bc-29a3b2c+12a3bc2+9a2b2c2)s(a)

    s(16a5bc+3a4b3-77a4b2c+3a4bc2+3a4c3+72a3b3c-20a3b2c2)

    s(20a5bc+4a4b3-31a4b2c-4a4bc2+4a4c3-46a3b3c+53a3b2c2)

    (3s(6a4c-31a3bc+6a3c2+19a2b2c)s(a2)-18s(c(a3-abc+(-8/3)(a2b-abc)+(1/3)(ab2-abc)-(bc2-abc))2))

    s(16a5bc+4a4b3-80a4b2c+3a4bc2+7a4c3+64a3b3c-14a3b2c2)

    s(2a4b3+9a3b4+abc(18a4-66a3b+10a2b2+11ab3+16a2bc))

    s(72a5bc+24a4b3+156a4b2c-453a4bc2+44a4c3+176a3b3c-19a3b2c2)
    """
    if any(coeff(i) for i in ((7,0,0), (6,1,0), (6,0,1), (5,2,0), (5,0,2))):
        return None
    if coeff((4,3,0)) < 0 or coeff((3,4,0)) < 0 or coeff((5,1,1)) < 0:
        return None
    rem = coeff.poly111()
    if rem < 0:
        return None

    if coeff((4,3,0)) == 0 and coeff((3,4,0)) == 0:
        # degenerated to quartic
        return _fast_solve_quartic(coeff,
            coeff((5,1,1)),
            coeff((4,2,1)),
            coeff((3,3,1)),
            coeff((2,4,1)),
            rem = rem / 3
        )

    if not coeff.is_rational:
        return None

    a, b, c = coeff.gens
    CyclicSum, CyclicProduct = coeff.cyclic_sum, coeff.cyclic_product

    if coeff((4,3,0)) == coeff((3,4,0)):
        # standard septic star
        # Idea: first we compute optimal u,v by virtually subtracting p(a)s(a^2b^2-a^2bc) to reach a nontrivial equality.
        # Then we subtract s(c(a^2c-b^2c-u(a^2b-abc)+v(ab^2-abc))^2) and then apply the quadratic theorem.
        u, v = sp.symbols('u v')
        coeff43 = coeff((4,3,0))
        m_, p_, n_, q_ = coeff((5,1,1)) / coeff43, coeff((4,2,1)) / coeff43, coeff((3,3,1)) / coeff43, coeff((2,4,1)) / coeff43
        z_ = m_ / 2

        def _compute_discriminant(u, v):
            m__, p__, n__, q__ = m_, p_ - (u**2 - 2*v), n_ + 2*u*v, q_ - (v**2 - 2*u)
            return 3*m__*(m__+n__) - (p__**2 + p__*q__ + q__**2), (m__, p__, n__, q__)

        eq1 = u**3*v + 2*u**2*z_ - u**2 - 2*u*v**2 - 4*u*z_ - 4*v**2*z_ + 2*v*z_ + 2*v - p_ * (u*v - 1)
        eq2 = -2*u**2*v - 4*u**2*z_ + u*v**3 + 2*u*z_ + 2*u + 2*v**2*z_ - v**2 - 4*v*z_ - q_ * (u*v - 1)
        eqv = sp.polys.resultant(eq1, eq2, u).as_poly(v)

        # solve the exact optimal (u,v)
        for v_ in nroots(eqv, method = 'factor', real = True, nonnegative = True):
            equ = eq1.subs(v, v_).as_poly(u)
            for u_ in nroots(equ, method = 'factor', real = True, nonnegative = True):
                if u_ * v_ >= 1 and _compute_discriminant(u_, v_)[0] >= 0:
                    break
            else:
                u_ = None

            if u_ is not None:
                break
        else:
            return None

        u, v = u_, v_
        # print(u, v)

        # now we have guaranteed discriminant >= 0 in the optimal value
        # we should make rational approximations
        for u_, v_ in zip_longest(
            rationalize_bound(u, direction = 0, compulsory = True),
            rationalize_bound(v, direction = 0, compulsory = True),
        ):

            det, (m__, p__, n__, q__) = _compute_discriminant(u_, v_)
            if det >= 0:
                m_, p_, n_, q_, u, v = m__, p__, n__, q__, u_, v_
                main_solution = coeff43 * CyclicSum(c*(a**2*c-b**2*c-u*a**2*b+v*a*b**2+(u-v)*a*b*c)**2)
                rest_solution = _fast_solve_quartic(coeff,
                    m__*coeff43, p__*coeff43, n__*coeff43, q__*coeff43, rem = rem/3, mul_abc = True)
                return main_solution + rest_solution


        return None


    if coeff((4,3,0)) == 0:
        p, q = 1 , 0
    else:
        p, q = coeff.domain.to_sympy(coeff((3,4,0)) / coeff((4,3,0))).as_numer_denom()

    x, y = sp.symbols('x y')
    if sp.ntheory.primetest.is_square(p) and sp.ntheory.primetest.is_square(q):
        t = coeff((4,3,0))

        if q == 0 and coeff((3,4,0)) == 0: # abc | poly
            pass
        elif t < 0:
            return None
        else:
            z = 0
            if q == 0:
                t = coeff((3,4,0))
            else:
                z = sp.sqrt(p / q)

            m, n, p, q = coeff((5,1,1)) / t, coeff((3,3,1)) / t, coeff((4,2,1)) / t, coeff((2,4,1)) / t

            if q != 0:
                n2 = n + 2*x**2 - 2*y**2 - 4*y*z + 4*y + 2*z**2 - 4*z + 2
                p2 = p - x**2 + 2*x*y - 2*x - y**2 - 2*y
                q2 = q - x**2 - 2*x*y - 2*x*z - y**2 + 2*y*z
            else:
                n2 = n + 2*x**2 - 2*y**2 + 4*y + 2
                p2 = p - x**2 + 2*x*y - y**2
                q2 = q - x**2 - 2*x*y + 2*x - y**2 - 2*y

            discriminant = -_quartic_det(m, p2, n2, q2).as_poly(x, y)

            # print('Place 1', m, p2, n2, q2, '(P,Q) =', (p, q))
            result = optimize_discriminant(discriminant, verbose = _VERBOSE_OPTIMIZE_DISCRIMINANT)
            # print(result, print(sp.latex(discriminant)), 'here')
            if result is None:
                return None

            u, v = result[x], result[y]

            # now we have guaranteed discriminant <= 0
            if coeff((4,3,0)) != 0:
                solution = t * CyclicSum(b*(a**2*b+(z-1-2*v)*a*b*c-z*b*c**2+(u+v)*a**2*c+(v-u)*(a*c**2))**2)
            else: #
                solution = t * CyclicSum(b*(b*c**2+(-2*v-1)*a*b*c+(u+v)*a**2*c+(v-u)*a*c**2)**2)


            poly = coeff.as_poly() - solution.doit().as_poly(a,b,c)
            # print(result, poly, discriminant.subs(result),'\n',discriminant)
            poly = sp.div(poly, (a*b*c).as_poly(a,b,c))[0]

            from .solver import _structural_sos_3vars_cyclic
            new_solution = _structural_sos_3vars_cyclic(poly)
            if new_solution is not None:
                return solution + new_solution * CyclicProduct(a)

            return None


    elif p > 0 and q > 0:
        # we shall NOTE that we actually do not require a/b is square
        # take coefficients like sqrt(a/b) also works -- though it is not pretty
        # but we can make a sufficiently small perturbation such that a/b is square
        # e.g. a/b = 7/4 = z^2 + epsilon
        # solve it by Newton's algorithm for squareroot, starting with z = floor(7/4) = 1
        # 1 -> (1 + 7/4)/2 = 11/8 -> (11/8+14/11)/2 = 233/176

        perturbation = CyclicSum(c*(a-b)**2*(a*b-a*c-b*c)**2)

        solution = try_perturbations(coeff.as_poly(),
            coeff.domain.to_sympy(coeff((3,4,0))), coeff.domain.to_sympy(coeff((4,3,0))), perturbation)
        if solution is not None:
            return solution

    return None


def _sos_struct_septic_biased(coeff: Coeff):
    """
    Solve septic hexagons without s(a5b2) and s(a4b3)

    Observations:
    s(ab4(a-c)2(ab-2ac+c2)2) >= 0 or s(a3(b-c)4(a-b)2) + p(a)s(a2b-2ab2+abc)2 >= 0
    s(a(a3c-a2bc-a2c2-ab3+3ab2c-abc2-b2c2+bc3)2) + p(a)s(a2b-2ab2+abc)2 >= 0


    Examples
    -------
    (s(a(a2c-ab2)(a2c-ab2-3abc))+5s(a3b3c-a3b2c2)-2s(a2b4c-a3b2c2))

    s(a(a2c-ab2)(a2c-ab2-3abc))+s((a2-b2+2(ab-ac)+5(bc-ab))2)abc

    s(a5c2-3a4bc2+a4c3+3a3b3c-2a3b2c2)+15s(a3b3c-a3b2c2)-7s(a2b4c-a3b2c2)

    s(a5c2-16a4bc2+a4c3+54a3b3c-40a3b2c2)

    s(4a5c2-6a4b2c-12a4bc2+8a4c3-11a3b3c+17a3b2c2)

    s(a5c2+a4b2c+a4bc2-7a3b3c+4a3b2c2)

    s(a3c-a2bc)p(a+b)+9p(a)s(a2b2-a2bc)-6p(a)s(a3b-a2bc)
    """

    if not coeff.is_rational:
        return None

    if coeff((5,2,0)) or coeff((4,3,0)):
        # reflect the polynomial so that coeff((5,2,0)) == 0
        sol = _sos_struct_septic_biased(coeff.reflect())
        return align_cyclic_group(sol, coeff.gens)

    if coeff((5,2,0)) or coeff((4,3,0)):
        return None

    a, b, c = coeff.gens
    CyclicSum, CyclicProduct = coeff.cyclic_sum, coeff.cyclic_product

    if coeff((3,4,0)) == coeff((2,5,0)) and coeff((3,4,0)) != 0:
        coeff34 = coeff((3,4,0))
        m, p, n, q = coeff((5,1,1)) / coeff34, coeff((4,2,1)) / coeff34 + 2, coeff((3,3,1)) / coeff34, coeff((2,4,1)) / coeff34
        if m > 0:
            det = m*(-3*m**2 - 3*m*n - 7*m - 4*n + p**2 + p*q - p + q**2 - 2*q - 3) + p**2
            if det <= 0:
                x = -(6*m + p + 2*q + 6)/(3*m + 4)
                # r = x.as_numer_denom()[1] # cancel the denominator is good

                n2 = n - (x**2 + 4*x + 3)
                q2 = q + (2*x + 3)
                u_, v_ = -(p + 2*q2) / 3 / m, -(2*p + q2) / 3 / m

                multiplier = CyclicSum(a)
                part1 = a**2*c - a*b**2*(x + 2) + a*b*c + b**2*c*(x + 1) - b*c**2
                part2 = a**3*c - a**2*b**2 - a**2*b*c*(x + 1) - a*b**3 + a*b**2*c*(x + 2) - a*b*c**2 + b**2*c**2
                part1, part2 = part1.together().as_coeff_Mul(), part2.together().as_coeff_Mul()

                y = [
                    part1[0]**2,
                    part2[0]**2 / 2,
                    m / 2,
                    (3*(m + n2) - (p**2 + q2**2 + p*q2) / m) / 6,
                    (m + p + n + q) + coeff((3,2,2)) / coeff34
                ]
                y = [_ * coeff34 for _ in y]
                exprs = [
                    CyclicSum(a*c*part1[1]**2),
                    CyclicSum(part2[1]**2),
                    CyclicSum((a**2-b**2+(u_-v_)*a*b-u_*a*c+v_*b*c)**2) * CyclicSum(a) * CyclicProduct(a),
                    CyclicSum(a**2*(b-c)**2) * CyclicSum(a) * CyclicProduct(a),
                    CyclicSum(a)**2 * CyclicProduct(a**2)
                ]
                return sum_y_exprs(y, exprs) / multiplier

        elif m == 0 and p >= -2:
            if p == -2:
                x = (-q - 3) / 2
            0

    return None


def _sos_struct_septic_hexagon(coeff: Coeff):
    """
    Solve septic without s(a7), s(a6b), s(a6c).

    Idea: subtract something to form a star.

    Examples
    -------
    s(a5b2-a5bc+a5c2-a4b3-2a4b2c-2a4bc2-a4c3+10a3b3c-5a3b2c2)

    s(4a5b2-2a5bc+4a5c2+8a4b3-8a4b2c+a4bc2-10a4c3+2a3b3c+a3b2c2)

    (s(a2(a-b)(a2+b2-3ac+3c2))s(a2+ab)-s(a(a3-a2c+0(a2b-abc)-(ac2-abc)+5/4(bc2-abc))2))

    s(2a5b2-5a5bc+8a5c2-5a4b3+21a4b2c-21a4bc2+a4c3-7a3b3c+6a3b2c2)

    s(20a5bc+4a5c2+4a4b3-23a4b2c-15a4bc2+8a4c3-78a3b3c+80a3b2c2)

    (s(a2(a-b)(a2+ab-5bc))s(a)-s(a(a-b)(a-c))2-3s(ac(a2-bc-(c2-bc)-3/2(ab-bc))2))s(a)

    (1/5(18s(a3(13b2+5c2)(13c2+5a2))-p(13a2+5b2)s(a))-585/64s(a(a2c-b2c-8/3(a2b-abc)+7/4(ab2-abc))2))
    """

    if any(coeff(i) for i in ((7,0,0), (6,1,0), (6,0,1))):
        return None

    if (coeff((5,2,0)) == 0 and coeff((4,3,0)) == 0) or (coeff((3,4,0)) == 0 and coeff((2,5,0)) == 0):
        solution = _sos_struct_septic_biased(coeff)
        if solution is not None:
            return solution

    if coeff((5,1,1)) < 0 and coeff((5,1,1))**2 > coeff((5,2,0)) * coeff((5,0,2)) * 4:
        return None

    a, b, c = coeff.gens
    CyclicSum, CyclicProduct = coeff.cyclic_sum, coeff.cyclic_product

    if coeff((5,2,0)) == coeff((2,5,0)):
        if coeff((5,2,0)) == 0 and coeff((2,5,0)) == 0:
            return _sos_struct_septic_star(coeff)

        # Try some quintic polynomial * s(ab):
        # s(ab)s(c(a^2-b^2+u(ab-ac)+v(bc-ab))^2) to eliminate the border
        # See more details in _sos_struct_quintic_hexagon


        # Theorem 1.
        # When Det = 64*p^3-16*p^2*q^2+8*p*q*z^2+32*p*q*z-256*p*q+64*q^3-z^4-24*z^3-192*z^2-512*z >= 0,
        # we have f(a,b,c) = s(a^4b + pa^3b^2 + qa^2b^3 + ab^4 + za^3bc - (p+q+z+2)a^2b^2c) >= 0.

        # Proof: Take w = z + 8, u = 4(2p^2-qw)/(w^2-4pq), v = 4(2q^2-pw)/(w^2-4pq),
        # and t = (4pq-w^2)^2/(8(2p+2q+w)(3(p-q)^2+(w-p-q)^2)),
        # then 1 - t = Det/... >= 0, and we have
        # f(a,b,c) = t*sum c(a^2-b^2+u(ab-ac)+v(bc-ab))^2 + (1 - t)*sum c(a-b)^4 >= 0.

        coeff410 = coeff((5,2,0))

        p, q, z = [coeff((4,3,0)) / coeff410, coeff((3,4,0)) / coeff410, sp.symbols('z')]
        m0, p0, n0, q0 = [coeff((5-i,1+i,1)) for i in range(4)]

        # extract the minimum root, which is the lower bound of z
        det = 64*p**3 - 16*p**2*q**2 + 8*p*q*z**2 + 32*p*q*z - 256*p*q + 64*q**3 - z**4 - 24*z**3 - 192*z**2 - 512*z
        det = det.as_poly(z, extension = True)
        if not coeff.is_rational:
            # First check if there is exact solution,
            # in which case after the subtraction, the quartic form is tight.
            dm, dp, dn, dq = 2*coeff410, (p+z+1)*coeff410, -(z+2)*coeff410, (q+z+1)*coeff410
            m1, p1, n1, q1 = m0 - dm, p0 - dp, n0 - dn, q0 - dq
            det2 = (3*m1*(m1+n1) - (p1**2+p1*q1+q1**2)).as_poly(z)
            # GCD does not work. However, det2 is quadratic, we can solve z directly.
            for z_ in sp.polys.roots(det2):
                z_ = sp.sqrtdenest(z_)
                if z_.is_real and det(z_) == 0:
                    z = z_
                    break

        if isinstance(z, Symbol):
            z = min(nroots(det, method = 'factor', real = True))

        # check whether we succeed if we subtract s(ab)s(a^4b + pa^3b^2 + qa^2b^3 + ab^4 + za^3bc - (p+q+z+2)a^2b^2c)
        # Theorem 2.
        # Denote f = a^2-b^2+u(ab-ac)+v(bc-ab)
        # s(ab)s(cf(a,b,c)^2) - 2abcs(f(a,b,c)^2) = s(c(af(b,c,a)+bf(c,a,b))^2) >= 0
        # Especially, s(c(a-b)4)s(ab) - 4abcs(a2-ab)2 = s(a(b-c)2(a2-2ab-2ac+3bc)2) >= 0


        def _compute_mpnq(z):
            w = z + 8
            g, u, v = [w**2-4*p*q, 4*(2*p**2-q*w), 4*(2*q**2-p*w)]
            denom = 1 / (8*(2*p + 2*q + w)*(3*(p-q)**2 + (w-p-q)**2))
            dm, dp, dn, dq = [2*coeff410, (p+z+1)*coeff410, -(z+2)*coeff410, (q+z+1)*coeff410]

            # additionally restore 2abcs(f(a,b,c)^2)
            t1 = 4 * denom * coeff410
            dm, dp, dn, dq = [dm - t1*g**2, dp - t1*g*(u-2*v), dn - t1*(u**2-u*v+v**2-g**2), dq - t1*g*(v-2*u)]

            # restore 4abcs(a2-ab)2
            t2 = (1 - g**2 * denom) * coeff410 * 4
            dm, dp, dn, dq = dm - t2, dp + 2*t2, dn - 3*t2, dq + 2*t2

            m1, p1, n1, q1 = m0 - dm, p0 - dp, n0 - dn, q0 - dq
            det = (3*m1*(m1+n1) - (p1**2+p1*q1+q1**2))
            return m1, p1, n1, q1, det

        def _verify(z, check_quintic_det = True):
            if p == q and z + 8 == p + q:
                return False
            if check_quintic_det:
                det_quintic = 64*p**3 - 16*p**2*q**2 + 8*p*q*z**2 + 32*p*q*z - 256*p*q + 64*q**3 - z**4 - 24*z**3 - 192*z**2 - 512*z
                if det_quintic < 0:
                    return False
            m1, p1, n1, q1, det = _compute_mpnq(z)
            if m1 > 0 and det >= 0:
                return True
            if m1 == 0 and p1 >= 0 and q1 >= 0 and (n1 >= 0 or 4*p1*q1 >= n1**2):
                return True
            return False

        # print(_compute_mpnq(z))
        if _verify(z, check_quintic_det = False):
            if isinstance(z, sp.Float):
                for z_ in rationalize_bound(z, direction = 1, compulsory = True):
                    if _verify(z_):
                        z = z_
                        break
                else:
                    return None

            m1, p1, n1, q1, det = _compute_mpnq(z)
            # print('z =', z, '\n(m,p,n,q) =', (m1,p1,n1,q1), '\ndet =', det)
            w = z + 8
            g, u, v = [w**2-4*p*q, 4*(2*p**2-q*w), 4*(2*q**2-p*w)]
            denom = 1 / (8*(2*p + 2*q + w)*(3*(p-q)**2 + (w-p-q)**2))
            f = lambda a,b,c: g*(a**2-b**2)+u*(a*b-a*c)+v*(b*c-a*b)

            y = [denom * coeff410, (1 - g**2*denom) * coeff410]
            exprs = [
                CyclicSum(c*(a*f(b,c,a)+b*f(c,a,b)).expand()**2),
                CyclicSum(a*(b-c)**2*(a**2-2*a*b-2*a*c+3*b*c)**2)
            ]

            main_solution = sum_y_exprs(y, exprs)
            remain_solution = _fast_solve_quartic(coeff, m1, p1, n1, q1, coeff)
            if remain_solution is not None:
                return main_solution + remain_solution


    if not coeff.is_rational:
        return None


    if coeff((5,2,0)) == 0:
        p, q = 1 , 0
    else:
        p, q = coeff.domain.to_sympy(coeff((2,5,0)) / coeff((5,2,0))).as_numer_denom()

    if sp.ntheory.primetest.is_square(p) and sp.ntheory.primetest.is_square(q):
        t = coeff((5,2,0)) / q if q != 0 else coeff((2,5,0))

        if t < 0:
            return None

        # 's((a(u(a2b-abc)-v(a2c-abc)+x(bc2-abc)+y(b2c-abc)+z(ac2-abc)+w(ab2-abc))2))' (u>0,v>0)
        # z^2 + 2uw = coeff((4,3,0)) = m,   w^2 - 2vz = coeff((3,4,0)) = n
        # => (w^2-n)^2 = 4v^2(m-2uw)
        # More generally, we can find w,z such that z^2+2uw < m and w^2-2vz < n
        # such that the remaining coeffs are slightly positive to form a hexagram (star)
        m = coeff((4,3,0)) / t
        n = coeff((3,4,0)) / t

        u = sp.sqrt(q)
        v = sp.sqrt(p)
        candidates = []

        if v == 0: # first reflect the problem, then reflect it back
            u , v = v , u
            p , q = q , p
            m , n = n , m

        x = Symbol('x')
        eq = ((x**2 - n)**2 - 4*p*(m - 2*u*x)).as_poly(x)
        for w in sp.polys.polyroots.roots(eq):
            if isinstance(w, Rational):
                z = (w*w - n) / 2 / v
                candidates.append((w, z))
                # print(w, v, z, n)
            elif w.is_real is None or w.is_real == False:
                continue
            else:
                w2 = complex(w).real
                w2 -= abs(w2) / 1000 # slight perturbation
                if m < 2*u*w2:
                    continue
                z2 = (m - 2*u*w2)**0.5
                z2 -= abs(z2) / 1000 # slight perturbation
                rounding = 1e-2
                for i in range(4):
                    w = rationalize(w2, rounding = rounding, reliable = False)
                    z = rationalize(z2, rounding = rounding, reliable = False)
                    rounding *= 0.1
                    if z*z + 2*u*w <= m and w*w - 2*v*z <= n:
                        candidates.append((w, z))

        for perturbation in (100, 3, 2):
            m2 = m - abs(m / perturbation)
            n2 = n - abs(n / perturbation)
            eq = ((x**2 - n2)**2 - 4*p*(m2 - 2*u*x)).as_poly(x)
            for w in sp.polys.roots(eq):
                if isinstance(w, Rational):
                    z = (w*w - n) / 2 / v
                    candidates.append((w, z))
                elif w.is_real is None or w.is_real == False:
                    continue
                else:
                    rounding = 1e-2
                    w2 = complex(w).real
                    if m + m2 < 4*u*w2:
                        continue
                    z2 = ((m + m2)/2 - 2*u*w2)**0.5
                    for i in range(4):
                        w = rationalize(w2, rounding = rounding, reliable = False)
                        z = rationalize(z2, rounding = rounding, reliable = False)
                        rounding *= 0.1
                        if z*z + 2*u*w <= m and w*w - 2*v*z <= n:
                            candidates.append((w, z))

        candidates = list(set(candidates))
        if coeff((2,5,0)) == 0: # reflect back
            u , v = v , u
            m , n = n , m
            p , q = q , p
            candidates = [(-z, -w) for w,z in candidates]

        # sort according to their weights
        weights = [abs(i[0].p) + abs(i[0].q) + abs(i[1].p) + abs(i[1].q) for i in candidates]
        indices = sorted(range(len(candidates)), key = lambda x: weights[x])
        candidates = [candidates[i] for i in indices]
        # print(candidates, u, v, m, n)

        for w, z in candidates:
            x, y = sp.symbols('x y')
            m, n, p, q = coeff((5,1,1)) / t, coeff((3,3,1)) / t, coeff((4,2,1)) / t, coeff((2,4,1)) / t
            m2 = m + 2*u*v
            n2 = n + 2*u*w - 2*u*y + 2*u*z - 2*v*w + 2*v*x - 2*v*z + 2*w**2 + 2*w*x + 2*w*y + 4*w*z - 2*x*y + 2*x*z + 2*y*z + 2*z**2
            p2 = p + 2*u**2 - 2*u*v + 2*u*w + 2*u*x + 2*u*y + 2*u*z + 2*v*w - 2*x*z - y**2
            q2 = q - 2*u*v - 2*u*z + 2*v**2 - 2*v*w - 2*v*x - 2*v*y - 2*v*z - 2*w*y - x**2

            discriminant = -_quartic_det(m2, p2, n2, q2).as_poly(x, y)
            # print('Place 2', m2, p2, n2, q2)
            result = optimize_discriminant(discriminant, soft = True, verbose = _VERBOSE_OPTIMIZE_DISCRIMINANT)
            if result is None:
                continue

            r, s = result[x], result[y]
            # print('w z =', w, z, 'a b =', a, b, discriminant)

            expr = (u*(a**2*b-a*b*c)-v*(a**2*c-a*b*c)+r*(b*c**2-a*b*c)+s*(b**2*c-a*b*c)+z*(a*c**2-a*b*c)+w*(a*b**2-a*b*c)).expand()
            solution = t * CyclicSum(a * expr**2)
            poly2 = coeff.as_poly() - solution.doit().as_poly(a,b,c)

            from .solver import _structural_sos_3vars_cyclic
            new_solution = _structural_sos_3vars_cyclic(poly2)
            if new_solution is not None:
                return solution + new_solution
            return None

    elif p > 0 and q > 0:
        perturbation = CyclicSum(a) * CyclicProduct((a-b)**2)
        solution = try_perturbations(coeff.as_poly(),
            coeff.domain.to_sympy(coeff((2,5,0))), coeff.domain.to_sympy(coeff((5,2,0))), perturbation)

        if solution is not None:
            return solution

    return None
