import sympy as sp
from sympy import Poly, Symbol, Rational, Float, Add, sqrt

from .quartic import sos_struct_quartic
from .septic_symmetric import sos_struct_septic_symmetric
from .utils import (
    Coeff, sum_y_exprs, nroots, rationalize_bound,
    zip_longest, align_cyclic_group, congruence_solve,
    sos_struct_handle_uncentered
)

def repeated_div(p, q):
    """Compute `p = q**k*r` so that q does not divide r. Return r."""
    div = p.div(q)
    while div[1].is_zero and (not p.is_zero):
        p = div[0]
        div = p.div(q)
    return p

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
                m / 2 * CyclicSum((a**2-b**2+(p+2*q)/3/m*(a*c-a*b)-(q+2*p)/3/m*(b*c-a*b)).together()**2),
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
    return None # still in development
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
    return None


def _sos_struct_septic_hexagon(coeff: Coeff):
    """
    Solve septic without s(a7), s(a6b), s(a6c).

    Examples
    -------
    s(a5b2-a5bc+a5c2-a4b3-2a4b2c-2a4bc2-a4c3+10a3b3c-5a3b2c2)

    s(4a5b2-2a5bc+4a5c2+8a4b3-8a4b2c+a4bc2-10a4c3+2a3b3c+a3b2c2)

    (s(a2(a-b)(a2+b2-3ac+3c2))s(a2+ab)-s(a(a3-a2c+0(a2b-abc)-(ac2-abc)+5/4(bc2-abc))2))

    s(2a5b2-5a5bc+8a5c2-5a4b3+21a4b2c-21a4bc2+a4c3-7a3b3c+6a3b2c2)

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

        p, q, z = [coeff((4,3,0)) / coeff410, coeff((3,4,0)) / coeff410, Symbol('z')]
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
            if isinstance(z, Float):
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

    return _sos_struct_septic_hexagon_sdp(coeff)


def _sos_struct_septic_star(coeff: Coeff):
    return _sos_struct_septic_star_sdp(coeff)


@sos_struct_handle_uncentered
def _sos_struct_septic_hexagon_sdp(coeff: Coeff):
    """
    Assume `F(a,b,c) = CyclicSum(a * vec' * M * vec) + a*b*c*g` where `g`
    has degree 4,

    ```
    vec = [
        c*(a**2 - b**2 + u*(a*b - a*c) + v*(-a*b + b*c)),
        b*(-a**2 + c**2 + u*(a*c - b*c) + v*(a*b - a*c)),
        -(u + v**2)*(-a*b*c + b*c**2) + (u**2 + v)*(-a*b*c + b**2*c) + (a*b**2 - a*c**2)*(-u*v + 1)
    ]
    ```

    and

    ```
    M = [[c502, m01, m02],
        [m01, c520, m12],
        [m02, m12, m22]]
    ```
    should be positive semidefinite, and `m02`, `m12` are linear with `m22` (see `_build_solution`
    for details). We need to find `u, v, m01, m22` such that the quartic discriminant of `g` is
    nonnegative, which is linear in `m01`, independent with `m22`, and degree-10 in `u, v`.
    """
    if not coeff.is_rational:
        return

    a, b, c = coeff.gens
    CyclicSum, CyclicProduct = coeff.cyclic_sum, coeff.cyclic_product

    c520, c430, c340, c502, c511, c421, c331, c241 = [
        coeff(_) for _ in [(5,2,0),(4,3,0),(3,4,0),(5,0,2),(5,1,1),(4,2,1),(3,3,1),(2,4,1)]]
    if c520 < 0 or c502 < 0:
        return

    # solve R(u,v) == S(u,v) == 0
    # this makes the remaining quartic be proportional to s((a2-b2+u(ab-ac)+v(bc-ab))2)
    R = {(0, 0): c340 + c421 - c430 - c502, (0, 1): -2*c340 + 2*c430 + 2*c511 - 4*c520, (0, 2): -3*c430 + 3*c520, (0, 3): -4*c520,
        (1, 0): -c340 - c430 + 2*c502 - c511 + 2*c520, (1, 1): -2*c421 - 2*c520, (1, 2): c340 - c430 - 4*c511 + 5*c520,
        (1, 3): 2*c430 - 2*c520, (1, 4): 3*c520, (2, 0): -c340 + c430 - c502 - c520, (2, 1): -2*c502 + 2*c511 - 2*c520,
        (2, 2): c421 + c520, (2, 3): 2*c511 - 2*c520, (3, 0): c340 - c430 - c502, (3, 2): c502 - c511 + c520,
        (4, 0): -c340 + c502, (5, 0): -c502}
    S = {(0, 0): c241 - c340 + c430 - c520, (0, 1): -c340 - c430 + 2*c502 - c511 + 2*c520, (0, 2): c340 - c430 - c502 - c520,
        (0, 3): -c340 + c430 - c520, (0, 4): -c430 + c520, (0, 5): -c520, (1, 0): 2*c340 - 2*c430 - 4*c502 + 2*c511,
        (1, 1): -2*c241 - 2*c502, (1, 2): -2*c502 + 2*c511 - 2*c520, (2, 0): -3*c340 + 3*c502,
        (2, 1): -c340 + c430 + 5*c502 - 4*c511, (2, 2): c241 + c502, (2, 3): c502 - c511 + c520,
        (3, 0): -4*c502, (3, 1): 2*c340 - 2*c502, (3, 2): -2*c502 + 2*c511, (4, 1): 3*c502}
    T = {(0, 0): c331 + c511, (0, 2): c340 - c430 - c511 + 2*c520, (0, 3): 2*c430 - 2*c520, (0, 4): 3*c520,
        (1, 1): -2*c331 + c340 + c430 - 2*c502 - c511 - 2*c520, (1, 2): -2*c340 + 2*c430 + 2*c502 + 2*c520,
        (1, 3): c340 - c430 + 2*c511 - c520, (1, 5): -c520, (2, 0): -c340 + c430 + 2*c502 - c511,
        (2, 1): 2*c340 - 2*c430 + 2*c502 + 2*c520, (2, 2): c331 + c340 + c430 + c502 - c511 + c520, (2, 4): -c511 + c520,
        (3, 0): 2*c340 - 2*c502, (3, 1): -c340 + c430 - c502 + 2*c511, (3, 3): -c502 + c511 - c520, (4, 0): 3*c502,
        (4, 2): c502 - c511, (5, 1): -c502}
    R, S, T = [coeff.from_dict(_, (a, b)).as_poly() for _ in [R,S,T]]

    res = R.resultant(S)
    divisor = coeff.from_list([1, -1, 1], (b,)).as_poly()
    res = repeated_div(res, divisor)

    success = False
    S_norm = S.l1_norm()
    for v_ in nroots(res, method='factor', real=True, nonnegative=True):
        for u_ in nroots(R.eval(1, v_), method='factor', real=True, nonnegative=True):
            if u_ * v_ <= 1:
                continue
            if T.eval((u_, v_)) >= 0 and abs(S.eval((u_, v_))) <= S_norm*1e-7:
                success = True
                break
        if success:
            break

    if not success:
        return
    # print('(u, v) =', (u_, v_))

    def _get_m01(u, v):
        """
        Discriminant of the remaining quartic is linear in `m01` and nonnegative
        slope. This function computes the threshold value of `m01` that makes the
        discriminant zero if possible.

        Meanwhile, `m01` should satisfy `c520*c502 >= m01**2` to ensure `M` to
        be positive semidefinite. It selects zero if the threshold exceeds the bound.
        """
        if c520 == 0 or c502 == 0:
            return 0
        r, s, t = R.eval((u, v)), S.eval((u, v)), T.eval((u, v))
        slope = (r*v + s*u + t)
        if slope == 0:
            threshold = -c511/2
        else:
            threshold = -c511/2 + (r**2 + r*s + s**2)/(6*(u*v - 1)**2*slope)
        if c520 * c502 >= threshold**2:
            return threshold
        if threshold > 0:
            return None
        return 0

    def _get_m22(u, v, m01):
        """
        Find `m22` such that `M` is positive semidefinite.
        Since the determinant of `M` is quadratic in `m22`, its best possible
        value is the symmetric axis of the quadratic function.
        """
        denom = ((u*v - 1)**2*(c502*u**2 + 2*c502*u + c502 + c520*v**2 \
                    + 2*c520*v + c520 - 2*m01*u*v - 2*m01*u - 2*m01*v - 2*m01))
        if denom == 0:
            return
        numer = (c340*c502*u**2 + c340*c502*u + c340*c520*v + c340*c520 - c340*m01*u*v - 2*c340*m01*u \
            - c340*m01 + c430*c502*u + c430*c502 + c430*c520*v**2 + c430*c520*v - c430*m01*u*v \
            - 2*c430*m01*v - c430*m01 + c502**2*u**3 + c502**2*u**2 - c502*c520*u**2*v - c502*c520*u*v**2 \
            + 2*c502*c520*u + 2*c502*c520*v + 2*c502*c520 + c502*m01*u**3*v - 3*c502*m01*u**2 - 2*c502*m01*u \
            + c520**2*v**3 + c520**2*v**2 + c520*m01*u*v**3 - 3*c520*m01*v**2 - 2*c520*m01*v \
            - 2*m01**2*u**2*v**2 + 4*m01**2*u*v - 2*m01**2)
        return numer/denom


    def _build_solution(u, v, m01):
        u, v, m01 = coeff.convert(u), coeff.convert(v), coeff.convert(m01)
        m22 = _get_m22(u, v, m01)
        if m22 is None:
            return None
        denom2 = 1/((u*v - 1)**2)

        lc = c511 + 2*m01
        m = lc
        p = (u - 2*v)*lc + R.eval((u, v)) * denom2
        n = (u**2 - u*v + v**2 - 1)*lc + T.eval((u, v)) * denom2
        q = (v - 2*u)*lc + S.eval((u, v)) * denom2

        quartic = _fast_solve_quartic(coeff, m, p, n, q)
        if quartic is None:
            return None

        m02 = (v + 1)/2*m22 + (-c340 - c430*v + c502*u**2*v - 2*c502*u - c520*v**2)/(2*(u*v - 1)**2)
        m12 = (u + 1)/2*m22 + (-c340*u - c430 - c502*u**2 + c520*u*v**2 - 2*c520*v)/(2*(u*v - 1)**2)
        M = coeff.as_matrix([[c502, m01, m02], [m01, c520, m12], [m02, m12, m22]], (3, 3))
        vec = [
            (a**2*c - b**2*c + u*(a*b*c - a*c**2) + v*(-a*b*c + b*c**2)),
            (-a**2*b + c**2*b + u*(a*b*c - b**2*c) + v*(a*b**2 - a*b*c)),
            -(u + v**2)*(-a*b*c + b*c**2) + (u**2 + v)*(-a*b*c + b**2*c) + (a*b**2 - a*c**2)*(-u*v + 1)
        ]
        sol = congruence_solve(M,
            mapping=lambda row: CyclicSum(a*Add(*[row[i]*vec[i] for i in range(3)]).together()**2))
        if sol is None:
            return None
        return sol + quartic


    for u__, v__ in zip_longest(
        rationalize_bound(u_, direction = 0, compulsory = True),
        rationalize_bound(v_, direction = 0, compulsory = True),
    ):
        if u__ * v__ <= 1 or u__ < 0:
            continue
        m01 = _get_m01(u__, v__)
        if m01 is None:
            continue
        sol = _build_solution(u__, v__, m01)
        if sol is not None:
            return sol


@sos_struct_handle_uncentered
def _sos_struct_septic_star_sdp(coeff: Coeff):
    """
    Solve `F(a,b,c) = s(ua4b3 + va3b4) + abcf(a,b,c) >= 0` where f is degree 4.

    Assume F(a,b,c) = CyclicSum(a*vec' * M * vec) + a*b*c*g(a,b,c)
    where `g` has degree 4, `vec = [(a*c**2-a*b*c), (a*b**2-a*b*c), (b*c**2-a*b*c), (c*b**2-a*b*c)]`
    and

    ```
    M = [[c430, m01, m02, m03],
        [m01, c340, m12, m13],
        [m02, m12, m22, m23],
        [m03, m13, m23, m33]]
    ```

    should be positive semidefinite. We wish to optimize the quartic discriminant of `g`.
    This is a convex quadratic SDP. After applying Barvinok-Pataki's thoerem,
    we can assume `M` has rank 1. This imposes `m01 = +- sqrt(c430*c340)`.

    In practice, we wish `m01` to be rational, and will subtract
    `(c340 - m01**2/c430)*CyclicSum(a**3*b**2*(b-c)**2)` from `F`.


    Examples
    ---------
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
    if not coeff.is_rational:
        return

    a, b, c = coeff.gens
    CyclicSum, CyclicProduct = coeff.cyclic_sum, coeff.cyclic_product

    c520, c430, c340, c502, c511, c421, c331, c241 = [
        coeff(_) for _ in [(5,2,0),(4,3,0),(3,4,0),(5,0,2),(5,1,1),(4,2,1),(3,3,1),(2,4,1)]]
    if c520 != 0 or c502 != 0 or c511 < 0:
        return
    if c430 <= 0 or c340 <= 0:
        return

    # the quartic discriminant (dependent on three parameters m01, m02, m03)
    disc0 = {
        (0, 0, 0): -c241**2*c430**2 - c241*c421*c430**2 + 3*c331*c430**2*c511 + 6*c340*c430**2*c511\
        - c421**2*c430**2 + 6*c430**3*c511 + 3*c430**2*c511**2, (0, 0, 1): 6*c430**2*c511,
        (0, 0, 2): c241*c430 + 2*c421*c430, (0, 0, 4): -1,
        (0, 1, 0): 2*c241*c430**2 + 4*c421*c430**2 + 6*c430**2*c511,
        (0, 1, 1): -6*c430*c511, (0, 1, 2): -4*c430, (0, 2, 0): 2*c241*c430 + c421*c430 - 4*c430**2,
        (0, 2, 2): -1, (0, 3, 0): -2*c430, (0, 4, 0): -1, (1, 0, 0): 12*c430**2*c511,
        (1, 0, 1): 4*c241*c430 + 2*c421*c430 + 6*c430*c511, (1, 0, 3): -2, (1, 1, 0): 6*c430*c511,
        (1, 1, 1): -4*c430, (1, 2, 1): -4, (2, 0, 2): -4
    }
    disc0 = coeff.from_dict(disc0, (a, b, c)).as_poly()

    # find m01, m02, m03 such that disc0 >= 0
    _sqrt = sqrt(c430 * c340)
    if not isinstance(_sqrt, Rational):
        _sqrt = _sqrt.n(15)
    _sqrt = coeff.convert(_sqrt)
    if _sqrt**2 > c430 * c340:
        _sqrt = c430 * c340 / _sqrt # so that abs(_sqrt) <= sqrt(c430 * c340)

    candidates = [-_sqrt, _sqrt]
    success = False
    for m01_ in candidates:
        disc = disc0.eval(0, m01_)
        diffu = disc.diff(0)
        diffv = disc.diff(1)
        res = diffu.resultant(diffv)
        for v_ in nroots(res, method='factor', real=True):
            for u_ in nroots(diffu.eval(1, v_), method='factor', real=True):
                if disc.eval((u_, v_)) >= 0:
                    success = True
                    break
            if success:
                break
        if success:
            break
    if not success:
        return None

    def _build_solution(u, v, m01):
        if m01**2 > c430*c340:
            return None

        m = c511
        p = (c421*c430 - 2*c430*u - v**2)/c430
        n = (c331*c430 + 2*c340*c430 + 2*c430**2 + 4*c430*m01 + 2*c430*u \
                + 2*c430*v + 2*m01*u + 2*m01*v - 2*u*v)/c430
        q = (c241*c430 - 2*m01*v - u**2)/c430
        quartic = _fast_solve_quartic(coeff, m, p, n, q)
        if quartic is None:
            return None

        ker = c430*(a*c**2-a*b*c) + m01 * (a*b**2-a*b*c) + u*(b*c**2-a*b*c) + v*(c*b**2-a*b*c)
        sol1 = CyclicSum(a*ker.together()**2) / c430 \
            + (c340 - m01**2/c430)*CyclicSum(a**3*b**2*(b-c)**2)
        return sol1 + quartic

    for u, v, z in zip_longest(
        rationalize_bound(u_, direction=0, compulsory=True),
        rationalize_bound(v_, direction=0, compulsory=True),
        [m01_] if m01_**2 == c430*c340 else \
            rationalize_bound(coeff.to_sympy(m01_).n(15),
                direction=-1 if m01_>0 else 1, compulsory=True)
    ):
        u, v, z = coeff.convert(u), coeff.convert(v), coeff.convert(z)
        sol = _build_solution(u, v, z)
        if sol is not None:
            return sol
    return None
