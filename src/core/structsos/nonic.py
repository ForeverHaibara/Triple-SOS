import sympy as sp

from .sextic_symmetric import _sos_struct_sextic_hexagram_symmetric, _sos_struct_sextic_tree
from .utils import (
    CyclicSum, CyclicProduct, Coeff,
    sum_y_exprs, rationalize_func, inverse_substitution, radsimp, nroots
)

a, b, c = sp.symbols('a b c')

def sos_struct_nonic(coeff, recurrsion, real = True):
    """
    Nonic is polynomial of degree 9.

    Examples
    --------
    s(ac2(a-b)4(b-c)2)-5p(a-b)2p(a)

    s(20a6b2c+20a6bc2+20a5b4+40a5b3c-34a5b2c2-108a5bc3+20a5c4-34a4b4c-43a4b3c2+31a4b2c3+68a3b3c3)

    s(a6b3+7a6c3-29a5b2c2+12a4bc4+9a3b3c3)

    Reference
    -------
    [1] https://tieba.baidu.com/p/8457240407

    [2] https://tieba.baidu.com/p/7303219331
    """
    if not any(coeff(_) for _ in ((9,0,0),(8,1,0),(7,2,0),(2,7,0),(7,1,1))):
        if not any(coeff(_) for _ in ((6,3,0),(3,6,0),(1,8,0))):
            if all(coeff((i,j,k)) == coeff((j,i,k)) for (i,j,k) in ((5,4,0),(6,2,1),(5,3,1),(4,3,2))):
                return _sos_struct_nonic_hexagram_symmetric(coeff, recurrsion)
            if (coeff((5,4,0)) == 0 and coeff((6,1,2)) == 0) or (coeff((4,5,0)) == 0 and coeff((6,2,1)) == 0):
                return _sos_struct_nonic_gear(coeff, recurrsion)

        if not any(coeff(_) for _ in ((6,2,1),(2,6,1),(5,4,0),(4,5,0))):
            if all(coeff((i,j,k)) == coeff((j,i,k)) for (i,j,k) in ((6,0,3),(5,3,1),(4,3,2))):
                return _sos_struct_nonic_hexagon_symmetric(coeff, recurrsion)

    if not any(coeff(_) for _ in 
        ((8,1,0),(7,2,0),(5,4,0),(5,0,4),(8,0,1),(7,0,2),
         (6,2,1),(6,1,2),(5,3,1),(5,1,3),(4,3,2),(4,2,3))
    ):
        if coeff((9,0,0)) > 0 and all(coeff((i,j,k)) == coeff((j,i,k)) for (i,j,k) in ((9,0,0),(6,3,0))):
            return _sos_struct_nonic_symmetric_tree(coeff)

    return None


def _sos_struct_nonic_symmetric_tree(coeff):
    """
    Solve problems with similar structure as s(a3-abc)s(a2+xab)s((a-b)2(a+b-xc)2).
    See details at _sos_struct_sextic_tree.

    The idea is to subtract (x >= 1):
    G(x) = s(a3-abc)/2s(a2+xab)s((a-b)2(a+b-xc)2)+(w-x3+3x-1)s(a(ab-c2)2(ac-b2)2)
    to remove the outer edge so that the rest is a multiple of a^2b^2c^2. Here the coefficient of a^9
    and a^6b^3 of G(x) are 1 and w respectively.

    Note that G'(x) = -3(2x-1)p(a)s(a)2s(a2-ab)2 <= 0. Hence x should be as large as possible, which
    would be the root of -x^3+3x-1+w == 0.

    Examples
    -------
    s(a9+3a3b3(a3+b3)-5a7bc-4a4b4c+2p(a3))

    s(4a9-25a7bc+14a6b3+14a6c3-16a4b4c+9a3b3c3)

    4/3s(a3-2/3abc)/2s(a2+3ab)s((a-b)2(a+b-3c)2)+p(a)s(a3b3-a2b2c2)+ s(a3(b3-c3)2)
    """
    c0 = coeff((9,0,0))
    if c0 <= 0:
        return None

    w, c6, c33, c411 = coeff((6,3,0)), coeff((7,1,1)), coeff((4,4,1)), coeff((5,2,2))
    w, c6, c33, c411 = radsimp([w / c0, c6 / c0, c33 / c0, c411 / c0])
    if w < -1:
        return None

    def _compute_tree_coeffs(x):
        c6_ = radsimp(c6 - (-3*x**2 + 3*x - 3))
        c33_ = radsimp(c33 - (-3*w - 6*x**2 + 6*x + 3))
        c411_ = radsimp(c411 - (-3*w + 18*x**2 - 18*x))
        return c6_, c33_, c411_

    def _check_valid_x(x):
        c6_, c33_, c411_ = _compute_tree_coeffs(x)
        if c6_ < 0:
            return False
        if c6_ == 0 and c33_ >= 0 and c411_ >= 0:
            return True
        if c6_ > 0:
            u, v = c33_ / c6_, c411_ / c6_
            if u >= 2 and v >= -6:
                return True
            if u >= -1:
                det = 27*u**2 + 27*u*v + 54*u + v**3 + 18*v**2 + 54*v
                if (v >= -6 and det <= 0) or (v < -6 and det >= 0):
                    return True
        return False

    x = sp.Symbol('x')
    eq = (-x**3 + 3*x - 1 + w).as_poly(x)
    if not coeff.is_rational:
        return None

    x = rationalize_func(eq, _check_valid_x, validation_initial = lambda x: x >= 1, direction = 1)

    if x is not None:
        c6_, c33_, c411_ = _compute_tree_coeffs(x)
        def _get_func_x(x):
            # return pretty form of s(a2+xab)s((a-b)2(a+b-xc)2)
            if x == 2:
                return 2 * CyclicSum(a)**2 * CyclicSum(a**2-b*c)**2
            p, q = x.as_numer_denom()
            return radsimp(1/q) * CyclicSum(q*a**2 + p*b*c) * CyclicSum((a-b)**2 * (a+b-x*c)**2)

        solution = sp.Add(
            c0/4 * CyclicSum(a) * CyclicSum((a-b)**2) * _get_func_x(x),
            radsimp(c0 * eq(x)) * CyclicSum(a*(a*b - c**2)**2 * (a*c - b**2)**2)
        )
        poly111 = radsimp(3 * c0 * (1 + 2*w + c6 + c33 + c411) + coeff((3,3,3)))
        if poly111 < 0:
            return None
        c6_, c33_, c411_ = radsimp([c6_ * c0, c33_ * c0, c411_ * c0])
        tree_coeffs = {
            (6,0,0): c6_, (3,3,0): c33_, (4,1,1): c411_,
            (2,2,2): radsimp((-c6_-c33_-c411_)*3 + poly111)
        }
        tree_solution = _sos_struct_sextic_tree(Coeff(tree_coeffs))
        if tree_solution is not None:
            return solution + tree_solution * CyclicProduct(a)

    return None


def _sos_struct_nonic_hexagon_symmetric(coeff, recurrsion):
    """
    Solve problems like s(a6b3+a3b6) + p(a)(...).

    Examples
    -------
    s(4a6b3+4a6c3-59a5b3c+30a5b2c2-59a5bc3+74a4b4c+90a4b3c2+90a4b2c3-174a3b3c3)

    s(a3)s(a3b3)+27p(a3)-6p(a2)s(ab(a+b))
    """
    c0 = coeff((6,3,0))
    if c0 <= 0:
        # this should be handled elsewhere
        return None

    c42, c33, c411, c321 = coeff((5,3,1)), coeff((4,4,1)), coeff((5,2,2)), coeff((4,3,2))
    c42, c33, c411, c321 = radsimp([c42 / c0, c33 / c0, c411 / c0, c321 / c0])

    if c42 >= 0:
        # first try simple case, use s(a(ab-c2)2(ac-b2)2)
        c33_ = c33 + 3 + 2 * c42
        c411_ = c411 + 3 + 2 * c42
        c321_ = c321 - 2 * c42
        if c33_ >= 0 and c411_ >= 0:
            if c33_ + c411_ + c321_ >= 0 or radsimp(c33_ * c411_ - (c33_ + c411_ + c321_)**2) >= 0:
                solution = c0 * CyclicSum(a*(a*b - c**2)**2 * (a*c - b**2)**2) + radsimp(c0 * c42) * CyclicProduct(a) * CyclicProduct((a-b)**2)
                c33_, c411_, c321_ = radsimp([c33_ * c0, c411_ * c0, c321_ * c0])
                hexagram_coeffs_ = {
                    (4,1,1): c411_, (3,3,0): c33_, (3,2,1): c321_, (2,3,1): c321_,
                    (2,2,2): radsimp((-c411_-c33_-c321_*2)*3 + coeff.poly111())
                }
                hexagram_solution = _sos_struct_sextic_hexagram_symmetric(Coeff(hexagram_coeffs_))
                if hexagram_solution is not None:
                    return solution + CyclicProduct(a) * hexagram_solution

    # Assuming c0 = 1
    # compute the coeffs after subtracting s(ab2-rabc)2s(ab2-abc)+s(ac2-rabc)2s(ac2-abc)+({c42}+(6r+3))p(a-b)2p(a)
    def _compute_hexagon_coeffs(r):
        diff = -12*r - 2*c42
        c33_ = c33 - diff
        c411_ = c411 - diff
        c321_ = c321 - (9*r**2 + 18*r + 2*c42)
        if not isinstance(r, sp.Symbol):
            c321_ = radsimp(c321_)
        return c33_, c411_, c321_

    def _check_valid_r(r):
        if c42 + 6*r + 3 < 0:
            return False
        c33_, c411_, c321_ = _compute_hexagon_coeffs(r)
        if c33_ < 0 or c411_ < 0:
            return False
        tolerance = 0 if isinstance(r, sp.Rational) else 1e-12
        if c33_ + c411_ + c321_ >= 0 or c33_ * c411_ - (c33_ + c411_ + c321_)**2 >= -tolerance:
            return True
        return False

    r = sp.Symbol('r')
    c33_, c411_, c321_ = _compute_hexagon_coeffs(r)
    det = (c33_ * c411_ - (c33_ + c411_ + c321_)**2).as_poly(r)

    # print(sp.latex(det.subs(r,'x')), det(2))
    if coeff.is_rational:
        r = rationalize_func(det, _check_valid_r, validation_initial = lambda r: r >= 0, direction = 1)
    else:
        r = None

    if r is not None:
        solution = sp.Add(
            c0 * CyclicSum(a*b*(b-r*c))**2 * (CyclicSum(a*b**2 - CyclicProduct(a))),
            c0 * CyclicSum(a*c*(c-r*b))**2 * (CyclicSum(a*c**2 - CyclicProduct(a))),
            radsimp(c0 * (c42 + 6*r + 3)) * CyclicProduct(a) * CyclicProduct((a-b)**2)
        )

        rest_poly = coeff.as_poly() - solution.doit().as_poly(a,b,c)
        if rest_poly.is_zero:
            return solution

        rest_solution = recurrsion(rest_poly, real = False)
        if rest_solution is None:
            return None
        return solution + rest_solution


def _sos_struct_nonic_hexagram_symmetric(coeff, recurrsion):
    """
    Observe that
    f(a,b,c) = s(c5(a-b)4) + x^2s(a2b2c(a-b)4) - 2xp(a)s(3a4b2-4a4bc+3a4c2-4a3b3+2a2b2c2) >= 0
    because
    f*s(ab(a-b)4) = p(a)(xs(ab(a-b)4) - s(3a4b2-4a4bc+3a4c2-4a3b3+2a2b2c2))^2 + R*p(a-b)^2
    where
    R(a,b,c) = p(a-b)^2*s(a)s(ab) + p(a)s((a-b)^2)s(a^2(b-c)^2) / 4 >= 0

    Examples
    -------
    s(a2b-abc)s(ab2-abc)s(a(b-c)2)-13p(a-b)2p(a)

    s(2a6b2c+2a6bc2+5a5b4-34a5b3c+30a5b2c2-34a5bc3+5a5c4+16a4b4c+19a4b3c2+19a4b2c3-30a3b3c3)
    """
    c1, c2, c3, c4, c5, c6 = [coeff(_) for _ in ((5,4,0),(6,2,1),(5,3,1),(4,4,1),(5,2,2),(4,3,2))]
    if c1 < 0 or c2 < 0:
        return None
    if c1 == 0:
        solution = recurrsion(coeff.as_poly().div((a*b*c).as_poly(a,b,c))[0])
        if solution is None:
            return None
        return CyclicProduct(a) * solution

    if c2 == 0:
        poly2 = CyclicSum(
            coeff((5,4,0)) * a**5*(b+c) + coeff((5,3,1)) * a**4*(b**2+c**2) + coeff((5,2,2)) * a**3*b**3\
            + coeff((4,4,1)) * a**4*b*c + coeff((4,3,2)) * a**3*b*c*(b+c)
        ).doit().as_poly(a,b,c) + (coeff((3,3,3)) * a**2*b**2*c**2).as_poly(a,b,c)
        solution = recurrsion(poly2)
        if solution is None:
            return None
        return inverse_substitution(solution, factor_degree = 1)

    def _compute_hexagram_coeffs(z):
        # Assume we subtract (c1)s(c5(a-b)4) +(c2)s(a2b2c(a-b)4) - 2zp(a)s(3a4b2-4a4bc+3a4c2-4a3b3+2a2b2c2) + wp(a-b)^2p(a)
        # where c1*c2 >= z^2
        # so that the remaining is a hexagram * p(a).
        w = c3 + 4*(c1 + c2) + 6*z
        c4_ = c4 - (-2*w+6*c2+8*z)
        c5_ = c5 - (-2*w+6*c1+8*z)
        c6_ = c6 - (2*w)
        return w, c4_, c5_, c6_


    def _compute_hexagram_discriminant(z):
        # We will check whether the hexagram is positive by discriminant
        # For details see _sos_struct_sextic_hexagram_symmetric

        w, c4_, c5_, c6_ = _compute_hexagram_coeffs(z)
        if w < 0 or c4_ < 0 or c5_ < 0:
            return False
        # print('z =', z, '(w,c4,c5,c6) =', (w, c4_, c5_, c6_))

        if c4_ == 0:
            return c5_ + c6_ >= 0
        if c5_ == 0:
            return c4_ + c6_ >= 0
        tolerance = 0 if isinstance(z, sp.Rational) else 1e-12
        if c4_*c5_ >= (c4_ + c5_ + c6_)**2 - tolerance or c4_ + c5_ + c6_ >= 0:
            return True
        return False


    def _is_valid_z(z):
        return z >= 0 and z**2 <= c1*c2 and _compute_hexagram_discriminant(z)
    eq_z = sp.Poly.from_list([1, 0, -c1*c2], sp.Symbol('z'))
    z = rationalize_func(eq_z, _is_valid_z, validation_initial = lambda z: z >= 0, direction = -1)
    if z is None:
        return None

    # Now we have z >= 0 such that the hexagram is positive
    w, c4_, c5_, c6_ = _compute_hexagram_coeffs(z)
    c7_ = coeff((3,3,3)) + 6*w + 12*z
    hexgram_coeffs_ = {
        (3,3,0): c4_,
        (4,1,1): c5_,
        (3,2,1): c6_,
        (3,1,2): c6_,
        (2,2,2): c7_,
    }
    hexagram = _sos_struct_sextic_hexagram_symmetric(Coeff(hexgram_coeffs_))
    if hexagram is None:
        return None

    ratio = radsimp(z / c2)
    c2_ratio2 = radsimp(c2 * ratio**2)
    solution = None
    if w >= 4* c2 * ratio:
        # Note that for simple case,
        # s(c5(a-b)4) + x^2s(a2b2c(a-b)4) - 2xp(a)s(3a4b2-4a4bc+3a4c2-4a3b3+2a2b2c2) + 4xp(a-b)^2p(a)
        # = s(c(a-b)4(xab-c2)2) >= 0
        solution = sp.Add(*[
            c2 * CyclicSum(c*(a-b)**4*(a*b - ratio*c**2)**2),
            (w - 4*c2*ratio) * CyclicProduct((a-b)**2) * CyclicProduct(a),
            (c1 - c2_ratio2) * CyclicSum(c**5*(a-b)**4),
            hexagram * CyclicProduct(a),
        ])

    elif w >= 0:
        p1 = a*b*(a-b)**4 - ratio * (3*a**4*b**2 - 4*a**4*b*c + 3*a**4*c**2 - 4*a**3*b**3 + 2*a**2*b**2*c**2)
        p1 = p1.expand().together().as_coeff_Mul()

        multiplier = CyclicSum(a*b*(a-b)**4)
        p2 = w * CyclicProduct(a) * multiplier\
            + c2_ratio2 * CyclicProduct((a-b)**2) * CyclicSum(a) * CyclicSum(a*b) \
            + c2_ratio2 / 4 * CyclicProduct(a) * CyclicSum((a-b)**2) * CyclicSum(a**2*(b-c)**2)
        p2 = p2.together().as_coeff_Mul()

        y = [
            p1[0]**2 * c2,
            c1 - c2_ratio2,
            p2[0],
            sp.S(1),
        ]
        exprs = [
            CyclicProduct(a) * CyclicSum(p1[1])**2,
            multiplier * CyclicSum(c**5*(a-b)**4),
            CyclicProduct((a-b)**2) * p2[1],
            hexagram * multiplier * CyclicProduct(a)
        ]
        solution = sum_y_exprs(y, exprs) / multiplier
    
    return solution


def _sos_struct_nonic_gear(coeff, recurrsion):
    """
    Solve problems like
    s(ac^2(a-b)^4(b-c)^2)-5p(a-b)^2p(a) >= 0

    There exists a very complicated solution by quadratic form.
    However, there exists much easier solution:
    s(c(a4b-3a3b2+5a3bc-3a3c2-a2b3-a2b2c-a2bc2+5ab3c-4ab2c2+4abc3-b3c2-b2c3)2)+6p(a)p(a-b)2s(a2-ab)+p(a)s(2a3b+a3c-6a2b2+3a2bc)2

    Examples
    -------
    s(a6b2c-a5b3c+a5c4-a4b3c2)

    s(ac2((b(a2-b2+1(ab-ac)+3(bc-ab)))+(((a2c-b2c)-2(a2b-abc)+5(ab2-abc))))2)

    References
    ----------
    [1] https://tieba.baidu.com/p/8457240407
    """
    if not (coeff((5,4,0)) == 0 and coeff((6,1,2)) == 0):
        # reflect the polynomial
        reflect_coeffs = lambda _: coeff((_[0], _[2], _[1]))
        solution = _sos_struct_nonic_gear(reflect_coeffs, recurrsion)
        if solution is not None:
            solution = solution.xreplace({b:c, c:b})
        return solution
    if coeff((4,5,0)) == 0 or coeff((6,2,1)) == 0:
        return None

    if True:
        # First try easy cases
        # Consider the following:
        # s(ac^2((b(?(a^2-b^2)+?(ab-ac)+?(bc-ab)))+((?(a^2c-b^2c)+?(a^2b-abc)+?(ab^2-abc))))^2) >= 0
        # Note that b(ab-ac) and ab^2-abc are equivalent, we can merge two into one.

        # Coefficients of (a^2-b^2) and (a^2c-b^2c) are determined by the vertices.
        # Coefficients of (bc-ab) and (ab^2-abc) are determined by the inner hexagon.
        # Finally we choose proper coefficient of (ab-ac) so that the remaining is symmetric,
        # which would be easy to handle.

        # First normalize the coefficient so that coeff((4,5,0)) == 1
        c1, c2, c42, c33, c24 = coeff((4,5,0)), coeff((6,2,1)), coeff((5,3,1)), coeff((4,4,1)), coeff((3,5,1))
        c41, c32, c23 = coeff((5,2,2)), coeff((4,3,2)), coeff((3,4,2))
        c2, c42, c33, c24 = c2 / c1, c42 / c1, c33 / c1, c24 / c1
        c41, c32, c23 = c41 / c1, c32 / c1, c23 / c1
        if c1 < 0 or c2 < 0:
            return None

        def _compute_params(z):
            # We expect that z^2 = c2 in the ideal case. However, when z is rational,
            # we perturb some s(ab^2c^2(a-b)^4) to make c2_sqrt rational.
            w = radsimp(c2 - z**2)
            c41_, c32_, c23_ = c41 + 3*w, c32 + 4*w, c23 - 6*w
            v = c24 / 2 - z # coeff of (a^2b-abc)
            u = radsimp(1 - c42 / 2 / z) # coeff of (bc-ab)

            frac0, frac1 = -(-3*u**2 - 2*u*z + 6*u + 3*v**2 + 6*v*z - 2*v - (c32_ - c23_) + 2*z**2 - 2), (6*(u + v + z - 1))
            if frac1 == 0:
                if frac0 != 0:
                    x = sp.nan
                else:
                    # x = 2 - 2*z - u - v # deprecated
                    x = 1 - v + sp.sqrt(-u - v + 1)
                    if not isinstance(x, sp.Rational):
                        x = 1 - v
            else:
                x = radsimp(frac0 / frac1)
            c41_ = c41_ - (2*u*z + v**2 + 2*v*z - 2*x*z + z**2)
            c33_ = c33 - (u**2 - 2*u - 2*v - 2*x + 1)
            c32_ = c32_ - (-2*u**2 - 2*u*v + 2*u*x - 2*u*z + 4*u + v**2 + 4*v*x + 2*v*z + x**2 + 4*x*z - 2*x - 2)
            return (u,v,x), radsimp((c41_, c33_, c32_))

        # linear combination of vertex 1 and 2

        def _check_valid_weight(vertex1, vertex2, w):
            if w is sp.nan or (not 0 <= w <= 1):
                return False
            # w * vertex1 + (1 - w) * vertex2
            c41_, c33_, c32_ = [w*vertex1[1][i] + (1-w)*vertex2[1][i] for i in range(3)]
            if c41_ < 0 or c33_ < 0:
                return False
            if c41_ + c33_ + c32_ >= 0 or c41_ * c33_ >= (c41_ + c33_ + c32_)**2:
                return True
            return False

        def _search_valid_weight(vertex1, vertex2):
            if vertex1[0][-1] is sp.nan or vertex2[0][-1] is sp.nan:
                return None
            x1, y1, z1 = vertex1[1]
            x2, y2, z2 = vertex2[1]
            candidates = [sp.S(0), sp.S(1), x2/(x2 - x1), y2/(y2 - y1)]
            for w_ in candidates:
                if _check_valid_weight(vertex1, vertex2, w_):
                    return w_
            
            # symmetric axis of a quadratic function
            g, h = x1 + y1 + z1, x2 + y2 + z2
            sym_axis = (2*g*h - 2*h**2 - x1*y2 - x2*y1 + 2*y1*y2)/(2*(-g**2 + 2*g*h - h**2 + x1*x2 - x1*y2 - x2*y1 + y1*y2))
            
            # t = sp.symbols('x')
            # eq = (x1*t + x2*(1-t))*(y1*t + y2*(1-t)) - (g*t + h*(1-t))**2
            # print(sp.latex(eq))
            if _check_valid_weight(vertex1, vertex2, sym_axis):
                return sym_axis

        def _is_valid_z(z):
            if not (z >= 0 and z**2 <= c2):
                return False
            vertex1 = _compute_params(z)
            vertex2 = _compute_params(-z)
            w = _search_valid_weight(vertex1, vertex2)
            return (w is not None)

        eq_z = sp.Poly.from_list([1, 0, -c2], sp.Symbol('z'))
        z = rationalize_func(eq_z, _is_valid_z, validation_initial = lambda z: z >= 0, direction = -1)

        if z is not None:
            vertex1 = _compute_params(z)
            vertex2 = _compute_params(-z)
            w = _search_valid_weight(vertex1, vertex2)
            c41_, c33_, c32_ = radsimp([w*vertex1[1][i] + (1-w)*vertex2[1][i] for i in range(3)])
            hexagram_coeffs_ = {
                (4,1,1): c41_, (3,3,0): c33_, (3,2,1): c32_, (2,3,1): c32_, (3,1,2): c32_,
                (2,2,2): radsimp((-c41_-c33_-c32_*2)*3 + coeff.poly111())
            }
            hexagram_solution = _sos_struct_sextic_hexagram_symmetric(Coeff(hexagram_coeffs_))
            if hexagram_solution is not None:
                def _get_ker(z, vertex, as_coeff_Mul = True):
                    u, v, x = vertex[0]
                    ker = b*(z*(a**2-b**2)+x*(a*b-a*c)+u*(b*c-a*b)) + a**2*c-b**2*c + v*(a**2*b-a*b*c)
                    if as_coeff_Mul:
                        ker = ker.expand().together().as_coeff_Mul()
                    return ker
                ker1 = _get_ker(z, vertex1)
                ker2 = _get_ker(-z, vertex2)
                y = radsimp([
                    ker1[0]**2 * w * c1,
                    ker2[0]**2 * (1-w) * c1,
                    (c2 - z**2) * c1,
                    c1
                ])
                exprs = [
                    CyclicSum(a*c**2*ker1[1]**2),
                    CyclicSum(a*c**2*ker2[1]**2),
                    CyclicSum(a*b**2*c**2*(a-b)**4),
                    hexagram_solution * CyclicProduct(a),
                ]
                return sum_y_exprs(y, exprs)