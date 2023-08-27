import sympy as sp

from .sextic_symmetric import _sos_struct_sextic_hexagram_symmetric
from .utils import CyclicSum, CyclicProduct, _sum_y_exprs, _make_coeffs, inverse_substitution
from ...utils.roots.rationalize import rationalize_bound

a, b, c = sp.symbols('a b c')

def sos_struct_nonic(poly, coeff, recurrsion):
    """
    Nonic is polynomial of degree 9.
    """
    if not any(
        coeff(_) for _ in ((9,0,0),(8,1,0),(7,2,0),(6,3,0),(3,6,0),(2,7,0),(1,8,0),(7,1,1))
    ):
        if all(coeff((i,j,k)) == coeff((j,i,k)) for (i,j,k) in ((5,4,0),(6,2,1),(5,3,1))):
            return _sos_struct_nonic_hexagram_symmetric(poly, coeff, recurrsion)
    return None


def _sos_struct_nonic_hexagram_symmetric(poly, coeff, recurrsion):
    """
    Observe that
    f(a,b,c) = s(c5(a-b)4) + x^2s(a2b2c(a-b)4) - 2xp(a)s(3a4b2-4a4bc+3a4c2-4a3b3+2a2b2c2) >= 0
    because
    f*s(ab(a-b)4) = p(a)(xs(ab(a-b)4) - s(3a4b2-4a4bc+3a4c2-4a3b3+2a2b2c2))^2 + R*p(a-b)^2
    where
    R(a,b,c) = p(a-b)^2*s(a^2b+ab^2+abc) + p(a)s((a-b)^2)s(a^2(b-c)^2) / 4 >= 0

    Examples
    -------
    s(a2b-abc)s(ab2-abc)s(a(b-c)2)-13p(a-b)2p(a)

    s(2a6b2c+2a6bc2+5a5b4-34a5b3c+30a5b2c2-34a5bc3+5a5c4+16a4b4c+19a4b3c2+19a4b2c3-30a3b3c3)
    """
    c1, c2, c3, c4, c5, c6 = [coeff(_) for _ in ((5,4,0),(6,2,1),(5,3,1),(4,4,1),(5,2,2),(4,3,2))]
    if c1 < 0 or c2 < 0:
        return None
    if c1 == 0:
        solution = recurrsion(poly.div((a*b*c).as_poly(a,b,c))[0])
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

    z = sp.sqrt(c1*c2)
    if not isinstance(z, sp.Rational):
        z = z.n(20)
    if not _compute_hexagram_discriminant(z):
        return None
    if not isinstance(z, sp.Rational):
        for z_ in rationalize_bound(z, direction = -1, compulsory = True):
            if z_ < 0 or z_**2 > c1*c2:
                continue
            if _compute_hexagram_discriminant(z_):
                z = z_
                break
        else:
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
    hexgram_coeffs = lambda _: hexgram_coeffs_.get(_, 0)
    hexagram = _sos_struct_sextic_hexagram_symmetric(hexgram_coeffs)
    if hexagram is None:
        return None

    ratio = z / c2
    c2_ratio2 = c2 * ratio**2
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
        solution = _sum_y_exprs(y, exprs) / multiplier
    
    return solution