import sympy as sp
from sympy import Poly

from .utils import CycSum, SymSum, quadratic_weighting, radsimp, intervals

a, b, c, d = sp.symbols("a b c d")

def quarternary_quartic(coeff, real=True):
    return _quarternary_quartic_real(coeff, real=real)


def _quarternary_quartic_fluroite(coeff, real=True):
    """
    This structure gets its name from the mineral fluroite, which is an octahedral crystal.
    It considers cyclic quartic 4-var polynomials with the form:

    s(?a2b2+?a2bc+?a2bd+?a2c2+?a2cd+?abcd) >= 0

    where a, b, c, d are the coefficients of the polynomial.

    Observe that
    s((x(dc-ab+bc-da)+y(dc-ab-bc+da)+z(cd-da)+w/2(ac-bd))2)-s((z(a-c)(b-d)+w(ac-bd))2)/4
    = ((2x+z)2+4y2)s(a2b2-abcd) = ((2x+z)2/2+2y2)s((ab-cd)2) >= 0,
    we write the polynomial as a linear combination of:

    s(a2b2-abcd), s((?(a-c)(b-d)+?(ac-bd))2), s(?ab-?ac)2

    Examples
    ---------
    s(26a2b2+90a2bc-36a2bd+65a2c2-108a2cd-37abcd)
    """
    if any(coeff(_) for _ in ((4,0,0,0),(3,1,0,0),(3,0,1,0),(3,0,0,1))):
        return None
    c2200, c2110, c2101, c2020, c2011, c1111 = [coeff(_) for _ in ((2,2,0,0),(2,1,1,0),(2,1,0,1),(2,0,2,0),(2,0,1,1),(1,1,1,1))]
    c_sum = (c2011 + c2110)/-8 # coeff of s(ab-ac)^2
    c_abcd = coeff.poly111()/4
    w1, w2, w3, w4 = c2200 - c_sum, c2011 + 4*c_sum, c2020 - 4*c_sum, c2101 - 2*c_sum
    c_square = w1 + w4/2 # coeff of s(a2b2-abcd)
    w1 -= c_square # = -w4/2
    expr = quadratic_weighting(w1, w2, w3,
        mapping=lambda x,y: CycSum((x*a*b-x*b*c-x*a*d+x*c*d-y*a*c+y*b*d).together()**2)/4)
    if expr is None:
        return None
    if all(_ >= 0 for _ in [c_sum, c_square, c_abcd]):
        return sp.Add(
            expr,
            c_sum * CycSum(a*b-a*c)**2,
            c_square/2 * CycSum((a*b-c*d)**2),
            c_abcd * CycSum(a*b*c*d)
        )

def _quarternary_quartic_real(coeff, real=True):
    """
    Solve cyclic quartic 4-var homogeneous polynomials. The idea is to subtract some

    s(((1-t)(a2-b2)+0(a2-c2)+(1+t)(a2-d2)+x(ab-cd)+y(bc-ad)+z(cd-bc)+w(ac-bd))2)

    so that the rest falls in the case of fluroite. This is heuristic and may not work for all cases.
    """
    if not any(coeff(_) for _ in ((4,0,0,0),(3,1,0,0),(3,0,1,0),(3,0,0,1))):
        return _quarternary_quartic_fluroite(coeff, real=real)

    c4000, poly1111 = coeff((4,0,0,0)), coeff.poly111()
    if c4000 <= 0 or poly1111 < 0:
        return None
    c3100, c3010, c3001 = radsimp([coeff(_)/c4000 for _ in ((3,1,0,0),(3,0,1,0),(3,0,0,1))])
    c2200, c2110, c2101, c2020, c2011 = radsimp([coeff(_)/c4000 for _ in ((2,2,0,0),(2,1,1,0),(2,1,0,1),(2,0,2,0),(2,0,1,1))])

    # hes = radsimp(c2011**2 + c2011*c2020 + 2*c2011*c2101 + 2*c2011*c3100 + 2*c2011 + 4*c2020*c2101 + c2020*c2110 + c2020*c3001 \
    #     + 4*c2020*c3010 + c2020*c3100 + 2*c2101*c2110 + 2*c2101*c3001 - 8*c2101*c3010 + 2*c2101*c3100 + 8*c2101 \
    #     + c2110**2 + 2*c2110*c3001 + 2*c2110 + c3001**2 + 2*c3001 - 8*c3010**2 + 8*c3010 + c3100**2 + 2*c3100)
    # if hes < 0:
    #     return None

    const_denom = radsimp(c2011 + 2*c2020 + c2110 + c3001 - 4*c3010 + c3100 + 4)
    const_sum = radsimp(-(c2011 + c2110 + c3001 + c3100)/4)
    if const_denom == 0 or const_sum < 0:
        return None
    const_z = radsimp(-(c2011 - c2110 - c3001 + c3100)/(4*const_denom))
    const_det = radsimp(-2*(c2011**2 + c2011*c2020 + 2*c2011*c2101 + 2*c2011*c3100 + 2*c2011 + 4*c2020*c2101 + c2020*c2110 \
        + c2020*c3001 + 4*c2020*c3010 + c2020*c3100 + 2*c2101*c2110 + 2*c2101*c3001 - 8*c2101*c3010 + 2*c2101*c3100 + 8*c2101\
        + c2110**2 + 2*c2110*c3001 + 2*c2110 + c3001**2 + 2*c3001 - 8*c3010**2 + 8*c3010 + c3100**2 + 2*c3100)/const_denom)
    if const_det < 0:
        return None

    t = sp.Symbol("t")
    c_sq_p0 = Poly(radsimp(
        [c2011 + 2*c2101 + c2110 + 4*c2200 - c3001**2 + c3001 + 2*c3010 - c3100**2 + c3100, 0,
         4*c2011 + 8*c2101 + 4*c2110 + 16*c2200 - 6*c3001**2 + 4*c3001 + 8*c3010 - 6*c3100**2 + 4*c3100 + 16, 0,
         3*c2011 + 6*c2101 + 3*c2110 + 12*c2200 - 9*c3001**2 + 3*c3001 + 6*c3010 - 9*c3100**2 + 3*c3100 + 16]
    ), t)
    c_sq_p1 = Poly(radsimp([-4*c3001 + 4*c3100, 0, -12*c3001 + 12*c3100]), t)
    z_p = Poly(radsimp([c3010*const_z, 0, (3*c3010-8)*const_z]), t)
    z_p2 = z_p**2
    c_sq_p = -8*z_p2 + c_sq_p1*z_p + c_sq_p0
    w1_p = Poly(radsimp([-c2011/4 - c2101 - c2110/4 - c3001/4 - c3010 - c3100/4, 0, 
        -3*c2011/4 - 3*c2101 - 3*c2110/4 - 3*c3001/4 - 3*c3010 - 3*c3100/4]), t) - z_p2
    w2_p0 = Poly(radsimp([c2011 - c2110 - c3001 + c3100, 0, 3*c2011 - 3*c2110 - 3*c3001 + 3*c3100]), t)
    w2_p1 = Poly([c3010, 0, 3*c3010 - 8], t)
    w2_p = w2_p1 * z_p + w2_p0
    w3_p = Poly(radsimp([-c3010**2/4, 0, c2011 + 2*c2020 + c2110 + c3001 - 3*c3010**2/2 + c3100 + 4,
        0, 3*c2011 + 6*c2020 + 3*c2110 + 3*c3001 - 9*c3010**2/4 + 3*c3100 - 4]), t)

    # find t such that w3 >= 0 and c_sq >= 0
    for t_ in intervals([w3_p, c_sq_p]):
        # print(t_, w3_p(t_), c_sq_p(t_))
        if w3_p(t_) >= 0 and c_sq_p(t_) >= 0:
            break
    else:
        return None

    def _get_params2(t):
        c_sum = (t**2 + 3)*const_sum
        c_square = c_sq_p(t) / (2*(t**2 + 1))
        w1 = w1_p(t)
        w2 = w2_p(t)
        w3 = w3_p(t)
        # det = w3 * (t**2 + 3) # = 4*w1*w3 - w2**2
        return radsimp([c_sum, c_square, w1, w2, w3])

    def _get_sol(t):
        z = z_p(t)
        w = radsimp(c3010*t**2/4 + 3*c3010/4)
        x = radsimp((c3001*t**3 + 3*c3001*t + c3100*t**2 + 3*c3100 + t**2*z + 2*t*z - z)/(2*t**2 + 2))
        y = radsimp((-c3001*t**2 - 3*c3001 + c3100*t**3 + 3*c3100*t + t**2*z - 2*t*z - z)/(2*t**2 + 2))

        expr = CycSum((2*a**2+(x)*a*b+(w)*a*c+(-y)*a*d+(t-1)*b**2+(y-z)*b*c+(-w)*b*d+(-x+z)*c*d+(-t-1)*d**2).together()**2)

        return expr

    def _get_sol2(t):
        c_sum, c_square, w1, w2, w3 = _get_params2(t)
        expr = quadratic_weighting(w1, w2, w3,
            mapping=lambda x,y: CycSum((x*a*b-x*b*c-x*a*d+x*c*d-y*a*c+y*b*d).together()**2)/4)
        if expr is None or c_sum < 0 or c_square < 0:
            return None
        mul = radsimp(c4000 / (2*t**2 + 6))
        return sp.Add(
            mul * _get_sol(t),
            mul * expr,
            radsimp(mul * c_sum) * CycSum(a*b-a*c)**2,
            radsimp(mul * c_square/2) * CycSum((a*b-c*d)**2),
            (poly1111/4) * CycSum(a*b*c*d)
        )
    return _get_sol2(t_)