import sympy as sp

from .utils import CyclicSum, CyclicProduct, SymSum, radsimp, rationalize_func
from ..sparse import sos_struct_quadratic

a, b, c, d = sp.symbols("a b c d")

def quarternary_cubic_symmetric(coeff, real = True):
    """
    Solve quarternary symmetric cubic polynomials. Symmetry is not checked here.

    References
    -----------
    [1] https://tieba.baidu.com/p/9033429329
    """
    c3000, c2100, c1110 = coeff((3,0,0,0)), coeff((2,1,0,0)), coeff((1,1,1,0))
    x, y, z = c3000, c2100/2 + c1110/6, c3000 + c2100
    if x >= 0 and y >= 0 and z >= 0:
        f1 = x * SymSum(a*(b-c)**2*(b+c-a)**2) + 4*x * a*b*c*d * SymSum(a)
        f2 = y * SymSum(a*b*c)
        f3 = z/4 * SymSum(a*(b-c)**2)
        return f1/SymSum(a*b) + f2 + f3

    x, y, z = c1110 + 3*c2100 + c3000, -c1110 - 3*c2100, c1110/3 + 2*c2100 + c3000
    if x >= 0 and y >= 0 and z >= 0:
        f1 = x * SymSum(a*(b-c)**2*(b+c-a)**2) + 4*x * a*b*c*d * SymSum(a)
        f2 = 2*y/3 * SymSum(a*(b-c)**2*(b+c-a)**2) + y/3 * SymSum(a*(b-c)**2*(b+c-d)**2) + y * SymSum(a*b*c*(a-b)**2)
        f3 = z/4 * SymSum(a*(b-c)**2)
        return (f1 + f2)/SymSum(a*b) + f3



#####################################################################
#
#                   "Nonhomogeneous" from 3-vars
#
#####################################################################

def _quarternary_cubic_partial_symmetric(coeff, real = False):
    """
    The function is equivalent to solving nonhomogeneous 3-var symmetric cubic polynomials in
    the form of
    c100*(a+b+c) + c000 + c1*(a^2+b^2+c^2-a*b-b*c-c*a) + c2*(a^2+b^2+c^2-2*(a*b+b*c+c*a)) + a*b*c >= 0.
    The poly is now homogenized by adding a new variable d.

    Theorem:
    s(a)2(s(a)+3abc+2s(a2-2ab))
    = s((b-c)2(b+c-a)2)+2s(ab(a-b)2)+3s(a)s(ab(c-1)2)+s(a)s((a-b)2)/2

    s(a)2(1+2abc+s(a2-2ab))
    = s((b-c)2(b+c-a)2)/2+s((a-b)2)/2+s(ab(a-b)2)+3s(ab(c-1)2)+2p(a)s(a-1)2

    We always represent the polynomial as 
          t * (3*v**2/4*(a+b+c) + v*(a**2+b**2+c**2-2*(a*b+b*c+c*a)) + a*b*c)
    + (1-t) * (4*v**3 + v*(a**2+b**2+c**2-2*(a*b+b*c+c*a)) + a*b*c) >= 0

    # TODO: It seems it is incomplete.

    Examples
    ---------
    4abc+9s(a2)-14s(ab)+4s(a)+4

    29/3s(a2)+13/3s(a)-58/3s(bc)+1+15abc

    1+2abc+s(a2-2ab)

    s(2a2-4ab)+3abc+s(a)

    s(a2-ab)-2s(a)+4+2abc
    """
    if coeff((3,0,0,0)) or coeff((2,1,0,0)):
        return None

    c100, c000, c200, c110, c111 = [coeff(_) for _ in [(1,0,0,2), (0,0,0,3), (2,0,0,1), (1,1,0,1), (1,1,1,0)]]
    if c111 == 0:
        return sos_struct_quadratic(coeff)
    elif c111 < 0 or c000 < 0:
        return None
    # normalize
    c100, c000, c200, c110 = radsimp([_/c111 for _ in [c100, c000, c200, c110]])

    c1 = 2*c200 + c110
    c2 = -c200 - c110
    if c1 < 0 or c2 < 0:
        return None

    def get_sol1(v):
        # solve 3*v**2/4*(a+b+c) + v*(a**2+b**2+c**2-2*(a*b+b*c+c*a)) + a*b*c >= 0
        # (8*a*b*v*(a - b)**2 + 2*a*b*(2*c - 3*v)**2*(a + b + c) + 3*v**2*(a - b)**2*(a + b + c) + 4*v*(b - c)**2*(-a + b + c)**2)
        return sp.Add(*[
            radsimp(4*v) * CyclicSum((b-c)**2*(b+c-a)**2) * d,
            radsimp(3*v**2) * CyclicSum(a) * CyclicSum((a-b)**2) * d**2,
            2 * CyclicSum(a) * CyclicSum(a*b*(2*c-3*v*d)**2),
            radsimp(8*v) * CyclicSum(a*b*(a-b)**2) * d,
        ]) / (8 * CyclicSum(a)**2)


    def get_sol2(v):
        # solve 4*v**3 + v*(a**2+b**2+c**2-2*(a*b+b*c+c*a)) + a*b*c >= 0
        return sp.Add(*[
            v * CyclicSum((b-c)**2*(b+c-a)**2) * d,
            radsimp(4*v**3) * CyclicSum((a-b)**2) * d**3,
            radsimp(6*v) * CyclicSum(a*b*(c-2*v*d)**2) * d,
            radsimp(2*v) * CyclicSum(a*b*(a-b)**2) * d,
            2 * CyclicProduct(a) * CyclicSum(a-2*v*d)**2
        ]) / (2 * CyclicSum(a)**2)

    
    # find x, y, t such that
    # normalized poly >= get_sol1(x) * t + get_sol2(y) * (1-t)
    # 3*x**2/4 * t <= c100, 4*y**3 * (1-t) <= c000, x*t + y*(1-t) = c2
    # Cancelling y, t by x, we require x such that 4*x*(3*c2*x - 4*c100)**3/3/(3*x**2 - 4*c100)**2 <= c000
    # and 3x^2/4 >= c100 (so that t <= 1).

    def get_const(x):
        if x == 0: return c000 - 4*c2**3 # t = 0, y = c2
        if x == c2: return c000 # t = 1, y = 0
        return radsimp(c000 - 4*x*(3*c2*x - 4*c100)**3/3/(3*x**2 - 4*c100)**2)
    
    def check_x(x):
        if x == 0: return c100 == 0 and get_const(x) >= 0 # t = 0, y = c2
        if x is None or (not x.is_finite) or (not 3*x**2/4 >= c100):
            return False
        r = get_const(x)
        return r.is_finite and r >= 0

    for x in [0, radsimp(4*c100/3/c2), None]:
        if check_x(x):
            break
    if x is None:
        det = radsimp(c000**2 + 6*c000*c100*c2 - 4*c000*c2**3 + 4*c100**3 - 3*c100**2*c2**2)
        if det < 0:
            return None
        # find x near 2*(c2 + sqrt(c2**2 - c100))/3
        if c2**2 - c100 < 0:
            return None

        x = rationalize_func(sp.Poly([9, -12*c2, 4*c100], sp.Symbol('x')), validation=check_x,
                            validation_initial=lambda x: x >= 2*c2/3, direction = 1)
    # print(x, get_const(x), 't =', radsimp(c100 / (3*x**2/4)))
    if x is None:
        return None
    res = get_const(x)
    if res < 0:
        return None
    t1 = radsimp(c100 / (3*x**2/4)) if x != 0 else 0
    t2 = 1 - t1
    if t1 < 0 or t2 < 0:
        return None
    y = (c2 - x*t1)/t2 if t2 != 0 else 0

    return sp.Add(
        radsimp(c111 * t1) * get_sol1(x),
        radsimp(c111 * t2) * get_sol2(y),
        radsimp(c111 * res) * d**3,
        radsimp(c111 * c1/2) * CyclicSum((a-b)**2)*d
    )