import sympy as sp

from .utils import SymSum

a, b, c, d = sp.symbols("a b c d")

def quarternary_cubic_symmetric(coeff, recurrsion = None, real = True):
    """
    Solve quarternary symmetric cubic polynomials. Symmetricity is not checked here.

    References
    -----------
    [1] https://tieba.baidu.com/p/9033429329
    """
    c3000, c2100, c1110 = coeff((3,0,0,0)), coeff((2,1,0,0)), coeff((1,1,1,0))
    x, y, z = c1110 + 3*c2100 + c3000, -c1110 - 3*c2100, c1110/3 + 2*c2100 + c3000
    if x >= 0 and y >= 0 and z >= 0:
        f1 = x * SymSum(a*(b-c)**2*(b+c-a)**2) + 4*x * a*b*c*d * SymSum(a)
        f2 = 2*y/3 * SymSum(a*(b-c)**2*(b+c-a)**2) + y/3 * SymSum(a*(b-c)**2*(b+c-d)**2) + y * SymSum(a*b*c*(a-b)**2)
        f3 = z/4 * SymSum(a*(b-c)**2)
        return (f1 + f2)/SymSum(a*b) + f3