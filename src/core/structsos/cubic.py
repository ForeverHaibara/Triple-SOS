import sympy as sp

from .utils import CyclicSum, CyclicProduct, _sum_y_exprs

a, b, c = sp.symbols('a b c')

def sos_struct_cubic(poly, coeff, recurrsion):
    if coeff((2,1,0)) == coeff((1,2,0)):
        return _sos_struct_cubic_symmetric(coeff)
    if coeff((3,0,0)) == 0:
        return _sos_struct_cubic_degenerate(coeff)

    solution = _sos_struct_cubic_parabola(coeff)
    if solution is not None:
        return solution

    poly2 = poly * (a + b + c).as_poly(a,b,c)
    solution = recurrsion(poly2)
    if solution is not None:
        return solution / CyclicSum(a)

    return None


def _sos_struct_cubic_symmetric(coeff):
    """
    Cubic symmetric inequality can be handled with Schur.
    """
    # if not coeff((2,1,0)) == coeff((1,2,0)):
    #     return None
    y = [
        coeff((3,0,0)),
        coeff((3,0,0)) + coeff((2,1,0)),
        3 * (coeff((3,0,0)) + coeff((2,1,0)) * 2) + coeff((1,1,1))
    ]
    if all(_ >= 0 for _ in y):
        exprs = [
            CyclicSum(a*(a-b)*(a-c)),
            CyclicSum(a*(b-c)**2),
            CyclicProduct(a)
        ]
        return _sum_y_exprs(y, exprs)
    return None


def _sos_struct_cubic_degenerate(coeff):
    # if coeff((3,0,0)) != 0:
    #     return None

    p, q, r = coeff((2,1,0)), coeff((1,2,0)), coeff((1,1,1))
    rem = 3 * (p + q) + r
    if p < 0 or q < 0 or rem < 0:
        return None

    if p == q:
        return CyclicSum(a*(b-c)**2) * p + rem * CyclicProduct(a)

    return p * CyclicSum(a**2*b - CyclicProduct(a)) + q * CyclicSum(a**2*c - CyclicProduct(a))\
             + rem * CyclicProduct(a)


def _sos_struct_cubic_parabola(coeff):
    """
    Although we can always multiply s(a) to convert the problem to a quartic one,
    sometimes the cubic inequality does not need to higher the degree.

    Apart from Schur, one of the cases is s(a(a+tb-(t+1)c)2), where the coefficient of s(a2b) and s(a2c)
    are (t^2+4t+1, t^2-2t-2), respectively. This is a parabola (x-y-9)^2 = 36(y+3).
    We will test whether the inequality is a linear (convex) combination of two points on the parabola.
    """
    m, p, q, r = coeff((3,0,0)), coeff((2,1,0)), coeff((1,2,0)), coeff((1,1,1))
    rem = 3 * (p + q + m) + r
    if m < 0 or rem < 0:
        return None

    p, q = p / m, q / m

    if p == q + 3:
        # then the line connecting (p,q) and (1,-2) is parallel to the symmetric axis of the parabola
        if p >= 1:
            return m * CyclicSum(a*(a-c)**2) + m*(p-1) * CyclicSum(a*(b-c)**2) + rem * CyclicProduct(a)

    elif p + 2*q == -3:
        # this is the tangent of parabola at (1,-2)
        if p == 1:
            return m * CyclicSum(a*(a-c)**2) + rem * CyclicProduct(a)

    elif p == 1:
        if -2 <= q <= 22:
            w1 = (q - 22) / (-24) * m
            w2 = m - w1
            return w1 * CyclicSum(a*(a-c)**2) + w2 * CyclicSum(a*(a-4*b+3*c)**2) + rem * CyclicProduct(a)

    else:
        x2 = (13*p**2 + 22*p*q + 18*p + q**2 - 18*q - 27)/(p - q - 3)**2
        # y2 = (q + 2)/(p - 1)*(x2 - 1) - 2

        w1 = (p - x2) / (1 - x2) * m
        w2 = m - w1

        if not 0 <= w1 <= 1:
            return None

        t = 2*(p + 2*q + 3)/(p - q - 3)
        return w1 * CyclicSum(a*(a-c)**2) + w2 * CyclicSum(a*(a+t*b-(t+1)*c)**2) + rem * CyclicProduct(a)


    return None