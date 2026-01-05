from .utils import Coeff

from sympy import Add

def quaternary_quintic_symmetric(coeff):
    """
    Solve quaternary symmetric quintic polynomials. Symmetry is not checked here.

    References
    -----------
    [1] 刘保乾. 一类四元五次对称不等式分拆探讨. 北京联合大学学报(自然科学版), 2005.
    """
    if coeff((5,0,0,0)) != 0:
        # not implemented
        return
    return _quaternary_quintic_symmetric_hexagon(coeff)


def _quaternary_quintic_symmetric_hexagon(coeff: Coeff):
    """
    Suppose a quaternary symmetric quintic `F` has zero coefficient at `a^5`,
    and that `F(1,1,1,1)=0`. Then it is nonnegative if and only if
    `F(x,1,1,1) >= 0` holds for all `x>=0`.

    This condition is sufficient for `F(x,x,1,1) >= 0` and `F(x,1,1,0) >= 0`
    to hold. And thus Vlad Timofte's theorem applies.

    Examples
    --------
    :: sym = "sym"

    => s(3/2a4b-3/2a3b2-3a3bc+7/2a2b2c-1/2a2bcd)

    => s(3/2a4b-a3b2-9/2a3bc+5a2b2c-a2bcd)

    => s(1/2a4b-1/2a3b2-3/2a3bc+2a2b2c-1/2a2bcd)

    => 4s(1/2a4b-1/2a3b2-1/2a3bc+1/2a2b2c)

    => s(1/2a3bc-1/2a2b2c)

    => s(1/2a2b2c-1/2a2bcd)

    => s(3/2a2b2c-3/2a2bcd + 4/7(a3bc-a2b2c) + 1/7a2bcd)
    """
    c5, c41, c32, c311, c221 = [coeff(_) for _ in
        [(5,0,0,0), (4,1,0,0), (3,2,0,0), (3,1,1,0), (2,2,1,0)]]

    if c5 != 0 or c41 < 0:
        return

    rem = coeff.poly111()
    if rem < 0:
        return

    c41w = c32 + c41
    c41r = c311 + c32 + 2*c41
    c41z = c221 + c311 + 2*c32 + 2*c41

    SymmetricSum = coeff.symmetric_sum
    a, b, c, d = coeff.gens

    if c41w < 0 or c41z < 0:
        return

    if c41r >= 0:
        # linear combinations of bases
        return Add(
            c41/4 * SymmetricSum(a*(b - c)**2*(b + c - a)**2),
            c41w/96 * SymmetricSum(a)*SymmetricSum((a - b)**2*(c - d)**2),
            c41r/4 * SymmetricSum(a*b*c*(a - b)**2),
            c41z/8 * SymmetricSum(a*b*(a + b)*(c - d)**2),
            rem/24 * a*b*c*d*SymmetricSum(a)
        )

    if c41 == 0:
        return

    # normalize so that c41 = 1
    r = c41r/c41
    z = c41z/c41

    if z*4 - r**2 >= 0:
        return Add(
            c41/16 * SymmetricSum(a*(c - d)**2*(2*a - r*b - 2*c - 2*d).together()**2),
            c41w/96 * SymmetricSum(a)*SymmetricSum((a - b)**2*(c - d)**2),
            c41*(z*4 - r**2)/32 * SymmetricSum(a*b*(a + b)*(c - d)**2),
            rem/24 * a*b*c*d*SymmetricSum(a)
        )
