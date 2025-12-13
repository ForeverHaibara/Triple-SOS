from sympy import Add
from sympy import MutableDenseMatrix as Matrix

from .utils import (
    Coeff, CommonExpr, congruence, sum_y_exprs
)
from ..utils import congruence_solve

def sos_struct_quadratic(coeff: Coeff, real = True):
    """
    Solve cyclic quadratic problems.
    It must be in the form CyclicSum(a**2 + x*a*b) where x >= -1.

    The function does not use the `real` argument.

    More detailedly, we shall also handle cases for real numbers.
    When -1 <= x <= 2, it is a linear combination of s(a2-ab) and s(a)2,
    and the inequality holds for all real numbers.
    When x > 2, it only holds for positive real numbers.
    """
    return CommonExpr.quadratic(coeff((2,0,0)), coeff((1,1,0)), coeff.gens)


def sos_struct_acyclic_quadratic(coeff: Coeff, real = True):
    """
    Solve quadratic acyclic 3-var polynomial inequalities.

    The function does not use the `real` argument.

    If the inequality is positive over a,b,c in R. Then it can is a semidefinite
    positive quadratic form. We can decompose it into a sum of squares.

    If the quadratic form is positive over a,b,c in R+, then it is called a
    copositive matrix. It can be shown that for nvars <= 4, a copositive
    matrix is the sum of a semidefinite positive matrix and an elementwise nonnegative matrix.
    See [1] for more details.

    We can also prove the property of the copositve matrix for nvars = 3.
    Denote the matrix as M where M =
    [[c00, c01/2, c02/2],
     [c01/2, c11, c12/2],
     [c02/2, c12/2, c22]].
    A necessary condition for M to be copositive is the diagonal elements are nonnegative.
    Next, we require c01/2 >= -sqrt(c00*c11), c02/2 >= -sqrt(c00*c22), c12/2 >= -sqrt(c11*c22).
    If all (c01, c02, c12) are positive, then it is trivial.
    If two of them are positive, but one is negative, say c01 < 0, then we set c02, c12 = 0 and
    M becomes PSD.
    If two of them are negative, but one is positive, say c02 > 0, then we set c02/2 = (c01/2)*(c12/2)/c11
    and M becomes PSD (this c02 is the symmetric axis of the determinant with respect to c02).

    Examples
    ---------
    a2+b2+2c2-2ab+(sqrt(3)a-(b-c))2

    (2b-3c+a)2+(2c-3a+b)2+7/3(a2-b2+c2)

    a2+2b2+5c2+2ac+5ab-6bc

    ab+(sqrt(2)a-b-c)2

    References
    -----------
    [1] P. H. Diananda, On non-negative forms in real variables some or all of which are non-negative
    """
    gens = coeff.gens
    c00, c01, c02, c11, c12, c22 = [coeff(_) for _ in ((2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2))]
    if c00 < 0 or c11 < 0 or c22 < 0:
        return None

    def quad_form(c01, c02, c12):
        return coeff.as_matrix([
            [c00, c01/2, c02/2],
            [c01/2, c11, c12/2],
            [c02/2, c12/2, c22],
        ], (3,3))

    def solve_psd(M):
        return congruence_solve(M, gens)

    def solve_nonnegative(M):
        if all(_ >= 0 for _ in M):
            return Add(*[
                M[i,j] * gens[i]*gens[j] for i in range(3) for j in range(3)])

    M = quad_form(c01, c02, c12)
    solution = solve_psd(M)
    if solution is not None:
        return solution

    if c01 >= 0 and c02 >= 0 and c12 >= 0:
        return solve_nonnegative(M)

    if len(list(filter(lambda x: x >= 0, (c01, c02, c12)))) == 2:
        # 1 negative, 2 positive
        c01, c02, c12 = min(c01, 0), min(c02, 0), min(c12, 0)
        psd = quad_form(c01, c02, c12)
        solution = solve_psd(psd)
        if solution is not None:
            solution = solution + solve_nonnegative(M - psd)
        return solution

    if c00 == 0 or c11 == 0 or c22 == 0:
        # In this case, at least two of c01, c02, c12 are nonnegative,
        # which is contradictory to above.
        return None

    # final case: 2 negative, 1 positive
    psd = M.copy()
    for i in range(3):
        if psd[i, (i+1)%3] >= 0:
            psd[i, (i+1)%3] = (psd[(i+1)%3, (i+2)%3] * psd[(i+2)%3, i%3] / psd[(i+2)%3, (i+2)%3])
            psd[(i+1)%3, i] = psd[i, (i+1)%3]
            solution2 = solve_nonnegative(M - psd)
            solution1 = solve_psd(psd)
            if solution1 is not None and solution2 is not None:
                return solution1 + solution2
