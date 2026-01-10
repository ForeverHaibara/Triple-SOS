from sympy import Add, factorial
# from sympy.combinatorics.named_groups import SymmetricGroup

from .utils import Coeff, rationalize_func
from ...sdp import congruence
from ...utils import verify_symmetry

def sos_struct_nvars_quartic_symmetric(poly, real=True):
    """
    Solve a homogeneous quartic symmetric polynomial inequality on real numbers for nvars >= 4.
    """
    if poly.total_degree() == 4 and verify_symmetry(poly, "sym"):
        return _sos_struct_nvars_quartic_symmetric_sdp(Coeff(poly))


def _sos_struct_nvars_quartic_symmetric_sdp(coeff: Coeff):
    """
    If a symmetric quartic polynomial is SOS, then it can be written in the form of
    ```
        PSD([Σ(a**2), (Σ(a))**2]) + Σ((a-b)**2*(PSD([a+b, Σ(a)]))) + λ * Σ((a-b)**2*(c-d)**2)
    ```
    where `Σ` denotes the complete symmetric sum.

    This is an application of the isotypic decomposition of symmetry-adapted SDPs. Details of
    SOS symmetric quartics can be found, for example, in [1].

    The form corresponds to an SDP with 3 blocks: `2*2, 2*2, 1*1`. And there are 5 constraints
    for the entries to match the coefficients. The degree of freedom is `3 + 3 + 1 - 5 = 2`.
    The function finds `x, y` such that `mat1 >> 0, mat2 >> 0, λ >= 0`.

    ## Computational Details

    The seven bases of the SDP entries are:
    ```
    bases = [
        1/n**2 * CyclicSum(a**2)**2,
        1/n**3 * CyclicSum(a**2)*CyclicSum(a)**2,
        1/n**4 * CyclicSum(a)**4,
        (n-1)/(2*m*n**3)*SymmetricSum((a-b)**2)*CyclicSum(a)**2,
        (n-1)/(2*m*n**2)*SymmetricSum((a-b)**2*(a+b))*CyclicSum(a),
        (n-1)/(2*m*n)*SymmetricSum((a-b)**2*(a+b)**2),
        (n-3)*(n-2)/(4*m*n)*SymmetricSum((a-b)**2*(c-d)**2),
    ]
    ```
    where `m = n!` and `SymmetricSum` has `m` terms.

    The target is to find `m1, m2, m3, m4, m5, m6, m7` such that
    ```
    sol = m1*bases[0] + 2*m2*bases[1] + m3*bases[2]
        + m4*bases[3] + 2*m5*bases[4] + m6*bases[5]
        + m7*bases[6]
    ```
    and `[[m1,m2],[m2,m3]] >> 0`, `[[m4,m5],[m5,m6]] >> 0`, `m7 >= 0`.
    We let `m4 = x` and `m6 = y` to be the free variables.

    The coefficients of the above bases in the basis of normalized newton sums
    (p4, p31, p22, p211, p1111) are given by:
    ```
    [
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, -1],
        [0, 1, 0, -1, 0],
        [1, 0, -1, 0, 0],
        [-1, 4, (n**2 - 3*n + 3)/(n - 1), -2*n**2/(n - 1), n**2/(n - 1)],
    ]
    ```

    Reference
    ----------
    [1] Grigoriy Blekherman and Cordian Riener. 2012. Symmetric non-negative forms and
    sums of squares. Discrete & Computational Geometry 65 (2012): 764-799.
    """
    n = len(coeff.gens)
    if n < 4:
        return None
    m = int(factorial(n))

    # convert the polynomial to a linear combination of
    # normalized newton sums: p4, p31, p22, p211, p1111
    # and compute the coefficients
    c4, c31, c22, c211, c1111 = [coeff(_ + (0,)*(n-len(_)))
        for _ in [(4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1)]]
    m, n = coeff.convert(m), coeff.convert(n)
    cvec = coeff.as_matrix([[c4, c31, c22, c211, c1111]], (1, 5))
    mat = coeff.as_matrix([
        # these are coefficients of symmetric monomial functions
        # represented in the basis of normalized newton sums
        [n, 0, 0, 0, 0],
        [-n, n**2, 0, 0, 0],
        [-n/2, 0, n**2/2, 0, 0],
        [n, -n**2, -n**2/2, n**3/2, 0],
        [-n/4, n**2/3, n**2/8, -n**3/4, n**4/24]
    ], (5, 5))
    pvec = cvec * mat
    p4, p31, p22, p211, p1111 = [coeff.wrap(_) for _ in pvec._rep.to_list_flat()]


    # for computing the entries of the SDP, see `compute_quad`
    domain = coeff.domain
    one = domain.one
    coeffs = [
        [0, -one/2, 1, 1, 0, 0, 0],
        [-(n-2)**2/(n-1), (n**2-2*n+2)/(n-1), -n**2/(n-1), 0, -2, 1, 1],
        [
            (n**2*p4 + n*p22 - 3*n*p4 - p22 + 3*p4)/(n - 1),
            -(2*n**2*p4 - n*p211 - n*p31 - 4*n*p4 + p211 + p31 + 4*p4)/(2*(n - 1)),
            (n**2*p4 + n*p1111 - p1111)/(n - 1),
            0,
            (p31 + 4*p4)/2,
            0,
            -p4
        ]
    ]

    def compute_quad(x, y):
        """
        Require `x, y` so that `mat1 >> 0, mat2 >> 0, m7 >= 0`.

        * The feasible set of det(mat1) >= 0 is in the form of
        `-(x-4*y)**2 + Bx + Cy + D >= 0` (parabola).
        * The feasible set of det(mat2) >= 0 is in the form of
        `xy >= (2*y + E)**2` where `y >= 0` (half hyperbola).
        * The feasible set of m7 >= 0 is in the form of
        `y >= p4` (half plane).
        """
        entries = []
        for cx, cy, c0 in zip(*coeffs):
            # all entries computable by x and y
            entries.append(cx * x + cy * y + c0)
        m1, m2, m3, m4, m5, m6, m7 = entries
        mat1 = coeff.as_matrix([[m1, m2], [m2, m3]], (2, 2))
        mat2 = coeff.as_matrix([[m4, m5], [m5, m6]], (2, 2))
        return mat1, mat2, m7

    def valid_sol(x, y):
        x, y = coeff.convert(x), coeff.convert(y)
        mat1, mat2, m7 = compute_quad(x, y)
        if m7 < 0:
            return None
        cong1 = congruence(mat1)
        cong2 = congruence(mat2)
        if cong1 is None or cong2 is None:
            return None
        return cong1, cong2, m7

    def make_sol(x, y):
        valid = valid_sol(x, y)
        if valid is None:
            return None
        cong1, cong2, m7 = valid

        # pg = SymmetricGroup(len(coeff.gens))
        # SymmetricSum = lambda expr: CyclicSum(expr, coeff.gens, pg, evaluate=False)
        SymmetricSum = coeff.symmetric_sum
        a, b, c, d = coeff.gens[:4]

        v = Add(*coeff.gens)
        sol = [
            # s * (u[0]/m*SymmetricSum(a**2) + u[1]/m**2*SymmetricSum(a)**2).together()**2
            #     for u, s in zip(cong1[0].tolist(), cong1[1])
            s/m**2 * (SymmetricSum((u[0] + u[1]/n)*a**2 + u[1]/n*(n-1)*a*b)).together()**2
                for u, s in zip(cong1[0].tolist(), cong1[1])
        ] + [
            (n-1)/(2*m*n)*s * SymmetricSum((a-b)**2*(u[1]*(a+b) + u[0]/n*v).together()**2)
                for u, s in zip(cong2[0].tolist(), cong2[1])
        ] + [
            (n-3)*(n-2)/(4*m*n)*m7 * SymmetricSum((a-b)**2*(c-d)**2)
        ]
        return Add(*sol)

    # Find x, y such that valid_sol(x, y) is not None
    # naive try:
    sol = make_sol(0, 0)
    if sol is not None:
        return sol

    if p4 >= 0:
        # try y == p4 such that m7 == 0

        # det(M2) == 0 and m7 == 0
        if p4 != 0:
            sol = make_sol(p31**2/4/p4, p4)
            if sol is not None:
                return sol

        # select the symmeric axis of det(M1)
        sol = make_sol(p211 + 2*p22 + p31 + 2*p4, p4)
        if sol is not None:
            return sol

    # let x == (-4*y + p31 + 4*p4)**2/(4*y) and det(M1) >= 0
    # which is a cubic polynomial >= 0

    a = coeff.gens[0]

    # this is the (numerator of the) determinant of `mat1`
    # when `x == (-4*y + p31 + 4*p4)**2/(4*y)`
    det = coeff.from_list([
        -64*(n - 2)**2*(p1111 + p211 + p22 + p31 + p4),
        16*(4*n**2*p1111*p4 + 4*n**2*p211*p4 + 4*n**2*p22*p4 + 4*n**2*p31*p4 + 4*n**2*p4**2 \
            + 4*n*p1111*p22 - 12*n*p1111*p4 - n*p211**2 - 6*n*p211*p31 - 24*n*p211*p4 \
            - 8*n*p22*p31 - 32*n*p22*p4 - 9*n*p31**2 - 48*n*p31*p4 - 48*n*p4**2 - 4*p1111*p22 \
            + 12*p1111*p4 + p211**2 + 6*p211*p31 + 24*p211*p4 + 8*p22*p31 + 32*p22*p4 \
            + 9*p31**2 + 48*p31*p4 + 48*p4**2),
        8*(n - 1)*(p31 + 4*p4)**2*(p211 + 2*p22 + 3*p31 + 6*p4),
        -(n - 1)*(p31 + 4*p4)**4
    ], (a,)).as_poly()

    detdiff = det.diff()
    detgcd = det.gcd(detdiff)
    if detgcd.total_degree() == 1:
        y = detgcd.rep.TC() / detgcd.rep.LC() * -1
        x = (-4*y + p31 + 4*p4)**2/(4*y)
        sol = make_sol(x, y)
        if sol is not None:
            return sol

    def validate_init(y):
        if y <= 0 or y <= p4:
            return False
        x = (-4*y + p31 + 4*p4)**2/(4*y)
        x, y = coeff.convert(x), coeff.convert(y)
        mat1, mat2, m7 = compute_quad(x, y)
        rep = mat1._rep.rep.to_list()
        return coeff.convert(rep[0][0]) >= 0 and coeff.convert(rep[1][1]) >= 0

    y = rationalize_func(det,
        lambda y: y > 0 and (valid_sol((-4*y + p31 + 4*p4)**2/(4*y), y) is not None),
        validation_initial = validate_init,
        direction = 1
    )
    # TODO: apply the cubic root isolation here
    if y is None:
        return None
    x = (-4*y + p31 + 4*p4)**2/(4*y)
    sol = make_sol(x, y)
    if sol is not None:
        return sol
