from sympy import Poly, Symbol, Dummy, Rational, Add, re

from .utils import Coeff, congruence

def compute_sym2(poly: Poly) -> Poly:
    """Compute poly(1,1,d,d)"""
    # TODO: move it elsewhere
    margin = poly.eval((1,1))
    s = {}
    for (a, b), v in margin.rep.terms():
        if a + b in s:
            s[a + b] += v
        else:
            s[a + b] = v
    _s = {(k,): v for k, v in s.items()}
    return margin.from_dict(
        _s, margin.gens[-1], domain=margin.domain)

def _free_symbol(coeff: Coeff):
    gens = set(coeff.gens)
    if hasattr(coeff.domain, "symbols"):
        for symbol in coeff.domain.symbols:
            gens = gens.union(symbol.free_symbols)
    for s in "efghij":
        if not (Symbol(s) in gens):
            return Symbol(s)
    return Dummy("x")

def quaternary_quartic_symmetric(coeff, real=True):
    """
    ## Real Variables

    Vlad Timofte's theorem on quaternary symmetric quartics:
    If a quaternary symmetric quartic form satisfies
    `F(x,1,1,1) >= 0` and `F(x,x,1,1) >= 0` for all real numbers `x`,
    then it is nonnegative.

    However, not all nonnegative symmetric quartics are sum of squares.

    Theorem 1: If a nonnegative symmetric quartic form has
    `F(1,1,1,1) = 0`, then it is sum of squares.

    Theorem 2: A nonnegative symmetric quartic form multiplied
    by `SymmetricSum((a-b)**2)` is always sum of squares (at degree 6).

    ## Positive Variables

    Theorem 3: If `F` is a quaternary symmetric quartic form and
    `F(1,1,1,1) = 0`. Then `F` is nonnegative on R+ if and only
    if `F(x,1,1,1) >= 0` and `F(x,1,0,0)` holds for all `x >= 0`.

    Discussions of quaternary symmetric quartic forms can
    also be found in [1,2].

    References
    ----------
    [1] T. Ando. Some Cubic and Quartic Inequalities of Four Variables. 2022.

    [2] 陈胜利. 不等式的分拆降维幂方法与可读证明. 哈尔滨工业大学出版社, 2016.
    """
    sol = _quaternary_quartic_symmetric_real(coeff)
    if sol is not None:
        return sol
    return _quaternary_quartic_symmetric_positive_centered(coeff)

def _quaternary_quartic_symmetric_real(coeff):
    """
    Solve quaternary symmetric quartic forms by first
    trying symbolic SDP on the degree-4 SOS cone,
    and then try multiplying `Σ((a-b)**2)`
    """
    sol = _quaternary_quartic_symmetric_sdp(coeff)
    if sol is not None:
        return sol

    sol = _quaternary_quartic_symmetric_real_full(coeff)
    return sol

def _quaternary_quartic_symmetric_sdp(coeff):
    """
    Solve symmetric quartic forms representable by
    sum-of-squares.

    Theorem: If a nonnegative symmetric quartic form has
    `F(1,1,1,1) = 0`, then it is sum of squares.

    Sum-of-squares symmetric quartic forms are handled
    generally in `_sos_struct_nvars_quartic_symmetric_sdp`.

    Examples
    --------
    :: sym = "sym", ineqs = []

    => s(2a2-3ab)2

    => 1/5s(ab)2

    => s((a-b)2(a+b-4c-4d)2)

    => s(a2b2-a2bc)

    => s(a4-a3b-a2bc+2a2b2-abcd)

    => s(26a4+(236-244*sqrt(2))a3b+(373-198*sqrt(2))a2b2+(1444-1036*sqrt(2))a2bc+(297-202*sqrt(2))abcd)
    """
    from ..nvars.quartic import _sos_struct_nvars_quartic_symmetric_sdp
    return _sos_struct_nvars_quartic_symmetric_sdp(coeff)


def _compute_44_sym_real_full_mat(coeff: Coeff, m11, m33, m, r):
    """
    Consider a quaternary symmetric quartic form `F` with
    coefficients `c4, c31, c22, c211, c1111`. Suppose
    ```
    F * Σ((a-b)**2) = Σ(vec.T @ mat @ vec)
                    + m/6 * Σ((a-b)**2*(b-c)**2*(c-a)**2)
                    + r   * Σ((a-b)**2*(c-d)**2*(a+b+c+d)**2)
    ```
    where
    ```
    vec = [
        3*a**3 - 5*b**3 + c**3 + d**3,
        a*b*c + a*b*d - 5*a*c*d + 3*b*c*d,
        4*a**2*b + a**2*c + a**2*d - 4*a*b**2 - a*c**2 - a*d**2
            - 3*b**2*c - 3*b**2*d + 3*b*c**2 + 3*b*d**2,
        a**2*b - 2*a**2*c - 2*a**2*d + a*b**2 - 2*a*c**2 - 2*a*d**2 + 2*b**2*c
            + 2*b**2*d + 2*b*c**2 + 2*b*d**2 - c**2*d - c*d**2
    ]
    ```
    and `Σ` is the 24-term complete symmetric sum wrt. a, b, c, d.
    The `vec` is computed from the isotopic decomposition of the SDP bases
    given the symmetry group S4.

    This is a semidefinite programming and there are 4 degrees of
    freedom: `m11, m33, m, r`. This function returns `mat` determined
    by the parameters `m11, m33, m, r`.
    """
    c4, c31, c22, c211, c1111 = [coeff(_) for _ in
        [(4,0,0,0),(3,1,0,0),(2,2,0,0),(2,1,1,0),(1,1,1,1)]]

    # only the upper triangular part is displayed here
    mat = [
        [
            c4/18,
            -c211/36 - c22/36 - c31/18 - m/18 + m11/2 + m33,
            -c22/72 + c31/72 - c4/24 + m33/4 + r/18,
            -c22/36 - c31/18 - c4/36 + m33/2 + r/9
        ],
        [
            0,
            m11,
            -c1111/72 - c22/72 - m/72 + m11/4 + m33/2 - r/18,
            c211/36 - m/36 + r/9
        ],
        [
            0,
            0,
            c22/36 - c31/12 - c4/36 - m/72 + m33/2 + r/9,
            -c211/36 + c22/36 + c31/24 + c4/72 + m/72 - m11/4 - m33/4
        ],
        [0, 0, 0, m33]
    ]
    for i in range(1, 4):
        for j in range(i):
            mat[i][j] = mat[j][i]
    return coeff.as_matrix(mat, (4, 4))


def _quaternary_quartic_symmetric_real_full(coeff: Coeff):
    """
    Solve general nonnegative symmetric quartic forms
    over real numbers by multiplying `SymmetricSum((a-b)**2)`
    and representing it in the form:
    ```
        Σ PSD([
            3*a**3 - 5*b**3 + c**3 + d**3,
            a*b*c + a*b*d - 5*a*c*d + 3*b*c*d,
            4*a**2*b + a**2*c + a**2*d - 4*a*b**2 - a*c**2 - a*d**2
                - 3*b**2*c - 3*b**2*d + 3*b*c**2 + 3*b*d**2,
            a**2*b - 2*a**2*c - 2*a**2*d + a*b**2 - 2*a*c**2 - 2*a*d**2 + 2*b**2*c
                + 2*b**2*d + 2*b*c**2 + 2*b*d**2 - c**2*d - c*d**2
        ])
        + m/6 * Σ((a-b)**2*(b-c)**2*(c-a)**2)
        + r   * Σ((a-b)**2*(c-d)**2*(a+b+c+d)**2)
    ```

    The detailed algorithm is as follows. First, subtract as many
    `zv * Σ((a-b)**2*(c-d)**2)/16` as possible so that there is a
    double root on `F(v,v,1,1)`. Then there are three cases:

    1. Case A: v = 1 => F(1,1,1,1) = 0 so F must be sum-of-squares.

    2. Case B: v = -1 => F(-1,-1,1,1) = 0. Then there is only one
    degree of freedom after facial reducton.

    3. Case C: v != 1, -1 => F(v,v,1,1) = F(1/v,1/v,1,1) = 0. Then
    there is only one degree of freedom after facial reduction.

    Both Case B and Case C will be a 3 * 3 SDP with only 1 degree of
    freedom. It can be solved via exact arithmetic if feasible.

    Examples
    --------
    :: sym = "sym", ineqs = []

    => s(a(a-b)(a-c)(a-d))/6 + abcd    # Lax-Lax form when e = 0 [1]

    => s(7a4-48a3b+27a2b2+48a2bc-30abcd)

    => s(1/6a4-27/31a3b+101/248a2b2+3/4a2bc-329/744abcd)

    => s(15/2a4-54a3b+71/2a2b2+54a2bc-39abcd)

    => s(a4-15/2a3b+11/2a2b2+15/2a2bc-6abcd)

    => s(40/3a4-120a3b+389/4a2b2+267/2a2bc-1273/12abcd)

    => s(3a2b2+6a2bc-abcd)    # Choi-Lam form [2]

    => s(1/2a2b2+3/2a2bc+1/12abcd)

    => s(7/2a2b2+15/2a2bc-11/12abcd)

    => s(sqrt(2)/4a2b2+a2bc+(sqrt(2)/4-1/3)abcd)

    References
    ----------
    [1] A. Lax and P. D. Lax. On sums of squares. Linear Algebra Appl. 20 (1978), 71-75.

    [2] M. D. Choi and T. Y. Lam. Extremal positive semidefinite forms. Math. Ann.
    231 (1977), 1-18.
    """
    c4, c31, c22, c211, c1111 = [coeff(_) for _ in
        [(4,0,0,0), (3,1,0,0), (2,2,0,0), (2,1,1,0), (1,1,1,1)]]
    if c4 < 0:
        return None
    if c4 == 0:
        if c31 != 0:
            return None
        if c22 < 0:
            return None
        if c22 == 0:
            if c211 != 0:
                return None
            if c1111 == 0:
                # the polynomial is zero
                return Add()
            return None

    poly1111 = c1111 + 12*c211 + 6*c22 + 12*c31 + 4*c4
    if poly1111 <= 0:
        # If poly(1,1,1,1) == 0 while poly is nonnegative, it must
        # be a sum-of-squares polynomial and should not be handled here
        return None

    # subtract as many `zv * Σ((a-b)**2*(c-d)**2)/16` as possible
    # so that there is a double root on F(v,v,1,1)
    zvs = [
        (c1111*c22 + 2*c1111*c31 + 2*c1111*c4 - 4*c211**2 + 4*c211*c22 + 8*c211*c4 \
          + 2*c22**2 - 12*c31**2 - 16*c31*c4 - 8*c4**2)/poly1111,
        (c1111 - 4*c211 + 6*c22 - 4*c31 + 4*c4)/16
    ]
    zvs = [zv for zv in zvs if zv >= 0]
    if not zvs:
        return None
    zv = min(zvs)


    # sympy polynomial ring / rational function field
    ring = coeff.domain[_free_symbol(coeff)]
    x = ring.gens[0]

    rest_coeff = coeff - coeff.from_dict({(2,2,0,0): zv, (2,1,1,0): -zv, (1,1,1,1): 6*zv})
    rest_coeff = Coeff(rest_coeff.as_poly().set_domain(ring), field=False)


    denom = coeff.domain.zero
    if zv == (c1111 - 4*c211 + 6*c22 - 4*c31 + 4*c4)/16:
        # Case B.
        m = -c1111/16 + c211/4 + 5*c22/8 - 7*c31/4 + 7*c4/4
        m11 = -c1111/72 + c22/12 + c4/18
        r = -c1111/32 - c211/8 + c22/16 + 3*c31/8 + 3*c4/8
        if m < 0 or r < 0 or m11 < 0:
            return None
        m, m11, r = [ring.convert_from(_, coeff.domain) for _ in [m, m11, r]]
        embed = coeff.as_matrix([[-1, 2, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], (4, 3))

        mat = _compute_44_sym_real_full_mat(rest_coeff, m11, x, m, r)
    else:
        # Case C.
        denom = -c1111 - 4*c211 + 2*c22 + 12*c31 + 12*c4
        if denom == 0:
            # degenerated case
            m = c211/2 + c22/2 - 5*c31/2 + c4
            m11 = c211/18 + c22/18 - c31/6 - c4/9
            m, m11 = [ring.convert_from(_, coeff.domain) for _ in [m, m11]]
            embed = rest_coeff.as_matrix([[-1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], (4, 3))

            mat = _compute_44_sym_real_full_mat(rest_coeff, m11, x, m, ring.zero)
        else:
            m11const = -(c1111*c31 + 2*c1111*c4 - 2*c211**2 - 4*c211*c22 + 16*c211*c4 \
                    - 2*c22**2 - 6*c22*c31 + 4*c22*c4 - 6*c31**2 + 4*c31*c4)/(9*poly1111)
            m33const = (-c1111**2*c4 - 2*c1111*c211**2 - 4*c1111*c211*c22 - 4*c1111*c211*c31 \
                    - 24*c1111*c211*c4 - 2*c1111*c22**2 - 4*c1111*c22*c31 - 12*c1111*c22*c4 \
                    + 6*c1111*c31**2 - 8*c1111*c31*c4 - 16*c211**3 - 36*c211**2*c22 \
                    - 104*c211**2*c4 - 24*c211*c22**2 + 24*c211*c22*c31 - 64*c211*c22*c4 \
                    + 240*c211*c31**2 + 176*c211*c31*c4 + 96*c211*c4**2 - 4*c22**3 \
                    + 24*c22**2*c31 + 4*c22**2*c4 + 204*c22*c31**2 + 224*c22*c31*c4 \
                    + 96*c22*c4**2 + 288*c31**3 + 504*c31**2*c4 + 352*c31*c4**2 + 80*c4**3)\
                    /(18*denom*poly1111)
            m11 = x/18 + m11const
            m33 = x*(c211 + c22 + 3*c31 + 2*c4)/(9*denom) + m33const

            if -c1111 - 6*c211 + 6*c31 + 8*c4 == 0:
                embed = rest_coeff.as_matrix([[1, 0, 0], [0, 2, 0], [0, 1, 0], [0, 0, 1]], (4, 3))
            else:
                z = (c211 + c22 + 3*c31 + 2*c4)/(-c1111 - 6*c211 + 6*c31 + 8*c4)
                z = ring.convert_from(z, coeff.domain)
                embed = rest_coeff.as_matrix([[2*z, -4*z, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], (4, 3))

            mat = _compute_44_sym_real_full_mat(rest_coeff, m11, m33, x, ring.zero)

    # entries mat and mat2 are linear polynomials in `x`
    mat2 = (embed.T * mat * embed)._rep

    # do not call .det() because it raise exception for RR domain
    # det = mat2.det() is a cubic polynomial in `x`
    M = mat2.to_list()
    det = M[0][0]*M[1][1]*M[2][2] + M[0][1]*M[1][2]*M[2][0] + M[0][2]*M[1][0]*M[2][1] - \
        M[0][2]*M[1][1]*M[2][0] - M[0][1]*M[1][0]*M[2][2] - M[0][0]*M[1][2]*M[2][1]
    detlist = [det.get((i,), coeff.domain.zero) for i in [3,2,1,0]]

    def make_sol(x, divide = True):
        x = coeff.convert(x)
        if x < 0:
            return None

        if zv == (c1111 - 4*c211 + 6*c22 - 4*c31 + 4*c4)/16:
            m   = -c1111/16 + c211/4 + 5*c22/8 - 7*c31/4 + 7*c4/4
            m11 = -c1111/72 + c22/12 + c4/18
            m33 = x
            r   = -c1111/32 - c211/8 + c22/16 + 3*c31/8 + 3*c4/8
            if r < 0:
                return None
        else:
            r = coeff.convert(0)
            if denom == 0:
                m   = c211/2 + c22/2 - 5*c31/2 + c4
                m11 = c211/18 + c22/18 - c31/6 - c4/9
                m33 = x
            else:
                m   = x
                m11 = m/18 + m11const
                m33 = m*(c211 + c22 + 3*c31 + 2*c4)/(9*denom) + m33const

        # add the entries contributed by zv * Σ((a-b)**2*(c-d)**2)/16
        m   = m   + 3*zv
        m11 = m11 + 2*zv/9
        m33 = m33 + zv/18

        mat = _compute_44_sym_real_full_mat(coeff, m11, m33, m, r)
        cong = congruence(mat)
        if cong is None:
            return None
        U, S = cong

        a, b, c, d = coeff.gens
        SymmetricSum = coeff.symmetric_sum
        vec = [
            3*a**3 - 5*b**3 + c**3 + d**3,
            a*b*c + a*b*d - 5*a*c*d + 3*b*c*d,
            4*a**2*b + a**2*c + a**2*d - 4*a*b**2 - a*c**2 - a*d**2 \
                - 3*b**2*c - 3*b**2*d + 3*b*c**2 + 3*b*d**2,
            a**2*b - 2*a**2*c - 2*a**2*d + a*b**2 - 2*a*c**2 - 2*a*d**2 + 2*b**2*c \
                + 2*b**2*d + 2*b*c**2 + 2*b*d**2 - c**2*d - c*d**2
        ]

        sol = Add(
            *[s * SymmetricSum(Add(*[ul[i] * vec[i] for i in range(4)]).together()**2)
                for s, ul in zip(S, U.tolist())],
            m/6 * SymmetricSum((a-b)**2*(b-c)**2*(c-a)**2),
            r/36 * SymmetricSum(a)**2 * SymmetricSum((a-b)**2*(c-d)**2),
        )
        if divide:
            sol = sol / SymmetricSum((a-b)**2)
        return sol

    def get_candidates(p):
        a3, a2, a1, a0 = p
        if a3 == 0:
            if a2 == 0:
                return [-a0/a1] if a1 != 0 else []
            return [-a1/a2/2]
        if 3*a1*a3 - a2**2 == 0:
            return [-a2/a3/3]
        x1 = (9*a0*a3**2 - 4*a1*a2*a3 + a2**3)/(a3*(3*a1*a3 - a2**2))
        x2 = (-9*a0*a3 + a1*a2)/(2*(3*a1*a3 - a2**2))
        return [x1, x2]

    candidates = [coeff.domain.zero] + get_candidates(detlist)
    for x in candidates:
        final_sol = make_sol(x)
        if final_sol is not None:
            if (coeff.is_rational or coeff.domain.is_AlgebraicField)\
                and detlist[0]*x**3 + detlist[1]*x**2 + detlist[2]*x + detlist[3] != 0:
                # round the solution first to obtain neater solution
                if not coeff.is_rational:
                    x = re(coeff.to_sympy(x).n(15))
                x = Rational(x)
                for prec in [0, 1, 2, 4, 8]:
                    _x = x.limit_denominator(10**prec)
                    sol = make_sol(_x)
                    if sol is not None:
                        return sol

            return final_sol


def _quaternary_quartic_symmetric_positive_centered(coeff: Coeff):
    """
    Solve quarternary symmetric quartic forms on R+ with `F(1,1,1,1) = 0`.
    The algorithm first subtracts sufficiently many `Σ((a-b)**2*(c-d)**2)`,
    `Σ(c*d*(a-b)**2)`, `Σ(a*b*(a-b)**2)` until it reaches one of the
    following:

    1. `F_ab(t) = Σ(1/6a4+(-t/3-1/3)a3b+(t/3-1/6)a2b2+(t^2/6+1/2)a2bc+(-t^2/6-1/6)abcd)`

    2. `F_c(t) = Σ(1/6a4+(-t/3-1/3)a3b+(t^2/36+t/18+19/36)a2b2
    +(t^2/9+5*t/9-8/9)a2bc+(-5*t^2/36-5*t/18+19/36)abcd)`

    Here `Σ` stands for the 24-term symmetric sum.

    Both `F(t)` has `F(t,1,1,1) = 0`. But `F_ab(1,1,0,0) = 0`, `F_c(u,1,0,0) = 0`
    where `u` is the root of `3*u**2 - (t + 1)*u + 3`. See also in [1], Theorem 1.4.

    Examples
    --------
    :: sym = "sym"

    => a4+b4+c4+d4-(a2+b2)(c2+d2)-(ab-cd)2    # Turkevich's inequality

    => 3(a4+b4+c4+d4)+4abcd-(a+b+c+d)(a3+b3+c3+d3)    # Suranyi's inequality

    => s(1/6a4-4/3a3b+5/6a2b2+2a2bc-5/3abcd)    # t = 3

    => s(1/2a4-6a3b+9/2a2b2+14a2bc-13abcd)    # t = 5

    => s(3/2a4-30a3b+59/2a2b2+118a2bc-119abcd)     # t = 9

    => s(37/6a4-148/3a3b+257/6a2b2+50a2bc-595/12abcd)

    => s(1/6a4+(-sqrt(2)/3)a3b+(-1/2+sqrt(2)/3)a2b2+(1-sqrt(2)/3)a2bc+(-2/3+sqrt(2)/3)abcd)    # t = sqrt(2)-1

    Reference
    ----------
    [1] T. Ando. Some Cubic and Quartic Inequalities of Four Variables. 2022.
    """
    c4, c31, c22, c211, c1111 = [coeff(_) for _ in
        [(4,0,0,0),(3,1,0,0),(2,2,0,0),(2,1,1,0),(1,1,1,1)]]
    if c4 <= 0:
        # c4 == 0 is not yet implemented
        return None

    # subtract poly1111 * abcd to assume F(1,1,1,1) = 0
    poly1111 = c1111 + 12*c211 + 6*c22 + 12*c31 + 4*c4
    if poly1111 < 0:
        return None
    poly1100 = c22 + 2*c31 + 2*c4
    if poly1100 < 0:
        return None
    poly1110 = 3*c211 + 3*c22 + 6*c31 + 3*c4
    if poly1110 < 0:
        return None

    ################################################
    # Case 0. Subtract some Σ(a*b*(a-b)**2)/12
    # so that the rest >= 0 for all real numbers
    # This does not lift the degree so we try it first

    # Find w >= 0 such that
    # F(x,x,1,1) =
    # F1100*x**2 + (4*c211 + 2*c22 + 8*c31 + 4*c4 - 4*w/3)*x + F1100 >= 0
    # F(x,1,1,1) =
    # c4*x**2 + (3*c31 + 2*c4 - w)*x + F1110 >= 0

    candidates = [
        coeff.convert(0),
        3*(4*c211 + 2*c22 + 8*c31 + 4*c4 + 2*poly1100)/4,
        3*(4*c211 + 2*c22 + 8*c31 + 4*c4 - 2*poly1100)/4,
        3*c31 + 2*c4
    ]
    def _verify(w):
        return w >= 0 and candidates[2] <= w <= candidates[1]\
            and 4*c4*poly1110 >= (3*c31 + 2*c4 - w)**2
    for w in candidates:
        if _verify(w):
            rest = coeff - coeff.from_dict(
                {(3,1,0,0): w/3, (2,2,0,0): -2*w/3, (1,1,1,1): poly1111})
            sol = _quaternary_quartic_symmetric_sdp(rest)
            if sol is not None:
                SymmetricSum = coeff.symmetric_sum
                a, b, c, d = coeff.gens
                return Add(
                    sol,
                    w/12 * SymmetricSum(a*b*(a-b)**2),
                    poly1111/24 * SymmetricSum(a*b*c*d)
                )
            return None

    ################################################


    # subtract sq0 * Σ((a-b)**2*(c-d)**2)/16
    # to make F(x,1,0,0) has double root
    k = c31/c4
    sq0 = poly1100
    if k < -4:
        sq0 = poly1100 - (4 + 2*k + k**2/4)*c4
    if sq0 < 0:
        return None

    # subtract sq2 * Σ(c*d*(a-b)**2)/12
    # so that either:
    # 1. F(0,1,1,1) == 0
    # 2. F(x,1,1,1) has double root at some t > 0, t != 1

    # Note that F(x,1,1,1) = c4*x**2 + (3*c31 + 2*c4)*x + F1110
    sq2 = poly1110
    t = -(3*c31 + 2*c4)/(2*c4) # sym axis
    if t >= 0:
        sq2 = sq2 - c4*t**2
    else:
        t = 0
    if sq2 < 0:
        return None

    sq1 = 0
    if sq0 == poly1100 and sq2 == poly1110:
        # now the followings hold:
        # 1. F(x,1,0,0) has double root at x == 1
        # 2. F(x,1,1,1) = x*(x - 1)**2*(x + c)
        # subtract sq1 * Σ(a*b*(a-b)**2)/12
        # so that
        # F(x,1,1,1) == c4 * x**2*(x - 1)**2
        # after subtraction, still F(x,1,0,0) >= 0
        sq1 = 3*c31 + 2*c4
    # else:
    #     # no Σ(a*b*(a-b)**2) can be subtracted
    #     pass

    SymmetricSum = coeff.symmetric_sum
    a, b, c, d = coeff.gens
    def get_sol_t(t, c4 = 1):
        """Solve F_ab(t) or F_c(t) >= 0 via explicit sum-of-squares."""
        # if t == 1:
        #     pass
        if t <= 5:
            # solution of F_ab(t)
            expr = 4*a**2 + (2 - 2*t)*a*b + 4*b**2 \
                + (-t - 3)*a*c + (-t - 3)*a*d + (-t - 3)*b*c + (-t - 3)*b*d \
                + (t - 1)*c**2 + (4*t + 4)*c*d + (t - 1)*d**2
            return Add(
                c4/16 * SymmetricSum((a-b)**2*expr.together()**2),
                c4*4*(t - 1)**2/3 * SymmetricSum(a*b*(a-b)**2*(c-d)**2),
                c4*(5 - t)*(3*t + 1)/18 * SymmetricSum((a-b)**2*(b-c)**2*(c-a)**2)
            ) / SymmetricSum((a-b)**2)
        else:
            # solution of F_c(t)
            expr = 3*a**2 + 3*b**2 + 3*c**2 + 3*d**2 \
                + (-t - 1)*a*b + (-t - 1)*a*c + (-t - 1)*a*d + (-t - 1)*b*c + (-t - 1)*b*d \
                + (5*t - 7)*c*d
            return Add(
                c4/9 * SymmetricSum((a-b)**2*expr.together()**2),
                c4*4*(t - 1)**2/3 * SymmetricSum(a*b*(a-b)**2*(c-d)**2),
            ) / SymmetricSum((a-b)**2)
        return None

    sol = get_sol_t(t, c4)
    if sol is None:
        return None

    return Add(
        sol,
        sq0/16 * SymmetricSum((c-d)**2*(a-b)**2),
        sq1/12 * SymmetricSum(a*b*(a-b)**2),
        sq2/12 * SymmetricSum(c*d*(a-b)**2),
        poly1111/24 * SymmetricSum(a*b*c*d),
    )
