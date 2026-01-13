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

def quaternary_quartic_symmetric(coeff, real=True):
    """
    Vlad Timofte's theorem on quaternary symmetric quartics:
    If a quaternary symmetric quartic polynomial satisfies
    `F(x,1,1,1) >= 0` and `F(x,x,1,1) >= 0` for all real numbers `x`,
    then it is nonnegative.

    However, not all nonnegative symmetric quartics are sum of squares.

    Theorem 1: If a nonnegative symmetric quartic polynomial has
    `F(1,1,1,1) = 0`, then it is sum of squares.

    Theorem 2: A nonnegative symmetric quartic polynomial multiplied
    by `SymmetricSum((a-b)**2)` is always sum of squares (at degree 6).

    Discussions of quaternary symmetric quartic polynomials can also be found in [1,2].

    References
    ----------
    [1] T. Ando. Some Cubic and Quartic Inequalities of Four Variables. 2022.

    [2] 陈胜利. 不等式的分拆降维幂方法与可读证明. 哈尔滨工业大学出版社, 2016.
    """
    return _quaternary_quartic_symmetric_real(coeff)

def _quaternary_quartic_symmetric_real(coeff):
    sol = _quaternary_quartic_symmetric_sdp(coeff)
    if sol is not None:
        return sol

    sol = _quaternary_quartic_symmetric_real_full(coeff)
    return sol

def _quaternary_quartic_symmetric_sdp(coeff):
    """
    Solve symmetric quartic polynomials representable by
    sum-of-squares.

    Theorem: If a nonnegative symmetric quartic polynomial has
    `F(1,1,1,1) = 0`, then it is sum of squares.

    Sum of squares symmetric quartic polynomials are handled
    generally in `_sos_struct_nvars_quartic_symmetric_sdp`.

    Examples
    --------
    :: sym = "sym"

    => s(2a2-3ab)2

    => 1/5s(ab)2

    => s((a-b)2(a+b-4c-4d)2)

    => s(a2b2-a2bc)

    => s(a4-a3b-a2bc+2a2b2-abcd)

    => s(26a4+(236-244*sqrt(2))a3b+(373-198*sqrt(2))a2b2+(1444-1036*sqrt(2))a2bc+(297-202*sqrt(2))abcd)
    """
    from ..nvars import _sos_struct_nvars_quartic_symmetric_sdp
    return _sos_struct_nvars_quartic_symmetric_sdp(coeff)


def _compute_44_sym_real_full_mat(coeff: Coeff, m11, m33, m):
    """
    Consider a quaternary symmetric quartic polynomial `F` with
    coefficients `c4, c31, c22, c211, c1111`. Suppose
    ```
    F * Σ((a-b)**2) = Σ(vec.T @ mat @ vec) + m/6 * Σ((a-b)**2*(b-c)**2*(c-a)**2)
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

    This is a semidefinite programming and there are 3 degrees of
    freedom: `m11, m33, m`. This function returns `mat` determined
    by the parameters `m11, m33, m`.
    """
    c4, c31, c22, c211, c1111 = [coeff(_) for _ in
        [(4,0,0,0),(3,1,0,0),(2,2,0,0),(2,1,1,0),(1,1,1,1)]]

    # only the upper triangular part is displayed here
    mat = [
        [
            c4/18,
            -c211/36 - c22/36 - c31/18 - m/18 + m11/2 + m33,
            -c22/72 + c31/72 - c4/24 + m33/4,
            -c22/36 - c31/18 - c4/36 + m33/2
        ],
        [
            0,
            m11,
            -c1111/72 - c22/72 - m/72 + m11/4 + m33/2,
            c211/36 - m/36
        ],
        [
            0,
            0,
            c22/36 - c31/12 - c4/36 - m/72 + m33/2,
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
    Solve general nonnegative symmetric quartic polynomials
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
        ]) + m/6 * Σ((a-b)**2*(b-c)**2*(c-a)**2)
    ```

    TODO: handle when (-1,-1,1,1) is double but not quartic root

    Examples
    --------
    :: sym = "sym"

    => s(a(a-b)(a-c)(a-d))/6 + abcd    # Lax-Lax form when e = 0 [1]

    => s(7a4-48a3b+27a2b2+48a2bc-30abcd)

    => s(1/6a4-27/31a3b+101/248a2b2+3/4a2bc-329/744abcd)

    => s(15/2a4-54a3b+71/2a2b2+54a2bc-39abcd)

    => s(a4-15/2a3b+11/2a2b2+15/2a2bc-6abcd)    # doctest:+SKIP

    => s(40/3a4-120a3b+389/4a2b2+267/2a2bc-1273/12abcd)    # doctest:+SKIP

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
        # be sum-of-squares polynomial and should not be handled here
        return None

    # subtract as many `zv * Σ((a-b)**2*(c-d)**2)/16` as possible
    # so that there is a double root on F(v,v,1,1)
    zvs = [
        (c1111*c22 + 2*c1111*c31 + 2*c1111*c4 - 4*c211**2 + 4*c211*c22 + 8*c211*c4 \
          + 2*c22**2 - 12*c31**2 - 16*c31*c4 - 8*c4**2)/poly1111,
        -(-c1111 + 4*c211 - 6*c22 + 4*c31 - 4*c4)/16
    ]
    zvs = [zv for zv in zvs if zv >= 0]
    if not zvs:
        return None
    zv = min(zvs)


    # sympy polynomial ring / rational function field
    def _free_symbol():
        gens = set(coeff.gens)
        if hasattr(coeff.domain, "symbols"):
            for symbol in coeff.domain.symbols:
                gens = gens.union(symbol.free_symbols)
        for s in "efghij":
            if not (Symbol(s) in gens):
                return Symbol(s)
        return Dummy("x")
    ring = coeff.domain[_free_symbol()]
    x = ring.gens[0]

    rest_coeff = coeff - coeff.from_dict({(2,2,0,0): zv, (2,1,1,0): -zv, (1,1,1,1): 6*zv})
    rest_coeff = Coeff(rest_coeff.as_poly().set_domain(ring), field=False)


    denom = -c1111 - 4*c211 + 2*c22 + 12*c31 + 12*c4
    if denom == 0:
        # degenerated case
        m = c211/2 + c22/2 - 5*c31/2 + c4
        m11 = c211/18 + c22/18 - c31/6 - c4/9
        m, m11 = [ring.convert_from(_, coeff.domain) for _ in [m, m11]]
        embed = rest_coeff.as_matrix([[-1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], (4, 3))

        mat = _compute_44_sym_real_full_mat(rest_coeff, m11, x, m)
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

        mat = _compute_44_sym_real_full_mat(rest_coeff, m11, m33, x)

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

        if denom == 0:
            m   = c211/2 + c22/2 - 5*c31/2 + c4
            m11 = c211/18 + c22/18 - c31/6 - c4/9
            m33 = x
        else:
            m   = x
            m11 = m/18 + m11const
            m33 = m*(c211 + c22 + 3*c31 + 2*c4)/(9*denom) + m33const

        m   = m   + 3*zv
        m11 = m11 + 2*zv/9
        m33 = m33 + zv/18

        mat = _compute_44_sym_real_full_mat(coeff, m11, m33, m)
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
            m/6 * SymmetricSum((a-b)**2*(b-c)**2*(c-a)**2)
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
