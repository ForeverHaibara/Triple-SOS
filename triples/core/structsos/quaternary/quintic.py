from .utils import Coeff

from sympy import Add

from .utils import congruence, sum_y_exprs

def quaternary_quintic_symmetric(coeff):
    """
    Solve quaternary symmetric quintic polynomials. Symmetry is not checked here.


    Examples
    --------
    :: sym = "sym"

    => s(1/3a5+(-2*sqrt(3)/3)a4b+(-1+2*sqrt(3)/3)a3b2+4/3a3bc-1/3a2b2c-1/3a2bcd)

    Below 13 inequalities are from [1]. They are the bases for
    the inequalities that are trivially nonnegative after the
    difference substitution.

    => s(3/2a5-11/2a4b+a3b2+6a3bc-a2b2c-2a2bcd) # doctest:+SKIP

    => s(2a5-10a4b+9/2a3b2+31/2a3bc-21/2a2b2c-3/2a2bcd) # doctest:+SKIP

    => s(1/2a5-a4b-1/2a3b2+a3bc+1/2a2b2c-1/2a2bcd)

    => s(2a5-8a4b+5/2a3b2+19/2a3bc-9/2a2b2c-3/2a2bcd) # doctest:+SKIP

    => s(3/2a5-15/2a4b+3a3b2+12a3bc-7a2b2c-2a2bcd) # doctest:+SKIP

    => s(1/2a3b2-1/2a3bc-1/2a2b2c+1/2a2bcd)

    => s(1/2a2b2c-1/2a2bcd)

    => s(1/2a5-5/2a4b+a3b2+9/2a3bc-3a2b2c-1/2a2bcd)

    => s(3/2a5-5a4b+1/2a3b2+6a3bc-3/2a2b2c-3/2a2bcd)

    => s(3/2a4b-3/2a3b2-3a3bc+7/2a2b2c-1/2a2bcd)

    => s(1/2a3bc-1/2a2b2c)

    => s(1/2a4b-1/2a3b2-3/2a3bc+2a2b2c-1/2a2bcd)

    => s(1/2a4b-1/2a3b2-1/2a3bc+1/2a2b2c)

    References
    -----------
    [1] 刘保乾. 一类四元五次对称不等式分拆探讨. 北京联合大学学报(自然科学版), 2005.
    """
    if coeff((5,0,0,0)) == 0:
        return _quaternary_quintic_symmetric_hexagon(coeff)

    t = coeff((4,1,0,0)) / coeff((5,0,0,0))

    if t > 0:
        # not implemented
        return
    if 4 - 3*t**2 >= 0:
        return _quaternary_quintic_symmetric_c3axis(coeff)
    if t >= -3:
        return _quaternary_quintic_symmetric_surface(coeff)


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

    => s(3/2a4b-a3b2-9/2a3bc+5a2b2c-a2bcd)

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


def _quaternary_quintic_symmetric_c3axis_t(t, coeff: Coeff, c5 = 1):
    """
    Given `-2/sqrt(3) <= t <= `, solve the symmetric quintic inequality:

        `s(1/6a5+(t/2)a4b+(-t/2-1/2)a3b2+(3*t^2/8+1/6)a3bc+(1/3-3*t^2/8)a2b2c-1/6a2bcd)`

    It has roots at `(-(3*t+2)/2),1,1,1), (1,1,0,0), (1,1,1,0), (1,1,1,1)`.

    The solution requires solving a 3-var feasible SDP:
    `n00, n01, n11 >= 0` such that `M >> 0`, `N >> 0`, `m1 >= 0`
    where `N = [[n00, n01], [n01, n11]]` and `M`, `m1` are defined in the code.

    In fact, it turns out that we can always set `n01 = (t+3)*n00`, `n11 = (t+3)**2*n00`.
    And taking  `n00 =((159+25*sqrt(3))/1128)`  is feasible for
    `-2/sqrt(3) <= t <= 0.886078266959562`.
    """
    coeffs = [
        [1, (2*t - 3)/8],
        [-(5*t + 16)/4, -(9*t**2 + 78*t - 16)/192],
        [(3*t + 4)/2, (t - 2)*(3*t - 4)/32],
        [-(t + 4)*(7*t + 4)/8, -(27*t**3 + 90*t**2 - 48*t + 80)/384],
        [(t + 4)*(7*t + 8)/4, (27*t**3 + 162*t**2 + 264*t + 176)/192]
    ]

    domain = coeff.domain

    def compute_mats_m1(n00):
        entries = []
        for cn, c0 in coeffs:
            entries.append(cn * n00 + c0)
        m01, m02, m11, m12, m22 = entries
        m00 = domain.one/2
        mat1 = coeff.as_matrix([
            [m00, m01, m02],
            [m01, m11, m12],
            [m02, m12, m22]
        ], (3, 3))

        n01 = n00*(t+3)
        n11 = n00*(t+3)**2
        mat2 = coeff.as_matrix([
            [n00, n01],
            [n01, n11]
        ], (2, 2))
        m1 = -16*n00 + 16*n01 - 4*n11 + 3*t**2/2 + 4*t + domain.one*8/3

        return mat1, mat2, m1

    def make_sol(n00):
        if coeff.wrap(n00) < 0:
            return None
        mat1, mat2, m1 = compute_mats_m1(n00)
        if m1 < 0:
            return None
        cong = congruence(mat1)
        if cong is None:
            return None
        # we do not need to decompose mat2 since it is
        # N = [[n00, n00*(t+3)], [n00*(t+3), n00*(t+3)**2]]

        U, S = cong
        a, b, c, d = coeff.gens
        SymmetricSum = coeff.symmetric_sum
        v = [
            a**2 + 3*t/2*a*b + b**2 + c*d*(-3*t/2 - 2),
            a*c + a*d + b*c + b*d - 2*a*b - 2*c*d,
            c**2 - 2*c*d + d**2
        ]
        def make_quad(s, u):
            if u[0] == 0 and u[1] == 0:
                return s*c5*2*u[2]**2 * SymmetricSum(a*b*(a-b)**2*(c-d)**4)
            return s*c5*2 * SymmetricSum(
                a*b*(a-b)**2*Add(*[u[i]*v[i] for i in range(3)]).together()**2)

        quad = [make_quad(s, u) for s, u in zip(S, U.tolist())]
        p0 = a**2 + b**2 + c**2 + d**2 - 2*a*c - 2*a*d - 2*b*c - 2*b*d + (t + 3)*a*b + (t + 3)*c*d
        rest = [
            m1*c5*2 * SymmetricSum(a*b*c*d*(a-b)**2*(c-d)**2),
            n00*c5*2 * SymmetricSum((a-b)**2*(c-d)**2*p0.together()**2)
        ]
        mul = SymmetricSum(a*(b-c)**2)
        return Add(*quad, *rest) / mul

    candidates = [
        # feasible when t >= -1.15008858649677
        domain.one/6,

        # feasible when t >= -1.15470053816744
        domain.one*33/184,

        # feasible when t >= -1.1547005383792489...
        domain.one*356/1985,

        # always feasible when t >= -2/sqrt(3) = -1.1547005383792517
        (13608*t**9 + 286011*t**8 + 1864296*t**7 + 6278040*t**6 + 22278696*t**5 \
         + 71285280*t**4 + 122150400*t**3 + 95672576*t**2 + 40778752*t + 19759104)\
            /(48*(2268*t**8 - 43065*t**7 - 524538*t**6 - 1528404*t**5 + 685904*t**4 \
                  + 12251840*t**3 + 30092672*t**2 + 36940800*t + 18597888))
    ]

    for n00 in candidates:
        sol = make_sol(n00)
        if sol is not None:
            return sol


def _quaternary_quintic_symmetric_surface_t(t, coeff: Coeff, c5 = 1):
    """
    Given `-3 <= t <= -2/sqrt(3)`, solve the symmetric quintic inequality:

        `s(1/6a5+(t/2)a4b+(-t/2-1/2)a3b2+(t^2/2)a3bc+(1/2-t^2/2)a2b2c-1/6a2bcd)`

    It has roots at `(-t-1,1,1,0), (1,1,0,0), (1,1,1,0), (1,1,1,1)`.

    The solution requires solving a 2-var feasible SDP:
    `m22, m3 >= 0` such that `M >> 0`, `m1 >= 0`, `m2 >= 0`,
    where
    ```
        m1 = m22*(2*t + 3)**2 - m3/(4*(t + 1)**2)
        m2 = -m22*(2*t + 3)**2 + 1/2
    ```
    and `M` is a 3*3 matrix defined in the code.
    """
    if 2*t + 3 == 0:
        # this should be separately handled to avoid division by zero
        a, b, c, d = coeff.gens
        SymmetricSum = coeff.symmetric_sum
        quad = [
            (32*a**2 + 54*a*b - 76*a*c - 76*a*d + 32*b**2 - 76*b*c - 76*b*d + 47*c**2 + 92*c*d + 47*d**2)**2/4864,
            (592*a**2 - 730*a*b + 592*b**2 - 375*c**2 + 296*c*d - 375*d**2)**2/1824000,
            73*(2*a**2 - 5*a*b + 2*b**2 + c*d)**2/3000
        ]
        quad = [2*c5*SymmetricSum(a*b*(a-b)**2*q.together()) for q in quad]
        p0 = 2*a**2 + 2*b**2 + 2*c**2 + 2*d**2 - 4*a*c - 4*a*d - 4*b*c - 4*b*d + 3*a*b + 3*c*d
        rest = [
            2*c5/16 * SymmetricSum((a-b)**2*(c-d)**2*p0**2),
            2*c5/4 * SymmetricSum(a*b*c*d*(a-b)**2*(c-d)**2)
        ]
        mul = SymmetricSum(a*(b-c)**2)
        return Add(*quad, *rest) / mul

    coeffs = [
        [
            t**2 + 2*t + 4,
            -(t + 3)/(2*(t + 1)**2*(2*t + 3)**2),
            -(t - 1)*(t + 2)/(4*(2*t + 3)**2)
        ],
        [
            2*t**2 + 6*t + 1,
            -(t**2 + 3*t + 1)/(2*(t + 1)**2*(2*t + 3)**2),
            -(2*t**3 + 5*t**2 - 6*t - 11)/(8*(2*t + 3)**2)
        ],
        [
            t + 1,
            -1/(4*(t + 1)**2*(2*t + 3)**2),
            -1/(8*(2*t + 3))
        ],
        [
            -4*(t + 2)*(2*t + 1),
            2*(t + 2)/((t + 1)*(2*t + 3)**2),
            (t**3 + 6*t**2 + 5*t - 2)/(2*(2*t + 3)**2)
        ],
        [
            2*(t + 2),
            -1/(2*(t + 1)*(2*t + 3)**2),
            -1/(4*(2*t + 3))
        ]
    ]

    def compute_mat_m1_m2(m22, m3):
        entries = []
        for c22, c3, c0 in coeffs:
            entries.append(
                c22*m22 + c3*m3 + c0
            )
        m00, m01, m02, m11, m12 = entries
        mat = coeff.as_matrix([
            [m00, m01, m02],
            [m01, m11, m12],
            [m02, m12, m22]
        ], (3, 3))
        m1 = m22*(2*t + 3)**2 - m3/(4*(t + 1)**2)
        m2 = (-2*m22*(2*t + 3)**2 + 1)/2
        return mat, m1, m2

    def make_sol(m22, m3):
        m22, m3 = coeff.wrap(m22), coeff.wrap(m3)
        if not (m3 >= 0):
            return None
        mat, m1, m2 = compute_mat_m1_m2(m22, m3)

        # print('make sol', t, m22, m3, '//',m1, m2,'//\n', mat, mat.n(15).eigenvals())

        if not (m1 >= 0 and m2 >= 0):
            return None
        cong = congruence(mat)
        if cong is None:
            return None

        U, S = cong
        a, b, c, d = coeff.gens
        SymmetricSum = coeff.symmetric_sum
        v = [
            a*b*(-2*t - 4) + a*c*(2*t + 3) + a*d*(2*t + 3) + b*c*(2*t + 3) \
                + b*d*(2*t + 3) + c**2*(-3*t - 4) + d**2*(-3*t - 4),
            -a*b + c**2*(-t - 1) + c*d*(2*t + 3) + d**2*(-t - 1),
            a**2*(2*t + 3) + a*b*(2*t**2 + 4*t + 2) + b**2*(2*t + 3) \
                + c**2*(-t**2 - 4*t - 4) + d**2*(-t**2 - 4*t - 4)
        ]
        quad = [
            s*c5*2 * SymmetricSum(a*b*(a-b)**2*Add(*[u[i]*v[i] for i in range(3)]).together()**2)
                for s, u in zip(S, U.tolist())
        ]
        p0 = a**2 + b**2 + c**2 + d**2 - 2*a*c - 2*a*d - 2*b*c - 2*b*d + (t + 3)*a*b + (t + 3)*c*d
        rest = [
            m1*c5*2 * SymmetricSum((a-b)**2*(c-d)**2*p0.together()**2),
            m2*c5*2 * SymmetricSum(a*b*(a-b)**4*(a + b + t*c + t*d).together()**2),
            m3*c5*2 * SymmetricSum(a*b*c*d*(a-b)**2*(c-d)**2)
        ]
        mul = SymmetricSum(a*(b-c)**2)
        return Add(*quad, *rest) / mul

    domain = coeff.domain
    if t < -2:
        # this is prettier than below (because it is linear), so try it first
        candidates = [
            # feasible when -2.88917053906166 < t < -2.03594184052306
            ((t + 3)/6, 0),

            # feasible when -2.16425772887787 < t < -1.93266974897326
            ((t + 3)/4, 0),

            # feasible when -3 <= t < -2.88735739523853
            (domain.one/54, 0)
        ]

    else:
        candidates = [
            # # feasible when -2.92016744568624 < t < -1.41628844163712
            # sol = make_sol(1/(2*t + 3)**2/4, domain.one/8),

            # # feasible when -1.53049108654972 < t < -1.18234597049344
            # sol = make_sol(1/(2*t + 3)**2/2, -t/2 - domain.one*6/11),

            # feasible when -2.53765349990436 < t <= -2/sqrt(3) = -1.15470053838
            (1/(2*t + 3)**2/2, -(t + 1)**2*(t**3 - 2*t**2 - 33*t - 94)/(4*(t**2 + 4*t + 15))),
        ]

    for _m22, _m3 in candidates:
        sol = make_sol(_m22, _m3)
        if sol is not None:
            return sol


def _quaternary_quintic_symmetric_c3axis(coeff: Coeff):
    c5, c41, c32, c311, c221, c2111 = [coeff(_) for _ in
        [(5,0,0,0), (4,1,0,0), (3,2,0,0), (3,1,1,0), (2,2,1,0), (2,1,1,1)]]
    if c5 == 0:
        return _quaternary_quintic_symmetric_hexagon(coeff)
    if c5 < 0:
        return None

    t = c41 / c5
    if not (t <= 0 and 4 - 3*t**2 >= 0):
        return

    y = [
        (c32 + c41 + c5)/96,
        (12*c311*c5 + 12*c32*c5 - 9*c41**2 + 12*c41*c5 + 8*c5**2)/(48*c5),
        (c221 + c311 + 2*c32 + 2*c41 + c5)/8,
        (c2111 + 3*c221 + 3*c311 + 3*c32 + 3*c41 + c5)/6
    ]
    if not all([_ >= 0 for _ in y]):
        return None

    ker = _quaternary_quintic_symmetric_c3axis_t(t, coeff, c5)
    if ker is None:
        return None

    a, b, c, d = coeff.gens
    SymmetricSum = coeff.symmetric_sum

    exprs = [
        SymmetricSum(a) * SymmetricSum((a-b)**2*(c-d)**2),
        SymmetricSum(a*b*c*(a-b)**2),
        SymmetricSum(a*b*(a+b)*(c-d)**2),
        SymmetricSum(a**2*b*c*d)
    ]
    return sum_y_exprs(y, exprs) + ker


def _quaternary_quintic_symmetric_surface(coeff: Coeff):
    c5, c41, c32, c311, c221, c2111 = [coeff(_) for _ in
        [(5,0,0,0), (4,1,0,0), (3,2,0,0), (3,1,1,0), (2,2,1,0), (2,1,1,1)]]
    if c5 == 0:
        return _quaternary_quintic_symmetric_hexagon(coeff)
    if c5 < 0:
        return None

    t = c41 / c5
    if not (t <= 0 and t >= -3 and 4 - 3*t**2 <= 0):
        return

    y = [
        (c32 + c41 + c5)/96,
        (c311*c5 + c32*c5 - c41**2 + c41*c5 + c5**2)/(4*c5),
        (c221 + c311 + 2*c32 + 2*c41 + c5)/8,
        (c2111 + 3*c221 + 3*c311 + 3*c32 + 3*c41 + c5)/6
    ]
    if not all([_ >= 0 for _ in y]):
        return None

    ker = _quaternary_quintic_symmetric_surface_t(t, coeff, c5)
    if ker is None:
        return None

    a, b, c, d = coeff.gens
    SymmetricSum = coeff.symmetric_sum

    exprs = [
        SymmetricSum(a) * SymmetricSum((a-b)**2*(c-d)**2),
        SymmetricSum(a*b*c*(a-b)**2),
        SymmetricSum(a*b*(a+b)*(c-d)**2),
        SymmetricSum(a**2*b*c*d)
    ]
    return sum_y_exprs(y, exprs) + ker
