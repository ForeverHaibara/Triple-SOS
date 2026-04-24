from typing import Tuple, Optional, Union

from sympy import Poly, Expr, Add, ZZ, QQ, FiniteField, sqrt, prod
from sympy.polys.polyclasses import ANP, DMP
from sympy.polys.polyerrors import CoercionFailed
from sympy.combinatorics.named_groups import CyclicGroup
from sympy.ntheory import factorint, nextprime, sqrt_mod
from sympy.external.gmpy import sqrt as isqrt
from sympy.utilities import subsets

from .utils import (
    Coeff, sos_struct_handle_uncentered, sos_struct_reorder_symmetry,
)
from ..univariate import prove_univariate
from ....utils import verify_symmetry, poly_reduce_by_symmetry
from ....utils.polytools import dmp_gf_factor, FLINT_VERSION


def _linear_invert(u, v, d: int = 0) -> Optional[Tuple[int, Expr, Expr]]:
    """
    Returns n, u2, v2 such that `n, u2, v2 >= 0` and
    `u2 = u - (u + v)*n` and `v2 = v + (u + v)*n`. If
    `d` is provided, `n` is chosen to be close to `d`.
    Returns None if no such solution exists.

    This is equivalent to `-v/(u + v) <= n <= u/(u + v)`.

    Note that `a^(-n) == (-n*a + (n+1))  (mod (a-1)^2)` and
    `(u*a + v)*(-n*a + (n+1)) + n*u*(a-1)**2 == (u - (u + v)*n)*a + v + (u + v)*n`,
    so this implies:
    `u*a + v == (u2*a + v2)*a**n (mod (a-1)**2)`
    """
    s = u + v
    if s == 0:
        # if u >= 0 and v >= 0:
        if u == 0 and v == 0:
            return d, u, v
        return None
    if s < 0 or u < 0:
        return None
    if -v <= d * s <= u:
        n = d
    elif d * s > u:
        n = u // s
    else: # d < -v/(u+v) and v < 0
        # n = (-v - 1)//s + 1
        n = -(v // s)
    u2, v2 = u - s * n, v + s * n
    if not (n >= 0 and u2 >= 0 and v2 >= 0):
        return None
    return n, u2, v2


def sos_struct_dense_symmetric(coeff, real=True):
    """
    Solve dense 3-var symmetric inequalities.
    Triggered only when the degree is at least 8 and the polynomial is symmetric.
    """
    if coeff.total_degree() < 8 or not coeff.is_symmetric():
        return None
    return _sos_struct_dense_symmetric(coeff, real)


def _sos_struct_dense_symmetric(coeff, real=True):
    """
    Solve dense 3-var symmetric inequalities. This function
    does not check the symmetry of the input polynomial, so it
    should only be called when the symmetry is guaranteed.
    """
    d = coeff.total_degree()
    methods = [_sos_struct_trivial_additive]
    if d < 8:
        from .solver import _structural_sos_3vars_cyclic
        methods.append(_structural_sos_3vars_cyclic)
    else:
        methods.append(sos_struct_liftfree_for_six)
        methods.append(_sos_struct_lift_for_six)

    for method in methods:
        solution = method(coeff, real=real)
        if solution is not None:
            return solution


def _sos_struct_trivial_additive(coeff: Coeff, real=True):
    """
    Solve trivial cyclic inequalities with nonnegative coefficients.
    """
    if any(_ < 0 for _ in map(coeff.wrap, coeff.values())):
        return None
    a, b, c = coeff.gens
    CyclicSum = coeff.cyclic_sum

    exprs = []
    to_sympy = coeff.domain.to_sympy
    for (i,j,k), v in coeff.items():
        if (i > j and i > k) or (i == j and i > k):
            exprs.append(to_sympy(v) * a**i * b**j * c**k)

    d = coeff.total_degree()
    if d % 3 == 0:
        d = d // 3
        v = coeff((d,d,d))
        if v != 0:
            exprs.append(v/3 * a**d * b**d * c**d)

    return CyclicSum(Add(*exprs))


def sym_axis(coeff: Coeff, d: int = -1) -> Poly:
    """Compute f(a,1,1)."""
    if d == -1: d = coeff.total_degree()
    coeff_list = [0] * (d+1)
    for m, v in coeff.items():
        coeff_list[m[0]] += v
    a = coeff.gens[0]
    return coeff.from_list(coeff_list[::-1], gens=(a,)).as_poly()


def _homogenize_sym_axis(coeff: Union[Coeff, Poly], sym: Poly, d: int) -> Expr:
    """Homogenize f(a,1,1) to f(a,b,c) given degree."""
    a, b, c = coeff.gens
    s = [0]
    for (m,), v in sym.terms():
        k, r = divmod(d-m, 2)
        if r == 0:
            s.append(v * a**m * b**k * c**k)
        else:
            s.append(v/2 * a**m * b**(k+1) * c**k)
            s.append(v/2 * a**m * b**k * c**(k+1))
    return Add(*s)

def _homogenize_sym_proof(coeff: Coeff, sym_proof, d: int) -> Expr:
    """Homogenize the result from prove_univariate."""
    a, b, c = coeff.gens
    exprs = []
    for i in range(len(sym_proof)):
        leading = sym_proof[i][0]
        ld = leading.as_poly(a).degree()
        part_expr = []
        for k, v in sym_proof[i][1]:
            v2 = _homogenize_sym_axis(coeff, v, v.degree())
            rd = (d - ld - v.degree()*2) // 2
            part_expr.append(k * v2**2 * b**rd * c**rd)
        part_expr = Add(*part_expr)
        if (d - ld) % 2 == 1:
            part_expr = part_expr * leading * (b + c) / 2
        else:
            part_expr = part_expr * leading
        exprs.append(part_expr)
    return Add(*exprs)


@sos_struct_handle_uncentered
def _sos_struct_lift_for_six(coeff: Coeff, real=True):
    """
    Solve high-degree (dense) symmetric inequalities by the method
    of lifting the degree for six. Idea: define f(a,1,1) to be the
    symmetric axis of the polynomial. If two tenary homogeneous symmetric polynomials
    have equal symmetric axis, then their difference is a multiple of (a-b)^2*(b-c)^2*(c-a)^2.

    We can subtract a nonnegative polynomial with equal symmetric axis and then
    divide the rest by (a-b)^2*(b-c)^2*(c-a)^2 to reduce the degree of the problem.
    We can repeat this process until the degree is small enough.

    Examples
    --------
    => s(a9)s(a3)+9p(a4)-6a3b3c3s(a3)

    => s(a12-3a8b2c2-a6b6+2a5b5c2+a4b4c4)

    => s(a2)s((s(ab)2+a2bc)(b2+ac)(c2+ab))-p(a2+bc)*3s(a3b+a3c+3a2bc)
    """
    d = coeff.total_degree()
    if d < 8:
        return None

    a, b, c = coeff.gens
    CyclicSum, CyclicProduct = coeff.cyclic_sum, coeff.cyclic_product

    sym = sym_axis(coeff, d)
    div = sym.div(Poly([1,-2,1], a, domain=sym.domain))
    if not div[1].is_zero:
        return None
    if div[0].LC() < 0 or div[0](0) < 0 or div[0](1) < 0:
        return None

    # if any(_ < 0 for _ in div[0].coeffs()):
    #     # raise NotImplementedError
    #     return None
    if d % 2 == 0:
        factors = div[0].factor_list()[1]
        if all(m % 2 == 0 for _, m in factors):
            # XXX: currently modp factorization is slow
            # unless flint is installed
            sol = _sos_struct_complex_factorizable(coeff,
                    modp=(FLINT_VERSION >= (0, 7, 0)))
            if sol is not None:
                return sol

    if all(_ >= 0 for _ in div[0].coeffs()):
        lifted_sym = _homogenize_sym_axis(coeff, div[0], d - 2)
    else:
        sym_proof = prove_univariate(div[0], (0, None), return_type='list')
        if sym_proof is None:
            return None
        lifted_sym = _homogenize_sym_proof(coeff, sym_proof, d - 2)
        # print(lifted_sym)


    def compute_diff(coeff: Coeff, sym2, mul, tail) -> Coeff:
        poly = coeff.as_poly() * mul.as_poly(a, b, c, domain=coeff.domain)
        diff = poly - CyclicSum(sym2 * tail).as_poly(a, b, c, domain=coeff.domain)
        diff = diff.div(CyclicProduct((a-b)**2).as_poly(a, b, c, domain=coeff.domain))
        return coeff.from_poly(diff[0])

    multipliers = [
        (CyclicSum((a-b)**2)/2, (a-b)**2*(a-c)**2)
        # CyclicSum(a*b*(a-b)**2)/2
    ]
    for mul, tail in multipliers:
        diff = compute_diff(coeff, lifted_sym, mul, tail)
        # print(diff.as_poly((a,b,c)).as_expr())
        sol_diff = _sos_struct_dense_symmetric(diff)
        if sol_diff is not None:
            return Add(
                CyclicSum(lifted_sym * tail),
                CyclicProduct((a-b)**2) * sol_diff
            ) / mul


@sos_struct_handle_uncentered
def sos_struct_liftfree_for_six(coeff: Coeff, real=True):
    """
    Solve high-degree (dense) symmetric inequalities without
    lifting the degree. This will be tried in prior because
    it avoids lifting the degree of the problem.

    The idea is to first subtract something like:

        `s(a^n*b^m*c^m*(a-b)*(a-c)*(u*a + v*(b+c)/2)) + p(a-b)^2*...`

    so that the symmetric axis of the remaining part is a multiple of (a-1)^4.
    Then the rest part can be seen as:

        `s(...*(a-b)^2*(a-c)^2) + p(a-b)^2*...`

    which would perhaps be trivially nonnegative if successful.

    Examples
    --------
    => s(a9)s(a3)+9p(a4)-6a3b3c3s(a3)

    => s(a2)s((s(ab)2+a2bc)(b2+ac)(c2+ab))-p(a2+bc)*3s(a3b+a3c+3a2bc)
    """
    d = coeff.total_degree()
    if d < 6:
        return None

    a, b, c = coeff.gens
    CyclicSum, CyclicProduct = coeff.cyclic_sum, coeff.cyclic_product

    sym = sym_axis(coeff, d)
    div = sym.div(Poly([1,-2,1], a, domain=sym.domain))
    if not div[1].is_zero:
        return None

    div2 = div[0].div(Poly([1,-2,1], a, domain=sym.domain))
    if div2[1].is_zero:
        return _sos_struct_liftfree_for_six_ord4(coeff, div2[0], real=real)

    # subtract some s(a^n*b^m*c^m*(a-b)*(a-c)*(a + (b+c))) + p(a-b)^2*...
    # so that the symmetric axis of the remaining part is a multiple of (a-1)^4
    u, v = div2[1].coeff_monomial((1,)), div2[1].coeff_monomial((0,))

    # find n, u2, v2 >= 0 that (u*a + v) == (u2*a + v2)*a**n (mod (a-1)**2)
    inv = _linear_invert(u, v, (d+1)//3 - 1)
    if inv is None:
        return None
    n, u2, v2 = inv
    if not n <= d - 3:
        return None

    m = (d - n - 3) // 2
    def _subtractor1(n, m):
        # Return some f(a,b,c), such that f(a,b,c) >= 0 holds
        # and f = s(a**n*b**m*c**m*(a-b)*(a-c)) (mod p(a-b)^2)
        # NOTE: n + 2*m + 2 = d >= 6
        if n >= m:
            if (n - m) % 2 == 0:
                return CyclicProduct(a**m) * CyclicSum((b+c-a)**(n-m)*(b-c)**2)/2
            else:
                if n == m + 1 and m >= 1:
                    # tighter
                    return CyclicProduct(a**(m-1)) * CyclicSum(a**2*(a*b+a*c-b**2-c**2)**2)/6
                if m >= 1:
                    return CyclicProduct(a**(m-1)) * CyclicSum(a**2*(b-c)**2*(b+c-a)**(n-m+1))/2
        else:
            return CyclicProduct(a**n) * CyclicSum(a**(2*(m-n))*(b-c)**2)/2
    def _subtractor2(n, m):
        # Return some f(a,b,c), such that f(a,b,c) >= 0 holds
        # and f = s(a**n*b**m*c**m*(a-b)*(a-c)*(b+c)/2) (mod p(a-b)^2)
        # NOTE: n + 2*m + 3 = d >= 6
        if n >= m:
            if (n - m) % 2 == 0:
                return CyclicProduct(a**m) * CyclicSum(a*(b+c-a)**(n-m)*(b-c)**2)/2
            else:
                return CyclicProduct(a**m) * CyclicSum(b*c*(b-c)**2*(b+c-a)**(n-m-1))/2
        else:
            return CyclicProduct(a**n) * CyclicSum(a**(2*(m-n)+1)*(b-c)**2)/2

    if (d - n - 3) % 2 == 0:
        # subtract s(a^n*b^m*c^m*(u2*a + v2*(b+c)/2)*(a-b)*(a-c)) + p(a-b)^2*...
        if n >= m and m == 0:
            return None
        subtractor = u2*_subtractor1(n+1, m) + v2 * _subtractor2(n, m)
    else:
        # subtract s(a^n*b^m*c^m*(u2*a*(b+c)/2 + v2*b*c)*(a-b)*(a-c)) + p(a-b)^2*...
        subtractor = u2 * _subtractor2(n+1, m) + v2 * _subtractor1(n, m+1)
    rem_poly = coeff - coeff.from_poly(subtractor.doit().as_poly(a,b,c,domain=coeff.domain))
    solution = _sos_struct_liftfree_for_six_ord4(rem_poly, real=real)
    if solution is not None:
        return subtractor + solution


def _sos_struct_liftfree_for_six_ord4(coeff: Coeff, div2=None, real=True):
    """
    Solve a high-degree (dense) symmetric inequality where (a-1)^4 is a factor
    of the symmetric axis. Such polynomial can be seen as:

        `s(...*(a-b)^2*(a-c)^2) + p(a-b)^2*...`

    which would be trivially nonnegative if successful.
    """
    a, b, c = coeff.gens
    CyclicSum, CyclicProduct = coeff.cyclic_sum, coeff.cyclic_product

    if div2 is None:
        sym = sym_axis(coeff, coeff.total_degree())
        div2 = sym.div(Poly([1,-4,6,-4,1], a, domain=sym.domain))
        if not div2[1].is_zero:
            return None
        div2 = div2[0]

    if div2.LC() < 0 or div2(0) < 0 or div2(1) < 0:
        return None

    d = coeff.total_degree()
    if all(_ >= 0 for _ in div2.coeffs()):
        lifted_sym = _homogenize_sym_axis(coeff, div2, d - 4)
    else:
        return None # not implemented

    poly = coeff.as_poly()
    diff = poly - CyclicSum((a-b)**2*(a-c)**2*lifted_sym).doit().as_poly(a, b, c, domain=coeff.domain)
    diff = diff.div(CyclicProduct((a-b)**2).doit().as_poly(a, b, c, domain=coeff.domain))[0]
    sol_diff = _sos_struct_dense_symmetric(coeff.from_poly(diff))
    if sol_diff is not None:
        return CyclicSum(lifted_sym * (a-b)**2*(a-c)**2) + CyclicProduct((a-b)**2) * sol_diff


def _conjugate_factor(poly: Poly, conj):
    const, factors = poly.factor_list()
    if const < 0:
        return None

    even_factors = [(f, m//2) for f, m in factors if m % 2 == 0]
    odd_factors = {f: m for f, m in factors if m % 2 == 1}
    if len(odd_factors) % 2 == 1:
        # odd factors must be pairwise conjugate
        return None

    def conj_poly(x):
        rep = {m: conj(v) for m, v in x.rep.to_dict().items()}
        return Poly.from_dict(rep, *x.gens, domain=x.domain)

    for f, mul in odd_factors.items():
        if mul == 0:
            # already processed
            continue
        conj_f = conj_poly(f)
        mul2 = odd_factors.get(conj_f, -1)
        if mul2 != mul:
            # not conjugate in pairs
            return None
        if f == conj_f:
            # the factor is real but has odd degree
            return None
        even_factors.append((f, mul))
        odd_factors[f] = 0
        odd_factors[conj_f] = 0
    return const, even_factors

def _sos_struct_complex_factorizable(coeff: Coeff, test=True, modp=True):
    """
    Solve a cyclic ternary polynomial inequality if it is
    factorizable over Q[sqrt(-3)].

    Examples
    --------
    => s(a2(a-b-c)4(a-b)(a-c))

    => 2/3s(a12-3a11b-3a11c+3a10b2+3a10bc+3a10c2+2a6b6-6a5b5c2)

    => s((a+b)4(a-c)2(b-c)2)-(8+4sqrt(6))s(ab)p(a-b)2
    """
    if not coeff.domain.is_Exact: # RR or CC
        return None
    d = coeff.total_degree()
    if d % 2 != 0:
        return None

    poly0 = coeff.as_poly().eval((1,))
    if test:
        # test whether the poly > 0 for real numbers
        for point in ((0, -1), (-2, -3)):
            if poly0(*point) < 0:
                return None

    if modp and (coeff.domain.is_QQ or coeff.domain.is_ZZ):
        return _sos_struct_complex_factorizable_fp(coeff)

    dom0 = coeff.domain
    dom = dom0.unify(QQ.algebraic_field(sqrt(-3)))

    # dehomogenize to factor faster
    poly = poly0.set_domain(dom)

    # map from ANP to its conjugate
    if dom0.is_QQ or dom0.is_ZZ:
        def conj(x):
            return ANP([-x.rep[0], x.rep[1]], x.mod, x.dom) if len(x.rep) == 2 else x
    elif dom0.is_Algebraic:
        z = ANP([1, 0], dom.mod.to_list(), QQ)
        try:
            conj_z = dom.convert(dom.to_sympy(z).conjugate())
        except CoercionFailed:
            return None
        def conj(x):
            s = dom.zero
            for ci in x.rep:
                s = conj_z * s + ci
            return s
    else:
        return None

    def conj_poly(x):
        rep = {m: conj(v) for m, v in x.rep.to_dict().items()}
        return Poly.from_dict(rep, *x.gens, domain=x.domain)

    if test:
        # test if the projection is factorizable, so that
        # we do not waste time on non-factorizable cases
        for point in (0, 7, -5):
            if _conjugate_factor(poly.eval((point,)), conj) is None:
                return None

    factors = _conjugate_factor(poly, conj)
    if factors is None:
        return None
    const, even_factors = factors

    half = poly.one
    for f, mul in even_factors:
        half *= f**mul
    conj_half = conj_poly(half)

    a = coeff.gens[0]
    real = (half + conj_half).mul_ground(dom.one/2).homogenize(a)
    imag = (half - conj_half).mul_ground(dom.convert(1/(2*sqrt(-3)))).homogenize(a)

    return _complex_factorizable_from_AB(coeff, real, imag, const)

def _complex_factorizable_from_AB(coeff: Coeff, A: Poly, B: Poly, const, k=3):
    """
    Returns `coeff.as_poly() == const*A**2 + 3*const*B**2` in
    a graceful form.
    """
    is_cyclic = coeff.is_cyclic()
    CyclicSum = coeff.cyclic_sum
    def sym_sq(p):
        if verify_symmetry(p, CyclicGroup(3)):
            return CyclicSum(poly_reduce_by_symmetry(p, CyclicGroup(3)).as_expr())**2
        if is_cyclic:
            return CyclicSum(p.as_expr()**2)/3
        return p.as_expr()**2
    sqa = sym_sq(A)
    sqb = sym_sq(B)

    return const*sqa + k*const*sqb


def _sos_struct_complex_factorizable_fp(coeff: Coeff):
    """
    Solve a homogeneous inequality F(a,b,c) = const*(A^2 + 3*B^2) >= 0
    where const in QQ and (F, A, B) in QQ[a,b,c] by computing on
    the Fp field. This is faster than computing on Q[sqrt(-3)].
    """
    poly = coeff.as_poly()
    if not (poly.domain.is_QQ or poly.domain.is_ZZ):
        return None
    poly = poly.eval((1,))
    const, factors = poly.factor_list()
    if const < 0:
        return None

    const = poly.domain.to_sympy(const)

    A, B = poly.one, poly.zero
    for factor, m in factors:
        if m > 0:
            fpow = factor**(m//2)
            A, B = A * fpow, B * fpow
        if m % 2 == 0:
            continue
        f = factor.set_domain(ZZ)

        # make the leading term a square
        lcfactors = factorint(int(f.LC()))
        mul = int(prod([f for f, m in lcfactors.items() if m % 2 == 1]))
        if mul != 1:
            f = f.mul_ground(mul)
            const = const / mul

        result = _sqf_complex_factorizable_fp(f)
        if result is None:
            return None
        A1, B1 = result
        A, B = A*A1 + 3*B*B1, A*B1 + B*A1
    a = coeff.gens[0]
    A, B = A.homogenize(a), B.homogenize(a)
    # return const*A.as_expr()**2 + 3*const*B.as_expr()**2
    return _complex_factorizable_from_AB(coeff, A, B, const)


def _sqf_complex_factorizable_fp(poly: Poly, p: Optional[int]=None):
    """
    Express `poly = A**2 + 3*B**2` assuming poly is on ZZ and square-free.

    Note
    -----
    * The poly should have domain ZZ.
    * The leading coefficient should be a square.

    Examples
    --------
    >>> from sympy.abc import a, b, c

    Examples
    --------
    >>> from sympy.abc import a, b
    >>> c = 1
    >>> p1 = (a**2+b**2+c**2)**2 - 3*(a**3*b + b**3*c + c**3*a)
    >>> p1 = p1.as_poly(a, b)
    >>> A, B = _sqf_complex_factorizable_fp(p1)
    >>> (A**2 + 3*B**2 - p1).is_zero
    True
    """
    if poly.is_zero:
        return poly, poly
    if not poly.domain.is_ZZ:
        return None
    def is_square(x):
        return isqrt(x)**2 == x
    if not is_square(poly.rep.LC()):
        return None
    d = poly.total_degree()
    if d % 2 != 0:
        return None

    u = len(poly.gens) - 1

    if p is None:
        # mignotte bound is too large (even for ZZ)
        # from sympy.polys.factortools import dmp_zz_mignotte_bound
        # bound = dmp_zz_mignotte_bound(f, u, ZZ)

        # we just use a heuristic bound
        bound = poly.max_norm() * 5 + 10

        p = 2*bound
        while p % 6 != 1:
            p = nextprime(p)
    w = sqrt_mod(-3, p)
    if w is None:
        return None
    K = FiniteField(p)
    invw = 1/K(w)

    poly_K = poly.set_domain(K)
    const, factors = dmp_gf_factor(poly_K.rep.to_list(), u, K)
    if len(factors) % 2 == 1:
        return None
    sqrt_const = sqrt_mod(int(const), p)
    if sqrt_const is None:
        return None

    factors = [(Poly.new(DMP(factor, K, u), *poly.gens), m) for factor, m in factors]
    degrees = [f.total_degree()*m for f, m in factors]

    def reconstruct(s):
        p1 = poly_K.one
        for i in s:
            f, m = factors[i]
            p1 *= f**m
        return p1
    def to_zz(p1):
        half_p = p//2
        p1 = p1.set_domain(ZZ)
        dt = {k: v if v <= half_p else v - p
              for k, v in p1.rep.to_dict().items()}
        return Poly.from_dict(dt, *poly.gens, domain=ZZ)

    point = [2]
    while len(point) <= u: # len(point) < len(poly.gens)
        point.append(nextprime(point[-1]))
    points = [point]

    inds = list(range(len(factors)))
    for s in subsets(inds, len(factors)//2):
        ns = tuple([i for i in range(len(factors)) if i not in s])
        if s > ns:
            continue
        ds = sum(degrees[i] for i in s)
        if ds * 2 != d:
            continue

        A = reconstruct(s).mul_ground(sqrt_const)
        B = reconstruct(ns).mul_ground(sqrt_const)
        A, B = to_zz((A + B)), to_zz((A - B).mul_ground(invw))

        if any(A(*point)**2 + B(*point)**2*3 != 4*poly(*point)
               for point in points):
            # quick check before expensive check
            continue

        if (A**2 + B**2*3 - 4*poly).is_zero:
            half = QQ.one/2
            return A.set_domain(QQ).mul_ground(half),\
                B.set_domain(QQ).mul_ground(half)
    return None

#####################################################################
#
#                              Acyclic
#
#####################################################################

@sos_struct_reorder_symmetry(groups=(2, 1))
def sos_struct_ternary_dense_partial_symmetric(coeff: Coeff, real=True):
    """
    Solve a homogeneous 3-var inequality `f(a,b,c) >= 0` where
    `f(a, b, c) == f(b, a, c)`.

    Examples
    --------
    => ((2a+2b+c)c-2ab)2(a+b)+3ab(4((a-b)2+c2)s(a)+8c(ab-c2)-9c2(a+b))

    => (a2+b2+2c2)3-4(2abc+(a+b)c2)2

    => s(a4(a-b)(a-c))+2a4(a-c)2+2b4(b-c)2
    """
    if all(v >= 0 for v in coeff.coeffs()):
        return coeff.as_poly().as_expr()
    if coeff.total_degree() <= 1:
        return None

    a, b, c = coeff.gens
    poly = coeff.as_poly()
    axis = poly.eval((1,1))
    div, rem = axis.div(Poly([1,-2,1], c, domain=axis.domain))
    if not rem.is_zero:
        return None
    div, rem = div.div(Poly([1,-2,1], c, domain=div.domain))

    subtractors = []
    factories = [
        _get_ternary_dense_partial_symmetric_default_subtractor,
        _get_ternary_dense_partial_symmetric_cubic_subtractor,
    ]
    for factory in factories:
        subtractors.extend(factory(coeff, rem))
    # print(subtractors)
    for subtractor in subtractors:
        remain = poly - subtractor.as_poly(a, b, c, domain=coeff.domain)
        sol = _ternary_dense_partial_symmetric_ord4(coeff.from_poly(remain))
        if sol is not None:
            return subtractor + sol

def _get_ternary_dense_partial_symmetric_default_subtractor(coeff: Coeff, rem: Poly) -> list:
    d = coeff.total_degree()
    a, b, c = coeff.gens

    # subtract something so that the remaining f(a,b,c) has (c-1)**4 | f(1, 1, c)
    inv = _linear_invert(rem.coeff_monomial((1,)), rem.coeff_monomial((0,)),
                         (d+1)//3 - 1)
    if inv is None:
        return []
    n, u2, v2 = inv

    subtractor = Add()
    m = (d - n - 3) // 2
    if m < 0:
        return []
    elif (d - n - 3) % 2 == 0:
        subtractor = a**m*b**m*c**n*(u2*c + v2/2*(a + b))*(2*c - a - b)**2/4
    else:
        m = (d - n - 3) // 2
        subtractor = a**m*b**m*c**n*(u2/2*c*(a + b) + v2*a*b)*(2*c - a - b)**2/4
    return [subtractor]

def _get_ternary_dense_partial_symmetric_cubic_subtractor(coeff: Coeff, rem: Poly) -> list:
    d = coeff.total_degree()
    if d <= 5:
        return []
    zero, one = rem.domain.zero, rem.domain.one

    m = (d - 6 + 1)//3
    n = d - 6 - 2*m
    rem_lift = rem * Poly([-n, n + 1], rem.gens[0], domain=rem.domain)
    rem_lift = rem_lift.div(Poly([1, -2, 1], rem.gens[0], domain=rem.domain))[1]
    rem_rep = rem_lift.rep.to_list()
    r2 = rem_rep[-1] if len(rem_rep) else zero
    r1 = rem_rep[-2] if len(rem_rep) >= 2 else zero

    if coeff.wrap(r1 + r2) <= 0:
        return []

    def ab_subtractor(u, v, w = zero, t = one):
        """f(1,0,t) = (df/db)(1,0,t) = 0, f(1,1,c) = (uc**2 + v*c + w)*(c - 1)"""
        return [
            -2*t**3*u + 3*t**2*u/2 - 3*t**2*v/2 + t*v - t*w + w/2,
            2*t**3*u - 3*t**2*u/2 + 3*t**2*v/2 - t*v + t*w - w,
            (2*t**3*u - 2*t**2*u + 2*t**2*v - 2*t*v + 2*t*w - w)/(2*t),
            (-2*t**3*u + 2*t**2*u - 2*t**2*v + t*v - t*w + w)/t,
            -u/2 + v/2,
            u
        ]

    def to_expr(lst):
        a, b, c = coeff.gens
        c300, c210, c201, c111, c102, c003 = lst
        return (r1 + r2)*a**m*b**m*c**n*Add(
            c**3*c003, a**3*c300, b**3*c300,
            a*c**2*c102, a*b**2*c210, b*c**2*c102,
            a**2*b*c210, a**2*c*c201, b**2*c*c201, a*b*c*c111
        ).together()**2

    u, v = r1/2/(r1 + r2), (r1 + 2*r2)/2/(r1 + r2)

    # c*(-v*c + u + 2*v) == u*c + v (mod (c - 1)**2)
    subtractors = [
        ab_subtractor(-v, u + 2*v, zero),
        ab_subtractor(zero, u, v)
    ]
    subtractors = [to_expr(lst) for lst in subtractors]
    return subtractors


def _ternary_dense_partial_symmetric_ord4(coeff: Coeff):
    """
    Solve a homogeneous 3-var inequality `f(a,b,c) >= 0` where
    `f(a, b, c) == f(b, a, c)` and `(c - 1)**4 | f(1, 1, c)`.
    """
    a, b, c = coeff.gens
    poly = coeff.as_poly()
    div2, rem2 = poly.eval((1,1)).div(Poly([1, -4, 6, -4, 1], c, domain=poly.domain))

    # print('div2', div2, 'rem2', rem2)
    if not rem2.is_zero:
        # not expected to happen
        return None

    if not all(v >= 0 for v in div2.coeffs()):
        return None

    d = coeff.total_degree()
    sym = _homogenize_sym_axis(Poly(0, c, a, b), div2, d - 4)
    remain2 = poly - (sym*(a - c)**2*(b - c)**2).as_poly(a, b, c, domain=poly.domain)
    div3, rem3 = remain2.div((a - b).as_poly(a, b, domain=remain2.domain)**2)
    if not rem3.is_zero:
        # not expected to happen
        return None

    from .solver import _structural_sos_3vars_acyclic
    sol = _structural_sos_3vars_acyclic(coeff.from_poly(div3))
    if sol is not None:
        return sym * (a-c)**2*(b-c)**2 + sol * (a-b)**2
