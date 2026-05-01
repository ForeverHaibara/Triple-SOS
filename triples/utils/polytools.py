"""
Compatibility & enhancement tools for SymPy polys module.
"""
from sympy import Poly, ZZ
from sympy import __version__ as _SYMPY_VERSION
try:
    from sympy.core.random import _randint
except ImportError:
    def _randint(seed):
        import random as _random
        rng = _random.Random()
        rng.seed(seed)
        return rng.randint
from sympy.external.importtools import version_tuple
from sympy.polys.rootoftools import ComplexRootOf as CRootOf
from sympy.polys.densebasic import (
    dmp_from_dict, dmp_to_dict, dmp_zero_p, dmp_degree_in,
    dmp_degree, dmp_degree_list, dmp_ground_LC, dmp_convert,
    dmp_validate, dmp_ground_p, dmp_raise, dmp_one, dmp_nest,
    dup_LC, dmp_LC,
)
from sympy.polys.densetools import (
    dup_primitive,
    dmp_diff_in, dmp_diff_eval_in,
    dmp_eval_in, dmp_eval_tail, dmp_ground_trunc
)
from sympy.polys.densearith import (
    dup_neg, dup_mul, dup_mul_ground,
    dmp_expand, dmp_sub, dmp_mul, dmp_div, dmp_pow, dmp_add_mul,
    dmp_mul_ground, dmp_quo, dmp_quo_ground
)
from sympy.polys.euclidtools import (
    dmp_primitive, dmp_inner_gcd, dmp_gcd
)
from sympy.polys.factortools import (
    dup_gf_factor, dmp_trial_division, dmp_zz_diophantine
)
from sympy.polys.polyerrors import EvaluationFailed, ExtraneousFactors
from sympy.polys.polyutils import _sort_factors
from sympy.polys.sqfreetools import dup_gf_sqf_list, dup_sqf_p
# from sympy.polys.sqfreetools import _dmp_check_degrees
from sympy.utilities import subsets

_IS_GROUND_TYPES_FLINT = False
_FLINT_VERSION = ''
try:
    from sympy.external.gmpy import GROUND_TYPES
    _IS_GROUND_TYPES_FLINT = (GROUND_TYPES == 'flint')
    from flint import __version__ as _FLINT_VERSION
except ImportError: # sympy <= 1.8 or no flint installed
    _IS_GROUND_TYPES_FLINT = False

SYMPY_VERSION = tuple(version_tuple(_SYMPY_VERSION))
FLINT_VERSION = tuple(version_tuple(_FLINT_VERSION))

if SYMPY_VERSION >= (1, 14):
    def poly_lift(poly: Poly) -> Poly:
        return poly.lift()

    def crootof_realroots_alg(poly: Poly):
        """Get real roots of a sqf poly on an algebraic field."""
        return list(set(CRootOf.real_roots(poly, radicals=False)))

    from sympy.polys.densetools import dup_sign_variations
    from sympy.polys.rootisolation import dup_count_real_roots
else:
    """
    Fixes SymPy counting roots on algebraic fields.
    """
    from collections import Counter
    from sympy.polys.polyclasses import DMP
    from sympy.polys.densebasic import (
        dup_LC, dup_convert, dup_degree, dmp_include
    )
    from sympy.polys.densetools import dup_eval
    from sympy.polys.euclidtools import dmp_resultant
    from sympy.polys.polyerrors import DomainError
    from sympy.polys.rootisolation import dup_sturm
    def dmp_alg_inject(f, u, K):
        fd = dmp_to_dict(f, u, K)
        h = {}
        for f_monom, g in fd.items():
            for g_monom, c in g.to_dict().items():
                h[g_monom + f_monom] = c
        F = dmp_from_dict(h, u + 1, K.dom)
        return F, u + 1, K.dom

    def dmp_lift(f, u, K):
        F, v, K2 = dmp_alg_inject(f, u, K)
        p_a = K.mod.to_list()
        P_A = dmp_include(p_a, list(range(1, v + 1)), 0, K2)
        return dmp_resultant(F, P_A, v, K2) # type: ignore

    # poly.lift() had a bug before 1.14:
    # https://github.com/sympy/sympy/pull/26812
    def poly_lift(poly: Poly) -> Poly:
        if not poly.domain.is_AlgebraicField:
            return poly
        rep = poly.rep
        dmp = DMP(dmp_lift(rep.rep, rep.lev, rep.dom), rep.dom.dom, rep.lev)
        return Poly.new(dmp, *poly.gens)

    # https://github.com/sympy/sympy/pull/26813
    def dup_sign_variations(f, K):
        """
        Compute the number of sign variations of ``f`` in ``K[x]``.

        Examples
        ========

        >>> from sympy.polys import ring, ZZ
        >>> R, x = ring("x", ZZ)

        >>> R.dup_sign_variations(x**4 - x**2 - x + 1)
        2

        """
        def is_negative_sympy(a):
            if not a:
                # XXX: requires zero equivalence testing in the domain
                return False
            else:
                # XXX: This is inefficient. It should not be necessary to use a
                # symbolic expression here at least for algebraic fields. If the
                # domain elements can be numerically evaluated to real values with
                # precision then this should work. We first need to rule out zero
                # elements though.
                return bool(K.to_sympy(a) < 0)

        # XXX: There should be a way to check for real numeric domains and
        # Domain.is_negative should be fixed to handle all real numeric domains.
        # It should not be necessary to special case all these different domains
        # in this otherwise generic function.
        if K.is_ZZ or K.is_QQ or K.is_RR:
            is_negative = K.is_negative
        elif K.is_AlgebraicField and K.ext.is_comparable:
            is_negative = is_negative_sympy
        elif ((K.is_PolynomialRing or K.is_FractionField) and len(K.symbols) == 1 and
            (K.dom.is_ZZ or K.dom.is_QQ or K.is_AlgebraicField) and
            K.symbols[0].is_transcendental and K.symbols[0].is_comparable):
            # We can handle a polynomial ring like QQ[E] if there is a single
            # transcendental generator because then zero equivalence is assured.
            is_negative = is_negative_sympy
        else:
            raise DomainError("sign variation counting not supported over %s" % K)

        prev, k = K.zero, 0

        for coeff in f:
            if is_negative(coeff*prev):
                k += 1

            if coeff:
                prev = coeff

        return k

    def dup_count_real_roots(f, K, inf=None, sup=None):
        """Returns the number of distinct real roots of ``f`` in ``[inf, sup]``. """
        if dup_degree(f) <= 0:
            return 0

        if not K.is_Field:
            R, K = K, K.get_field()
            f = dup_convert(f, R, K)

        sturm = dup_sturm(f, K)

        if inf is None:
            signs_inf = dup_sign_variations([ dup_LC(s, K)*(-1)**dup_degree(s) for s in sturm ], K)
        else:
            signs_inf = dup_sign_variations([ dup_eval(s, inf, K) for s in sturm ], K)

        if sup is None:
            signs_sup = dup_sign_variations([ dup_LC(s, K) for s in sturm ], K)
        else:
            signs_sup = dup_sign_variations([ dup_eval(s, sup, K) for s in sturm ], K)

        count = abs(signs_inf - signs_sup)

        if inf is not None and not dup_eval(f, inf, K):
            count += 1

        return count

    def _which_roots(f, candidates, num_roots):
        fe = f.as_expr()
        x = f.gens[0]
        prec = 10
        candidates = list(Counter(candidates).keys())

        while len(candidates) > num_roots:
            potential_candidates = []
            for r in candidates:
                # If f(r) != 0 then f(r).evalf() gives a float/complex with precision.
                f_r = fe.xreplace({x: r}).evalf(prec, maxn=2*prec)
                if abs(f_r)._prec < 2:
                    potential_candidates.append(r)

            candidates = potential_candidates
            prec *= 2

        return candidates

    def count_real_roots(f, inf=None, sup=None):
        """Returns the number of distinct real roots of ``f`` in ``[inf, sup]``. """
        if isinstance(f, Poly):
            rep = f.rep
        if hasattr(rep, 'to_DMP_Python'):
            rep = rep.to_DMP_Python()
        return dup_count_real_roots(rep.to_list(), f.domain, inf, sup)

    def crootof_realroots_alg(poly: Poly):
        rts = poly_lift(poly).real_roots()
        cnt = count_real_roots(poly)
        return _which_roots(poly, rts, cnt)

###############################################################################
#
#          Polynomial Factorization over Finite Fields
#
###############################################################################

def dmp_gf_sqf_list(f, u, K, all = False):
    """
    Musser's algorithm for square-free factorization of
    multivariate polynomials over finite fields.
    """
    if dmp_zero_p(f, u):
        return K.zero, []
    if u == 0:
        return dup_gf_sqf_list(f, K, all=all)
    if all:
        raise NotImplementedError("all=True not implemented")

    content, f = dmp_primitive(f, u, K)
    coeff, result = dmp_gf_sqf_list(content, u - 1, K, all=all)

    result = [([fac], m) for fac, m in result]
    lc = dmp_ground_LC(f, u, K)
    if lc != K.one:
        f = dmp_mul_ground(f, K.one/lc, u, K)
        coeff = coeff * lc

    # compute gcd(f, df/dx0, ... df/dxu)
    c = f
    for i in range(u + 1):
        df = dmp_validate(dmp_diff_in(f, 1, i, u, K), K)[0]
        if dmp_zero_p(df, u):
            continue
        c = dmp_gcd(c, df, u, K)
    w = dmp_quo(f, c, u, K)

    i = 1
    while not dmp_ground_p(w, None, u):
        y, fac, c = dmp_inner_gcd(w, c, u, K)
        if not dmp_ground_p(fac, None, u):
            result.append((fac, i))
        w = y
        i += 1

    if not dmp_ground_p(c, None, u):
        p = K.characteristic()
        rep = dmp_to_dict(c, u, K)
        l = 0
        while True:
            if any(any(i%p for i in m) for m in rep):
                break
            if len(rep) == 1 and not any(next(iter(rep.keys()))):
                # constant polynomial
                break
            rep = {tuple(i//p for i in m): v for m, v in rep.items()}
            l += 1
        c = dmp_from_dict(rep, u, K)
        new_coeff, new_result = dmp_gf_sqf_list(c, u, K, all=all)

        coeff *= new_coeff
        pl = p**l
        result.extend([(h, m * pl) for h, m in new_result])

    return coeff, result


def _dmp_gf_factor_flint(f, u, K):
    if K.mod < 2**64:
        from flint import nmod_mpoly_ctx
        ctx_func = nmod_mpoly_ctx
    else:
        from flint import fmpz_mod_mpoly_ctx
        ctx_func = fmpz_mod_mpoly_ctx
    ctx = ctx_func.get([chr(i) for i in range(65, 65 + u + 1)], K.mod)
    dt = {k: int(v) for k, v in dmp_to_dict(f, u, K).items()}
    p = ctx.from_dict(dt)
    c, factors = p.factor()
    factors = [(dmp_convert(dmp_from_dict(f.to_dict(), u, K),
                    u, K, ZZ), m) for f, m in factors]
    return K(int(c)), factors


def dmp_gf_wang_test_points(f, T, ct, A, u, K):
    """Wang/EEZ: Test evaluation points for suitability. """
    if not dmp_eval_tail(dmp_LC(f, K), A, u - 1, K):
        raise EvaluationFailed('no luck')

    g = dmp_eval_tail(f, A, u, K)

    if not dup_sqf_p(g, K):
        raise EvaluationFailed('no luck')

    c, h = dup_primitive(g, K)

    if K.is_negative(dup_LC(h, K)):
        c, h = -c, dup_neg(h, K)

    v = u - 1

    E = [ dmp_eval_tail(t, A, v, K) for t, _ in T ]
    return c, h, E


def dmp_gf_wang_lead_coeffs(f, T, cs, E, H, A, u, K):
    """Wang/EEZ: Compute correct leading coefficients. """
    v = u - 1
    k = len(H)

    L = f[0]

    L_A = dmp_eval_tail(L, A, v, K)

    if K.is_zero(L_A):
        raise EvaluationFailed("Evaluation point makes leading coefficient zero.")

    HHH, CCC = [],[]

    for h in H:
        lc = dup_LC(h, K)
        multiplier = L_A / lc
        h_new = dup_mul_ground(h, multiplier, K)
        HHH.append(h_new)
        CCC.append(L)

    if k == 1:
        return f, HHH, CCC

    L_pow = dmp_pow(L, k - 1, v, K)
    f_new = dmp_mul(f, [L_pow], u, K)

    return f_new, HHH, CCC


def dmp_gf_wang_hensel_lifting(f, H, LC, A, p, u, K):
    """Wang/EEZ: Parallel Hensel lifting algorithm. """
    S, n, v = [f], len(A), u - 1

    H = list(H)

    for i, a in enumerate(reversed(A[1:])):
        s = dmp_eval_in(S[0], a, n - i, u - i, K)
        S.insert(0, dmp_ground_trunc(s, p, v - i, K))

    d = max(dmp_degree_list(f, u)[1:])

    for j, s, a in zip(range(2, n + 2), S, A):
        G, w = list(H), j - 1

        I, J = A[:j - 2], A[j - 1:]

        for i, (h, lc) in enumerate(zip(H, LC)):
            lc = dmp_ground_trunc(dmp_eval_tail(lc, J, v, K), p, w - 1, K)
            H[i] = [lc] + dmp_raise(h[1:], 1, w - 1, K)

        m = dmp_nest([K.one, -a], w, K)
        M = dmp_one(w, K)

        c = dmp_sub(s, dmp_expand(H, w, K), w, K)

        dj = dmp_degree_in(s, w, w)

        for k in range(0, dj):
            if dmp_zero_p(c, w):
                break

            M = dmp_mul(M, m, w, K)
            C = dmp_diff_eval_in(c, k + 1, a, w, w, K)

            if not dmp_zero_p(C, w - 1):
                C = dmp_quo_ground(C, K.factorial(K(k) + 1), w - 1, K)
                T = dmp_zz_diophantine(G, C, I, d, p, w - 1, K)

                for i, (h, t) in enumerate(zip(H, T)):
                    h = dmp_add_mul(h, dmp_raise(t, 1, w - 1, K), M, w, K)
                    H[i] = dmp_ground_trunc(h, p, w, K)

                h = dmp_sub(s, dmp_expand(H, w, K), w, K)
                c = dmp_ground_trunc(h, p, w, K)

    # if dmp_expand(H, u, K) != f:
    #     raise ExtraneousFactors  # pragma: no cover
    # else:
    return H

def dmp_gf_wang(f, u, K, seed=None):
    """
    Factor primitive square-free polynomials in GF(p)[X].
    """
    randint = _randint(seed)

    if dmp_ground_p(f, None, u):
        return dmp_ground_LC(f, u, K), []
    if not u:
        return dup_gf_factor(f, K)

    p = K.characteristic()
    ct, T = dmp_gf_factor(dmp_LC(f, K), u - 1, K)

    lc = dmp_ground_LC(f, u, K)
    if lc != K.one:
        f = dmp_mul_ground(f, K.one/lc, u, K)

    history = set()
    d = sum(dmp_degree_list(f, u))

    for _ in range(20):
        A = tuple([K(randint(0, p - 1)) for _ in range(u)])
        if A in history:
            continue
        history.add(A)

        try:
            cs, s, E = dmp_gf_wang_test_points(f, T, ct, A, u, K)
        except EvaluationFailed:
            continue

        _, factors = dup_gf_factor(s, K)
        H = [h for h, _ in factors]

        try:
            f1, H, LC = dmp_gf_wang_lead_coeffs(f, T, cs, E, H, A, u, K)
        except EvaluationFailed:
            continue

        H = [dmp_convert(h, 0, K, ZZ) for h in H]
        fz = dmp_convert(f1, u, K, ZZ)
        A = [ZZ(int(a)) for a in A]
        LC = [dmp_convert(g, u - 1, K, ZZ) for g in LC]

        hensel = dmp_gf_wang_hensel_lifting(fz, H, LC, A, p, u, ZZ)
        hensel = [dmp_convert(h, u, ZZ, K) for h in hensel]

        result = []
        rem = list(range(len(hensel)))
        g = f

        while rem and not dmp_ground_p(g, None, u):
            found = False

            for size in list(range(1, len(rem) // 2 + 1)) + [len(rem)]:
                for S in subsets(rem, size):
                    G = dmp_one(u, K)
                    for i in S:
                        G = dmp_mul(G, hensel[i], u, K)

                    if sum(dmp_degree_list(G, u)) > d:
                        continue

                    q, r = dmp_div(g, G, u, K)
                    if dmp_zero_p(r, u):
                        result.append(G)
                        g = q
                        rem = [i for i in rem if i not in S]
                        found = True
                        break
                if found:
                    break

            # This evaluation point does not yield a complete factorization.
            if not found:
                break

        # Only accept a complete decomposition.
        if rem or not dmp_ground_p(g, None, u):
            continue

        result = dmp_trial_division(f, result, u, K)
        result = _sort_factors(result)
        return lc, result

    raise ExtraneousFactors


def dmp_gf_factor(f, u, K):
    """
    Factor multivariate polynomials over `GF(p)[X]`
    where `p` should be prime and this is not checked.
    """
    # from sympy.ntheory import isprime
    # if not isprime(p):
    #     raise NotImplementedError('multivariate polynomials over GF(p**n)')
    if _IS_GROUND_TYPES_FLINT and FLINT_VERSION >= (0, 7, 0):
        return _dmp_gf_factor_flint(f, u, K)

    if not u:
        return dup_gf_factor(f, K)

    content, f = dmp_primitive(f, u, K)
    c, cont_result = dmp_gf_factor(content, u - 1, K)
    cont_result = [([fac], m) for fac, m in cont_result]

    c2, sqf = dmp_gf_sqf_list(f, u, K)
    c *= c2
    for g, m in sqf:
        try:
            coeff, result = dmp_gf_wang(g, u, K)
        except ExtraneousFactors:
            # fallback to slow method
            # raise ExtraneousFactors
            coeff, result = dmp_gf_kron(g, u, K)
        c *= coeff**m
        cont_result.extend([(h, m*l) for h, l in result])

    result = _sort_factors(cont_result)
    return c, result


def dmp_gf_kron(f, u, K):
    """
    Factor primitive square-free polynomials in `GF(p)[X]`
    using Kronecker's substitution.
    """
    lc = dmp_ground_LC(f, u, K)
    if lc != K.one:
        f = dmp_mul_ground(f, K.one/lc, u, K)

    d = sum(dmp_degree_list(f, u))
    if d <= 0:
        return lc, []

    p = K.characteristic()
    F = f
    f_dict = dmp_to_dict(f, u, K)

    if all(not any(e % p for e in expv) for expv in f_dict):
        f_pth = dmp_from_dict({tuple(e // p for e in expv): coeff
            for expv, coeff in f_dict.items()}, u, K)
        coeff, factors = dmp_gf_factor(f_pth, u, K)
        return K.mul(lc, coeff), [(g, k*p) for g, k in factors]

    n = u + 1
    bounds = [dmp_degree_in(F, i, u) + 1 for i in range(n)]
    B = max(bounds) + 1

    powers = [1]
    for _ in range(1, n):
        powers.append(powers[-1]*B)

    def _encode(expv):
        return sum(e*w for e, w in zip(reversed(expv), powers))

    def _decode(m):
        expv = [0]*n

        for i in range(n):
            expv[n - i - 1] = m % B
            m //= B

        if m:
            return None

        return tuple(expv)

    def _to_univariate(g):
        G = {(_encode(e),): v for e, v in dmp_to_dict(g, u, K).items()}
        return dmp_from_dict(G, 0, K)

    def _from_univariate(g):
        G = {}

        for (m,), coeff in dmp_to_dict(g, 0, K).items():
            expv = _decode(m)

            if expv is None:
                return None

            if any(e > dmp_degree_in(F, i, u) for i, e in enumerate(expv)):
                return None

            G[expv] = coeff

        return dmp_from_dict(G, u, K)

    U = _to_univariate(F)
    _, ufactors = dup_gf_factor(U, K)

    if not ufactors:
        return lc, []

    atoms = []
    for h, k in ufactors:
        for _ in range(k):
            atoms.append(h)

    result = []
    rem = list(range(len(atoms)))

    while rem and not dmp_ground_p(f, None, u):
        found = False

        for size in range(1, len(rem)//2 + 1):
            for S in subsets(rem, size):
                h = [K.one]
                deg_sum = sum(dmp_degree(atoms[i], 0) for i in S)
                expv = _decode(deg_sum)

                if expv is None or any(
                    e > dmp_degree_in(f, i, u) for i, e in enumerate(expv)):
                    continue

                for i in S:
                    h = dup_mul(h, atoms[i], K)
                g = _from_univariate(h)
                if g is None:
                    continue

                dg = sum(dmp_degree_list(g, u))

                if dg <= 0 or dg >= d:
                    continue

                q, r = dmp_div(f, g, u, K)

                if dmp_zero_p(r, u):
                    result.append(g)
                    f = q
                    rem = [i for i in rem if i not in S]
                    found = True
                    break
            if found:
                break

        if not found:
            break

    if not dmp_ground_p(f, None, u):
        result.append(f)

    factors = []
    for g in result:
        if g not in factors:
            factors.append(g)

    result = dmp_trial_division(F, factors, u, K)

    # _dmp_check_degrees(F, u, result)

    return lc, result
