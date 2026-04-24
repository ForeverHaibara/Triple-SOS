"""
Compatibility & enhancement tools for SymPy polys module.
"""
from sympy import Poly, ZZ
from sympy import __version__ as _SYMPY_VERSION
from sympy.external.importtools import version_tuple
from sympy.polys.rootoftools import ComplexRootOf as CRootOf
from sympy.polys.densebasic import (
    dmp_from_dict, dmp_to_dict, dmp_zero_p, dmp_degree_in,
    dmp_degree, dmp_degree_list, dmp_ground_LC, dmp_convert
)
from sympy.polys.densetools import dmp_ground_monic
from sympy.polys.densearith import dup_mul, dmp_div
from sympy.polys.factortools import dup_gf_factor, dmp_trial_division
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
    factors = [(dmp_convert(dmp_from_dict(f.to_dict(), u, K), u, K, ZZ), m) for f, m in factors]
    return K(int(c)), factors

def dmp_gf_factor(f, u, K):
    """
    Factor multivariate polynomials over finite fields.
    K.mod should be prime and this is not checked.

    TODO: 1. review the code 2. use faster algorithms
    """
    p = K.mod

    # from sympy.ntheory import isprime
    # if not isprime(p):
    #     raise NotImplementedError('multivariate polynomials over GF(p**n)')
    if _IS_GROUND_TYPES_FLINT and FLINT_VERSION >= (0, 7, 0):
        return _dmp_gf_factor_flint(f, u, K)

    if not u:
        return dup_gf_factor(f, K)

    lc = dmp_ground_LC(f, u, K)
    f = dmp_ground_monic(f, u, K)

    if all(d <= 0 for d in dmp_degree_list(f, u)):
        return lc, []

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
        G = {( _encode(expv), ): coeff for expv, coeff in dmp_to_dict(g, u, K).items()}
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

    while rem and dmp_degree(f, u) > 0:
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

                dg = dmp_degree(g, u)

                if dg <= 0 or dg >= dmp_degree(f, u):
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

    if dmp_degree(f, u) > 0:
        result.append(f)

    factors = []
    for g in result:
        if g not in factors:
            factors.append(g)

    result = dmp_trial_division(F, factors, u, K)

    # _dmp_check_degrees(F, u, result)

    return lc, result
