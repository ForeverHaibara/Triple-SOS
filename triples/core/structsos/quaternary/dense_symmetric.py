from typing import Tuple, Optional

from sympy import Poly, Symbol, Add
from sympy.combinatorics import PermutationGroup, Permutation

from .utils import Coeff, CyclicSum, sos_struct_reorder_symmetry
from ..univariate import prove_univariate
from ....utils import poly_reduce_by_symmetry, arraylize_sp, invarraylize

def _sym_sum_poly(poly: Poly) -> Poly:
    """Compute the symmetric sum of a polynomial efficiently."""
    rep = arraylize_sp(poly, expand_cyc=True, sym=True)
    return invarraylize(rep, poly.gens, degree=poly.total_degree(), sym=True)


def _lift_sym_axis_to_D4(
    poly: Poly,
    gens: Tuple[Symbol, Symbol, Symbol, Symbol],
    degree: int
) -> Optional[Poly]:
    """
    Given a nonhomogeneous f(a,b) such that `f(a,b) == f(b,a)`,
    compute a homogeneous g(a,b,c,d) so that
    `f(a,b) == g(a,b,1,1) (mod (a-b)**2)` and `g(a,b,c,d)` is D4-symmetric
    (assuming g exists).

    The conversion is not unique so we only return one of them
    by employing a linear-time algorithm.
    """
    q = {}
    for (n, m), v in poly.rep.terms():
        if n < m:
            continue
        d_rem = degree - n - m
        if d_rem * 2 > degree:
            continue
        if n == m and d_rem == n + m:
            # this should be handled separately
            continue
        if d_rem % 2 != 0:
            v = v/2
        q[(n, m, (d_rem+1)//2, d_rem//2)] = v

    D4 = [[0, 1, 2, 3],
        [1, 0, 2, 3],
        [2, 3, 0, 1],
        [3, 2, 0, 1],
        [0, 1, 3, 2],
        [1, 0, 3, 2],
        [2, 3, 1, 0],
        [3, 2, 1, 0]]
    def mp(k, a):
        return tuple(k[i] for i in a)

    for a in D4:
        q.update({mp(k, a): v for k, v in q.items()})

    if degree % 4 == 0:
        # determine the coeff of (d/4, d/4, d/4, d/4) by the relation
        # poly(1,1) == q(1,1,1,1)
        const = sum(poly.rep.coeffs()) if not poly.is_zero else poly.domain.zero
        const2 = sum(q.values()) if q else poly.domain.zero
        q[(degree//4,)*4] = const - const2

    q2 = Poly(q, gens, domain=poly.domain)
    return q2


def quaternary_dense_symmetric(coeff: Coeff, real=True):
    if coeff.total_degree() <= 4:
        # should not be handled here
        return None
    return _quaternary_dense_symmetric(coeff, real=real)

def _quaternary_dense_symmetric(coeff: Coeff, real=True):
    return _quaternary_dense_symmetric_vanish2(coeff)


def _quaternary_dense_symmetric_vanish2(coeff: Coeff):
    """
    Solve symmetric quaternary inequalities with
    `(a-1)**2*(b-1)**2 | F(a,b,1,1)`.

    Examples
    --------
    :: sym = "sym"

    => s((a+b-c-d)2(a-c)(a-d)(b-c)(b-d))

    => s(a4b(a-c)(a-d)(b-c)(b-d)) # doctest:+SKIP

    => s(a4bd(27ac2+acd+bc2+5bcd+2bd2)(a-c)(a-d)(b-c)(b-d))

    => 4s(ab(c-d)2((a+b)2-(a-b)2))s(ab(c-d)2(a+b-c-d)2)-4s(ab(a+b)(c-d)2(a+b-c-d))2
    """
    poly = coeff.as_poly()
    a, b, c, d = poly.gens
    per = lambda _: Poly(_, poly.gens, domain = poly.domain)
    margin = poly.eval((1,1))
    sym, rem = margin.div(((c-1)*(d-1)).as_poly(c,d, domain=poly.domain)**2)
    if not rem.is_zero:
        return None
    wrap = coeff.wrap

    # if sym.div((c-d).as_poly(c,d,domain=poly.domain)**2)[1].is_zero:
    #     return _quaternary_dense_symmetric_vanish_xyzz(coeff)

    dih = _lift_sym_axis_to_D4(sym, (a,b,c,d), poly.total_degree() - 4)
    alt_proj, alt_rem = (sym - dih.eval((1,1))).div((c-d).as_poly(c,d,domain=poly.domain)**2)
    if not alt_rem.is_zero:
        # not expected to happen
        return None

    # print(f'dih = {dih}\nalt_proj = {alt_proj}')
    if any(wrap(_) < 0 for _ in alt_proj.rep.coeffs()):
        return None

    from .solver import _structural_sos_4vars_dihedral
    dih_sol = _structural_sos_4vars_dihedral(dih)
    if dih_sol is None:
        return None

    n = poly.total_degree() - 6
    alt = per({((n-(i+j))//2, (n-(i+j)+1)//2, i, j): v/2
        for (i,j), v in alt_proj.rep.terms()})
    alt = alt + per({(m[1], m[0], m[2], m[3]): v
        for m, v in alt.rep.terms()})

    main_poly = _sym_sum_poly(
        per((a-c)*(b-c)*(a-d)*(b-d))**2 * (alt*(per(c-d)**2)*4 + dih*2)
    )

    rest = (poly * _sym_sum_poly(per((a-b)*(c-d))**2) - main_poly)
    rest, rem = rest.div(per((a-b)*(a-c)*(b-c)*(a-d)*(b-d)*(c-d))**2)
    if not rem.is_zero:
        # not expected to happen
        return None
    # print(f'rest = {rest}\nrem = {rem}')
    if any(wrap(_) < 0 for _ in rest.rep.coeffs()):
        return None

    SymmetricSum = coeff.symmetric_sum
    const, rest2 = poly_reduce_by_symmetry(rest, "sym").primitive()
    rest_expr = const * SymmetricSum(rest2.as_expr())
    disc = ((a - c)**2*(b - c)**2*(a - d)**2*(b - d)**2*(c - d)**2*(a - b)**2)

    const, alt = alt.primitive()
    main = 4*const*SymmetricSum((a - c)**2*(b - c)**2*(a - d)**2*(b - d)**2*(c - d)**2*alt.as_expr())\
        + 2*SymmetricSum((a - c)**2*(a - d)**2*(b - c)**2*(b - d)**2*dih_sol)
    sol = (rest_expr*disc + main)/SymmetricSum((a-b)**2*(c-d)**2)
    return sol


#####################################################################
#
#                              Acyclic
#
#####################################################################

@sos_struct_reorder_symmetry(groups=(2, 2))
def quaternary_dense_dihedral(coeff: Coeff):
    return _quaternary_dense_dihedral_by_level(coeff)

def _quaternary_dense_dihedral_by_level(coeff: Coeff):
    """
    Solve a 4-var homogeneous dihedral inequality
    by writing it as `Σ f(a,b)*c**r*d**s` where
    `f(a,b)` is symmetric with respect to a, b.
    """
    a, b, c, d = coeff.gens
    dih = PermutationGroup(Permutation([2,3,1,0]), Permutation([1,0,2,3]))
    reduced = poly_reduce_by_symmetry(coeff.as_poly(), dih)

    wrap = coeff.wrap
    is_nng_cfs = all(wrap(_) >= 0 for _ in coeff.coeffs())
    def _nng_fallback():
        """
        When it is nonnegative over real numbers but has
        nonnegative coefficients, fall back to solver for
        non-negative coefficients, which generates neater solutions.
        """
        return CyclicSum(reduced.as_expr(), coeff.gens, dih)

    # marginalize over c, d
    margin = reduced.eject(a, b)
    levels = margin.rep.terms()

    if is_nng_cfs and any(r % 2 != 0 or s % 2 != 0 for (r, s), cfs in levels):
        # has odd-degree terms -> fall back to solver for non-negative coefficients
        return _nng_fallback()

    level_proof = {}
    def hom_proof(proof, gen: Symbol, degree: int):
        """
        Restore the solution from the homogenized proof.
        """
        lst = []
        for i, (extra, source) in enumerate(proof):
            for c, v in source:
                if v.is_zero:
                    continue
                v = v.homogenize(gen)
                codegree = degree - 2*v.total_degree() - i
                lst.append(c * extra * v.as_expr()**2 * gen**(codegree))
        return Add(*lst)

    degree = coeff.total_degree()
    for (r, s), cfs in levels:
        # compute q = f(a,b) + f(b,a)
        q = Poly(dict(cfs), a, b, domain=coeff.domain)
        q = q + Poly({(m, n): v for (n, m), v in cfs.items()},
                     a, b, domain=coeff.domain)
        q = q.eval((1,)).mul_ground(q.domain.one/2)

        # print(r, s,  Poly(dict(cfs), coeff.gens[:2], domain=coeff.domain))
        proof = prove_univariate(q, return_type='list')
        if proof is None:
            # prove on R+
            if is_nng_cfs:
                return _nng_fallback()
            proof = prove_univariate(q, (0, None), return_type='list')
        if proof is None:
            return None
        level_proof[(r, s)] = hom_proof(proof, coeff.gens[0], degree - r - s)

    sol = Add(*[
        CyclicSum(c**r*d**s*proof, coeff.gens, dih)
            for (r, s), proof in level_proof.items()
    ])
    return sol
