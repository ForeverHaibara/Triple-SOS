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


def _lift_sym_axis_to_D4_part1(
    poly: Poly,
    gens: Tuple[Symbol, Symbol],
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

    if degree % 2 == 0:
        # determine the coeff of (r, s, r, s) by the relation
        # poly(1,1) == q(1,1,1,1)
        r, s = (degree + 2)//4, degree//4
        const = sum(poly.rep.coeffs()) if not poly.is_zero else poly.domain.zero
        q[(r, s, r, s)] = poly.domain.zero
        q[(s, r, s, r)] = poly.domain.zero
        const2 = sum(q.values()) if q else poly.domain.zero
        q[(r, s, r, s)] = (const - const2)/2
        q[(s, r, s, r)] += (const - const2)/2


    q2 = Poly(q, poly.gens + gens, domain=poly.domain)
    return q2

def _lift_sym_axis_to_D4(
    poly: Poly,
    gens: Tuple[Symbol, Symbol],
    degree: int,
    mod_poly: Optional[Poly] = None
):
    """
    Given a nonhomogeneous f(a,b) such that `f(a,b) == f(b,a)`,
    compute a homogeneous g(a,b,c,d) so that
    `f(a,b) == g(a,b,1,1)` and `g(a,b,c,d)` is D4-symmetric
    (assuming g exists).

    The conversion is not unique so we only return one of them
    by employing a linear-time algorithm.
    """
    if mod_poly is None:
        mod_poly = _lift_sym_axis_to_D4_part1(poly, gens, degree)

    a, b = poly.gens
    c, d = gens
    rest, _ = (poly - mod_poly.eval(3,1).eval(2,1)).div((a-b).as_poly(a,b)**2)

    dt = {}
    for (m, n), v in rest.rep.terms():
        if m < n:
            continue
        r, s = (degree - m - n + 1 - 2)//2, (degree - m - n - 2)//2
        if r != s:
            v = v/2
        dt[(m, n, r, s)] = v
        dt[(n, m, r, s)] = v
        dt[(m, n, s, r)] = v
        dt[(n, m, s, r)] = v

    all_gens = (a, b, c, d)
    rest1 = (a-b).as_poly(all_gens)**2 * Poly(dt, all_gens, domain=poly.domain)
    dt2 = {(r, s, m, n): v for (m, n, r, s), v in dt.items()}
    rest2 = (c-d).as_poly(all_gens)**2 * Poly(dt2, all_gens, domain=poly.domain)

    return mod_poly + rest1 + rest2


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
    sol = _quaternary_dense_symmetric_vanish2_liftfree(coeff)
    if sol is not None:
        return sol

    dih = _lift_sym_axis_to_D4_part1(sym, (a, b), poly.total_degree() - 4)
    dih.gens = (a, b, c, d)
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


def _quaternary_dense_symmetric_vanish2_liftfree(coeff: Coeff, sym=None):
    """
    Solve symmetric quaternary inequalities with
    `(a-1)**2*(b-1)**2 | F(a,b,1,1)`.
    without lifting the degree. This is done by noting
    that
    `(d-a)*(d-c)*(b-a)*(b-c) + (c-a)*(c-d)*(b-a)*(b-d) = (a-b)**2*(c-d)**2`


    Examples
    --------
    :: sym = "sym"

    => s(1/4a4b4-a4b3c+1/2a4b2c2+a3b3cd-a3b2c2d+1/4a2b2c2d2)

    => s(a2b3(a-b)2(c-d)2)

    => s(a6b4c2+a6b4cd-a6b3c3-a6b2c2d2+a5b5c2+a5b5cd-4a5b4c2d-2a5b3c3d+2a5b3c2d2-a4b4c4+2a4b4c3d+a4b4c2d2-a4b3c3d2+a3b3c3d3)
    """
    a, b, c, d = coeff.gens
    per = lambda _: Poly(_, coeff.gens, domain = coeff.domain)
    if sym is None:
        poly = coeff.as_poly()
        per = lambda _: Poly(_, poly.gens, domain = poly.domain)
        margin = poly.eval((1,1))
        sym, rem = margin.div(((c-1)*(d-1)).as_poly(c,d, domain=poly.domain)**2)
        if not rem.is_zero:
            return None
    wrap = coeff.wrap

    # F = SymmetricSum(D(a,b,c,d)*(a-c)*(a-d)*(b-c)*(b-d))/8 (mod discriminant)
    dih = _lift_sym_axis_to_D4(sym, (a, b), coeff.total_degree() - 4)
    dih.gens = (a, b, c, d)
    dih_c = Poly({(r, n, m, s): v for (m, n, r, s), v in dih.rep.terms()},
                a, b, c, d, domain=dih.domain)
    dih_d = Poly({(s, n, r, m): v for (m, n, r, s), v in dih.rep.terms()},
                a, b, c, d, domain=dih.domain)
    # F = SymmetricSum(D(a,b,c,d)*(a-b)**2*(c-d)**2)/16 (mod discriminant)
    dih = dih_c + dih_d - dih

    from .solver import _structural_sos_4vars_dihedral
    dih_sol = _structural_sos_4vars_dihedral(dih)
    if dih_sol is None:
        return None

    main_poly = _sym_sum_poly(dih * per((a-b)*(c-d))**2)

    rest = (coeff.as_poly()*16 - main_poly)
    rest, rem = rest.div(per((a-b)*(a-c)*(b-c)*(a-d)*(b-d)*(c-d))**2)

    if not rem.is_zero:
        # not expected to happen
        return None
    # print(f'rest = {rest}\nrem = {rem}')
    if any(wrap(_) < 0 for _ in rest.rep.coeffs()):
        return None

    disc = 0
    if not rest.is_zero:
        disc = ((a - c)**2*(b - c)**2*(a - d)**2*(b - d)**2*(c - d)**2*(a - b)**2)

    SymmetricSum = coeff.symmetric_sum
    const, rest2 = poly_reduce_by_symmetry(rest, "sym").primitive()
    rest_expr = const * SymmetricSum(rest2.as_expr())
    return SymmetricSum(dih_sol * (a-b)**2*(c-d)**2)/16 + disc * rest_expr/16

#####################################################################
#
#                            Asymmetric
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

    Examples
    --------
    => (a+c)(b+d)(ac+bd)((s(ab)+ac+bd))-3abcds(a)2 # doctest:+SKIP
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
