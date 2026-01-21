from sympy import Poly

from .utils import Coeff
from ....utils import poly_reduce_by_symmetry, arraylize_sp, invarraylize

def _sym_sum_poly(poly: Poly) -> Poly:
    """Return the symmetric sum of a polynomial efficiently."""
    rep = arraylize_sp(poly, expand_cyc=True, sym=True)
    return invarraylize(rep, poly.gens, degree=poly.total_degree(), sym=True)


def quaternary_dense_symmetric(coeff: Coeff, real=True):
    if coeff.total_degree() <= 4:
        # should not be handled here
        return None
    return _quaternary_dense_symmetric(coeff, real=real)

def _quaternary_dense_symmetric(coeff: Coeff, real=True):
    return _quarternary_dense_symmetric_vanish_xyzz(coeff)

def _quarternary_dense_symmetric_vanish_xyzz(coeff: Coeff):
    """
    Examples
    --------
    :: sym = "sym"

    => 4s(ab(c-d)2((a+b)2-(a-b)2))s(ab(c-d)2(a+b-c-d)2)-4s(ab(a+b)(c-d)2(a+b-c-d))2
    """
    poly = coeff.as_poly()
    a, b, c, d = poly.gens
    per = lambda _: Poly(_, poly.gens, domain = poly.domain)

    margin = poly.eval((1,1))
    sym, rem = margin.div(((c-1)*(d-1)*(c-d)).as_poly(c,d,domain=poly.domain)**2)
    if not rem.is_zero:
        return None
    wrap = coeff.wrap
    if any(wrap(_) < 0 for _ in sym.rep.coeffs()):
        return None

    n = poly.total_degree() - 6
    lift = per({((n-(i+j))//2, (n-(i+j)+1)//2, i, j): v/2
        for (i,j), v in sym.rep.terms()})
    lift = lift + per({(m[1], m[0], m[2], m[3]): v
        for m, v in lift.rep.terms()})

    main_poly = _sym_sum_poly(per((a-c)*(b-c)*(a-d)*(b-d)*(c-d))**2 * lift).mul_ground(4)

    rest = (poly * _sym_sum_poly(per((a-b)*(c-d))**2) - main_poly)
    rest, rem = rest.div(per((a-b)*(a-c)*(b-c)*(a-d)*(b-d)*(c-d))**2)
    if not rem.is_zero:
        return None
    if any(wrap(_) < 0 for _ in rest.rep.coeffs()):
        return None

    SymmetricSum = coeff.symmetric_sum
    const, rest2 = poly_reduce_by_symmetry(rest, "sym").primitive()
    rest_expr = const * SymmetricSum(rest2.as_expr())
    disc = ((a - c)**2*(b - c)**2*(a - d)**2*(b - d)**2*(c - d)**2*(a - b)**2)

    const, lift = lift.primitive()
    main = 4*const*SymmetricSum((a - c)**2*(b - c)**2*(a - d)**2*(b - d)**2*(c - d)**2*lift.as_expr())
    sol = (rest_expr*disc + main)/SymmetricSum((a-b)**2*(c-d)**2)
    return sol
