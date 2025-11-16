import sympy as sp

from .utils import Coeff, quadratic_weighting
from ...sdp import congruence
from ...utils import SymmetricSum
from ...utils.roots import common_region_of_conics

def sos_struct_nvars_quartic_symmetric(poly, real=True):
    """
    Solve a homogeneous quartic symmetric polynomial inequality on real numbers for nvars >= 4.
    """
    if poly.total_degree() == 4 and Coeff(poly).is_symmetric():
        return _sos_struct_nvars_quartic_symmetric_sdp(poly)


def _sos_struct_nvars_quartic_symmetric_sdp(poly):
    """
    Solve a quartic symmetric polynomial inequality on real numbers for nvars >= 4.
    This function does not check the degree & symmetry of the polynomial.

    Idea: if the polynomial is SOS, then it can be written as
    SymmetricSum((a-b)**2 * quad_form1) + quad_form2, where quad_form2 is a quadratic form
    with respect to SymmetricSum(a**2) and SymmetricSum(a*b). We perform symbolic SDP to find
    quad_form1 and quad_form2 to be PSD.
    """
    n = len(poly.gens)
    if n <= 3:
        return
    m = sp.factorial(n - 2)
    q1, q3, u = sp.Dummy('q1'), sp.Dummy('q3'), sp.Dummy('u')
    degrees = [[4],[3,1],[2,2],[2,1,1],[1,1,1,1]]
    degrees = [_ + [0]*(n - len(_)) for _ in degrees]
    c4 ,c3, c22, c21, c1 = [poly.coeff_monomial(_) for _ in degrees]

    cq = (c4/(n-1) + c3 + c22/2 + (n-2)*(c21/2 + (n-3)*c1/24))/n
    q2 = cq - q1/(n-1) - q3*(n-1)/4
    x = (c4 - q1)/(2*m*(n-1))
    cy = (2*q1 + q3 - c22)/4/m + x
    y = cy/2 + (n-2)*u/2
    v = (6*q3 - c1)/48/m
    cr = ((c3 - 2*q2)/(2*m) + 2*x - cy)/(2*(n-2))
    r = cr - u/2

    constraints = [
        u - v,
        u + (n - 3)*v,
        x + y,
        x - y,
        (u + (n - 3)*v) * (x + y) - 2*(n - 2)*r**2,
        q1 * q3 - q2**2,
    ]

    def _create_quad(x, y, u, v, r):
        mat = [[v for i in range(n)] for j in range(n)]
        mat[0][0] = mat[1][1] = x
        mat[1][0] = mat[0][1] = y
        for i in range(2,n):
            mat[0][i] = mat[1][i] = mat[i][1] = mat[i][0] = r
            mat[i][i] = u
        mat = sp.Matrix(mat)
        return mat

    def _get_solution(x, y, u, v, r):
        a, b = poly.gens[:2]
        sol = None
        # fallback to default (not expected to be used)
        cong = congruence(_create_quad(x, y, u, v, r))
        if cong is not None:
            vec = sp.Matrix(list(poly.gens))
            sol = sum(i * line**2 for i, line in zip(cong[1], cong[0]*vec)).together()
            sol = SymmetricSum((a-b)**2 * sol, poly.gens)
        return sol
    def _get_solution2(q1, q2, q3):
        a, b = poly.gens[:2]
        sol2 = quadratic_weighting(q1, q2*2, q3, a**2/(m*(n-1)), a*b/(m*2),
                mapping = lambda x,y: \
                    SymmetricSum((a**2/(m*(n-1))*x + a*b/(m*2)*y).together(), poly.gens)**2)
        return sol2

    u_candidates = [2*cr, (2*x-cy)/(n-2), v]

    # the following make constraints[-2]==0, but it is not linear but fractional
    # u_.candidates.append(-(((n-3)*v*(2*x+cy)-4*(n-2)*cr**2)/((2*x+cy)+4*(n-2)*cr+(n-2)*(n-3)*v)))

    for u_ in u_candidates:
        cons = [_.subs(u, u_).as_poly(q1, q3) for _ in constraints]
        q1q3 = common_region_of_conics(cons)
        if q1q3 is not None:
            q1_, q3_, u_ = q1q3[0], q1q3[1], u_.subs({q1: q1q3[0], q3: q1q3[1]})
            if not u_.is_finite:
                continue
            params = {u: u_, q1: q1_, q3: q3_}
            params = [_.subs(params) for _ in [x, y, u, v, r, q1, q2, q3]]
            sol = _get_solution(*params[:5])
            sol2 = _get_solution2(*params[5:])
            if sol is not None and sol2 is not None:
                return sol + sol2
        # print('Constraints =', [_.subs(params) for _ in constraints], 'PSD =', mat.is_positive_semidefinite)
    return None
