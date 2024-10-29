import sympy as sp

from .utils import (
    CyclicSum, CyclicProduct, Coeff, radsimp, prove_univariate, SS,
    sos_struct_handle_uncentered
)

a, b, c = sp.symbols('a b c')

def sos_struct_dense_symmetric(coeff, real=True):
    """
    Solve dense 3-var symmetric inequalities.
    Triggered only when the degree is at least 8 and the polynomial is symmetric.
    """
    if coeff.degree() < 8 or not coeff.is_symmetric():
        return None
    return _sos_struct_dense_symmetric(coeff, real)




def _sos_struct_dense_symmetric(coeff, real=True):
    """
    Solve dense 3-var symmetric inequalities. This function
    does not check the symmetry of the input polynomial, so it
    should only be called when the symmetry is guaranteed.
    """
    d = coeff.degree()
    methods = [_sos_struct_trivial_additive]
    if d < 8:
        methods.append(SS.structsos.ternary._structural_sos_3vars_cyclic)
    else:
        methods.append(_sos_struct_lift_for_six)

    for method in methods:
        solution = method(coeff, real=real)
        if solution is not None:
            return solution


def _sos_struct_trivial_additive(coeff, real=True):
    """
    Solve trivial cyclic inequalities with nonnegative coefficients.
    """
    if any(_ < 0 for _ in coeff.coeffs.values()):
        return None
    exprs = []
    for (i,j,k), v in coeff.coeffs.items():
        if (i > j and i > k) or (i == j and i > k):
            exprs.append(v * a**i * b**j * c**k)

    d = coeff.degree()
    if d % 3 == 0:
        d = d // 3
        v = coeff((d,d,d))
        if v != 0:
            exprs.append(v/3 * a**d * b**d * c**d)

    return CyclicSum(sp.Add(*exprs))



def _homogenize_sym_axis(sym: sp.Poly, d: int) -> sp.Expr:
    # homogenize f(a,1,1) to f(a,b,c)
    s = [0]
    for (m,), v in sym.terms():
        k, r = divmod(d-m, 2)
        if r == 0:
            s.append(v * a**m * b**k * c**k)
        else:
            s.append(v/2 * a**m * b**(k+1) * c**k)
            s.append(v/2 * a**m * b**k * c**(k+1))
    return sp.Add(*s)

@sos_struct_handle_uncentered
def _sos_struct_lift_for_six(coeff, real=True):
    """
    Solve high-degree (dense) symmetric inequalities by the method
    of lifting the degree for six.

    Examples
    ---------
    s(a9)s(a3)+9p(a4)-6a3b3c3s(a3)

    s(a12-3a8b2c2-a6b6+2a5b5c2+a4b4c4)

    s(a2)s((s(ab)2+a2bc)(b2+ac)(c2+ab))-p(a2+bc)*3s(a3b+a3c+3a2bc)
    """
    d = coeff.degree()
    if d < 8:
        return None

    a, b, c = sp.symbols('a b c')
    def sym_axis(coeff):
        # compute f(a,1,1)
        coeff_list = [0] * (d+1)
        for m, v in coeff.coeffs.items():
            coeff_list[m[0]] += v
        return sp.Poly(radsimp(coeff_list)[::-1], a)

    sym = sym_axis(coeff)
    div = sym.div(sp.Poly([1,-2,1], a))
    if not div[1].is_zero:
        return None

    # if any(_ < 0 for _ in div[0].coeffs()):
    #     # raise NotImplementedError
    #     return None

    if all(_ >= 0 for _ in div[0].coeffs()):
        lifted_sym = _homogenize_sym_axis(div[0], d - 2)
    else:
        sym_proof = prove_univariate(div[0], return_raw=True)
        if sym_proof is None:
            return None
        exprs = []
        for i in range(len(sym_proof)):
            leading = sym_proof[i][0]
            ld = leading.as_poly(a).degree()
            part_expr = []
            for k, v in zip(sym_proof[i][1], sym_proof[i][2]):
                v2 = _homogenize_sym_axis(v, v.degree())
                rd = (d - 2 - ld - v.degree()*2) // 2
                part_expr.append(k * v2**2 * b**rd * c**rd)
            part_expr = sp.Add(*part_expr)
            if (d - 2 - ld) % 2 == 1:
                part_expr = part_expr * leading * (b + c) / 2
            else:
                part_expr = part_expr * leading
            exprs.append(part_expr)
        lifted_sym = sp.Add(*exprs)
        # print(lifted_sym)


    def compute_diff(coeff, sym2, mul, tail) -> Coeff:
        poly = coeff.as_poly((a, b, c)) * mul.as_poly(a, b, c)
        diff = poly - CyclicSum(sym2 * tail).as_poly(a, b, c)
        diff = diff.div(CyclicProduct((a-b)**2).as_poly(a, b, c))
        return Coeff(diff[0])

    multipliers = [
        (CyclicSum((a-b)**2)/2, (a-b)**2*(a-c)**2)
        # CyclicSum(a*b*(a-b)**2)/2
    ]
    for mul, tail in multipliers:
        diff = compute_diff(coeff, lifted_sym, mul, tail)
        # print(diff.as_poly((a,b,c)).as_expr())
        sol_diff = _sos_struct_dense_symmetric(diff)
        if sol_diff is not None:
            return sp.Add(
                CyclicSum(lifted_sym * tail),
                CyclicProduct((a-b)**2) * sol_diff
            ) / mul