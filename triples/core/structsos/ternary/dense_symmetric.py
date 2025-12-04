from typing import Tuple, List

import sympy as sp

from .utils import (
    CyclicSum, CyclicProduct, Coeff, radsimp, SS,
    sos_struct_handle_uncentered
)
from ..univariate import prove_univariate

a, b, c = sp.symbols('a b c')

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
        methods.append(SS.structsos.ternary._structural_sos_3vars_cyclic)
    else:
        methods.append(sos_struct_liftfree_for_six)
        methods.append(_sos_struct_lift_for_six)

    for method in methods:
        solution = method(coeff, real=real)
        if solution is not None:
            return solution


def _sos_struct_trivial_additive(coeff, real=True):
    """
    Solve trivial cyclic inequalities with nonnegative coefficients.
    """
    if any(_ < 0 for _ in coeff.values()):
        return None
    exprs = []
    for (i,j,k), v in coeff.items():
        if (i > j and i > k) or (i == j and i > k):
            exprs.append(v * a**i * b**j * c**k)

    d = coeff.total_degree()
    if d % 3 == 0:
        d = d // 3
        v = coeff((d,d,d))
        if v != 0:
            exprs.append(v/3 * a**d * b**d * c**d)

    return CyclicSum(sp.Add(*exprs))




def sym_axis(coeff: Coeff, d: int = -1) -> sp.Poly:
    """Compute f(a,1,1)."""
    if d == -1: d = coeff.total_degree()
    coeff_list = [0] * (d+1)
    for m, v in coeff.items():
        coeff_list[m[0]] += v
    return sp.Poly(radsimp(coeff_list)[::-1], a)

def _homogenize_sym_axis(sym: sp.Poly, d: int, abc: Tuple[sp.Symbol] = None) -> sp.Expr:
    """Homogenize f(a,1,1) to f(a,b,c) given degree."""
    if abc is None: abc = sp.symbols('a b c')
    a, b, c = abc
    s = [0]
    for (m,), v in sym.terms():
        k, r = divmod(d-m, 2)
        if r == 0:
            s.append(v * a**m * b**k * c**k)
        else:
            s.append(v/2 * a**m * b**(k+1) * c**k)
            s.append(v/2 * a**m * b**k * c**(k+1))
    return sp.Add(*s)

def _homogenize_sym_proof(sym_proof, d: int, abc: Tuple[sp.Symbol] = None) -> sp.Expr:
    """Homogenize the result from prove_univariate."""
    if abc is None: abc = sp.symbols('a b c')
    a, b, c = abc
    exprs = []
    for i in range(len(sym_proof)):
        leading = sym_proof[i][0]
        ld = leading.as_poly(a).degree()
        part_expr = []
        for k, v in sym_proof[i][1]:
            v2 = _homogenize_sym_axis(v, v.degree(), abc)
            rd = (d - ld - v.degree()*2) // 2
            part_expr.append(k * v2**2 * b**rd * c**rd)
        part_expr = sp.Add(*part_expr)
        if (d - ld) % 2 == 1:
            part_expr = part_expr * leading * (b + c) / 2
        else:
            part_expr = part_expr * leading
        exprs.append(part_expr)
    return sp.Add(*exprs)


@sos_struct_handle_uncentered
def _sos_struct_lift_for_six(coeff, real=True):
    """
    Solve high-degree (dense) symmetric inequalities by the method
    of lifting the degree for six. Idea: define f(a,1,1) to be the
    symmetric axis of the polynomial. If two tenary homogeneous symmetric polynomials
    have equal symmetric axis, then their difference is a multiple of (a-b)^2*(b-c)^2*(c-a)^2.

    We can subtract a nonnegative polynomial with equal symmetric axis and then
    divide the rest by (a-b)^2*(b-c)^2*(c-a)^2 to reduce the degree of the problem.
    We can repeat this process until the degree is small enough.

    Examples
    ---------
    s(a9)s(a3)+9p(a4)-6a3b3c3s(a3)

    s(a12-3a8b2c2-a6b6+2a5b5c2+a4b4c4)

    s(a2)s((s(ab)2+a2bc)(b2+ac)(c2+ab))-p(a2+bc)*3s(a3b+a3c+3a2bc)
    """
    d = coeff.total_degree()
    if d < 8:
        return None

    a, b, c = coeff.gens
    sym = sym_axis(coeff, d)
    div = sym.div(sp.Poly([1,-2,1], a))
    if not div[1].is_zero:
        return None
    if div[0].LC() < 0 or div[0](0) < 0 or div[0](1) < 0:
        return None

    # if any(_ < 0 for _ in div[0].coeffs()):
    #     # raise NotImplementedError
    #     return None

    if all(_ >= 0 for _ in div[0].coeffs()):
        lifted_sym = _homogenize_sym_axis(div[0], d - 2, (a, b, c))
    else:
        sym_proof = prove_univariate(div[0], (0, None), return_type='list')
        if sym_proof is None:
            return None
        lifted_sym = _homogenize_sym_proof(sym_proof, d - 2, (a, b, c))
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


@sos_struct_handle_uncentered
def sos_struct_liftfree_for_six(coeff, real=True):
    """
    Solve high-degree (dense) symmetric inequalities without
    lifting the degree. This will be tried in prior because
    it avoids lifting the degree of the problem.

    The idea is to first subtract something like:

        s(a^n*b^m*c^m*(a-b)*(a-c)*(u*a + v*(b+c)/2)) + p(a-b)^2*...

    so that the symmetric axis of the remaining part is a multiple of (a-1)^4.
    Then the rest part can be seen as:

        s(...*(a-b)^2*(a-c)^2) + p(a-b)^2*...

    which would perhaps be trivially nonnegative if successful.

    Examples
    ---------
    s(a9)s(a3)+9p(a4)-6a3b3c3s(a3)

    s(a2)s((s(ab)2+a2bc)(b2+ac)(c2+ab))-p(a2+bc)*3s(a3b+a3c+3a2bc)
    """
    d = coeff.total_degree()
    if d < 6:
        return None

    a, b, c = coeff.gens
    sym = sym_axis(coeff, d)
    div = sym.div(sp.Poly([1,-2,1], a))
    if not div[1].is_zero:
        return None

    div2 = div[0].div(sp.Poly([1,-2,1], a))
    if div2[1].is_zero:
        return _sos_struct_liftfree_for_six_ord4(coeff, div2[0], real=real)

    # subtract some s(a^n*b^m*c^m*(a-b)*(a-c)*(a + (b+c))) + p(a-b)^2*...
    # so that the symmetric axis of the remaining part is a multiple of (a-1)^4
    u, v = div2[1].coeff_monomial((1,)), div2[1].coeff_monomial((0,))

    # note that a^n = (-n*a + (n+1))  (mod (a-1)^2)
    # we find integer n such that (u*a + v)*(-n*a + (n+1)) + nu(a-1)^2 >= 0
    # it is equivalent to -v/(u+v) <= n <= u/(u+v)
    if u + v <= 0 or u < 0:
        return None
    expected = (d+1)//3 - 1
    if -v <= expected * (u+v) <= u:
        n = expected
    elif expected * (u+v) > u:
        n = u // (u+v)
    else: # expected < -v/(u+v) and v < 0
        n = (-v - 1)//(u+v) + 1
    u2, v2 = u - (u+v)*n, v + (u+v)*n
    if not (n <= d - 3 and n >= 0 and u2 >= 0 and v2 >= 0):
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
    rem_poly = coeff.as_poly((a,b,c)) - subtractor.doit().as_poly((a,b,c))
    solution = _sos_struct_liftfree_for_six_ord4(Coeff(rem_poly), real=real)
    if solution is not None:
        return subtractor + solution


def _sos_struct_liftfree_for_six_ord4(coeff, div2=None, real=True):
    """
    Solve a high-degree (dense) symmetric inequality where (a-1)^4 is a factor
    of the symmetric axis. Such polynomial can be seen as:

        s(...*(a-b)^2*(a-c)^2) + p(a-b)^2*...

    which would perhaps be trivially nonnegative if successful.
    """
    a, b, c = coeff.gens
    if div2 is None:
        sym = sym_axis(coeff, coeff.total_degree())
        div2 = sym.div(sp.Poly([1,-4,6,-4,1], a))
        if not div2[1].is_zero:
            return None
        div2 = div2[0]

    if div2.LC() < 0 or div2(0) < 0 or div2(1) < 0:
        return None

    d = coeff.total_degree()
    if all(_ >= 0 for _ in div2.coeffs()):
        lifted_sym = _homogenize_sym_axis(div2, d - 4, (a, b, c))
    else:
        return None # not implemented

    poly = coeff.as_poly((a, b, c))
    diff = poly - CyclicSum((a-b)**2*(a-c)**2*lifted_sym).as_poly(a, b, c)
    diff = diff.div(CyclicProduct((a-b)**2).as_poly(a, b, c))[0]
    sol_diff = _sos_struct_dense_symmetric(Coeff(diff))
    if sol_diff is not None:
        return CyclicSum(lifted_sym * (a-b)**2*(a-c)**2) + CyclicProduct((a-b)**2) * sol_diff
