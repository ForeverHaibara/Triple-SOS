from ....utils import Coeff, CyclicSum

def sos_struct_nvars_linear(coeff: Coeff, **kwargs):
    """
    Solve a linear inequality. Supports non-homogeneous polynomials also.

    Examples
    --------
    :: ineqs = [a,b,c,d,e], gens = [a,b,c,d,e]

    => 5a + 2b + 3(c - b)/4 + 4e

    => 4(a+b+c+d+e)/7

    => (3 - 2sqrt(2))(a - b) + 2b + sqrt(2)c/2

    => 1/sqrt(3) + (b - c - 1)/4 + 3(c + d)

    => 4a + 4c + 4d
    """
    d = coeff.total_degree()
    if d > 1 or not coeff.domain.is_Numerical:
        return None

    n = len(coeff.gens)
    constant = coeff((0,)*n)
    if d == 0 and constant >= 0:
        return constant

    # d == 1
    wrap = coeff.wrap
    if not all(wrap(c) >= 0 for c in coeff.coeffs()):
        return None

    # explore the symmetry
    common_coeff = None
    monom = [0] * n
    for i in range(n):
        monom[i] = 1
        v = coeff(tuple(monom))
        if common_coeff is not None and v != common_coeff:
            # not symmetric
            break
        common_coeff = v
        monom[i] = 0
    else:
        # the polynomial is symmetric
        gens = coeff.gens
        return common_coeff * CyclicSum(gens[0], gens) + constant

    return coeff.as_poly().as_expr()
