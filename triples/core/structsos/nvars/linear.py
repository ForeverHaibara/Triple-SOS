from typing import Union

from sympy import Poly
from ....utils import Coeff, CyclicSum

def sos_struct_nvars_linear(poly: Union[Poly, Coeff], **kwargs):
    """
    Solve a linear inequality. Supports non-homogeneous polynomials also.
    """
    if isinstance(poly, Coeff):
        poly = poly.as_poly()
    d = poly.total_degree()
    if d > 1 or not poly.domain.is_Numerical:
        return None
    coeffs = poly.coeffs()
    if d == 0 and coeffs[0] >= 0:
        return coeffs[0]

    # d == 1
    if not all(i >= 0 for i in coeffs):
        return None

    # explore the symmetry
    common_coeff = None
    for gen in poly.gens:
        v = poly.coeff_monomial(gen)
        if common_coeff is not None and v != common_coeff:
            # not symmetric
            break
        common_coeff = v
    else:
        # the polynomial is symmetric
        constant = poly.coeff_monomial(1)
        return common_coeff * CyclicSum(poly.gens[0], poly.gens) + constant

    return poly.as_expr()
