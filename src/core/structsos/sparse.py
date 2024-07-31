from typing import Callable, List, Optional, Union

import sympy as sp

from .utils import Coeff, PolynomialUnsolvableError
from ...utils import CyclicSum, CyclicProduct, congruence

def sos_struct_extract_factors(coeff: Coeff, recurrsion: Callable, symbols: Optional[List[sp.Symbol]] = None, real: bool = True, **kwargs):
    """
    Try solving the inequality by extracting factors and changing of variables.
    It handles cyclic and acyclic inequalities both.

    For instance,
    CyclicSum(a**2bc*(a-b)*(a-c)) is converted to CyclicSum(a*(a-b)*(a-c))
    by extracting CyclicProduct(a).
    CyclicSum(a**2*(a**2-b**2)*(a**2-c**2)) is converted to proving
    Cyclic(a*(a-b)*(a-c)) by making substitution a^2,b^2,c^2 -> a,b,c.
    """
    if symbols is None:
        symbols = sp.symbols(f"a:{chr(96 + coeff.nvars)}")

    monoms, new_coeff = coeff.cancel_abc()
    if any(i > 0 for i in monoms):
        solution = recurrsion(new_coeff, real = real and all(i % 2 == 0 for i in monoms), **kwargs)
        if solution is not None:
            if coeff.nvars == 3 and all(i == monoms[0] for i in monoms):
                multiplier = CyclicProduct(symbols[0]**monoms[0])
            else:
                multiplier = sp.Mul(*[s**i for s, i in zip(symbols, monoms)])
            return multiplier * solution
        raise PolynomialUnsolvableError

    i, new_coeff = coeff.cancel_k()
    if i > 1:
        solution = recurrsion(new_coeff, real = False if (i % 2 == 0) else real, **kwargs)
        if solution is not None:
            return solution.xreplace(dict((s, s**i) for s in symbols))
        raise PolynomialUnsolvableError
    

def sos_struct_linear(poly: sp.Poly):
    """
    Solve a linear inequality. Supports non-homogeneous polynomials also.
    """
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


def sos_struct_quadratic(poly: sp.Poly):
    """
    Solve a quadratic inequality on real numbers.
    """
    coeff = Coeff(poly)
    nvars = coeff.nvars
    mat = sp.zeros(nvars + 1)
    for k, v in coeff.coeffs.items():
        inds = []
        for i in range(nvars):
            if k[i] > 2:
                return None
            elif k[i] == 2:
                inds = (i, i)
                break
            elif k[i] == 1:
                if len(inds) == 2:
                    return None
                inds.append(i)
        if len(inds) == 1:
            inds.append(nvars)
        elif len(inds) == 0:
            inds = (nvars, nvars)
        if inds[0] == inds[1]:
            mat[inds[0], inds[0]] = v
        else:
            mat[inds[0], inds[1]] = v/2
            mat[inds[1], inds[0]] = v/2
    res = congruence(mat)
    if res is None:
        return None
    U, S = res
    genvec = sp.Matrix(list(poly.gens) + [1])
    return sum(S[i] * (U[i, :] * genvec)[0,0]**2 for i in range(nvars + 1))