from typing import Callable, List, Optional, Union, Dict

import sympy as sp

from .utils import Coeff, PolynomialUnsolvableError, PolynomialNonpositiveError
from ...utils import CyclicSum, CyclicProduct, congruence

def _null_solver(*args, **kwargs):
    return None

def _get_defaulted_gens(poly: Union[sp.Poly,Coeff]):
    if isinstance(poly, sp.Poly):
        return poly.gens
    return sp.symbols(f"a:{chr(96 + poly.nvars)}") if len(poly.coeffs) > 0 else []

def sos_struct_extract_factors(poly: Union[sp.Poly, Coeff], solver: Callable, real: bool = True, **kwargs):
    """
    Wrap a solver to handle factorizable polynomials in advance.

    Cases.
    + If the polynomial is a*b*c * f(a,b,c), it first solves f(a,b,c) and then multiplies the result by a*b*c.
    + If the polynomial is f(a^k, b^k, c^k), it first solves f(a,b,c) and then replaces a,b,c with a^k,b^k,c^k.
    """
    coeff = poly if isinstance(poly, Coeff) else Coeff(poly)
    symbols = _get_defaulted_gens(poly)

    def coeff_to_poly(new_coeff):
        if isinstance(poly, sp.Poly):
            return new_coeff.as_poly(*symbols)
        return new_coeff

    monoms, new_coeff = coeff.cancel_abc()
    if any(i > 0 for i in monoms):
        new_coeff = coeff_to_poly(new_coeff)
        solution = solver(new_coeff, real = real and all(i % 2 == 0 for i in monoms), **kwargs)
        if solution is not None:
            if coeff.nvars == 3 and all(i == monoms[0] for i in monoms):
                multiplier = CyclicProduct(symbols[0]**monoms[0], symbols)
            else:
                multiplier = sp.Mul(*[s**i for s, i in zip(symbols, monoms)])
            return multiplier * solution
        return None

    i, new_coeff = coeff.cancel_k()
    if i > 1:
        new_coeff = coeff_to_poly(new_coeff)
        solution = solver(new_coeff, real = False if (i % 2 == 0) else real, **kwargs)
        if solution is not None:
            solution = solution.xreplace(dict((s, s**i) for s in symbols))
            return solution
        return None

    return solver(poly, **kwargs)


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
    gens = _get_defaulted_gens(poly)
    genvec = sp.Matrix(list(gens) + [1])
    return sum(S[i] * (U[i, :] * genvec)[0,0]**2 for i in range(nvars + 1))


def sos_struct_common(poly: Union[sp.Poly, Coeff], *solvers, **kwargs):
    """
    A method wrapper for multiple solvers.
    """
    def _wrapped_solver(poly, **kwargs):
        solution = None
        for solver in solvers:
            if solver is None:
                continue
            try:
                solution = solver(poly, **kwargs)
                if solution is not None:
                    break
            
            except PolynomialUnsolvableError as e:
                # When we are sure that the polynomial is nonpositive,
                # we can return None directly.
                if isinstance(e, PolynomialNonpositiveError):
                    return None
        return solution
    
    # cancel abc
    
    return sos_struct_extract_factors(poly, _wrapped_solver, **kwargs)
    


def sos_struct_degree_specified_solver(solvers: Dict[int, Callable], homogeneous: bool = False) -> Callable:
    """
    A method wrapper for structural SOS with degree specified solvers.

    When the degree <= 2 and the solver is not provided, it uses the default solvers.
    """
    def _sos_struct_degree_specified_solver(poly: Union[sp.Poly, Coeff], *args, **kwargs):
        if homogeneous and isinstance(poly, Coeff):
            degree = poly.degree()
        else:
            degree = poly.total_degree()
        solver = solvers.get(degree, None)
        if solver is None:
            if degree == 1:
                solver = sos_struct_linear
            elif degree == 2:
                solver = sos_struct_quadratic
            else:
                solver = _null_solver
        return solver(poly, *args, **kwargs)
    return _sos_struct_degree_specified_solver