from typing import Tuple, List, Optional

import sympy as sp

from ...utils import deg, arraylize_sp

def _get_pqr(new_gens: Optional[List[sp.Symbol]] = None) -> Tuple[sp.Symbol, sp.Symbol, sp.Symbol]:
    """Return p,q,r from new_gens. If None, create new symbols."""
    if new_gens is not None:
        p, q, r = new_gens
    else:
        p, q, r = sp.symbols('p q r')
    return p, q, r


def _pqr_get_basis(degree: int) -> List[Tuple[Tuple[int, int, int], sp.Poly]]:
    """
    Return a list of ((i, j, k), p**i * q**j * r**k) for i+j+k == degree.
    """
    a, b, c = sp.symbols('a b c')
    p, q, r = [a+b+c, a*b+b*c+c*a, a*b*c]
    p, q, r = [_.as_poly() for _ in (p, q, r)]
    basis = []
    base = 1
    for k in range(degree // 3 + 1):
        base2 = base
        for j in range((degree - 3*k) // 2 + 1):
            i = degree - 3 * k - 2 * j
            if i < 0: break
            basis.append(((i, j, k), base2 * p ** i))
            base2 = q * base2
        base = r * base
    return basis

def pqr_coeffs_sym(poly: sp.Poly) -> List[Tuple[sp.Expr, Tuple[Tuple[int, int, int], sp.Poly]]]:
    """
    Helper function for pqr_cyc.
    When poly == f(p,q,r) + (a-b)*(b-c)*(c-a) * g(p,q,r)
    return the coefficients of f and g.
    """
    degree = deg(poly)
    pqr_basis = _pqr_get_basis(degree)
    
    coeffs = sp.Matrix([arraylize_sp(_[1]) for _ in pqr_basis])
    coeffs = coeffs.reshape(len(pqr_basis), coeffs.shape[0] // len(pqr_basis)).T
    array = arraylize_sp(poly)
    target = sp.Matrix(array).reshape(array.shape[0] * array.shape[1], 1)
    y = coeffs.LUsolve(target)
    return zip(y, pqr_basis)

def pqr_sym(poly: sp.Poly, new_gens: Optional[List[sp.Symbol]] = None) -> sp.Expr:
    """
    Express a 3-variable symmetric polynomial P in p,q,r form,
    such that P(p,q,r) = f(p,q,r) + (a-b)*(b-c)*(c-a) * g(p,q,r)
    where p = a+b+c, q = ab+bc+ca, r = abc.

    Parameters
    ----------
    poly : sp.Poly
        A 3-variable cyclic polynomial.
    new_gens : Optional[List[sp.Symbol]]
        If None, use p,q,r to stand for the new vars.
        Otherwise, use the given list of symbols.

    Returns
    ----------
    part_sym : sp.Expr
        The f(p,q,r) part.
    part_cyc : sp.Expr
        The g(p,q,r) part.
    """
    p, q, r = _get_pqr(new_gens)
    return sum(y * p**i * q**j * r**k for y, ((i, j, k), _) in pqr_coeffs_sym(poly))

def pqr_coeffs_cyc(poly: sp.Poly) -> Tuple[sp.Expr, sp.Expr]:
    """
    Helper function for pqr_cyc.
    When poly == f(p,q,r) + (a-b)*(b-c)*(c-a) * g(p,q,r)
    return the coefficients of f and g.
    """
    degree = deg(poly)
    if degree < 3:
        return pqr_coeffs_sym(poly, degree), tuple()

    pqr_basis = _pqr_get_basis(degree)
    pqr_basis2 = _pqr_get_basis(degree - 3)
    
    a, b, c = sp.symbols('a b c')
    d = ((a-b)*(b-c)*(c-a)).as_poly()

    m1, m2 = len(pqr_basis), len(pqr_basis2)
    coeffs = sp.Matrix([arraylize_sp(_[1]) for _ in pqr_basis]
                    + [arraylize_sp(_[1] * d) for _ in pqr_basis2])
    coeffs = coeffs.reshape(m1 + m2, coeffs.shape[0] // (m1 + m2)).T
    array = arraylize_sp(poly)
    target = sp.Matrix(array).reshape(array.shape[0] * array.shape[1], 1)
    y = coeffs.LUsolve(target)

    return zip(y[:m1], pqr_basis), zip(y[m1:], pqr_basis2)

def pqr_cyc(poly: sp.Poly, new_gens: Optional[List[sp.Symbol]] = None) -> Tuple[sp.Expr, sp.Expr]:
    """
    Express a 3-variable cyclic polynomial P in p,q,r form,
    such that P(p,q,r) = f(p,q,r) + (a-b)*(b-c)*(c-a) * g(p,q,r)
    where p = a+b+c, q = ab+bc+ca, r = abc.

    Parameters
    ----------
    poly : sp.Poly
        A 3-variable cyclic polynomial.
    new_gens : Optional[List[sp.Symbol]]
        If None, use p,q,r to stand for the new vars.
        Otherwise, use the given list of symbols.

    Returns
    ----------
    part_sym : sp.Expr
        The f(p,q,r) part.
    part_cyc : sp.Expr
        The g(p,q,r) part.
    """
    p, q, r = _get_pqr(new_gens)
    result = pqr_coeffs_cyc(poly)
    part_sym = sum(y * p**i * q**j * r**k for y, ((i, j, k), _) in result[0])
    part_cyc = sum(y * p**i * q**j * r**k for y, ((i, j, k), _) in result[1])
    return part_sym, part_cyc

def pqr_ker(new_gens: Optional[List[sp.Symbol]] = None) -> sp.Expr:
    """
    Return the pqr representation of ((a-b)*(b-c)*(c-a))^2.

    It should be -4*p**3*r + p**2*q**2 + 18*p*q*r - 4*q**3 - 27*r**2.
    """
    p, q, r = _get_pqr(new_gens)
    return -4*p**3*r + p**2*q**2 + 18*p*q*r - 4*q**3 - 27*r**2

def pqr_pqrt(a, b, c = sp.S(1)) -> Tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """
    Compute the p,q,r,t with p = 1 given a, b, c.
    """
    w = c + a + b
    q = (a*c+a*b+b*c) / w / w
    return sp.S(1), q, c*a*b/w/w/w, sp.sqrt(1-3*q)