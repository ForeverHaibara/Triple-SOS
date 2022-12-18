
import sympy as sp

from ...utils.text_process import deg
from ...utils.basis_generator import arraylize_sp, generate_expr

def _pqr_get_basis(degree):
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

def pqr_coeffs_sym(poly, degree = None, dict_monom = None, inv_monom = None):
    degree = degree or deg(poly)
    pqr_basis = _pqr_get_basis(degree)

    if dict_monom is None or inv_monom is None:
        dict_monom, inv_monom = generate_expr(degree)
    
    coeffs = sp.Matrix([arraylize_sp(_[1], dict_monom, inv_monom) for _ in pqr_basis])
    coeffs = coeffs.reshape(len(pqr_basis), coeffs.shape[0] // len(pqr_basis)).T
    target = sp.Matrix(arraylize_sp(poly, dict_monom, inv_monom)).reshape((len(inv_monom)), 1)
    y = coeffs.LUsolve(target)
    return zip(y, pqr_basis)

def pqr_sym(poly, **kwargs):
    p, q, r = sp.symbols('p q r')
    return sum(y * p**i * q**j * r**k for y, ((i, j, k), _) in pqr_coeffs_sym(poly, **kwargs))

def pqr_coeffs_cyc(poly, degree = None, dict_monom = None, inv_monom = None):
    """
    When poly == f(p,q,r) + (a-b)*(b-c)*(c-a) * g(p,q,r)
    return f and g
    """
    degree = degree or deg(poly)
    if degree < 3:
        return pqr_coeffs_sym(poly, degree, dict_monom, inv_monom), tuple()

    pqr_basis = _pqr_get_basis(degree)
    pqr_basis2 = _pqr_get_basis(degree - 3)

    if dict_monom is None or inv_monom is None:
        dict_monom, inv_monom = generate_expr(degree)
    
    a, b, c = sp.symbols('a b c')
    d = ((a-b)*(b-c)*(c-a)).as_poly()

    m1, m2 = len(pqr_basis), len(pqr_basis2)
    coeffs = sp.Matrix([arraylize_sp(_[1], dict_monom, inv_monom) for _ in pqr_basis]
                    + [arraylize_sp(_[1] * d, dict_monom, inv_monom) for _ in pqr_basis2])
    coeffs = coeffs.reshape(m1 + m2, coeffs.shape[0] // (m1 + m2)).T
    target = sp.Matrix(arraylize_sp(poly, dict_monom, inv_monom)).reshape((len(inv_monom)), 1)
    y = coeffs.LUsolve(target)

    return zip(y[:m1], pqr_basis), zip(y[m1:], pqr_basis2)

def pqr_cyc(poly, **kwargs):
    p, q, r = sp.symbols('p q r')
    result = pqr_coeffs_cyc(poly, **kwargs)
    part_sym = sum(y * p**i * q**j * r**k for y, ((i, j, k), _) in result[0])
    part_cyc = sum(y * p**i * q**j * r**k for y, ((i, j, k), _) in result[1])
    return part_sym, part_cyc

def pqr_ker():
    """
    Return the pqr representation of ((a-b)*(b-c)*(c-a))^2
    """
    p, q, r = sp.symbols('p q r')
    return -4*p**3*r + p**2*q**2 + 18*p*q*r - 4*q**3 - 27*r**2

def pqr_pqrt(a, b, c = sp.S(1)):
    """
    Compute the p,q,r,t with p = 1 given a, b, c.
    """
    w = c + a + b
    q = (a*c+a*b+b*c) / w / w
    return 1, q, c*a*b/w/w/w, sp.sqrt(1-3*q)