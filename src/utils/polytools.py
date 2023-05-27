import sympy as sp

def deg(poly: sp.polys.Poly):
    """
    Return the degree of a polynomial. We shall assume the polynomial is homogeneous.

    Since the polynomial is homogeneous, we can simply return the sum of the exponents of the first monomial.
    This avoids calling `monoms` and significantly improves the speed.
    """
    rep = poly.rep.rep
    n = len(rep) + len(rep[0]) + len(rep[0][0]) - 3
    # only exception is poly == 0 (n = -1)
    return max(n, 0)

    # too slow:
    # return sum(poly.monoms()[0])



def verify(y, polys, poly, tol: float = 1e-10) -> bool:
    """
    Verify whether the fraction approximation is valid
    by substracting the partial sums and checking whether the remainder is zero.

    Deprecated.
    """
    try:
        for coeff, f in zip(y, polys):
            if coeff[0] != 0:
                if coeff[1] != -1:
                    if not isinstance(coeff[0], sp.Expr):
                        poly = poly - sp.Rational(coeff[0] , coeff[1]) * f
                    else:
                        v = coeff[0] / coeff[1]
                        coeff_dom = preprocessText_getdomain(str(v))
                        if coeff_dom != sp.QQ:
                            v = sp.polys.polytools.Poly(str(v)+'+a', domain=coeff_dom)\
                                 - sp.polys.polytools.Poly('a',domain=coeff_dom)
                        poly = poly - v * f
                else:
                    poly = poly - coeff[0] * f

        for coeff in poly.coeffs():
            # some coefficient is larger than tolerance, approximation failed
            if abs(coeff) > tol:
                return False
        return True
    except:
        return False


def verify_isstrict(func, root, tol=1e-9):
    """
    Verify whether a root is strict.

    Warning: Better get the function regularized beforehand.

    Deprecated.
    """
    return abs(func(root)) < tol


def verify_hom_cyclic(poly, fast=True):
    """
    Check whether a polynomial is homogenous and 3-var-cyclic

    Parameters
    ----------
    poly : sympy.Poly
        The polynomial to be checked.
    fast : bool
        Whether to use a faster algorithm. Default is True. It is currently in experimental stage. 
        The faster algorithm is based on direct operations on poly.rep.rep.

    Returns
    ----------
    is_hom : bool
        Whether the polynomial is homogenous.
    is_cyc : bool
        Whether the polynomial is 3-var-cyclic.
    """
    if fast:
        rep = poly.rep.rep
        len0, len1, len2 = len(rep), len(rep[0]), len(rep[0][0])
        n = len0 + len1 + len2 - 3
        if n < 1:
            return True, True
        
        t = n + 1
        coeffs = [0] * t**2

        for i in range(len(rep)):
            xi = rep[i]
            if len(xi):
                monom_i = len0 - i - 1
                v = monom_i * t + len(xi)
                rem = n - monom_i - len(xi)
                for j in range(len(xi)):
                    xj = xi[j]
                    v -= 1 # this is a pointer to `coeffs`
                    # monom_j = len(xi) - j - 1
                    rem += 1 # rem = n - monom_i - monom_j
                    if len(xj):
                        monom_k = len(xj) - 1
                        if monom_k != rem or any(xj[1:]):
                            return False, False
                        coeffs[v] = xj[0]

    else:
        # slow algorithm, abandoned
        monoms = poly.monoms()
        n = sum(monoms[0])

        if len(poly.gens) != 3:
            for monom in monoms:
                if sum(monom) != n:
                    return False, False
            return True, False

        t = n + 1
        coeffs = [0] * t**2 # for i in range(t**2)]
        for coeff, monom in zip(poly.coeffs(), monoms):
            if sum(monom) != n:
                return False, False
            coeffs[monom[0]*t + monom[1]] = coeff
        
    for i in range((n-1)//3+1, n+1):
        # 0 <= k = n-i-j <= i
        for j in range(max(0,n-2*i), min(i+1,n-i+1)):
            # a^i * b^j * c^{n-i-j}
            k = n-i-j
            u = coeffs[i*t + j]
            v = coeffs[j*t + k]
            if u == v:
                w = coeffs[k*t + i]
                if u == w:
                    continue 
            return True, False 
    return True, True


def monom_of(x, m):
    return x.coeff_monomial(m)