import sympy as sp

def deg(poly):
    """
    Return the degree of a polynomial.
    """
    return sum(poly.monoms()[0])


def verify(y, polys, poly, tol: float = 1e-10) -> bool:
    '''
    Verify whether the fraction approximation is valid
    by substracting the partial sums and checking whether the remainder is zero.
    '''
    try:
        for coeff, f in zip(y, polys):
            if coeff[0] != 0:
                if coeff[1] != -1:
                    if not isinstance(coeff[0], sp.Expr):
                        poly = poly - sp.Rational(coeff[0] , coeff[1]) * f
                    else:
                        v = coeff[0] / coeff[1]
                        coeff_dom = PreprocessText_GetDomain(str(v))
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
    '''
    Verify whether a root is strict.

    Warning: Better get the function regularized beforehand.
    '''
    return abs(func(root)) < tol



def verify_hom_cyclic(poly):
    """
    Check whether a polynomial is homogenous and 3-var-cyclic

    Returns
    -------
    is_hom : bool
        Whether the polynomial is homogenous.
    is_cyc : bool
        Whether the polynomial is 3-var-cyclic.
    """
    n = sum(poly.monoms()[0])

    if len(poly.args) != 4:
        for monom in poly.monoms():
            if sum(monom) != n:
                return False, False
        return True, False

    coeffs = {}
    for coeff, monom in zip(poly.coeffs(), poly.monoms()):
        if sum(monom) != n:
            return False, False
        coeffs[(monom[0], monom[1])] = coeff 
        
    for i in range((n-1)//3+1, n+1):
        # 0 <= k = n-i-j <= i
        for j in range(max(0,n-2*i), min(i+1,n-i+1)):
            # a^i * b^j * c^{n-i-j}
            u = coeffs.get((i,j))
            v = coeffs.get((j,n-i-j))
            if u == v: # Nones are treated the same
                w = coeffs.get((n-i-j,i))
                if u == w:
                    continue 
            return True, False 
    return True, True 