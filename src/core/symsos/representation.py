import sympy as sp

from ..pqrsos import pqr_sym

def _extract_factor(poly, factor, symbol):
    """
    Given a polynomial and a factor, find degree and remainder such
    that `poly = factor ^ degree * remainder`

    Parameters
    ----------
    poly : sympy.Poly
        The polynomial to be factorized.
    factor : sympy.Poly
        The factor to be extracted.
    symbol : sympy.Symbol
        Poly and factor is treated as single variate polynomials of `symbol`.

    Returns
    -------
    degree : int
        The degree of the factor.
    remainder : sympy.Poly
        The remainder of the polynomial after extracting the factor.    
    """
    poly = poly.as_poly(symbol)
    factor = factor.as_poly(symbol)

    if poly.is_zero:
        return poly

    degree = 0
    while True:
        quotient, remain = divmod(poly, factor)
        if remain.is_zero:
            poly = quotient
            degree += 1
        else:
            break

    return degree, poly


def _sym_representation_positive(poly_pqr, return_poly = False):
    """
    Given a polynomial represented in pqr form where
    `p = a+b+c`, `q = ab+bc+ca`, `r = abc`, return the
    new form of polynomial with respect to
    ```
    x = s(a(a-b)(a-c))
    y = s(a(b-c)^2)
    z = p(a-b)^2 * s(a)^3 / p(a)
    w = x * y^2 / p(a)
    ```

    Actually we can solve that
    ```
    w   = (y - 2*x)^2 + z
    p^3 = (z*(x + 4*y) + 4*(x + y)^3) / w
    q   = y * ((x + y)*(4*x + y) + z) / w / p
    r   = x * y^2 / w
    ```

    Reference
    -------
    [1] https://zhuanlan.zhihu.com/p/616532245
    """
    x, y, z, p, q, r, w = sp.symbols('x y z p q r w')

    # to avoid cubic root, we keep p here
    poly = sp.together(poly_pqr.subs({
        q: y * (4*x**2 + 5*x*y + y**2 + z) / w / p,
        r: x * y**2 / w
    }))

    numerator, denominator = sp.fraction(poly)

    # numerator
    # do not collect p here, because terms with p naturally contains factors w
    # so that the w can be cancelled in each term
    numerator = numerator.subs(p, sp.cancel((z*(x+4*y)+4*(x+y)**3) / w)**(sp.S(1)/3))
    numerator = numerator.subs(w, y**2 - 4*x*y + 4*x**2 + z)

    degree, numerator = _extract_factor(numerator, (y**2 - 4*x*y + 4*x**2 + z), z)
    denominator /= w ** degree
    numerator = numerator.as_poly(z)

    if return_poly:
        return numerator, denominator

    numerator = sum(coeff.factor() * z**deg for ((deg, ), coeff) in numerator.terms())

    # denominator is in the form of p**r * w**s
    # w = (abc(s(a(b-c)2)-2s(a(a-b)(a-c)))2+s(a3)p(a-b)2) / (abc)
    # w = s(a(b-c)2)2s(a(a-b)(a-c)) / abc
    return numerator / denominator


def _sym_representation_real(poly_pqr, return_poly = False):
    """
    Given a polynomial represented in pqr form where
    `p = a+b+c`, `q = ab+bc+ca`, `r = abc`, return the
    new form of polynomial with respect to
    ```
    x = s(a^3 - abc)
    y = p(2*a - b - c) / 2
    z = 27/4 * p(a-b)^2
    ```

    Actually we can solve that
    ```
    w^3 = y^2 + z
    p   = x / w
    q   = (x^2 - y^2 - z) / (3*w^2)
    r   = (x^3/w^3 - 3*x + 2*y) / 27
    ```

    Reference
    -------
    [1] https://zhuanlan.zhihu.com/p/616532245
    """
    x, y, z, p, q, r, w = sp.symbols('x y z p q r w')

    # to avoid cubic root, we keep p here
    poly = sp.together(poly_pqr.subs({
        p: x / w,
        q: (x**2 - y**2 - z) / 3 / w**2,
        r: (x**3 / w**3 - 3*x + 2*y) / 27
    }))

    numerator, denominator = sp.fraction(poly)

    # numerator
    numerator = numerator.subs(w, sp.Pow(y**2 + z, sp.S(1)/3))

    degree, numerator = _extract_factor(numerator, (y**2 + z), z)
    denominator /= w ** (3 * degree)
    numerator = numerator.as_poly(z)

    if return_poly:
        return numerator, denominator

    numerator = sum(coeff.factor() * z**deg for ((deg, ), coeff) in numerator.terms())

    # denominator is in the form of w**s
    return numerator / denominator


def sym_representation(poly, is_pqr = None, positive = True, return_poly = False):
    """
    Represent a polynoimal to the symmetric form.

    Please refer to functions `_sym_representation_positive` and `_sym_representation_real`
    for the details of the representation.
    
    Parameters
    ----------
    poly : sympy.Poly or sympy.Expr
        The polynomial to be represented. Could be either a polynomial of (a,b,c)
        or its pqr representation.
    is_pqr : bool
        Whether the input polynomial is in pqr form. If it is None,
        it will automatically infer from the input polynomial.
    positive : bool
        Whether use the positive representation or the real representation.

    Returns
    -------
    sympy.Expr
        The polynomial in the new representation.
    """
    if poly.is_zero:
        return poly

    if is_pqr is None:
        a = sp.symbols('a')
        if a in poly.free_symbols:
            is_pqr = False
        else:
            is_pqr = True

    if not is_pqr:
        poly = pqr_sym(poly)

    if positive:
        return _sym_representation_positive(poly, return_poly)
    else:
        return _sym_representation_real(poly, return_poly)
