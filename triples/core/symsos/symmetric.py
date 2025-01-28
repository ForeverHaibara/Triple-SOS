import sympy as sp

from .basic import SymmetricTransform, extract_factor
from ...utils import CyclicSum, CyclicProduct

class SymmetricPositive(SymmetricTransform):
    """
    Given a polynomial represented in pqr form where
    `p = a+b+c`, `q = ab+bc+ca`, `r = abc`, return the
    new form of polynomial with respect to
    ```
    x = a*(a-b)*(a-c) + b*(b-c)*(b-a) + c*(c-a)*(c-b)
    y = a*(b-c)^2 + b*(c-a)^2 + c*(a-b)^2
    z = (a-b)^2 * (b-c)^2 * (c-a)^2 * (a+b+c)^3 / (a*b*c)
    w = x * y^2 / (a*b*c)
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
    @classmethod
    def _transform_pqr(cls, poly_pqr, original_symbols, new_symbols, return_poly=True):
        p, q, r = poly_pqr.gens
        a, b, c = original_symbols
        x, y, z = new_symbols
        degree = 0 if poly_pqr.is_zero else (lambda d:d[0]+2*d[1]+3*d[2])(poly_pqr.monoms()[0])

        # Note that q/p^2 = (...)/(p*w), r/p^3 = (...)/(p*w),
        # we first dehomogenize by setting p = 1
        poly_qr = poly_pqr.subs(p, 1).as_poly(q, r)
        hom_degree = poly_qr.total_degree()
        numerator = poly_qr.homogenize(p)(
            y * ((x + y)*(4*x + y) + z),
            x * y**2,
            z*(x + 4*y) + 4*(x + y)**3
        )
        # poly = p**degree * numerator / (p*w)**hom_degree

        p_degree = 3*hom_degree - degree # negative degree is acceptable
        w_degree = hom_degree

        # denominator = p**p_degree * w**w_degree
        # return numerator / denominator

        # numerator can be factored by w (depending on the multiplicity at (1,1,1))
        numerator = numerator.as_poly(x, y, z)
        wpoly = ((y - 2*x)**2 + z).as_poly(x, y, z)
        extract_w_degree, numerator = extract_factor(numerator, wpoly, [(2, 1, -9)])
        w_degree -= extract_w_degree

        numerator = numerator.as_poly(x, y, z)
        p0 = CyclicSum(a,(a,b,c))
        # w0 = CyclicSum(a*(a-b)*(a-c),(a,b,c))*CyclicSum(a*(b-c)**2,(a,b,c))**2 / CyclicProduct(a,(a,b,c))
        # w0 = (CyclicProduct((a-b)**2,(a,b,c))*p0**3 + 
        #         CyclicProduct(a,(a,b,c))*CyclicProduct((a+b-2*c)**2,(a,b,c)))/CyclicProduct(a,(a,b,c))
        _SCHUR = (CyclicSum((b-c)**2*(b+c-a)**2, (a,b,c)) + 2*CyclicSum(b*c*(b-c)**2, (a,b,c)))/2/CyclicSum(a, (a,b,c))
        w0 = _SCHUR * CyclicSum(a*(b-c)**2,(a,b,c))**2 / CyclicProduct(a,(a,b,c))
        denominator = p0**p_degree * w0**w_degree
        if return_poly:
            return numerator, denominator

        numerator = numerator.as_poly(z)
        numerator = sum(coeff.factor() * z**deg for ((deg, ), coeff) in numerator.terms())
        return (numerator / denominator).together()

    @classmethod
    def get_inv_dict(cls, symbols, new_symbols):
        a, b, c = symbols
        x, y, z = new_symbols
        # _SCHUR = CyclicSum(a*(a-b)*(a-c), (a,b,c))
        _SCHUR = (CyclicSum((b-c)**2*(b+c-a)**2, (a,b,c)) + 2*CyclicSum(b*c*(b-c)**2, (a,b,c)))/2/CyclicSum(a, (a,b,c))
        return {
            x: _SCHUR,
            y: CyclicSum(a*(b-c)**2, (a,b,c)),
            z: CyclicProduct((a-b)**2, (a,b,c)) * CyclicSum(a, (a,b,c))**3 / CyclicProduct(a, (a,b,c)),            
        }

    @classmethod
    def get_default_constraints(cls, symbols):
        x, y, z = symbols
        return {
            sp.Poly(x, (x,y,z)): x,
            sp.Poly(y, (x,y,z)): y,
            sp.Poly(z, (x,y,z)): z
        }, dict()

class SymmetricReal(SymmetricTransform):
    """
    Given a polynomial represented in pqr form where
    `p = a+b+c`, `q = ab+bc+ca`, `r = abc`, return the
    new form of polynomial with respect to
    ```
    x = a^3 + b^3 + c^3 - 3*a*b*c
    y = (2*a - b - c) * (2*b - c - a) * (2*c - a - b)/2
    z = 27/4 * (a - b)^2 * (b - c)^2 * (c - a)^2
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
    @classmethod
    def _transform_pqr(cls, poly_pqr, original_symbols, new_symbols, return_poly=True):
        p, q, r = poly_pqr.gens
        a, b, c = original_symbols
        x, y, z = new_symbols
        degree = 0 if poly_pqr.is_zero else (lambda d:d[0]+2*d[1]+3*d[2])(poly_pqr.monoms()[0])

        # Note that q/p^2 = (..)/(27x^3), r/p^3 = (..)/(27x^3),
        # we first dehomogenize by setting p = 1
        poly_qr = poly_pqr.subs(p, 1).as_poly(q, r)
        hom_degree = poly_qr.total_degree()
        numerator = poly_qr.homogenize(p)(
            9*x*(x**2 - y**2 - z),
            x**3 + (2*y - 3*x)*(y**2 + z),
            27*x**3
        )
        # poly = p**degree * numerator / (27*x**3)**hom_degree

        p_degree = 3*hom_degree - degree # negative degree is acceptable
        w_degree = 3*hom_degree

        # denominator = p**p_degree * w**w_degree * 27**hom_degree
        # return numerator / denominator

        # numerator can be factored by x or w^3 (depending on the multiplicity at (1,1,1))
        numerator = (numerator / 27**hom_degree).as_poly(x, y, z)

        if not numerator.is_zero:
            min_x_degree = min(numerator.monoms(), key=lambda _:_[0])[0]
            if min_x_degree > 0:
                # x = p*w
                numerator = sp.Poly(dict(((i-min_x_degree, j, k), v) for (i,j,k), v in numerator.terms()), x, y, z)
                p_degree -= min_x_degree
                w_degree -= min_x_degree

        wpoly = (y**2 + z).as_poly(x, y, z)
        extract_w_degree, numerator = extract_factor(numerator, wpoly, [(3, 2, -4)])
        w_degree -= extract_w_degree * 3

        numerator = numerator.as_poly(x, y, z)
        p0 = CyclicSum(a,(a,b,c))
        w0 = CyclicSum((a-b)**2,(a,b,c))/2
        denominator = p0**p_degree * w0**w_degree
        if return_poly:
            return numerator, denominator

        numerator = numerator.as_poly(z)
        numerator = sum(coeff.factor() * z**deg for ((deg, ), coeff) in numerator.terms())
        return (numerator / denominator).together()
    
    @classmethod
    def get_inv_dict(cls, symbols, new_symbols):
        a, b, c = symbols
        x, y, z = new_symbols
        return {
            x: CyclicSum(a, (a,b,c)) * CyclicSum((a-b)**2, (a,b,c)) / 2,
            y: CyclicProduct(2*a - b - c, (a,b,c)) / 2,
            z: sp.S(27)/4 * CyclicProduct((a-b)**2, (a,b,c))
        }

    @classmethod
    def get_default_constraints(cls, symbols):
        x, y, z = symbols
        return {
            sp.Poly(z, (x,y,z)): z
        }, dict()