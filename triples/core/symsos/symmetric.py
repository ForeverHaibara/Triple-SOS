import sympy as sp
from sympy.combinatorics import AlternatingGroup

from .basic import SymmetricTransform, extract_factor
from ...utils import CyclicSum, CyclicProduct, SymmetricSum, SymmetricProduct

class SymmetricPositive(SymmetricTransform):
    """
    Given a polynomial represented in pqr form where
    `p = a+b+c`, `q = ab+bc+ca`, `r = abc`, return the
    new form of the polynomial with respect to
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

    This algorithm was proposed in [1, Chapter 4] and mentioned in [2].

    References
    ----------
    [1] 陈胜利.不等式的分拆降维幂方法与可读证明.哈尔滨工业大学出版社,2016.
    [2] https://zhuanlan.zhihu.com/p/616532245
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
    new form of the polynomial with respect to
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

    This algorithm was proposed in [1].

    The algorithm can be extended to 4-var case [2], where
    ```
    D = (a-b)^2*(b-c)^2*(c-d)^2*(d-a)^2*(a-c)^2*(b-d)^2
    x = S[a]S[(a-b)^2*(b-c)^2*(c-a)^2]*S[(a-b)^2]/(144*(a+b-c-d)*(a+c-b-d)*(a+d-b-c))
    y = S[(a-b)^2*(b-c)^2*(c-a)^2]/6
    z = S[(a-b)^2*(c-d)^4]/2 - y
    w = D*S[(a-b)^2]^3/(4*(a+b-c-d)^2*(a+c-b-d)^2*(a+d-b-c)^2)
    ```

    The notation "S[.]" stands for the 24-term complete symmetric summation.
    It can be solved that

    ```
    D = w*y^2*z / (16*((8*y+z)*w + (2*y+z)^3))
    p/(r2*q2^2) = 1024*x*D*(w + (y-z)^2)/(w*y^3*z)
    q2^3 = -w*y^2*z/(1024*D*(w + (y-z)^2))
    r2^2 = y^2*z/(8*(w + (y-z)^2))
    s2/q2^2 = 4*D*(w - 8*y^2 - 2*y*z + z^2)/(w*y^2)
    q2 = -3*p^2/8 + q
    r2 = p^3/8 - p*q/2 + r
    s2 = -3*p^4/256 + p^2*q/16 - p*r/4 + s
    ```

    where `p = a+b+c+d`, `q = ab+bc+cd+da+ac+bd`, `r = abc+bcd+cda+dab`, `s = abcd`.

    References
    ----------
    [1] https://zhuanlan.zhihu.com/p/616532245

    [2] https://zhuanlan.zhihu.com/p/20969491385
    """
    @classmethod
    def _transform_pqr(cls, poly_pqr, original_symbols, new_symbols, return_poly=True):
        func = None
        nvars = len(poly_pqr.gens)
        cls._check_nvars(nvars)
        if nvars == 3:
            func = _symmetric_real_3vars
        elif nvars == 4:
            func = _symmetric_real_4vars
        return func(poly_pqr, original_symbols, new_symbols, return_poly=return_poly)

    @classmethod
    def _check_nvars(cls, nvars: int) -> bool:
        if nvars != 3 and nvars != 4:
            raise ValueError("Only 3-var or 4-var polynomials are supported.")
        return True

    @classmethod
    def get_inv_dict(cls, symbols, new_symbols):
        nvars = len(symbols)
        if nvars == 3:
            a, b, c = symbols
            x, y, z = new_symbols
            return {
                x: CyclicSum(a, (a,b,c)) * CyclicSum((a-b)**2, (a,b,c)) / 2,
                y: CyclicProduct(2*a - b - c, (a,b,c)) / 2,
                z: sp.S(27)/4 * CyclicProduct((a-b)**2, (a,b,c))
            }
        else: # if nvars == 4:
            a, b, c, d = symbols
            x, y, z, w = new_symbols
            rr = (a - b - c + d)*(a - b + c - d)*(a + b - c - d)
            disc = (a - b)**2*(a - c)**2*(a - d)**2*(b - c)**2*(b - d)**2*(c - d)**2
            y_ = SymmetricSum((a-b)**2*(b-c)**2*(c-a)**2, (a,b,c,d))/6
            w_ = disc * SymmetricSum((a-b)**2,(a,b,c,d))**3 / (4*rr**2)
            J = -(a*b - 2*a*c + a*d + b*c - 2*b*d + c*d)*(a*b + a*c - 2*a*d - 2*b*c + b*d + c*d)*(2*a*b - a*c - a*d - b*c - b*d + 2*c*d)
            # z_ = y_ - 4*J
            ker1 = w_ + 16*J**2
            z_ = ker1*rr**2 / (8*y_**2)
            # ker2 = w_*y_**2*z_ / 16 / disc
            x_ = SymmetricSum(a,(a,b,c,d))*SymmetricSum((a-b)**2,(a,b,c,d))*y_/24/rr
  
            return {x: x_, y: y_, z: z_, w: w_}

    @classmethod
    def get_default_constraints(cls, symbols):
        nvars = len(symbols)
        if nvars == 3:
            x, y, z = symbols
            return {sp.Poly(z, (x,y,z)): z}, dict()
        else: # if nvars == 4:
            x, y, z, w = symbols
            return {
                sp.Poly(y, (x,y,z,w)): y,
                sp.Poly(z, (x,y,z,w)): z,
                sp.Poly(w, (x,y,z,w)): w
            }, dict()


def _symmetric_real_3vars(poly_pqr, original_symbols, new_symbols, return_poly=True):
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


def _symmetric_real_4vars(poly_pqr, original_symbols, new_symbols, return_poly=True):
    p, q, r, s = poly_pqr.gens
    x, y, z, w = new_symbols
    a, b, c, d = original_symbols
    p2, q2, r2, s2 = sp.Dummy("p2"), sp.Dummy("q2"), sp.Dummy("r2"), sp.Dummy("s2")
    degree = 0 if poly_pqr.is_zero else (lambda _: _[0]+_[1]*2+_[2]*3+_[3]*4)(poly_pqr.monoms()[0])
    poly_shifted = poly_pqr(p2, 3*p2**2/8 + q2, p2**3/16 + p2*q2/2 + r2, p2**4/256 + p2**2*q2/16 + p2*r2/4 + s2).as_poly(p2, q2, r2, s2)

    # dehomogenize p2=r2*q2^2*p, s2=q2^2*s (symbols p,q,r,s are reused and are different now)
    poly_dehom = poly_shifted(r*q**2*p, q, r, q**2*s).as_poly(p, q, r, s)
    q_degree = -((degree * 2) % 3)
    r_degree = -(degree % 2)
    # q^3 -> q, r^2 -> r
    poly_dehom = sp.Poly(dict(((i,j//3,k//2,l), _) for (i,j,k,l), _ in poly_dehom.terms()), p, q, r, s)

    # Naive implementation:
    # ker1, ker2 = w + (y - z)**2, (27*y**2*z + (w + (y - z)**2)*(8*y + z))
    # numerator = poly_dehom(
    #     64*x*ker1/ker2/y,
    #     -ker2/(64*ker1),
    #     y**2*z/(8*ker1),
    #     -z*(9*y**2 - ker1)/(4*ker2))
    # )
    # denominator = q2 ** q_degree * r2 ** r_degree
    # return numerator.factor() / denominator

    # fraction-free algorithm
    hom_degree = poly_dehom.total_degree()
    poly_dehom = poly_dehom.homogenize(p2)
    ker1, ker2 = r2, s2  # any temporary free symbols (defer the substitution of true values)
    numerator = poly_dehom(
        4096*x*ker1**2,
        -y*ker2**2,
        8*y**3*z*ker2,
        -16*y*z*(9*y**2 - ker1)*ker1,
        64*y*ker1*ker2
    ).as_poly(x, y, z, ker1, ker2)
    _min_degree = lambda poly, i: 0 if poly.is_zero else min(_[i] for _ in poly.monoms())
    min_degrees = [_min_degree(numerator, i) for i in range(5)]
    numerator = sp.Poly(dict(
        (tuple(mi - mini for mi, mini in zip(m, min_degrees)), _)
        for m, _ in numerator.terms()
    ), x, y, z, ker1, ker2).as_poly(ker1, ker2)

    # degrees of y, ker1, ker2 in denominator is lifted by homogenization
    # and the degrees are reduced after cancellation
    rem_degrees = [0, hom_degree, 0, hom_degree, hom_degree]
    rem_degrees = [ri - mi for ri, mi in zip(rem_degrees, min_degrees)]

    # substitute the actual value in
    ker1, ker2 = w + (y - z)**2, (8*w*y + w*z + 8*y**3 + 12*y**2*z + 6*y*z**2 + z**3)
    numerator = numerator(ker1, ker2).as_poly(x, y, z, w)
    inv_denominator, numerator = numerator.primitive()
    if inv_denominator < 0:
        inv_denominator, numerator = -inv_denominator, -numerator
    denominator = 1/inv_denominator

    # try if numerator can be divided by ker1
    ker1poly = (w + (y - z)**2).as_poly(x,y,z,w)
    extract_ker1_degree, numerator = extract_factor(numerator, ker1poly, [(5,3,1,-4)])
    rem_degrees[3] -= extract_ker1_degree

    xd, yd, zd, k1d, k2d = rem_degrees
    # Naive implementation:
    # denominator = 64**hom_degree * x**xd * y**yd * z**zd * ker1**k1d * ker2**k2d

    # ker2 == w*y**2*z / 16 / Discriminant
    yd = yd + 2*k2d
    zd = zd + k2d
    wd = k2d
    disc = (a-b)**2*(b-c)**2*(c-d)**2*(a-d)**2*(a-c)**2*(b-d)**2
    if r_degree % 2 == 1:
        rd = r_degree
        # r2*x == S[a]*S[(a-b)^2]*y/192
        denominator *= SymmetricSum(a,(a,b,c,d))**rd * SymmetricSum((a-b)**2,(a,b,c,d))**rd / sp.S(192)**rd
        yd = yd + rd
        xd = xd - rd
        r_degree = 0

    disc_pow_k2d = disc**k2d
    if k2d % 4 == 0:
        disc_pow_k2d = SymmetricProduct((a-b)**2, (a,b,c,d))**(k2d//4)
    elif k2d % 2 == 0:
        disc_pow_k2d = CyclicProduct((a-b)**2, (a,b,c,d), AlternatingGroup(4))**(k2d//2)

    denominator *= sp.S(2)**(6*hom_degree - 4*k2d)\
        * x**xd * y**yd * z**zd * w**wd * (w + (y - z)**2)**k1d / disc_pow_k2d

    q2_ = -SymmetricSum((a-b)**2, (a,b,c,d))/32
    r2_ = (a+b-c-d)*(a+c-d-b)*(a+d-b-c)/8
    if q_degree % 2 == 1: # q2 -> -q2
        numerator = -numerator
    denominator *= (-q2_)**q_degree * r2_**r_degree

    if return_poly:
        return numerator, denominator
    return (numerator.as_expr() / denominator).together()