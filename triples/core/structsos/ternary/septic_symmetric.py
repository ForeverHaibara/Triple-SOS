from functools import partial

import sympy as sp
from sympy import Poly, Expr, Rational, Add

from .utils import (
    Coeff, CommonExpr, DomainExpr, quadratic_weighting
)
from .cubic import _sos_struct_cubic_symmetric
from .sextic_symmetric import _restructure_quartic_polynomial
from ..univariate import prove_univariate


def sos_struct_septic_symmetric(coeff, real=False):
    if not all(coeff((i,j,k)) == coeff((j,i,k)) for (i,j,k) in ((6,1,0),(5,2,0),(4,3,0),(4,2,1))):
        return None

    from .dense_symmetric import sos_struct_liftfree_for_six
    solution = sos_struct_liftfree_for_six(coeff)
    if solution is not None:
        return solution

    if coeff.is_rational:
        return _sos_struct_septic_symmetric_quadratic_form(coeff.as_poly(), coeff)


def _sos_struct_septic_symmetric_quadratic_form(poly, coeff: Coeff):
    """
    Let F0 = s(a7+a6b+a6c+a5bc-2a4b3-2a4c3) and G0 = s((b+c)(a2+a(b+c)+2bc)2(a-b)(a-c)).
    Let f(a,b,c) = g(a,b,c) = s(xa^2+yab).
    Then we have
    F_{x,y} = F0 - 2s(a5-a3b2+a3bc-a3c2)f(a,b,c) + s(a(a-b)(a-c))f(a,b,c)^2 >= 0
    G_{x,y} = G0 - 2s((b+c)(a2+a(b+c)+2bc)(a-b)(a-c))g(a,b,c) + s(a(b-c)^2)g(a,b,c)^2 >= 0

    See proof at class _septic_sym_axis.
    Such F_{x,y}, G_{x,y} has the property that the symmetric axis is a multiple of (a-1)^2 * (...)^2.
    For more general septic symmetric polynomials, we can first decompose its symmetric axis
    into several F_{x,y}, G_{x,y} and then combine them together.

    For a more primary case, see `_sos_struct_sextic_symmetric_quadratic_form`.

    Examples
    ---------
    s(a7-3a5bc+2a3b2c2)

    s((b+c)(b2+c2-a2)2(a-b)(a-c))-8p(a-b)2s(a)

    s(4a6b+4a6c-16a5b2-16a5c2+13a4b3+15a4b2c+15a4bc2+13a4c3-50a3b3c+18a3b2c2)

    s(a5(a-b)(a-c))-3s(a)p(a-b)2

    s(a7+a6b+a6c-9a5b2+13a5bc-9a5c2+7a4b3-9a4b2c-9a4bc2+7a4c3-9a3b3c+15a3b2c2)

    s(9a7-33a6b-33a6c+45a5b2+103a5bc+45a5c2-21a4b3-123a4b2c-123a4bc2-21a4c3+122a3b3c+30a3b2c2)

    s(a)(s(a2(a-b)(a-c)(a-4b)(a-4c))+0s(a2(a-b)(a-c)(a-3b)(a-3c))+11p(a-b)2)

    1/361s(11664a7-33696a6b-33696a6c+31104a5b2+99720a5bc+31104a5c2-9072a4b3-94476a4b2c-94476a4bc2-9072a4c3+65929a3b3c+34967a3b2c2)

    s(a7+39a6b+39a6c-123a5b2+1531a5bc-123a5c2+83a4b3-4607a4b2c-4607a4bc2+83a4c3+12158a3b3c-4474a3b2c2)

    s(a)(s(a2(a2-b2)(a2-c2))-p(a)s(a(a-b)(a-c)))-8p(a-b)2s(a)

    3s(a)p(2a2+b2+c2)-4s(b(2a2+b2+c2)(2c2+a2+b2))s(a2)-10p(a-b)2s(a)
    """
    a, b, c = coeff.gens
    CyclicSum, CyclicProduct = coeff.cyclic_sum, coeff.cyclic_product

    sym = poly.subs({b:1,c:1}).div(Poly([1,-2,1], a))
    if not sym[1].is_zero:
        return None

    # write the symmetric axis in sum-of-squares form
    sym = prove_univariate(sym[0], (0, None), return_type = 'list')
    if sym is None:
        return None
    # print(sym)

    def as_poly(f):
        if isinstance(f, Poly):
            return f
        elif isinstance(f, Expr):
            return f.as_poly(a)
        elif isinstance(f, int):
            return Poly([f], a)

    sym_f = as_poly(sum(i*j**2 for i, j in sym[1][1]))
    sym_g = as_poly(sum(i*j**2 for i, j in sym[0][1]) * Rational(1,2))

    ft_coeff, f_coeff, fx, fy, f_rem_coeff, f_rem_ratio = _restructure_quartic_polynomial(sym_f)
    gt_coeff, g_coeff, gx, gy, g_rem_coeff, g_rem_ratio = _restructure_quartic_polynomial(sym_g)

    # Compute the remaining coeff of p(a-b)^2 * s(a)
    # Theorem:
    # s(a^2-ab)^2 * s(a(a-b)(a-c)) = 1/4*s(a(a-b)^2(a-c)^2(2a-b-c)^2) + 3/4*p(a-b)^2*s(a)
    # s(a^2-ab)^2 * s(a(b-c)^2)    = s(a(b-c)^4(2a-b-c)^2) + 3*p(a-b)^2*s(a)
    ker_coeff = coeff((5,2,0)) - f_coeff*(fx - fy)**2 - g_coeff*(gx**2 + 2*gx*gy - 2*gx - 2*gy + 1)
    ker_coeff -= Rational(13,4) * ft_coeff - 4 * gt_coeff

    if f_rem_ratio is sp.oo:
        ker_coeff -= f_rem_coeff
    else:
        ker_coeff -= f_rem_coeff*(f_rem_ratio - 1)**2
    if g_rem_ratio is not sp.oo:
        ker_coeff -= g_rem_coeff*(2*g_rem_ratio + 1)

    # print(f_coeff, fx, fy, ft_coeff, f_rem_coeff, f_rem_ratio)
    # print(g_coeff, gx, gy, gt_coeff, g_rem_coeff, g_rem_ratio)
    # print('Ker_coeff =', ker_coeff)

    if fx is sp.nan or fy is sp.nan or gx is sp.nan or gy is sp.nan:
        return None

    f_g_solution = _septic_sym_axis(coeff).solve(f_coeff, fx, fy, g_coeff, gx, gy, ker_coeff)
    if f_g_solution is None:
        return None

    p1 = (ft_coeff/4*(a-b)**2*(a-c)**2 + gt_coeff*(b-c)**4).as_coeff_Mul()

    solution = Add(
        f_g_solution,
        p1[0] * CyclicSum(a * (2*a-b-c)**2 * p1[1]),
        _septic_sym_axis(coeff).rem_poly(f_rem_coeff, f_rem_ratio) * CommonExpr.schur(3, (a,b,c)),
        _septic_sym_axis(coeff).rem_poly(g_rem_coeff, g_rem_ratio) * CyclicSum(a*(b-c)**2)
    )
    return solution

class _septic_sym_axis(DomainExpr):
    """
    Let F0 = s(a7+a6b+a6c+a5bc-2a4b3-2a4c3) and f(a,b,c) = s(xa^2+yab)

    Define
    F_{x,y}(a,b,c) = F0 - 2s(a5-a3b2+a3bc-a3c2)f(a,b,c) + s(a(a-b)(a-c))f(a,b,c)^2.

    Theorem 1:
    F_{x,y} >= 0 because
    F_{x,y} * s(a(a-b)(a-c)) = (s(a(a-b)(a-c))f(a,b,c) - s(a5-a3b2+a3bc-a3c2))^2 + p(a)s(a)p(a-b)^2 >= 0.


    Let G0 = s((b+c)(a2+a(b+c)+2bc)2(a-b)(a-c)) and g(a,b,c) = s(xa^2+yab)
    Define
    G_{x,y}(a,b,c) = G0 - 2s((b+c)(a2+a(b+c)+2bc)(a-b)(a-c))g(a,b,c) + s(a(b-c)^2)g(a,b,c)^/2.

    Theorem 2:
    G_{x,y} >= 0 because
    G_{x,y} * s(a(b-c)^2) = (s(a(b-c)^2)g(a,b,c) - s((b+c)(a2+a(b+c)+2bc)(a-b)(a-c)))^2 + 4p(a)s(a)p(a-b)^2 >= 0.

    Also, we have:
    F_{x,y}(a,1,1)/a = G_{x,y}(a,1,1)/2 = (a - 1)**2 * ((x - 1)*a**2 + (2*y - 2)*a + 2*x + y - 2)**2

    The class provides various methods to solve the above inequalities.
    F(x, y)
        Return f, ker_coeff. such that F(x,y) + ker_coeff * s(a) * p(a-b)^2 == f.
    G(x, y)
        Return g, ker_coeff. such that G(x,y) + ker_coeff * s(a) * p(a-b)^2 == g.
    """

    def _F_regular(self, x, y):
        a, b, c = self.gens
        CyclicSum, CyclicProduct = self.cyclic_sum, self.cyclic_product

        f1 = (x - 1)*a**5 + (y - x)*a**4*(b+c) + (1 - y)*a**3*(b**2+c**2) \
            + (3*x - y - 1)*a**3*b*c + (y - 2*x)*a**2*b**2*c
        p1 = CyclicSum(f1.expand())**2 + CyclicProduct(a) * CyclicSum(a) * CyclicProduct((a-b)**2)
        return p1 / CommonExpr.schur(3, (a,b,c)), 0

    def _F_square(self, x, y):
        """
        When x + y == 5/3,
        F_{x,y} = 1/9 * s(a(a-b)^2(a-c)^2(3*(x-1)*a-(3*x-1)*(b+c)/2)^2) + (x-1)(9x-13)/12 * s(a) * p(a-b)^2
        """
        a, b, c = self.gens
        CyclicSum, CyclicProduct = self.cyclic_sum, self.cyclic_product

        if x + y == Rational(5,3):
            p1 = (3*(x - 1)*a - (3*x-1)/2*b - (3*x-1)/2*c).together()
            return CyclicSum(a*(a-b)**2*(a-c)**2*p1**2) / 9, -(x-1)*(9*x-13)/12

        return None, sp.oo

    def _F_sos(self, x, y, z_type = 0):
        """
        F(x, y) * s(a) = CyclicSum((a-b)**2*p1**2)/18 + CyclicSum(a*b*(a-b)**2*p2**2) + rem * p(a-b)^2
        where
        rem = CyclicSum((-x**2/3 - 2*x*z + 2*z)*a**2 + (-2*x**2/3 + 2*x*z - 2*y*z - z**2 + 2*z)*a*b).

        Here z is arbitrary. Find proper z such that the remainder term is nonnegative.
        """
        a, b, c = self.gens
        CyclicSum, CyclicProduct = self.cyclic_sum, self.cyclic_product

        p1 = (3*x - 3)*a**3 + (4*x + 3*y - 6)*a**2*b + (-4*x + 3*y)*a**2*c + (4*x + 3*y - 6)*a*b**2 + (-4*x + 3*y)*a*b*c \
            + (6*x - 3*y)*a*c**2 + (3*x - 3)*b**3 + (-4*x + 3*y)*b**2*c + (6*x - 3*y)*b*c**2 + (3 - 5*x)*c**3

        z = [2 - y, -x**2/(6*(x - 1))][z_type]
        if z is sp.oo or z is sp.nan or z is sp.zoo:
            return None, sp.oo

        c1, c2 = -x**2/3 - 2*x*z + 2*z, -2*x**2/3 + 2*x*z - 2*y*z - z**2 + 2*z
        p2 = a*b*(-x + y + z - 1) + c**2*z + (a*c + b*c)*(x + y - z - 1) + (a**2 + b**2)*(x - 1)

        if 2*c1 >= c2:
            # s(c_1*a^2 + c_2*a*b) >= (c_1 + c_2)/3 * s(a)^2
            ker_coeff = -(c1 + c2)/3
            rem = (2*c1 - c2)/6 * CyclicSum((a-b)**2)
        else:
            # s(c_1*a^2 + c_2*a*b) >= c_1 * s(a)^2
            ker_coeff = -c1
            rem = (c2 - 2*c1) * CyclicSum(a*b)

        sol = Add(
            CyclicSum((a-b)**2*p1**2) / 18,
            CyclicSum(a*b*(a-b)**2*p2**2),
            rem * CyclicProduct((a-b)**2)
        )
        return sol / CyclicSum(a), ker_coeff

    def _F_border(self, x, y):
        """
        (Conjecture) When 6*x**2 + 9*x*y - 20*x + 3*y**2 - 15*y + 17 >= 0,
        F_{x,y} >= (2*x + y - 3)**2 * p(a-b)^2 * s(a). And on the border there is a root (1,1,0) of order 4.

        The splitting curve 6*x**2 + 9*x*y - 20*x + 3*y**2 - 15*y + 17 = 0 can be parametrized by
        ((5*t**2 + 9*t + 3)/(3*(t + 1)*(2*t + 1)), (6*t**2 + 8*t + 3)/(3*(t + 1)*(2*t + 1)))
        """
        a, b, c = self.gens
        CyclicSum, CyclicProduct = self.cyclic_sum, self.cyclic_product

        ker_coeff = -(2*x + y - 3)**2
        if y == 1 or 2*x + y - 3 == 0:
            # Degenerated case. Example: s(a5(a-b)(a-c)) - 3p(a-b)2s(a)
            # p1 = (F_sos_p1 + (9-7*x-3*y)*(a-c)*(b-c)*(a+b-2*c)) / 3
            p1 = (x - 1)*a**3 + (1 - x)*a**2*b + (x + 2*y - 3)*a**2*c + (1 - x)*a*b**2 + (8*x + 5*y - 12)*a*b*c \
                + (-5*x - 4*y + 9)*a*c**2 + (x - 1)*b**3 + (x + 2*y - 3)*b**2*c + (-5*x - 4*y + 9)*b*c**2 + (3*x + 2*y - 5)*c**3
            # p2 = (F_sos_p2 at z = 3 - x - y)
            p2 = a*b*(2 - 2*x) + c**2*(-x - y + 3) + (a*c + b*c)*(2*x + 2*y - 4) + (a**2 + b**2)*(x - 1)
            sol = Add(
                CyclicSum((a-b)**2*p1**2) / 2,
                CyclicSum(a*b*(a-b)**2*p2**2),
            )
            return sol / CyclicSum(a), ker_coeff

        if x != 1 and (y - 1)*(16*x - y - 15) >= 0:
            det = (72*x**3 + 108*x**2*y - 300*x**2 + 54*x*y**2 - 318*x*y + 434*x + 9*y**3 - 84*y**2 + 243*y - 218)/(2*(x - 1))
            p1 = a**4*(x - 1) + a**3*b*(2*x + 2*y - 4) + a**3*c*(3 - 3*x) + a**2*b**2*(-2*x - 2*y + 6) + a**2*b*c*(x + y - 3) \
                + a**2*c**2*(3*x - 3) + a*b**3*(2*x + 2*y - 4) + a*b**2*c*(x + y - 3) + a*b*c**2*(-4*x - 4*y + 8) + a*c**3*(1 - x) \
                + b**4*(x - 1) + b**3*c*(3 - 3*x) + b**2*c**2*(3*x - 3) + b*c**3*(1 - x)
            p2 = (a - c)*(b - c)*((x + y - 2)*a**2 + (-4*x - 3*y + 6)*a*b + (1 - y)*a*c + (x + y - 2)*b**2 + (1 - y)*b*c + (1 - x)*c**2)

            u = (y - 1)/(4*(x - 1))
            def _mapping(_vec):
                c1, c2 = _vec
                return CyclicSum(c*(c1*p1 + c2*p2).expand()**2)
            qw = quadratic_weighting(self._coeff, u, u, 1, mapping = _mapping)
            if qw is not None and det >= 0: # actually this is always true in this case
                sol = qw + det * CyclicProduct(a) * CyclicProduct((a-b)**2)
                # p2 = (a**2 + (2*u - 1)*b*c).together().as_coeff_Mul()
                p2 = CommonExpr.quadratic(sp.S(1), 2*u-1, (a,b,c))
                return sol / p2, ker_coeff

        det = 6*x**2 + 9*x*y - 20*x + 3*y**2 - 15*y + 17
        det2 = (6*x + 3*y - 8) * det / (x - 1)
        if x != 1 and det2 >= 0:
            u = (2*x + y - 3) / (2*(x - 1))
            if 0 <= u <= 1:
                p1 = a**2*(x - 1) + a*b*(y - 1) + a*c*(y - 1) + b**2*(-x - y + 2) + b*c*(4*x + 3*y - 6) + c**2*(-x - y + 2)
                sol = Add(
                    u * CyclicSum(a**2*(a - b)**2*(a - c)**2*p1**2),
                    u * CyclicSum(a*(b + c)*(a - b)**2*(a - c)**2*p1**2),
                    (1 - u) * CyclicSum(a*(a - b)*(a - c)*((x - 1)*a**2 + (y - 1)*(a*b + a*c) + (2*x + y - 2)*b*c))**2,
                    det2 * CyclicProduct((a-b)**2) * CyclicSum(a) * CyclicProduct(a)
                )
                # multiplier = CyclicSum(a*(a-b)*(a-c) + u*a*(b-c)**2)
                multiplier = _sos_struct_cubic_symmetric(
                    self._coeff.from_dict({(3,0,0):1, (2,1,0): u-1, (1,2,0): u-1, (1,1,1):3-6*u}))
                return sol / multiplier, ker_coeff
        return None, sp.oo


    def _G_regular(self, x, y):
        """
        This is deprecated. It can be completely replaced with _G_trivial.
        It is here for illustration purpose.
        """
        a, b, c = self.gens
        CyclicSum, CyclicProduct = self.cyclic_sum, self.cyclic_product

        g1 = (x - 1)*a**4*(b+c) + (x + y - 1)*a**3*(b**2+c**2) \
            + (2*y + 2 - 6*x)*a**3*b*c + (2*x - 4*y + 2)*a**2*b**2*c
        p1 = CyclicSum(g1.expand())**2 + 4 * CyclicProduct(a) * CyclicSum(a) * CyclicProduct((a-b)**2)
        return p1 / CyclicSum(a*(b-c)**2), 0

    def _G_trivial(self, x, y):
        """
        G(x, y) = CyclicSum(a*(b-c)**2*p1**2)
        """
        a, b, c = self.gens
        CyclicSum, CyclicProduct = self.cyclic_sum, self.cyclic_product

        # w = y - 2
        # p1 = a**2*(w + x - y + 1) + a*b*(-w + 2*y - 2) + a*c*(-w + 2*y - 2) + b**2*(x - 1) + b*c*w + c**2*(x - 1)
        p1 = a**2*(x - 1) + a*b*y + a*c*y + b**2*(x - 1) + b*c*(y - 2) + c**2*(x - 1)
        return CyclicSum(a*(b-c)**2*p1**2), 0

    def _G_border(self, x, y):
        """
        When (x - 1)*(2*x + y - 2) >= 0, we have
        G(x, y) >= 4*(x - 1)*(2*x + y - 2) * p(a-b)^2 * s(a)
        and there exists roots on the border (a, 1, 0) with (x - 1)*a**2 + (-4*x - y + 4)*a + x - 1 = 0.
        """
        a, b, c = self.gens
        CyclicSum, CyclicProduct = self.cyclic_sum, self.cyclic_product

        ker_coeff = -4*(x - 1)*(2*x + y - 2)
        if ker_coeff >= 0:
            # this should be already covered by _G_trivial
            return None, sp.oo
        if (y - 1)*(2*x + y - 2) >= 0:
            p1 = (x - 1)*a**2 + (-4*x - y + 4)*a*c + (x - 1)*b**2 + (-4*x - y + 4)*b*c + (x - 1)*c**2 + (4*x + 3*y - 6)*a*b
            p2 = 4*(x - 1)*(2*x + y - 2) * CyclicSum(a*b*(a-b)**2) + 8*(y - 1)*(2*x + y - 2) * CyclicSum(a**2*(b-c)**2)
            sol = Add(
                CyclicSum(c*(a-b)**2*p1**2),
                p2 * CyclicProduct(a)
            )
            return sol, ker_coeff
        # not implemented
        return None, sp.oo

    def _G_sos(self, x, y, zw_type = 0):
        """
        G(x, y) * s(a) = CyclicSum(a*b*(a-b)**2*p2**2) + 2CyclicSum(a**2*(b-c)**2*p1**2) + rem * p(a-b)^2
        where
        rem = CyclicSum(c_1*a**2 + c_2*a*b),
        c_1 = -2*w**2 - 4*w*x + 4*w*y - 4*w + 2*x**2 + 4*x*y - 2*x*z - 8*x - 2*y**2 + 4*y + 2*z - 2
        c_2 = -4*w*y - x**2 + 2*x*y + 2*x*z - 2*x + 4*y**2 - 2*y*z - 6*y - z**2 + 2*z - 1

        Here z, w is arbitrary. Find proper z, w such that the remainder term is nonnegative.
        Note: Two conics (with respect to z, w) always tangent at (z, w) = (x + 1, y - 2).
        Also, 2*c1 - c2 = 0 represents two straight lines, one of which is the tangent, and the other
        gives (z, w) = (x - 2*y + 1, y - 2*x) to maximize min{c_1, (c_1 + c_2)/3}
        """
        a, b, c = self.gens
        CyclicSum, CyclicProduct = self.cyclic_sum, self.cyclic_product

        def _compute_c1_c2(x, y, z, w):
            c1 = -2*w**2 - 4*w*x + 4*w*y - 4*w + 2*x**2 + 4*x*y - 2*x*z - 8*x - 2*y**2 + 4*y + 2*z - 2
            c2 = -4*w*y - x**2 + 2*x*y + 2*x*z - 2*x + 4*y**2 - 2*y*z - 6*y - z**2 + 2*z - 1
            return c1, c2

        if zw_type == 0:
            z, w = x + 1, y - 2
            c1, c2 = sp.S(0), sp.S(0)
        else: # zw_type == 1:
            z, w = 2 - y, -x - 1
            c1, c2 = _compute_c1_c2(x, y, z, w)
            if min(c1, (c1 + c2)/3) <= 4*y*(x - 1):
                # use a better choice of z, w
                z, w = x - 2*y + 1, y - 2*x
                c1, c2 = 4*y*(x - 1), 8*y*(x - 1)

        p1 = a**2*(w + x - y + 1) + a*b*(-w + 2*y - 2) + a*c*(-w + 2*y - 2) + b**2*(x - 1) + b*c*w + c**2*(x - 1)
        p2 = a*b*(-x + y + z - 1) + c**2*z + (a*c + b*c)*(x + y - z - 1) + (a**2 + b**2)*(x - 1)

        # ker_coeff = -min{c_1, (c_1 + c_2)/3}
        if 2*c1 >= c2:
            # s(c_1*a^2 + c_2*a*b) >= (c_1 + c_2)/3 * s(a)^2
            ker_coeff = -(c1 + c2)/3
            rem = (2*c1 - c2)/6 * CyclicSum((a-b)**2)
        else:
            # s(c_1*a^2 + c_2*a*b) >= c_1 * s(a)^2
            ker_coeff = -c1
            rem = (c2 - 2*c1) * CyclicSum(a*b)

        sol = Add(
            CyclicSum(a*b*(a-b)**2*p2**2),
            2 * CyclicSum(a**2*(b-c)**2*p1**2),
            rem * CyclicProduct((a-b)**2)
        )
        return sol / CyclicSum(a), ker_coeff

    def rem_poly(self, rem_coeff, rem_ratio):
        a, b, c = self.gens
        CyclicSum, CyclicProduct = self.cyclic_sum, self.cyclic_product
        return rem_coeff * (CyclicSum(a**2 + rem_ratio*a*b)**2 if not rem_ratio is sp.oo else CyclicSum(a*b)**2)

    def solve(self, f_coeff, fx, fy, g_coeff, gx, gy, ker_coeff):
        a, b, c = self.gens
        CyclicSum, CyclicProduct = self.cyclic_sum, self.cyclic_product

        fx, fy, gx, gy = sp.S(fx), sp.S(fy), sp.S(gx), sp.S(gy)

        F_SOLVERS = [
            self._F_square,
            partial(self._F_sos, z_type = 0),
            partial(self._F_sos, z_type = 1),
            self._F_border,
            self._F_regular,
        ]
        G_SOLVERS = [
            self._G_trivial,
            self._G_border,
            # partial(self._G_sos, zw_type = 0),
            partial(self._G_sos, zw_type = 1),
            # self._G_regular,
        ]
        for F_solver in F_SOLVERS:
            pf, ker_coeff1 = F_solver(fx, fy)
            if pf is None:
                continue

            for G_solver in G_SOLVERS:
                pg, ker_coeff2 = G_solver(gx, gy)
                if pg is None:
                    continue

                rem_coeff = ker_coeff - f_coeff * ker_coeff1 - g_coeff * ker_coeff2
                # print([type(_) for _ in [rem_coeff, ker_coeff, f_coeff, ker_coeff1, g_coeff, ker_coeff2]])
                if rem_coeff >= 0:
                    return Add(
                        f_coeff * pf,
                        g_coeff * pg,
                        rem_coeff * CyclicSum(a) * CyclicProduct((a-b)**2)
                    )
