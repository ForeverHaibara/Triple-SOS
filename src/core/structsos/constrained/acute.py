import sympy as sp

from ..solution import SolutionStructuralSimple
from ..ternary.utils import CommonExpr
from ..ternary import sos_struct_cubic, sos_struct_sextic
from ..utils import (
    Coeff, CyclicSum, CyclicProduct,
    uniquely_named_symbol, radsimp, rationalize_func
)

a, b, c = sp.symbols("a b c")


def constrained_acute(poly, ineq_constraints, eq_constraints):
    gens = poly.gens
    if len(gens) != 3:
        return None
    a, b, c = gens
    cons = tuple(sp.Poly(_, gens) for _ in (b**2+c**2-a**2, c**2+a**2-b**2, a**2+b**2-c**2))
    cons = tuple(ineq_constraints.get(_) for _ in cons)
    if any(_ is None for _ in cons):
        return None

    cons = tuple(cons) + tuple(eq_constraints.values())


    coeff = Coeff(poly)
    if not poly.is_homogeneous or not coeff.is_cyclic():
        return None

    Fname = uniquely_named_symbol("_F", gens + tuple(ineq_constraints.values()))
    Gname = uniquely_named_symbol("_G", gens + tuple(ineq_constraints.values()))
    F, G = sp.Function(Fname), sp.Function(Gname)

    degree = coeff.degree()
    solution = None
    SOLVERS = {
        2: _constrained_acute_quadratic,
        3: _constrained_acute_cubic,
    }
    solver = SOLVERS.get(degree)
    if solver is not None:
        solution = solver(coeff, F)
    if solution is None:
        return None

    extra_checker = lambda x: x if isinstance(x, F) else None
    solution = SolutionStructuralSimple._extract_nonnegative_exprs(solution, func_name=Gname, extra_checker=extra_checker)
    
    if solution is None:
        return None

    if poly.gens != (sp.symbols("a b c")):
        solution = solution.xreplace(dict(zip(sp.symbols("a b c"), poly.gens)))

    replacement = {F(a): cons[0], F(b): cons[1], F(c): cons[2]}
    for k, v in ineq_constraints.items():
        if len(k.free_symbols) == 1 and k.is_monomial and k.LC() >= 0:
            replacement[G(k.free_symbols.pop())] = v/k.LC()
    solution = solution.xreplace(replacement)
    if solution.has(F) or solution.has(G):
        return None
    return solution



def _constrained_acute_quadratic(coeff, F):
    """
    s(2ab-a2) = s((b2+c2-a2)a+2abc)/(s(a))
    """
    c2, c1 = coeff((2,0,0)), coeff((1,1,0))
    if c2 >= 0 and c2 + c1 >= 0:
        return CommonExpr.quadratic(c2, c1)
    if c2 < 0 and c1 + 2*c2 >= 0:
        # s(2ab-a2) + ts(a2)
        return sp.Add(c1/2 * (CyclicSum(F(a)*a) + 6*CyclicProduct(a))/CyclicSum(a),
                      (c2 + c1/2)*CyclicSum(a**2))


def _constrained_acute_cubic(coeff, F):
    """
    Consider f(a,b,c) = s(-a^3 + x*(a^2*b+a^2*c) + y/3*a*b*c).
    Then f >= 0 for acute triangles iff x >= 1 and (x, y) is above the curves:

        f(sqrt(2),1,1) = (2*sqrt(2)+6)*x + sqrt(2)*y + (-2*sqrt(2)-2)
        f(1,1,1) = 6*x + y - 3
        det(border) = 7*x**2 - 2*x*y - 2*x - y**2 - 2*y - 9

    Examples
    ---------
    p(b+c-a)

    (s(-a3+9/7(a2b+ab2))-32/7abc)

    (s(-a3+(3/sqrt(2)-1)(a2b+ab2))+(sqrt(2)-5)abc)

    s(-a3+(8sqrt(2)-3)/7(a2b+ab2)-4abc/3)

    s(-a3+a2(b+c)-abc+sqrt(2)/2a(b-c)2)
    """
    c300, c210, c120, c111 = coeff((3,0,0)), coeff((2,1,0)), coeff((1,2,0)), coeff((1,1,1))
    if c210 != c120:
        return None
    if c300 >= 0:
        return sos_struct_cubic(coeff)
    x, y = radsimp(c210/-c300), radsimp(c111/-c300)

    def _solve_trivial(t):
        """
        Solve s(-a3+x(a2b+ab2))+yabc >= 0 for acute triangles with
        x = t and y = 6 - 6*t.
        """
        return CyclicSum(F(a)*a) + (t-1)*CyclicSum(a*(b-c)**2)
    def _solve_parabola(t):
        """
        Solve s(-a3+x(a2b+ab2))+yabc >= 0 for acute triangles with
        x = -(t**2 - 2*t + 9)/(t**2 - 2*t - 7) and y = 16*t/(t**2 - 2*t - 7)
        where t**2 - 2*t - 7 < 0.
        Hint: this is quadratic with respect to t.
        """
        if t == 1:
            return (8*CyclicProduct(a**2) + CyclicProduct(F(a)))/(CyclicSum(a)*CyclicSum(a**2))
        else:
            p = ((t-1)*a**3+(t-1)*a**2*b+(t-1)*a**2*c-sp.S(8)/3*a*b*c).together().as_coeff_Mul()
            p = p[0]**2*CyclicSum(p[1])**2
            return radsimp(1/(-t**2+2*t+7))*(p + 8*CyclicProduct(F(a)))/(CyclicSum(a)*CyclicSum(a**2))
    def _solve_bottom(t):
        """
        Solve s(-a3+x(a2b+ab2))+yabc >= 0 for acute triangles with
        x = t and y = 3 - 6*t.
        """
        if t >= 2:
            # 9/2s((a2+b2-c2)(a2+c2-b2)(b-c)2) + 8s(a3-abc-a(b-c)2/2)2
            p = (9*CyclicSum(F(b)*F(c)*(b-c)**2) + 4*CyclicSum(a*(2*a**2-b**2-c**2))**2)
            return p/(2*CyclicSum(a)*CyclicSum(a**2)) + (t-2)*CyclicSum(a*(b-c)**2)

        # 4(x-1)2(s(-a3+a2(b+c)-abc+(x-1)a(b-c)2)s(a3+(8-2x)/3abc)-s((b2+a2-c2)(c2+a2-b2)(b-c)2))-s((b+c-(2x-2)a)(a-b)(a-c))2
        original = CyclicSum(-a**3+t*a**2*(b+c)+(1-2*t)*a*b*c)
        multiplier = CyclicSum(a**3 + (8 - 2*t)/3*a*b*c)
        ker = CyclicSum(F(b)*F(c)*(b-c)**2) + 1/(t - 1)**2/4*CyclicSum((b+c-(2*t-2)*a)*(a-b)*(a-c))**2
        ker2 = CyclicSum((a**2+b**2-c**2)*(c**2+a**2-b**2)*(b-c)**2) + 1/(t - 1)**2/4*CyclicSum((b+c-(2*t-2)*a)*(a-b)*(a-c))**2
        rem = (original * multiplier - ker2).doit().as_poly(a,b,c)
        print(t, rem)
        rem_sol = sos_struct_sextic(Coeff(rem))
        if rem_sol is not None:
            return (rem_sol + ker)/multiplier


    def get_sol(x, y):
        if x < 1:
            return None
        if 6*x + y - 6 >= 0:
            return _solve_trivial(x) + (6*x + y - 6)*CyclicProduct(a)
        if x == 1:
            if y >= -2:
                return _solve_parabola(1) + (y+2)*CyclicProduct(a)
            return None
        if 6*x + y - 4 > 0:
            t = radsimp((8*x + y - 6)/(6*x + y - 4))
            w1 = radsimp((t - x)/(t - 1))
            if 0 <= w1 and w1 <= 1:
                return w1*CyclicSum(F(a)*a**2)/CyclicSum(a) + (1-w1)*_solve_trivial(t)
        if 6*x + y < 3: # poly(1,1,1) = 6*x + y - 3 < 0
            return None
        if 14*x**2 - 4*x*y - 4*x - y**2 + 4*y - 2 < 0:
            # This constraint can be factorized on QQ(sqrt(2))
            # and it implies poly(sqrt(2),1,1) < 0.
            return None
        if x + y + 1 != 0:
            t = radsimp((9 - 7*x + y)/(x + y + 1))
            if t**2 - 2*t - 7 < 0:
                xt = radsimp(-(t**2 - 2*t + 9)/(t**2 - 2*t - 7))
                # yt = radsimp(16*t/(t**2 - 2*t - 7))
                w1 = radsimp((xt - x)/(xt - 1))
                if 0 <= w1 and w1 <= 1:
                    return w1*_solve_parabola(1) + (1-w1)*_solve_parabola(t)
        
        # 3 <= 6*x + y <= 4
        # Find t such that ux := ux1/ux2 >= 1 + 1/sqrt(2).
        # This is equivalent to eqt := (ux1 - ux2)^2*2 - ux2^2 >= 0 and ux1 / ux2 >= 1.
        _t = sp.Symbol('t')
        ux1 = sp.Poly(radsimp([-3*x + y - 3, 22*x - 2*y + 6, 21*x + 9*y - 27]), _t)
        ux2 = sp.Poly(radsimp([-6*x - y - 6, 12*x + 2*y + 28, 42*x + 7*y - 54]), _t)
        # eqt = sp.Poly(radsimp([
        #     -18*x**2 + 12*x*y - 36*x + 7*y**2 + 12*y - 18,
        #     4*(66*x**2 + 20*x*y + 84*x - 7*y**2 - 36*y + 18),
        #     2*(154*x**2 - 92*x*y - 812*x + 29*y**2 + 228*y - 70),
        #     -4*(462*x**2 - 20*x*y - 468*x + 15*y**2 + 196*y - 162),
        #     -882*x**2 - 756*x*y + 2268*x - 41*y**2 + 972*y - 1458
        # ]), _t)
        def _get_ux_w(t):
            ux2t = ux2(t)
            if ux2t == 0: return sp.oo, 0
            ux = ux1(t)/ux2t
            xt = -(t**2 - 2*t + 9)/(t**2 - 2*t - 7)
            if ux != xt:
                w = (ux - x)/(ux - xt)
            else:
                uy = 3 - 6*ux
                yt = 16*t/(t**2 - 2*t - 7)
                w = (uy - y)/(uy - yt)
            return ux, w
        def _validation(t):
            ux, w = _get_ux_w(t)
            return (ux is not sp.oo) and (ux - 1)**2*2 >= 1 and w >= 0 and w <= 1
        if 14*x**2 - 4*x*y - 4*x - y**2 + 4*y - 2 == 0:
            t = 2*sp.sqrt(2) - 1
        elif _validation(sp.S(1)):
            t = sp.S(1)
        else:
            t = rationalize_func(sp.Poly([1,2,-7], _t), 
                    validation=_validation, validation_initial=lambda t: t>0, direction=-1)
        if t is None:
            if isinstance(x, sp.Rational) and isinstance(y, sp.Rational):
                return None
            t = 2*sp.sqrt(2) - 1
            if not _validation(t):
                return None
        # t is not None
        ux, w = radsimp(_get_ux_w(t))
        if 0 <= w and w <= 1:
            sol_bottom = _solve_bottom(ux)
            if sol_bottom is None:
                return None
            return w*_solve_parabola(t) + (1-w)*sol_bottom

    sol = get_sol(x, y)
    if sol is not None:
        return (-c300) * sol
    
def _constrained_acute_quartic(coeff, F):
    """
    """
    pass