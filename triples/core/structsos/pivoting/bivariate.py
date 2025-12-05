import sympy as sp
from sympy.core.symbol import uniquely_named_symbol

from ..univariate import prove_univariate

def structural_sos_2vars(poly, ineq_constraints, eq_constraints):
    """
    Solve bivariate homogeneous polynomial inequality. This is equivalent
    to univariate case after dehomogenization.

    For a homogeneous polynomial inequality f(a,b) >= 0, the feasible region
    is a convex cone with vertex at the origin on the XY plane. The boundary
    of the cone is linear.
    """
    sgn, factors = poly.factor_list()

    sols = []
    for factor, multiplicity in factors:
        if multiplicity % 2 == 0:
            sols.append(factor.as_expr() ** multiplicity)
            continue
        sol = _sos_struct_bivariate_linear_ineq(factor, ineq_constraints, eq_constraints)
        if sol is None:
            sgn = -sgn
            sol = _sos_struct_bivariate_linear_ineq(-factor, ineq_constraints, eq_constraints)
        if sol is None:
            return None
        sols.append(sol ** multiplicity)

    if sgn < 0:
        return None

    return sgn * sp.Mul(*sols)


class HalfspaceIntersection2D():
    """
    Given multiple 2d homogenos halfplanes a*x + b*y >= 0, find the intersection
    represented by at most 2 linear constraints.

    See also
    ---------
    scipy.spatial.HalfspaceIntersection
    """
    def __init__(self):
        self.normals = []
        self.exprs = []
        self._degenerated = False
    @property
    def degenerated(self):
        return self._degenerated
    @classmethod
    def cross(cls, n1, n2):
        a1, b1 = n1
        a2, b2 = n2
        return a1 * b2 - a2 * b1
    @classmethod
    def dot(cls, n1, n2):
        a1, b1 = n1
        a2, b2 = n2
        return a1 * a2 + b1 * b2
    def _replace(self, i, n, expr):
        self.normals[i] = n
        self.exprs[i] = expr
        return
    def add(self, n, expr):
        """
        Add a constraint [expr]: a*x + b*y >= 0 where n = (a,b) is the normal vector.

        Parameters
        ----------
        n: Tuple
            normal vector (a, b), a*x + b*y >= 0
        expr: Any
            Additional information about the constraint. It is stored along with the normal vector.
        """
        if self._degenerated:
            return
        if len(self.normals) == 0:
            self.normals.append(n)
            self.exprs.append(expr)
            return
        elif len(self.normals) == 1:
            # ensure the normals are in counter-clockwise order
            c1 = self.cross(self.normals[0], n)
            if c1 < 0:
                self.normals.insert(0, n)
                self.exprs.insert(0, expr)
            elif c1 > 0 or (c1 == 0 and self.dot(self.normals[0], n) < 0):
                self.normals.append(n)
                self.exprs.append(expr)
            return
        n1, n2 = self.normals
        c1, c2 = self.cross(n1, n), self.cross(n2, n)
        if c1 == 0:
            if self.dot(n1, n) < 0:
                return self._replace(1, n, expr)
            return
        if c2 == 0:
            if self.dot(n2, n) < 0:
                return self._replace(0, n, expr)
            return
        if c1 > 0 and c2 < 0: # n is between n1 and n2
            return
        elif c1 > 0 and c2 > 0:
            return self._replace(1, n, expr)
        elif c1 < 0 and c2 > 0:
            self._degenerated = True
            return
        elif c1 < 0 and c2 < 0:
            return self._replace(0, n, expr)

    def as_bounds(self):
        return list(zip(self.normals, self.exprs))


def _sos_struct_bivariate_linear_ineq(poly, ineq_constraints, eq_constraints):
    hs = HalfspaceIntersection2D()

    for ineq, e in ineq_constraints.items():
        if ineq.is_linear:
            u, v = ineq.coeff_monomial((1, 0)), ineq.coeff_monomial((0, 1))
            # u*a + v*b >= 0
            hs.add((u, v), e)

    if hs.degenerated: # TODO: shall we solve degenerated case?
        return None

    a, b = poly.gens
    d = poly.homogeneous_order()

    def _hom_sos(coeffs, polys, d, numer, denom):
        numer = sp.signsimp(numer)
        denom = sp.signsimp(denom)
        def map_poly(p):
            d2 = p.total_degree()
            all_coeffs = [p.coeff_monomial((d2-i, )) for i in range(d2+1)]
            expr = sp.Add(*(c*numer**(d2-i)*denom**i for i, c in enumerate(all_coeffs))).expand()
            expr = sp.signsimp(expr.together())
            expr = expr**2 * denom**((d//2-d2)*2)
            return expr
        polys = [map_poly(p) for p in polys]
        return sp.Add(*(c*p for c, p in zip(coeffs, polys)))

    bound = hs.as_bounds()
    if len(bound) == 0:
        if d % 2 != 0:
            return None
        all_coeffs = [poly.coeff_monomial((d-i, i)) for i in range(d+1)]
        sol = prove_univariate(sp.Poly(all_coeffs, a), (-sp.oo, sp.oo))
        if sol is None:
            return None
        sol = (sol.xreplace({a: a/b}) * b**d).together()
        return sol
    return
    if len(bound) == 1:
        # 1 constraint = halfspace: represented by boundary + normal
        # u*a + v*b = x, u*b - v*a = y (x >= 0)
        # a = (u*x - v*y) / (u^2 + v^2), b = (v*x + u*y) / (u^2 + v^2)
        (u, v), expr = bound[0]
        y = uniquely_named_symbol('y', (a, b, *expr.free_symbols))
        poly2 = poly.subs({a: (u - v*y)/(u**2 + v**2), b: (v + u*y)/(u**2 + v**2)}).as_poly(y)
        sol = prove_univariate_interval(poly2, (-sp.oo, sp.oo), return_raw=True)
        if sol is None:
            return None
        ubva, uavb = sp.signsimp(sp.together(u*b - v*a)), sp.signsimp(sp.together(u*a + v*b))
        if d % 2 == 0:
            sol = _hom_sos(sol[0][1], sol[0][2], d, ubva, uavb)
        else:
            sol = _hom_sos(sol[0][1], sol[0][2], d-1, ubva, uavb) * expr
        return sol
    if len(bound) == 2:
        # 2 constraints:
        # u*a + v*b = x, w*a + z*b = y (x >= 0, y >= 0)
        # a = (v*y - z*x) / (v*w - u*z), b = (u*y - w*x) / (u*z - v*w)
        (u, v), expr1 = bound[0]
        (w, z), expr2 = bound[1]
        y = uniquely_named_symbol('y', (a, b, *expr1.free_symbols, *expr2.free_symbols))
        poly2 = poly.subs({a: (v*y - z)/(v*w - u*z), b: (u*y - w)/(u*z - v*w)}).as_poly(y)
        sol = prove_univariate_interval(poly2, (0, sp.oo), return_raw=True)
        if sol is None:
            return None
        wazb, uavb = sp.signsimp(sp.together(w*a + z*b)), sp.signsimp(sp.together(u*a + v*b))

        # p1 = sp.Add(*(c*p.as_expr()**2 for c, p in zip(*sol[0][1:])))
        # p2 = sp.Add(*(c*p.as_expr()**2 for c, p in zip(*sol[1][1:])))
        # if d % 2 == 1:
        #     sol = (expr1 * p1 + expr2 * p2).subs(y, wazb/uavb).together() * uavb**(d-1)
        # elif d >= 2:
        #     sol = ((u*a+v*b)**2 * p1 + expr2*expr1 * p2).subs(y, wazb/uavb).together() * uavb**(d-2)
        # elif d == 0: # will not happen
        #     sol = (p1 + expr2/expr1 * p2).subs(y, wazb/uavb).together() * uavb**d

        if d % 2 == 1:
            p1 = _hom_sos(sol[0][1], sol[0][2], d-1, wazb, uavb)
            p2 = _hom_sos(sol[1][1], sol[1][2], d-1, wazb, uavb)
            sol = expr1 * p1 + expr2 * p2
        elif d >= 2:
            p1 = _hom_sos(sol[0][1], sol[0][2], d, wazb, uavb)
            p2 = _hom_sos(sol[1][1], sol[1][2], d-2, wazb, uavb)
            sol = p1 + expr2*expr1 * p2
        elif d == 0: # will not happen?
            p1 = _hom_sos(sol[0][1], sol[0][2], d, wazb, uavb)
            p2 = _hom_sos(sol[1][1], sol[1][2], d, wazb, uavb)
            sol = p1 + expr2/expr1 * p2
        return sol
