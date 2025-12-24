from itertools import permutations, combinations
from typing import List, Tuple, Dict, Callable, Optional
from math import gcd

from sympy import Poly, Expr, Integer, Rational, Add, Mul
from sympy.combinatorics import CyclicGroup

from .utils import Coeff, DomainExpr
from .dense_symmetric import sos_struct_dense_symmetric
from .quadratic import sos_struct_quadratic
from .cubic import sos_struct_cubic
from .quartic import sos_struct_quartic
from ..univariate import prove_univariate

def sos_struct_sparse(coeff, real = True):
    """
    Solver to very sparse inequalities like AM-GM.

    The function does not use `recursion` for minimal dependency.

    This method does not present solution for a,b,c in R in prior, but in R+.
    Inequalities of degree 2 and 4 are skipped because they might be
    handled in R.
    """
    a, b, c = coeff.gens
    CyclicSum, CyclicProduct = coeff.cyclic_sum, coeff.cyclic_product

    if len(coeff) > 6:
        return None

    degree = coeff.total_degree()
    if degree < 5:
        if degree == 0:
            v = coeff((0,0,0))
            return v if v >= 0 else None
        elif degree == 1:
            v = coeff((1,0,0))
            return v * CyclicSum(a) if v >= 0 else None
        elif degree == 2:
            return sos_struct_quadratic(coeff)
        elif degree == 3:
            return sos_struct_cubic(coeff)
        elif degree == 4:
            # quartic should be handled by _sos_struct_quartic
            # because it presents proof for real numbers
            return sos_struct_quartic(coeff, None)

    if len(coeff) <= 6:
        return _sos_struct_sparse_amgm(coeff)



#####################################################################
#
#                          AMGM solver
#
#####################################################################

class CoeffMonom:
    """Closely related to CyclicSum(a**i*b**j*c**k) and the 3var coefficient triangle."""
    __slots__ = ['monom']
    def __init__(self, *monom):
        self.monom = monom
    def __repr__(self) -> str:
        return f"CoeffMonom({', '.join(str(i) for i in self.monom)})"
    def __str__(self) -> str:
        return self.__repr__()
    def degree(self) -> int:
        return sum(self.monom)
    def norm(self) -> int:
        return sum(abs(i) for i in self.monom)
    def total_degree(self) -> int:
        return sum(self.monom)
    def __eq__(self, other) -> bool:
        return self.monom == other.monom
    def __ne__(self, other) -> bool:
        return self.monom != other.monom
    def __sub__(self, other) -> 'CoeffMonom':
        return CoeffMonom(*[self[i]-other[i] for i in range(3)])
    def __add__(self, other) -> 'CoeffMonom':
        return CoeffMonom(*[self[i]+other[i] for i in range(3)])
    def __mul__(self, other) -> 'CoeffMonom':
        return CoeffMonom(*[self[i]*other for i in range(3)])
    def __rmul__(self, other) -> 'CoeffMonom':
        return CoeffMonom(*[other*self[i] for i in range(3)])
    def __floordiv__(self, other) -> 'CoeffMonom':
        if hasattr(other, '__iter__'): return CoeffMonom(*[self[i]//other[i] for i in range(3)])
        return CoeffMonom(*[self[i]//other for i in range(3)])
    def __abs__(self) -> 'CoeffMonom':
        return CoeffMonom(*[abs(self[i]) for i in range(3)])
    def __hash__(self) -> int:
        return hash(self.monom)
    def __getitem__(self, i) -> int:
        return self.monom[i]
    def __iter__(self):
        return iter(self.monom)
    def std(self) -> 'CoeffMonom':
        x, y, z = self.monom
        return CoeffMonom(*max((x,y,z), (y,z,x), (z,x,y)))
    def dot(self, other) -> int:
        return sum(self[i]*other[i] for i in range(3))
    def gcd(self) -> int:
        x, y, z = self.monom
        return gcd(x, gcd(y,z))
    def factor_list(self) -> Tuple[int, 'CoeffMonom']:
        g = self.gcd()
        return g, CoeffMonom(*[i//g for i in self.monom])
    def area(self) -> int:
        x, y, z = self.monom
        return x**2+y**2+z**2-x*y-y*z-z*x
    def as_monom(self, a, b, c) -> Expr:
        return a**self[0]*b**self[1]*c**self[2]
    def cycle(self) -> Tuple['CoeffMonom']:
        x, y, z = self.monom
        return (self, CoeffMonom(y,z,x), CoeffMonom(z,x,y))
    def next(self) -> 'CoeffMonom':
        x, y, z = self.monom
        return CoeffMonom(y,z,x)
    @property
    def is_center(self) -> bool:
        return self.monom[0] == self.monom[1] and self.monom[1] == self.monom[2]
    @classmethod
    def is_collinear(cls, p1, p2, p3) -> bool:
        # cls.assert_equal_degrees(p1, p2, p3)
        return p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]) == 0
    @classmethod
    def integer_intersection(cls, p1, p2, p3, p4) -> Optional['CoeffMonom']:
        # cls.assert_equal_degrees(p1, p2, p3, p4)
        (p10, p11, p12), (p20, p21, p22), (p30, p31, p32), (p40, p41, p42) = p1, p2, p3, p4
        det = p10*p31 - p10*p41 - p11*p30 + p11*p40 - p20*p31 + p20*p41 + p21*p30 - p21*p40
        p50 = p10*p21*p30 - p10*p21*p40 - p10*p30*p41 + p10*p31*p40 - p11*p20*p30 + p11*p20*p40 + p20*p30*p41 - p20*p31*p40
        p51 = p10*p21*p31 - p10*p21*p41 - p11*p20*p31 + p11*p20*p41 - p11*p30*p41 + p11*p31*p40 + p21*p30*p41 - p21*p31*p40
        if det == 0 or p50 % det != 0 or p51 % det != 0: return None
        return CoeffMonom(p50//det, p51//det, sum(p1) - p50//det - p51//det)
    def is_equal_degree(self, other) -> bool:
        return sum(other) == self.degree()
    @classmethod
    def assert_equal_degree(cls, *args) -> bool:
        d = sum(args[0]) if cls is CoeffMonom else sum(cls)
        if not all(sum(i) == d for i in args): raise ValueError("Monomials must have the same degree.")
        return True
    def weight(self, point) -> Dict['CoeffMonom', Rational]:
        self.assert_equal_degree(point)
        x, y, z = point
        u, v, w = self.monom
        if u==v and v==w and x==y and y==z: return {(u,v,w): Integer(1)}
        det = Integer(3*u*v*w - (u**3+v**3+w**3))
        deta = Integer(x*(v*w-u**2)+y*(w*u-v**2)+z*(u*v-w**2))
        detb = Integer(x*(w*u-v**2)+y*(u*v-w**2)+z*(v*w-u**2))
        detc = Integer(x*(u*v-w**2)+y*(v*w-u**2)+z*(w*u-v**2))
        return {(u,v,w): deta/det, (v,w,u): detb/det, (w,u,v): detc/det}
    def is_circumscribing(self, other) -> Optional['CoeffMonom']:
        if not self.is_equal_degree(other): raise ValueError("Monomials must have the same degree.")
        other = CoeffMonom(*other) if not isinstance(other, CoeffMonom) else other
        nex = self.next()
        for p in other.cycle():
            if CoeffMonom.is_collinear(self, p, nex):
                return p
        return None
    def __rshift__(self, other) -> bool:
        """Test whether two coefficient triangles are inclusive."""
        if not isinstance(other, CoeffMonom): other = CoeffMonom(*other)
        if self.is_center and self.assert_equal_degree(other): return other.is_center
        return all(_ >= 0 for _ in self.weight(other.monom).values())
    def __lshift__(self, other) -> bool:
        """Test whether two coefficient triangles are inclusive."""
        if not isinstance(other, CoeffMonom): other = CoeffMonom(*other)
        if self.is_center and self.assert_equal_degree(other): return True
        return all(_ >= 0 for _ in other.weight(self.monom).values())


class AMGM3(DomainExpr):
    def solve(self, c1, m1, c2, m2):
        """Solve c1*CyclicSum(a**m1[0]*b**m1[1]*c**m1[2]) + c2*CyclicSum(a**m2[0]*b**m2[1]*c**m2[2]) >= 0."""
        a, b, c = self.gens
        CyclicSum, CyclicProduct = self.cyclic_sum, self.cyclic_product

        if c1 >= 0 and c2 >= 0: # TODO: handle real numbers??
            return c1*CyclicSum(a**m1[0]*b**m1[1]*c**m1[2]) + c2*CyclicSum(a**m2[0]*b**m2[1]*c**m2[2])
        if c1 < 0 and c2 < 0:
            return None
        m1 = CoeffMonom(*m1)
        m2 = CoeffMonom(*m2)
        if c1 + c2 >= 0:
            if c2 >= 0 and m2 >> m1:
                c1, m1, c2, m2 = c2, m2, c1, m1
            elif not (c1 >= 0 and m1 >> m2):
                return None
            sol = self._solve(m1, m2)
            if sol is None: return None
            return c1*sol + (c1 + c2)*CyclicSum(m2.as_monom(a,b,c))


    def _solve(self, m1: CoeffMonom, m2: CoeffMonom):
        funcs = [self._solve_center, self._solve_circumscribing,
                 self._solve_spiral1,
                 self._solve_spiral2] #, self._solve_amgm]
        for func in funcs:
            sol = func(m1, m2)
            if sol is not None: return sol
        return None

    def _solve_amgm(self, m1: CoeffMonom, m2: CoeffMonom):
        a, b, c = self.gens
        CyclicSum, CyclicProduct = self.cyclic_sum, self.cyclic_product

        weight = m1.weight(m2)
        m2expr = a**m2[0]*b**m2[1]*c**m2[2]
        return CyclicSum((sum(w*a**i*b**j*c**k for (i,j,k), w in weight.items()) - m2expr).together())

    def local_ineq(self, m1, p, m2) -> Optional[Tuple[Expr, int, int]]:
        """
        Solve CyclicSum(..*m1.as_monom(a,b,c) + ..*p.as_monom(a,b,c) + ..*m2.as_monom(a,b,c)) >= 0
        where m1, p, m2 are collinear.
        Return expr, x, y such that
        expr = CyclicSum(x*m1.as_monom(a,b,c) - (x+y)*p.as_monom(a,b,c) + y*m2.as_monom(a,b,c)) >= 0.

        Theorem: The polynomial inequality u*x^(u+v) - (u+v)*x^u + v >= 0 holds for x >= 0. This
        is trivial by AMGM, or by factoring out (x-1)^2.

        Moreover, when u+v is even, the inequality holds for all real number x. When u is even,
        it is trivial as it is a polynomial with respect to a^2. When u is odd, we can show that

            u*x^(u+v) - (u+v)*x^u + v >= u*x^(u+v) - (u+v)/2*x^(u+1) - (u+v)/2*x^(u-1) + v

        where the right-hand-side can be split into two even degree AMGMs. In fact, the right-
        hand-side can be factored by (a-1)^2*(a+1)^2.
        """
        a, b, c = self.gens
        CyclicSum, CyclicProduct = self.cyclic_sum, self.cyclic_product

        v1, v2 = p - m1, m2 - p
        if v1.dot(v2) > 0:
            d1, d2 = v1.norm(), v2.norm()
            dgcd = abs(gcd(d1, d2))
            d1, d2 = int(d1//dgcd), int(d2//dgcd)
            x = a
            f1 = Poly({(d1+d2,): d1, (d1,): -(d1+d2), (0,):d2}, x)
            # solve f1 >= 0

            if (d1 + d2) % 2 == 1:
                dm = divmod(f1, Poly([1,-2,1], x))
                if not dm[1].is_zero: return None # should not happen
                f1sol = dm[0].as_expr()*(x - 1)**2
            else:
                # when (d1 + d2) is even, f1 >= 0 holds for all real number x
                # since d1 and d2 is coprime, d1 must be odd
                f2 = f1 - Poly({(d1+1,): (d1+d2)//2, (d1,):-(d1+d2), (d1-1,):(d1+d2)//2}, x)
                dm = divmod(f2, Poly([1, 0, -2, 0, 1], x))
                if not dm[1].is_zero: return None # should not happen
                f1sol = dm[0].as_expr()*(x**2 - 1)**2 + (d1+d2)//2*(x - 1)**2*x**(d1 - 1)

            x2 = (v2//d2).as_monom(a, b, c)
            sol = m1.as_monom(a, b, c) * f1sol.xreplace({x: x2})
            return CyclicSum(sol.together()), d2, d1

    def _solve_circumscribing(self, m1: CoeffMonom, m2: CoeffMonom):
        p = m1.is_circumscribing(m2)
        if p is not None:
            # e.g. s(a6-a4c2)
            sol = self.local_ineq(m1, p, m1.next())
            if sol is not None: return sol[0] / (sol[1] + sol[2])

        p = m2.is_circumscribing(m1)
        if p is not None:
            # e.g. s(a7b2-a5b3c)
            sol = self.local_ineq(p, m2, m2.next())
            if sol is not None: return sol[0] / sol[1]
            sol = self.local_ineq(p, m2.next(), m2)
            if sol is not None: return sol[0] / sol[1]

    def _solve_center(self, m1: CoeffMonom, m2: CoeffMonom):
        if not m2.is_center:
            return None

        a, b, c = self.gens
        CyclicSum, CyclicProduct = self.cyclic_sum, self.cyclic_product

        d = m2[0]
        m1 = m1.std()
        if m1[1] < d and m1[2] < d:
            # in a parallelogram region
            if m1[1] <= m1[2]:
                r = m1[2]
                v = CoeffMonom(3*d-2*r, r, r)
                v2 = CoeffMonom(r, 3*d-2*r, r)
            else:
                r = m1[1]
                v = CoeffMonom(3*d-2*r, r, r)
                v2 = CoeffMonom(r, r, 3*d-2*r)
            if m1 == v:
                sol1 = (Integer(0), Integer(1), Integer(1))
            else:
                sol1 = self.local_ineq(m1, v, v2)
            if sol1 is None:
                return None
            sol2 = CyclicProduct(a**r)*CyclicSum(a**(d-r))*CyclicSum((a**(d-r)-b**(d-r))**2)/2
            return sol1[0]/sol1[1] + sol2
        elif m1[1] == d:
            r = m1[2] # (2d-r, d, r)
            s = d - r
            return CyclicProduct(a**r) * CyclicSum(a**s*(a**s+2*c**s)*(b**s-c**s)**2)/(2*CyclicSum(a**s))
        elif m1[2] == d:
            r = m1[1]
            s = d - r
            return CyclicProduct(a**r) * CyclicSum(a**s*(a**s+2*b**s)*(b**s-c**s)**2)/(2*CyclicSum(a**s))
        else: # m1[1] > d or m1[2] > d
            # e.g. s(a6b5c-a4b4c4) = s(a6b5c-a5b5c2) + s(a5b5c2-a4b4c4)
            if m1[1] > d:
                r = m1[1]
                v = CoeffMonom(r, r, 3*d-2*r)
                v2 = CoeffMonom(3*d-2*r, r, r)
            else:
                r = m1[2]
                v = CoeffMonom(r, r, 3*d-2*r)
                v2 = CoeffMonom(r, 3*d-2*r, r)
            if m1 == v:
                sol1 = (Integer(0), Integer(1), Integer(1))
            else:
                sol1 = self.local_ineq(m1, v, v2)
            if sol1 is None:
                return None
            s = r - d
            sol2 = CyclicProduct(a**(3*d-2*r))*CyclicSum(a**s*b**s)*CyclicSum(a**(2*s)*(b**s-c**s)**2)/2
            return sol1[0]/sol1[1] + sol2

    def _solve_spiral1(self, m1: CoeffMonom, m2: CoeffMonom):
        nex = m1.next()
        nexx = nex.next()
        for p1 in m2.cycle():
            p = CoeffMonom.integer_intersection(m1, p1, nex, nexx)
            if p is not None:
                if (m1 - p1).dot(p1 - p) > 0:
                    sol1 = self.local_ineq(m1, p1, p)
                    sol2 = self._solve_circumscribing(m1, p)
                    if sol1 is not None and sol2 is not None:
                        return sol1[0]/(sol1[1]+sol1[2]) + sol2*sol1[2]/(sol1[1]+sol1[2])

        # e.g. s(a8b3-a6b3c2)
        for p1, p2 in permutations(m2.cycle(), 2):
            p = CoeffMonom.integer_intersection(m1, p1, nex, p2)
            if p is not None:
                if (m1 - p1).dot(p1 - p) > 0:
                    sol1 = self.local_ineq(m1, p1, p)
                    if p.is_center:
                        sol2 = self._solve_center(m1, p)
                        if sol2 is not None:
                            return sol1[0]/(sol1[1]+sol1[2]) + sol2*sol1[2]/(sol1[1]+sol1[2])
                        return None
                    sol2 = self._solve_circumscribing(m1, p)
                    if sol1 is not None and sol2 is not None:
                        # TODO: Simplify AMGM3(self._coeff).solve(1,(9,0,4),-1,(8,1,4))
                        def merge_cyc_sums(e1, e2):
                            e = e1 + e2
                            (v1, e1), (v2, e2) = e1.as_coeff_Mul(), e2.as_coeff_Mul()
                            if not isinstance(e1, CyclicSum) or not isinstance(e2, CyclicSum): return e
                            if not isinstance(e1.args[0], Mul) or not isinstance(e2.args[0], Mul): return e
                            if e1.args[1] != e2.args[1] or e1.args[2] != e2.args[2]: return e
                            return CyclicSum((v1*e1.args[0] + v2*e2.args[0]).together(), e1.args[1], e1.args[2])
                        return merge_cyc_sums(sol1[0], sol2*sol1[2])/(sol1[1]+sol1[2])
                else:
                    sol1 = self._solve_circumscribing(m1, p)
                    sol2 = self._solve_circumscribing(p, p1)
                    if sol1 is not None and sol2 is not None:
                        return sol1 + sol2
                # return None

        for p1, p2 in combinations(m2.cycle(), 2):
            # e.g. s(a4b4-a3b3c2) = s(a4b4-a3bc4) + s(a3bc4-a3b3c2)
            p = CoeffMonom.integer_intersection(m1, nex, p1, p2)
            if p is not None:
                if (m1 - p).dot(p - nex) > 0:
                    sol1 = self.local_ineq(m1, p, nex)
                    sol2 = self._solve_circumscribing(p, p1)
                    if sol1 is not None and sol2 is not None:
                        return sol1[0] / (sol1[1] + sol1[2]) + sol2
                # return None

    def _solve_spiral2(self, m1: CoeffMonom, m2: CoeffMonom):
        m1 = m1.std()
        d = min(m2)
        if m1[1] > m1[2]:
            v = CoeffMonom(m1[0]+m1[2]-d, m1[1], d)
        else:
            v = CoeffMonom(m1[0]+m1[1]-d, d, m1[2])
        if v[0] >= 0 and v != m2 and m1 >> v and v >> m2:
            sol1 = self._solve(m1, v)
            sol2 = self._solve(v, m2)
            if sol1 is not None and sol2 is not None:
                return sol1 + sol2

def _sos_struct_sparse_amgm(coeff):
    """
    Solve
    sum coeff(large) * a^u*b^v*c^w + sum coeff(small) * a^x*b^y*c^z >= 0
    where triangle Cyclic(x,y,z) is contained in the triangle Cyclic(u,v,w).
    Also, |x-y|+|y-z|+|z-x| > 0.

    In general this is only an AM-GM inequality.

    However, we shall handle special cases more carerfully. Because AM-GM
    is sometimes not so beautiful as sum of squares.
    For example,
    s(a6-a4bc) = s(a2)s((a2-b2)2)/4+s(a2(a2-bc)2)/2 >= 0 for all real numbers a,b,c.
    """
    if not len(coeff) <= 6:
        return None
    monoms = set(CoeffMonom(*m).std() for m in coeff.keys())
    monoms = list(monoms)
    if len(monoms) == 1:
        degree = coeff.total_degree()
        monoms.append(CoeffMonom(degree, 0, 0))

    def getv(m):
        v = coeff(m)
        return v if not m.is_center else v/3
    if len(monoms) == 2:
        return AMGM3(coeff).solve(getv(monoms[0]), monoms[0], getv(monoms[1]), monoms[1])


#####################################################################
#
#                          Heuristic method
#
#####################################################################

def _acc_dict(items: List[Tuple]) -> Dict:
    """
    Accumulate the coefficients in a dictionary.
    """
    d = {}
    for k, v in items:
        if k in d:
            d[k] += v
        else:
            d[k] = v
    d = {k: v for k,v in d.items() if v != 0}
    return d

def _separate_product_wrapper(recursion: Callable, coeff: Coeff) -> Callable:
    """
    A wrapper of recursion function to avoid nested CyclicProduct(a).
    For instance, if we have CyclicProduct(a) * (CyclicProduct(a)*F(a,b,c) + G(a,b,c)),
    we had better expand it to CyclicProduct(a**2) * F(a,b,c) + CyclicProduct(a) * G(a,b,c).
    """
    from ....utils import CyclicSum, CyclicProduct
    a = coeff.gens[0]
    cg = CyclicGroup(len(coeff))
    def _extract_cyclic_prod(x: Expr) -> Tuple[int, Expr]:
        """
        Given x, return (d, r) such that x = r * CyclicProduct(a**d).
        """
        if not isinstance(x, Expr) or not (x.is_Mul or x.is_Pow or isinstance(x, CyclicProduct)):
            return 0, x
        if isinstance(x, CyclicProduct) and x.args[1] == coeff.gens and x.args[2] == cg:
            if x.args[0] == a:
                return 1, Integer(1)
            elif x.args[0].is_Pow and x.args[0].base == a:
                return x.args[0].exp, Integer(1)
        elif x.is_Pow:
            d, r = _extract_cyclic_prod(x.base)
            return d * x.exp, r**x.exp
        elif x.is_Mul:
            rs = []
            d = 0
            for arg in x.args:
                dd, r = _extract_cyclic_prod(arg)
                d += dd
                rs.append(r)
            return d, Mul(*rs)
        return 0, x

    def _new_recursion(x: Coeff, **kwargs) -> Optional[Expr]:
        x = recursion(x, **kwargs)
        if x is None:
            return x
        d, r = _extract_cyclic_prod(x)
        if r.is_Add:
            new_args = []
            for arg in r.args:
                d2, r2 = _extract_cyclic_prod(arg)
                new_args.append(CyclicProduct(a**(d + d2), coeff.gens) * r2)
            return Add(*new_args)
        return x

    return _new_recursion


class Pnrms(DomainExpr):
    """
    Represent s(a^n(b^r-c^r)) * s(a^m(b^s-c^s)).
    """
    def coeff(self, n, r, m, s, v = 1) -> Coeff:
        return self._coeff.from_dict(_acc_dict([
            ((r + s, m + n, 0), v), ((r + s, 0, m + n), v), ((m + n, r + s, 0), v), ((m + n, 0, r + s), v),
            ((0, r + s, m + n), v), ((0, m + n, r + s), v), ((m + r, n + s, 0), -v), ((m + r, 0, n + s), -v),
            ((n + s, m + r, 0), -v), ((n + s, 0, m + r), -v), ((0, m + r, n + s), -v), ((0, n + s, m + r), -v),
            ((r, m, n + s), v), ((r, n + s, m), v), ((s, n, m + r), v), ((s, m + r, n), v),
            ((m, r, n + s), v), ((m, n + s, r), v), ((n, s, m + r), v), ((n, m + r, s), v),
            ((m + r, s, n), v), ((m + r, n, s), v), ((n + s, r, m), v), ((n + s, m, r), v),
            ((r, s, m + n), -v), ((r, m + n, s), -v), ((s, r, m + n), -v), ((s, m + n, r), -v),
            ((m, n, r + s), -v), ((m, r + s, n), -v), ((n, m, r + s), -v), ((n, r + s, m), -v),
            ((r + s, m, n), -v), ((r + s, n, m), -v), ((m + n, r, s), -v), ((m + n, s, r), -v),
        ]))

    def as_expr(self, n, r, m, s, v = 1) -> Expr:
        a, b, c = self.gens
        CyclicSum, CyclicProduct = self.cyclic_sum, self.cyclic_product

        if n == r or m == s or r == 0 or s == 0:
            return Integer(0)
        sol = None
        if n == m and r == s:
            sol = CyclicSum(a**n * (b**r - c**r))**2
        f1, f2 = self._half_side(n, r), self._half_side(m, s)
        if f1 is not None and f2 is not None:
            sol = CyclicProduct((a-b)**2) * f1 * f2
        else:
            sol = CyclicSum(a**n * (b**r - c**r)) * CyclicSum(a**m * (b**s - c**s))
        return v * sol

    def _half_side(self, n, r) -> Expr:
        """
        Return s(a^n(b^r-c^r)) / [(b-a)(c-b)(a-c)]
        """
        a, b, c = self.gens
        CyclicSum, CyclicProduct = self.cyclic_sum, self.cyclic_product

        if n + r <= 12:
            RESULT = {
                (2, 1): Integer(1),
                (3, 1): CyclicSum(a),
                (4, 1): CyclicSum(a**2+b*c),
                (5, 1): CyclicProduct(a) + CyclicSum(a) * CyclicSum(a**2),
                (6, 1): CyclicSum(a**2*(a+b)*(a+c)) + CyclicSum(a**2*b**2),
                (7, 1): CyclicSum(a**3*(a+b)*(a+c)) + CyclicSum(a) * CyclicSum(a**2*b**2),
                (8, 1): CyclicSum(a**2*(2*a**2+b**2+c**2)*(a+b)*(a+c))/2 + CyclicProduct(a**2+b**2)/2, # s(a2(2a2+b2+c2)(a+b)(a+c))/2+1/2(b2+c2)(a2+b2)(a2+c2)
                (9, 1): CyclicSum(a**3*(2*a**2+b**2+c**2)*(a+b)*(a+c))/2 + CyclicSum(a) * CyclicProduct(a**2+b**2)/2, # s(a3(2a2+b2+c2)(a+b)(a+c))/2+1/2s(a)(b2+c2)(a2+b2)(a2+c2)
                (3, 2): CyclicSum(a*b),
                (4, 2): CyclicProduct(a+b),
                (5, 2): CyclicSum(a*b) * CyclicSum(a**2) + CyclicSum(a**2*(b+c)**2) / 2,
                (6, 2): CyclicProduct(a+b) * CyclicSum(a**2),
                (7, 2): CyclicProduct(a+b) * (CyclicProduct(a) + CyclicSum(a**3)) + CyclicSum(a**3*b**3),
                (8, 2): CyclicProduct(a+b) * CyclicSum(a**4 + b**2*c**2),
                (4, 3): CyclicSum(a**2*(b+c)**2) / 2,
                (5, 3): CyclicSum(a) * CyclicSum(a**2*b**2) + CyclicSum((a+b)**2) * CyclicProduct(a) / 2,
                (6, 3): CyclicProduct(a**2+a*b+b**2),
                (7, 3): CyclicSum(a**3*(b**2+b*c+c**2)*(a+b)*(a+c)) + 2 * CyclicSum(a) * CyclicProduct(a**2),
                (8, 3): CyclicSum(a**4*(b**2+c**2)*(a+b)*(a+c)) + CyclicSum(a**2*b*(a+c)) * CyclicSum(a**2*c*(a+b)),
                (5, 4): CyclicSum(a**2*b**2) * CyclicSum(a*b) + CyclicProduct(a**2),
                (6, 4): CyclicProduct(a+b) * CyclicSum(a**2*b**2),
                (7, 4): 2*CyclicProduct(a**2)*CyclicSum(a**2) + CyclicSum(a*b)*(CyclicSum(a**3*b**3) + CyclicSum(a**2)*CyclicSum(a**2*b**2)), # 2p(a2)s(a2)+s(ab)(s(a3b3)+s(a2)s(a2b2)),
                (8, 4): CyclicProduct(a+b) * CyclicProduct(a**2+b**2),
                (6, 5): CyclicSum(b**3*c**3*(a+b)*(a+c)) + CyclicSum(a**2) * CyclicProduct(a**2)
            }
            return RESULT.get((n, r), None)


class Hnmr(DomainExpr):
    """
    Represent s(a^n*(b^m+c^m)*(a^r-b^r)*(a^r-c^r)) when n >= 0,
    and s(b^(-n)*c^(-n)*(b^m+c^m))*(a^r-b^r)*(a^r-c^r)) when n < 0.

    We require m >= 0 and r >= 0 always.

    We have recursion identity:
    H(n,m,r) = P(2r,r,n,m-r) + (a^r*b^r*c^r) * H(n-r,m-2r,r)

    Also, we have inverse substituion (n,m,r) -> (m-n-r,m,r).
    When n <= 0 or n + r >= m, we have H(n,m,r) >= 0.
    """

    def coeff(self, n, m, r, v = 1) -> Coeff:
        if n >= 0:
            coeffs = [
                ((m, n + 2*r, 0), v), ((m, 0, n + 2*r), v), ((n + 2*r, m, 0), v), ((n + 2*r, 0, m), v),
                ((0, m, n + 2*r), v), ((0, n + 2*r, m), v), ((m + r, n + r, 0), -v), ((m + r, 0, n + r), -v),
                ((n + r, m + r, 0), -v), ((n + r, 0, m + r), -v), ((0, m + r, n + r), -v), ((0, n + r, m + r), -v),
                ((r, n, m + r), v), ((r, m + r, n), v), ((n, r, m + r), v), ((n, m + r, r), v),
                ((m + r, r, n), v), ((m + r, n, r), v), ((r, m, n + r), -v), ((r, n + r, m), -v),
                ((m, r, n + r), -v), ((m, n + r, r), -v), ((n + r, r, m), -v), ((n + r, m, r), -v)
            ]
        else:
            n = -n
            coeffs = [
                ((n + r, m + n + r, 0), v), ((n + r, 0, m + n + r), v), ((m + n + r, n + r, 0), v), ((m + n + r, 0, n + r), v),
                ((0, n + r, m + n + r), v), ((0, m + n + r, n + r), v), ((n, 2*r, m + n), v), ((n, m + n, 2*r), v),
                ((2*r, n, m + n), v), ((2*r, m + n, n), v), ((m + n, n, 2*r), v), ((m + n, 2*r, n), v),
                ((r, n, m + n + r), -v), ((r, n + r, m + n), -v), ((r, m + n, n + r), -v), ((r, m + n + r, n), -v),
                ((n, r, m + n + r), -v), ((n, m + n + r, r), -v), ((n + r, r, m + n), -v), ((n + r, m + n, r), -v),
                ((m + n, r, n + r), -v), ((m + n, n + r, r), -v), ((m + n + r, r, n), -v), ((m + n + r, n, r), -v)
            ]
        return self._coeff.from_dict(_acc_dict(coeffs))

    def as_expr(self, n, m, r, v = 1) -> Expr:
        a, b, c = self.gens
        CyclicSum, CyclicProduct = self.cyclic_sum, self.cyclic_product
        coeff = self._coeff

        if r == 0:
            return Integer(0)
        sol = None
        if n >= 0:
            if n == m:
                sol = CyclicSum(a**n*b**n*(a**r-b**r)**2)
            elif n == m - r:
                if n >= r:
                    sol = CyclicSum(a**(n-r)*b**(n-r)*(a**r-b**r)**2) * CyclicProduct(a**r)
                else: # if n < r:
                    sol = CyclicSum(c**(r-n)*(a**r-b**r)**2) * CyclicProduct(a**n)
            elif m == 2*r and n >= r:
                sol = Pnrms(coeff).as_expr(2*r, r, n, m - r) + CyclicProduct(a**r) * self.as_expr(n - r, m - 2*r, r)
            elif m == r:
                if n % r == 0 and (n // r) <= 8:
                    RESULT = {
                        2: CyclicSum(a*(b-c)**2*(b+c-a)**2),
                        4: CyclicSum(a*(b-c)**2*(b+c-a)**4) + 2 * CyclicSum(a) * CyclicProduct((a-b)**2),
                        5: CyclicSum(a*b*(a-b)**2*(a**2+b**2-c**2)**2) + CyclicProduct((a-b)**2) * CyclicSum((a-b)**2) / 2,
                        6: CyclicSum(a*(b-c)**2*(b+c-a)**2*(b**2+c**2-a**2)**2) + CyclicProduct((a-b)**2) * (3*CyclicSum(a*(b-c)**2) + 22*CyclicProduct(a)),
                        7: CyclicSum(a*b*(a-b)**2*(a**3 + 2*a*b*c - 2*a*c**2 + b**3 - 2*b*c**2 + c**3)**2) + CyclicProduct((a-b)**2) * (CyclicSum(a*b)*CyclicSum(a**2) + CyclicSum((a**2-b**2)**2)/2),
                        8: CyclicSum(a*(b-c)**2*(b+c-a)**2*(b**3+c**3-a**3)**2) + CyclicProduct((a-b)**2) * (CyclicSum(a*b*(a+b))*CyclicSum(a)**2 + CyclicSum((a-b)**2) * CyclicProduct(a))
                    }
                    sol = RESULT.get(n // r, None)
                    if sol is not None:
                        sol = sol.xreplace({a:a**r, b:b**r, c:c**r})
                elif n == 1 and r >= 2:
                    sol = Pnrms(coeff).as_expr(r+1, 1, r, r-1) + CyclicProduct(a) * CyclicSum(c**(r-2)*(a**r-b**r)**2)
                else:
                    d = gcd(n, r)
                    RESULT = {
                        (1, 2): CyclicSum(a) * CyclicProduct((a-b)**2) + CyclicProduct(a) * CyclicSum((a-b)**2),
                        (3, 2): CyclicProduct((a-b)**2) * (CyclicSum(a)*CyclicSum(a**2) + CyclicProduct(a+b)) + CyclicProduct(a) * CyclicSum((a**2-b**2)**2*(a+b-c)**2), # (s(a)s(a2)+p(a+b))p(a-b)2+p(a)s((a2-b2)2(a+b-c)2)
                        (5, 2): CyclicProduct((a-b)**2) * (CyclicSum(a)**2*CyclicSum(a**3) + CyclicSum(a*b)*CyclicProduct(a+b)) + CyclicProduct(a) * CyclicSum((a**2-b**2)**2*(a**2+b**2-c**2)**2), # p(a-b)2(s(a3)s(a)2+p(a+b)s(ab))+s((a2-b2)2(a2+b2-c2)2)p(a)
                    }
                    sol = RESULT.get((n // d, r // d), None)
                    if sol is not None:
                        sol = sol.xreplace({a:a**d, b:b**d, c:c**d})
                if sol is None:
                    if n >= m + r:
                        numerator = CyclicSum(a**n*(b**m+c**m)*(a**r-b**r)**2*(a**r-c**r)**2) + Pnrms(coeff).as_expr(2*r, r, n, m + r)
                        denominator = CyclicSum((a**r - b**r)**2) / 2
                        sol = numerator / denominator
                    else:
                        if n >= r:
                            numerator = CyclicSum(a**(n-r)*(b**m+c**m)*(a**r-b**r)**2*(a**r-c**r)**2) * CyclicProduct(a**r)
                        else:
                            numerator = CyclicSum(b**(r-n)*c**(r-n)*(b**m+c**m)*(a**r-b**r)**2*(a**r-c**r)**2) * CyclicProduct(a**n)
                        numerator += Pnrms(coeff).as_expr(2*r, r, n + m + r, m + r)
                        denominator = CyclicSum(a**(2*r)*(b**r - c**r)**2) / 2
                        sol = numerator / denominator
            if sol is None:
                sol = CyclicSum(a**n*(b**m+c**m)*(a**r-b**r)*(a**r-c**r))
        else:
            n = -n
            if n == r:
                sol = CyclicSum(c**(m+2*r)*(a**r-b**r)**2)
            if sol is None:
                sol = CyclicSum(b**n*c**n*(b**m+c**m)*(a**r-b**r)*(a**r-c**r))
        return v * sol



def sos_struct_heuristic(coeff: Coeff, real=True):
    """
    Solve high-degree but sparse inequalities by heuristic method.
    It subtracts some structures from the inequality and calls
    the recursion function to solve the problem.

    WARNING: Only call this function when degree > 6. And make sure that
    coeff.clear_zero() to remove zero terms on the border.

    Examples
    -------
    s(ab(a-b)2(a4-3a3b+2a2b2+3b4))

    s(c8(a-b)2(a4-3a3b+2a2b2+3b4))
    """
    degree = coeff.total_degree()
    # assert degree > 6, "Degree must be greater than 6 in heuristic method."
    if degree <= 6:
        return None
    if True:
        solution = sos_struct_dense_symmetric(coeff, real = real)
        if solution is not None:
            return solution

    from .solver import _structural_sos_3vars_cyclic
    recursion = _structural_sos_3vars_cyclic
    recursion = _separate_product_wrapper(recursion, coeff)

    if coeff((degree, 0, 0)):
        # not implemented
        return None

    monoms = coeff.terms() # this is sorted
    if monoms[0][1] < 0 or monoms[-1][1] < 0 or monoms[-1][0][0] != 0:
        return None

    a, b, c = coeff.gens
    CyclicSum, CyclicProduct = coeff.cyclic_sum, coeff.cyclic_product

    border1, border2 = [], []
    for (i,j,k), v in monoms[::-1]:
        if i != 0:
            break
        if v != 0:
            border1.append((j, v))
    border1 = border1[::-1]
    i0 = monoms[0][0][0]
    for (i,j,k), v in monoms:
        if i != i0:
            break
        if v != 0:
            border2.append((j, v))

    if len(border1) == 1 and border1[0][0] * 2 == degree:
        return None

    # print('Coeff =' , coeff.as_poly())
    # print('Border1 =', border1)
    # print('Border2 =', border2)

    for border in (border1, border2):
        if len(border) * 3 == len(coeff):
            # all coefficients are on the border
            border_poly = Poly.from_dict(dict(border), gens = (a,))
            border_proof = prove_univariate(border_poly, (0, None), return_type='expr')
            if border_proof is not None:
                if border is border1:
                    border_proof = border_proof.xreplace({a: c}).xreplace({c: a / b}).together() * b**degree
                else:
                    border_proof = border_proof.xreplace({a: b / c}).together() * a**i0 * c**(degree - i0)
                border_proof = CyclicSum(border_proof)
            return border_proof

    c0 = border1[0][1]
    c0_ = border1[-1][1]
    if c0 < 0 or c0_ < 0:
        return None
    if border1[0][0] + border1[-1][0] == degree and c0 == c0_ and i0 == border1[0][0]:
        if len(border1) <= 4 and len(border2) <= 4:
            # symmetric hexagon
            gap11, gap12, gap21, gap22 = -1, -1, -1, -1
            if len(border1) >= 3 and border1[1][0] + border1[-2][0] == degree:
                gap11 = border1[0][0] - border1[1][0]
                gap12 = border1[0][0] - border1[-2][0]
            elif len(border1) <= 2:
                gap11 = 0
                gap12 = 0

            if len(border2) >= 3 and border2[1][0] + border2[-2][0] == degree - i0:
                gap21 = border2[0][0] - border2[1][0]
                gap22 = border2[0][0] - border2[-2][0]
            elif len(border2) <= 2:
                gap21 = 0
                gap22 = 0

            # print('Symmetric Hexagon Gap =', (gap11, gap12, gap21, gap22))
            if gap11 != -1 and gap21 != -1:
                if gap11 != 0 and gap21 != 0:
                    r_, s_ = gap21, gap22
                    for n_, m_ in ((r_ + gap11, s_ + gap12), (r_ + gap12, s_ + gap11)):
                        # print('>> s(a%d(b%d-c%d))s(a%d(b%d-c%d))' % (n_, r_, r_, m_, s_, s_))

                        solution = recursion(coeff - Pnrms(coeff).coeff(n_, r_, m_, s_, c0), real = False)
                        if solution is not None:
                            return solution + Pnrms(coeff).as_expr(n_, r_, m_, s_, c0)
                elif gap11 != 0 and gap21 == 0:
                    for r_ in (gap11, gap12):
                        n_ = border1[0][0] - 2*r_
                        m_ = degree - border1[0][0]
                        if n_ >= 0 and m_ >= 0 and m_ <= n_ + r_:
                            # print('>> s(a%d(b%d+c%d)(a%d-b%d)(a%d-c%d))' % (n_, m_, m_, r_, r_, r_, r_))

                            solution = recursion(coeff - Hnmr(coeff).coeff(n_, m_, r_, c0 if m_ else c0/2), real = False)
                            if solution is not None:
                                return solution + Hnmr(coeff).as_expr(n_, m_, r_, c0 if m_ else c0/2)

                        if m_ > r_ and n_ + r_ > m_:
                            # print('>> s(a%d(b%d-c%d))s(a%d(b%d-c%d))' % (2*r_, r_, r_, n_, m_-r_, m_-r_))

                            solution = recursion(coeff - Pnrms(coeff).coeff(2*r_, r_, n_, m_-r_, c0), real = False)
                            if solution is not None:
                                return solution + Pnrms(coeff).as_expr(2*r_, r_, n_, m_-r_, c0)

                elif gap11 == 0 and gap21 != 0:
                    for r_ in (gap21, gap22):
                        n_ = degree - border1[0][0] - r_
                        m_ = 2 * border1[0][0] - degree
                        if n_ >= 0 and m_ >= 0:
                            # print('>> s(b%dc%d(b%d+c%d)(a%d-b%d)(a%d-c%d))' % (n_, n_, m_, m_, r_, r_, r_, r_))

                            solution = recursion(coeff - Hnmr(coeff).coeff(-n_, m_, r_, c0 if m_ else c0/2), real = False)
                            if solution is not None:
                                return solution + Hnmr(coeff).as_expr(-n_, m_, r_, c0 if m_ else c0/2)

                        # if m_ >= r_:
                        #     print('>> s(a%d(b%d-c%d))s(a%d(b%d-c%d))' % (2*r_, r_, r_, n_, m_-r_, m_-r_))

                        #     solution = recursion(coeff - Pnrms(coeff).coeff(2*r_, r_, n_, m_-r_, c0), real = False)
                        #     if solution is not None:
                        #         return solution + Pnrms(coeff).coeff(2*r_, r_, n_, m_-r_, c0)


    return None
