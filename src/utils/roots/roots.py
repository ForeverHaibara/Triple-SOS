from typing import Optional, Union, Tuple

import numpy as np
import sympy as sp

from .rationalize import rationalize
from ..basis_generator import generate_expr

def _reg_matrix(M):
    # normalize so that the largest entry in each column is 1
    reg = np.max(np.abs(M), axis = 0)
    reg = 1 / np.tile(reg, (M.shape[0], 1))
    reg = sp.Matrix(reg)
    M = M.multiply_elementwise(reg)
    return M

def _algebraic_extension(vec):
    if isinstance(vec[0], sp.Rational):
        return [vec]
    field = vec[0].mod

    # if len(field) == 3 and field[0] == 1 and field[1] == 0:
    #     f = lambda x: sp.Rational(x.numerator, x.denominator)
    #     return [[f(x.rep[0])  if len(x.rep) == 2 else sp.S(0) for x in vec], 
    #             [f(x.rep[-1]) if len(x.rep) else sp.S(0) for x in vec]]

    f = lambda x: sp.Rational(x.numerator, x.denominator)
    vecs = []
    for i in range(1, len(field)):
        vecs.append([f(x.rep[-i]) if len(x.rep) >= i else sp.S(0) for x in vec])
    return vecs

def _derv(n, i):
    # compute n*(n-1)*..*(n-i+1)
    if i == 1: return n
    if n < i:
        return 0
    return sp.prod((n - j) for j in range(i))

class Root():
    """
    Cyclic root of CyclicSum((a^2-b^2+u(ab-ac)+v(bc-ab))^2).
    Clearly, it should satisfy a^2-b^2+u(ab-ac)+v(bc-ab) = 0 and its permutations.
    For example, Vasile inequality is the case of (u, v) = (1, 2).

    When uv = 1, it degenerates to a root on border, (u, 0, 1) and permutations.
    When uv != 1, the root is computed by:
    x = ((v - u)(uv + u + v - 2) + u^3 + 1)/(1 - uv)
    y = ((u - v)(uv + u + v - 2) - v^3 - 1)/(1 - uv)

    Then b/c is a root of t^3 + xt^2 + yt - 1 = 0.
    And a/c is determined by a/c = ((b/c)^2 + (b/c)(u - v) - 1)/((b/c)u - v).
    """
    __slots__ = ('root', '_uv', '_ker', '_cyclic_sum_cache')

    def __new__(cls, root):
        """
        Return RootRational if each entry of root is rational.
        """
        if all(isinstance(r, (sp.Rational, int)) for r in root):
            return super().__new__(RootRational)
        return super().__new__(cls)

    def __init__(self, root):
        if len(root) == 2:
            root = root + (1,)
        self.root = root
        self._uv = None
        self._ker = None
        self._cyclic_sum_cache = {}

    def __getitem__(self, i):
        return self.root[i]

    @classmethod
    def from_uv(cls, uv: Tuple[sp.Expr, sp.Expr]) -> 'Root':
        """
        Construct a root from u, v.
        """
        u, v = uv
        u, v = sp.S(u), sp.S(v)
        ker = (u*u - u*v + v*v + u + v + 1)
        sab = (u + v - 1) / ker
        abc = (u*v - 1) / ker**2
        x = sp.symbols('x')
        poly = (x**3 - x**2 + sab * x - abc).as_poly(x)
        if cls is RootAlgebraic:
            if poly.domain is sp.QQ or poly.domain is sp.ZZ:
                a, b, c = poly.all_roots()
            else:
                raise ValueError('RootAlgebraic.from_uv currently does not support irrational generating polynomials.')
        else:
            a, b, c = sp.polys.nroots(poly)
        root = cls((a, b, c))
        u_, v_ = root.uv()
        if abs(u_ - u) + abs(v_ - v) > abs(v_ - u) + abs(u_ - v):
            a, b, c = c, b, a
            root = cls((a, b, c))
        if not root.__class__ is RootAlgebraic:
            root._uv = (u, v)
            root._ker = ker
        return root

    @property
    def is_corner(self):
        if self.root[0] != 0:
            return self.root[2] == 0 and self.root[1] == 0
        elif self.root[1] != 0:
            return self.root[2] == 0 and self.root[0] == 0
        elif self.root[2] != 0:
            return self.root[0] == 0 and self.root[1] == 0
        return True

    @property
    def is_border(self):
        return self.root[0] == 0 or self.root[1] == 0 or self.root[2] == 0

    @property
    def is_symmetric(self):
        return self.root[0] == self.root[1] or self.root[1] == self.root[2] or self.root[0] == self.root[2]

    @property
    def is_centered(self):
        return self.root[0] == self.root[1] == self.root[2]

    @property
    def is_nontrivial(self):
        return not self.is_border and not self.is_symmetric

    def uv(self):
        if self._uv is None:
            if self.is_centered:
                self._uv = (sp.S(2), sp.S(2))
                return self._uv
            elif self.is_corner:
                self._uv = (sp.oo, sp.oo)
                return self._uv

            a, b, c = self.root
            if c == 0:
                a, b, c = b, c, a
            if c != 1:
                a, b = a/c, b/c
            t = ((a*b-a)*(a-b)-(b*(a-1))**2)

            # basic quadratic form
            u = ((b*b-a*a)*(a-b) - (b-a*b)*(1-b*b))/t
            v = ((a*b-a)*(1-b*b) - (b*b-a*a)*(b-b*a))/t

            # u, v = rationalize(u, reliable = True), rationalize(v, reliable = True)

            self._uv = (u, v)
        return self._uv

    def __str__(self):
        return str(self.root)

    def __repr__(self):
        return repr(self.root)

    def __hash__(self):
        return hash(self.root)

    def __eq__(self, other):
        def sd(x):
            r = x.root
            s = sum(r)
            return (r[0]/s, r[1]/s, r[2]/s)
        return sd(self) == sd(other)

    def _permuted_root(self, permute = 0):
        a, b, c = self.root
        if permute == 1:
            a, b, c = b, c, a
        elif permute == 2:
            a, b, c = c, a, b
        return a, b, c

    def standardize(self, cyc: bool = False, inplace: bool = False) -> 'Root':
        """
        Standardize the root by setting a + b + c == 3.

        Parameters
        ----------
        cyc : bool
            Whether to standardize cyclically. If True, return
            (a,b,c) such that a=max(a,b,c). If a == max(b,c), then
            return (a,b,c) such that b=max(b,c).

        inplace : bool
            Whether to modify the root inplace. If False, return
            a new Root object. If True, return self.

        Returns
        ----------
        root : Root
            The standardized root. If inplace is True, return self
            after modification.
        """
        if self.is_centered:
            root = (sp.S(1), sp.S(1), sp.S(1))
        else:
            s = sum(self.root) / 3
            root = (self.root[0] / s, self.root[1] / s, self.root[2] / s)
            if cyc:
                if not self.is_symmetric:
                    m = max(root)
                    for i in range(3):
                        if root[i] == m:
                            root = (root[i], root[(i+1)%3], root[(i+2)%3])
                            break
                else:
                    m = max(root)
                    for i in range(3):
                        if root[i] == m and root[(i+2)%3] < m:
                            root = (root[i], root[(i+1)%3], root[(i+2)%3])
                            break

        if inplace:
            self.root = root
            return self
        return Root(root)

    def eval(self, poly, rational = False) -> sp.Expr:
        """
        Evaluate poly(*root). The function only accepts sympy
        polynomials as input. For more complicated usage,
        please use subs().

        Parameters
        ----------
        poly : sympy.Poly
            The polynomial to be evaluated on.
        rational : bool
            Whether to convert the root to a Rational number and evaluate. This
            boosts the performance.
        """
        root = self.root
        if rational:
            root = tuple(map(sp.Rational, root))
        return poly(*root)

    def ker(self):
        """
        Return u^2 - uv + v^2 + u + v + 1.
        """
        if self._ker is None:
            u, v = self.uv()
            self._ker = (u*u - u*v + v*v + u + v + 1)
        return self._ker

    def cyclic_sum(self, monom: Tuple[int, int, int]) -> sp.Expr:
        """
        Return CyclicSum(a**i * b**j * c**k) assuming standardlization a + b + c = 1,
        that is, s(a^i*b^j*c^k) / s(a)^(i+j+k).

        Note: when i == j == k, it will return 3 * p(a^i). Be careful with the coefficient.
        """
        i, j, k = monom
        m = min(monom)
        u, v = self.uv()
        if m >= 1:
            # is multiple of abc
            return ((u*v - 1) / self.ker()**2) ** m * self.cyclic_sum((i-m, j-m, k-m))

        if k != 0:
            if j == 0:
                i, j, k = k, i, j
            elif i == 0:
                i, j, k = j, k, i
        if i == 0:
            i, j = j, i

        s = self._cyclic_sum_cache.get((i,j,k), None)
        if s is not None:
            return s

        m = max(monom)
        if m >= 3:
            # can reduce the degree by poly remainder
            a, b = sp.symbols('a b')
            mod_poly = self.poly(a)
            poly_a = (a**i).as_poly(a) % mod_poly
            poly_b = (b**j).as_poly(b) % mod_poly.replace(a, b)
            s = sp.S(0)
            for term1 in poly_a.terms():
                for term2 in poly_b.terms():
                    s += term1[1] * term2[1] * self.cyclic_sum((term1[0][0], term2[0][0], 0))
            self._cyclic_sum_cache[(i,j,k)] = s
            return s


        if i == 0:
            if j == 0:
                s = sp.S(3)
            elif j == 1:
                s = sp.S(1)
            elif j == 2:
                s = 1 - 2 * self.cyclic_sum((1,1,0))
        elif i == 1:
            if j == 0:
                s = sp.S(1)
            elif j == 1:
                s = (u + v - 1) / self.ker()
            elif j == 2:
                s = ((v - u) * (u*v + u + v - 2) + u**3 + 1) / self.ker()**2
        elif i == 2:
            if j == 0:
                s = 1 - 2 * self.cyclic_sum((1,1,0))
            elif j == 1:
                s = ((u - v) * (u*v + u + v - 2) + v**3 + 1) / self.ker()**2
            elif j == 2:
                s = (u**2 + v**2 - 2*(u+v) + 3) / self.ker()**2
        self._cyclic_sum_cache[(i,j,k)] = s
        return s

    def poly(self, x = None):
        """
        Return a polynomial whose three roots are (proportional to) a, b, c.
        """
        if x is None:
            x = sp.symbols('x')
        poly = x**3 - x**2 + self.cyclic_sum((1,1,0)) * x - self.cyclic_sum((1,1,1)) / 3
        return poly.as_poly(x)

    def as_vec(self,
            n: int,
            diff: Optional[Tuple[int,int,int]] = None,
            cyc: bool = False,
            permute: bool = 0,
            numer: bool = False
        ) -> Union[sp.Matrix, np.ndarray]:
        """
        Construct the vector of all monomials of degree n. For example, for n = 3,
        return f(a,b,c) = [a^3,a^2*b,a^2*c,a*b^2,a*b*c,a*c^2,b^3,b^2*c,b*c^2,c^3].

        Parameters
        ----------
        n : int
            The degree of the monomials.
        diff : Tuple[int,int,int]
            If diff is not None, take partial derivative of the monomials of
            certain degrees.
        cyc : bool
            If cyc, only reserve one of the cyclic permutations of the monomials.
        permute : int
            If permute, permute the root first by permute times. For example, if
            permute = 1, then return f(b,c,a).
        numer : bool
            If numer, return a numpy array instead of sympy Matrix.
        """
        monoms = generate_expr(3, n, cyc = cyc)[1]
        a, b, c = self._permuted_root(permute)

        if diff is None:
            vec = ([a**i*b**j*c**k for i, j, k in monoms])
        else:
            u, v, w = diff
            vec = [0] * len(monoms)
            for ind, (i, j, k) in enumerate(monoms):
                if i < u or j < v or k < w:
                    continue
                vec[ind] = _derv(i,u)*_derv(j,v)*_derv(k,w)*a**(i-u)*b**(j-v)*c**(k-w)

        if numer:
            vec = np.array(vec).astype(np.float64)
        else:
            vec = sp.Matrix(vec)
        return vec

    def span(self, n: int) -> sp.Matrix:
        """
        Construct the space spanned by the root and its cyclic permutaton of degree n.
        In general, it is a matrix with 3 columns.
        For example, for n = 3, f(a,b,c) = [a^3,a^2*b,a^2*c,a*b^2,a*b*c,a*c^2,b^3,b^2*c,b*c^2,c^3],
        then sum(f(a,b,c)), sum(a*f(a,b,c)) and sum(a^2*f(a,b,c)) are the three columns.
        """
        monoms = generate_expr(3, n, cyc = False)[1]
        if self.is_centered:
            return sp.ones(len(monoms), 1)

        u, v = self.uv()
        if u == v:
            # on the symmetric axis, the three roots are (1,1,u-1)
            vecs = [None, None, None]
            a, b, c = (1, 1, u - 1) if u != sp.oo else (0, 0, 1)
            for column in range(3):
                a, b, c = c, a, b
                vec = [0] * len(monoms)
                for ind, (i, j, k) in enumerate(monoms):
                    vec[ind] = a**i * b**j * c**k
                vecs[column] = sp.Matrix(vec)
            
        else:
            vecs = [None, None, None]
            for column in range(3):
                vec = [0] * len(monoms)
                for ind, (i, j, k) in enumerate(monoms):
                    vec[ind] = self.cyclic_sum((i + column, j, k))
                vecs[column] = sp.Matrix(vec)

        M = sp.Matrix.hstack(*vecs)

        return _reg_matrix(M)

    def subs(self, expr: Union[sp.Expr, sp.Matrix]) -> Union[sp.Expr, sp.Matrix]:
        """
        Substitute the root into an expression or a matrix of expressions.
        """
        a, b, c = sp.symbols('a b c')
        return expr.subs({a: self.root[0], b: self.root[1], c: self.root[2]})

    def approximate(self, tolerance = 1e-3, approximate_tolerance = 1e-6):
        """
        For roots on border / symmetric axis, we can approximate the root.
        Only supports constant roots.
        """
        def n(a, b, tol = tolerance):
            # whether a, b are close enough
            return abs(a - b) < tol
        a, b, c = self.root
        s = (a + b + c) / 3
        r = [a/s, b/s, c/s]
        if n(r[0], r[1]) and n(r[1], r[2]) and n(r[0], r[2]):
            return RootRational((1,1,1))

        r = [0 if n(x, 0) else x for x in r]
        if sum(x != 0 for x in r) == 1:
            # two of them are zero
            r = [0 if x == 0 else 1 for x in r]
            return RootRational(r)

        perms = [(0,1,2),(1,2,0),(2,0,1)]
        for perm in perms:
            if n(r[perm[0]], r[perm[1]]):
                r[perm[2]] = r[perm[2]] / r[perm[0]]
                r[perm[0]] = r[perm[1]] = 1
            elif n(r[perm[0]], 0):
                r[perm[2]] = r[perm[2]] / r[perm[1]]
                r[perm[1]] = 1

        for i in range(3):
            v = rationalize(r[i], rounding=.1, reliable=False)
            if n(v, r[i], approximate_tolerance):
                r[i] = v

        return Root((r[0], r[1], r[2]))


class RootAlgebraic(Root):
    """
    Root of a 3-var homogeneous cyclic polynomial.
    """
    __slots__ = ('root', '_uv', '_ker', 'K', '_cyclic_sum_cache', 'root_anp', '_inv_sum', '_power_cache')
    def __new__(cls, root):
        return super().__new__(cls, root)

    def __init__(self, root):
        root = tuple(sp.S(r) for r in root)
        if len(root) == 2:
            root = root + (sp.S(1),)
        self.root = root

        self.K = None
        self.root_anp = None
        for i in range(3):
            if not isinstance(root[i], sp.Rational):
                self.K = sp.QQ.algebraic_field(root[i])
                try:
                    self.root_anp = [
                        self.K.from_sympy(r) for r in root
                    ]
                except: # Coercion Failed
                    self.K = None
                if self.K is not None:
                    break
        if self.root_anp is None:
            self.root_anp = self.root

        if self.is_centered:
            self._uv = (sp.S(2), sp.S(2))
        elif self.is_corner:
            self._uv = (sp.oo, sp.oo)
        else:
            self._uv = self._initialize_uv(self.root_anp)
        u, v = self._uv
        self._ker = (u**2 - u*v + v**2 + u + v + 1)

        if self.K is not None:
            self._inv_sum = self.K.from_sympy(sp.S(1)) / sum(self.root_anp)
        else:
            self._inv_sum = (sp.S(1)) / sum(self.root_anp)

        self._power_cache = {0: {}, 1: {}, 2: {}, -1: {}}

    @classmethod
    def _initialize_uv(cls, root):
        a, b, c = root
        if c != 1:
            if c == 0 or (hasattr(c, 'rep') and len(c.rep) == 0):
                if (isinstance(b, sp.Rational) and b != 0) or (hasattr(b, 'rep') and len(b.rep) > 0):
                    a, b, c = c, a, b
                elif (isinstance(a, sp.Rational) and a != 0) or (hasattr(a, 'rep') and len(a.rep) > 0):
                    a, b, c = b, c, a
            a, b = a/c, b/c
        t = ((a*b-a)*(a-b)-(b*(a-1))**2)

        # basic quadratic form
        u = ((b*b-a*a)*(a-b) - (b-a*b)*(1-b*b))/t
        v = ((a*b-a)*(1-b*b) - (b*b-a*a)*(b-b*a))/t
        return u, v
    
    def uv(self, to_sympy = True):
        if self.K is None or not to_sympy:
            return self._uv
        return (self.K.to_sympy(self._uv[0]), self.K.to_sympy(self._uv[1]))

    def ker(self):
        if self.K is None:
            return self._ker
        return self.K.to_sympy(self._ker)

    def to_sympy(self, *args, **kwargs):
        return self.K.to_sympy(*args, **kwargs)

    def from_sympy(self, *args, **kwargs):
        return self.K.from_sympy(*args, **kwargs)

    def __str__(self):
        if self.K is None:
            return str(self.root)
        return str(tuple(r.n(15) if not isinstance(r, sp.Rational) else r for r in self.root))

    def __repr__(self):
        if self.K is None:
            return str(self.root)
        return str(tuple(r.n(15) if not isinstance(r, sp.Rational) else r for r in self.root))

    def _single_power(self, i: int, degree: int) -> sp.polys.polyclasses.ANP:
        """
        Return self.root_anp[i] ** degree.
        """
        k = self._power_cache[i].get(degree)
        if k is None:
            if i >= 0:
                self._power_cache[i][degree] = self.root_anp[i] ** degree
            elif i == -1:
                self._power_cache[i][degree] = self._inv_sum ** degree
            k = self._power_cache[i][degree]
        return k

    def cyclic_sum(self, monom: Tuple[int, int, int], to_sympy = True) -> Union[sp.Expr, sp.polys.polyclasses.ANP]:
        """
        Return CyclicSum(a**i * b**j * c**k) assuming standardlization a + b + c = 1,
        that is, s(a^i*b^j*c^k) / s(a)^(i+j+k).

        Note: when i == j == k, it will return 3 * p(a^i). Be careful with the coefficient.
        """
        s = 0
        _single_power = self._single_power
        for i in range(3):
            s = _single_power(0, monom[i%3]) * _single_power(1, monom[(i+1)%3]) * _single_power(2, monom[(i+2)%3]) + s
        s *= _single_power(-1, sum(monom))
        if self.K is not None and to_sympy:
            s = self.K.to_sympy(s)
        return s

    def span(self, n: int) -> sp.Matrix:
        monoms = generate_expr(3, n, cyc = False)[1]

        vecs = [None]
        _single_power = self._single_power
        for column in range(1):
            vec = [0] * len(monoms)
            for ind, (i, j, k) in enumerate(monoms):
                vec[ind] = _single_power(0, i) * _single_power(1, j) * _single_power(2, k)
            vecs[column] = vec.copy()
    
        # return sp.Matrix(vecs).T

        vecs_extended = []
        for vec in vecs:
            vecs_extended.extend(_algebraic_extension(vec))            
        M = sp.Matrix(vecs_extended).T

        return _reg_matrix(M)

    def _subs_poly(self, poly: sp.Poly) -> sp.polys.polyclasses.ANP:
        """
        Substitute the root into a polynomial. This returns an ANP object,
        which is different from the general method self.eval(poly). This
        function is only available for RootAlgebraic class.
        """
        s = 0
        for (i,j,k), coeff in poly.terms():
            s += self._single_power(0, i) * self._single_power(1, j) * self._single_power(2, k) * coeff
        return s
        

    def subs(self, expr: Union[sp.Expr, sp.Matrix], to_sympy = True) -> Union[sp.Expr, sp.Matrix]:
        """
        Substitute the root into an expression or a matrix of expressions.
        """
        is_single_expr = not isinstance(expr, sp.MatrixBase)
        if is_single_expr:
            expr = [expr]

        a, b, c = sp.symbols('a b c')
        results = []
        for expr0 in expr:
            try:
                if isinstance(expr0, sp.Poly):
                    s = self._subs_poly(expr0)
                else:
                    frac0, frac1 = sp.fraction(expr0.together())
                    frac0, frac1 = frac0.as_poly(a,b,c), frac1.as_poly(a,b,c)
                    s = self._subs_poly(frac0) / self._subs_poly(frac1)
            except:
                s = expr0.subs({a: self.root[0], b: self.root[1], c: self.root[2]})
            results.append(s)

        if to_sympy:
            results = [self.K.to_sympy(r) for r in results]
        if is_single_expr:
            return results[0]
        if not to_sympy:
            return results # ANP objects cannot be converted to sympy Matrix
        return sp.Matrix(results).reshape(*expr.shape)


class RootRational(RootAlgebraic):
    __slots__ = ('root', '_uv', '_ker', '_sum')

    def __init__(self, root):
        root = tuple(sp.S(r) for r in root)
        if len(root) == 2:
            root = (root[0], root[1], 1)
        self.root = root

        if self.is_centered:
            self._uv = (sp.S(2), sp.S(2))
        elif self.is_corner:
            self._uv = (sp.oo, sp.oo)
        else:
            a, b, c = root
            if c == 0:
                a, b, c = b, c, a
            if c != 1:
                a, b = a / c, b / c
            t = ((a*b-a)*(a-b)-(b*(a-1))**2)
            u = ((b*b-a*a)*(a-b) - (b-a*b)*(1-b*b))/t
            v = ((a*b-a)*(1-b*b) - (b*b-a*a)*(b-b*a))/t
            self._uv = (u, v)

        u, v = self._uv
        self._ker = (u**2 - u*v + v**2 + u + v + 1)
        self._sum = sum(root)

    def __str__(self):
        return super(RootAlgebraic, self).__str__()

    def __repr__(self):
        return super(RootAlgebraic, self).__repr__()

    def uv(self, to_sympy = True):
        return self._uv

    def ker(self):
        return self._ker

    def cyclic_sum(self, monom, to_sympy=True):
        i, j, k = monom
        a, b, c = self.root
        return (a**i * b**j * c**k + a**k * b**i * c**j + a**j * b**k * c**i) / self._sum**(i+j+k)

    def span(self, n: int) -> sp.Matrix:
        monoms = generate_expr(3, n, cyc = False)[1]
        if self.is_centered:
            return sp.ones(len(monoms), 1)

        a, b, c = self.root
        vecs = [None, None, None]
        for column in range(3):
            vec = [0] * len(monoms)
            for ind, (i, j, k) in enumerate(monoms):
                vec[ind] = a**i * b**j * c**k
            vecs[column] = sp.Matrix(vec)
            a, b, c = c, a, b
        return sp.Matrix.hstack(*vecs)

    def subs(self, expr: Union[sp.Expr, sp.Matrix], to_sympy = True) -> Union[sp.Expr, sp.Matrix]:
        """
        Substitute the root into an expression or a matrix of expressions.
        """
        a, b, c = sp.symbols('a b c')
        return expr.subs({a: self.root[0], b: self.root[1], c: self.root[2]})