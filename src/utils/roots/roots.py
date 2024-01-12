from typing import Optional

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
    __slots__ = ('root', 'uv_', 'ker_', 'cyclic_sum_cache_')

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
        self.uv_ = None
        self.ker_ = None
        self.cyclic_sum_cache_ = {}

    def __getitem__(self, i):
        return self.root[i]

    @classmethod
    def from_uv(cls, uv):
        u, v = uv
        ker = (u*u - u*v + v*v + u + v + 1)
        sab = (u + v - 1) / ker
        abc = (u*v - 1) / ker**2
        x = sp.symbols('x')
        a, b, c = sp.polys.nroots((x**3 - x**2 + sab * x - abc).as_poly(x))
        root = cls((a, b, c))
        u_, v_ = root.uv()
        if abs(u_ - u) + abs(v_ - v) > abs(v_ - u) + abs(u_ - v):
            a, b, c = c, b, a
            root = cls((a, b, c))
        root.uv_ = (sp.S(u), sp.S(v))
        root.ker_ = ker
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
        if self.uv_ is None:
            if self.is_centered:
                self.uv_ = (sp.S(2), sp.S(2))
                return self.uv_
            elif self.is_corner:
                self.uv_ = (sp.oo, sp.oo)
                return self.uv_

            a, b, c = self.root
            if c != 1:
                a, b = a/c, b/c
            t = ((a*b-a)*(a-b)-(b*(a-1))**2)

            # basic quadratic form
            u = ((b*b-a*a)*(a-b) - (b-a*b)*(1-b*b))/t
            v = ((a*b-a)*(1-b*b) - (b*b-a*a)*(b-b*a))/t

            # u, v = rationalize(u, reliable = True), rationalize(v, reliable = True)

            self.uv_ = (u, v)
        return self.uv_

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

    def eval(self, poly, rational = False):
        """
        Evaluate poly(*root).

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
        if self.ker_ is None:
            u, v = self.uv()
            self.ker_ = (u*u - u*v + v*v + u + v + 1)
        return self.ker_

    def cyclic_sum(self, monom):
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

        s = self.cyclic_sum_cache_.get((i,j,k), None)
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
            self.cyclic_sum_cache_[(i,j,k)] = s
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
        self.cyclic_sum_cache_[(i,j,k)] = s
        return s

    def poly(self, x = None):
        """
        Return a polynomial whose three roots are (proportional to) a, b, c.
        """
        if x is None:
            x = sp.symbols('x')
        poly = x**3 - x**2 + self.cyclic_sum((1,1,0)) * x - self.cyclic_sum((1,1,1)) / 3
        return poly.as_poly(x)

    def as_vec(self, n, cyc = False, permute = 0, numer = True):
        """
        Construct the vector of all monomials of degree n. For example, for n = 3,
        return f(a,b,c) = [a^3,a^2*b,a^2*c,a*b^2,a*b*c,a*c^2,b^3,b^2*c,b*c^2,c^3].
        """
        monoms = generate_expr(n, cyc)[1]
        a, b, c = self._permuted_root(permute)

        vec = ([a**i*b**j*c**k for i, j, k in monoms])
        if numer:
            vec = np.array(vec).astype(np.float64)
        else:
            vec = sp.Matrix(vec)
        return vec

    def span(self, n):
        """
        Construct the space spanned by the root and its cyclic permutaton of degree n.
        In general, it is a matrix with 3 columns.
        For example, for n = 3, f(a,b,c) = [a^3,a^2*b,a^2*c,a*b^2,a*b*c,a*c^2,b^3,b^2*c,b*c^2,c^3],
        then sum(f(a,b,c)), sum(a*f(a,b,c)) and sum(a^2*f(a,b,c)) are the three columns.

        TODO:
        1. Prove that the three vectors are linearly independent.
        2. Handle cases when u, v are not rational. -> See RootUV
        """
        monoms = generate_expr(n, cyc = False)[1]
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
    __slots__ = ('root', 'uv_', 'ker_', 'cyclic_sum_cache_', 'K', 'root_anp', 'inv_sum_', 'power_cache_')
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
            self.uv_ = (sp.S(2), sp.S(2))
        elif self.is_corner:
            self.uv_ = (sp.oo, sp.oo)
        else:
            self.uv_ = self._initialize_uv(self.root_anp)
        u, v = self.uv_
        self.ker_ = (u**2 - u*v + v**2 + u + v + 1)

        if self.K is not None:
            self.inv_sum_ = self.K.from_sympy(sp.S(1)) / sum(self.root_anp)
        else:
            self.inv_sum_ = (sp.S(1)) / sum(self.root_anp)

        self.power_cache_ = {0: {}, 1: {}, 2: {}, -1: {}}

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
    
    def uv(self):
        if self.K is None:
            return self.uv_
        return (self.K.to_sympy(self.uv_[0]), self.K.to_sympy(self.uv_[1]))

    def ker(self):
        if self.K is None:
            return self.ker_
        return self.K.to_sympy(self.ker_)

    def __str__(self):
        if self.K is None:
            return str(self.root)
        return str(tuple(r.n(15) if not isinstance(r, sp.Rational) else r for r in self.root))

    def __repr__(self):
        if self.K is None:
            return str(self.root)
        return str(tuple(r.n(15) if not isinstance(r, sp.Rational) else r for r in self.root))

    def single_power_(self, i, degree):
        # return self.root_anp[i] ** degree
        k = self.power_cache_[i].get(degree)
        if k is None:
            if i >= 0:
                self.power_cache_[i][degree] = self.root_anp[i] ** degree
            elif i == -1:
                self.power_cache_[i][degree] = self.inv_sum_ ** degree
            k = self.power_cache_[i][degree]
        return k

    def cyclic_sum(self, monom, to_sympy = True):
        """
        Return CyclicSum(a**i * b**j * c**k) assuming standardlization a + b + c = 1,
        that is, s(a^i*b^j*c^k) / s(a)^(i+j+k).

        Note: when i == j == k, it will return 3 * p(a^i). Be careful with the coefficient.
        """
        s = 0
        single_power_ = self.single_power_
        for i in range(3):
            s = single_power_(0, monom[i%3]) * single_power_(1, monom[(i+1)%3]) * single_power_(2, monom[(i+2)%3]) + s
        s *= single_power_(-1, sum(monom))
        if self.K is not None and to_sympy:
            s = self.K.to_sympy(s)
        return s

    def span(self, n):
        monoms = generate_expr(n, cyc = False)[1]

        vecs = [None]
        single_power_ = self.single_power_
        for column in range(1):
            vec = [0] * len(monoms)
            for ind, (i, j, k) in enumerate(monoms):
                vec[ind] = single_power_(0, i) * single_power_(1, j) * single_power_(2, k)
            vecs[column] = vec.copy()
    
        # return sp.Matrix(vecs).T

        vecs_extended = []
        for vec in vecs:
            vecs_extended.extend(_algebraic_extension(vec))            
        M = sp.Matrix(vecs_extended).T

        return _reg_matrix(M)


class RootRational(RootAlgebraic):
    __slots__ = ('root', 'uv_', 'ker_', 's__')

    def __init__(self, root):
        root = tuple(sp.S(r) for r in root)
        if len(root) == 2:
            root = (root[0], root[1], 1)
        self.root = root

        if self.is_centered:
            self.uv_ = (sp.S(2), sp.S(2))
        elif self.is_corner:
            self.uv_ = (sp.oo, sp.oo)
        else:
            a, b, c = root
            if c != 1:
                a, b = a / c, b / c
            t = ((a*b-a)*(a-b)-(b*(a-1))**2)
            u = ((b*b-a*a)*(a-b) - (b-a*b)*(1-b*b))/t
            v = ((a*b-a)*(1-b*b) - (b*b-a*a)*(b-b*a))/t
            self.uv_ = (u, v)

        u, v = self.uv_
        self.ker_ = (u**2 - u*v + v**2 + u + v + 1)
        self.s__ = sum(root)

    def __str__(self):
        return super(RootAlgebraic, self).__str__()

    def __repr__(self):
        return super(RootAlgebraic, self).__repr__()

    def uv(self):
        return self.uv_

    def ker(self):
        return self.ker_

    def cyclic_sum(self, monom, to_sympy=True):
        i, j, k = monom
        a, b, c = self.root
        return (a**i * b**j * c**k + a**k * b**i * c**j + a**j * b**k * c**i) / self.s__**(i+j+k)

    def span(self, n):
        monoms = generate_expr(n, cyc = False)[1]
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
