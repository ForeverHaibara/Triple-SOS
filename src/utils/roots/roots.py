import numpy as np
import sympy as sp

from .rationalize import rationalize
from ..basis_generator import generate_expr

class Root():
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

    @property
    def uv(self):
        if self.uv_ is None:
            if self.is_centered:
                self.uv_ = (sp.S(0), sp.S(0))
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

            u, v = rationalize(u, reliable = True), rationalize(v, reliable = True)

            self.uv_ = (u, v)
        return self.uv_

    def __str__(self):
        return str(self.root)

    def __repr__(self):
        return repr(self.root)

    @property
    def ker(self):
        if self.ker_ is None:
            u, v = self.uv
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
        u, v = self.uv
        if m >= 1:
            # is multiple of abc
            return ((u*v - 1) / self.ker**2) ** m * self.cyclic_sum((i-m, j-m, k-m))

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
            poly_b = (b**j).as_poly(b) % mod_poly.xreplace({a:b})
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
                s = (u + v - 1) / self.ker
            elif j == 2:
                s = ((v - u) * (u*v + u + v - 2) + u**3 + 1) / self.ker**2
        elif i == 2:
            if j == 0:
                s = 1 - 2 * self.cyclic_sum((1,1,0))
            elif j == 1:
                s = ((u - v) * (u*v + u + v - 2) + v**3 + 1) / self.ker**2
            elif j == 2:
                s = (u**2 + v**2 - 2*(u+v) + 3) / self.ker**2
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
        a, b, c = self.root
        if permute == 1:
            a, b, c = b, c, a
        elif permute == 2:
            a, b, c = c, a, b

        vec = np.array([a**i*b**j*c**k for i, j, k in monoms])
        if numer:
            vec = vec.astype(np.float64)
        return vec

    def span(self, n):
        """
        Construct the space spanned by the root and its cyclic permutaton of degree n.
        In general, it is a matrix with 3 columns.
        For example, for n = 3, f(a,b,c) = [a^3,a^2*b,a^2*c,a*b^2,a*b*c,a*c^2,b^3,b^2*c,b*c^2,c^3],
        then sum(f(a,b,c)), sum(a*f(a,b,c)) and sum(a^2*f(a,b,c)) are the three columns.

        TODO:
        1. Prove that the three vectors are linearly independent.
        2. Handle cases when u, v are not rational.
        """
        monoms = generate_expr(n, cyc = False)[1]
        if self.is_centered:
            return sp.ones(len(monoms), 1)

        u, v = self.uv
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

        # normalize so that the largest entry in each column is 1
        reg = np.max(np.abs(M), axis = 0)
        reg = 1 / np.tile(reg, (M.shape[0], 1))
        reg = sp.Matrix(reg)
        M = M.multiply_elementwise(reg)
        return M
