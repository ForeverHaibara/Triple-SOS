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

            u, v = rationalize(u, reliable = True), rationalize(v, reliable = True)

            self.uv_ = (u, v)
        return self.uv_

    def __str__(self):
        return str(self.root)

    def __repr__(self):
        return repr(self.root)

    def __hash__(self):
        return hash(self.root)

    def _permuted_root(self, permute = 0):
        a, b, c = self.root
        if permute == 1:
            a, b, c = b, c, a
        elif permute == 2:
            a, b, c = c, a, b
        return a, b, c

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
        2. Handle cases when u, v are not rational. -> See RootAlgebraic
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


# class RootSymmetricAxis(Root):
#     is_symmetric = True
#     def __new__(cls, r):
#         if r == 1:
#             # construct RootCentered instead
#             # return RootCentered(r) # this will cause infinite recursion
#             return RootCentered()
#         else:
#             # return Root.__new__(cls, r) # Error: object.__new__() takes only one argument
#             return Root.__new__(cls)

#     def __init__(self, r):
#         r = sp.S(r)
#         self.root = (r, 1, 1)
#         self.uv_ = (r + 1, r + 1)
#         self.ker_ = (r + 2)**2

#     def span(self, n, rational = True):
#         monoms = generate_expr(n, cyc = False)[1]
#         vecs = [None, None, None]
#         a, b, c = self.root
#         if isinstance(a, sp.Rational):
#             for column in range(3):
#                 a, b, c = c, a, b
#                 vec = [0] * len(monoms)
#                 for ind, (i, j, k) in enumerate(monoms):
#                     vec[ind] = a**i * b**j * c**k
#                 vecs[column] = sp.Matrix(vec)
#         elif 0:
#             # quadratic root
#             # important: input the minimal polynomial
#             0
#         M = sp.Matrix.hstack(*vecs)
#         return _reg_matrix(M)


# class RootCentered(RootSymmetricAxis):
#     is_centered = True
#     def __new__(cls, r = 1):
#         return Root.__new__(cls)

#     def __init__(self, r = 1):
#         self.root = (1, 1, 1)
#         self.uv_ = (sp.S(2), sp.S(2))
#         self.ker_ = sp.S(16)

#     def as_vec(self, n, cyc = False, permute = 0, numer = True):
#         monoms = generate_expr(n, cyc = cyc)[1]
#         if numer:
#             vec = np.ones(len(monoms), dtype = np.float64)
#         else:
#             vec = sp.ones(len(monoms), 1)
#         return vec

#     def span(self, n, rational = True):
#         monoms = generate_expr(n, cyc = False)[1]
#         return sp.ones(len(monoms), 1)


class RootAlgebraic(Root):
    """
    When we have exact values for u, v, we can construct the root algebraically
    rather than numerically. This is useful when u, v are algebraic numbers but not rational.
    """
    def __new__(cls, u, v, K = None):
        if cls == RootAlgebraic:
            if K is None:
                u, v = sp.S(u), sp.S(v)
                if u == v:
                    if u != sp.oo:
                        return RootRational((u - 1, 1, 1))
                    return RootRational((0, 0, 1))
                if u * v == 1:
                    return RootRational((u, 0, 1))
                x = ((v-u)*(u*v+u+v-2)+u**3+1)/(1-u*v)
                y = ((v-u)*(u*v+u+v-2)-v**3-1)/(1-u*v)
                r = sp.symbols('r')
                poly = (r**3 + x*r**2 + y*r - 1).as_poly(r)
                roots = sp.polys.roots(poly, multiple = True, cubics = False)
                if len(roots) == 3 and all(isinstance(r, sp.Rational) for r in roots):
                    b = roots[0]
                    a = (b**2 + b*(u - v) - 1) / (b*u - v)
                    return RootRational((a, b, sp.S(1)))
            else:
                if u == v:
                    return RootAlgebraicSymmetricAxis(u, v, K = K)
                if abs(u * v - 1) < 1e-12:
                    return RootAlgebraicBorder(u, v, K = K)

        return object.__new__(cls)


    def __init__(self, u, v, K = None):
        self.root = self._compute_root_from_uv(u, v, K)

        if K is not None:
            if not isinstance(u, sp.polys.AlgebraicField):
                u = K.from_sympy(u)
            if not isinstance(v, sp.polys.AlgebraicField):
                v = K.from_sympy(v)
        self.uv_ = (u, v)
        self.K = K


        self.ker_ = (u*u - u*v + v*v + u + v + 1)
        self.cyclic_sum_cache_ = {}

    @classmethod
    def _compute_root_from_uv(cls, u, v, K = None):
        if K is not None:
            if isinstance(u, sp.polys.AlgebraicField):
                u = K.to_sympy(u)
            if isinstance(v, sp.polys.AlgebraicField):
                v = K.to_sympy(v)
            u, v = u.n(20), v.n(20)
    
            if abs(u*v - u - v) < 1e-12:
                # degenerated case where bu - v = 0
                return ((v - 1)**2, v - 1, sp.S(1))
            if abs(u*v - 1) < 1e-12:
                return (u, 0, 1)

        x = ((v-u)*(u*v+u+v-2)+u**3+1)/(1-u*v)
        y = ((v-u)*(u*v+u+v-2)-v**3-1)/(1-u*v)
        r = sp.symbols('r')
        poly = (r**3 + x*r**2 + y*r - 1).as_poly(r)

        roots = sp.polys.nroots(poly)
        b = roots[0]
        a = (b**2 + b*(u - v) - 1) / (b*u - v)
        return (a, b, sp.S(1))

    def uv(self):
        if self.K is None:
            return self.uv_
        return (self.K.to_sympy(self.uv_[0]), self.K.to_sympy(self.uv_[1]))

    def ker(self):
        if self.K is None:
            return self.ker_
        return self.K.to_sympy(self.ker_)

    def poly(self, x=None):
        if x is None:
            x = sp.symbols('x')
        poly = x**3 - x**2 + self.cyclic_sum((1,1,0)) * x - self.cyclic_sum((1,1,1)) / 3
        return poly.as_poly(x, domain = self.K)

    def cyclic_sum(self, monom, to_sympy = True):
        """
        Return CyclicSum(a**i * b**j * c**k) assuming standardlization a + b + c = 1,
        that is, s(a^i*b^j*c^k) / s(a)^(i+j+k).

        Note: when i == j == k, it will return 3 * p(a^i). Be careful with the coefficient.
        """
        # problem:
        # 1. the poly is not integercoefficient, how to use % operator -> solution: construct poly in the field
        # 2. be aware to translate constant to ANP
        to_sympy = False if self.K is None else to_sympy
        f = (lambda x: sp.S(x)) if self.K is None else (lambda x: self.K.from_sympy(sp.S(x)))
        i, j, k = monom
        m = min(monom)
        u, v = self.uv_
        if m >= 1:
            # is multiple of abc
            s = ((u*v - f(1)) / self.ker_**2) ** m * self.cyclic_sum((i-m, j-m, k-m), to_sympy = False)
            return self.K.to_sympy(s) if to_sympy else s

        if k != 0:
            if j == 0:
                i, j, k = k, i, j
            elif i == 0:
                i, j, k = j, k, i
        if i == 0:
            i, j = j, i

        s = self.cyclic_sum_cache_.get((i,j,k), None)
        if s is not None:
            return self.K.to_sympy(s) if to_sympy else s

        m = max(monom)
        if m >= 3:
            # can reduce the degree by poly remainder
            a, b = sp.symbols('a b')
            mod_poly = self.poly(a)
            poly_a = (a**i).as_poly(a, domain = self.K) % mod_poly
            poly_b = (b**j).as_poly(b, domain = self.K) % mod_poly.replace(a, b)
            s = f(0)
            for term1 in poly_a.terms():
                for term2 in poly_b.terms():
                    s += f(term1[1]) * f(term2[1]) * self.cyclic_sum((term1[0][0], term2[0][0], 0), to_sympy = False)
            self.cyclic_sum_cache_[(i,j,k)] = s
            return self.K.to_sympy(s) if to_sympy else s


        if i == 0:
            if j == 0:
                s = f(3)
            elif j == 1:
                s = f(1)
            elif j == 2:
                s = f(1) - f(2) * self.cyclic_sum((1,1,0), to_sympy = False)
        elif i == 1:
            if j == 0:
                s = f(1)
            elif j == 1:
                s = (u + v - f(1)) / self.ker_
            elif j == 2:
                s = ((v - u) * (u*v + u + v - f(2)) + u**3 + f(1)) / self.ker_**2
        elif i == 2:
            if j == 0:
                s = f(1) - f(2) * self.cyclic_sum((1,1,0), to_sympy = False)
            elif j == 1:
                s = ((u - v) * (u*v + u + v - f(2)) + v**3 + f(1)) / self.ker_**2
            elif j == 2:
                s = (u**2 + v**2 - f(2) * (u+v) + f(3)) / self.ker_**2
        self.cyclic_sum_cache_[(i,j,k)] = s
        return self.K.to_sympy(s) if to_sympy else s

    def span(self, n):
        monoms = generate_expr(n, cyc = False)[1]

        vecs = [None, None, None]
        for column in range(3):
            vec = [0] * len(monoms)
            for ind, (i, j, k) in enumerate(monoms):
                vec[ind] = self.cyclic_sum((i + column, j, k), to_sympy = False)
            vecs[column] = vec.copy()
    
        # return sp.Matrix(vecs).T

        vecs_extended = []
        for vec in vecs:
            vecs_extended.extend(_algebraic_extension(vec))            
        M = sp.Matrix(vecs_extended).T

        return _reg_matrix(M)


class RootRational(Root):
    def __new__(cls, root):
        return Root.__new__(cls)
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


class RootAlgebraicBorder(RootAlgebraic):
    """
    When uv = 1, the root is on the border. It can be handled more carefully.
    Because there is much faster algorithm for spanning subspace.

    However, this class is only a special case of RootAlgebraic. If a root on 
    the border is completely rational, please use RootRational instead.
    """
    def __init__(self, u, v, K = None):
        self.root = (u.n(20), sp.S(0), sp.S(1))

        if K is not None:
            if not isinstance(u, sp.polys.AlgebraicField):
                u = K.from_sympy(u)
            if not isinstance(v, sp.polys.AlgebraicField):
                v = K.from_sympy(v)
        self.uv_ = (u, v)
        self.K = K

        self.ker_ = (u*u + v*v + u + v)
        self.cyclic_sum_cache_ = {}

    @property
    def is_border(self):
        return True

    def span(self, n):
        monoms = generate_expr(n, cyc = False)[1]

        a, b, c = self.uv_[0], 0, 1
        vecs = [None, None, None]
        for column in range(3):
            vec = [0] * len(monoms)
            for ind, (i, j, k) in enumerate(monoms):
                vec[ind] = a**i * b**j * c**k
            vecs[column] = vec.copy()
            a, b, c = c, a, b
    
        # return sp.Matrix(vecs).T

        vecs_extended = []
        for vec in vecs:
            vecs_extended.extend(_algebraic_extension(vec))            
        M = sp.Matrix(vecs_extended).T

        return _reg_matrix(M)


class RootAlgebraicSymmetricAxis(RootAlgebraic):
    """
    When u = v, the root is on the symmetric axis. It can be handled more carefully.

    However, this class is only a special case of RootAlgebraic. If a root on
    the symmetric axis is completely rational, please use RootRational instead.
    """
    def __init__(self, u, v, K = None):
        self.root = (u.n(20) - 1, sp.S(1), sp.S(1))

        if K is not None:
            if not isinstance(u, sp.polys.AlgebraicField):
                u = K.from_sympy(u)
            if not isinstance(v, sp.polys.AlgebraicField):
                v = K.from_sympy(v)

        self.uv_ = (u, v)
        self.K = K

        self.ker_ = (u + 1)**2
        self.cyclic_sum_cache_ = {}

    @property
    def is_symmetric(self):
        return True

    def span(self, n):
        monoms = generate_expr(n, cyc = False)[1]

        a, b, c = self.uv_[0] - 1, 1, 1
        vecs = [None, None, None]
        for column in range(3):
            vec = [0] * len(monoms)
            for ind, (i, j, k) in enumerate(monoms):
                vec[ind] = a**i * b**j * c**k
            vecs[column] = vec.copy()
            a, b, c = c, a, b
    
        # return sp.Matrix(vecs).T

        vecs_extended = []
        for vec in vecs:
            vecs_extended.extend(_algebraic_extension(vec))            
        M = sp.Matrix(vecs_extended).T

        return _reg_matrix(M)