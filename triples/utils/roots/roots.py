from typing import Optional, Union, List, Tuple

import numpy as np
import sympy as sp
from sympy import Poly, Expr, Symbol, Rational
from sympy.combinatorics import PermutationGroup, CyclicGroup
from sympy.polys.constructor import construct_domain
from sympy.polys.domains import Domain
from sympy.polys.domains.gaussiandomains import GaussianElement
from sympy.polys.matrices.sdm import SDM
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.numberfields.subfield import primitive_element
from sympy.polys.polyclasses import DMP, ANP
from sympy.polys.rootoftools import ComplexRootOf as CRootOf

# from .rationalize import rationalize
from ..monomials import generate_monoms

EXRAW = sp.EXRAW if hasattr(sp, 'EXRAW') else sp.EX

try:
    setattr(CRootOf, 'is_algebraic', True)
except:
    pass


def _reg_matrix(M):
    # normalize so that the largest entry in each column is 1
    reg = np.max(np.abs(M), axis = 0)
    reg = 1 / np.tile(reg, (M.shape[0], 1))
    reg = sp.Matrix(reg)
    M = M.multiply_elementwise(reg)
    return M

def _algebraic_extension(vec: List[ANP], domain: Domain) -> sp.Matrix:
    if len(vec) == 0:
        return sp.Matrix(0, 0, [])

    def default(vec, domain):
        sdm = SDM({i: {0: x} for i, x in enumerate(vec) if x != 0}, (len(vec), 1), domain)
        return sdm

    sdm = None
    if domain.is_QQ or domain.is_ZZ or (not hasattr(domain, 'mod')):
        sdm = default(vec, domain)
    else:
        if hasattr(vec[0], 'rep'):
            rep = lambda z: z.rep
        elif isinstance(vec[0], GaussianElement):
            rep = lambda z: (z.y, z.x)
        else:
            sdm = default(vec, domain)

        if sdm is None:
            mod = domain.mod
            mod = mod.to_list() if hasattr(mod, 'to_list') else mod
            zero = domain.zero

            sdm = {}
            for row, x in enumerate(vec):
                l = len(rep(x))
                for i in range(1, l + 1): # len(x.rep) = l >= i
                    if rep(x)[-i] == zero:
                        continue
                    if not (row in sdm):
                        sdm[row] = {}
                    sdm[row][i-1] = rep(x)[-i]
            sdm = SDM(sdm, (len(vec), len(mod) - 1), sp.QQ)

    return sp.Matrix._fromrep(DomainMatrix.from_rep(sdm))

def _derv(n: int, i: int) -> int:
    """
    Compute n! / (n-i)!.
    """
    if i == 1: return n
    if n < i:
        return 0
    return sp.factorial(n) // sp.factorial(n - i)



class Root():
    """
    A tuple (vector) to represent a point in n-dimensional space with
    specialized operations. It has a domain property to control the
    arithmetic operations, which is similar to sympy's DomainMatrix.
    """
    domain: Domain
    def __init__(self, root: List[Expr], domain: Optional[Domain]=None, rep: Optional[List[ANP]]=None):
        """
        Initialize the root.
        """
        root = tuple(sp.S(r) for r in root)
        self.rep = rep
        self.domain = domain
        if domain is None:
            if all(len(r.free_symbols) == 0 for r in root):
                self.domain, self.rep = construct_domain(root, extension=True)
            else:
                self.domain = EXRAW

        if self.rep is None:
            self.rep = [self.domain.from_sympy(r) for r in root]
        self.root = root

        self._make_single_power_cached_func()

    def __getitem__(self, i):
        """Get the i-th element of the root."""
        return self.root[i]

    def __len__(self):
        """Return the number of variables."""
        return len(self.rep)

    @property
    def nvars(self):
        """Return the number of variables. Identical to __len__()."""
        return len(self.rep)

    @property
    def is_Rational(self):
        """Whether the root is rational."""
        return self.domain.is_QQ or self.domain.is_ZZ

    @property
    def is_algebraic(self):
        """Whether the root is algebraic."""
        return self.is_Rational or self.domain.is_AlgebraicField\
            or self.domain.is_QQ_I or self.domain.is_ZZ_I

    @property
    def is_corner(self):
        """Whether there is at most one nonzero in the root."""
        return sum(_ != 0 for _ in self.root) <= 1

    @property
    def is_border(self):
        """Whether there is a zero in the root."""
        return any(_ == 0 for _ in self.root)

    @property
    def is_symmetric(self):
        """Whether at least two coordinates of the root are equal."""
        return len(set(self.root)) < self.nvars

    @property
    def is_center(self):
        """Whether all coordinates of the root are equal."""
        if len(self.root) == 1: return True
        p = self.root[0]
        return all(p == _ for _ in self.root[1:])

    @property
    def is_centered(self):
        """Whether all coordinates of the root are equal."""
        return self.is_center

    @property
    def is_nontrivial(self):
        return (not self.is_border) and (not self.is_symmetric)

    def __str__(self):
        return str(self.root)

    def __repr__(self):
        return repr(self.root)

    def __hash__(self):
        return hash(self.root)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Root):
            return False
        return self.domain == __value.domain and self.rep == __value.rep

    def evalf(self, *args, **kwargs) -> 'Root':
        """
        Return the numerical evaluation of the root. See also sympy.evalf().

        Examples
        ----------
        >>> from sympy import sqrt, cbrt
        >>> root = Root((3, sqrt(2), 2 + cbrt(2))).evalf(5)
        >>> root
        (3.0000, 1.4142, 3.2599)
        >>> root.domain
        RR
        """
        root = tuple(r.evalf(*args, **kwargs) for r in self.root)
        return Root(root)

    def n(self, *args, **kwargs) -> 'Root':
        """
        Return the numerical evaluation of the root. See also sympy.n().

        Examples
        ----------
        >>> from sympy import sqrt, cbrt
        >>> root = Root((3, sqrt(2), 2 + cbrt(2))).n(5)
        >>> root
        (3.0000, 1.4142, 3.2599)
        >>> root.domain
        RR
        """
        root = tuple(r.n(*args, **kwargs) for r in self.root)
        return Root(root)

    def from_sympy(self, *args, **kwargs) -> ANP:
        """Wrapper for self.domain.from_sympy()."""
        return self.domain.from_sympy(*args, **kwargs)

    def to_sympy(self, *args, **kwargs) -> Expr:
        """Wrapper for self.domain.to_sympy()."""
        return self.domain.to_sympy(*args, **kwargs)

    def set_domain(self, domain: Domain) -> 'Root':
        if domain == self.domain:
            return self
        convert = domain.convert
        rep = [convert(r) for r in self.rep]
        return Root(self.root, domain=domain, rep=rep)

    def to_field(self) -> 'Root':
        """
        Return an equivalent root with domain set to the field.
        
        Examples
        ----------
        >>> from sympy import QQ, ZZ
        >>> root = Root((3, 2, 1), domain=ZZ).to_field()
        >>> root
        (3, 2, 1)
        >>> root.domain
        QQ
        """
        if self.domain.is_Field:
            return self
        domain = self.domain.get_field()
        return self.set_domain(domain)

    def eval(self, poly: Poly) -> Expr:
        """
        Evaluate poly(*root). The function only accepts sympy
        polynomials as input. For more complicated usage,
        please use subs().

        Parameters
        ----------
        poly : sympy.Poly
            The polynomial to be evaluated on.

        Returns
        ----------
        expr : sympy.Expr
            The evaluation of the polynomial.

        Examples
        ----------
        >>> from sympy import Poly
        >>> from sympy.abc import a, b, c
        >>> Root((3, 2, 1)).eval(Poly(a**2 + b**2 + c**2, a, b, c))
        14

        >>> from sympy import sqrt
        >>> Root(((1 + sqrt(5))/5, (1 - sqrt(5))/5)).eval(Poly(a**2 + a*b + 2*b**2, a, b))
        14/25 - 2*sqrt(5)/25
        """
        if not (len(poly.gens) == self.nvars):
            raise ValueError('The number of variables in the polynomial does not match the root.')
        return self.to_sympy(self._subs_poly_rep(poly))

    def subs(self, expr: Union[sp.Basic], symbols: Optional[List[Symbol]] = None) -> Union[sp.Basic]:
        """
        Substitute the root into an expression or a matrix of expressions.
        """
        if symbols is None:
            symbols = expr.free_symbols
        return expr.subs(dict(zip(symbols, self.root)))


    def _single_power_monomial(self, monomial: List[int]) -> ANP:
        """Compute r[0]**monomial[0] * r[1]**monomial[1] * ...
        and return the result as a low-level ANP object."""
        return sp.prod([self._single_power(i, p) for i, p in enumerate(monomial)])

    def _make_single_power_cached_func(self):
        """
        Make a function self._single_power = self.rep[i] ** degree.
        Wrap it with cache if it is algebraic but not rational.
        """
        if (not self.is_Rational) and self.is_algebraic:
            self._single_power_cache = dict((key, {}) for key in range(-1, len(self.root)))
            def _single_power(i: int, degree: int) -> ANP:
                """
                Return self.rep[i] ** degree.
                """
                k = self._single_power_cache[i].get(degree)
                if k is None:
                    if i >= 0:
                        self._single_power_cache[i][degree] = self.rep[i] ** degree
                    elif i == -1:
                        self._single_power_cache[i][degree] = self.rep[0] ** degree
                    k = self._single_power_cache[i][degree]
                return k
            self._single_power = _single_power
        else:
            self._single_power = lambda i, degree: self.rep[i] ** degree

    def as_vec(self, n: int, diff: Optional[Tuple[int, ...]] = None,
        numer: bool = False, **options
    ) -> Union[sp.Matrix, np.ndarray]:
        """
        Evaluate the root at monomials of degree n.

        Parameters
        ----------
        n : int
            The degree of the monomials.
        diff : Optional[Tuple[int, ...]]
            The order of differentiation in each variable if not None.
        numer : bool
            If numer, return a numpy float64 array instead of a sympy Matrix.
        options : dict
            Other options for the function `generate_monoms`.

        Returns
        ----------
        vec : Union[sp.Matrix, np.ndarray]
            The vector of evaluated monomials of degree n.

        Examples
        ----------
        >>> from sympy.abc import a, b, c
        >>> print(Root((a, b, c)).as_vec(2))
        Matrix([[a**2], [a*b], [a*c], [b**2], [b*c], [c**2]])
        >>> print(Root((a, b, c)).as_vec(2, diff=(1, 0, 0)))
        Matrix([[2*a], [b], [c], [0], [0], [0]])

        Numerical evaluation is also supported.

        >>> from sympy import sqrt
        >>> print(Root((3, sqrt(2), 1)).as_vec(2))
        Matrix([[9], [3*sqrt(2)], [3], [2], [sqrt(2)], [1]])
        >>> Root((3, 2, 1)).as_vec(2, numer=True)
        array([9., 6., 3., 4., 2., 1.])

        Other options for `generate_monoms` include hom, cyc, sym, symmetry, etc.

        >>> print(Root((a, b, c)).as_vec(2, hom=False))
        Matrix([[a**2], [a*b], [a*c], [a], [b**2], [b*c], [b], [c**2], [c], [1]])
        """
        monoms = generate_monoms(self.nvars, n, **options)[1]

        vec = [None] * len(monoms)
        _single_power = self._single_power_monomial

        if diff is None:
            for ind, monom in enumerate(monoms):
                vec[ind] = _single_power(monom)
        else:
            zero = self.domain.zero
            for ind, monom in enumerate(monoms):
                if any(order_m < order_diff for order_m, order_diff in zip(monom, diff)):
                    vec[ind] = zero
                else:
                    dervs = [_derv(order_m, order_diff) for order_m, order_diff in zip(monom, diff)]
                    powers = [order_m - order_diff for order_m, order_diff in zip(monom, diff)]
                    vec[ind] = int(sp.prod(dervs)) * _single_power(powers)

        sdm = SDM({i: {0: x} for i, x in enumerate(vec) if x != 0}, (len(vec), 1), self.domain)
        vec = sp.Matrix._fromrep(DomainMatrix.from_rep(sdm))
        if numer:
            vec = np.array(vec).astype(np.float64).flatten()
        return vec

    def span(self, n: int, diff: Optional[Tuple[int, ...]] = None,
             normalize: bool = False, **options) -> sp.Matrix:
        """
        Compute the rational span of the Root.as_vec(n, diff, **options).
        It degenerates to `as_vec` if the root is not algebraic.

        Parameters
        ----------
        n : int
            The degree of the monomials.
        diff : Optional[Tuple[int, ...]]
            The order of differentiation in each variable if not None.
        normalize : bool
            Whether to normalize the span so that the largest entry in each column is 1.
            Valid only if the root is algebraic.
        options : dict
            Other options for the function `generate_monoms`.

        Returns
        ----------
        M : sp.Matrix
            The matrix of the span of the monomials.

        Examples
        ----------
        >>> from sympy import sqrt
        >>> print(Root((3, 1 + sqrt(2), 1)).span(2))
        Matrix([[9, 0], [3, 3], [3, 0], [3, 2], [1, 1], [1, 0]])

        Note the difference between `span` and `as_vec`: `span` converts
        algebraic vectors to a matrix of rational numbers.

        >>> print(Root((3, 1 + sqrt(2), 1)).as_vec(2))
        Matrix([[9], [3 + 3*sqrt(2)], [3], [2*sqrt(2) + 3], [1 + sqrt(2)], [1]])

        When the root is rational or not algebraic, `span` is the same as `as_vec`.

        >>> print(Root((3, 2, 1)).span(2))
        Matrix([[9], [6], [3], [4], [2], [1]])

        Other options for `generate_monoms` include hom, cyc, sym, symmetry, etc.

        >>> print(Root((3, 1 + sqrt(2), 1)).span(2, hom=False))
        Matrix([[9, 0], [3, 3], [3, 0], [3, 0], [3, 2], [1, 1], [1, 1], [1, 0], [1, 0], [1, 0]])
        """
        vec = self.as_vec(n, diff, numer=False, **options)

        if self.is_algebraic and not self.is_Rational:
            vec = vec._rep.rep.to_list_flat()
            M = _algebraic_extension(vec, self.domain)
        else:
            M = vec
        if normalize and self.is_algebraic:
            M = _reg_matrix(M)
        return M

    def _subs_poly_rep(self, poly: Poly) -> ANP:
        """
        Substitute the root into a polynomial. This returns an ANP object,
        which is different from the general method self.eval(poly). This
        function is only available for RootAlgebraic class.
        """
        if not (self.domain.is_QQ or self.domain.is_ZZ
                or poly.domain.is_QQ or poly.domain.is_ZZ or self.domain == poly.domain):
            poly = poly.set_domain(poly.domain.unify(self.domain))
            self = self.set_domain(poly.domain)
        s = poly.domain.zero
        for monom, coeff in poly.rep.terms():
            s += self._single_power_monomial(monom) * coeff
        return s

    def cyclic_sum(self, monom: List[int], perm_group: Optional[PermutationGroup] = None,
            standardize: bool = False, to_sympy: bool = True) -> Expr:
        """
        
        """
        nvars = len(self.root)
        if len(monom) != nvars:
            raise ValueError('The monomial does not match the number of variables.')
        # if not all(isinstance(_, int) or _.is_Integer for _ in monom):
        #     raise ValueError('The monomial should be a list of integers.')
        if any(_ < 0 for _ in monom):
            self = self.to_field()
        if perm_group is None:
            perm_group = CyclicGroup(nvars)

        s = 0
        if all(_ == monom[0] for _ in monom):
            # special case: all the same
            s = int(perm_group.order()) * self._single_power_monomial(monom)
        else:
            for perm in perm_group.elements:
                s += self._single_power_monomial(perm(monom))

        domain = self.domain
        if standardize:
            s2 = sum(self.rep)**sum(monom)
            if s2 == domain.zero:
                return sp.nan if s == domain.zero else sp.zoo
            if not domain.is_Field:
                domain = self.domain.get_field()
                s, s2 = domain.convert(s), domain.convert(s2)
            s = s / s2
        if to_sympy:
            s = domain.to_sympy(s)
        return s

    def poly(self, x: Optional[Symbol] = None) -> Poly:
        """
        Return a univariate polynomial with given roots.
        """
        raise NotImplementedError


    def uv(self, to_sympy = True):
        """
        Compute the u, v of a ternary root (a, b, c) defined by:

            sab3  = a*b^3 + b*c^3 + c*a^3
            sac3  = a*c^3 + c*b^3 + b*a^3
            sa2b2 = a^2*b^2 + b^2*c^2 + c^2*a^2
            sa2bc = a^2*b*c + b^2*c*a + c^2*a*b
            u     = (sab3 - sa2bc) / (sa2b2 - sa2bc)
            v     = (sac3 - sa2bc) / (sa2b2 - sa2bc)

        It satisfies the equations:

            a^2 - b^2 + u*(a*b - a*c) + v*(b*c - a*b) = 0
            b^2 - c^2 + u*(b*c - b*a) + v*(c*a - b*c) = 0
            c^2 - a^2 + u*(c*a - c*b) + v*(a*b - c*a) = 0

        Moreover, there are the relations:

            (a*b+b*c+c*a)/(a+b+c)^2     = (u + v - 1)/(u^2 - u*v + v^2 + u + v + 1)
            a*b*c/(a+b+c)^3             = (u*v - 1)/(u^2 - u*v + v^2 + u + v + 1)^2
            (a-b)*(b-c)*(c-a)/(a+b+c)^3 = (u - v)*(u^2 - u*v + v^2 - 2*u - 2*v + 4)/(u^2 - u*v + v^2 + u + v + 1)^2

        Returns
        ----------
        u, v : Expr
            The u, v of the ternary root.

        Examples
        ----------
        >>> from sympy import Poly, CRootOf, EX
        >>> from sympy.abc import a, b, c, x
        >>> Root((6, 4, 2)).uv()
        (17/13, 29/13)
        >>> Root((1, 1, 1)).uv()
        (2, 2)
        >>> Root((a, a, 1), domain=EX).uv()
        (1 + 1/a, 1 + 1/a)

        The nontrivial equality case of the Vasile Cirtoaje's inequality
        has (u, v) = (1, 2).

        >>> poly = Poly(49*x**3 - 49*x**2 + 14*x - 1, x)
        >>> root = Root((CRootOf(poly, 2), CRootOf(poly, 1), CRootOf(poly, 0))); root.n(6)
        (0.543134, 0.349292, 0.107574)
        >>> root.uv()
        (1, 2)
        >>> root.eval(Poly((a**2+b**2+c**2)**2 - 3*(a**3*b+b**3*c+c**3*a), a, b, c))
        0
        """
        if len(self) != 3:
            raise NotImplementedError('The method uv() is only available for ternary roots.')
        if self.is_center:
            return (sp.Integer(2), sp.Integer(2))
        if self.is_corner:
            return (sp.zoo, sp.zoo)
        cyclic_sum = self.cyclic_sum
        sab3 = cyclic_sum((1,3,0), to_sympy=False)
        sac3 = cyclic_sum((1,0,3), to_sympy=False)
        sa2b2 = cyclic_sum((2,2,0), to_sympy=False)
        sa2bc = cyclic_sum((2,1,1), to_sympy=False)
        domain = self.domain
        if not domain.is_Field:
            domain = domain.get_field()
            convert = domain.convert
            sab3, sac3, sa2b2, sa2bc = map(convert, (sab3, sac3, sa2b2, sa2bc))
        inv = domain.one / (sa2b2 - sa2bc)
        u = (sab3 - sa2bc) * inv
        v = (sac3 - sa2bc) * inv
        if to_sympy:
            u, v = domain.to_sympy(u), domain.to_sympy(v)
        return u, v

    @classmethod
    def from_uv(cls, u: Expr, v: Expr) -> 'Root':
        """
        Construct a ternary root from u, v. See also the method uv().

        When (u, v) = (-1, -1), it raises a ValueError because
        any ternary root satisfying a+b+c=0 has (u, v) = (-1, -1).

        When (u, v) are real numbers and not (-1, -1), the resulting
        (a, b, c) is unique up to a cyclic permutation and a scaling.
        Assuming a + b + c = 1, then should satisfy the cubic equation:

            x^3 - x^2 + sab*x - abc = 0

        where sab and abc are given by:

            sab = (u + v - 1) / (u^2 - u*v + v^2 + u + v + 1)
            abc = (u*v - 1) / (u^2 - u*v + v^2 + u + v + 1)^2.

        The discriminant of the cubic equation is a square in QQ(u, v)
        and the roots are real. Also, the Galois group of the cubic
        equation is cyclic and there exists a quadratic polynomial
        that permutes the roots when u != v:

            alpha = -(u^2 - u*v + v^2 + u + v + 1) / (u - v)
            beta  = (u^2 - u*v + v^2 + v) / (u - v)
            gamma = (1 - v) / (u - v)
            b     = alpha*a^2 + beta*a + gamma
            c     = alpha*b^2 + beta*b + gamma
            a     = alpha*c^2 + beta*c + gamma


        Parameters
        ----------
        u : Expr
            The u of the ternary root.
        v : Expr
            The v of the ternary root.

        Examples
        ----------
        >>> from sympy import Rational, Poly
        >>> from sympy.abc import a, b, c
        >>> Root.from_uv(Rational(17, 13), Rational(29, 13))
        (1/2, 1/3, 1/6)
        >>> Root.from_uv(1, 2).n(6)
        (0.543134, 0.349292, 0.107574)
        >>> Root.from_uv(3, 5).uv()
        (3, 5)
        >>> Root.from_uv(-2, 5).eval(Poly(a**2 - b**2 - 2*(a*b - a*c) + 5*(b*c - a*b), a, b, c))
        0
        """
        u0, v0 = sp.S(u), sp.S(v)

        is_real = (u0.is_real and v0.is_real) in (sp.true, True)

        domain, (u, v) = construct_domain((u0, v0), extension=True, field=True)
        one, zero = domain.one, domain.zero
        if u == -one and v == -one: # infinity line
            raise ValueError('Argument uv = (-1, -1) is not well-defined.')

        if u == v:
            a0 = one/(v + one)
            b0 = a0
            c0 = (v - one)*a0
            a, b, c = domain.to_sympy(a0), domain.to_sympy(b0), domain.to_sympy(c0)
        else:
            ker = u**2 - u*v + v**2 + u + v + 1
            invker = one / ker
            sab = (u + v - 1) * invker
            abc = (u*v - 1) * invker**2

            poly = sp.Basic.__new__(Poly)
            poly.rep = DMP([one, -one, sab, -abc], domain, 0)
            poly.gens = (sp.Symbol('x'),)
            a, b, c = poly.all_roots(radicals=False) if poly.domain.is_Exact else poly.nroots()

            if not (a in domain):
                u, v = domain.to_sympy(u), domain.to_sympy(v)
                domain = domain.algebraic_field(a) if not domain.is_QQ_I else sp.QQ.algebraic_field(a, sp.I)
                u, v, one = domain.convert(u), domain.convert(v), domain.one
            invuv = one / (u - v)
            alpha = -(u**2 - u*v + v**2 + u + v + one)*invuv
            beta = (u**2 - u*v + v**2 + v)*invuv
            gamma = (one - v)*invuv
            perm = lambda x: alpha*x**2 + beta*x + gamma

            a0 = domain.convert(a)
            b0 = perm(a0)
            c0 = perm(b0)

            wrong_order = False
            if not is_real:
                b1, c1 = domain.to_sympy(b0), domain.to_sympy(c0)
                if abs(b - b1) + abs(c - c1) > abs(b - c1) + abs(c - b1):
                    wrong_order = True
            elif u0 < v0:
                wrong_order = True
            if wrong_order:
                b0, c0 = c0, b0
                a, b, c = c, b, a
                a0, b0, c0 = c0, b0, a0
        return Root((a,b,c), domain=domain, rep=[a0,b0,c0])


    def ker(self, to_sympy = True):
        raise NotImplementedError


# ###################################################################
# #
# #
# #                         Ternary roots
# #
# #
# ###################################################################

# class TernaryMixin():
#     """
#     Abstract mixin class for roots of 3-var cyclic polynomials.
#     """
#     def __init__(self):
#         if self.is_centered:
#             self._uv = (sp.S(2), sp.S(2))
#         elif self.is_corner:
#             self._uv = (sp.oo, sp.oo)
#         else:
#             a, b, c = self.root if not hasattr(self, 'rep') else self.rep
#             one = sp.S(1) if not hasattr(self, 'K') else self.domain.one
#             sa2b2 = a**2*b**2 + b**2*c**2 + c**2*a**2
#             sa2bc = a*b*c * (a + b + c)
#             sab3 = a*b**3 + b*c**3 + c*a**3
#             sa3b = a**3*b + b**3*c + c**3*a
#             inv_sa2b2_sa2bc = one / (sa2b2 - sa2bc)
#             u = (sab3 - sa2bc) * inv_sa2b2_sa2bc
#             v = (sa3b - sa2bc) * inv_sa2b2_sa2bc
#             self._uv = (u, v)

#         u, v = self._uv
#         self._ker = u**2 - u*v + v**2 + u + v + 1


#     def ker(self):
#         """
#         Return u^2 - uv + v^2 + u + v + 1.
#         """
#         ker = self._ker
#         if hasattr(self, 'K') and self.domain is not None:
#             return self.domain.to_sympy(ker)
#         return ker

#     def poly(self, x: Symbol = None) -> Poly:
#         """
#         Return a polynomial whose three roots are (proportional to) a, b, c.
#         """
#         x = x or Symbol('x')
#         poly = x**3 - x**2 + self.cyclic_sum((1,1,0)) * x - self.cyclic_sum((1,1,1)) / 3
#         return poly.as_poly(x)



    
#     def standardize(self, cyc: bool = False, inplace: bool = False) -> 'Root':
#         """
#         Standardize the root by setting a + b + c == 3.

#         Parameters
#         ----------
#         cyc : bool
#             Whether to standardize cyclically. If True, return
#             (a,b,c) such that a=max(a,b,c). If a == max(b,c), then
#             return (a,b,c) such that b=max(b,c).

#         inplace : bool
#             Whether to modify the root inplace. If False, return
#             a new Root object. If True, return self.

#         Returns
#         ----------
#         root : Root
#             The standardized root. If inplace is True, return self
#             after modification.
#         """
#         if self.is_centered:
#             root = (sp.S(1), sp.S(1), sp.S(1))
#         else:
#             s = sum(self.root) / 3
#             root = (self.root[0] / s, self.root[1] / s, self.root[2] / s)
#             if cyc:
#                 if not self.is_symmetric:
#                     m = max(root)
#                     for i in range(3):
#                         if root[i] == m:
#                             root = (root[i], root[(i+1)%3], root[(i+2)%3])
#                             break
#                 else:
#                     m = max(root)
#                     for i in range(3):
#                         if root[i] == m and root[(i+2)%3] < m:
#                             root = (root[i], root[(i+1)%3], root[(i+2)%3])
#                             break

#         if inplace:
#             self.root = root
#             return self
#         return RootTernary(root)


#     def approximate(self, tolerance = 1e-3, approximate_tolerance = 1e-6):
#         """
#         Approximate the root of a ternary equation, especially it is on the border or symmetric axis.
#         Currently only supports constant roots.
#         """
#         def n(a, b, tol = tolerance):
#             # whether a, b are close enough
#             return abs(a - b) < tol
#         a, b, c = self.root
#         s = (a + b + c) / 3
#         r = [a/s, b/s, c/s]
#         if n(r[0], r[1]) and n(r[1], r[2]) and n(r[0], r[2]):
#             return RootRationalTernary((1,1,1))

#         r = [0 if n(x, 0) else x for x in r]
#         if sum(x != 0 for x in r) == 1:
#             # two of them are zero
#             r = [0 if x == 0 else 1 for x in r]
#             return RootRationalTernary(r)

#         perms = [(0,1,2),(1,2,0),(2,0,1)]
#         for perm in perms:
#             if n(r[perm[0]], r[perm[1]]):
#                 r[perm[2]] = r[perm[2]] / r[perm[0]]
#                 r[perm[0]] = r[perm[1]] = 1
#             elif n(r[perm[0]], 0):
#                 r[perm[2]] = r[perm[2]] / r[perm[1]]
#                 r[perm[1]] = 1

#         for i in range(3):
#             v = rationalize(r[i], rounding=.1, reliable=False)
#             if n(v, r[i], approximate_tolerance):
#                 r[i] = v

#         return RootTernary((r[0], r[1], r[2]))


# class RootTernary(Root, TernaryMixin):
#     """
#     Cyclic root of CyclicSum((a^2-b^2+u(ab-ac)+v(bc-ab))^2).
#     Clearly, it should satisfy a^2-b^2+u(ab-ac)+v(bc-ab) = 0 and its permutations.
#     For example, Vasile inequality is the case of (u, v) = (1, 2).

#     When uv = 1, it degenerates to a root on border, (u, 0, 1) and permutations.
#     When uv != 1, the root is computed by:
#     x = ((v - u)(uv + u + v - 2) + u^3 + 1)/(1 - uv)
#     y = ((u - v)(uv + u + v - 2) - v^3 - 1)/(1 - uv)

#     Then b/c is a root of t^3 + xt^2 + yt - 1 = 0.
#     And a/c is determined by a/c = ((b/c)^2 + (b/c)(u - v) - 1)/((b/c)u - v).
#     """