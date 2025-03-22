from typing import Optional, Union, List, Tuple

import numpy as np
import sympy as sp
from sympy import Poly, Expr, Symbol, Rational, Integer
from sympy.combinatorics import PermutationGroup, CyclicGroup
from sympy.polys.constructor import construct_domain
from sympy.polys.domains import Domain
from sympy.polys.domains.gaussiandomains import GaussianElement
from sympy.polys.matrices.sdm import SDM
from sympy.polys.matrices.domainmatrix import DomainMatrix
# from sympy.polys.numberfields.subfield import primitive_element
from sympy.polys.polyclasses import DMP, ANP
from sympy.polys.rootoftools import ComplexRootOf as CRootOf

# from .rationalize import rationalize
from ..monomials import generate_monoms

EXRAW = sp.EXRAW if hasattr(sp, 'EXRAW') else sp.EX

try:
    setattr(CRootOf, 'is_algebraic', True)
except:
    pass


def _reg_matrix(M: sp.Matrix) -> sp.Matrix:
    """Normalize so that the largest entry in each column is 1"""
    rep = M._rep.rep.to_sdm()
    domain = rep.domain
    if not (domain.is_QQ or domain.is_ZZ):
        # not expected to reach here given that
        # the input is a matrix of rational numbers
        return M
    if domain.is_ZZ:
        domain = domain.get_field()
        rep = rep.convert_to(domain)
    one, zero = domain.one, domain.zero

    colmax = {}
    for row in rep.values():
        for col, val in row.items():
            if col not in colmax:
                colmax[col] = abs(val)
            elif val != zero:
                colmax[col] = max(colmax[col], abs(val))

    colmax = {col: one / val for col, val in colmax.items()}
    newsdm = {r: {col: val * colmax[col] for col, val in row.items()} for r, row in rep.items()}
    newsdm = SDM(newsdm, rep.shape, domain)
    return sp.Matrix._fromrep(DomainMatrix.from_rep(newsdm))

def _algebraic_extension(vec: List[ANP], domain: Domain) -> sp.Matrix:
    """
    Convert a column vector of algebraic numbers to a matrix of rational numbers.
    """
    if len(vec) == 0:
        return sp.Matrix(0, 0, [])

    def default(vec, domain):
        # Fails to convert to a matrix of rational numbers:
        # return the original vector as a matrix.
        sdm = SDM({i: {0: x} for i, x in enumerate(vec) if x != 0}, (len(vec), 1), domain)
        return sdm

    rep, mod, sdm = None, None, None
    if domain.is_QQ_I or domain.is_ZZ_I:
        mod = domain.mod if hasattr(domain, 'mod') else \
            [domain.dom.one, domain.dom.zero, domain.dom.one] # version compatibility
    elif (not domain.is_QQ) and (not domain.is_ZZ) and hasattr(domain, 'mod'):
        mod = domain.mod

    if hasattr(vec[0], 'rep'):
        rep = lambda z: z.rep
    elif isinstance(vec[0], GaussianElement):
        rep = lambda z: (z.y, z.x)

    if mod is not None and rep is not None:
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
    else:
        sdm = default(vec, domain)

    return sp.Matrix._fromrep(DomainMatrix.from_rep(sdm))

def _derv(n: int, i: int) -> int:
    """Compute n! / (n-i)!."""
    if i == 1: return n
    if n < i:
        return 0
    return sp.factorial(n) // sp.factorial(n - i)

def _root_op(f, g, op, broadcast_f=True, broadcast_g=True, field=False):
    """Perform the operation op between two Root instances f and g."""
    _rootify = lambda x: Root((x,)) if not isinstance(x, Root) else x
    f, g = _rootify(f), _rootify(g)

    domain = f.domain.unify(g.domain)
    if domain == f.domain:
        domain = f.domain
    elif domain == g.domain:
        domain = g.domain
    if field and not domain.is_Field:
        domain = domain.get_field()

    f, g = f.set_domain(domain), g.set_domain(domain)

    if len(f) != len(g):
        if broadcast_f and len(f) == 1:
            f = Root((f.root[0],) * len(g), domain=f.domain, rep=[f.rep[0]] * len(g))
        elif broadcast_g and len(g) == 1:
            g = Root((g.root[0],) * len(f), domain=g.domain, rep=[g.rep[0]] * len(f))
        else:
            raise ValueError('The number of variables in the roots does not match.')

    root = [getattr(a, op)(b) for a, b in zip(f.root, g.root)]
    rep = [getattr(a, op)(b) for a, b in zip(f.rep, g.rep)]
    return Root(root, domain=domain, rep=rep)

class Root():
    """
    A tuple (vector) to represent a point in n-dimensional space with
    specialized operations. It has a domain property to control the
    arithmetic operations, which is similar to sympy's DomainMatrix.

    Examples
    ----------
    >>> from sympy.abc import a, b, c
    >>> from sympy import sqrt, Poly
    >>> root = Root((1 + sqrt(2), 1 - sqrt(2))); root
    (1 + sqrt(2), 1 - sqrt(2))

    Root class supports commonly used arithmetic operations.

    >>> root.eval(Poly(a**2 + b**2, a, b))
    6
    >>> root * 2 + 1
    (2*sqrt(2) + 3, 3 - 2*sqrt(2))
    >>> root.elementary_polynomials()
    [1, 2, -1]
    """
    domain: Domain
    def __init__(self, root: List[Expr], domain: Optional[Domain]=None, rep: Optional[List[ANP]]=None):
        root = tuple(sp.S(r) for r in root)
        self.rep = rep
        self.domain = domain
        if domain is None:
            if all(len(r.free_symbols) == 0 for r in root):
                domain, rep = construct_domain(root, extension=True)
                if domain.is_AlgebraicField:
                    content, mp = domain.ext.minpoly.primitive()
                    if content != 1:
                        # See also https://github.com/sympy/sympy/issues/27798
                        setattr(domain.ext, 'minpoly', mp)
                        domain = domain.__class__(domain.dom, domain.ext)
                        rep = [ANP(r.rep, domain.mod, domain.dom) for r in rep]
                self.domain, self.rep = domain, rep
            else:
                # do not rely on whether the default symbolic domain is EX or EXRAW
                self.domain = EXRAW

        if self.rep is None:
            self.rep = tuple(self.domain.from_sympy(r) for r in root)
        if not isinstance(self.rep, tuple): # make it hashable and immutable
            self.rep = tuple(self.rep)
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
    def is_zero(self):
        """Whether all coordinates of the root are zero."""
        zero = self.domain.zero
        return all(_ == zero for _ in self.rep)

    @property
    def is_corner(self):
        """Whether there is at most one nonzero in the root."""
        zero = self.domain.zero
        return sum(_ != zero for _ in self.rep) <= 1

    @property
    def is_border(self):
        """Whether there is a zero in the root."""
        zero = self.domain.zero
        return any(_ == zero for _ in self.rep)

    @property
    def is_symmetric(self):
        """Whether at least two coordinates of the root are equal."""
        return len(set(self.rep)) < self.nvars

    @property
    def is_center(self):
        """Whether all coordinates of the root are equal."""
        if len(self) == 1: return True
        p = self.rep[0]
        return all(p == _ for _ in self.rep[1:])

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
        return hash((self.domain, self.rep))

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Root):
            return False
        return self.domain == __value.domain and self.rep == __value.rep

    def __add__(self, other: Expr) -> 'Root':
        return _root_op(self, other, '__add__')

    def __radd__(self, other: Expr) -> 'Root':
        return _root_op(other, self, '__add__')

    def __sub__(self, other: Expr) -> 'Root':
        return _root_op(self, other, '__sub__')

    def __rsub__(self, other: Expr) -> 'Root':
        return _root_op(other, self, '__sub__')

    def __neg__(self) -> 'Root':
        return Root((-r for r in self.root), domain=self.domain, rep=[-r for r in self.rep])

    def __mul__(self, other: Expr) -> 'Root':
        return _root_op(self, other, '__mul__')

    def __rmul__(self, other: Expr) -> 'Root':
        return _root_op(other, self, '__mul__')

    def __truediv__(self, other: Expr) -> 'Root':
        return _root_op(self, other, '__truediv__', field=True) # broadcast_f=False)

    def __rtruediv__(self, other: Expr) -> 'Root':
        return _root_op(other, self, '__truediv__', field=True) # broadcast_g=False)

    def __pow__(self, other: Expr) -> 'Root':
        if not isinstance(other, (int, Integer)):
            raise ValueError('The power should be an integer')
        if other < 0:
            self = self.to_field()
        root = [r**other for r in self.root]
        rep = [r**other for r in self.rep]
        return Root(root, domain=self.domain, rep=rep)

    def __abs__(self) -> 'Root':
        return Root(tuple(abs(r) for r in self.root))

    def sum(self, to_sympy: bool = True) -> Expr:
        """
        Evaluate the sum of the root.

        Examples
        ----------
        >>> from sympy import sqrt
        >>> Root((2, 1 + sqrt(2), 1 - 2*sqrt(2))).sum()
        4 - sqrt(2)
        """
        s = sum(self.rep)
        if to_sympy:
            s = self.domain.to_sympy(s)
        return s

    def simplify(self) -> 'Root':
        root = [self.domain.to_sympy(r) for r in self.rep]
        return Root(root, domain=self.domain, rep=self.rep)

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
        if not self.is_Rational:
            _convert = lambda x: domain.convert(self.domain.to_sympy(x))
        else:
            _convert = domain.convert
        rep = [_convert(r) for r in self.rep]
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
        Compute the cyclic sum of the root over a monomial.

        Parameters
        ----------
        monom : List[int]
            The monomial to be summed over.
        perm_group : Optional[PermutationGroup]
            The permutation group to be used. Default is the cyclic group.
        standardize : bool
            Whether to standardize the result by dividing by (sum(root)**sum(monom)).
            This is useful for homogeneous expressions.

        Returns
        ----------
        s : Expr
            The cyclic sum of the root.

        Examples
        ----------
        >>> from sympy.abc import a, b, c
        >>> Root((a, b, c)).cyclic_sum((3, 2, 1))
        a**3*b**2*c + a**2*b*c**3 + a*b**3*c**2
        >>> Root((2, 3, 5)).cyclic_sum((1, 1, 1))
        90

        The permutation group can be specified.

        >>> from sympy.combinatorics import SymmetricGroup
        >>> Root((a, b ,c)).cyclic_sum((3, 2, 1), perm_group=SymmetricGroup(3))
        a**3*b**2*c + a**3*b*c**2 + a**2*b**3*c + a**2*b*c**3 + a*b**3*c**2 + a*b**2*c**3

        >>> Root((2, 3, 5)).cyclic_sum((1, 1, 1), standardize=True)
        9/100
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

    def elementary_polynomials(self, to_sympy: bool = True) -> List[Expr]:
        """
        Compute the 0th~nth elementary polynomials of the root.

        Returns
        ----------
        sigmas : List[Expr]
            The list of elementary polynomials. Starting from σ₀ = 1.

        Examples
        ----------
        >>> from sympy.abc import a, b, c
        >>> from sympy import EX
        >>> Root((a, b, c), domain=EX).elementary_polynomials()
        [1, a + b + c, a*b + a*c + b*c, a*b*c]
        >>> Root((1, 2, 3, 4)).elementary_polynomials()
        [1, 10, 35, 50, 24]
        """
        self = self.to_field()
        def newton_to_elementary(p_list, domain):
            n = len(p_list)
            sigma = [domain.one]  # σ₀ = 1
            for k in range(1, n+1):
                total = domain.zero
                for i in range(1, k+1):
                    sign = domain.one if i % 2 == 1 else -domain.one
                    p_i = p_list[i-1]
                    total += sign * sigma[k - i] * p_i
                sigma_k = total / (domain.one * k)
                sigma.append(sigma_k)
            return sigma
        nvars = len(self)
        domain = self.domain
        newtons = [self.cyclic_sum((i,) + (0,)*(nvars-1), to_sympy=False) for i in range(1, nvars+1)]
        sigmas = newton_to_elementary(newtons, domain)
        if to_sympy:
            sigmas = [domain.to_sympy(_) for _ in sigmas]
        return sigmas

    def poly(self, x: Optional[Symbol] = None) -> Poly:
        """
        Return a monic univariate polynomial with given roots.

        Parameters
        ----------
        x : Optional[Symbol]
            The symbol for the polynomial.

        Returns
        ----------
        >>> from sympy import exp, pi, I, Symbol, Poly, CRootOf
        >>> from sympy.abc import x, y
        >>> Root((1, 2, 3, 4)).poly(y).as_expr().factor()
        (y - 4)*(y - 3)*(y - 2)*(y - 1)
        >>> Root((1, exp(2*pi*I/3), exp(4*pi*I/3))).poly()
        Poly(x**3 - 1, x, domain='QQ<exp(-2*I*pi/3)>')
        
        >>> poly = Poly(49*x**3 - 49*x**2 + 14*x - 1, x)
        >>> root = Root((CRootOf(poly, 2), CRootOf(poly, 1), CRootOf(poly, 0))); root.n(6)
        (0.543134, 0.349292, 0.107574)
        >>> root.poly()
        Poly(x**3 - x**2 + 2/7*x - 1/49, x, domain='QQ<CRootOf(49*x**3 - 49*x**2 + 14*x - 1, 0)>')
        """
        domain = self.to_field().domain
        sigmas = self.elementary_polynomials(to_sympy=False)
        signed_sigmas = [x if i % 2 == 0 else -x for i, x in enumerate(sigmas)]
        if x is None:
            x = sp.Symbol('x')
        poly = Poly.new(DMP(signed_sigmas, domain, 0), x)
        return poly

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
        >>> Root((2, 4, 6)).uv()
        (29/13, 17/13)
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

            poly = Poly.new(DMP([one, -one, sab, -abc], domain, 0), sp.Symbol('x'))
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

            # (a, b, c) and (a, c, b) are different permutations
            # check the order of the roots
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

    def as_trig(self) -> 'Root':
        """
        Try to convert a ternary root to a trigonometric form when its
        u, v are rational numbers. It bases on the Kronecker-Weber theorem
        that every abelian extension of the rationals is contained in a
        cyclotomic field and that roots of equations with abelian galois group
        can be expressed as a rational linear combination of exp(2kπi/n).
        The implementation is experimental and may be very
        slow for high order cyclotomic fields.

        Returns
        ----------
        root : Root
            The trigonometric form of the root.
        
        Examples
        ----------
        >>> Root.from_uv(1, 2)[0]
        CRootOf(49*x**3 - 49*x**2 + 14*x - 1, 2)
        >>> Root.from_uv(1, 2).as_trig()[0]
        -2*cos(3*pi/7)/7 + 2*cos(2*pi/7)/7 + 3/7

        >>> from sympy.abc import x
        >>> from sympy import Poly
        >>> root = Root(Poly(x**3 - 3*x**2 + 1, x).all_roots()); root
        (CRootOf(x**3 - 3*x**2 + 1, 0), CRootOf(x**3 - 3*x**2 + 1, 1), CRootOf(x**3 - 3*x**2 + 1, 2))
        >>> root.uv()
        (2, -1)
        >>> root.as_trig()
        (1 - 2*cos(2*pi/9), 1 - 2*cos(4*pi/9), 2*cos(4*pi/9) + 1 + 2*cos(2*pi/9))
        """
        if len(self) != 3:
            raise NotImplementedError('The method as_trig() is only available for ternary roots.')
        if (self.is_Rational) or (not self.is_algebraic):
            return self

        self = self.to_field()
        domain = self.domain
        u, v = self.uv()
        if not (u.is_Rational and v.is_Rational):
            return self
        u, v = domain.from_sympy(u), domain.from_sympy(v)
        invker = domain.one / (u**2 - u*v + v**2 + u + v + 1)
        sqrtdisc = (u - v)*(u**2 - u*v - 2*u + v**2 - 2*v + 4)*invker**2
        sqrtdisc = domain.to_sympy(sqrtdisc)

        def _get_conductor(sqrtdisc):
            """
            The conductor of a rational cubic polynomial with cyclic galois group
            must take the form of p1*p2*...*pn where pi are distinct integers
            from the set {9} + {p is prime and p = 1 (mod 3)}.
            """
            p, q = sqrtdisc.numerator, sqrtdisc.denominator
            # convert p/q to integer by multiplying u^3
            p = sp.factorint(abs(p))
            for k, v in sp.factorint(abs(q)).items():
                p[k] = (p.get(k, 0) - v) % 3
                if p[k] < 0:
                    p[k] += 3
            if 3 in p:
                p.pop(3)
                p[9] = 1

            conductor = sp.prod([k for k, v in p.items() if k % 3 == 1 or k == 9])
            # print('Sqrt[D] =', sqrtdisc, '-> Conductor =', conductor)
            return conductor

        conductor = _get_conductor(sqrtdisc)
        cyclo = sp.QQ.algebraic_field(sp.cos(2*sp.pi/conductor))
        ext = cyclo.convert(domain.ext)

        rep = [r.rep for r in self.rep]
        rep = [sum(r[-k]*ext**(k-1) for k in range(1, 1+len(r))) for r in rep]

        def _to_trig_seq(n, r, one=1):
            """Convert a polynomial in cos(2π/n) to a linear combination of cos(2kπ/n)."""
            r = r[::-1]
            coeff = [0] * n
            for m in range(len(r)):
                a_m = r[m]
                if a_m == 0:
                    continue
                
                # expand ((z + 1/z)/2)^m
                current_c = one  # C(m,0) = 1
                for k in range(0, m//2 + 1):
                    t = m - 2 * k
                    contribution = a_m * current_c / (2 ** m)
                    # no need to record t < 0
                    coeff[t] += contribution
                    
                    # compute C(m,k+1)
                    current_c = current_c * (m - k) / (k + 1)

            coeff = [coeff[0]] + [2 * k for k in coeff[1:]]
            return coeff
        def _to_trig(n, r, one=1):
            seq = _to_trig_seq(n, r, one)
            return sum([c*sp.cos(2*sp.pi*k/n) for k, c in enumerate(seq)])
        # root = [cyclo.to_sympy(r) for r in rep]
        root = [_to_trig(conductor, r.rep, sp.QQ.one) for r in rep]
        return Root(root, domain=cyclo, rep=rep)

    # def ker(self, to_sympy = True):
    #     """
    #     Return u^2 - uv + v^2 + u + v + 1.
    #     """
    #     raise NotImplementedError
    #     ker = self._ker
    #     if hasattr(self, 'K') and self.domain is not None:
    #         return self.domain.to_sympy(ker)
    #     return ker

    def approximate(self, tolerance = 1e-6):
        """
        Try to convert a numerical root to algebraic by approximating
        the coefficient of its polynomial. It assumes all elements are
        the roots of a rational coefficient polynomial.

        Parameters
        ----------
        tolerance : float
            The tolerance for approximating the coefficients. It is passed
            to the sympy.nsimplify function.

        Returns
        ----------
        root : Root
            The approximated root.

        Examples
        ----------
        >>> root = Root((-0.1773629621, 0.2175678816, 0.9597950805))
        >>> root.approximate()
        (CRootOf(27*x**3 - 27*x**2 + 1, 0), CRootOf(27*x**3 - 27*x**2 + 1, 1), CRootOf(27*x**3 - 27*x**2 + 1, 2))
        >>> Root((2.414213,-0.414213)).approximate(tolerance=1e-2)
        (CRootOf(x**2 - 2*x - 1, 1), CRootOf(x**2 - 2*x - 1, 0))
        >>> Root((.5+1j, .5-1j)).approximate()
        (CRootOf(4*x**2 - 4*x + 5, 1), CRootOf(4*x**2 - 4*x + 5, 0))
        """
        if not (self.domain.is_RR or self.domain.is_CC):
            return self
        poly = self.poly()
        all_coeffs = [sp.nsimplify(_, rational=True, tolerance=tolerance) for _ in poly.all_coeffs()]
        poly = Poly(all_coeffs, poly.gen)
        roots = poly.all_roots(radicals=False)

        # sort original numerical roots
        domain = self.domain
        is_imag = lambda x: domain.imag(x) != 0
        repkeys = [(is_imag(r), r.real, r.imag, i)
                        for i, r in enumerate(self.rep)]
        repkeys = sorted(repkeys)

        sorted_roots = [None] * len(roots)
        for key, root in zip(repkeys, roots):
            sorted_roots[key[-1]] = root
        return Root(sorted_roots)