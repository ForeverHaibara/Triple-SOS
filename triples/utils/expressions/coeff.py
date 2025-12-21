from typing import Union, Dict, List, Tuple, Optional, Callable, Iterator

from sympy import Poly, Expr, Basic, Symbol, Rational, sympify
from sympy import MutableDenseMatrix as Matrix
from sympy.combinatorics import Permutation, PermutationGroup
from sympy.polys.rings import PolyElement
from sympy.polys.domains import Domain
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.ddm import DDM
from sympy.polys.polyclasses import DMP
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities.iterables import iterable

from .exraw import EXRAW
from .cyclic import CyclicSum, CyclicProduct, SymmetricSum, SymmetricProduct
from ..monomials import verify_symmetry

default_prover = lambda x: (x if (x >= 0) else None)
default_prover_implicit = lambda x: x >= 0
identity1 = lambda self, x: x

class PartialOrder:
    """
    Partial order on a domain. Supports a callback function to
    do comparisons between elements.

    >>> from sympy import Symbol, QQ, count_roots
    >>> x = Symbol("x", real=True)

    Define a prover to check whether a univariate polynomial
    is positive over reals. Returns itself if yes. Below we use
    `.as_expr()` to convert an element in `QQ[x]` to sympy `Expr`.

    >>> prover = lambda p: p if count_roots(p.as_expr()) == 0 and p(0) > 0 else None
    >>> po = PartialOrder(QQ[x], prover, wrapper=PartialOrderElement)
    >>> po.convert(x**4 + 1), po.convert(x)
    (PartialOrderElement(x**4 + 1), PartialOrderElement(x))
    >>> po.convert(x**4 + 1) > po.convert(x)
    True

    The above is True because `x**4 + 1 - x > 0` over reals, as asserted by
    the prover. In general, sympy will not check `x**4 + 1 > x` automatically
    if they are raw sympy expressions.

    >>> x**4 + 1 > x
    x**4 + 1 > x
    >>> type(x**4 + 1 > x)
    <class 'sympy.core.relational.StrictGreaterThan'>

    The comparison always checks whether the difference is in the positive
    cone defined by the prover and returns False if not. It is possible
    where both `a <= b` and `b <= a` are False.

    >>> po.convert(x**4) >= po.convert(1)
    False
    >>> po.convert(x**4) <= po.convert(1)
    False
    >>> po.convert(1) <= po.convert(x**4)
    False
    >>> po.convert(1) >= po.convert(x**4)
    False
    """
    domain: Domain
    _prover: Callable[[object], Optional[Expr]]
    _prover_implicit: Callable[[object], Optional[bool]]
    _wrapper: Callable[['PartialOrder', object], object]
    def __init__(self, domain: Domain, prover=None, prover_implicit=None, wrapper=None):
        self.domain = domain
        self._prover = prover if prover is not None else default_prover
        self._prover_implicit = prover_implicit if prover_implicit is not None else (lambda x: self._prover(x) is not None)
        self._wrapper = wrapper if wrapper is not None else identity1

    def prove(self, x) -> Optional[Expr]:
        if isinstance(x, PartialOrderElement):
            x = x.arg
        return self._prover(x)

    def prove_implicit(self, x) -> Optional[bool]:
        if isinstance(x, PartialOrderElement):
            x = x.arg
        return self._prover_implicit(x)

    def wrap(self, x):
        if isinstance(x, PartialOrderElement):
            return x
        return self._wrapper(self, x)

    def convert(self, x) -> object:
        if isinstance(x, PartialOrderElement):
            if self.domain.of_type(x.arg):
                return x
            x = x.arg
        return self.wrap(self.domain.convert(x))

    def to_sympy(self, x) -> Expr:
        if isinstance(x, PartialOrderElement):
            x = x.arg
        return self.domain.to_sympy(x)

    @classmethod
    def from_domain(cls, domain: Domain) -> 'PartialOrder':
        """Create a PartialOrder from a Domain."""
        if domain.is_QQ or domain.is_RR or domain.is_EXRAW or domain.is_RR or domain.is_CC:
            _prover = default_prover
            _prover_implicit = default_prover_implicit
            _wrapper = identity1
        elif domain.is_Algebraic or domain.is_Poly or domain.is_Frac:
            def _algebraic_prover(x):
                z = domain.to_sympy(x)
                return x if z >= 0 else None
            def _algebraic_prover_implicit(x):
                return domain.to_sympy(x) >= 0
            _prover = _algebraic_prover
            _prover_implicit = _algebraic_prover_implicit
            _wrapper = PartialOrderElement

        else:
        # if domain.is_Poly or domain.is_Frac:
        #     self._prover = lambda x: x
        #     self._prover_implicit = lambda x: True
            _prover, _prover_implicit = default_prover, default_prover_implicit
            _wrapper = PartialOrderElement

        return cls(domain, _prover, _prover_implicit, _wrapper)

    def from_rep(self, rep: PolyElement) -> 'Coeff':
        """Create a Coeff instance from rep"""
        return Coeff.new(rep, self)

    def from_dict(self, rep: dict, gens: Tuple[Symbol, ...]) -> 'Coeff':
        """Create a Coeff instance from rep"""
        dt = {k: v if not isinstance(v, PartialOrderElement) else v.arg for k, v in rep.items()}
        ring = self.domain[tuple(gens)].one.ring
        return self.from_rep(ring.from_dict(dt))

    def from_list(self, rep: list, gens: Tuple[Symbol, ...]) -> 'Coeff':
        def _rebuild(rep):
            if len(rep) == 0:
                return rep
            if isinstance(rep[0], list):
                return [_rebuild(r) for r in rep]
            return [v if not isinstance(v, PartialOrderElement) else v.arg for v in rep]
        l = _rebuild(rep)
        ring = self.domain[tuple(gens)].one.ring
        return self.from_rep(ring.from_list(l))

    def from_poly(self, rep: Poly, gens: Optional[Tuple[Symbol, ...]]=None) -> 'Coeff':
        dmp = rep.set_domain(self.domain).rep
        return self.from_dict(dmp.to_dict(), rep.gens if gens is None else gens)

    def as_matrix(self, rep: List[List], shape: Tuple[int, int]) -> Matrix:
        if isinstance(rep, Matrix):
            return rep
        def conv(z):
            if isinstance(z, PartialOrderElement):
                z = z.arg
            return self.domain.convert(z)
        rows = [[conv(v) for v in r] for r in rep]
        ddm = DDM(rows, shape, self.domain)
        return Matrix._fromrep(DomainMatrix.from_rep(ddm))


class Coeff():
    """
    Sparse polynomial representation for INTERNAL USAGE.

    Wrapper of sympy `PolyElement` that supports `__call__` and other methods.
    The outputs from `__call__` might be wrapped to support basic operators with
    other types. The output type is low-level to exploit the fast field arithmetic.

    >>> from sympy import Poly, QQ, RR, Rational, sqrt
    >>> from sympy.abc import a
    >>> K = QQ.algebraic_field(sqrt(7))
    >>> v = K.convert(sqrt(7) + 1)
    >>> v # doctest: +SKIP
    ANP([1, 1], [1, 0, -7], QQ)

    >>> v > 5 # doctest: +SKIP
    Traceback (most recent call last):
    ...
    UnificationFailed: Cannot unify ANP([1, 1], [1, 0, -7], QQ) with 5

    >>> v + a # doctest: +SKIP
    Traceback (most recent call last):
    ...
    TypeError: unsupported operand type(s) for +: 'ANP' and 'Symbol'

    If accessed from a `Coeff` wrapper, then the element will support
    these operators better:

    >>> poly = Poly({(0,): v}, a, domain = K)
    >>> Coeff(poly)((0,)) # doctest: +SKIP
    PartialOrderElement(ANP([1, 1], [1, 0, -7], QQ))

    >>> Coeff(poly)((0,)) > 5
    False

    >>> Coeff(poly)((0,)) + a
    a + 1 + sqrt(7)

    >>> Coeff(poly)((0,))**2 # doctest: +SKIP
    PartialOrderElement(ANP([2, 8], [1, 0, -7], QQ))

    Whether the output element is wrapper by `PartialOrderElement` is
    dependent on the domain. Typically, QQ, RR, EXRAW domains are not
    wrapped as they naturally support these operators.

    >>> poly = (a**2 - 2*a + 5).as_poly(a, domain = QQ)
    >>> type(Coeff(poly)((1,))) # doctest: +SKIP
    flint.types.fmpq.fmpq
    >>> Coeff(poly)((1,)) + a
    a - 2

    ### Casting Rules

    The following table illustrates the type conversion rules
    between various classes. Note that Python tries the left-hand
    operand first, so the behaviour of (xxx) + (coeff output)
    could be undefined or even raise an error.    

    |     Left     |    Right     |    Result    |
    |--------------|--------------|--------------|
    |  python int  | coeff output | coeff output |
    | coeff output |  python int  | coeff output |
    | coeff output | coeff output | coeff output |
    | coeff output |domain element| coeff output |
    |domain element| coeff output |  UNDEFINED   |
    |  sympy expr  | coeff output |  sympy expr  |
    | coeff output |  sympy expr  |  sympy expr  |
    |sympy rational| coeff output |  UNDEFINED   |
    |sympy rational| coeff output |  sympy expr  |

    The following DomainElement + PartialOrderElement could
    raise an error dependent on whether python-flint is installed.

    >>> poly = Poly({(0,): K.convert(sqrt(7) + 1)}, a, domain = K)
    >>> K.domain.one + Coeff(poly)((0,)) # doctest: +SKIP
    PartialOrderElement(ANP([1, 2], [1, 0, -7], QQ))

    The following shows how the order affects the result type:

    >>> poly = Poly({(0,): 7**.5}, a, domain = RR) 
    >>> Coeff(poly)((0,)) + Rational(3, 5) # doctest: +SKIP
    mpf('3.2457513110645908')
    >>> Rational(3, 5) + Coeff(poly)((0,)) # doctest: +SKIP
    3.24575131106459
    """
    rep: PolyElement

    def __init__(self, arg, partial_order=None, field = True, no_ex = True):
        if isinstance(arg, Coeff):
            # make a copy
            self.rep = arg.rep
        elif isinstance(arg, dict):
            if len(arg) == 0:
                gens = (Symbol("a"),)
            else:
                nvars = len(next(iter(arg.keys())))
                gens = tuple([Symbol(chr(97 + i)) for i in range(nvars)])
            rep_dom = EXRAW[gens]
            self.rep = rep_dom.one.ring.from_dict(arg)
        elif isinstance(arg, Poly):
            if field:
                arg = arg.to_field()
            if no_ex and arg.domain.is_EX:
                arg = arg.set_domain(EXRAW)
            rep_dom = arg.domain[arg.gens]
            self.rep = rep_dom.one.ring.from_dict(arg.rep.to_dict())
        elif isinstance(arg, PolyElement):
            self.rep = arg

        self._partial_order = self._default_partial_order(partial_order)

    def _default_partial_order(self, partial_order):
        if partial_order is None:
            partial_order = PartialOrder.from_domain(self.domain)
        return partial_order

    @classmethod
    def new(cls, rep: PolyElement, partial_order=None) -> 'Coeff':
        obj = super().__new__(cls)
        obj.rep = rep
        obj._partial_order = obj._default_partial_order(partial_order)
        return obj

    def from_rep(self, rep: PolyElement) -> 'Coeff':
        return self.new(rep, self._partial_order)

    def from_dict(self, rep: dict, gens: Optional[Tuple[Symbol, ...]] = None) -> 'Coeff':
        return self._partial_order.from_dict(rep, self.gens if gens is None else gens)

    def from_list(self, rep: list, gens: Optional[Tuple[Symbol, ...]] = None) -> 'Coeff':
        return self._partial_order.from_list(rep, self.gens if gens is None else gens)

    def from_poly(self, rep: Poly, gens: Optional[Tuple[Symbol, ...]] = None) -> 'Coeff':
        return self._partial_order.from_poly(rep, gens)

    def as_matrix(self, rep: List[List], shape: Tuple[int, int]) -> Matrix:
        return self._partial_order.as_matrix(rep, shape)

    def prove(self, x) -> Optional[Expr]:
        return self._partial_order.prove(x)

    def prove_implicit(self, x) -> Optional[bool]:
        return self._partial_order.prove_implicit(x)

    def wrap(self, x):
        return self._partial_order.wrap(x)

    def convert(self, x, wrap=True):
        z = self._partial_order.convert(x)
        if wrap:
            z = self.wrap(z)
        return z

    def to_sympy(self, x) -> Expr:
        return self._partial_order.to_sympy(x)

    @property
    def gens(self) -> Tuple[Symbol, ...]:
        return self.rep.ring.symbols

    @property
    def nvars(self) -> int:
        return self.rep.ring.ngens

    @property
    def ring(self):
        return self.rep.ring

    @property
    def domain(self) -> Domain:
        return self.rep.ring.domain

    @property
    def is_rational(self) -> bool:
        return self.domain.is_QQ or self.domain.is_ZZ

    def __iter__(self):
        return self.rep.__iter__()

    def monoms(self) -> List[Tuple[int, ...]]:
        return self.rep.monoms()

    def coeffs(self) -> List[Expr]:
        return self.rep.coeffs()

    def terms(self) -> List[Tuple[Tuple[int, ...], Expr]]:
        return self.rep.terms()

    def keys(self) -> Iterator[Tuple[int, ...]]:
        return self.rep.keys()

    def values(self) -> Iterator[Expr]:
        return self.rep.values()

    def items(self) -> Iterator[Tuple[Tuple[int, ...], Expr]]:
        return self.rep.items()

    def itermonoms(self) -> Iterator[Tuple[int, ...]]:
        return self.rep.itermonoms()

    def itercoeffs(self) -> Iterator[Expr]:
        return self.rep.itercoeffs()

    def iterterms(self) -> Iterator[Tuple[Tuple[int, ...], Expr]]:
        return self.rep.iterterms()

    def listmonoms(self) -> List[Tuple[int, ...]]:
        return self.rep.listmonoms()

    def listcoeffs(self) -> List[Expr]:
        return self.rep.listcoeffs()

    def listterms(self) -> List[Tuple[Tuple[int, ...], Expr]]:
        return self.rep.listterms()

    def __len__(self) -> int:
        return len(self.rep)

    def __hash__(self) -> int:
        return hash(self.rep)

    @property
    def is_zero(self) -> bool:
        return self.rep.is_zero

    @property
    def is_homogeneous(self) -> bool:
        monoms = self.monoms()
        if len(monoms) == 0:
            return True
        d = sum(monoms[0])
        return all(sum(m) == d for m in monoms[1:])

    def total_degree(self) -> int:
        return max((sum(m) for m in self.monoms()), default=0)

    def as_poly(self, *args) -> Poly:
        if len(args) == 0:
            args = self.gens
        elif len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]
        else:
            raise TypeError("Unknown type args")
        dmp = DMP.from_dict(dict(self.rep), len(self.gens)-1, self.domain)
        return Poly.new(dmp, *args)

    def set_domain(self, domain) -> 'Coeff':
        ring = domain[self.gens]
        return self.set_ring(ring)

    def set_ring(self, ring) -> 'Coeff':
        if ring == self.ring:
            return self
        return self.from_rep(self.rep.set_ring(ring))

    def __add__(self, other) -> 'Coeff':
        if isinstance(other, Poly):
            return Coeff(self.as_poly() - other)
        if isinstance(other, Coeff):
            return self.from_rep(self.rep + other.rep)
        return NotImplemented

    def __sub__(self, other) -> 'Coeff':
        if isinstance(other, Poly):
            return Coeff(self.as_poly() - other)
        if isinstance(other, Coeff):
            return self.from_rep(self.rep - other.rep)
        return NotImplemented

    def __radd__(self, other) -> 'Coeff':
        if isinstance(other, Poly):
            return Coeff(other - self.as_poly())
        if isinstance(other, Coeff):
            return self.from_rep(other.rep + self.rep)
        return NotImplemented

    def __rsub__(self, other) -> 'Coeff':
        if isinstance(other, Poly):
            return Coeff(other - self.as_poly())
        if isinstance(other, Coeff):
            return self.from_rep(other.rep - self.rep)
        return NotImplemented

    def __pos__(self) -> 'Coeff':
        return self

    def __neg__(self) -> 'Coeff':
        return self.from_rep(-self.rep)

    def __mul__(self, other) -> 'Coeff':
        if isinstance(other, (int, Rational)):
            return self.from_rep(self.rep * self.domain.convert(other))
        if self.domain.of_type(other):
            return self.from_rep(self.rep * other)
        if isinstance(other, (Poly, Expr, float)):
            return Coeff(self.as_poly() * other)
        if isinstance(other, Coeff):
            return self.from_rep(self.rep * other.rep)
        return NotImplemented

    def __rmul__(self, other) -> 'Coeff':
        return self.__mul__(other)

    def __call__(self, *x) -> Expr:
        """
        Coeff((i,j,k)) -> returns the coefficient of a^i * b^j * c^k.
        """
        if len(x) == 1 and iterable(x[0]):
            # x is ((a,b,c), )
            x = x[0]
        if not isinstance(x, tuple):
            x = tuple(x)
        return self.wrap(self.rep.get(x, self.domain.zero))

    def poly111(self) -> Expr:
        z = self.domain.zero
        for c in self.coeffs():
            z = z + c
        return self.wrap(z)

    def is_cyclic(self, perm_group: Optional[Union[Permutation, List[Permutation], PermutationGroup]] = None) -> bool:
        """
        Check whether the coefficients are cyclic with respect to a permutation group.
        If not specified, it assumes to be the cyclic group.

        Examples
        ---------
        >>> from sympy.abc import a, b, c, d
        >>> coeff = Coeff((a**2*b+b**2*c+c**2*d+d**2*a).as_poly(a, b, c, d))
        >>> coeff.is_cyclic()
        True
        """
        if perm_group is None:
            perm_group = "cyc"
        return verify_symmetry(self.as_poly(), perm_group)

    def is_symmetric(self, perm_group: Optional[Union[Permutation, List[Permutation], PermutationGroup]] = None) -> bool:
        """
        Check whether the coefficients are symmetric with respect to a permutation group.
        If not specified, it assumes to be the symmetric group. When the perm_group
        argument is given, it acts the same as `is_cyclic()`.

        Examples
        ---------
        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> from sympy.abc import a, b, c
        >>> coeff = Coeff((a**2+b**2+c*(a+b)+4*c**2).as_poly(a, b, c))
        >>> coeff.is_symmetric(PermutationGroup(Permutation((1,0,2))))
        True
        >>> coeff.is_symmetric()
        False
        """
        if perm_group is None:
            perm_group = "sym"
        return verify_symmetry(self.as_poly(), perm_group)

    def reflect(self) -> 'Coeff':
        if self.nvars <= 1:
            return self
        return self.reorder([1, 0] + list(range(2, self.nvars)))

    def reorder(self, perm: List[int]) -> 'Coeff':
        order = lambda x: tuple([x[i] for i in perm])
        new_gens = order(self.gens)
        return self.from_dict({order(k): v for k, v in self.items()}, new_gens)

    def clear_zero(self) -> None:
        """
        Clear the coefficients that are zero. In-place. Used internally.
        """
        self.rep = self.ring.from_dict({k: v for k, v in self.items() if v != 0})

    def cancel_abc(self) -> Tuple[Tuple[int, ...], 'Coeff']:
        """
        Assume poly = a^i*b^j*c^k * poly2.
        Return ((i,j,k), Coeff(poly2)).
        """
        if self.is_zero:
            return ((0,) * self.nvars, self)
        monoms = self.monoms()
        if len(monoms) == 0:
            return ((0,) * self.nvars, self)
        d = self.total_degree() + 1
        common = [d] * self.nvars
        for monom in monoms:
            common = [min(i, j) for i, j in zip(common, monom)]
            if all(_ == 0 for _ in common):
                return ((0,) * self.nvars, self)

        common = tuple(common)
        new_coeff = self.from_dict({tuple([i - j for i, j in zip(m, common)]): v for m, v in self.items()})
        return common, new_coeff

    def cancel_k(self) -> Tuple[int, 'Coeff']:
        """
        Assume poly = Sum_{uvw}(x_{uvw} * a^{d*u} * b^{d*v} * c^{d*w}).
        Write poly2 = Sum_{uvw}(x_{uvw} * a^{u} * b^{v} * c^{w}).
        Return (d, Coeff(poly2))
        """
        from math import gcd
        monoms = self.monoms()
        if len(monoms) == 0:
            return (1, self)
        d = 0
        for monom in monoms:
            for u in monom:
                d = gcd(d, u)
                if d == 1:
                    return (1, self)

        d = int(d)
        if d == 0:
            return 0, self

        new_coeff = self.from_dict({tuple([i//d for i in m]): v for m, v in self.items()})
        return d, new_coeff

    def cyclic_sum(self, expr) -> Expr:
        return CyclicSum(expr, self.gens)

    def cyclic_product(self, expr) -> Expr:
        return CyclicProduct(expr, self.gens)

    def symmetric_sum(self, expr) -> Expr:
        return SymmetricSum(expr, self.gens)

    def symmetric_product(self, expr) -> Expr:
        return SymmetricProduct(expr, self.gens)


class PartialOrderElement:
    """
    Element that supports comparison operators with other PartialOrderElement.
    See `Coeff` for details.
    """
    def __init__(self, partial_order, arg):
        self.partial_order = partial_order
        self.arg = arg

    def from_arg(self, new_arg):
        return PartialOrderElement(self.partial_order, new_arg)

    @property
    def domain(self):
        return self.partial_order.domain

    def __str__(self):
        return f"PartialOrderElement({self.arg!s})"

    def __repr__(self):
        return f"PartialOrderElement({self.arg!r})"

    def __add__(self, other):
        if isinstance(other, PartialOrderElement):
            return self.from_arg(self.arg + other.arg)
        elif isinstance(other, Basic):
            return self.as_expr() + other
        try:
            _other = self.domain.convert(other)
        except CoercionFailed:
            return NotImplemented
        return self.from_arg(self.arg + _other)

    def __sub__(self, other):
        if isinstance(other, PartialOrderElement):
            return self.from_arg(self.arg - other.arg)
        elif isinstance(other, Basic):
            return self.as_expr() - other
        try:
            _other = self.domain.convert(other)
        except CoercionFailed:
            return NotImplemented
        return self.from_arg(self.arg - _other)

    def __mul__(self, other):
        if isinstance(other, PartialOrderElement):
            return self.from_arg(self.arg * other.arg)
        elif isinstance(other, Basic):
            return self.as_expr() * other
        elif isinstance(other, Coeff):
            return self.arg * other
        try:
            _other = self.domain.convert(other)
        except CoercionFailed:
            return NotImplemented
        return self.from_arg(self.arg * _other)

    def __truediv__(self, other):
        if isinstance(other, PartialOrderElement):
            return self.from_arg(self.arg / other.arg)
        elif isinstance(other, Basic):
            return self.as_expr() / other
        elif isinstance(other, Coeff):
            return self.arg / other
        try:
            _other = self.domain.convert(other)
        except CoercionFailed:
            return NotImplemented
        return self.from_arg(self.arg / _other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        if isinstance(other, PartialOrderElement):
            return self.from_arg(other.arg - self.arg)
        elif isinstance(other, Basic):
            return other - self.as_expr()
        try:
            _other = self.domain.convert(other)
        except CoercionFailed:
            return NotImplemented
        return self.from_arg(_other - self.arg)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        if isinstance(other, PartialOrderElement):
            return self.from_arg(other.arg / self.arg)
        elif isinstance(other, Basic):
            return other / self.as_expr()
        elif isinstance(other, Coeff):
            return other / self.arg
        try:
            _other = self.domain.convert(other)
        except CoercionFailed:
            return NotImplemented
        return self.from_arg(_other / self.arg)

    def __pow__(self, other):
        if isinstance(other, PartialOrderElement):
            return self.from_arg(self.arg ** other.arg)
        elif isinstance(other, Basic):
            return self.as_expr() ** other
        return self.from_arg(self.arg ** other)

    def __pos__(self):
        return self.from_arg(+self.arg)

    def __neg__(self):
        return self.from_arg(-self.arg)

    def __eq__(self, other):
        if isinstance(other, PartialOrderElement):
            return self.arg == other.arg
        if self.partial_order.domain.of_type(other):
            return self.arg == other
        try:
            v = self.partial_order.domain.convert(other)
            return self.arg == v
        except CoercionFailed:
            pass
        return NotImplemented

    def __bool__(self):
        return self.arg != 0

    def __hash__(self):
        return hash(self.arg)

    def _sympy_(self):
        return self.as_expr()

    def as_expr(self):
        return self.partial_order.to_sympy(self.arg)

    def __le__(self, other):
        return self.partial_order.prove_implicit(other - self)

    def __ge__(self, other):
        return self.partial_order.prove_implicit(self - other)

    def __lt__(self, other):
        return self != other and self.__le__(other)

    def __gt__(self, other):
        return self != other and self.__ge__(other)

    def __abs__(self):
        if (-self) > 0:
            return -self
        return self
