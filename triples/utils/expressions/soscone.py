from collections.abc import Iterable
from typing import (
    Tuple, List, Union, Optional, Callable, Iterator,
    Type, Generic, TypeVar, TYPE_CHECKING
)

from sympy import Expr, Add, Mul, Integer, Rational, UnevaluatedExpr, sqrt
from sympy.core.sympify import CantSympify, sympify
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.polyerrors import CoercionFailed

from .cyclic import CyclicExpr
from .exraw import EXRAW


# For type annotations purposes, DO NOT IMPORT Ef, TExpr, Domain from this module.
# XXX: move it elsewhere
# from sympy.polys.domains.domain import Ef, FieldElement # SymPy >= 1.15
from sympy.polys.domains.domain import Domain as _Domain
class FieldElement: ...
Ef = TypeVar('Ef', bound=FieldElement)
class Domain(_Domain, Generic[Ef]): ...

class TExpr(FieldElement, Generic[Ef]):
    """Only for static type checking."""
    def __mul__(self, other: Union[Ef, 'TExpr[Ef]']) -> 'TExpr[Ef]': ...
    def __rmul__(self, other: Union[Ef, 'TExpr[Ef]']) -> 'TExpr[Ef]': ...


class SOSCone(Generic[Ef]):
    algebra: Domain[TExpr[Ef]]
    domain: Domain[Ef]
    dtype: Type['SOSElement[Ef]']
    def __init__(self, algebra: Domain[TExpr[Ef]], domain: Domain[Ef]):
        self.algebra = algebra
        self.domain = domain
        self.dtype = SOSElement
        self.one = SOSElement(self, [(self.domain.one, self.algebra.one)])
        self.zero = SOSElement(self, [])

    def __str__(self) -> str:
        return f"SOSCone(algebra={self.algebra!s}, domain={self.domain!s})"

    def __repr__(self) -> str:
        return f"SOSCone(algebra={self.algebra!r}, domain={self.domain!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SOSCone):
            return self.algebra == other.algebra
        else:
            return NotImplemented

    def __hash__(self) -> int:
        return hash((self.algebra,))

    def of_type(self, x: object) -> bool:
        return isinstance(x, SOSElement) and x.cone == self

    def sign(self, x: Ef) -> int:
        domain = self.domain
        if domain.is_zero(x):
            return 0
        if domain.is_ZZ or domain.is_QQ:
            if domain.is_positive(x):
                return 1
            return -1
        return 1 if domain.to_sympy(x) > 0 else -1

    def sum(self, elements: Iterator['SOSElement[Ef]']) -> 'SOSElement[Ef]':
        e = self.zero
        for e2 in elements:
            e += e2
        return e

    def prod(self, elements: Iterator['SOSElement[Ef]']) -> 'SOSElement[Ef]':
        e = self.one
        for e2 in elements:
            e *= e2
        return e

    def from_list(self, lst: List[Tuple[Ef, TExpr[Ef]]]) -> 'SOSElement[Ef]':
        return SOSElement.new(self, lst)

    def from_ground(self, x: Ef) -> 'SOSElement[Ef]':
        return self.from_list([(x, self.algebra.one)])

    def from_sympy(self, expr: object) -> Optional['SOSElement[Ef]']:
        expr = sympify(expr)
        try:
            return self._rebuild_expr(expr)
        except CoercionFailed:
            pass
        return None

    def _rebuild_expr(self, x) -> 'SOSElement[Ef]':
        dom, alg = self.domain, self.algebra
        if x.is_Add:
            return self.sum([self._rebuild_expr(t) for t in x.args])
        elif x.is_Mul:
            return self.prod([self._rebuild_expr(t) for t in x.args]) 
        elif x.is_Pow:
            if isinstance(x.exp, Rational) and (int(x.exp.numerator) % 2 == 0
                    or int(x.exp.denominator) % 2 == 0):
                sqrt_x = alg.from_sympy(x.base**(x.exp/2))
                return self.from_list([(dom.one, sqrt_x)])
            if isinstance(x.exp, Integer):
                y = self._rebuild_expr(x.base)
                return y**int(x.exp)
        if x.is_constant(simplify=False):
            if x == 0:
                return self.zero
            elif x > 0:
                return self.from_ground(self.domain.convert(x))
            raise CoercionFailed(f"Cannot convert {x!r} to SOSElement.")
        if isinstance(x, CyclicExpr):
            return self._rebuild_expr(x.doit(deep=False))
        elif isinstance(x, UnevaluatedExpr):
            return self._rebuild_expr(x.args[0])
        raise CoercionFailed(f"Cannot convert {x!r} to SOSElement.")

    def unify(self, cone: 'SOSCone') -> 'SOSCone':
        if self == cone:
            return self
        return SOSCone(self.algebra.unify(cone.algebra), self.domain.unify(cone.domain))


class SOSElement(DomainElement, CantSympify, Generic[Ef]):
    cone: SOSCone[Ef]
    _items: List[Tuple[Ef, TExpr[Ef]]]
    def __new__(cls, cone: SOSCone[Ef], items: List[Tuple[object, object]]):
        _items = []
        domain = cone.domain
        algebra = cone.algebra
        for _c, _v in items:
            c, v = domain.convert(_c), algebra.convert(_v) # type: ignore
            if domain.is_zero(c) or algebra.is_zero(v):
                continue
            if cone.sign(c) < 0:
                raise ValueError("Coeffs must be non-negative.")
            _items.append((c, v))
        return cls.new(cone, _items)

    @classmethod
    def new(cls, cone: SOSCone[Ef], items: List[Tuple[Ef, TExpr[Ef]]]) -> 'SOSElement[Ef]':
        obj = super().__new__(cls)
        obj.cone = cone
        obj._items = items
        return obj

    def per(self, items: List[Tuple[Ef, TExpr[Ef]]]) -> 'SOSElement[Ef]':
        return self.new(self.cone, items)

    @property
    def domain(self) -> Domain[Ef]:
        return self.cone.domain

    @property
    def algebra(self) -> Domain[TExpr[Ef]]:
        return self.cone.algebra

    def __str__(self) -> str:
        if self:
            return " + ".join(f"{c!s}*({v!s})**2" for c, v in self)
        return "0"

    def __repr__(self) -> str:
        if self:
            return " + ".join(f"{c!r}*({v!r})**2" for c, v in self)
        return "0"

    def __bool__(self) -> bool:
        return bool(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Tuple[Ef, TExpr[Ef]]]:
        return iter(self._items)

    def coeffs(self) -> List[Ef]:
        return [c for c, _ in self]

    def values(self) -> List[TExpr[Ef]]:
        return [v for _, v in self]

    def items(self) -> List[Tuple[Ef, TExpr[Ef]]]:
        return self._items[:]

    @property
    def is_zero(self) -> bool:
        # return self.algebra.is_zero(self.to_algebra())
        dom, alg = self.domain, self.algebra
        return all(dom.is_zero(c) or alg.is_zero(v) for c, v in self)

    @property
    def zero(self) -> 'SOSElement[Ef]':
        return self.cone.zero

    @property
    def one(self) -> 'SOSElement[Ef]':
        return self.cone.one

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SOSElement):
            return self.to_algebra() == other.to_algebra()
        else:
            return NotImplemented

    def to_algebra(self) -> TExpr[Ef]:
        if len(self) == 0:
            return self.algebra.zero
        return sum([c * v**2 for c, v in self]) # type: ignore

    def as_expr(self) -> Expr:
        dom, alg = self.domain, self.algebra
        return Add(*[dom.to_sympy(c) * alg.to_sympy(v)**2 for c, v in self]) # type: ignore

    def from_ground(self, arg: Ef) -> 'SOSElement[Ef]':
        return self.per([(arg, self.cone.algebra.one)])

    def __add__(self, other: object) -> 'SOSElement[Ef]':
        if isinstance(other, SOSElement):
            if self.cone == other.cone:
                return self._add(other)
            raise ValueError("Cannot add SOSElements of different cones.")

        if self.domain.of_type(other):
            _other = other
        else:
            try:
                _other = self.domain.convert(other) # type: ignore
            except CoercionFailed:
                return NotImplemented
        sgn = self.cone.sign(_other)
        if sgn == 0:
            return self
        elif sgn < 0:
            raise ValueError("Cannot add negative coeff.")
        return self._add_ground(_other)

    def __radd__(self, other: object) -> 'SOSElement[Ef]':
        return self.__add__(other)

    def _add(self, other: 'SOSElement[Ef]') -> 'SOSElement[Ef]':
        return self.per(self._items + other._items)

    def _add_ground(self, other: Ef) -> 'SOSElement[Ef]':
        return self.per(self._items + [(other, self.cone.algebra.one)])

    def __sub__(self, other: object) -> 'SOSElement[Ef]':
        raise ValueError("Cannot subtract SOSElement.")

    def __rsub__(self, other: object) -> 'SOSElement[Ef]':
        raise ValueError("Cannot subtract SOSElement.")

    def __pos__(self) -> 'SOSElement[Ef]':
        return self

    def __neg__(self) -> 'SOSElement[Ef]':
        raise ValueError("Cannot negate SOSElement.")

    def __mul__(self, other: object) -> 'SOSElement[Ef]':
        if isinstance(other, SOSElement):
            if self.cone == other.cone:
                return self._mul(other)
            raise ValueError("Cannot multiply SOSElements of different cones.")
    
        if self.domain.of_type(other):
            _other = other
        else:
            try:
                _other = self.domain.convert(other) # type: ignore
            except CoercionFailed:
                return NotImplemented
        sgn = self.cone.sign(_other)
        if sgn == 0:
            return self.zero
        elif sgn < 0:
            raise ValueError("Cannot multiply SOSElement with negative coeff.")
        return self._mul_ground(_other)

    def __rmul__(self, other: object) -> 'SOSElement[Ef]':
        if isinstance(other, SOSElement):
            if self.cone == other.cone:
                return other._mul(self)
            raise ValueError("Cannot multiply SOSElements of different cones.")
    
        if self.domain.of_type(other):
            _other = other
        else:
            try:
                _other = self.domain.convert(other) # type: ignore
            except CoercionFailed:
                return NotImplemented
        sgn = self.cone.sign(_other)
        if sgn == 0:
            return self.zero
        elif sgn < 0:
            raise ValueError("Cannot multiply SOSElement with negative coeff.")
        return self._mul_ground(_other)

    def _mul(self, other: 'SOSElement[Ef]') -> 'SOSElement[Ef]':
        return self.per([(c * other_c, v * other_v) \
            for c, v in self for other_c, other_v in other.items()])

    def _mul_ground(self, other: Ef) -> 'SOSElement[Ef]':
        return self.per([(c * other, v) for c, v in self])

    def _rmul_ground(self, other: Ef) -> 'SOSElement[Ef]':
        return self.per([(other * c, v) for c, v in self])

    def __truediv__(self, other: object) -> 'SOSElement[Ef]':
        if isinstance(other, SOSElement):
            if self.cone == other.cone:
                return self._truediv(other)
            raise ValueError("Cannot divide SOSElements of different cones.")

        if self.domain.of_type(other):
            _other = other
        else:
            try:
                _other = self.domain.convert(other) # type: ignore
            except CoercionFailed:
                return NotImplemented
        sgn = self.cone.sign(_other)
        if sgn == 0:
            raise ZeroDivisionError("Cannot divide SOSElement with zero coeff.")
        elif sgn < 0:
            raise ValueError("Cannot divide SOSElement with negative coeff.")
        return self._truediv_ground(_other)

    def __rtruediv__(self, other: object) -> 'SOSElement[Ef]':
        if isinstance(other, SOSElement):
            if self.cone == other.cone:
                return other._truediv(self)
            raise ValueError("Cannot divide SOSElements of different cones.")
        if self.domain.of_type(other):
            _other = other
        else:
            try:
                _other = self.domain.convert(other) # type: ignore
            except CoercionFailed:
                return NotImplemented
        sgn = self.cone.sign(_other)
        if sgn == 0:
            raise ZeroDivisionError("Cannot divide SOSElement with zero coeff.")
        elif sgn < 0:
            raise ValueError("Cannot divide SOSElement with negative coeff.")
        return self.inverse()._rmul_ground(_other)

    def _truediv(self, other: 'SOSElement[Ef]') -> 'SOSElement[Ef]':
        return self._mul(other.inverse())

    def _truediv_ground(self, other: Ef) -> 'SOSElement[Ef]':
        return self.per([(c / other, v) for c, v in self])

    def inverse(self) -> 'SOSElement[Ef]':
        s = self.to_algebra()
        if self.algebra.is_zero(s):
            raise ZeroDivisionError("Cannot invert zero SOSElement.")
        return self.per([(c, v / s) for c, v in self])

    def __pow__(self, n: int) -> 'SOSElement[Ef]':
        if not isinstance(n, int):
            raise TypeError("Cannot raise SOSElement to non-integer power.")
        # if n < 0:
        #     raise ValueError("Cannot raise SOSElement to negative power.")
        return self._pow_int(n)

    def _pow_int(self, n: int) -> 'SOSElement[Ef]':
        if n == 1:
            return self
        if n < 0:
            return self.inverse()._pow_int(-n)
        if self.is_zero:
            if n == 0:
                raise ValueError("0**0")
            if n > 0:
                return self.zero
            raise ZeroDivisionError("Cannot raise zero SOSElement to negative power.")
        if n == 0:
            return self.one
        m = abs(n)
        s = self.to_algebra()
        sm2 = s**(m//2)
        if m % 2 == 0:
            return self.per([(self.domain.one, sm2)])
        return self.per([(c, v*sm2) for c, v in self])

    def mul_sqr(self, x: object) -> 'SOSElement[Ef]':
        if self.algebra.of_type(x):
            return self._mul_sqr_algebra(x)
        if isinstance(x, SOSElement):
            if self.cone == x.cone:
                return self._mul_sqr_algebra(x.to_algebra())
            else:
                raise ValueError("Cannot multiply SOSElements of different cones.")
        if self.domain.of_type(x):
            _x = x
        else:
            _x = self.domain.convert(x) # type: ignore
        return self._mul_ground(_x**2)

    def _mul_sqr_algebra(self, x: TExpr[Ef]) -> 'SOSElement[Ef]':
        return self.per([(c, v*x) for c, v in self])

    def collect(self) -> 'SOSElement[Ef]':
        dt = {}
        for c, v in self:
            if v in dt:
                dt[v] += c
            else:
                dt[v] = c
        return self.per([(c, v) for v, c in dt.items()])

    def primitive(self) -> 'SOSElement[Ef]':
        terms = []
        for c, v in self:
            c2, v2 = v.primitive() # type: ignore
            c = c * c2**2
            terms.append((c, v2))
        return self.per(terms)

    def applyfunc(self, func: Callable[[TExpr[Ef]], TExpr[Ef]]) -> 'SOSElement[Ef]':
        terms = []
        for c, v in self:
            v = func(v)
            terms.append((c, v))
        return self.per(terms)

    def convert(self, cone: SOSCone) -> 'SOSElement':
        dom0, alg0 = self.domain, self.algebra
        dom, alg = cone.domain, cone.algebra
        return SOSElement(cone,
            [(dom.convert_from(c, dom0), alg.convert_from(v, alg0)) for c, v in self])


EXRAWSOSCone = SOSCone(EXRAW, EXRAW)


class SOSlist(Generic[Ef]):
    """
    A class to represent a (weighted) sum-of-squares expression.

    Attributes
    ----------
    rep: SOSElement
        The internal representation of the SOSlist.
        It uses sympy domains.

    Methods
    ----------
    items() -> List[Tuple[Expr, Expr]]:
        Returns the items of the SOSlist.
        It equals to `sum(c * v**2 for c, v in self.items())
    coeffs() -> List[Expr]:
        Returns the coefficient of each item, i.e., `[c for c, v in self.items()]`
    values() -> List[Expr]:
        Returns the expression part of each item, i.e., `[v for c, v in self.items()]`
    as_expr() -> Expr:
        Convert the SOSlist to a sympy expression instance.

    Examples
    ---------

    ## Basic Tutorials
    
    ### Building SOSlist by `SOSlist.from_sympy`

    `SOSlist.from_sympy` automatically recognizes expressions that are in the form
    of sum-of-squares and convert them to SOSlists. The result can be checked using
    `.items()`.

    >>> from sympy.abc import a, b, c, x, y, z
    >>> sl = SOSlist.from_sympy((2*a**2 - 1)**2/4 + ((1 - 2*a)**2 + 2)/4)
    >>> sl
    1/2*(1)**2 + 1/4*(1 - 2*a)**2 + 1/4*(2*a**2 - 1)**2
    >>> sl.items()
    [(1/2, 1), (1/4, 1 - 2*a), (1/4, 2*a**2 - 1)]
    >>> sl.as_expr()
    (1 - 2*a)**2/4 + (2*a**2 - 1)**2/4 + 1/2

    Expressions only involving add, mul, div of sum-of-squares elements are supported.
    For example, the following codes converts the sum-of-squares proof to the Motzkin polynomial.

    >>> sl = SOSlist.from_sympy(
    ... (x**2*y**2*(x**2+y**2-2*z**2)**2 + z**2*(x**2*(y**2-z**2)**2 + y**2*(x**2-z**2)**2))/(x**2+y**2))
    >>> sl.items() # doctest:+SKIP
    [(1, x**2*z*(y**2 - z**2)/(x**2 + y**2)),
     (1, x*y*z*(x**2 - z**2)/(x**2 + y**2)),
     (1, x**2*y*(x**2 + y**2 - 2*z**2)/(x**2 + y**2)),
     (1, x*y*z*(y**2 - z**2)/(x**2 + y**2)),
     (1, y**2*z*(x**2 - z**2)/(x**2 + y**2)),
     (1, x*y**2*(x**2 + y**2 - 2*z**2)/(x**2 + y**2))]
    >>> sum(c * v**2 for c, v in sl.items()).factor()
    x**4*y**2 + x**2*y**4 - 3*x**2*y**2*z**2 + z**6

    It returns None if the expression is not explicitly in the form of sum-of-squares.

    >>> SOSlist.from_sympy(x**2 - 2*x + 5) is None
    True
    >>> SOSlist.from_sympy((x - 1)**2 + 4)
    4*(1)**2 + 1*(x - 1)**2

    ### Building SOSlist from a list

    An SOSlist can be initialized directly from a list of tuples `(c, v)` to
    represent `sum(c * v**2 for c, v in lst)`.

    >>> SOSlist([(1, a - b), (2, b - c), (3, a - c)])
    1*(a - b)**2 + 2*(b - c)**2 + 3*(a - c)**2
    >>> (SOSlist([(1, 1 - 2*a), (1, 2*a**2 - 1)]) + 2)/4
    1/4*(1 - 2*a)**2 + 1/4*(2*a**2 - 1)**2 + 1/2*(1)**2
    """
    rep: SOSElement[Ef]
    def __new__(cls, arg):
        if isinstance(arg, SOSlist):
            return arg
        elif isinstance(arg, SOSElement):
            return cls.new(arg)
        elif isinstance(arg, Iterable):
            return cls.new(SOSElement(EXRAWSOSCone, list(arg)))
        raise TypeError(f"Cannot convert {arg!r} to SOSlist.")

    @property
    def cone(self) -> SOSCone[Ef]:
        return self.rep.cone

    @property
    def domain(self) -> Domain[Ef]:
        return self.rep.domain

    @property
    def algebra(self) -> Domain[TExpr[Ef]]:
        return self.rep.algebra

    @classmethod
    def new(cls, rep: SOSElement[Ef]) -> 'SOSlist[Ef]':
        obj = super().__new__(cls)
        obj.rep = rep
        return obj

    def __str__(self) -> str:
        return str(self.rep)

    def __repr__(self) -> str:
        return repr(self.rep)

    def _repr_latex_(self) -> str:
        return self.as_expr()._repr_latex_()

    def __bool__(self) -> bool:
        return bool(self.rep)

    def __len__(self) -> int:
        return len(self.rep)

    def __iter__(self):
        return iter(self.rep)

    def coeffs(self) -> List[Expr]:
        dom = self.domain
        return [dom.to_sympy(c) for c in self.rep.coeffs()]

    def values(self) -> List[Expr]:
        alg = self.algebra
        return [alg.to_sympy(v) for v in self.rep.values()]

    def items(self) -> List[Tuple[Expr, Expr]]:
        dom = self.domain
        alg = self.algebra
        return [(dom.to_sympy(c), alg.to_sympy(v)) for c, v in self.rep.items()]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SOSlist):
            return self.rep == other.rep
        return NotImplemented

    @property
    def is_zero(self) -> bool:
        return self.rep.is_zero

    @property
    def zero(self) -> 'SOSlist[Ef]':
        return self.new(self.cone.zero)

    @property
    def one(self) -> 'SOSlist[Ef]':
        return self.new(self.cone.one)

    @property
    def expr(self) -> Expr:
        return self.as_expr()

    def as_expr(self) -> Expr:
        return self.rep.as_expr()

    @classmethod
    def from_sympy(cls, arg) -> Optional['SOSlist']:
        rep = EXRAWSOSCone.from_sympy(arg)
        if isinstance(rep, SOSElement):
            return cls.new(rep)
        return None

    def __add__(self, other: object) -> 'SOSlist[Ef]':
        if isinstance(other, SOSlist):
            return self.new(self.rep + other.rep)
        return self.new(self.rep + other)

    def __sub__(self, other: object) -> 'SOSlist[Ef]':
        raise ValueError("Cannot subtract SOSlist.")

    def __radd__(self, other: object) -> 'SOSlist[Ef]':
        if isinstance(other, SOSlist):
            return self.new(other.rep + self.rep)
        return self.new(other + self.rep)

    def __rsub__(self, other: object) -> 'SOSlist[Ef]':
        raise ValueError("Cannot subtract SOSlist.")

    def __pos__(self) -> 'SOSlist[Ef]':
        return self
    
    def __neg__(self) -> 'SOSlist[Ef]':
        raise ValueError("Cannot negate SOSlist.")

    def __mul__(self, other: object) -> 'SOSlist[Ef]':
        if isinstance(other, SOSlist):
            return self.new(self.rep * other.rep)
        return self.new(self.rep * other)

    def __rmul__(self, other: object) -> 'SOSlist[Ef]':
        if isinstance(other, SOSlist):
            return self.new(other.rep * self.rep)
        return self.new(other * self.rep)

    def __truediv__(self, other: object) -> 'SOSlist[Ef]':
        if isinstance(other, SOSlist):
            return self.new(self.rep / other.rep)
        return self.new(self.rep / other)

    def __rtruediv__(self, other: object) -> 'SOSlist[Ef]':
        if isinstance(other, SOSlist):
            return self.new(other.rep / self.rep)
        return self.new(other / self.rep)

    def __pow__(self, n: int) -> 'SOSlist[Ef]':
        return self.new(self.rep**n)

    def inverse(self) -> 'SOSlist[Ef]':
        return self.new(self.rep.inverse())

    def mul_sqr(self, x: Expr) -> 'SOSlist[Ef]':
        """
        Compute self * expr**2

        Examples
        ---------
        >>> from sympy.abc import a, b
        >>> SOSlist([(1, a), (2, b)]).mul_sqr(b + 1).items()
        [(1, a*(b + 1)), (2, b*(b + 1))]
        """
        return self.new(self.rep.mul_sqr(self.algebra.from_sympy(x)))

    def collect(self) -> 'SOSlist[Ef]':
        """
        Collect duplicative terms.

        Examples
        ---------
        >>> from sympy.abc import a, b
        >>> SOSlist([(1, a), (2, b), (1, a + 1), (3, a)]).collect().items()
        [(4, a), (2, b), (1, a + 1)]
        """
        return self.new(self.rep.collect())

    def primitive(self) -> 'SOSlist[Ef]':
        """
        Extract the constant of each term.

        Examples
        ---------
        >>> from sympy.abc import a, b
        >>> SOSlist([(3, (a + b)/2), (2, (4*a**3 - 12*a + 8))]).primitive().items()
        [(3/4, a + b), (32, a**3 - 3*a + 2)]
        """
        return self.new(self.rep.primitive())

    def applyfunc(self, func: Callable[[Expr], Expr]) -> 'SOSlist':
        return self.new(EXRAWSOSCone.from_list([(c, func(v)) for c, v in self.items()])) # type: ignore

    def normalize(self) -> 'SOSlist':
        """
        Make all coefficients unit.

        Examples
        ---------
        >>> from sympy.abc import a, b
        >>> SOSlist([(3, (a + b)/2), (2, (4*a**3 - 12*a + 8))]).normalize().items()
        [(1, sqrt(3)*(a/2 + b/2)), (1, sqrt(2)*(4*a**3 - 12*a + 8))]
        """
        return self.new(EXRAWSOSCone.from_list([(Integer(1), sqrt(c)*v) for c, v in self.items()])) # type: ignore

    def convert(self, cone: SOSCone) -> 'SOSlist':
        """
        Convert the SOSlist to a given cone.

        Examples
        ---------
        >>> from sympy.abc import a, b
        >>> from sympy import ZZ
        >>> l = SOSlist([(1, a), (2, b)]).convert(SOSCone(ZZ[a,b], ZZ))
        >>> l.rep.items() # doctest: +SKIP
        [(1, a), (2, b)]
        >>> ZZ[a, b].of_type(l.rep.items()[0][1])
        True
        """
        return self.new(self.rep.convert(cone))
