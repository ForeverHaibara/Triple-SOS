from typing import List, Tuple, Dict, FrozenSet, Union, Optional, Callable

from sympy import Expr, Add, Mul, Rational, Integer, UnevaluatedExpr, fraction, sympify, latex, sqrt, true
from sympy import Tuple as stuple

from .cyclic import CyclicExpr

def is_true(x) -> bool:
    return x in (true, True)

class SOSlist:
    zero = None
    one  = None

    def __new__(cls, items: List[Tuple[Expr, Expr]]):
        _items = []
        for c, v in items:
            c, v = sympify(c), sympify(v)
            if c == 0 or v == 0:
                continue
            if is_true(c < 0):
                raise ValueError("Coeffs must be non-negative.")
            _items.append((c, v))
        return cls.new(_items)

    @classmethod
    def new(cls, items: List[Tuple[Expr, Expr]]) -> 'SOSlist':
        obj = object.__new__(cls)
        obj._items = items
        return obj

    def coeffs(self) -> List[Expr]:
        """
        Returns the coefficients of the SOSlist.

        Examples
        ---------
        >>> from sympy.abc import a, b
        >>> SOSlist([(1, a), (2, b)]).coeffs()
        [1, 2]
        """
        return [c for c, v in self._items]

    def values(self) -> List[Expr]:
        """
        Returns the values of the SOSlist.

        Examples
        ---------
        >>> from sympy.abc import a, b
        >>> SOSlist([(1, a), (2, b)]).values()
        [a, b]
        """
        return [v for c, v in self._items]

    def items(self):
        """
        Returns the items of the SOSlist.

        Examples
        ---------
        >>> from sympy.abc import a, b
        >>> SOSlist([(1, a), (2, b)]).items()
        [(1, a), (2, b)]
        """
        return self._items[:]

    def copy(self) -> 'SOSlist':
        """
        Returns a copy of the SOSlist.
        """
        return SOSlist.new(self.items())

    def __iter__(self):
        return iter(self.items())

    def __len__(self) -> int:
        """
        Returns the length of the SOSlist.

        Examples
        ---------
        >>> from sympy.abc import a, b
        >>> len(SOSlist([(1, a), (2, b)]))
        2
        """
        return len(self._items)

    @property
    def is_zero(self) -> bool:
        """
        Identify whether the SOSlist is zero by checking len(self) == 0

        Examples
        ---------
        >>> SOSlist([]).is_zero
        True
        >>> from sympy.abc import a, b
        >>> SOSlist([(1, a), (2, b)]).is_zero
        False
        """
        return len(self._items) == 0

    def as_expr(self) -> Expr:
        """
        Returns the sympy expression of the SOSlist.
        This is equivalent to `sum(c * v**2 for c, v in self.items())`

        Examples
        ---------
        >>> from sympy.abc import a, b
        >>> SOSlist([(1, a), (2, b)]).as_expr()
        a**2 + 2*b**2
        """
        return Add(*[c * v**2 for c, v in self._items])

    def __str__(self) -> str:
        return 'SOSlist(' + str(self._items) + ')'

    def __repr__(self) -> str:
        return 'SOSlist(' + repr(self._items) + ')'

    def _repr_latex_(self) -> str:
        s = ', '.join([f"\\left({latex(c)}, {latex(v)}\\right)" for c, v in self._items])
        return f'$\\displaystyle \\mathrm{{SOSlist}}\\left(\\left[{s}\\right]\\right)$'

    def __add__(self, other: 'SOSlist') -> 'SOSlist':
        """
        Add two SOSlist objects. The terms are not collected.

        Examples
        --------
        >>> from sympy.abc import a, b, c
        >>> l = SOSlist([(1, a), (2, b)]) + SOSlist([(3, a), (4, b), (2, c + 1)]); l
        SOSlist([(1, a), (2, b), (3, a), (4, b), (2, c + 1)])

        To collect terms, use the `collect` method.
        >>> l.collect()
        SOSlist([(4, a), (6, b), (2, c + 1)])
        """
        return SOSlist.new(self.items() + other.items())

    def __neg__(self) -> 'SOSlist':
        raise ValueError("Negation of SOSlist is not defined")

    def __sub__(self, other: 'SOSlist') -> 'SOSlist':
        raise ValueError("Subtraction of SOSlist is not defined")

    def __mul__(self, other: Union['SOSlist', Expr]) -> 'SOSlist':
        """
        Multiply an SOSlist with another SOSlist or a constant. The terms are not collected.

        Examples
        ---------
        >>> from sympy.abc import a, b, c, d
        >>> from sympy import Rational
        >>> SOSlist([(1, a), (2, b)]) * Rational(2, 3)
        SOSlist([(2/3, a), (4/3, b)])

        >>> SOSlist([(1, a), (2, b)]) * SOSlist([(3, a), (4, b)])
        SOSlist([(3, a**2), (4, a*b), (6, a*b), (8, b**2)])

        Multiplication with a negative number is not allowed.
        >>> SOSlist([(1, a), (2, b)]) * Rational(-2, 3) # doctest: +SKIP
        Traceback (most recent call last):
        ...
        ValueError: SOSlist only allows multiplication with SOSlist or a nonnegative constant.
        """
        if isinstance(other, SOSlist):
            items = []
            for c1, v1 in self.items():
                for c2, v2 in other.items():
                    items.append((c1 * c2, v1 * v2))
            return SOSlist.new(items)
        else:
            other = sympify(other)
            if other.is_constant(simplify=False) and is_true(other >= 0):
                return SOSlist.new([(other * c, v) for c, v in self.items()])
        raise ValueError("SOSlist only allows multiplication with SOSlist or a nonnegative constant.")

    def __rmul__(self, other: Union['SOSlist', Expr]) -> 'SOSlist':
        if isinstance(other, SOSlist):
            items = []
            for c1, v1 in self.items():
                for c2, v2 in other.items():
                    items.append((c1 * c2, v1 * v2))
            return SOSlist.new(items)
        else:
            other = sympify(other)
            if other.is_constant(simplify=False) and is_true(other >= 0):
                return SOSlist.new([(other * c, v) for c, v in self.items()])
        raise ValueError("SOSlist only allows multiplication with SOSlist or a nonnegative constant.")

    def __truediv__(l, r) -> 'SOSlist':
        if isinstance(r, SOSlist):
            if r.is_zero:
                raise ZeroDivisionError("division by zero")
            mul = l*r
            expr = r.as_expr()
            return SOSlist.new([(c, v/expr) for c, v in mul.items()])
        else:
            r = sympify(r)
            if r == 0:
                raise ValueError("division by zero")
            if r.is_constant(simplify=False) and is_true(r > 0):
                return SOSlist.new([(c / r, v) for c, v in l.items()])
        raise ValueError("SOSlist only allows division by SOSlist or a positive constant.")

    def __rtruediv__(r, l) -> 'SOSlist':
        if not isinstance(l, SOSlist):
            l = sympify(l)
            if l.is_constant(simplify=False) and is_true(l > 0):
                l = SOSlist([(l, Integer(1))])
            else:
                raise ValueError("SOSlist only allows division of an SOSlist or a positive constant.")
        if r.is_zero:
            raise ZeroDivisionError("division by zero")
        mul = l*r
        expr = r.as_expr()
        return SOSlist.new([(c, v/expr) for c, v in mul.items()])

    def __pow__(self, other: Expr) -> 'SOSlist':
        other = sympify(other)
        if isinstance(other, Rational) and (other.numerator % 2 == 0 or other.denominator % 2 == 0):
            sqr = self.as_expr()**(other/2)
            if self.is_zero:
                return self.zero
            return SOSlist.new([(Integer(1), sqr)])
        if (not isinstance(other, Integer)):
            raise ValueError("SOSlist only allows integer or even powers.")
        if self.is_zero:
            return self.zero
        sqr = self.as_expr()**(other//2)
        if other % 2 == 0:
            return SOSlist.new([(Integer(1), sqr)])
        return SOSlist.new([(c, v*sqr) for c, v in self.items()])

    def mul_sqr(self, expr: Expr) -> 'SOSlist':
        """
        Compute self * expr**2

        Examples
        ---------
        >>> from sympy.abc import a, b
        >>> SOSlist([(1, a), (2, b)]).mul_sqr(b + 1)
        SOSlist([(1, a*(b + 1)), (2, b*(b + 1))])
        """
        return SOSlist.new([(c, v*expr) for c, v in self.items()])

    @classmethod
    def sum(cls, lists: List['SOSlist']) -> 'SOSlist':
        items = []
        for lst in lists:
            items.extend(lst.items())
        return SOSlist.new(items)

    @classmethod
    def prod(cls, lists: List['SOSlist']) -> 'SOSlist':
        if len(lists) == 0:
            return cls.one
        item = lists[0]
        for lst in lists[1:]:
            item = item * lst
        return item

    @classmethod
    def from_sympy(cls, expr: Expr) -> Optional['SOSlist']:
        """
        Convert an SOS (sum-of-squares) sympy expression to an SOSlist instance.

        Examples
        ---------
        >>> from sympy.abc import a, b, c, d
        >>> SOSlist.from_sympy(3*a**2 + 4*(a - b)**2 + 5*(a + b - c)**2)
        SOSlist([(3, a), (4, a - b), (5, a + b - c)])

        Summations, multiplications and exponentials are supported.

        >>> SOSlist.from_sympy((a**2+b**2)*(c**2+d**2))
        SOSlist([(1, a*c), (1, a*d), (1, b*c), (1, b*d)])
        >>> SOSlist.from_sympy((a*b + 1)**2 / (a**2 + 2))
        SOSlist([(2, (a*b + 1)/(a**2 + 2)), (1, a*(a*b + 1)/(a**2 + 2))])

        The function only converts expressions that are explicitly in the form of SOS.
        An expression that is implicitly SOS (or not positive definite) cannot be converted.

        >>> SOSlist.from_sympy(a**2 - 2*a*b + b**2 + 5) is None
        True
        >>> SOSlist.from_sympy((a - b)**2 + 5)
        SOSlist([(5, 1), (1, a - b)])
        >>> SOSlist.from_sympy(a**3 + 1) is None
        True
        """
        expr = sympify(expr)
        def _recur_build(x: Expr) -> Optional['SOSlist']:
            if x.is_Add:
                args = []
                for t in x.args:
                    y = _recur_build(t)
                    if y is None:
                        return None
                    args.append(y)
                return cls.sum(args)
            elif x.is_Mul:
                args = []
                common_args = []
                for t in x.args:
                    # if t.is_Pow:
                    #     if isinstance(t.exp, Rational) and (int(t.exp.numerator) % 2 == 0
                    #             or int(t.exp.denominator) % 2 == 0):
                    #         common_args.append(t.base**(t.exp/2))
                    #         continue
                    y = _recur_build(t)
                    if y is None:
                        return None
                    args.append(y)
                return cls.prod(args).mul_sqr(Mul(*common_args))
            elif x.is_Pow:
                if isinstance(x.exp, Rational) and (int(x.exp.numerator) % 2 == 0
                        or int(x.exp.denominator) % 2 == 0):
                    return cls.new([(Integer(1), x.base**(x.exp/2))])
                if isinstance(x.exp, Integer):
                    y = _recur_build(x.base)
                    if y is None:
                        return None
                    return y**x.exp
            if x.is_constant(simplify=False) and is_true(x > 0):
                if x == 0:
                    return cls.zero
                return cls.new([(x, Integer(1))])
            if isinstance(x, CyclicExpr):
                return _recur_build(x.doit(deep=False))
            elif isinstance(x, UnevaluatedExpr):
                return _recur_build(x.args[0])
        return _recur_build(expr)


    def collect(self) -> 'SOSlist':
        """
        Collect duplicative terms.

        Examples
        ---------
        >>> from sympy.abc import a, b
        >>> SOSlist([(1, a), (2, b), (1, a + 1), (3, a)]).collect()
        SOSlist([(4, a), (2, b), (1, a + 1)])
        """
        dt = {}
        for c, v in self.items():
            if v in dt:
                dt[v] += c
            else:
                dt[v] = c
        return SOSlist.new([(c, v) for v, c in dt.items()])

    def primitive(self) -> 'SOSlist':
        """
        Extract the constant of each term.

        Examples
        ---------
        >>> from sympy.abc import a, b
        >>> SOSlist([(3, (a + b)/2), (2, (4*a**3 - 12*a + 8))]).primitive()
        SOSlist([(3/4, a + b), (32, a**3 - 3*a + 2)])
        """
        terms = []
        for c, v in self.items():
            c2, v2 = v.primitive()
            c = c * c2**2
            terms.append((c, v2))
        return SOSlist.new(terms)

    def normalize(self) -> 'SOSlist':
        """
        Make all coefficients unit.

        Examples
        ---------
        >>> from sympy.abc import a, b
        >>> SOSlist([(3, (a + b)/2), (2, (4*a**3 - 12*a + 8))]).normalize()
        SOSlist([(1, sqrt(3)*(a/2 + b/2)), (1, sqrt(2)*(4*a**3 - 12*a + 8))])
        """
        return SOSlist.new([(Integer(1), v*sqrt(c)) for c, v in self.items()])

    def applyfunc(self, func: Callable[[Expr], Expr]) -> 'SOSlist':
        terms = []
        for c, v in self.items():
            v = func(v)
            terms.append((c, v))
        return SOSlist.new(terms)

SOSlist.zero = SOSlist.new([])
SOSlist.one  = SOSlist.new([(Integer(1), Integer(1))])

PSATZ_UNIT = {frozenset(): SOSlist.one}

class Preorder(list):
    def __init__(self, preorder: List[Expr]):
        super().__init__([sympify(_) for _ in preorder])

    def as_expr(self, preorder: Dict[FrozenSet[int], Expr]) -> Expr:
        args = []
        for inds, v in preorder.items():
            args.append(Mul(*[self[i] for i in inds], v.as_expr()))
        return Add(*args)

class Ideal(list):
    def __init__(self, ideal: List[Expr]):
        super().__init__([sympify(_) for _ in ideal])

    def as_expr(self, ideal: Dict[int, Expr]) -> Expr:
        return Add(*[self[i] * v for i, v in ideal.items()])


def _preorder_ideal_add(preorder, ideal, p1, i1, p2, i2):
    """Add (p1 + i1) and (p2 + i2) given preorder and ideal."""
    p3 = p1.copy()
    for m, v in p2.items():
        if m in p3:
            p3[m] = p3[m] + v
        else:
            p3[m] = v
    i3 = i1.copy()
    for m, v in i2.items():
        if m in i3:
            i3[m] = i3[m] + v
        else:
            i3[m] = v
    return p3, i3

def _preorder_ideal_mul(preorder, ideal, p1, i1, p2, i2):
    """Mul (p1 + i1) and (p2 + i2) given preorder and ideal."""
    # p3 = p1 * p2
    p3 = {}
    for m1, v1 in p1.items():
        for m2, v2 in p2.items():
            m3 = m1 ^ m2
            m4 = m1 & m2
            mul = Mul(*[preorder[i] for i in m4])
            v = (v1 * v2).mul_sqr(mul)
            if m3 in p3:
                p3[m3] = p3[m3] + v
            else:
                p3[m3] = v

    p1expr = preorder.as_expr(p1)
    p2expr = preorder.as_expr(p2)
    i2expr = ideal.as_expr(i2)

    # i3 = i1 * (p2 + i2)
    z = p2expr + i2expr
    i3 = {m: v * z for m, v in i1.items()}

    # i3 = i3 + p1 * i2
    for m, v in i2.items():
        if m in i3:
            i3[m] = i3[m] + p1expr * v
        else:
            i3[m] = p1expr * v
    return p3, i3


class PSatz:
    """
    The certificate of F >= 0 given inequality constraints G1,...,Gn >= 0 and equality
    constraints H1,...,Hm == 0 can be established by a variant of positivstellensatz:

        F = (P1 + I1)/(P2 + I2)

    where P1 and P2 are in the preorder generated by G1,...,Gn, while I1 and I2 are in the ideal
    <H1,...,Hm>.

    This class provides methods to convert a sympy expression to a PSatz instance and some basic
    operations.
    """
    preorder: Preorder
    ideal: Ideal
    numer_preorder: Dict[FrozenSet[int], SOSlist]
    numer_ideal: Dict[int, Expr]
    denom_preorder: Dict[FrozenSet[int], SOSlist]
    denom_ideal: Dict[int, Expr]

    zero = None
    one  = None

    def __new__(cls, preorder: Preorder, ideal: Ideal,
        numer_preorder: Optional[Dict[FrozenSet[int], SOSlist]]=None,
        numer_ideal: Optional[Dict[int, Expr]]=None,
        denom_preorder: Optional[Dict[FrozenSet[int], SOSlist]]=None,
        denom_ideal: Optional[Dict[int, Expr]]=None
    ):
        preorder = Preorder(preorder) if not isinstance(preorder, Preorder) else preorder
        ideal = Ideal(ideal) if not isinstance(ideal, Ideal) else ideal

        numer_preorder = numer_preorder or {}
        numer_ideal = numer_ideal or {}

        if denom_preorder is None and denom_ideal is None:
            denom_preorder = PSATZ_UNIT
        denom_preorder = denom_preorder or {}
        denom_ideal = denom_ideal or {}
        return cls.new(preorder, ideal,
            numer_preorder, numer_ideal, denom_preorder, denom_ideal)

    @classmethod
    def new(cls, preorder, ideal, numer_preorder, numer_ideal, denom_preorder, denom_ideal) -> 'PSatz':
        obj = object.__new__(cls)
        obj.preorder = preorder
        obj.ideal = ideal
        obj.numer_preorder = numer_preorder
        obj.numer_ideal = numer_ideal
        obj.denom_preorder = denom_preorder
        obj.denom_ideal = denom_ideal
        return obj

    def per(self, numer_preorder, numer_ideal, denom_preorder, denom_ideal) -> 'PSatz':
        return PSatz(self.preorder, self.ideal,
            numer_preorder, numer_ideal,
            denom_preorder, denom_ideal)

    @property
    def numerator(self) -> Expr:
        return self.preorder.as_expr(self.numer_preorder) + self.ideal.as_expr(self.numer_ideal)

    @property
    def denominator(self) -> Expr:
        return self.preorder.as_expr(self.denom_preorder) + self.ideal.as_expr(self.denom_ideal)

    @property
    def is_zero(self) -> bool:
        return len(self.numer_preorder) == 0 and len(self.numer_ideal) == 0

    @property
    def is_denominator_free(self) -> bool:
        return len(self.denom_ideal) == 0 and len(self.numer_preorder) == 1 and \
            self.numer_preorder.get(frozenset()) == SOSlist.one

    def as_expr(self) -> Expr:
        return self.numerator / self.denominator

    def __str__(self):
        return f"PSatz(preorder={self.preorder}, ideal={self.ideal}, " \
            f"numer_preorder={self.numer_preorder}, numer_ideal={self.numer_ideal}, " \
            f"denom_preorder={self.denom_preorder}, denom_ideal={self.denom_ideal})"

    def __repr__(self):
        return self.__str__()

    def __add__(a, b):
        if not isinstance(b, PSatz):
            raise TypeError(f"expect PSatz, but got {type(b)}")
        if b.preorder != a.preorder:
            raise ValueError("preorder must be the same")
        if b.ideal != a.ideal:
            raise ValueError("ideal must be the same")

        return PSatz.add(a, b)

    def add(a, b):
        preorder, ideal = a.preorder, a.ideal
        # (p1 + i1)/(p2 + i2) + (p3 + i3)/(p4 + i4)
        p1, i1, p2, i2 = a.numer_preorder, a.numer_ideal, a.denom_preorder, a.denom_ideal
        p3, i3, p4, i4 = b.numer_preorder, b.numer_ideal, b.denom_preorder, b.denom_ideal
        if p2 == p4 and i2 == i4:
            p5, i5 = _preorder_ideal_add(preorder, ideal, p1, i1, p3, i3)
            return PSatz.new(preorder, ideal, p5, i5, p2, i2)
        p5, i5 = _preorder_ideal_add(preorder, ideal,
            *_preorder_ideal_mul(preorder, ideal, p1, i1, p4, i4),
            *_preorder_ideal_mul(preorder, ideal, p3, i3, p2, i2)
        )
        p6, i6 = _preorder_ideal_mul(preorder, ideal, p2, i2, p4, i4)
        return PSatz.new(preorder, ideal, p5, i5, p6, i6)

    def __mul__(a, b):
        if not isinstance(b, PSatz):
            raise TypeError(f"expect PSatz, but got {type(b)}")
        if b.preorder != a.preorder:
            raise ValueError("preorder must be the same")
        if b.ideal != a.ideal:
            raise ValueError("ideal must be the same")
        return PSatz.mul(a, b)

    def mul(a, b):
        preorder, ideal = a.preorder, a.ideal
        # (p1 + i1)/(p2 + i2) * (p3 + i3)/(p4 + i4)
        p1, i1, p2, i2 = a.numer_preorder, a.numer_ideal, a.denom_preorder, a.denom_ideal
        p3, i3, p4, i4 = b.numer_preorder, b.numer_ideal, b.denom_preorder, b.denom_ideal
        p5, i5 = _preorder_ideal_mul(preorder, ideal, p1, i1, p3, i3)
        p6, i6 = _preorder_ideal_mul(preorder, ideal, p2, i2, p4, i4)
        return PSatz.new(preorder, ideal, p5, i5, p6, i6)

    def __pow__(self, n):
        if not isinstance(n, (int, Integer)):
            raise TypeError(f"expect int, but got {type(n)}")
        preorder, ideal = self.preorder, self.ideal
        sgn = 1
        if n == 0:
            return PSatz(preorder, ideal, PSATZ_UNIT, {}, {}, {})
        if n < 0:
            sgn = -1
            n = -n

        if n % 2 == 0:
            p1, i1, p2, i2 = {frozenset(): SOSlist([(Integer(1), self.numerator**(n//2))])}, {},\
                {frozenset(): SOSlist([(Integer(1), self.denominator**(n//2))])}, {}
        else:
            p1, i1, p2, i2 = self.numer_preorder, self.numer_ideal, self.denom_preorder, self.denom_ideal
            if n != 1:
                numer_sqr = self.numerator**(n//2)
                numer_sqr2 = numer_sqr**2
                denom_sqr = self.denominator**(n//2)
                denom_sqr2 = denom_sqr**2
                p1, i1, p2, i2 = {m: v.mul_sqr(numer_sqr) for m, v in p1.items()},\
                    {m: v * numer_sqr2 for m, v in i1.items()},\
                    {m: v.mul_sqr(denom_sqr) for m, v in p2.items()},\
                    {m: v * denom_sqr2 for m, v in i2.items()}

        if sgn < 0: # inverse
            p1, p2 = p2, p1
            i1, i2 = i2, i1
        return PSatz.new(preorder, ideal, p1, i1, p2, i2)

    @classmethod
    def sum(cls, ps: List['PSatz']) -> 'PSatz':
        if len(ps) == 0:
            return cls.zero
        p0 = ps[0]
        for p in ps[1:]:
            p0 = p0 + p
        return p0

    @classmethod
    def prod(cls, ps: List['PSatz']) -> 'PSatz':
        if len(ps) == 0:
            return cls.one
        p0 = ps[0]
        for p in ps[1:]:
            p0 = p0 * p
        return p0

    def convert(self, expr: Expr) -> Optional['PSatz']:
        return self.from_sympy(self.preorder, self.ideal, expr)

    @classmethod
    def from_sympy(cls, preorder: Preorder, ideal: Ideal, expr: Expr) -> Optional['PSatz']:
        """
        Convert a sympy expression to a PSatz.

        Parameters
        ----------
        preorder : Preorder or list[Expr]
            The (generators of the) preorder of the PSatz.
        ideal : Ideal or list[Expr]
            The (generators of the) ideal of the PSatz.
        expr : Expr
            The expression to convert.

        Returns
        -------
        Optional[PSatz]
            The converted PSatz.


        Examples
        ---------
        >>> from sympy.abc import a, b, c, x, y
        >>> PSatz.from_sympy([a,b,c], [], a*(b-c)**2 + 2*a*b*c*(a+b-c)**2 + a**2 + 2) # doctest: +NORMALIZE_WHITESPACE
        PSatz(preorder=[a, b, c], ideal=[], numer_preorder={frozenset(): SOSlist([(2, 1), (1, a)]),
            frozenset({0}): SOSlist([(1, b - c)]), frozenset({0, 1, 2}): SOSlist([(2, a + b - c)])},
            numer_ideal={}, denom_preorder={frozenset(): SOSlist([(1, 1)])}, denom_ideal={})

        >>> PSatz.from_sympy([a,b], [x], 2*c*x + b*(x + 1)**2 + x*(2*a + b - 5)) # doctest: +NORMALIZE_WHITESPACE
        PSatz(preorder=[a, b], ideal=[x], numer_preorder={frozenset({1}): SOSlist([(1, x + 1)])},
            numer_ideal={0: 2*a + b + 2*c - 5}, denom_preorder={frozenset(): SOSlist([(1, 1)])}, denom_ideal={})

        >>> PSatz.from_sympy([a], [x], (b**2 - a*x)/(a*(2*a+b)**2 + x + a)) # doctest: +NORMALIZE_WHITESPACE
        PSatz(preorder=[a], ideal=[x], numer_preorder={frozenset(): SOSlist([(1, b)])},
            numer_ideal={0: -a}, denom_preorder={frozenset({0}): SOSlist([(1, 1), (1, 2*a + b)])}, denom_ideal={0: 1})


        The `PSatz.from_sympy` function only identifies expressions that are explicitly in the desired form.

        >>> PSatz.from_sympy([1 - a], [], a**2 - a**3) is None
        True

        >>> PSatz.from_sympy([1 - a], [], a**2*(1 - a)) # doctest: +NORMALIZE_WHITESPACE
        PSatz(preorder=[1 - a], ideal=[], numer_preorder={frozenset({0}): SOSlist([(1, a)])},
            numer_ideal={}, denom_preorder={frozenset(): SOSlist([(1, 1)])}, denom_ideal={})


        To avoid the issue, it is suggested to use `sympy.UnevaluatedExpr` to prevent expressions
        from being expanded.

        >>> PSatz.from_sympy([], [x - 2, y + 2], x + y) is None
        True

        >>> from sympy import UnevaluatedExpr as ue
        >>> ue(x - 2) + ue(y + 2)
        (x - 2) + (y + 2)
        >>> PSatz.from_sympy([], [x - 2, y + 2], ue(x - 2) + ue(y + 2)) # doctest: +NORMALIZE_WHITESPACE
        PSatz(preorder=[], ideal=[x - 2, y + 2], numer_preorder={},
            numer_ideal={0: 1, 1: 1}, denom_preorder={frozenset(): SOSlist([(1, 1)])}, denom_ideal={})
        """
        preorder = Preorder(preorder) if not isinstance(preorder, Preorder) else preorder
        ideal = Ideal(ideal) if not isinstance(ideal, Ideal) else ideal

        mp = {k: i for i, k in enumerate(preorder)}
        mi = {k: i for i, k in enumerate(ideal)}

        f = lambda p, i: PSatz.new(preorder, ideal, p, i, PSATZ_UNIT, {})

        expr = sympify(expr)
        def _is_pure_ideal(x: Expr) -> Optional['PSatz']:
            """Whether an expression lies in the ideal."""
            if x in mi:
                return f({}, {mi[x]: Integer(1)})
            if x.is_Add:
                args = []
                for a in x.args:
                    args.append(_is_pure_ideal(a))
                    if args[-1] is None:
                        return None
                return cls.sum(args)
            elif x.is_Mul or x.is_Pow:
                xargs = x.args if x.is_Mul else (x,)
                for i, a in enumerate(xargs):
                    power = 1
                    if a.is_Pow:
                        a, power = a.base, a.exp
                        if not (isinstance(power, Integer) and power > 0):
                            continue
                    y = _is_pure_ideal(a)
                    if y is not None:
                        other = Mul(*[x.args[j] for j in range(len(x.args)) if j != i])
                        if power > 1:
                            other = other * (a**(power - 1))
                        other = fraction(other)
                        numer = other[0] * other[1]
                        return PSatz.new(preorder, ideal,
                                {}, {m: v*numer for m, v in y.numer_ideal.items()},
                                {m: v.mul_sqr(other[1]) for m, v in y.denom_preorder.items()},
                                y.denom_ideal)
            elif isinstance(x, CyclicExpr):
                return _is_pure_ideal(x.doit(deep=False))
            elif isinstance(x, UnevaluatedExpr):
                return _is_pure_ideal(x.args[0])

            return None

        def _recur_build(x: Expr) -> Optional['PSatz']:
            """Return 1. whether it is psatz 2. whether it lies in the pure ideal"""
            if x.is_Pow:
                if isinstance(x.exp, Integer):
                    if x.exp % 2 == 0:
                        base = SOSlist([(Integer(1), x.base**(abs(x.exp)//2))])
                        if x.exp > 0:
                            return f({frozenset(): base}, {})
                        else:
                            return PSatz.new(preorder, ideal,
                                PSATZ_UNIT, {}, {frozenset(): base}, {})
                    else:
                        y = _recur_build(x.base)
                        if y is None:
                            return None
                        return y**x.exp
            if x in mi:
                return f({}, {mi[x]: Integer(1)})
            if x in mp:
                return f({frozenset({mp[x]}): SOSlist.one}, {})
            if x.is_Add:
                args = []
                for a in x.args:
                    args.append(_recur_build(a))
                    if args[-1] is None:
                        return None
                return cls.sum(args)
            elif x.is_Mul:
                args = []
                for a in x.args:
                    args.append(_recur_build(a))
                    if args[-1] is None:
                        break
                else:
                    return cls.prod(args)
                # not every term is in the preorder + ideal,
                # and we identify whether a term is pure ideal
                return _is_pure_ideal(x)
            if x.is_constant(simplify=False) and is_true(x > 0):
                if x == 0:
                    return f({}, {})
                return f({frozenset(): SOSlist([(x, Integer(1))])}, {})
            if isinstance(x, CyclicExpr):
                return _recur_build(x.doit(deep=False))
            elif isinstance(x, UnevaluatedExpr):
                return _recur_build(x.args[0])
        return _recur_build(expr)

    def mul_sqr(self, expr: Union[Expr, Tuple[Expr, Expr]]) -> 'PSatz':
        """
        Return `self * expr**2`
        """
        if not isinstance(expr, (tuple, stuple)):
            expr = sympify(expr)
            expr = fraction(expr.doit().together())
        numer, denom = expr

        if numer == 0:
            return self.per({}, {}, PSATZ_UNIT, {})

        numer_preorder = {k: v.mul_sqr(numer) for k, v in self.numer_preorder.items()}
        denom_preorder = {k: v.mul_sqr(denom) for k, v in self.denom_preorder.items()}
        numer_ideal    = {k: v*numer**2 for k, v in self.numer_ideal.items()}
        denom_ideal    = {k: v*denom**2 for k, v in self.denom_ideal.items()}
        return self.per(numer_preorder, numer_ideal, denom_preorder, denom_ideal)

    def marginalize(self, ind: int, pop: bool = False) -> Tuple['PSatz', 'PSatz', 'PSatz', 'PSatz']:
        """
        Return `ps1, ps2, ps3, ps4` so that `self == (ps1 + ps2)/(ps3 + ps4)`
        and `ps1` and `ps3` contain terms involving `preorder[ind]`.
        """
        def _separate(p, i):
            if pop:
                p1 = {frozenset(set(k) - {ind}): v for k, v in p.items() if ind in k}
            else:
                p1 = {k: v for k, v in p.items() if ind in k}
            p2 = {k: v for k, v in p.items() if ind not in k}
            return  self.per(p1, {}, PSATZ_UNIT, {}),\
                    self.per(p2, i, PSATZ_UNIT, {})
        ps1, ps2 = _separate(self.numer_preorder, self.numer_ideal)
        ps3, ps4 = _separate(self.denom_preorder, self.denom_ideal)
        return ps1, ps2, ps3, ps4

    def join(a: 'PSatz', b: 'PSatz', ind: int, F: Optional[Expr] = None) -> 'PSatz':
        """
        Join two PSatzs to eliminate the `ind`-th preorder generator. The `ind`-th
        preorder generator of two PSatzs should imply opposite values.

        If the two PSatzs imply:
        ```
            F = (f * ps1 + ps2)/(f * ps3 + ps4) = (-f * ps5 + ps6)/(-f * ps7 + ps8)
        ```
        where `f` and `-f` are the `ind`-th preorder generators, then
        ```
            f = (ps2 - F * ps4)/(F * ps3 - ps1) = -(ps6 - F * ps8)/(F * ps7 - ps5)
            F = (F**2*(ps3*ps8 + ps4*ps7) + ps1*ps6 + ps2*ps5)/(ps1*ps8 + ps2*ps7 + ps3*ps6 + ps4*ps5)
        ```

        Examples
        ---------
        Consider (a, c, x, y, z) >= 0 and 4ac - b^2 = -by + z. Prove that:

            F = a*x^2 + b*x + c >= 0

        The inequality can be proved by simple arguments over the cases b >= 0 and -b >= 0. To
        establish a "joint" proof for the two cases, we let (i) u = b >= 0 (ii) v = -b >= 0
        and join the proof with respect to `u` and `v` (v = -u) by `PSatz.join`.

        >>> from sympy.abc import a, b, c, u, v, x, y, z
        >>> p1 = a*x**2 + u*x + c
        >>> p2 = a*(x + b/(2*a))**2 + (v*y + z)/(4*a) + (4*a*c - b**2 + b*y - z)/(4*a)
        >>> p1 = PSatz.from_sympy([a,c,u,x,y,z], [4*a*c-b**2+b*y-z], p1)
        >>> p2 = PSatz.from_sympy([a,c,v,x,y,z], [4*a*c-b**2+b*y-z], p2)
        >>> p3 = p1.join(p2, 2, F=a*x**2+b*x+c)
        >>> p3.as_expr().together()
        (a*x**2*y + c*y + x*z + x*(2*a*x + b)**2 + x*(4*a*c - b**2 + b*y - z))/(4*a*x + y)
        >>> p3.as_expr().factor()
        a*x**2 + b*x + c
        """
        if (a.preorder is not b.preorder and any(a.preorder[i] != b.preorder[i]
                for i in range(len(a.preorder)) if i != ind)) or a.ideal != b.ideal:
            raise ValueError('preorder or ideal not equal')

        if F is not None:
            F = fraction(sympify(F).as_expr().doit().together())
        else:
            F = (a.numerator, a.denominator)
        ps1, ps2, ps3, ps4 = a.marginalize(ind, pop=True)
        ps5, ps6, ps7, ps8 = b.marginalize(ind, pop=True)
        for ps in (ps5, ps6, ps7, ps8):
            ps.preorder = ps1.preorder # align preorder
        A = ps3*ps8 + ps4*ps7
        B = ps1*ps6 + ps2*ps5
        C = ps1*ps8 + ps2*ps7 + ps3*ps6 + ps4*ps5
        # return (F**2*A + B)/C
        A = A.mul_sqr((F[0], Integer(1)))
        B = B.mul_sqr((F[1], Integer(1)))
        C = C.mul_sqr((F[1], Integer(1)))
        D = A + B
        p1, i1 = D.numer_preorder, D.numer_ideal
        p2, i2 = C.numer_preorder, C.numer_ideal
        return a.per(p1, i1, p2, i2)


PSatz.zero = PSatz.new([], [], {}, {}, PSATZ_UNIT, {})
PSatz.one  = PSatz.new([], [], PSATZ_UNIT, {}, PSATZ_UNIT, {})
