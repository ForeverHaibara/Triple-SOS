from typing import (
    List, Tuple, Dict, FrozenSet, Union, Optional, Callable,
    TypeVar, Generic, Iterator
)

from sympy import Expr, Add, Mul, Rational, Integer, UnevaluatedExpr, fraction, sympify, latex, sqrt, true
from sympy import Tuple as stuple
from sympy.core.sympify import CantSympify, sympify
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.polyerrors import CoercionFailed

from .cyclic import CyclicExpr
from .exraw import HAS_EXRAW
from .soscone import EXRAWSOSCone, SOSCone, SOSElement, SOSlist
from .soscone import Ef, TExpr, Domain # for type annotations


def _pretty_print(psatz_elem, func=str):
    preorder, ideal = psatz_elem.preorder, psatz_elem.ideal

    # def _is_num(s: str) -> bool:
    #     if s.isdecimal():
    #         return True
    #     import re
    #     num_re = re.compile(r'^-?(?:\d+\.\d+|\d+)$')
    #     return bool(num_re.match(s))

    def _preorder_term(monom: FrozenSet[int], v: SOSElement) -> str:
        s = "*".join([f"({func(preorder[i])})" for i in monom]) + f"*({func(v)})"
        if not monom:
            s = s[1:] # strip the leading "*"
        return s

    def _ideal_term(ind: int, v: TExpr) -> str:
        return f"({func(ideal[ind])})*({func(v)})"

    def _pretty(p, i) -> str:
        ss = [_preorder_term(k, v) for k, v in p.items()]\
            + [_ideal_term(ind, v) for ind, v in i.items()]
        if len(ss) == 0:
            return "0"
        if len(ss) == 1:
            return  f"({ss[0]})"
        return f"({' + '.join(ss)})"

    numer = _pretty(psatz_elem.numer_preorder, psatz_elem.numer_ideal)
    denom = _pretty(psatz_elem.denom_preorder, psatz_elem.denom_ideal)

    if denom == "1":
        return numer
    return f"{numer}/{denom}"


class PSatzDomain(Generic[Ef]):
    cone: SOSCone[Ef]
    preorder: List[TExpr[Ef]]
    ideal: List[TExpr[Ef]]
    def __init__(self, cone: SOSCone[Ef], preorder: List[TExpr[Ef]] = [], ideal: List[TExpr[Ef]] = []):
        self.cone = cone
        self.preorder = preorder
        self.ideal = ideal
        onezero = {frozenset(): cone.one}, {}
        self.one = PSatzElement.new(self, *onezero, *onezero)
        self.zero = PSatzElement.new(self, {}, {}, *onezero)

    def __str__(self) -> str:
        return f"PSatzDomain(cone={self.cone!s}, preorder={self.preorder!s}, ideal={self.ideal!s})"

    def __repr__(self) -> str:
        return f"PSatzDomain(cone={self.cone!r}, preorder={self.preorder!r}, ideal={self.ideal!r})"

    def __eq__(self, other: 'PSatzDomain[Ef]') -> bool:
        return self.cone == other.cone and self.preorder == other.preorder and self.ideal == other.ideal

    @property
    def algebra(self) -> Domain[TExpr[Ef]]:
        return self.cone.algebra

    @property
    def domain(self) -> Domain[Ef]:
        return self.cone.domain

    @property
    def onezero(self) -> Tuple[Dict[FrozenSet[int], SOSElement[Ef]], Dict[int, TExpr[Ef]]]:
        return (self.one.denom_preorder, self.one.denom_ideal)

    def preorder_term(self, monom: Iterator[int]) -> TExpr[Ef]:
        z = self.algebra.one
        for i in monom:
            z = z * self.preorder[i]
        return z

    def preorder_term_expr(self, monom: Iterator[int], preorder_alias: Optional[List[Expr]]=None, evaluate: bool=True) -> Expr:
        if preorder_alias is not None:
            return Mul(*[preorder_alias[i] for i in monom])
        alg = self.algebra
        _eval = (lambda x: x) if evaluate else UnevaluatedExpr
        return Mul(*[_eval(alg.to_sympy(self.preorder[i])) for i in monom])

    def ideal_term(self, ind: int) -> TExpr[Ef]:
        return self.ideal[ind]

    def ideal_term_expr(self, ind: int, ideal_alias: Optional[List[Expr]]=None, evaluate: bool=True) -> Expr:
        if ideal_alias is not None:
            return ideal_alias[ind]
        alg = self.algebra
        _eval = (lambda x: x) if evaluate else UnevaluatedExpr
        return _eval(alg.to_sympy(self.ideal[ind]))

    def _to_algebra(self, preorder: Dict[FrozenSet[int], SOSElement[Ef]], ideal: Dict[int, TExpr[Ef]]) -> TExpr[Ef]:
        z = self.algebra.zero
        for m, v in preorder.items():
            z += self.preorder_term(m) * v.to_algebra()
        for m, v in ideal.items():
            z += self.ideal[m] * v
        return z

    def _to_expr(self, preorder: Dict[FrozenSet[int], SOSElement[Ef]], ideal: Dict[int, TExpr[Ef]],
            preorder_alias: Optional[List[Expr]]=None, ideal_alias: Optional[List[Expr]]=None, evaluate: bool=True) -> Expr:
        alg = self.algebra
        return Add(
            *[self.preorder_term_expr(k, preorder_alias, evaluate) * v.as_expr() for k, v in preorder.items()],
            *[self.ideal_term_expr(i, ideal_alias, evaluate) * alg.to_sympy(v) for i, v in ideal.items()],
        )

    def sum(self, elements: List['PSatzElement[Ef]']) -> 'PSatzElement[Ef]':
        e = self.zero
        for e2 in elements:
            e += e2
        return e

    def prod(self, elements: List['PSatzElement[Ef]']) -> 'PSatzElement[Ef]':
        e = self.one
        for e2 in elements:
            e *= e2
        return e

    def from_ground(self, x: Ef) -> 'PSatzElement[Ef]':
        return self.from_sos_element(self.cone.from_ground(x))

    def from_sos_element(self, element: SOSElement[Ef]) -> 'PSatzElement[Ef]':
        # if element.cone != self.cone:
        #     raise ValueError("Cannot construct a PSatzElement from other cones.")
        return PSatzElement.new(self, {frozenset(): element}, self.zero.numer_ideal, *self.onezero)

    def from_sympy(self, expr: object,
        preorder_alias: Optional[Union[List[TExpr[Ef]], Dict[List[TExpr[Ef]], int]]]=None,
        ideal_alias: Optional[Union[List[TExpr[Ef]], Dict[List[TExpr[Ef]], int]]]=None
    ) -> Optional['PSatzElement']:
        expr = sympify(expr)
        alg = self.algebra
        if preorder_alias is None:
            preorder_alias = [alg.to_sympy(v) for v in self.preorder]
        if not isinstance(preorder_alias, (dict,)):
            preorder_alias = {v: i for i, v in enumerate(preorder_alias)}
        if ideal_alias is None:
            ideal_alias = [alg.to_sympy(v) for v in self.ideal]
        if not isinstance(ideal_alias, (dict,)):
            ideal_alias = {v: i for i, v in enumerate(ideal_alias)}
        try:
            return self._rebuild_expr(expr, preorder_alias, ideal_alias)
        except CoercionFailed:
            pass
        return None

    def _rebuild_expr_in_ideal(self, x, mp, mi) -> 'PSatzElement':
        """
        Whether an expression lies in the ideal.
        Raises CoercionFailed for expressions that cannot be converted to elements in the ideal.
        """
        _rebuild_expr_in_ideal = lambda z: self._rebuild_expr_in_ideal(z, mp, mi)
        if x in mi:
            return PSatzElement.new(self, {}, {mi[x]: self.algebra.one}, *self.onezero)
        if x.is_Add:
            return self.sum([_rebuild_expr_in_ideal(a) for a in x.args])
        elif x.is_Mul or x.is_Pow:
            xargs = x.args if x.is_Mul else (x,)
            for i, a in enumerate(xargs):
                power = 1
                if a.is_Pow:
                    a, power = a.base, a.exp
                    if not (isinstance(power, Integer) and power > 0):
                        continue
                try:
                    y = _rebuild_expr_in_ideal(a)
                    if y is not None and power > 0:
                        other = Mul(*[x.args[j] for j in range(len(x.args)) if j != i])
                        if power > 1:
                            other = other * (a**(power - 1))
                        other = fraction(other)
                        denom = self.algebra.from_sympy(other[1])
                        numer = self.algebra.from_sympy(other[0]) * denom
                        return PSatzElement.new(self,
                                {}, {m: v*numer for m, v in y.numer_ideal.items()},
                                {m: v.mul_sqr(denom) for m, v in y.denom_preorder.items()},
                                y.denom_ideal)
                except CoercionFailed:
                    pass
        elif isinstance(x, CyclicExpr):
            return _rebuild_expr_in_ideal(x.doit(deep=False))
        elif isinstance(x, UnevaluatedExpr):
            return _rebuild_expr_in_ideal(x.args[0])
        raise CoercionFailed(f"Cannot convert {x!r} to an element in the ideal.")

    def _rebuild_expr(self, x, mp, mi) -> 'PSatzElement[Ef]':
        _rebuild_expr = lambda z: self._rebuild_expr(z, mp, mi)

        f = lambda p, i: PSatzElement.new(self, p, i, *self.onezero)

        if x.is_Pow:
            if isinstance(x.exp, Integer):
                if x.exp % 2 == 0:
                    base = SOSElement.new(self.cone, 
                        [(self.domain.one, self.algebra.from_sympy(x.base)**(abs(int(x.exp))//2))])
                    if x.exp > 0:
                        return self.from_sos_element(base)
                    else:
                        return PSatzElement.new(self, *self.onezero, {frozenset(): base}, {})
                else:
                    return _rebuild_expr(x.base)**int(x.exp)
        if x in mi:
            return f({}, {mi[x]: self.algebra.one})
        if x in mp:
            return f({frozenset({mp[x]}): self.cone.one}, {})
        if x.is_Add:
            try:
                return self.sum([_rebuild_expr(a) for a in x.args])
            except CoercionFailed:
                pass
            # e.g., sqrt(2) - 1 --> goto x.is_constant() == True case
        elif x.is_Mul:
            try:
                return self.prod([_rebuild_expr(a) for a in x.args])
            except CoercionFailed:
                pass
            # not every term is in the preorder + ideal,
            # and we identify whether a term is purely in ideal
            return self._rebuild_expr_in_ideal(x, mp, mi)
        if x.is_constant(simplify=False):
            if x == 0:
                return self.zero
            elif x > 0:
                return self.from_ground(x)
            raise CoercionFailed(f"Cannot convert {x!r} to PSatzElement.")
        if isinstance(x, CyclicExpr):
            return _rebuild_expr(x.doit(deep=False))
        elif isinstance(x, UnevaluatedExpr):
            return _rebuild_expr(x.args[0])
        raise CoercionFailed(f"Cannot convert {x!r} to PSatzElement.")


class PSatzElement(DomainElement, CantSympify, Generic[Ef]):
    psatz_domain: PSatzDomain[Ef]
    numer_preorder: Dict[FrozenSet[int], SOSElement[Ef]]
    numer_ideal: Dict[int, TExpr[Ef]]
    denom_preorder: Dict[FrozenSet[int], SOSElement[Ef]]
    denom_ideal: Dict[int, TExpr[Ef]]
    def __new__(cls, psatz_domain: PSatzDomain[Ef],
        numer_preorder: Optional[Dict[FrozenSet[int], SOSElement]]=None,
        numer_ideal: Optional[Dict[int, TExpr[Ef]]]=None,
        denom_preorder: Optional[Dict[FrozenSet[int], SOSElement]]=None,
        denom_ideal: Optional[Dict[int, TExpr[Ef]]]=None
    ):
        numer_preorder = numer_preorder or psatz_domain.zero.numer_preorder
        numer_ideal = numer_ideal or psatz_domain.zero.numer_ideal

        if denom_preorder is None and denom_ideal is None:
            denom_preorder = psatz_domain.one.denom_preorder
            denom_ideal = psatz_domain.one.denom_ideal
        denom_preorder = denom_preorder or psatz_domain.zero.numer_preorder # defaults to empty dict
        denom_ideal = denom_ideal or psatz_domain.one.denom_ideal
        return cls.new(psatz_domain,
            numer_preorder, numer_ideal, denom_preorder, denom_ideal)
    
    @classmethod
    def new(cls, psatz_domain: PSatzDomain, numer_preorder, numer_ideal, denom_preorder, denom_ideal) -> 'PSatzElement':
        obj = super().__new__(cls)
        obj.psatz_domain = psatz_domain
        obj.numer_preorder = numer_preorder
        obj.numer_ideal = numer_ideal
        obj.denom_preorder = denom_preorder
        obj.denom_ideal = denom_ideal
        return obj

    def per(self, numer_preorder, numer_ideal, denom_preorder, denom_ideal) -> 'PSatzElement[Ef]':
        return self.new(self.psatz_domain, numer_preorder, numer_ideal, denom_preorder, denom_ideal)

    def __str__(self) -> str:
        return _pretty_print(self, func=str)

    def __repr__(self) -> str:
        return _pretty_print(self, func=repr)

    def _repr_latex_(self) -> str:
        return self.as_expr()._repr_latex_()

    @property
    def preorder(self) -> Dict[FrozenSet[int], TExpr[Ef]]:
        return self.psatz_domain.preorder

    @property
    def ideal(self) -> Dict[int, TExpr[Ef]]:
        return self.psatz_domain.ideal

    @property
    def cone(self) -> SOSCone[Ef]:
        return self.psatz_domain.cone

    @property
    def algebra(self) -> Domain[TExpr[Ef]]:
        return self.psatz_domain.algebra

    @property
    def domain(self) -> Domain[Ef]:
        return self.psatz_domain.domain

    @property
    def is_zero(self) -> bool:
        alg = self.algebra
        return all(v.is_zero for v in self.numer_preorder.values())\
             and all(alg.is_zero(v) for v in self.numer_ideal.values())

    @property
    def zero(self) -> 'PSatzElement[Ef]':
        return self.psatz_domain.zero

    @property
    def one(self) -> 'PSatzElement[Ef]':
        return self.psatz_domain.one

    @property
    def onezero(self) -> Tuple[Dict[FrozenSet[int], SOSElement[Ef]], Dict[int, TExpr[Ef]]]:
        return self.psatz_domain.onezero

    @property
    def numerator(self) -> TExpr[Ef]:
        return self.psatz_domain._to_algebra(self.numer_preorder, self.numer_ideal)

    @property
    def denominator(self) -> TExpr[Ef]:
        return self.psatz_domain._to_algebra(self.denom_preorder, self.denom_ideal)

    @property
    def is_in_ideal(self) -> bool:
        return all(v.is_zero for v in self.numer_preorder.values())

    def as_expr(self, preorder_alias: Optional[List[Expr]]=None, ideal_alias: Optional[List[Expr]]=None,
            evaluate: bool=True) -> Expr:
        numer = self.psatz_domain._to_expr(self.numer_preorder, self.numer_ideal,
            preorder_alias, ideal_alias, evaluate=evaluate)
        denom = self.psatz_domain._to_expr(self.denom_preorder, self.denom_ideal,
            preorder_alias, ideal_alias, evaluate=evaluate)
        return numer / denom

    def __add__(self, other: object) -> 'PSatzElement[Ef]':
        if isinstance(other, PSatzElement):
            if self.psatz_domain == other.psatz_domain:
                return self._add(other)
            raise ValueError("PSatzElement domains must be equal")
        return NotImplemented

    def _add(a: 'PSatzElement[Ef]', b: 'PSatzElement[Ef]') -> 'PSatzElement[Ef]':
        preorder, ideal = a.preorder, a.ideal
        # (p1 + i1)/(p2 + i2) + (p3 + i3)/(p4 + i4)
        p1, i1, p2, i2 = a.numer_preorder, a.numer_ideal, a.denom_preorder, a.denom_ideal
        p3, i3, p4, i4 = b.numer_preorder, b.numer_ideal, b.denom_preorder, b.denom_ideal
        if p2 == p4 and i2 == i4:
            p5, i5 = _preorder_ideal_add(a.psatz_domain, p1, i1, p3, i3)
            return a.per(p5, i5, p2, i2)
        p5, i5 = _preorder_ideal_add(a.psatz_domain,
            *_preorder_ideal_mul(a.psatz_domain, p1, i1, p4, i4),
            *_preorder_ideal_mul(a.psatz_domain, p3, i3, p2, i2)
        )
        p6, i6 = _preorder_ideal_mul(a.psatz_domain, p2, i2, p4, i4)
        return a.per(p5, i5, p6, i6)

    def __sub__(self, other: object) -> 'PSatzElement[Ef]':
        if isinstance(other, PSatzElement):
            if self.psatz_domain == other.psatz_domain:
                if not other.is_in_ideal:
                    raise ValueError("Cannot subtract PSatzElement that is not in the ideal.")
                return self._sub(other)
            raise ValueError("Cannot subtract PSatzElements of different PSatzDomains.")
        return NotImplemented

    def _sub(a: 'PSatzElement[Ef]', b: 'PSatzElement[Ef]') -> 'PSatzElement[Ef]':
        return a._add(b._neg())

    def __pos__(self) -> 'PSatzElement[Ef]':
        return self

    def __neg__(self) -> 'PSatzElement[Ef]':
        if not self.is_in_ideal:
            raise ValueError("Cannot negate PSatzElement that is not in the ideal.")
        return self._neg()

    def _neg(self) -> 'PSatzElement[Ef]':
        return self.per(self.numer_preorder, {k: -v for k, v in self.numer_ideal.items()},
                            self.denom_preorder, self.denom_ideal)

    def __mul__(self, other: object) -> 'PSatzElement[Ef]':
        if isinstance(other, PSatzElement):
            if self.psatz_domain == other.psatz_domain:
                return self._mul(other)
            raise ValueError("Cannot multiply PSatzElements of different PSatzDomains.")
        
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
            raise ValueError("Cannot multiply PSatzElement with negative coeff.")
        return self._mul_ground(_other)

    def __rmul__(self, other: object) -> 'PSatzElement[Ef]':
        if isinstance(other, PSatzElement):
            if self.psatz_domain == other.psatz_domain:
                return self._mul(other)
            raise ValueError("Cannot multiply PSatzElements of different cones.")
        
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
            raise ValueError("Cannot multiply PSatzElement with negative coeff.")
        return self._rmul_ground(_other)

    def _mul(a: 'PSatzElement[Ef]', b: 'PSatzElement[Ef]') -> 'PSatzElement[Ef]':
        preorder, ideal = a.preorder, a.ideal
        # (p1 + i1)/(p2 + i2) * (p3 + i3)/(p4 + i4)
        p1, i1, p2, i2 = a.numer_preorder, a.numer_ideal, a.denom_preorder, a.denom_ideal
        p3, i3, p4, i4 = b.numer_preorder, b.numer_ideal, b.denom_preorder, b.denom_ideal
        p5, i5 = _preorder_ideal_mul(a.psatz_domain, p1, i1, p3, i3)
        p6, i6 = _preorder_ideal_mul(a.psatz_domain, p2, i2, p4, i4)
        return a.per(p5, i5, p6, i6)

    def _mul_ground(a: 'PSatzElement[Ef]', b: Ef) -> 'PSatzElement[Ef]':
        preorder, ideal = a.preorder, a.ideal
        # (p1 + i1)/(p2 + i2) * b
        p1, i1, p2, i2 = a.numer_preorder, a.numer_ideal, a.denom_preorder, a.denom_ideal
        p3 = {k: v._mul_ground(b) for k, v in p1.items()}
        i3 = {k: v * b for k, v in i1.items()}
        return a.per(p3, i3, p2, i2)

    def _rmul_ground(a: 'PSatzElement[Ef]', b: Ef) -> 'PSatzElement[Ef]':
        preorder, ideal = a.preorder, a.ideal
        # b * (p1 + i1)/(p2 + i2)
        p1, i1, p2, i2 = a.numer_preorder, a.numer_ideal, a.denom_preorder, a.denom_ideal
        p3 = {k: v._rmul_ground(b) for k, v in p1.items()}
        i3 = {k: b * v for k, v in i1.items()}
        return a.per(p3, i3, p2, i2)

    def __truediv__(self, other: object) -> 'PSatzElement[Ef]':
        if isinstance(other, PSatzElement):
            if self.psatz_domain == other.psatz_domain:
                return self._truediv(other)
            raise ValueError("Cannot divide PSatzElements of different PSatzDomains.")

        if self.domain.of_type(other):
            _other = other
        else:
            try:
                _other = self.domain.convert(other) # type: ignore
            except CoercionFailed:
                return NotImplemented
        sgn = self.cone.sign(_other)
        if sgn == 0:
            raise ZeroDivisionError("Cannot divide PSatzElement with zero coeff.")
        elif sgn < 0:
            raise ValueError("Cannot divide PSatzElement with negative coeff.")
        return self._truediv_ground(_other)

    def __rtruediv__(self, other: object) -> 'PSatzElement[Ef]':
        if isinstance(other, PSatzElement):
            if self.psatz_domain == other.psatz_domain:
                return other._truediv(self)
            raise ValueError("Cannot divide PSatzElements of different PSatzDomains.")

        if self.domain.of_type(other):
            _other = other
        else:
            try:
                _other = self.domain.convert(other) # type: ignore
            except CoercionFailed:
                return NotImplemented
        sgn = self.cone.sign(_other)
        if sgn == 0:
            raise ZeroDivisionError("Cannot divide PSatzElement with zero coeff.")
        elif sgn < 0:
            raise ValueError("Cannot divide PSatzElement with negative coeff.")
        return self._rtruediv_ground(_other)

    def _truediv(a: 'PSatzElement[Ef]', b: 'PSatzElement[Ef]') -> 'PSatzElement[Ef]':
        return a._mul(b.inverse())

    def _truediv_ground(a: 'PSatzElement[Ef]', b: Ef) -> 'PSatzElement[Ef]':
        preorder, ideal = a.preorder, a.ideal
        # b * (p1 + i1)/(p2 + i2)
        p1, i1, p2, i2 = a.numer_preorder, a.numer_ideal, a.denom_preorder, a.denom_ideal
        p3 = {k: v._mul_ground(b) for k, v in p2.items()}
        i3 = {k: v * b for k, v in i2.items()}
        return a.per(p1, i1, p3, i3)

    def _rtruediv_ground(a: 'PSatzElement[Ef]', b: Ef) -> 'PSatzElement[Ef]':
        return a.inverse()._rmul_ground(b)

    def inverse(self) -> 'PSatzElement[Ef]':
        if self.is_zero:
            raise ZeroDivisionError("Cannot invert zero PSatzElement.")
        return self.per(self.denom_preorder, self.denom_ideal, self.numer_preorder, self.numer_ideal)

    def __pow__(self, n: int) -> 'PSatzElement[Ef]':
        if not isinstance(n, int):
            raise TypeError("Cannot raise SOSElement to non-integer power.")
        return self._pow_int(n)

    def _pow_int(self, n: int) -> 'PSatzElement[Ef]':
        if n == 1:
            return self
        if n < 0:
            return self.inverse()._pow_int(-n)
        if self.is_zero:
            if n == 0:
                raise ValueError("0**0")
            if n > 0:
                return self.zero
            raise ZeroDivisionError("Cannot raise zero PSatzElement to negative power.")
        if n == 0:
            return self.one

        # n > 0

        if n % 2 == 0:
            cone = self.cone
            dom = self.domain
            p1, i1, p2, i2 = {frozenset(): SOSElement.new(cone, [(dom.one, self.numerator**(n//2))])}, {},\
                {frozenset(): SOSElement.new(cone, [(dom.one, self.denominator**(n//2))])}, {}
        else:
            p1, i1, p2, i2 = self.numer_preorder, self.numer_ideal, self.denom_preorder, self.denom_ideal
            if n != 1:
                numer_sqrt = self.numerator**(n//2)
                numer_sqr2 = numer_sqrt**2
                denom_sqrt = self.denominator**(n//2)
                denom_sqr2 = denom_sqrt**2
                p1, i1, p2, i2 = {m: v.mul_sqr(numer_sqrt) for m, v in p1.items()},\
                    {m: v * numer_sqr2 for m, v in i1.items()},\
                    {m: v.mul_sqr(denom_sqrt) for m, v in p2.items()},\
                    {m: v * denom_sqr2 for m, v in i2.items()}

        return self.per(p1, i1, p2, i2)

    def mul_sqr(self, numer: Optional[TExpr[Ef]] = None, denom: Optional[TExpr[Ef]] = None) -> 'PSatzElement[Ef]':
        p1, i1, p2, i2 = self.numer_preorder, self.numer_ideal, self.denom_preorder, self.denom_ideal
        if numer is not None:
            if self.algebra.is_zero(numer):
                return self.zero
            numer_sqr = numer**2
            p1 = {k: v.mul_sqr(numer) for k, v in p1.items()}
            i1 = {k: v * numer_sqr for k, v in i1.items()}
        if denom is not None:
            denom_sqr = denom**2
            p2 = {k: v.mul_sqr(denom) for k, v in p2.items()}
            i2 = {k: v * denom_sqr for k, v in i2.items()}
        return self.per(p1, i1, p2, i2)

    def marginalize(self, ind: int, pop: bool = False) \
            -> Tuple['PSatzElement[Ef]', 'PSatzElement[Ef]', 'PSatzElement[Ef]', 'PSatzElement[Ef]']:
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
            return  self.per(p1, {}, *self.onezero),\
                    self.per(p2, i, *self.onezero)
        ps1, ps2 = _separate(self.numer_preorder, self.numer_ideal)
        ps3, ps4 = _separate(self.denom_preorder, self.denom_ideal)
        return ps1, ps2, ps3, ps4

    def join(a: 'PSatzElement[Ef]', b: 'PSatzElement[Ef]', ind: int,
            numer: Optional[TExpr[Ef]] = None, denom: Optional[TExpr[Ef]] = None) -> 'PSatzElement[Ef]':
        """
        Join two PSatzs to eliminate the `ind`-th preorder generator. The `ind`-th
        preorder generator of two PSatzs should imply opposite values.

        Suppose `F = numer/denom`. If the two PSatzs imply:
        ```
            F = (f * ps1 + ps2)/(f * ps3 + ps4) = (-f * ps5 + ps6)/(-f * ps7 + ps8)
        ```
        where `f` and `-f` are the `ind`-th preorder generators, then
        ```
            f = (ps2 - F * ps4)/(F * ps3 - ps1) = -(ps6 - F * ps8)/(F * ps7 - ps5)
            F = (F**2*(ps3*ps8 + ps4*ps7) + ps1*ps6 + ps2*ps5)/(ps1*ps8 + ps2*ps7 + ps3*ps6 + ps4*ps5)
        ```
        """
        if (a.preorder is not b.preorder and any(a.preorder[i] != b.preorder[i]
                for i in range(len(a.preorder)) if i != ind)) or a.ideal != b.ideal:
            # only the i-th generator is allowed to be different
            raise ValueError("Cannot join PSatzElements with different cones or ideals.")

        if numer is None or denom is None:
            if not (numer is None and denom is None):
                raise ValueError("Arguments numer and denom should be both provided or be both None,"\
                                +f" but got {numer!r} and {denom!r}.")
            numer, denom = a.numerator, a.denominator
        ps1, ps2, ps3, ps4 = a.marginalize(ind, pop=True)
        ps5, ps6, ps7, ps8 = b.marginalize(ind, pop=True)
        for ps in (ps5, ps6, ps7, ps8):
            ps.psatz_domain = ps1.psatz_domain # align preorder
        A = ps3*ps8 + ps4*ps7
        B = ps1*ps6 + ps2*ps5
        C = ps1*ps8 + ps2*ps7 + ps3*ps6 + ps4*ps5
        # return (F**2*A + B)/C
        A = A.mul_sqr(numer)
        B = B.mul_sqr(denom)
        C = C.mul_sqr(denom)
        D = A + B
        p1, i1 = D.numer_preorder, D.numer_ideal
        p2, i2 = C.numer_preorder, C.numer_ideal
        return a.per(p1, i1, p2, i2)

def _preorder_ideal_add(psatz_domain: PSatzDomain[Ef],
    p1: Dict[FrozenSet[int], SOSElement[Ef]],
    i1: Dict[int, SOSElement[Ef]],
    p2: Dict[FrozenSet[int], SOSElement[Ef]],
    i2: Dict[int, SOSElement[Ef]]
):
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


def _preorder_ideal_mul(psatz_domain: PSatzDomain[Ef],
    p1: Dict[FrozenSet[int], SOSElement[Ef]],
    i1: Dict[int, SOSElement[Ef]],
    p2: Dict[FrozenSet[int], SOSElement[Ef]],
    i2: Dict[int, SOSElement[Ef]]
):
    """Mul (p1 + i1) and (p2 + i2) given preorder and ideal."""
    # p3 = p1 * p2
    p3 = {}
    for m1, v1 in p1.items():
        for m2, v2 in p2.items():
            m3 = m1 ^ m2
            m4 = m1 & m2
            mul = psatz_domain.preorder_term(m4)
            v = (v1 * v2).mul_sqr(mul)
            if m3 in p3:
                p3[m3] = p3[m3] + v
            else:
                p3[m3] = v

    p1expr = psatz_domain._to_algebra(p1, {})
    p2expr = psatz_domain._to_algebra(p2, {})
    i2expr = psatz_domain._to_algebra({}, i2)

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


class PSatz(Generic[Ef]):
    """
    Consider the semialgebraic set formed by inequalities `G1,...,Gn >= 0`
    and equalities `H1,...,Hm == 0`. An element is nonnegative over
    the set if it can be represented as

        `(sum(Gi1*...Gik * SOS_i) + sum(Hi * fi))/(sum(Gj1*...Gjk * SOS_j) + sum(Hj * fj))`

    Here we assume the denominator is always nonzero. The PSatz stores such elements
    in a structured way and provides arithmetics (add, mul, div, pow) for them.    

    Attributes
    ----------
    rep: PSatzElement
        The internal representation of the PSatz.
        It uses sympy domains.
    preorder: List[Expr]
        The generators of the preorder of the PSatz.
    ideal: List[Expr]
        The generators of the ideal of the PSatz.
    numer_preorder: Dict[FrozenSet[int], SOSlist]
        The preorder of the numerator.
    numer_ideal: Dict[int, Expr]
        The ideal of the numerator. 
    denom_preorder: Dict[FrozenSet[int], SOSlist]
        The preorder of the denominator.
    denom_ideal: Dict[int, Expr]
        The ideal of the denominator.


    Examples
    ---------

    ## Basic Tutorials

    ### Building PSatz by `PSatz.from_sympy`

    The code below converts `a*(b-c)**2 + 2*a*b*c*(a+b-c)**2 + a**2 + 2` to a PSatz
    element given `a, b, c >= 0`.

    >>> from sympy.abc import a, b, c, x, y
    >>> ps = PSatz.from_sympy(a*(b-c)**2 + 2*a*b*c*(a+b-c)**2 + a**2 + 2, [a,b,c], [])
    >>> ps
    ((2*(1)**2 + 1*(a)**2) + (a)*(1*(b - c)**2) + (a)*(b)*(c)*(2*(a + b - c)**2))/((1*(1)**2))

    To access the SOSlists associated with each generator of the preorder,
    use `numer_preorder`.
    >>> ps.numer_preorder
    {frozenset(): 2*(1)**2 + 1*(a)**2, frozenset({0}): 1*(b - c)**2, frozenset({0, 1, 2}): 2*(a + b - c)**2}
    >>> ps.numer_preorder[frozenset()]
    2*(1)**2 + 1*(a)**2
    >>> type(ps.numer_preorder[frozenset()])
    <class 'triples.utils.expressions.soscone.SOSlist'>
    >>> ps.numer_preorder[frozenset()].items()
    [(2, 1), (1, a)]

    PSatz instances can also be constructed directly without `.from_sympy`:

    >>> PSatz(a*(b-c)**2 + 2*a*b*c*(a+b-c)**2 + a**2 + 2, [a,b,c], [])
    ((2*(1)**2 + 1*(a)**2) + (a)*(1*(b - c)**2) + (a)*(b)*(c)*(2*(a + b - c)**2))/((1*(1)**2))

    However, when the expression is not in the desired form, `PSatz.from_sympy` returns `None`.
    But `PSatz(...)` will raise an Exception:

    >>> PSatz.from_sympy(a**2 - 2, [a], []) is None
    True
    >>> PSatz(a**2 - 2, [a], []) # doctest: +SKIP
    Traceback (most recent call last):
    ...
    TypeError: Cannot convert a**2 - 2 to PSatz.


    ### General PSatz

    In general, a PSatz instance represents:

        `(ps.numer_preorder + ps.numer_ideal)/(ps.denom_preorder + ps.denom_ideal)`

    In `PSatz.from_sympy`, the second and the third arguments specify the generators
    of the preorder and the ideal (inequalities and equalities), repsectively.
    If not provided, they are considered as empty lists.

    Consider `(b**2 - a*x)/(a*(2*a+b)**2 + x + a) >= 0` given `a >= 0, x == 0`. We
    convert it to a PSatz instance with:
    
    >>> ps = PSatz.from_sympy((b**2 - a*x - 3*x)/(a*(2*a+b)**2 + x + a), [a], [x])
    >>> ps
    ((1*(b)**2) + (x)*(-a - 3))/((a)*(1*(1)**2 + 1*(2*a + b)**2) + (x)*(1))
    >>> ps.numer_preorder
    {frozenset(): 1*(b)**2}
    >>> ps.numer_ideal
    {0: -a - 3}
    >>> ps.denom_preorder
    {frozenset({0}): 1*(1)**2 + 1*(2*a + b)**2}
    >>> ps.denom_ideal
    {0: 1}


    ### Using aliases

    The `PSatz.from_sympy` function only identifies expressions that are explicitly in the desired form.

    >>> PSatz.from_sympy(a**2 - a**3, [1 - a], []) is None
    True

    >>> PSatz.from_sympy(a**2*(1 - a), [1 - a], [])
    ((1 - a)*(1*(a)**2))/((1*(1)**2))

    Consider `2*(a + 2)*(a - 1)**2 + (a**3 - 2)**2 + (a + 2) + 5 >= 0` given `a + 2 >= 0`.
    However, a direct input will fail because (a + 2) + 5 will be simplified to (a + 7).
    To resolve the problem, it is possible to replace (a + 2) by symbols and then use a dict
    to indicate their actual values:

    >>> PSatz.from_sympy(2*x*(a - 1)**2 + (a**3 - 2)**2 + x + 5 , {x: a + 2}, [])
    ((5*(1)**2 + 1*(a**3 - 2)**2) + (a + 2)*(1*(1)**2 + 2*(a - 1)**2))/((1*(1)**2))


    ### Using unevaluated expressions

    To avoid expressions from being simplified that break the desired PSatz from, it is also
    suggested to use `sympy.UnevaluatedExpr` to prevent expressions from being expanded.

    >>> PSatz.from_sympy(x + y, [], [x - 2, y + 2]) is None
    True

    >>> from sympy import UnevaluatedExpr as ue
    >>> ue(x - 2) + ue(y + 2)
    (x - 2) + (y + 2)
    >>> ps = PSatz.from_sympy(ue(x - 2) + ue(y + 2), [], [x - 2, y + 2])
    >>> ps
    ((x - 2)*(1) + (y + 2)*(1))/((1*(1)**2))

    Use `evaluate=False` in `PSatz.as_expr` to keep the terms unevaluated:

    >>> ps.as_expr()
    x + y
    >>> ps.as_expr(evaluate=False)
    (x - 2) + (y + 2)


    ## Low-level Operations

    The following is an introduction of low-level operations over PSatz.

    The relationship between `PSatz` and `PSatzElement` is similar to that between
    `sympy.Poly` and `sympy.polys.polyclasses.DMP`. The latter stores the coefficients
    and expressions in a low-level, compact data structure using the knowledge of
    the domain or algebra, while the former is an interface that converts the
    low-level data structure to `sympy.Expr` when accessing.

    ### Specifying domains

    The internal represetation `rep` of a Psatz is equipped with a domain and an algebra.
    They should be `sympy.polys.domains.domain.Domain` instances.

    >>> from sympy import ZZ
    >>> ps1 = PSatz.from_sympy((b**2 - a*x - 3*x)/(a*(2*a+b)**2 + x + a), [a], [x])
    >>> ps1.cone
    SOSCone(algebra=EXRAW, domain=EXRAW)
    >>> ps2 = PSatz.from_sympy((b**2 - a*x - 3*x)/(a*(2*a+b)**2 + x + a), [a], [x], algebra=ZZ[a,b,x])
    >>> ps2.cone
    SOSCone(algebra=ZZ[a,b,x], domain=ZZ)

    The algebra and the domain of the SOSCone control the internal computation of PSatz.
    For example, on algebra EXRAW, expressions are not expanded by default. But on the domain ZZ[a,b,x],
    polynomials are always expanded.

    >>> ps1**2
    ((1*(b**2 + x*(-a - 3))**2))/((1*(a*((2*a + b)**2 + 1) + x)**2))
    >>> ps2**2
    ((1*(-a*x + b**2 - 3*x)**2))/((1*(4*a**3 + 4*a**2*b + a*b**2 + a + x)**2))

    ### Building from PSatzElements

    PSatz instances are always built from PSatzElement instances internally.
    Consider representing `((x - y)**2 + 4*(x*y - 1))/(x + y + 2) >= 0` given `x, y >= 0` and `x*y == 1`.
    To build its PSatz from PSatzElement, it should be:

    >>> cone = SOSCone(ZZ[x,y], ZZ)
    >>> numer_preorder = {frozenset(): SOSElement(cone, [(1, x - y)])}
    >>> numer_ideal = {0: cone.algebra(4)}
    >>> denom_preorder = {frozenset(): cone.one * 2, frozenset({0}): cone.one, frozenset({1}): cone.one}
    >>> ps_dom = PSatzDomain(cone, [cone.algebra(x), cone.algebra(y)], [cone.algebra(x*y-1)])
    >>> ps_elem = PSatzElement(ps_dom, numer_preorder, numer_ideal, denom_preorder, {})
    >>> ps = PSatz.new(ps_elem)
    >>> ps
    ((1*(x - y)**2) + (x*y - 1)*(4))/((2*(1)**2) + (x)*(1*(1)**2) + (y)*(1*(1)**2))
    >>> ps.as_expr().factor()
    x + y - 2


    See Also
    --------
    triples.utils.expressions.soscone.SOSlist
    """

    rep: PSatzElement[Ef]
    def __new__(cls, arg,
        preorder: Optional[Union[List[Expr], Dict[Expr, Expr]]]=None,
        ideal: Optional[Union[List[Expr], Dict[Expr, Expr]]]=None,
        cone: Optional[SOSCone]=None,
        algebra: Optional[Domain[TExpr[Ef]]]=None,
        domain: Optional[Domain[Ef]]=None
    ):
        if isinstance(arg, PSatz):
            return arg
        elif isinstance(arg, PSatzElement):
            return cls.new(arg)
            
        obj = cls.from_sympy(arg, preorder=preorder, ideal=ideal,
                cone=cone, algebra=algebra, domain=domain)
        if obj is not None:
            return obj
        raise TypeError(f"Cannot convert {arg!r} to PSatz.")

    @property
    def psatz_domain(self) -> PSatzDomain[Ef]:
        return self.rep.psatz_domain

    @property
    def preorder(self) -> List[Expr]:
        alg = self.algebra
        return [alg.to_sympy(v) for v in self.psatz_domain.preorder]

    @property
    def ideal(self) -> List[Expr]:
        alg = self.algebra
        return [alg.to_sympy(v) for v in self.psatz_domain.ideal]

    @property
    def cone(self) -> SOSCone[Ef]:
        return self.psatz_domain.cone

    @property
    def algebra(self) -> Domain[TExpr[Ef]]:
        return self.psatz_domain.algebra

    @property
    def domain(self) -> Domain[Ef]:
        return self.psatz_domain.domain

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

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PSatz):
            return self.rep == other.rep
        return NotImplemented

    @property
    def is_zero(self) -> bool:
        return self.rep.is_zero

    @property
    def zero(self) -> 'PSatz[Ef]':
        return self.new(self.rep.zero)

    @property
    def one(self) -> 'PSatz[Ef]':
        return self.new(self.rep.one)

    @property
    def expr(self) -> Expr:
        return self.as_expr()

    def as_expr(self, preorder_alias: Optional[List[Expr]]=None, ideal_alias: Optional[List[Expr]]=None,
            evaluate: bool=True) -> Expr:
        return self.rep.as_expr(preorder_alias=preorder_alias, ideal_alias=ideal_alias, evaluate=evaluate)

    @property
    def numer_preorder(self) -> Dict[FrozenSet[int], SOSlist]:
        return {k: SOSlist.new(v) for k, v in self.rep.numer_preorder.items()}

    @property
    def numer_ideal(self) -> Dict[int, Expr]:
        alg = self.algebra
        return {k: alg.to_sympy(v) for k, v in self.rep.numer_ideal.items()}

    @property
    def denom_preorder(self) -> Dict[FrozenSet[int], SOSlist]:
        return {k: SOSlist.new(v) for k, v in self.rep.denom_preorder.items()}

    @property
    def denom_ideal(self) -> Dict[int, Expr]:
        alg = self.algebra
        return {k: alg.to_sympy(v) for k, v in self.rep.denom_ideal.items()}

    @classmethod
    def from_sympy(cls, arg,
        preorder: Optional[Union[List[Expr], Dict[Expr, Expr]]]=None,
        ideal: Optional[Union[List[Expr], Dict[Expr, Expr]]]=None,
        cone: Optional[SOSCone]=None,
        algebra: Optional[Domain[TExpr[Ef]]]=None,
        domain: Optional[Domain[Ef]]=None
    ) -> Optional['PSatz']:
        # construction from sympy expressions
        arg = sympify(arg)
        preorder = preorder or {}
        ideal = ideal or {}
        if not isinstance(preorder, dict):
            preorder = [sympify(_) for _ in preorder]
            preorder = {v: v.as_expr() for v in preorder}
        else:
            preorder = {sympify(k): sympify(v).as_expr() for k, v in preorder.items()}
        if not isinstance(ideal, dict):
            ideal = [sympify(_) for _ in ideal]
            ideal = {v: v.as_expr() for v in ideal}
        else:
            ideal = {sympify(k): sympify(v).as_expr() for k, v in ideal.items()}

        # build the cone and psatz_element from keyword arguments
        if domain is None and algebra is None:
            if cone is None:
                cone = EXRAWSOSCone
            algebra = cone.algebra
            domain = cone.domain
        elif cone is not None:
            raise TypeError("Cannot specify cone when domain or algebra is given.")
        elif algebra is not None:
            domain = algebra.domain
        elif domain is not None:
            if not (domain.is_EX or (HAS_EXRAW and domain.is_EXRAW)):
                fs = arg.free_symbols.union(
                    *[_.free_symbols for _ in preorder.values()],
                    *[_.free_symbols for _ in ideal.values()]    
                )
                fs = sorted(list(fs), key=lambda x: x.name)
                algebra = domain[*fs]
            else:
                algebra = domain
        if cone is None:
            cone = SOSCone(algebra, domain)

        psatz_domain = PSatzDomain(cone, 
            [algebra.from_sympy(v) for v in preorder.values()], 
            [algebra.from_sympy(v) for v in ideal.values()]
        )
        rep = psatz_domain.from_sympy(arg,
            preorder_alias = {k: i for i, k in enumerate(preorder)},
            ideal_alias = {k: i for i, k in enumerate(ideal)}
        )
        if rep is not None:
            return cls.new(rep)
        return None

    def __add__(self, other: object) -> 'PSatz[Ef]':
        if isinstance(other, PSatz):
            return self.new(self.rep + other.rep)
        return self.new(self.rep + other)

    def __sub__(self, other: object) -> 'PSatz[Ef]':
        if isinstance(other, PSatz):
            return self.new(self.rep - other.rep)
        return self.new(self.rep - other)

    def __radd__(self, other: object) -> 'PSatz[Ef]':
        if isinstance(other, PSatz):
            return self.new(other.rep + self.rep)
        return self.new(other + self.rep)

    def __rsub__(self, other: object) -> 'PSatz[Ef]':
        if isinstance(other, PSatz):
            return self.new(other.rep - self.rep)
        return self.new(other - self.rep)

    def __pos__(self) -> 'PSatz[Ef]':
        return self
    
    def __neg__(self) -> 'PSatz[Ef]':
        return self.new(self.rep.__neg__())

    def __mul__(self, other: object) -> 'PSatz[Ef]':
        if isinstance(other, PSatz):
            return self.new(self.rep * other.rep)
        return self.new(self.rep * other)

    def __rmul__(self, other: object) -> 'PSatz[Ef]':
        if isinstance(other, PSatz):
            return self.new(other.rep * self.rep)
        return self.new(other * self.rep)

    def __truediv__(self, other: object) -> 'PSatz[Ef]':
        if isinstance(other, PSatz):
            return self.new(self.rep / other.rep)
        return self.new(self.rep / other)

    def __rtruediv__(self, other: object) -> 'PSatz[Ef]':
        if isinstance(other, PSatz):
            return self.new(other.rep / self.rep)
        return self.new(other / self.rep)

    def __pow__(self, n: int) -> 'PSatz[Ef]':
        return self.new(self.rep**n)

    def inverse(self) -> 'PSatz[Ef]':
        return self.new(self.rep.inverse())

    def mul_sqr(self, numer: Optional[Expr]=None, denom: Optional[Expr]=None, frac: bool=True) -> 'PSatz[Ef]':
        if frac and denom is None and numer is not None:
            numer, denom = fraction(sympify(numer).as_expr().doit().together())
        _numer = self.algebra.from_sympy(numer) if numer is not None else numer
        _denom = self.algebra.from_sympy(denom) if denom is not None else denom
        return self.new(self.rep.mul_sqr(_numer, _denom))

    def marginalize(self, ind: int, pop: bool = False) -> Tuple['PSatz[Ef]', 'PSatz[Ef]', 'PSatz[Ef]', 'PSatz[Ef]']:
        return tuple([self.new(_) for _ in self.rep.marginalize(ind, pop=pop)])

    def join(self, other: 'PSatz[Ef]', ind: int,
            numer: Optional[Expr]=None, denom: Optional[Expr]=None, frac: bool=True) -> 'PSatz[Ef]':
        """        
        Join two PSatzs to eliminate the `ind`-th preorder generator. The `ind`-th
        preorder generator of two PSatzs should imply opposite values.

        Suppose `F = numer/denom`. If the two PSatzs imply:
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
        >>> p1 = PSatz.from_sympy(p1, [a,c,u,x,y,z], [4*a*c-b**2+b*y-z])
        >>> p2 = PSatz.from_sympy(p2, [a,c,v,x,y,z], [4*a*c-b**2+b*y-z])
        >>> p3 = p1.join(p2, 2, a*x**2+b*x+c)
        >>> p3.as_expr().together()
        (a*x**2*y + c*y + x*z + x*(2*a*x + b)**2 + x*(4*a*c - b**2 + b*y - z))/(4*a*x + y)
        >>> p3.as_expr().factor()
        a*x**2 + b*x + c
        """
        if frac and denom is None and numer is not None:
            numer, denom = fraction(sympify(numer).as_expr().doit().together())
        _numer = self.algebra.from_sympy(numer) if numer is not None else numer
        _denom = self.algebra.from_sympy(denom) if denom is not None else denom
        return self.new(self.rep.join(other.rep, ind, _numer, _denom))
