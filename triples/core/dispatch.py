"""
This module contains utility functions for dispatching
based on the type of instances in `InequalityProblem`.

To control the behaviour of `InequalityProblem` on new types, you can either:
1. Implement the default behaviour for new types.
2. Register / override the methods for @singledispatch methods.
3. Inherits from `InequalityProblem` and override its methods.
"""
from functools import singledispatch
from typing import (
    Dict, List, Tuple, Set, Optional, Union, Iterable, Callable,
    Any, TypeVar, Generic
)

from sympy import (
    Basic, Expr, Symbol, Dummy, Poly, Integer, Rational, Function, Add, Mul, Pow,
    sympify, signsimp, fraction
)
from sympy import __version__ as SYMPY_VERSION
from sympy.external.importtools import version_tuple
from sympy.combinatorics import Permutation
from sympy.polys.rings import PolyElement
from sympy.polys.fields import FracElement
from sympy.polys.domains.domain import DomainElement
from sympy.polys.polyerrors import BasePolynomialError


# fix the bug in sqf_list before 1.13.0
# https://github.com/sympy/sympy/pull/26182
SYMPY_VERSION_TUPLE = tuple(version_tuple(SYMPY_VERSION))
if SYMPY_VERSION_TUPLE >= (1, 13):
    _sqf_list = lambda p: p.sqf_list()
else:
    _sqf_list = lambda p: p.factor_list() # it would be slower, but correct

if SYMPY_VERSION_TUPLE >= (1, 14):
    _polyelement_init = PolyElement
    _fracelement_init = FracElement
else:
    def _polyelement_init(ring, init):
        return ring.zero.new(init)
    def _fracelement_init(field, numer, denom):
        obj = object.__new__(FracElement)
        obj.field, obj.numer, obj.denom = field, numer, denom
        return obj

from ..utils.expressions.exraw import HAS_EXRAW

T = TypeVar('T')

@singledispatch
def _dtype_free_symbols(x: T) -> Set[Symbol]:
    return x.free_symbols

@singledispatch
def _dtype_gens(x: T) -> Tuple[Symbol, ...]:
    return x.gens

@singledispatch
def _dtype_is_zero(x: T) -> Optional[bool]:
    return x.is_zero

@singledispatch
def _dtype_convert(x: T, y: Any) -> T:
    return x.convert(y)

@singledispatch
def _dtype_homogenize(x: T, s: Symbol) -> T:
    return x.homogenize(s)

@singledispatch
def _dtype_is_homogeneous(x: T) -> Optional[bool]:
    return x.is_homogeneous

@singledispatch
def _dtype_sqf_list(x: T) -> Tuple[Expr, List[Tuple[T, int]]]:
    return x.sqf_list()

@singledispatch
def _dtype_make_reorder_func(x: T, gens: Tuple[Symbol, ...]) -> Callable[[Permutation], T]:
    """Return a callable `f` such that `f(perm) == x.xreplace(dict(zip(gens, perm(gens))))`."""
    return lambda perm: x.xreplace(dict(zip(gens, perm(gens))))


###############################################################
#                      Implementation
###############################################################

@_dtype_convert.register(Expr)
def _expr_convert(x: Expr, y: Any) -> Expr:
    return sympify(y).as_expr()

@_dtype_gens.register(Expr)
def _expr_gens(x: Expr) -> Tuple[Symbol, ...]:
    return ()

@_dtype_homogenize.register(Expr)
def _expr_homogenize(x: Expr, s: Symbol) -> Expr:
    z = x.xreplace({k: k/s for k in x.free_symbols})

    def extract_mul(x: Expr) -> Tuple[int, Expr]:
        """Return (d, z) such that x = s**d * z"""
        if x.is_Mul or x.is_Pow:
            xargs = Mul.make_args(x)
            power = 0
            other_args = []
            for a in xargs:
                if a == s:
                    power += 1
                elif a.is_Pow and a.base == s:
                    power += a.exp
                else:
                    other_args.append(a)
            return (power, Mul(*other_args))
        return (0, x)

    def extract(x: Add) -> Add:
        """Collect all terms of an Add expression by the power of s."""
        args = Add.make_args(x)
        ex = [extract_mul(a) for a in args]
        dmin = min([_[0] for _ in ex], default=0)
        return s**dmin * Add(*[s**(d - dmin)*v for d, v in ex])

    z = z.replace(lambda x: x.is_Add, extract)

    # e.g., s**2*a + s**2*b + s**2*c -> s**2*(a + b + c) -> a + b + c since s == 1
    d, z = extract_mul(z)
    return z

@_dtype_is_homogeneous.register(Expr)
def _expr_is_homogeneous(x: Expr) -> Optional[bool]:
    # use real=True, positive=True for squareroots, e.g. sqrt(a*b) -> sqrt(a*b)/s
    s = Dummy("1", real=True, positive=True)
    z = _expr_homogenize(x, s)
    return not z.has(s)

@_dtype_sqf_list.register(Expr)
def _expr_sqf_list(x: Expr) -> Tuple[Expr, List[Tuple[Expr, int]]]:
    if x.is_Mul:
        return (Integer(1), [(a, 1) if not (a.is_Pow and a.exp.is_Rational)
            else (a.base**(Integer(1)/a.exp.q), int(a.exp.p)) for a in x.args])
    if x.is_Pow and x.exp.is_Rational:
        return (Integer(1), [(x.base**(Integer(1)/x.exp.q), int(x.exp.p))])
    return (Integer(1), [(x, 1)])

@_dtype_make_reorder_func.register(Expr)
def _expr_make_reorder_func(x: Expr, gens: Tuple[Symbol, ...]) -> Callable[[Permutation], T]:
    return lambda perm: signsimp(x.xreplace(dict(zip(gens, perm(gens)))))



@_dtype_convert.register(Poly)
def _poly_convert(x: Poly, y: Any) -> Poly:
    try:
        # try to unify the domain if possible
        return Poly(y, x.gens, domain=x.domain)
    except BasePolynomialError: # CoercionFailed
        pass
    return Poly(y, x.gens)

@_dtype_sqf_list.register(Poly)
def _poly_sqf_list(x: Poly) -> Tuple[Expr, List[Tuple[Poly, int]]]:
    return _sqf_list(x)

@_dtype_make_reorder_func.register(Poly)
def _poly_make_reorder_func(x: Poly, gens: Tuple[Symbol, ...]) -> Callable[[Permutation], T]:
    if x.gens == gens:
        return lambda perm: Poly.new(x.reorder(*perm.__invert__()(gens)).rep, *gens)
    return lambda perm: Poly(x.as_expr().xreplace(dict(zip(gens, perm(gens)))), *gens)



@_dtype_gens.register(DomainElement)
def _domainelement_gens(x: DomainElement) -> Tuple[Symbol, ...]:
    return x.parent().gens

@_dtype_is_zero.register(DomainElement)
def _domainelement_is_zero(x: DomainElement) -> Optional[bool]:
    return x.parent().zero == x

@_dtype_convert.register(DomainElement)
def _domainelement_convert(x: DomainElement, y: Any) -> DomainElement:
    return x.parent()(y)


@_dtype_free_symbols.register(PolyElement)
def _polyelement_free_symbols(x: PolyElement) -> Set[Symbol]:
    symbols = set([g for g, d in zip(x.ring.gens, x.degrees()) if d > 0])
    domain = x.ring.domain
    if domain.is_Composite:
        for gen in domain.symbols:
            symbols |= gen.free_symbols
    elif domain.is_EX:
        for coeff in x.coeffs():
            symbols |= coeff.ex.free_symbols
    elif HAS_EXRAW and domain.is_EXRAW:
        for coeff in x.coeffs():
            symbols |= coeff.free_symbols
    return symbols


@_dtype_homogenize.register(PolyElement)
def _polyelement_homogenize(x: PolyElement, s: Symbol) -> PolyElement:
    """Homogenize a polynomial with respect to a symbol."""
    xterms = list(x.items())
    d = max([sum(t) for t, v in xterms], default=0)
    terms = [(t + (d - sum(t),), v) for t, v in xterms]
    ring = x.ring.__class__(x.ring.symbols + (s,), x.ring.domain, x.ring.order)
    return _polyelement_init(ring, dict(terms))

@_dtype_is_homogeneous.register(PolyElement)
def _polyelement_is_homogeneous(x: PolyElement) -> bool:
    """Check if a polynomial is homogeneous with respect to a symbol."""
    if x.is_zero: return True
    monoms = list(x.monoms())
    d = sum(monoms[0])
    return all(sum(m) == d for m in monoms)



@_dtype_free_symbols.register(FracElement)
def _fracelement_free_symbols(x: FracElement) -> Set[Symbol]:
    return _polyelement_free_symbols(x.numer) | _polyelement_free_symbols(x.denom)

@_dtype_is_zero.register(FracElement)
def _fracelement_is_zero(x: FracElement) -> bool:
    return x.numer.is_zero and (not x.denom.is_zero)

@_dtype_is_homogeneous.register(FracElement)
def _fracelement_is_homogeneous(x: FracElement) -> bool:
    if x.numer.is_zero: return True
    return _polyelement_is_homogeneous(x.numer) and _polyelement_is_homogeneous(x.denom)

@_dtype_homogenize.register(FracElement)
def _fracelement_homogenize(x: FracElement, s: Symbol) -> FracElement:
    numer, denom = _polyelement_homogenize(x.numer, s), _polyelement_homogenize(x.denom, s)
    return _fracelement_init(numer.ring.to_field(), *numer.cancel(denom))
