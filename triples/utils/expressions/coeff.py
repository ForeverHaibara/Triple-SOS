from typing import Union, Dict, List, Tuple, Optional, Iterator

from sympy import Poly, Expr, Symbol, Rational
from sympy.combinatorics import Permutation, PermutationGroup
from sympy.polys.rings import PolyElement
from sympy.polys.polyclasses import DMP
from sympy.utilities.iterables import iterable

from .exraw import EXRAW
from .cyclic import CyclicSum, CyclicProduct, SymmetricSum, SymmetricProduct
from ..monomials import verify_symmetry

class Coeff():
    """
    Wrapper of sympy PolyElement.
    """
    rep: PolyElement

    def __init__(self, arg, is_rational: bool = True, field = True):
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
            rep_dom = arg.domain[arg.gens]
            self.rep = rep_dom.one.ring.from_dict(arg.rep.to_dict())
        elif isinstance(arg, PolyElement):
            self.rep = arg

    @classmethod
    def new(cls, rep: PolyElement) -> 'Coeff':
        obj = super().__new__(cls)
        obj.rep = rep
        return obj

    def from_dict(self, rep: dict) -> 'Coeff':
        return self.new(self.ring.from_dict(rep))

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
    def domain(self):
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
        dmp = DMP.from_dict(dict(self.rep), len(self.gens)-1, self.domain)
        return Poly.new(dmp, *args)

    def set_domain(self, domain) -> 'Coeff':
        ring = domain[self.gens]
        return self.set_ring(ring)

    def set_ring(self, ring) -> 'Coeff':
        if ring == self.ring:
            return self
        return self.new(self.rep.set_ring(ring))

    def __add__(self, other) -> 'Coeff':
        if isinstance(other, Poly):
            return Coeff(self.as_poly() - other)
        if isinstance(other, Coeff):
            return self.new(self.rep + other.rep)
        return NotImplemented

    def __sub__(self, other) -> 'Coeff':
        if isinstance(other, Poly):
            return Coeff(self.as_poly() - other)
        if isinstance(other, Coeff):
            return self.new(self.rep - other.rep)
        return NotImplemented

    def __radd__(self, other) -> 'Coeff':
        if isinstance(other, Poly):
            return Coeff(other - self.as_poly())
        if isinstance(other, Coeff):
            return self.new(other.rep + self.rep)
        return NotImplemented

    def __rsub__(self, other) -> 'Coeff':
        if isinstance(other, Poly):
            return Coeff(other - self.as_poly())
        if isinstance(other, Coeff):
            return self.new(other.rep - self.rep)
        return NotImplemented

    def __pos__(self) -> 'Coeff':
        return self

    def __neg__(self) -> 'Coeff':
        return self.new(-self.rep)

    def __mul__(self, other) -> 'Coeff':
        if isinstance(other, (int, Rational)):
            return self.new(self.rep * self.domain.convert(other))
        if self.domain.of_type(other):
            return self.new(self.rep * other)
        if isinstance(other, (Poly, Expr, float)):
            return Coeff(self.as_poly() * other)
        if isinstance(other, Coeff):
            return self.new(self.rep * other.rep)
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
        return self.rep.get(x, self.domain.zero)

    def poly111(self) -> Expr:
        z = self.domain.zero
        for c in self.coeffs():
            z = z + c
        return z

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
        if self.is_zero:
            return self.from_dict({})
        if self.nvars == 1:
            return self.from_dict(dict(self.rep))
        refl = lambda z: tuple((z[1], z[0],) + z[2:])
        reflected = dict([(refl(k), v) for k, v in self.items()])
        new_coeff = self.from_dict(reflected)
        return new_coeff

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
