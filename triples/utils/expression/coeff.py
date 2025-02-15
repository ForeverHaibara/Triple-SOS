from typing import Union, Dict, List, Tuple, Optional

import sympy as sp
from sympy.combinatorics import Permutation, PermutationGroup
from sympy.combinatorics.named_groups import CyclicGroup, SymmetricGroup, AlternatingGroup
from sympy.polys.polytools import Poly
from sympy.utilities.iterables import iterable

class Coeff():
    """
    A standard class for representing a polynomial with coefficients.
    """
    def __new__(cls, *args, **kwargs):
        # if it is already a Coeff object, return it.
        if len(args) == 1 and isinstance(args[0], cls):
            return args[0]
        return super().__new__(cls)

    def __init__(self, coeffs: Union[Poly, Dict], is_rational: bool = True):
        if hasattr(self, 'coeffs'):
            # if the object is already initialized,
            # (this is the case when called from __new__ by Coeff(Coeff(...)) )
            # then return without reinitializing.
            return
        if isinstance(coeffs, Poly):
            self._nvars = len(coeffs.gens)
            poly = coeffs
            coeffs = {}
            for monom, coeff in poly.terms():
                if not isinstance(coeff, sp.Rational): #isinstance(coeff, sp.Float): # and degree > 4
                    # if isinstance(coeff, sp.Float):
                    #     coeff = rationalize(coeff, reliable = True)
                    # else:
                    is_rational = False
                    # coeff = coeff.as_numer_denom()
                coeffs[monom] = coeff
        else:
            self._nvars = len(next(iter(coeffs.keys()))) if len(coeffs) > 0 else 0
            
        self.coeffs = coeffs
        self.is_rational = is_rational

    @property
    def nvars(self) -> int:
        return self._nvars

    def __call__(self, *x) -> sp.Expr:
        """
        Coeff((i,j,k)) -> returns the coefficient of a^i * b^j * c^k.
        """
        if len(x) == 1 and iterable(x[0]):
            # x is ((a,b,c), )
            x = x[0]
        if not isinstance(x, tuple):
            x = tuple(x)
        return self.coeffs.get(x, sp.S(0))

    def __len__(self) -> int:
        """
        Number of coefficients. Sometimes the zero coefficients are not included.
        """
        return len(self.coeffs)

    def is_cyclic(self, perm_group: Optional[Union[Permutation, List[Permutation], PermutationGroup]] = None) -> bool:
        """
        Check whether the coefficients are cyclic with respect to a permutation group.
        If not specified, it assumes to be the cyclic group.

        Examples
        ---------
        >>> coeff = Coeff((a**2*b+b**2*c+c**2*d+d**2*a).as_poly(a, b, c, d))
        >>> coeff.is_cyclic()
        True
        """
        if perm_group is None:
            perm_group = CyclicGroup(self.nvars)

        if self.nvars == 1:
            return True
        
        perms = []
        if isinstance(perm_group, PermutationGroup):
            # checking the generators is enough
            perms = perm_group.generators
        elif isinstance(perm_group, Permutation):
            perms = [perm_group]
        else:
            perms = perm_group

        for perm in perms:
            if perm.size != self.nvars:
                return False
            for k, v in self.coeffs.items():
                if self(perm(k)) != v:
                    return False
        return True

    def is_symmetric(self, perm_group: Optional[Union[Permutation, List[Permutation], PermutationGroup]] = None) -> bool:
        """
        Check whether the coefficients are symmetric with respect to a permutation group.
        If not specified, it assumes to be the symmetric group. When the perm_group
        argument is given, it acts the same as `is_cyclic()`.

        Examples
        ---------
        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> coeff = Coeff((a**2+b**2+c*(a+b)+4*c**2).as_poly(a, b, c))
        >>> coeff.is_symmetric(PermutationGroup(Permutation((1,0,2))))
        True
        >>> coeff.is_symmetric()
        False
        """
        if perm_group is None:
            perm_group = SymmetricGroup(self.nvars)
        return self.is_cyclic(perm_group)

    def reflect(self) -> 'Coeff':
        """
        Reflect the coefficients of a, b, c with respect to a,b.
        Returns a deepcopy.
        """
        if self.is_zero:
            return Coeff({}, is_rational = True)

        assert self.nvars == 3, "The number of variables must be 3."

        reflected_coeffs = dict([((j,i,k), v) for (i,j,k), v in self.coeffs.items()])
        new_coeff = Coeff(reflected_coeffs, is_rational = self.is_rational)
        return new_coeff

    def clear_zero(self) -> None:
        """
        Clear the coefficients that are zero.
        """
        self.coeffs = {k:v for k,v in self.coeffs.items() if v != 0}

    def as_poly(self, *args) -> Poly:
        """
        Return the polynomial of given variables. If args is not given, it uses a-z.
        """
        if len(args) == 0:
            if self.is_zero:
                args = sp.symbols('a')
            else:
                args = sp.symbols(f'a:{chr(96+self.nvars)}')
        elif len(args) == 1 and iterable(args[0]):
            args = args[0]
        return Poly.from_dict(self.coeffs, gens = args)

    def is_homogeneous(self) -> bool:
        """
        Whether the polynomial is homogeneous.
        """
        if self.is_zero:
            return True
        degree = self.degree()
        return all(sum(k) == degree for k in self.coeffs)

    def degree(self) -> int:
        """
        Return the degree of the polynomial. Only works for homogeneous polynomials.
        Please use `total_degree()` for non-homogeneous polynomials.
        """
        if len(self.coeffs) == 0:
            return 0
        for k in self.coeffs:
            return sum(k)

    def total_degree(self) -> int:
        """
        Return the total degree of the polynomial.
        """
        return max(sum(k) for k in self.coeffs)

    @property
    def is_zero(self) -> bool:
        """
        Whether the polynomial is zero.
        """
        return len(self.coeffs) == 0

    def poly111(self) -> sp.Expr:
        """
        Evalutate the polynomial at (1,1,...,1).
        """
        return sum(self.coeffs.values())

    def items(self):
        return self.coeffs.items()

    def __operator__(self, other, operator) -> 'Coeff':
        new_coeffs = self.coeffs.copy()
        for k, v2 in other.items():
            v1 = self(k)
            v3 = operator(v1, v2)
            if v3 == 0 and v1 != 0:
                del new_coeffs[k]
            elif v3 != 0:
                new_coeffs[k] = v3
        new_coeffs = dict(sorted(new_coeffs.items(), reverse=True))
        other_rational = (not isinstance(other, Coeff)) or other.is_rational
        is_rational = self.is_rational and other_rational
        return Coeff(new_coeffs, is_rational = is_rational)

    def __add__(self, other) -> 'Coeff':
        return self.__operator__(other, lambda x, y: x + y)

    def __sub__(self, other) -> 'Coeff':
        return self.__operator__(other, lambda x, y: x - y)

    # def __mul__(self, other) -> 'Coeff':
    #     return self.__operator__(other, lambda x, y: x * y)

    # def __truediv__(self, other) -> 'Coeff':
    #     return self.__operator__(other, lambda x, y: x / y)

    def __pow__(self, other) -> 'Coeff':
        return self.__operator__(other, lambda x, y: x ** y)

    def cancel_abc(self) -> Tuple[List[int], 'Coeff']:
        """
        Assume poly = a^i*b^j*c^k * poly2.
        Return ((i,j,k), Coeff(poly2)).
        """
        if self.is_zero:
            return ((0,) * self.nvars, self)
        all_monoms = list(self.coeffs.keys())
        d = self.degree() + 1
        common = [d] * self.nvars
        for monom in all_monoms:
            common = [min(i, j) for i, j in zip(common, monom)]
            if all(_ == 0 for _ in common):
                return ((0,) * self.nvars, self)

        common = tuple(common)
        new_coeff = Coeff({tuple([i - j for i, j in zip(monom, common)]): _ for monom, _ in self.coeffs.items() if _ != 0})
        new_coeff.is_rational = self.is_rational
        return common, new_coeff


    def cancel_k(self) -> Tuple[int, 'Coeff']:
        """
        Assume poly = Sum_{uvw}(x_{uvw} * a^{d*u} * b^{d*v} * c^{d*w}).
        Write poly2 = Sum_{uvw}(x_{uvw} * a^{u} * b^{v} * c^{w}).
        Return (d, Coeff(poly2))
        """
        if self.is_zero:
            return (1, self)
        all_monoms = list(self.coeffs.keys())
        d = 0
        for monom in all_monoms:
            for u in monom:
                d = sp.gcd(d, u)
                if d == 1:
                    return (1, self)

        d = int(d)
        if d == 0:
            return 0, self

        new_coeff = Coeff({tuple([i//d for i in monom]): _ for monom, _ in self.coeffs.items() if _ != 0})
        new_coeff.is_rational = self.is_rational
        return d, new_coeff



def identify_symmetry(poly: sp.Poly, homogenizer: Optional[sp.Symbol] = None) -> PermutationGroup:
    """
    Identify the symmetry group of the polynomial heuristically.
    It only identifies very simple groups like complete symmetric and cyclic groups.
    TODO: Implement an algorithm to identify all symmetric groups.

    Reference
    ----------
    [1] https://cs.stackexchange.com/questions/64335/how-to-find-the-symmetry-group-of-a-polynomial
    """
    coeff = Coeff(poly)
    nvars = len(poly.gens)
    if coeff.is_symmetric():
        return SymmetricGroup(nvars)
    if coeff.is_cyclic():
        return CyclicGroup(nvars)
    if nvars > 3:
        alt = AlternatingGroup(nvars)
        if coeff.is_symmetric(alt):
            return alt

    if homogenizer is not None and nvars > 2:
        # check the symmetry of the polynomial before homogenization
        a = list(range(1, nvars - 1))
        a.append(0)
        a.append(nvars - 1)
        gen1 = Permutation(a)
        a = list(range(nvars))
        a[0], a[1] = a[1], a[0]
        gen2 = Permutation(a)
        G = PermutationGroup([gen1, gen2])
        if coeff.is_symmetric(G):
            return G

        G = PermutationGroup([gen1])
        if coeff.is_symmetric(G):
            return G

    return PermutationGroup(Permutation(list(range(nvars))))