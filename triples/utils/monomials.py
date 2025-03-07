from typing import Union, Optional, Dict, List, Tuple, Callable
from warnings import warn

import numpy as np
import sympy as sp
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polyclasses import DMP
from sympy.combinatorics import Permutation, PermutationGroup, CyclicGroup, SymmetricGroup

try:
    from sympy.polys.matrices.sdm import SDM
    from sympy.polys.matrices import DomainMatrix
except ImportError: # sympy <= 1.7
    SDM = None

try:
    from sympy.external.gmpy import GROUND_TYPES
    _IS_GROUND_TYPES_FLINT = (GROUND_TYPES == 'flint')
except ImportError: # sympy <= 1.8 or no flint installed
    _IS_GROUND_TYPES_FLINT = False

def generate_partitions(nvars: int, degree: int, equal: bool = False) -> List[Tuple[int, ...]]:
    """
    Generate all tuples (a0,a1,...,an) such that n = nvars and sum(ai) <= degree.
    If equal is True, then it requires sum(ai) == degree.
    """
    if nvars == 0:
        return [()] if degree == 0 or (not equal) else []
    def generate_tuples(current_tuple: Tuple[int, ...], current_sum: int, remaining_vars: int) -> List[Tuple[int, ...]]:
        if remaining_vars == 0:
            return [current_tuple]
        else:
            tuples = []
            for i in range(degree - current_sum, -1, -1):
                tuples.extend(generate_tuples(current_tuple + (i,), current_sum + i, remaining_vars - 1))
            return tuples

    monoms = generate_tuples((), 0, nvars) if not equal else generate_tuples((), 0, nvars - 1)
    if equal:
        monoms = [m + (degree - sum(m),) for m in monoms]
    return monoms
    # inv_monoms = list(filter(self.is_standard_monom, inv_monoms))
    # dict_monoms = {t: i for i, t in enumerate(inv_monoms)}
    # return dict_monoms, inv_monoms

def _poly_rep(poly: Union[sp.Poly, DMP]) -> DMP:
    return poly.rep if isinstance(poly, sp.Poly) else poly

class MonomialManager():
    """
    Class to compute polynomial monomials given the symmetry of variables and homogeneity.
    """
    def __init__(self, nvars: int, perm_group = None, is_homogeneous: bool = True) -> None:
        self.nvars = nvars
        self._is_homogeneous = bool(is_homogeneous)
        if isinstance(perm_group, MonomialManager):
            perm_group = perm_group.perm_group
        self._perm_group = perm_group if perm_group is not None else PermutationGroup(Permutation(list(range(nvars))))
        if self._perm_group.degree != nvars:
            raise ValueError(f"The degree of the permutation group ({self._perm_group.degree}) does not match the number of variables ({nvars}).")

        self._monoms = {}

    def __str__(self) -> str:
        return "MonomialManager(nvars=%d, perm_group=%s, is_homogeneous=%s)" % (
            self.nvars, str(self._perm_group).replace('\n', '').replace('  ',''), self._is_homogeneous)

    def __repr__(self) -> str:
        return self.__str__()

    def copy(self) -> 'MonomialManager':
        return MonomialManager(self.nvars, perm_group=self._perm_group, is_homogeneous=self._is_homogeneous)

    def __hash__(self) -> int:
        return hash((self.__class__, self.nvars, self._perm_group, self._is_homogeneous))

    def __eq__(self, other: 'MonomialManager') -> bool:
        if self is other:
            return True
        return self.nvars == other.nvars and self._perm_group == other._perm_group and self._is_homogeneous == other._is_homogeneous

    @property
    def is_homogeneous(self) -> bool:
        return self._is_homogeneous

    @property
    def perm_group(self) -> PermutationGroup:
        return self._perm_group

    @property
    def is_trivial(self) -> bool:
        return self._perm_group.is_trivial

    @classmethod
    def from_perm_group(cls, perm_group: PermutationGroup, is_homogeneous: bool = True) -> 'MonomialManager':
        """
        Create a `MonomialManager` object from a permutation group. The `nvars` argument is inferred from 
        the degree of the group and is therefore not needed.
        """
        return MonomialManager(perm_group.degree, perm_group=perm_group, is_homogeneous=is_homogeneous)

    def _register_monoms(self, degree: int) -> Tuple[Dict[Tuple[int, ...], int], List[Tuple[int, ...]]]:
        """
        Register dict_monoms and inv_monoms for degree if not computed.
        """
        x = self._monoms.get(degree, None)
        if x is None:
            candidates = generate_partitions(self.nvars, degree, equal = self._is_homogeneous)
            monoms = list(filter(self.is_standard_monom, candidates))
            dict_monoms = {t: i for i, t in enumerate(monoms)}
            x = (dict_monoms, monoms)
            self._monoms[degree] = x
        return x

    def dict_monoms(self, degree: int) -> Dict[Tuple[int, ...], int]:
        """
        Return the dictionary of monomials of given degree with indices as values.
        """
        return self._register_monoms(degree)[0]

    def inv_monoms(self, degree: int) -> List[Tuple[int, ...]]:
        """
        Return the list of monomials of given degree. Monomials can be accessed by indices.
        """
        return self._register_monoms(degree)[1]

    def index(self, monom: Tuple[int, ...]) -> Optional[int]:
        """
        Return the index of the monom in the vector representation of the polynomial.
        Note that it is NOT the index of a permutation group.
        """
        degree = sum(monom)
        return self.dict_monoms(degree).get(self.standard_monom(monom))

    def length(self, degree: int) -> int:
        return len(self.inv_monoms(degree))

    def __contains__(self, monom: Tuple[int, ...]) -> bool:
        """
        Return True if a monom is contained in the reduction rule.
        """
        return monom in self.dict_monoms(sum(monom))

    def base(self) -> 'MonomialManager':
        """
        Return a copy with the permutation group set to the trivial group.
        """
        obj = self.copy()
        obj._perm_group = PermutationGroup(Permutation(list(range(self.nvars))))
        return obj

    def order(self) -> int:
        """
        Return the order of the permutation group.
        """
        return self._perm_group.order()

    def permute(self, monom: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """
        Permute the monom according to the permutation group.
        """
        return [tuple(perm(monom)) for perm in self._perm_group.elements]

    def standard_monom(self, monom: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        For monomials in cyclic groups, only one representative is needed.
        For example, a^3b^2c, b^3c^2a, and c^3a^2b are equivalent wrt. CyclicGroup(3),
        and the standard monom (the representative) is (3, 2, 1), standing for a^3b^2c.
        The function returns the standard monom for the input monom. 
        This is chosen to be the lexicographically largest monom among all permutations of the input monom.
        """
        return max(self.permute(monom))

    def is_standard_monom(self, monom: Tuple[int, ...]) -> bool:
        """
        For monoms in cyclic groups, only one representative is needed.
        For example, a^3b^2c, b^3c^2a, and c^3a^2b are equivalent in MonomialCyclic,
        and the standard monom (the representative) is a^3b^2c. The function
        returns True if the input monom is the standard monom.
        """
        return monom == self.standard_monom(monom)

    def permute_vec(self, vec: sp.Matrix, degree: int) -> sp.Matrix:
        """
        Permute the vector representation of a polynomial according to the permutation group.
        It returns a matrix with all permutations of the monoms.

        Note that the vector representation is on the base of MonomialManager class, which does
        not reduce any monomials by symmetry.
        """
        base = self.base()
        dict_monoms = base.dict_monoms(degree)
        inv_monoms = base.inv_monoms(degree)
        f = lambda i: dict_monoms[i]
        # get all permutations of all monoms first
        all_vecs = [[0] * vec.shape[0] for _ in range(self.order())]
        for v, m in zip(vec, inv_monoms):
            for ind, j in enumerate(map(f, self.permute(m))):
                # write the value to the corresponding column
                all_vecs[ind][j] = v
        return sp.Matrix(all_vecs).T
 

    # def _standard_monom(self, monom: Tuple[int, ...]) -> Tuple[int, ...]:
    #     warn("_standard_monom is deprecated. Use standard_monom instead.", DeprecationWarning, stacklevel=2)
    #     return self.standard_monom(monom)

    def _assert_equal_nvars(self, t: Union[int, Tuple[int, ...]]) -> bool:
        if (t if isinstance(t, int) else len(t)) != self.nvars:
            raise ValueError("Number of variables does not match. Expected %d but received %d." % (self.nvars, len(t)))
            return False
        return True

    def _arraylize_list(self, poly: Union[sp.Poly, DMP], expand_cyc: bool = False) -> List:
        rep = _poly_rep(poly)
        self._assert_equal_nvars(rep.lev + 1)
        degree = rep.total_degree()
        dict_monoms = self.dict_monoms(degree)
        array = [rep.dom.zero] * len(dict_monoms)
        if not expand_cyc:
            for monom, coeff in rep.terms():
                v = dict_monoms.get(monom)
                if v is not None:
                    array[v] = coeff
        else:
            for monom, coeff in rep.terms():
                for monom2 in self.permute(monom):
                    v = dict_monoms.get(monom2)
                    if v is not None:
                        array[v] += coeff
        return array

    def arraylize_np(self, poly: Union[sp.Poly, DMP], expand_cyc: bool = False) -> np.ndarray:
        """
        Return the vector representation of the polynomial in numpy array.
        """
        vec = self._arraylize_list(poly, expand_cyc = expand_cyc)
        rep = _poly_rep(poly)
        if not (rep.dom is ZZ or rep.dom is QQ):
            to_sympy = rep.dom.to_sympy
            vec = [to_sympy(v) for v in vec]
        elif _IS_GROUND_TYPES_FLINT:
            if rep.dom is QQ:
                vec = [int(v.numerator) / int(v.denominator) for v in vec]
            else:
                vec = [int(v) for v in vec]
        return np.array(vec).astype(np.float64)

    def arraylize_sp(self, poly: Union[sp.Poly, DMP], expand_cyc: bool = False) -> sp.Matrix:
        """
        Return the vector representation of the polynomial in sympy matrix (column vector).
        """
        vec = self._arraylize_list(poly, expand_cyc = expand_cyc)
        rep = _poly_rep(poly)
        if SDM is not None:
            sdm = dict((i, {0: v}) for i, v in enumerate(vec) if v)
            return sp.Matrix._fromrep(DomainMatrix.from_rep(SDM(sdm, (len(vec), 1), rep.dom)))
        else: # sympy <= 1.7
            to_sympy = rep.dom.to_sympy
            vec = [to_sympy(v) for v in vec]
            return sp.Matrix(vec)

    def invarraylize(self, array: Union[List, np.ndarray, sp.Matrix], gens: List[sp.Symbol], degree: int) -> sp.Poly:
        """
        Reverse the arraylize_np function to get the polynomial from its vector representation.
        """
        self._assert_equal_nvars(gens)
        inv_monoms = self.inv_monoms(degree)
        terms_dict = {}
        for coeff, monom in zip(array, inv_monoms):
            for monom2 in self.permute(monom):
                terms_dict[monom2] = coeff
        return sp.Poly(terms_dict, gens)

    def cyclic_sum(self, expr: sp.Expr, gens: List[sp.Symbol]) -> sp.Expr:
        """
        Sum up a given expression according to the permutation group.

        This function should be hijacked by the CyclicExpr module.
        """
        s = []
        expr0 = expr
        for perm in self._perm_group.elements:
            perm_gens = perm(gens)
            expr = expr0.xreplace(dict(zip(gens, perm_gens)))
            s.append(expr)
        return sp.Add(*s)


def _parse_options(nvars, **options) -> MonomialManager:
    """Fetch the corresponding MonomialManager object given options."""
    if nvars < 0:
        raise ValueError("Number of variables should be non-negative.")

    hom = options.pop('hom', True)
    cyc = options.pop('cyc', False)
    sym = options.pop('sym', False)
    symmetry = options.pop('symmetry', None)

    if len(options) > 0:
        raise KeyError(f"Unknown options: {','.join(options.keys())}."
                       ' Only "hom", "cyc", "sym", and "symmetry" are allowed.')

    if symmetry is not None and (cyc or sym):
        raise ValueError("Cannot specify both symmetry and cyc or sym.")  
    elif cyc:
        if sym:
            raise ValueError("Cannot specify both cyc and sym.")
        symmetry = CyclicGroup(nvars)
    elif sym:
        symmetry = SymmetricGroup(nvars)

    if isinstance(symmetry, MonomialManager):
        return symmetry
    if symmetry is None:
        return MonomialManager(nvars, is_homogeneous = hom)
    if isinstance(symmetry, PermutationGroup):
        return MonomialManager.from_perm_group(symmetry, is_homogeneous = hom)
    
    raise ValueError(f"Invalid symmetry type {type(symmetry)}. Expected MonomialManager or PermutationGroup.")


def arraylize_np(poly: Union[sp.Poly, DMP], expand_cyc: bool = False, **options) -> np.ndarray:
    """
    Convert a sympy polynomial to a numpy vector of coefficients.

    Parameters
    -----------
    poly: Poly
        The sympy polynomial.

    expand_cyc: bool
        Whether to compute the cyclic sum of the polynomial given
        the symmetry group.

    Keyword Arguments
    -----------------
    hom: bool
        If True, only homogeneous monomials are considered.
        Default is True.
    cyc: bool
        If True, monomials are reduced by a cyclic group.
        Default is False.
    sym: bool
        If True, monomials are reduced by a symmetric group.
        Default is False.
    symmetry: PermutationGroup
        Sympy permutation group object for the monomials. This specifies the symmetry
        beyond cyclic and symmetric groups.

    Returns
    ---------
    vec: Matrix
        Sympy matrix (vector) that stores the coefficients of the polynomial.

    Examples
    ---------
    >>> from sympy.abc import a, b, c
    >>> print(arraylize_np(((a-b)**2+(b-c)**2+(c-a)**2).as_poly(a,b,c)))
    [ 2. -2. -2.  2. -2.  2.]

    >>> print(arraylize_np(((a**2+b**2+c**2)**2-3*(a**3*b+b**3*c+c**3*a)).as_poly(a,b,c), cyc = True))
    [ 1. -3.  0.  2.  0.]

    >>> print(arraylize_np((a*b*c).as_poly(a,b,c), sym = True))
    [0. 0. 1.]

    >>> print(arraylize_np((b**2*c + c**2*a).as_poly(a,b,c), expand_cyc = True, cyc = True))
    [0. 2. 0. 0.]

    See Also
    ----------
    arraylize_sp, invarraylize, generate_monoms
    """
    nvars = (poly.rep if isinstance(poly, sp.Poly) else poly).lev + 1
    option = _parse_options(nvars, **options)
    return option.arraylize_np(poly, expand_cyc = expand_cyc)

def arraylize_sp(poly: Union[sp.Poly, DMP], expand_cyc: bool = False, **options) -> sp.Matrix:
    """
    Convert a sympy polynomial to a sympy vector of coefficients.

    Parameters
    -----------
    poly: Poly
        The sympy polynomial.

    expand_cyc: bool
        Whether to compute the cyclic sum of the polynomial given
        the symmetry group.

    Keyword Arguments
    -----------------
    hom: bool
        If True, only homogeneous monomials are considered.
        Default is True.
    cyc: bool
        If True, monomials are reduced by a cyclic group.
        Default is False.
    sym: bool
        If True, monomials are reduced by a symmetric group.
        Default is False.
    symmetry: PermutationGroup
        Sympy permutation group object for the monomials. This specifies the symmetry
        beyond cyclic and symmetric groups.

    Returns
    ---------
    vec: Matrix
        Sympy matrix (vector) that stores the coefficients of the polynomial.

    Examples
    ---------
    >>> from sympy.abc import a, b, c
    >>> print(arraylize_sp(((a-b)**2+(b-c)**2+(c-a)**2).as_poly(a,b,c)))
    Matrix([[2], [-2], [-2], [2], [-2], [2]])

    >>> print(arraylize_sp(((a**2+b**2+c**2)**2-3*(a**3*b+b**3*c+c**3*a)).as_poly(a,b,c), cyc = True))
    Matrix([[1], [-3], [0], [2], [0]])

    >>> print(arraylize_sp((a*b*c).as_poly(a,b,c), sym = True))
    Matrix([[0], [0], [1]])

    >>> print(arraylize_sp((b**2*c + c**2*a).as_poly(a,b,c), expand_cyc = True, cyc = True))
    Matrix([[0], [2], [0], [0]])

    See Also
    ----------
    arraylize_np, invarraylize, generate_monoms
    """
    nvars = (poly.rep if isinstance(poly, sp.Poly) else poly).lev + 1
    option = _parse_options(nvars, **options)
    return option.arraylize_sp(poly, expand_cyc = expand_cyc)

def invarraylize(array: Union[List, np.ndarray, sp.Matrix], gens: List[sp.Symbol], degree: int, **options) -> sp.Poly:
    """
    Convert a vector representation of polynomial back to the sympy polynomial.

    Parameters
    -----------
    array: List or ndarray or Matrix
        1D iterable object representing the vector of coefficients.
    gens: List[Symbol]
        A list of symbols as the generators of the polynomial.
    degree: int
        The total degree of the polynomial.

    Keyword Arguments
    -----------------
    hom: bool
        If True, only homogeneous monomials are considered.
        Default is True.
    cyc: bool
        If True, monomials are reduced by a cyclic group.
        Default is False.
    sym: bool
        If True, monomials are reduced by a symmetric group.
        Default is False.
    symmetry: PermutationGroup
        Sympy permutation group object for the monomials. This specifies the symmetry
        beyond cyclic and symmetric groups.
        

    Returns
    ---------
    poly: Poly
        Sympy polynomial.

    Examples
    ---------
    >>> from sympy.abc import a, b, c, x, y, z
    >>> v1 = arraylize_sp(((a**2+b**2+c**2)**2-3*(a**3*b+b**3*c+c**3*a)).as_poly(a,b,c), cyc = True)
    >>> invarraylize(v1, (x, y, z), 3, cyc = True)
    Poly(x**3 - 3*x**2*y + 2*x*y*z - 3*x*z**2 + y**3 - 3*y**2*z + z**3, x, y, z, domain='ZZ')

    >>> invarraylize([x,y], (a,b,c), 2, sym = True)
    Poly(x*a**2 + y*a*b + y*a*c + x*b**2 + y*b*c + x*c**2, a, b, c, domain='ZZ[x,y]')

    See Also
    ----------
    arraylize_np, arraylize_sp, generate_monoms
    """
    option = _parse_options(len(gens), **options)
    return option.invarraylize(array, gens, degree)

def generate_monoms(nvars: int, degree: int, **options) -> Tuple[Dict[Tuple[int, ...], int], List[Tuple[int, ...]]]:
    """
    Generate monomials of given number of variables and degree.

    Returns a dictionary of monomials with indices as values and a list of monomials.

    Parameters
    ----------
    nvars: int
        Number of variables.
    degree: int
        Degree of monomials.

    Keyword Arguments
    -----------------
    hom: bool
        If True, only homogeneous monomials are generated.
        Default is True.
    cyc: bool
        If True, the monomials are generated with cyclic symmetry.
        Default is False.
    sym: bool
        If True, the monomials are generated with symmetric symmetry.
        Default is False.
    symmetry: PermutationGroup
        Sympy permutation group object for the monomials. This specifies the symmetry
        beyond cyclic and symmetric groups.

    Returns
    ----------
    dict_monoms : Dict[Tuple[int, ...], int]
        Dictionary of monomials with indices as values.
    inv_monoms : List[Tuple[int, ...]]
        List of monomials.

    Examples
    ----------
    >>> generate_monoms(3, 3, cyc = True) # doctest: +NORMALIZE_WHITESPACE
    ({(3, 0, 0): 0, (2, 1, 0): 1, (2, 0, 1): 2, (1, 1, 1): 3},
     [(3, 0, 0), (2, 1, 0), (2, 0, 1), (1, 1, 1)])

    >>> from sympy.combinatorics import AlternatingGroup
    >>> generate_monoms(4, 3, symmetry=AlternatingGroup(4)) # doctest: +NORMALIZE_WHITESPACE
    ({(3, 0, 0, 0): 0, (2, 1, 0, 0): 1, (1, 1, 1, 0): 2},
     [(3, 0, 0, 0), (2, 1, 0, 0), (1, 1, 1, 0)])

    >>> generate_monoms(3, 2, hom = False, sym = True) # doctest: +NORMALIZE_WHITESPACE
    ({(2, 0, 0): 0, (1, 1, 0): 1, (1, 0, 0): 2, (0, 0, 0): 3},
     [(2, 0, 0), (1, 1, 0), (1, 0, 0), (0, 0, 0)])
    """
    option = _parse_options(nvars, **options)
    return option._register_monoms(degree)


def generate_expr(nvars: int, degree: int, **options) -> Tuple[Dict[Tuple[int, ...], int], List[Tuple[int, ...]]]:
    warn("generate_expr is deprecated. Use generate_monoms instead.", DeprecationWarning, stacklevel=2)
    return generate_monoms(nvars, degree, **options)