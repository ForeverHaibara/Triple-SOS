"""
This module provides functions to manipulate the group symmetry of a polynomial,
and also utilities to compute monomial representations under the group symmetry.
"""
from collections import defaultdict
from typing import (Dict, List, Tuple, Iterable, Callable,
    Union, Optional, Any, overload, TypeVar
)
import numpy as np
from sympy import Poly, Expr, Symbol, Add, ZZ, QQ, factorial, prod
from sympy.matrices import Matrix, MatrixBase
from sympy.polys.polyclasses import DMP
from sympy.polys.rings import PolyElement
from sympy.polys.domains import Domain
from sympy.combinatorics import (Permutation, PermutationGroup,
    CyclicGroup, SymmetricGroup, AlternatingGroup, DihedralGroup
)
from ..sdp.arithmetic import rep_matrix_from_list

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

def generate_partitions(d_list: Union[int, List[int]], degree: int,
        equal: bool = False, descending: bool = True) -> List[Tuple[int, ...]]:
    """
    Generate all tuples (a0,a1,...,an) such that n = len(d_list) and sum(ai*di) <= degree.
    If equal is True, then it requires sum(ai*di) == degree.

    When d_list is an integer, it assumes d_list = [1, 1, ..., 1] of length nvars.
    """
    if isinstance(d_list, int):
        nvars = d_list
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
        return monoms if descending else monoms[::-1]

    n = len(d_list)
    if n == 0:
        return [tuple()]

    powers = []
    i = 0
    current_degree = 0
    current_powers = [0 for _ in range(n)]
    while True:
        if i == n - 1:
            if degree >= current_degree:
                if not equal:
                    for j in range(1 + (degree - current_degree)//d_list[i]):
                        current_powers[i] = j
                        powers.append(tuple(current_powers))
                elif (degree - current_degree) % d_list[i] == 0:
                    current_powers[i] = (degree - current_degree) // d_list[i]
                    powers.append(tuple(current_powers))
            i -= 1
            current_powers[i] += 1
            current_degree += d_list[i]
        else:
            if current_degree > degree:
                # reset the current power
                current_degree -= d_list[i] * current_powers[i]
                current_powers[i] = 0
                i -= 1
                if i < 0:
                    break
                current_powers[i] += 1
                current_degree += d_list[i]
            else:
                i += 1
    return powers[::-1] if descending else powers


def _poly_rep(poly: Union[Poly, DMP, PolyElement]) -> Tuple[List[Tuple], Domain, int, int]:
    """Return [(monom, coeff)], domain, ngens, degree"""
    if isinstance(poly, Poly):
        poly = poly.rep
    if isinstance(poly, DMP):
        return poly.terms(), poly.dom, poly.lev + 1, poly.total_degree()
    if isinstance(poly, PolyElement):
        degree = max(map(sum, poly.keys()), default=0)
        return list(poly.items()), poly.ring.domain, poly.ring.ngens, degree


class MonomialManager():
    """
    Class to compute polynomial monomials given the symmetry of variables and homogeneity.
    Monomials are sorted in graded lexicographical (grlex) order by default.
    """
    nvars: int
    _perm_group: PermutationGroup
    _is_homogeneous: bool

    OPTIMIZE_SYMMETRY = False
    def __init__(self,
        nvars: int,
        perm_group: Optional[PermutationGroup] = None,
        is_homogeneous: bool = True,
    ) -> None:
        self.nvars = nvars
        self._is_homogeneous = bool(is_homogeneous)
        if isinstance(perm_group, MonomialManager):
            perm_group = perm_group.perm_group
        self._perm_group = perm_group if perm_group is not None else PermutationGroup(Permutation(list(range(nvars))))
        if self._perm_group.degree != nvars:
            raise ValueError(f"The degree of the permutation group ({self._perm_group.degree}) does not match the number of variables ({nvars}).")

        self._monoms = {}

        if self.OPTIMIZE_SYMMETRY:
            orbits = self._perm_group.orbits()
            orbits = [sorted(list(o)) for o in orbits]# if len(o) > 1]
            order = self._perm_group.order()
            if prod([factorial(len(m)) for m in orbits]) == order:
                # internal direct product of symmetric groups
                def standard_monom(monom: Tuple[int, ...]) -> Tuple[int, ...]:
                    result = list(monom)
                    for orbit in orbits:
                        values = [result[i] for i in orbit]
                        values.sort(reverse=True)
                        for i, val in zip(orbit, values):
                            result[i] = val
                    return tuple(result)

                setattr(self, 'standard_monom', standard_monom)

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

    @property
    def is_symmetric(self) -> bool:
        return self._perm_group.is_symmetric

    @classmethod
    def from_perm_group(cls, perm_group: PermutationGroup, is_homogeneous: bool = True) -> 'MonomialManager':
        """
        Create a `MonomialManager` object from a permutation group. The `nvars` argument is inferred from
        the degree of the group and is therefore not needed.
        """
        return MonomialManager(perm_group.degree, perm_group=perm_group, is_homogeneous=is_homogeneous)

    @classmethod
    def add(cls, *monomials) -> Tuple[int, ...]:
        """
        Add multiple monomials (element-wise).
        """
        return tuple(map(sum, zip(*monomials)))

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
        # WARNING: This should be implemented when OPTIMIZE_SYMMETRY is True.
        obj = self.copy()
        obj._perm_group = PermutationGroup(Permutation(list(range(self.nvars))))
        def standard_monom(monom: Tuple[int, ...]) -> Tuple[int, ...]:
            return monom
        setattr(obj, 'standard_monom', standard_monom)
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
        return [tuple([monom[i] for i in perm._array_form]) for perm in self._perm_group.elements]

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

    def permute_vec(self, vec: Matrix, degree: int) -> Matrix:
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

        # Dense version (too slow when converting between reps and sympy objs)
        # all_vecs = [[0] * vec.shape[0] for _ in range(self.order())]
        # for v, m in zip(vec, inv_monoms):
        #     for ind, j in enumerate(map(f, self.permute(m))):
        #         # write the value to the corresponding column
        #         all_vecs[ind][j] = v
        # return Matrix(all_vecs).T

        sdm = {i: {} for i in range(vec.shape[0])}
        rep = vec._rep.rep.to_sdm()
        for i in rep.keys():
            v, m = rep[i][0], inv_monoms[i]
            for ind, j in enumerate(map(f, self.permute(m))):
                sdm[j][ind] = v
        return Matrix._fromrep(DomainMatrix.from_rep(SDM(sdm, (vec.shape[0], self.order()), rep.domain)))


    # def _standard_monom(self, monom: Tuple[int, ...]) -> Tuple[int, ...]:
    #     warn("_standard_monom is deprecated. Use standard_monom instead.", DeprecationWarning, stacklevel=2)
    #     return self.standard_monom(monom)

    def _assert_equal_nvars(self, t: Union[int, Iterable]) -> bool:
        if (t if isinstance(t, int) else len(t)) != self.nvars:
            raise ValueError("Number of variables does not match. Expected %d but received %d." % (self.nvars, len(t)))
            return False
        return True

    def _arraylize_list(self, poly: Union[Poly, DMP, PolyElement], degree: Optional[int] = None, expand_cyc: bool = False) -> List:
        rep, dom, ngens, _degree = _poly_rep(poly)
        if degree is None:
            degree = _degree
        self._assert_equal_nvars(ngens)
        dict_monoms = self.dict_monoms(degree)
        array = [dom.zero] * len(dict_monoms)
        if not expand_cyc:
            for monom, coeff in rep:
                v = dict_monoms.get(monom)
                if v is not None:
                    array[v] = coeff
        else:
            for monom, coeff in rep:
                for monom2 in self.permute(monom):
                    v = dict_monoms.get(monom2)
                    if v is not None:
                        array[v] += coeff
        return array

    def arraylize_np(self, poly: Union[Poly, DMP, PolyElement], degree: Optional[int] = None, expand_cyc: bool = False) -> np.ndarray:
        """
        Return the vector representation of the polynomial in numpy array.
        """
        vec = self._arraylize_list(poly, degree = degree, expand_cyc = expand_cyc)
        rep, dom, ngens, _degree = _poly_rep(poly)
        if not (dom is ZZ or dom is QQ):
            to_sympy = dom.to_sympy
            vec = [to_sympy(v) for v in vec]
        elif _IS_GROUND_TYPES_FLINT:
            if dom is QQ:
                vec = [int(v.numerator) / int(v.denominator) for v in vec]
            else:
                vec = [int(v) for v in vec]
        return np.array(vec).astype(np.float64)

    def arraylize_sp(self, poly: Union[Poly, DMP, PolyElement], degree: Optional[int] = None, expand_cyc: bool = False) -> Matrix:
        """
        Return the vector representation of the polynomial in sympy matrix (column vector).
        """
        vec = self._arraylize_list(poly, degree = degree, expand_cyc = expand_cyc)
        rep, dom, ngens, _degree = _poly_rep(poly)
        if SDM is not None:
            sdm = dict((i, {0: v}) for i, v in enumerate(vec) if v)
            return Matrix._fromrep(DomainMatrix.from_rep(SDM(sdm, (len(vec), 1), dom)))
        else: # sympy <= 1.7
            to_sympy = dom.to_sympy
            vec = [to_sympy(v) for v in vec]
            return Matrix(vec)

    def invarraylize(self, array: Union[List, np.ndarray, Matrix], gens: List[Symbol], degree: int) -> Poly:
        """
        Reverse the arraylize_np function to get the polynomial from its vector representation.
        """
        self._assert_equal_nvars(gens)
        inv_monoms = self.inv_monoms(degree)
        terms_dict = {}
        permute = self.permute if (not self.is_trivial) else lambda x: (x,)
        if SDM is not None and isinstance(array, MatrixBase):
            rep = array._rep.rep.to_sdm()
            domain = rep.domain
            zero = domain.zero
            # be careful to handle column / row vectors
            if rep.shape[0] == 1:
                rep_list = list(rep.get(0, dict()).items())
            elif rep.shape[1] == 1:
                rep_list = [(i, v.get(0, zero)) for i, v in rep.items()]
            elif rep.shape[0] == 0 or rep.shape[1] == 0:
                return Poly(0, *gens)
            else:
                raise ValueError(f"Array must be a vector, but received shape {array.shape}.")

            for i, z in rep_list:
                if z != zero:
                    monom = inv_monoms[i]
                    for monom2 in permute(monom):
                        terms_dict[monom2] = z
            rep = DMP.from_dict(terms_dict, len(gens)-1, domain)
            return Poly.new(rep, *gens)

        elif isinstance(array, np.ndarray):
            array = array.tolist()
        for coeff, monom in zip(array, inv_monoms):
            for monom2 in permute(monom):
                terms_dict[monom2] = coeff
        return Poly(terms_dict, gens)

    def cyclic_sum(self, expr: Expr, gens: List[Symbol]) -> Expr:
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
        return Add(*s)


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


def arraylize_np(
    poly: Union[Poly, DMP, PolyElement],
    degree: Optional[int] = None,
    expand_cyc: bool = False,
    **options
) -> np.ndarray:
    """
    Convert a sympy polynomial to a numpy vector of coefficients.
    Monomials are sorted in graded lexicographical (grlex) order.

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
    vec: np.ndarray
        Numpy vector that stores the coefficients of the polynomial.

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

    For non-homogeneous polynomials, `hom=False` must be provided. The `degree`
    argument should also be provided if its degree is lower than the desired degree.

    >>> print(arraylize_np((a**2 + 2*b + 3).as_poly(a,b), hom = False))
    [1. 0. 0. 0. 2. 3.]

    >>> print(arraylize_np((a**2 + 2*b + 3).as_poly(a,b), degree = 3, hom = False))
    [0. 0. 1. 0. 0. 0. 0. 0. 2. 3.]

    See Also
    ----------
    arraylize_sp, invarraylize, generate_monoms
    """
    nvars = (poly.rep if isinstance(poly, Poly) else poly).lev + 1
    option = _parse_options(nvars, **options)
    return option.arraylize_np(poly, degree = degree, expand_cyc = expand_cyc)


def arraylize_sp(
    poly: Union[Poly, DMP, PolyElement],
    degree: Optional[int] = None,
    expand_cyc: bool = False,
    **options
) -> Matrix:
    """
    Convert a sympy polynomial to a sympy vector of coefficients.
    Monomials are sorted in graded lexicographical (grlex) order.

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

    For non-homogeneous polynomials, `hom=False` must be provided. The `degree`
    argument should also be provided if its degree is lower than the desired degree.

    >>> print(arraylize_sp((a**2 + 2*b + 3).as_poly(a,b), hom = False))
    Matrix([[1], [0], [0], [0], [2], [3]])

    >>> print(arraylize_sp((a**2 + 2*b + 3).as_poly(a,b), degree = 3, hom = False))
    Matrix([[0], [0], [1], [0], [0], [0], [0], [0], [2], [3]])

    See Also
    ----------
    arraylize_np, invarraylize, generate_monoms
    """
    nvars = (poly.rep if isinstance(poly, Poly) else poly).lev + 1
    option = _parse_options(nvars, **options)
    return option.arraylize_sp(poly, degree = degree, expand_cyc = expand_cyc)


def invarraylize(array: Union[List, np.ndarray, Matrix], gens: List[Symbol], degree: int, **options) -> Poly:
    """
    Convert a vector representation of polynomial back to the sympy polynomial.
    Monomials are sorted in graded lexicographical (grlex) order.

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
    Monomials are sorted in graded lexicographical (grlex) order.

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


def verify_closure(l: List, f: Callable, get_rep: Optional[Callable]=None) -> bool:
    """
    Verify the list `l` is closed under an operator `f`.
    """
    if get_rep is None:
        get_rep = lambda x: x
    rep_set = set()
    f_set = set()
    for p in l:
        rep = get_rep(p)
        f_rep = get_rep(f(p))
        # if rep == f_rep:
        #     # it is invariant itself
        #     continue
        rep_set.add(rep)
        f_set.add(f_rep)
    for r in f_set:
        if r not in rep_set:
            return False
    return True


def parse_symmetry(symmetry: Union[PermutationGroup, str], n: int) -> PermutationGroup:
    if isinstance(symmetry, str):
        maps = {
            "cyc": CyclicGroup,
            "sym": SymmetricGroup,
            "alt": AlternatingGroup,
            "dih": DihedralGroup,
            "trivial": lambda n: PermutationGroup(Permutation(list(range(n))))
        }
        if symmetry in maps:
            symmetry = maps[symmetry](n)
        else:
            raise ValueError(
                f"Expected one of {tuple(maps.keys())} as symmetry, but received {symmetry}")
    elif not isinstance(symmetry, PermutationGroup):
        raise TypeError("Symmetry should be either PermutationGroup or str.")
    return symmetry


def verify_symmetry(
    polys: Union[List[Poly], Poly],
    symmetry: Union[str, Permutation, PermutationGroup, List[Permutation]]
) -> bool:
    """
    Verify whether the polynomials are symmetric with respect to the permutation group.

    Parameters
    ----------
    polys : Union[List[Poly], Poly]
        A list of polynomials or a single polynomial. Must have the same generators.
    symmetry : Union[str, Permutation, PermutationGroup]
        A permutation or a permutation group to verify. If string, it should be one of
        ["cyc", "sym", "alt", "dih", "trivial"].

    Returns
    ----------
    bool
        Whether the polynomials are symmetric with respect to the permutation group.

    Examples
    ----------
    >>> from sympy.combinatorics import Permutation, PermutationGroup, SymmetricGroup
    >>> from sympy.abc import a, b, c, d
    >>> verify_symmetry((a*(a-b)*(a-c)+b*(b-c)*(b-a)+c*(c-a)*(c-b)).as_poly(a,b,c), 'sym')
    True

    >>> f = lambda x: x.as_poly(a,b,c,d)
    >>> verify_symmetry([f(a+1),f(b+1),f(c+1),f(d+1)], SymmetricGroup(4))
    True

    >>> perm_group = PermutationGroup(Permutation([2,3,0,1]))
    >>> verify_symmetry([f(a-c+b-d)], perm_group)
    False
    >>> verify_symmetry([f(a-c+b-d), f(c-a+d-b)], perm_group)
    True
    """
    if isinstance(polys, Poly):
        polys = [polys]
    if len(polys) == 0:
        return True
    for p in polys:
        gens = p.gens
        break
    if len(polys) > 1 and any(p.gens != gens for p in polys):
        raise ValueError("All polynomials should have the same generators.")

    if isinstance(symmetry, str):
        symmetry = parse_symmetry(symmetry, len(gens))
    if isinstance(symmetry, PermutationGroup):
        if symmetry.degree != len(gens):
            raise ValueError("The permutation group should have the same degree as the number of generators.")
        perms = symmetry.generators
    elif isinstance(symmetry, Permutation):
        if symmetry.size != len(gens):
            raise ValueError("The permutation should have the same size as the number of generators.")
        perms = [symmetry]

    get_rep = lambda p: p.rep
    for perm in perms:
        reorder = lambda x: x.reorder(*perm(gens))
        if not verify_closure(polys, reorder, get_rep):
            return False
    return True


def _identify_symmetry_from_blackbox(
    f: Callable[[Permutation], bool],
    G: Union[PermutationGroup, int]
) -> PermutationGroup:
    """
    Identify symmetry by calling a black-box function `f` on each permutation.
    If `G` is integer, then it implies the degree of the permutation.
    If `G` is a permutation group, then it finds a subgroup of `G`.
    """
    # List a few candidates: symmetric, alternating, cyclic groups...
    def _rotated(n, start=0):
        return list(range(start+1, n+start)) + [start]
    def _reflected(n, start=0):
        return [start+1, start] + list(range(start+2, n+start))

    verified = [] # storing permutations that fit the input
    candidates = [] # a list of permutations

    if isinstance(G, int):
        nvars = G
        G = SymmetricGroup(nvars)
    else:
        nvars = G.degree

    if nvars > 1:
        candidates.append(_rotated(nvars))
        if nvars > 2:
            candidates.append(_reflected(nvars))

    for perm in map(Permutation, candidates):
        if f(perm):
            verified.append(perm)
    if len(verified) == 2:
        # reflection + cyclic -> complete symmetric group
        # but it should be a subgroup of G, hence return G
        return G
    verified = [arg for arg in verified if arg in G]

    candidates = []
    # bi-symmetric group etc.
    if nvars > 3:
        half = nvars // 2
        p1 = _rotated(half) + _rotated(half, half)
        p2 = _reflected(half) + _reflected(half, half)
        p3 = list(range(half,half*2)) + list(range(half))
        if nvars % 2 == 1:
            for p in [p1, p2, p3]:
                p.append(nvars - 1)
                candidates.append(p)
                p = [0] + [_ + 1 for _ in p[:-1]]
                candidates.append(p)
        else:
            for p in [p1, p2, p3]:
                candidates.append(p)

    if nvars > 2:
        candidates.append(_rotated(nvars - 1) + [nvars - 1])
        candidates.append([0] + _rotated(nvars - 1, 1))
        if nvars > 3:
            candidates.append(_reflected(nvars - 1) + [nvars - 1])
            candidates.append([0] + _reflected(nvars - 1, 1))

    is_sym = G._is_sym
    for perm in map(Permutation, candidates):
        if (is_sym or (perm in G)) and f(perm):
            verified.append(perm)

    if len(verified) == 0:
        verified.append(Permutation(list(range(nvars))))

    return PermutationGroup(*verified)


def identify_symmetry_from_lists(
    lst_of_lsts: List[List[Poly]],
    G: Optional[PermutationGroup]=None
) -> PermutationGroup:
    """
    Infer a symmetric group so that each list of (list of polynomials)
    is symmetric with respect to the rule. It only identifies very
    common groups like complete symmetric and cyclic groups.

    TODO: Implement a complete algorithm to identify all symmetric groups.

    Parameters
    ----------
    lst_of_lsts : List[List[Poly]]
        A list of lists of polynomials.
    G: Optional[PermutationGroup]
        The permutation group to be used. If provided, the
        returned group must be a subgroup of G.

    Returns
    ----------
    PermutationGroup
        The inferred permutation group.

    Examples
    ----------
    >>> from sympy.abc import a, b, c
    >>> identify_symmetry_from_lists([[(a+b+c-3).as_poly(a,b,c)],
    ... [a.as_poly(a,b,c), b.as_poly(a,b,c), c.as_poly(a,b,c)]]).is_symmetric
    True

    >>> identify_symmetry_from_lists([[(a+b+c-3).as_poly(a,b,c)],
    ... [(2*a+b).as_poly(a,b,c), (2*b+c).as_poly(a,b,c), (2*c+a).as_poly(a,b,c)]])
    PermutationGroup([
        (0 1 2)])

    See Also
    ----------
    identify_symmetry

    Reference
    ----------
    [1] https://cs.stackexchange.com/questions/64335/how-to-find-the-symmetry-group-of-a-polynomial
    """
    gens = ()
    for l in lst_of_lsts:
        for p in l:
            gens = p.gens
            break
        if gens:
            break
    for l in lst_of_lsts:
        for p in l:
            if p.gens != gens:
                raise ValueError("All polynomials should have the same generators.")

    def verify(perm):
        return all(verify_symmetry(l, perm) for l in lst_of_lsts)

    nvars = len(gens)
    if G is not None:
        if nvars != 0 and nvars != G.degree:
            raise ValueError("The degree of the permutation group must"
                             " be the same as the number of variables.")
        if nvars == 0:
            return G
        if G.is_trivial:
            return G
        return _identify_symmetry_from_blackbox(verify, G)
    return _identify_symmetry_from_blackbox(verify, nvars)


def identify_symmetry(poly: Poly, G: Optional[PermutationGroup]=None) -> PermutationGroup:
    """
    Infer a symmetric group so that the polynomial is symmetric with respect to the rule.
    It only identifies very simple groups like complete symmetric and cyclic groups.

    Parameters
    ----------
    poly: Poly
        The polynomial to be identified.
    G: Optional[PermutationGroup]
        The permutation group to be used. If provided, the
        returned group must be a subgroup of G.

    Returns
    ----------
    PermutationGroup
        The inferred permutation group.

    Examples
    ----------
    >>> from sympy.abc import a, b, c, d
    >>> identify_symmetry((a*b).as_poly(a,b,c,d)) # doctest:+SKIP
    PermutationGroup([
     (3)(0 1),
     (0 1)(2 3)])
    >>> identify_symmetry((a*b).as_poly(a,b,c,d)).order()
    4

    >>> from sympy.combinatorics import DihedralGroup
    >>> identify_symmetry((a*b).as_poly(a,b,c,d), DihedralGroup(4)) # doctest:+NORMALIZE_WHITESPACE
    PermutationGroup([
     (0 1)(2 3)])
    >>> identify_symmetry((a*b).as_poly(a,b,c,d), DihedralGroup(4)).order()
    2
    """
    return identify_symmetry_from_lists([[poly]], G)


def arraylize_up_to_symmetry(
    poly: Poly,
    perm_group: PermutationGroup,
    degree: Optional[int] = None,
    return_type: str = 'matrix',
    **options
) -> Union[List, Matrix]:
    """
    Get the canonical representation of the poly up to given symmetry.
    Two polynomials that are equivalent up to the symmetry should have the same representation.

    Parameters
    ----------
    poly: Poly
        The polynomial to be arraylized.
    perm_group: PermutationGroup
        Sympy permutation group object for the polynomial. This specifies the symmetry
        beyond cyclic and symmetric groups.
    return_type: str
        The type of the return value.
        If 'list', the return value is a list of coefficients.
        If 'matrix', the return value is a matrix.
        Default is 'matrix'.

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

    Returns
    --------
    matrix: Matrix
        Standard representation of the poly up to given symmetry.

    Examples
    ---------
    >>> from sympy.abc import a, b, c
    >>> from sympy.combinatorics import CyclicGroup
    >>> arraylize_up_to_symmetry((b**2 - 2*a*c + 3).as_poly(a,b,c), CyclicGroup(3), hom=False, return_type='list') # doctest:+SKIP
    [1, 0, 0, 0, 0, -2, 0, 0, 0, 3]
    >>> arraylize_up_to_symmetry((c**2 - 2*b*a + 3).as_poly(a,b,c), CyclicGroup(3), hom=False).tolist()
    [[1], [0], [0], [0], [0], [-2], [0], [0], [0], [3]]

    Note that the `arraylize_sp` method does not canonicalize the polynomial by symmetry:
    >>> arraylize_sp((b**2 - 2*a*c + 3).as_poly(a,b,c), hom=False).tolist()
    [[0], [0], [-2], [0], [1], [0], [0], [0], [0], [3]]
    """
    base = _parse_options(len(poly.gens), **options)
    domain = poly.domain
    vec = base.arraylize_sp(poly, degree=degree)
    shape = vec.shape[0]

    rep = vec._rep.rep.to_list_flat() # avoid conversion from rep to sympy
    # getvalue = lambda i: vec[i,0] # get a single value
    getvalue = lambda i: rep[i]
    # if shape <= 1:
    #     return tuple(getvalue(i) for i in range(shape))

    # # The naive implementation below could take minutes for calling 50 times on a 6-order group
    # mat = symmetry.permute_vec(vec, t.total_degree())
    # cols = [tuple(mat[:, i]) for i in range(mat.shape[1])]
    # return max(cols)

    # We should highly optimize the algorithm.
    dict_monoms = base.dict_monoms(poly.total_degree())
    inv_monoms = base.inv_monoms(poly.total_degree())

    def v(perm, i):
        """Get the value of index i in the vector after permutation"""
        # return getvalue(dict_monoms[tuple(perm(inv_monoms[i]))])

        # because perm.__call__ has sanity checks, we should make it faster
        ii = inv_monoms[i]
        pii = tuple([ii[j] for j in perm._array_form])
        return getvalue(dict_monoms[pii])

    perms = list(perm_group.elements)
    queue, queue_len, best_perm = [domain.zero]*shape, 0, perms[0]

    domain = vec._rep.domain
    key = lambda z: z
    if domain.is_EX:
        key = lambda z: z.ex.sort_key()
    elif domain.is_EXRAW:
        key = lambda z: z.sort_key()

    for perm in perms[1:]:
        for j in range(shape):
            s = v(perm, j)
            if j >= queue_len:
                # compare the next element
                queue[j] = v(best_perm, j)
                queue_len += 1
            key_s, key_j = key(s), key(queue[j])
            if key_s > key_j:
                queue[j], queue_len, best_perm = s, j + 1, perm
                break
            elif key_s < key_j:
                break
    for j in range(queue_len, shape): # fill the rest
        queue[j] = v(best_perm, j)
    if return_type == 'list':
        return queue
    return rep_matrix_from_list(queue, shape, domain)


@overload
def clear_polys_by_symmetry(
    polys: List[Union[Poly, Expr]],
    symbols: Tuple[Symbol, ...],
    symmetry: Union[PermutationGroup, MonomialManager],
) -> List[Union[Poly, Expr]]: ...
@overload
def clear_polys_by_symmetry(
    polys: List[Tuple[Union[Poly, Expr], Any]],
    symbols: Tuple[Symbol, ...],
    symmetry: Union[PermutationGroup, MonomialManager],
) -> List[Tuple[Union[Poly, Expr], Any]]: ...
@overload
def clear_polys_by_symmetry(
    polys: Dict[Union[Poly, Expr], Any],
    symbols: Tuple[Symbol, ...],
    symmetry: Union[PermutationGroup, MonomialManager],
) -> Dict[Union[Poly, Expr], Any]: ...


def clear_polys_by_symmetry(polys, symbols, symmetry):
    """
    Remove duplicate polynomials up to given symmetry. This function
    accepts list or dict inputs and preserves the input structure.

    Parameters
    ----------
    polys: list or dict
        A list or dict of polynomials or sympy expressions.
    symbols: list of Symbol
        The symbols in the polynomials.
    symmetry: Union[PermutationGroup, MonomialManager]
        The reference symmetry object.

    Returns
    --------
    polys: list or dict
        A list or dict of polynomials with duplicates removed. If a dict, the keys are the polynomials
        and the values are the corresponding data.

    Examples
    ---------
    >>> from sympy.abc import a, b, c
    >>> from sympy.combinatorics import CyclicGroup
    >>> polys = [b**2 - 2*a*c + 3, c**2 - 2*b*a + 3, b**2 - 2*b*c + 3, a*b - 1, b*c - 1]
    >>> clear_polys_by_symmetry(polys, (a,b,c), CyclicGroup(3))
    [-2*a*c + b**2 + 3, b**2 - 2*b*c + 3, a*b - 1]
    >>> clear_polys_by_symmetry({polys[i]: i for i in range(5)}, (a,b,c), CyclicGroup(3))
    {-2*a*c + b**2 + 3: 0, b**2 - 2*b*c + 3: 2, a*b - 1: 3}
    """
    base, perm_group = None, None
    if isinstance(symmetry, PermutationGroup):
        base = MonomialManager(len(symbols), is_homogeneous=False)
        perm_group = symmetry
    else:
        base = symmetry.base()
        perm_group = symmetry.perm_group

    def _get_rep(t):
        if isinstance(t, tuple):
            t = t[0]
        if not isinstance(t, Poly):
            t = Poly(t, symbols)
        rep = arraylize_up_to_symmetry(t, perm_group, symmetry=base, return_type='list')
        return (t.total_degree(),) + tuple(rep)

    is_dict = isinstance(polys, dict)
    if is_dict:
        if perm_group.is_trivial:
            return polys # .copy()
        polys = polys.items()

    # clear duplicate expressions while preserving order
    reps = set()
    collected = []
    for i, t in enumerate(polys):
        t_rep = _get_rep(t)
        if t_rep not in reps:
            reps.add(t_rep)
            collected.append(t)
    if is_dict:
        collected = dict(collected)
    return collected


def poly_reduce_by_symmetry(poly: Poly, symmetry: Union[str, PermutationGroup]) -> Poly:
    """
    Given a polynomial which is symmetric with respect to the permutation group,
    return a new_poly such that `CyclicSum(new_poly, new_poly.gens, perm_group) == poly`.
    Users should ensure that the given poly is symmetric with respect to the permutation group
    and this is not checked.

    Parameters
    ----------
    poly: Poly
        The polynomial to be reduced.
     symmetry: Union[str, PermutationGroup]
        The permutation group to be considered. If it is a string, it should be one of
        ["cyc", "sym", "alt", "dih", "trivial"].

    Returns
    ----------
    Poly
        The reduced polynomial.

    Examples
    ----------
    >>> from sympy.abc import a, b, c, d
    >>> from sympy.combinatorics import SymmetricGroup, DihedralGroup, CyclicGroup
    >>> p1 = (a**2+b**2+c**2+d**2+a*b+b*c+c*d+d*a+a*c+b*d).as_poly(a,b,c,d)
    >>> poly_reduce_by_symmetry(p1, SymmetricGroup(4))
    Poly(1/6*a**2 + 1/4*a*b, a, b, c, d, domain='QQ')
    >>> poly_reduce_by_symmetry(p1, DihedralGroup(4))
    Poly(1/2*a**2 + 1/2*a*b + 1/4*a*c, a, b, c, d, domain='QQ')
    >>> poly_reduce_by_symmetry(p1, CyclicGroup(4))
    Poly(a**2 + a*b + 1/2*a*c, a, b, c, d, domain='QQ')
    """
    if symmetry is None:
        return poly
    perm_group = parse_symmetry(symmetry, len(poly.gens))

    extracted = []
    perm_group_gens = perm_group.generators
    perm_order = perm_group.order()
    ufs = {}
    # monomials invariant under the permutation group is recorded in ufs
    def ufs_find(monom):
        v = ufs.get(monom, monom)
        if v == monom:
            return monom
        w = ufs_find(v)
        ufs[monom] = w
        return w
    for m1, coeff in poly.terms():
        for p in perm_group_gens:
            m2 = tuple(p(m1))
            f1, f2 = ufs_find(m1), ufs_find(m2)
            # merge to the maximum
            if f1 > f2:
                ufs[f2] = f1
            else:
                ufs[f1] = f2

    ufs_size = defaultdict(int)
    for m in ufs.keys():
        ufs_size[ufs_find(m)] += 1

    def get_order(monom):
        # get the multiplicity of the monomials given the permutation group
        # i.e. how many permutations make it invariant
        return perm_order // ufs_size[ufs_find(monom)]

    # only reserve the keys for ufs[monom] == monom
    for monom, coeff in poly.terms():
        if ufs_find(monom) == monom:
            order = get_order(monom)
            extracted.append((monom, coeff/order))
    return Poly(dict(extracted), poly.gens)
