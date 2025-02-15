from functools import lru_cache
from typing import Any, Union, Tuple, List, Dict, Callable, Optional
from time import time

import sympy as sp
import numpy as np
from scipy.sparse import coo_matrix
from sympy.combinatorics import PermutationGroup, Permutation
from sympy.core.singleton import S

from ...utils import arraylize_np, arraylize_sp, MonomialManager

_VERBOSE_GENERATE_QUAD_DIFF = False

def tuple_sum(t1: Tuple[int, ...], t2: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(x + y for x, y in zip(t1, t2))

class _callable_expr():
    """
    Callable expression is a wrapper of sympy expression that can be called with symbols,
    it is more like a function. It accepts an addition kwarg poly=True/False.

    Example
    ========
    >>> _callable_expr.from_expr(a**3*b**2, (a,b))((x,y))
    x**3*y**2

    >>> e = _callable_expr.from_expr(sp.Function("F")(a,b,c), (a,b,c), (a**3+b**3+c**3).as_poly(a,b,c))
    >>> e((a,b,c))
    F(a,b,c)
    >>> e((a,b,c), poly=True)
    Poly(a**3 + b**3 + c**3, a, b, c, domain='ZZ')
    
    """
    __slots__ = ['_func']
    def __init__(self, func: Callable[[Tuple[sp.Symbol, ...], Any], sp.Expr]):
        self._func = func
    def __call__(self, symbols: Tuple[sp.Symbol, ...], *args, **kwargs) -> sp.Expr:
        return self._func(symbols, *args, **kwargs)

    @classmethod
    def from_expr(cls, expr: sp.Expr, symbols: Tuple[sp.Symbol, ...], p: Optional[sp.Poly] = None) -> '_callable_expr':
        if p is None:
            def func(s, poly=False):
                e = expr.xreplace(dict(zip(symbols, s)))
                if poly: e = e.as_poly(s)
                return e
        else:
            def func(s, poly=False):
                if not poly:
                    return expr.xreplace(dict(zip(symbols, s)))
                return p.as_expr().xreplace(dict(zip(symbols, s))).as_poly(s)
        return cls(func)

    def default(self, nvars: int) -> sp.Expr:
        """
        Get the defaulted value of the expression given nvars.
        """
        symbols = sp.symbols(f'x:{nvars}')
        return self._func(symbols)


class LinearBasis():
    def nvars(self) -> int:
        raise NotImplementedError
    def _get_default_symbols(self) -> Tuple[sp.Symbol, ...]:
        return tuple(sp.symbols(f'x:{self.nvars()}'))
    def as_expr(self, symbols) -> sp.Expr:
        raise NotImplementedError
    def as_poly(self, symbols) -> sp.Poly:
        return self.as_expr(symbols).doit().as_poly(symbols)
    def degree(self) -> int:
        return self.as_poly(self._get_default_symbols()).total_degree()
    def as_array_np(self, **kwargs) -> np.ndarray:
        return arraylize_np(self.as_poly(self._get_default_symbols()), **kwargs)
    def as_array_sp(self, **kwargs) -> sp.Matrix:
        return arraylize_sp(self.as_poly(self._get_default_symbols()), **kwargs)

class LinearBasisExpr(LinearBasis):
    __slots__ = ['_expr', '_symbols']
    def __init__(self, expr: sp.Expr, symbols: Tuple[int, ...]):
        self._expr = expr.as_expr()
        self._symbols = symbols
    def nvars(self) -> int:
        return len(self._symbols)
    def as_expr(self, symbols) -> sp.Expr:
        return self._expr.xreplace(dict(zip(self._symbols, symbols)))

class LinearBasisTangent(LinearBasis):
    _degree_step = 1
    __slots__ = ['_powers', '_tangent']
    def __init__(self, powers: Tuple[int, ...], tangent: sp.Expr, symbols: Tuple[sp.Symbol, ...]):
        self._powers = powers
        self._tangent = _callable_expr.from_expr(tangent, symbols)
    @property
    def powers(self) -> Tuple[int, ...]:
        return self._powers
    @property
    def tangent(self) -> _callable_expr:
        return self._tangent
    def nvars(self) -> int:
        return len(self._powers)
    def as_expr(self, symbols) -> sp.Expr:
        return sp.Mul(*(x**i for x, i in zip(symbols, self._powers))) * self._tangent(symbols).as_expr()
    def as_poly(self, symbols) -> sp.Poly:
        return sp.Poly.from_dict({self._powers: 1}, symbols) * self._tangent(symbols, poly=True)
    def __neg__(self) -> 'LinearBasisTangent':
        return self.__class__.from_callable_expr(self._powers, lambda *args, **kwargs: -self._tangent(*args, **kwargs).as_expr())

    def to_even(self, symbols: List[sp.Expr]) -> 'LinearBasisTangentEven':
        """
        Convert the linear basis to an even basis.
        """
        rem_powers = tuple(d % 2 for d in self._powers)
        even_powers = tuple(d - r for d, r in zip(self._powers, rem_powers))
        def _new_tangent(s, poly=False):
            if poly: return self._tangent(s, poly=True)
            monom = sp.Mul(*(symbols[i] for i, d in enumerate(rem_powers) if d))
            return self._tangent(s, poly=False).as_expr() * monom
        return LinearBasisTangentEven.from_callable_expr(even_powers, _callable_expr(_new_tangent))

    @classmethod
    def from_callable_expr(cls, powers: Tuple[int, ...], tangent: _callable_expr) -> 'LinearBasisTangent':
        """
        Create a LinearBasisTangent from powers and a callable expression. This is intended for
        internal use only.
        """
        obj = cls.__new__(cls)
        obj._powers = powers
        obj._tangent = tangent
        return obj

    @classmethod
    def generate(cls, tangent: sp.Expr, symbols: Tuple[int, ...], degree: int, tangent_p: Optional[sp.Poly] = None, require_equal: bool = True) -> List['LinearBasisTangent']:
        """
        Generate all possible linear bases of the form x1^a1 * x2^a2 * ... * xn^an * tangent
        with total degree == degree or total degree <= degree.
        """
        if tangent_p is None:
            tangent_degree = tangent.as_poly(symbols).total_degree()
        else:
            tangent_degree = tangent_p.total_degree()
        degree = degree - tangent_degree
        step = cls._degree_step
        if degree < 0 or degree % step != 0:
            return []
        tangent = _callable_expr.from_expr(tangent, symbols, p=tangent_p)
        return [LinearBasisTangent.from_callable_expr(tuple(i*step for i in comb), tangent) for comb in \
                _degree_combinations([cls._degree_step] * len(symbols), degree, require_equal=require_equal)]

    @classmethod
    def generate_quad_diff(cls, 
            tangent: sp.Expr, symbols: Tuple[sp.Symbol, ...], degree: int, symmetry: PermutationGroup,
            tangent_p: Optional[sp.Poly] = None, quad_diff: bool = True
        ) -> Tuple[List['LinearBasisTangent'], np.ndarray]:
        """
        Generate all possible linear bases of the form x1^a1 * x2^a2 * ... * xn^an * (x1-x2)^(2b_12) * ... * (xi-xj)^(2b_ij) * tangent
        with total degree == degree.
        Also, return the matrix representation of the bases.
        """
        basis, mat, perm_group = None, None, None
        cache = _get_tangent_cache_key(cls, tangent, symbols) if quad_diff else None
        if cache is not None:
            perm_group = symmetry.perm_group if isinstance(symmetry, MonomialManager) else symmetry
            basis = cache.get((degree, len(symbols)))
            if basis is not None:
                mat = cache.get((degree, perm_group))
                if mat is not None:
                    return basis, mat

        if not isinstance(tangent_p, sp.Poly):
            p = tangent.as_poly(symbols)
        else:
            p = tangent_p
        d = p.total_degree()
        if p.is_zero or len(p.free_symbols_in_domain) or d > degree:
            return [], np.array([], dtype='float')

        if quad_diff:
            quad_diff = quadratic_difference(symbols)
            cross_tangents = cross_exprs(quad_diff, symbols, degree - d)
        else:
            cross_tangents = [S.One]

        if _VERBOSE_GENERATE_QUAD_DIFF:
            print('GenerateQuadDiff cross_tangents num =', len(cross_tangents))
            time0 = time()

        if basis is None:
            # no cache, generate the bases first
            basis = []
            for t in cross_tangents:
                p2 = t.as_poly(symbols) * p
                basis += cls.generate(t * tangent, symbols, degree, tangent_p=p2, require_equal=True)

            if _VERBOSE_GENERATE_QUAD_DIFF:
                print('>> Time for generating bases instances:', time() - time0)
                time0 = time()

        if mat is None:
            # convert the bases to matrix
            # mat = np.array([x.as_array_np(expand_cyc=True, symmetry=symmetry) for x in basis]) # too slow
            mat = []
            step = cls._degree_step

            symmetry = MonomialManager(len(symbols), symmetry)
            for t in cross_tangents:
                p2 = t.doit().as_poly(symbols) * p

                degree_comb_mat = _degree_combinations([step] * len(symbols), degree - p2.homogeneous_order(), require_equal=True)
                degree_comb_mat = np.array(degree_comb_mat, dtype='int32') * step

                mat2 = _get_matrix_of_lifted_degrees(p2, degree_comb_mat, symmetry, degree)
                mat.append(mat2)

            mat = np.vstack(mat) if len(mat) > 0 else np.array([], dtype='float')

            if _VERBOSE_GENERATE_QUAD_DIFF:
                print('>> Time for converting bases to matrix:', time() - time0)

        if cache is not None:
            # cache the result
            cache[(degree, len(symbols))] = basis
            cache[(degree, perm_group)] = mat

        return basis, mat


class LinearBasisTangentEven(LinearBasisTangent):
    """
    Ensure the degree of each monomial is even.
    """
    _degree_step = 2


def _degree_combinations(d_list: List[int], degree: int, require_equal = False) -> List[Tuple[int, ...]]:
    """
    Find a1, a2, ..., an such that a1*d1 + a2*d2 + ... + an*dn <= degree.
    """
    n = len(d_list)
    if n == 0:
        return []

    powers = []
    i = 0
    current_degree = 0
    current_powers = [0 for _ in range(n)]
    while True:
        if i == n - 1:
            if degree >= current_degree:
                if not require_equal:
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
    return powers

def cross_exprs(exprs: List[sp.Expr], symbols: Tuple[sp.Symbol, ...], degree: int) -> List[sp.Expr]:
    """
    Generate cross products of exprs within given degree.
    """
    polys = [_.as_poly(symbols) for _ in exprs]
    poly_degrees = [_.total_degree() for _ in polys]

    # remove zero-degree polynomials
    polys = [p for p, d in zip(polys, poly_degrees) if d > 0]
    poly_degrees = [d for d in poly_degrees if d > 0]
    if len(polys) == 0:
        return []

    # find all a1*d1 + a2*d2 + ... + an*dn <= degree
    powers = _degree_combinations(poly_degrees, degree)
    # map the powers to expressions
    return [sp.Mul(*(x**i for x, i in zip(exprs, p))) for p in powers]

def quadratic_difference(symbols: Tuple[sp.Symbol, ...]) -> List[sp.Expr]:
    """
    Generate all expressions of the form (ai - aj)^2

    Example
    ========
    >>> quadratic_difference((a, b, c))
    [(a - b)**2, (a - c)**2, (b - c)**2]
    """
    exprs = []
    symbols = sorted(list(symbols), key=lambda x: x.name)
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            exprs.append((symbols[i] - symbols[j])**2)
    return exprs

###########################################################
# Fast operations for computing basis
###########################################################


def _define_common_tangents() -> List[sp.Expr]:
    # define keys of quad_diff bases that should be cached
    a, b, c, d = sp.symbols('x:4')
    return [
        S.One,
        (a**2 - b*c)**2, (b**2 - a*c)**2, (c**2 - a*b)**2,
        (a**3 - b*c**2)**2, (a**3 - b**2*c)**2, (b**3 - a*c**2)**2,
        (b**3 - a**2*c)**2, (c**3 - a*b**2)**2, (c**3 - a**2*b)**2,
    ]

_CACHED_TANGENT_BASIS = dict((k, {}) for k in _define_common_tangents())
_CACHED_TANGENT_BASIS_EVEN = dict((k, {}) for k in _define_common_tangents())

def _get_tangent_cache_key(cls, tangent: sp.Expr, symbols: Tuple[int, ...]) -> Optional[Dict]:
    """
    Given a tangent and symbols, return the cache key if it is in the cache.
    """
    callable_tangent = _callable_expr.from_expr(tangent, symbols)
    std_tangent = callable_tangent.default(len(symbols))
    cache = _CACHED_TANGENT_BASIS if cls is LinearBasisTangent else _CACHED_TANGENT_BASIS_EVEN
    return cache.get(std_tangent)



def _compute_sym_multiplicity(arr: np.ndarray, need_sort: bool = True) -> np.ndarray:
    """
    Compute by row the multiplicity of each row invariant in the
    completely symmetric permutation.
    """
    X, N = arr.shape
    if need_sort:
        arr = np.sort(arr, axis=1)
    mask = (arr[:, 1:] != arr[:, :-1]).astype('int')
    indices = np.tile(np.arange(1,N), (X, 1)) * mask
    indices = np.hstack([np.zeros((X,1),dtype='int'), indices, np.full((X,1),N,dtype='int')])
    ind_cum_max = np.maximum.accumulate(indices, axis=1)
    length_of_group = np.diff(ind_cum_max, axis=1)

    factorial = np.cumprod([1] + list(range(1, N+1)))
    group_factorial = factorial[length_of_group]
    group_invariant = np.prod(group_factorial, axis=1)
    return group_invariant


@lru_cache()
def _get_reduced_indices(symmetry: MonomialManager, degree: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the indices of monomials after being reduced by symmetry.

    Since it is called multiple times in generating, the function is wrapper by lru_cache.
    However, this might need optimization for parallel use.
    
    Returns
    -----------
    full_encod_to_reduced_indices : np.ndarray
        A mapping from encoding of full monomials to encoding of reduced monomials.
    multiplicity : np.ndarray
        The multiplicity of each reduced monomial in cyclic sums.
        It equals to the number of permutations that make the reduced monomial invariant.
    """
    _DTYPE = 'int32'
    encoding = np.array([(degree + 1)**i for i in range(symmetry.nvars)], dtype=_DTYPE)
    reduced_monoms = symmetry.inv_monoms(degree)
    reduced_monoms = np.array(reduced_monoms, dtype=_DTYPE)
    reduced_encodings = reduced_monoms @ encoding

    # mapping reduced encoding to indices
    reduced_indices = np.zeros(reduced_encodings.max() + 1, dtype=_DTYPE)
    reduced_indices[reduced_encodings] = np.arange(reduced_monoms.shape[0])

    no_symmetry = symmetry.base()
    full_monoms = no_symmetry.inv_monoms(degree)

    if symmetry.perm_group.is_trivial:
        full_to_reduced = full_monoms
        multiplicity = np.ones(reduced_monoms.shape[0], dtype=_DTYPE)
    elif symmetry.perm_group.is_symmetric:
        full_to_reduced = np.sort(np.array(full_monoms, dtype=_DTYPE), axis=1)[:, ::-1]
        multiplicity = _compute_sym_multiplicity(reduced_monoms, need_sort=False)
    else:
        full_to_reduced = [symmetry.standard_monom(m) for m in full_monoms]
        multiplicity = [sum(_ == m for _ in symmetry.permute(m)) for m in 
                            symmetry.inv_monoms(degree)]
        multiplicity = np.array(multiplicity, dtype=_DTYPE)


    full_encoding = np.array(full_monoms, dtype=_DTYPE) @ encoding
    full_to_reduced = np.array(full_to_reduced, dtype=_DTYPE) @ encoding

    full_encod_to_reduced_indices = np.zeros(full_encoding.max() + 1, dtype=_DTYPE)
    full_encod_to_reduced_indices[full_encoding] = reduced_indices[full_to_reduced]
    return full_encod_to_reduced_indices, multiplicity


def _count_contribution_of_monoms(A: np.ndarray, v: np.ndarray, M: int) -> np.ndarray:
    """
    (Written by Deepseek R1)

    Parameters
    -----------
    A : np.ndarray with shape (X, N), each element to be an integer in [0, M)
    v : np.ndarray with shape (N,)
    M : int
    
    Returns
    -----------
    B : np.ndarray
        B[i,m] = sum(v[j] for j where A[i,j] == m)
    """
    A = np.asarray(A)
    v = np.asarray(v)
    X, N = A.shape
    # assert v.shape == (N,)

    # it seems scipy is slower?
    if False:
        rows = np.repeat(np.arange(X), N)  # row coors [0,0,0,1,1,1,...]
        cols = A.ravel()                   # column coors [A[0,0],A[0,1],...,A[1,0],...]
        data = np.tile(v, X)               # values [v[0],v[1],...,v[0],v[1],...]
        
        return coo_matrix((data, (rows, cols)), shape=(X, M)).toarray()
    else:
        B = np.zeros((X, M), dtype=v.dtype)

        for m in range(M):
            mask = (A == m)
            B[:, m] = np.dot(mask, v)

        return B

def _get_matrix_of_lifted_degrees(poly: sp.Poly, degree_comb_mat: np.ndarray,
        symmetry: MonomialManager, degree: int) -> np.ndarray:

    if degree_comb_mat.shape[0] == 0:
        return np.array([], dtype='float')

    symbols = poly.gens

    # # This a naive implementation
    # def _naive_implementation():
    #     mat = [None] * degree_comb_mat.shape[0]
    #     poly_from_dict = sp.Poly.from_dict
    #     p2dict = poly.as_dict()
    #     for power in degree_comb_mat:
    #         new_p_dict = dict((tuple_sum(power, k), v) for k, v in p2dict.items())
    #         new_p = poly_from_dict(new_p_dict, symbols)
    #         mat[mat_ind] = symmetry.arraylize_np(new_p, expand_cyc=True)
    #         mat_ind += 1
    #     return np.vstack(mat) if len(mat) > 0 else np.array([], dtype='float')

    # Below is a faster, low-level implementation
    # But it is not equivalent to the naive implementation
    nvars = len(symbols)

    # encoding is a vector dot operation that maps monomials to a single integer
    # e.g. (4,3,2,1) -> 4 + 3*5 + 2*5^2 + 1*5^3 = 4 + 15 + 50 + 125 = 194
    # i.e. (4,3,2,1) * (1,5,25,125) = 194
    # Since it is linear, encoding of sum of monomials is the sum of encodings.
    # Don't worry about overflow, if there are too many monomials, then it is
    # impossible to solve the problem anyway.
    _DTYPE = 'int32'
    encoding = np.array([(degree + 1)**i for i in range(nvars)], dtype=_DTYPE)

    source_symmetry = symmetry.base()
    source_monoms = source_symmetry.inv_monoms(poly.total_degree())  # a list of monomials
    source_monoms = np.array(source_monoms, dtype=_DTYPE) @ encoding

    degree_comb_mat = degree_comb_mat.astype(_DTYPE) @ encoding

    target_monoms = symmetry.inv_monoms(degree)
    length_of_target = len(target_monoms)
    target_monoms = np.array(target_monoms, dtype=_DTYPE) @ encoding

    # add source monoms with degree_comb_mat
    # TODO: will the matrix cause MemoryError?
    source_monoms = np.tile(source_monoms.reshape((1, -1)), (degree_comb_mat.shape[0], 1))
    new_monoms = source_monoms + degree_comb_mat.reshape((-1, 1))

    # map encoding to indices
    inv_target_monoms, multiplicity = _get_reduced_indices(symmetry, degree)

    new_monoms = inv_target_monoms[new_monoms] # map to the indices of target monoms

    poly_vec = source_symmetry.arraylize_np(poly) #, expand_cyc=True)

    new_mat = _count_contribution_of_monoms(new_monoms, poly_vec, length_of_target)
    new_mat = new_mat * multiplicity.reshape((1, -1))

    return new_mat