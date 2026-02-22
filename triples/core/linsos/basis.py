from functools import lru_cache, wraps
from typing import Any, Union, Tuple, List, Dict, Callable, Optional
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix
from sympy import Poly, Expr, Symbol, Mul, Pow, Integer, Basic
from sympy import symbols as sp_symbols
from sympy.matrices import MutableDenseMatrix as Matrix
from sympy.combinatorics import PermutationGroup
from sympy.polys.polyclasses import DMP
from sympy.polys.rings import PolyRing, PolyElement

from ...utils import arraylize_np, arraylize_sp, MonomialManager
from ...utils.monomials import generate_partitions


_VERBOSE_GENERATE_QUAD_DIFF = False

def tuple_sum(t1: Tuple[int, ...], t2: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(x + y for x, y in zip(t1, t2))

class _callable_expr():
    """
    Callable expression is a wrapper of sympy expression that can be called with symbols,
    it is more like a function. It accepts an addition kwarg poly=True/False.

    Example
    ========
    >>> from sympy.abc import a, b, c, x, y
    >>> from sympy import Function
    >>> _callable_expr.from_expr(a**3*b**2, (a,b))((x,y))
    x**3*y**2

    >>> e = _callable_expr.from_expr(Function("F")(a,b,c), (a,b,c), (a**3+b**3+c**3).as_poly(a,b,c))
    >>> e((a,b,c))
    F(a, b, c)
    >>> e((a,b,c), poly=True)
    Poly(a**3 + b**3 + c**3, a, b, c, domain='ZZ')

    """
    __slots__ = ['_func']
    def __init__(self, func: Callable[[Tuple[Symbol, ...], Any], Expr]):
        self._func = func
    def __call__(self, symbols: Tuple[Symbol, ...], *args, **kwargs) -> Expr:
        return self._func(symbols, *args, **kwargs)

    @classmethod
    def from_expr(cls, expr: Expr, symbols: Tuple[Symbol, ...], p: Optional[Poly] = None) -> '_callable_expr':
        if p is None:
            def func(s, poly=False):
                e = expr.xreplace(dict(zip(symbols, s)))
                if poly: e = e.as_poly(s)
                return e
        else:
            def func(s, poly=False):
                if not poly:
                    return expr.xreplace(dict(zip(symbols, s)))
                # new_p = p.as_expr().xreplace(dict(zip(symbols, s))).as_poly(s)
                new_p = Basic.__new__(Poly)
                new_p.gens = tuple(s)
                if isinstance(p, PolyElement):
                    new_p.rep = DMP.from_dict(dict(p), len(new_p.gens) - 1, p.ring.domain)
                elif isinstance(p, Poly):
                    new_p.rep = p.rep
                elif isinstance(p, DMP):
                    new_p.rep = p
                return new_p

        return cls(func)

    def default(self, nvars: int) -> Expr:
        """
        Get the defaulted value of the expression given nvars.
        """
        symbols = sp_symbols(f'x:{nvars}')
        return self._func(symbols)


class LinearBasis():
    def nvars(self) -> int:
        raise NotImplementedError
    def _get_default_symbols(self) -> Tuple[Symbol, ...]:
        return tuple(sp_symbols(f'x:{self.nvars()}'))
    def as_expr(self, symbols) -> Expr:
        raise NotImplementedError
    def as_poly(self, symbols) -> Poly:
        return self.as_expr(symbols).doit().as_poly(symbols)
    def degree(self) -> int:
        return self.as_poly(self._get_default_symbols()).total_degree()
    def as_array_np(self, **kwargs) -> np.ndarray:
        return arraylize_np(self.as_poly(self._get_default_symbols()), **kwargs)
    def as_array_sp(self, **kwargs) -> Matrix:
        return arraylize_sp(self.as_poly(self._get_default_symbols()), **kwargs)

class LinearBasisExpr(LinearBasis):
    __slots__ = ['_expr', '_symbols']
    def __init__(self, expr: Expr, symbols: Tuple[int, ...]):
        self._expr = expr.as_expr()
        self._symbols = symbols
    def nvars(self) -> int:
        return len(self._symbols)
    def as_expr(self, symbols) -> Expr:
        return self._expr.xreplace(dict(zip(self._symbols, symbols)))

class LinearBasisTangent(LinearBasis):
    _degree_step = 1
    __slots__ = ['_powers', '_tangent']
    def __init__(self, powers: Tuple[int, ...], tangent: Expr, symbols: Tuple[Symbol, ...]):
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
    def as_expr(self, symbols) -> Expr:
        return Mul(*(x**i for x, i in zip(symbols, self._powers))) * self._tangent(symbols).as_expr()
    def as_poly(self, symbols) -> Poly:
        return Poly.from_dict({self._powers: 1}, symbols) * self._tangent(symbols, poly=True)
    def __neg__(self) -> 'LinearBasisTangent':
        return self.__class__.from_callable_expr(self._powers, lambda *args, **kwargs: -self._tangent(*args, **kwargs).as_expr())
    def __len__(self) -> int:
        return 1
    def to_even(self, symbols: List[Expr]) -> 'LinearBasisTangentEven':
        """
        Convert the linear basis to an even basis.
        """
        rem_powers = tuple(d % 2 for d in self._powers)
        even_powers = tuple(d - r for d, r in zip(self._powers, rem_powers))
        def _new_tangent(s, poly=False):
            if poly: return self._tangent(s, poly=True)
            monom = Mul(*(symbols[i] for i, d in enumerate(rem_powers) if d))
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
    def generate(cls, tangent: Expr, symbols: Tuple[int, ...], degree: int, tangent_p: Optional[Poly] = None, require_equal: bool = True) -> List['LinearBasisTangent']:
        """
        Generate all possible linear bases of the form x1^a1 * x2^a2 * ... * xn^an * tangent
        with total degree == degree or total degree <= degree.
        """
        if tangent_p is None:
            tangent_degree = tangent.doit().as_poly(symbols).total_degree()
        elif isinstance(tangent_p, PolyElement):
            tangent_degree = max(map(sum, tangent_p.keys()), default=0)
        else:
            tangent_degree = tangent_p.total_degree()
        degree = degree - tangent_degree
        step = cls._degree_step
        if degree < 0 or degree % step != 0:
            return []
        tangent = _callable_expr.from_expr(tangent, symbols, p=tangent_p)
        return [LinearBasisTangent.from_callable_expr(tuple(i*step for i in comb), tangent) for comb in \
                generate_partitions([cls._degree_step] * len(symbols), degree, equal=require_equal, descending=False)]

    @classmethod
    def generate_quad_diff(cls,
            tangent: Expr, symbols: Tuple[Symbol, ...], degree: int, symmetry: PermutationGroup,
            tangent_p: Optional[Poly] = None, quad_diff_order: Union[bool, int] = 8,
        ) -> Tuple[List['LinearBasisTangent'], np.ndarray]:
        """
        Generate all possible linear bases of the form x1^a1 * x2^a2 * ... * xn^an * (x1-x2)^(2b_12) * ... * (xi-xj)^(2b_ij) * tangent
        with total degree == degree.
        Also, return the matrix representation of the bases.

        Parameters
        ----------
        tangent: Expr
            The sympy expression of the tangent.
        symbols: Tuple[Symbol, ...]
            A tuple of symbols.
        degree: int
            The total degree of the generated bases.
        symmetry: PermutationGroup
            The permutation group of the symmetry. Bases are summed
            over the permutation group before converting to matrix representation.
        tangent_p: Optional[Poly]
            The sympy polynomial of the tangent. If the tangent parameter
            is an alias of the polynomial that does not actually
            form a polynomial, then this parameter should be used.
            See also the example below.
        quad_diff_order: Union[bool, int]
            If an intger, generate only quadratic differences with degree <= quad_diff.
            If True, no upper bound is set. If False, do not generate quadratic differences.
            Set this smaller to reduce the number of bases generated and improve speed.

        Examples
        --------
        >>> from sympy.abc import a, b, c
        >>> from sympy.combinatorics import CyclicGroup
        >>> from sympy import Function
        >>> bases, mat = LinearBasisTangent.generate_quad_diff(\
                Function("F")(a,b,c), (a,b,c), 4, CyclicGroup(3),\
                tangent_p=((a+b-c)**2).as_poly(a,b,c), quad_diff_order=4)

        >>> [_.__class__.__name__ for _ in [bases[0], mat]]
        ['LinearBasisTangent', 'ndarray']

        >>> (bases[8].as_expr((a,b,c)), mat[8])
        ((a - b)**2*F(a, b, c), array([ 2., -2., -2.,  0.,  2.]))
        """
        # 1. standardize the input
        if tangent_p is None:
            tangent_p = tangent.doit().as_poly(symbols)

        if isinstance(quad_diff_order, bool):
            quad_diff_order = 2147483647 if quad_diff_order else 0
        quad_diff_order = max(0, min(quad_diff_order, degree - tangent_p.total_degree()))

        # 2. get cross(quad_diff) * tangent_p
        cross_exprs_mul_p, cross_polys_mul_p = _get_cross_exprs_and_polys_of_quad_diff(
            symbols, quad_diff_order, tangent, tangent_p
        )

        # 3. Construct the basis objects
        # This is not slow compared to other parts
        # as it is only involves object creation.
        basis = []
        if _VERBOSE_GENERATE_QUAD_DIFF:
            print('>> Time for converting cross_tangents to polys:', perf_counter() - time0)
            time0 = perf_counter()

        for t2, p2 in zip(cross_exprs_mul_p, cross_polys_mul_p):
            basis += cls.generate(t2, symbols, degree, tangent_p=p2, require_equal=True)

        if _VERBOSE_GENERATE_QUAD_DIFF:
            print('>> Time for generating bases instances:', perf_counter() - time0)
            time0 = perf_counter()

        # 4. Convert polys to the numpy matrix representation.
        mat = _get_matrix_of_quad_diff(tangent_p, degree, quad_diff_order, cls._degree_step, symmetry)

        return basis, mat


class LinearBasisTangentEven(LinearBasisTangent):
    """
    Ensure the degree of each monomial is even.
    """
    _degree_step = 2


def cross_exprs(exprs: List[Expr], symbols: Tuple[Symbol, ...], degree: int) -> List[Expr]:
    """
    Given expressions f1, f2, ..., fn,
    generate all expressions of the form f1^a1 * f2^a2 * ... * fn^an
    bounded by the degree.

    Parameters
    ----------
    exprs: Expr
        A list of sympy expressions.
    symbols: Tuple[Symbol, ...]
        A tuple of symbols.
    degree: int
        The maximum degree of the cross products.

    Returns
    -------
    List[Expr]
        A list of sympy expressions.
    """
    polys = [_.doit().as_poly(symbols) for _ in exprs]
    poly_degrees = [_.total_degree() for _ in polys]

    # remove zero-degree polynomials
    polys = [p for p, d in zip(polys, poly_degrees) if d > 0]
    poly_degrees = [d for d in poly_degrees if d > 0]
    if len(polys) == 0:
        return []

    # find all a1*d1 + a2*d2 + ... + an*dn <= degree
    powers = generate_partitions(poly_degrees, degree, descending=False)
    # map the powers to expressions
    new_exprs = [Mul(*(x**i for x, i in zip(exprs, p))) for p in powers]

    return new_exprs

def quadratic_difference(symbols: Tuple[Symbol, ...]) -> List[Expr]:
    """
    Generate all expressions of the form (ai - aj)^2

    Example
    ========
    >>> from sympy.abc import a, b, c
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
# Fast operations for computing bases
###########################################################

def switchable_lru_cache(maxsize=128, typed=False, enabled=True):
    def decorator(func):
        cached_func = lru_cache(maxsize, typed)(func)
        wrapper = wraps(func)(SwitchableWrapper(func, cached_func))
        wrapper.enabled = enabled
        return wrapper
    return decorator

class SwitchableWrapper:
    def __init__(self, func, cached_func):
        self.func = func
        self.cached_func = cached_func
        self.enabled = True

    def __call__(self, *args, **kwargs):
        if self.enabled:
            return self.cached_func(*args, **kwargs)
        return self.func(*args, **kwargs)
    def enable_cache(self):
        self.enabled = True
    def disable_cache(self):
        self.enabled = False
    def cache_clear(self):
        self.cached_func.cache_clear()
    def clear_cache(self):
        self.cache_clear()

@switchable_lru_cache()
def _get_cross_dmps_of_quad_diff(quad_diff_order: int, tangent_dmp: DMP) -> List[PolyElement]:
    """
    Compute the DMP of polynomials of the form prod((ai - aj)^2) * tangent.

    This is a low-level and optimized function for generating bases,
    which computes polynomials by dynamic programming rather than
    converting expressions to polynomials. It is cached by lru_cache.

    Parameters
    ----------
    quad_diff_order: int
        The maximum degree of the quadratic differences.
    tangent_dmp: DMP
        The sympy polynomial representation (DMP object) of the tangent.
    """
    tangent_dmp = tangent_dmp.rep if isinstance(tangent_dmp, Poly) else tangent_dmp
    nvars = tangent_dmp.lev + 1
    ndiff = nvars * (nvars - 1) // 2
    powers = generate_partitions([2] * ndiff, quad_diff_order, descending=False)
    domain = tangent_dmp.dom

    rng = PolyRing(f'x:{nvars}', domain)
    rng_zero = rng.zero
    smp = rng_zero.new(tangent_dmp.to_dict())

    # polys are the DMPs of (ai - aj)^2 for all i < j
    polys, lst = [None] * ndiff, [0] * nvars
    cnt, lev, one, negtwo = 0, nvars - 1, domain.one, domain.one * -2
    for i in range(nvars):
        for j in range(i+1, nvars):
            coeffs = {}
            lst[i] = 2
            coeffs[tuple(lst)] = one
            lst[i] = 0
            lst[j] = 2
            coeffs[tuple(lst)] = one
            lst[j] = 1
            lst[i] = 1
            coeffs[tuple(lst)] = negtwo
            lst[i] = 0
            lst[j] = 0
            # polys[cnt] = DMP.from_dict(coeffs, lev, domain)
            polys[cnt] = rng_zero.new(coeffs)
            cnt += 1

    # cache = {(0,) * ndiff: tangent_dmp} # tangent_dmp.one(nvars - 1, tangent_dmp.dom)
    cache = {(0,) * ndiff: smp}

    if _VERBOSE_GENERATE_QUAD_DIFF:
        time0 = perf_counter()

    # compute the polynomials of prod((ai - aj)^2) * tangent
    # via dynamic programming
    def _compute_power_with_cache(polys, cache, power):
        cache_p = cache.get(power)
        if cache_p is None:
            first_nonzero_ind = next(i for i, p in enumerate(power) if p)
            reduced_power = (0,) * first_nonzero_ind \
                + (power[first_nonzero_ind]-1,) + power[first_nonzero_ind+1:]
            cache_p = polys[first_nonzero_ind] * \
                _compute_power_with_cache(polys, cache, reduced_power)
            cache[power] = cache_p
        return cache_p

    new_poly_reps = [None] * len(powers)
    for i, power in enumerate(powers):
        new_poly_reps[i] = _compute_power_with_cache(polys, cache, power)

    if _VERBOSE_GENERATE_QUAD_DIFF:
        print('>> Time for computing polys in cross_exprs:', perf_counter() - time0)
        time0 = perf_counter()
    return new_poly_reps


def _get_cross_exprs_and_polys_of_quad_diff(symbols: Tuple[Symbol, ...],
        quad_diff_order: int, tangent: Expr, tangent_p: Poly) -> Tuple[List[Expr], List[PolyElement]]:
    """
    Generate all sympy expressions of the form prod((ai - aj)^2) * tangent and return the polynomials,
    the degree of prod((ai - aj)^2) is bounded by quad_diff_order.

    This is a low-level and optimized function for generating bases,
    which computes polynomials by dynamic programming rather than
    converting expressions to polynomials.

    Parameters
    ----------
    symbols: Tuple[Symbol, ...]
        A tuple of symbols.
    quad_diff_order: int
        The maximum degree of the quadratic differences.
    tangent: Expr
        The sympy expression of the tangent.
    tangent_p: Poly
        The sympy polynomial of the tangent.

    Returns
    ---------
    Tuple[List[Expr], List[PolyElement]]
        A list of sympy expressions and a list of corresponding polynomials.
    """
    # # This is a naive implementation
    # def _naive_implementation():
    #     quad_diff = quadratic_difference(symbols)
    #     exprs = cross_exprs(quad_diff, symbols, quad_diff_order)
    #     exprs_mul_p = [tangent * e for e in exprs]
    #     polys_mul_p = [tangent_p * e.as_poly(symbols) for e in exprs]
    #     # Do not use [e.as_poly(symbols) for e in exprs]
    #     # since we cannot ensure tangent.as_poly(symbols) == tangent_p
    #     return exprs_mul_p, polys_mul_p
    # return _naive_implementation()

    # Faster implementation
    nvars = len(symbols)
    # symbols = sorted(list(symbols), key=lambda x: x.name) # sorting makes rep reordered
    inds = [(i,j) for i in range(nvars) for j in range(nvars) if i < j]
    powers = generate_partitions([2] * (nvars*(nvars-1)//2), quad_diff_order, descending=False)

    exprs = [
        Mul(tangent,
            *((Pow(symbols[i] - symbols[j], 2*p) if p else Integer(1))
                for (i,j), p in zip(inds, power))) for power in powers
    ]

    dmps = _get_cross_dmps_of_quad_diff(quad_diff_order, tangent_p.rep)

    # _new_func, _new_func_arg = Basic.__new__, Poly
    # polys = [_new_func(_new_func_arg) for _ in range(len(exprs))]
    # for new_p, new_rep in zip(polys, dmps):
    #     new_p.rep = new_rep
    #     new_p.gens = symbols

    polys = dmps
    return exprs, polys


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


@switchable_lru_cache()
def _get_reduced_indices(symmetry: MonomialManager, symmetry_base: MonomialManager, degree: int) -> Tuple[np.ndarray, np.ndarray]:
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

    no_symmetry = symmetry_base
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
    if True:
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

@switchable_lru_cache()
def _get_matrix_of_quad_diff(
    tangent_dmp: DMP,
    degree: int,
    quad_diff_order: int,
    step: int,
    symmetry: PermutationGroup
) -> np.ndarray:
    """
    Generate the matrix representation of all bases of the form

        x1^a1 * x2^a2 * ... * xn^an * (x1-x2)^(2b_12) * ... * (xi-xj)^(2b_ij) * tangent

    with total degree == degree and (2b_12 + ... + 2b_ij) <= quad_diff_order and
    a1 % step == a2 % step == ... == an % step == 0.

    This is a low-level and optimized function for generating bases, and is
    cached by lru_cache.

    Parameters
    -----------
    tangent_dmp : DMP
        The sympy polynomial representation (DMP object) of the tangent.
    degree : int
        The total degree of the generated bases.
    quad_diff_order : int
        The maximum degree of the quadratic differences.
    step : int
        The step of the generated bases.
    symmetry : PermutationGroup
        The permutation group of the symmetry.

    Returns
    -----------
    np.ndarray
        The matrix representation of the bases.
    """
    polys = _get_cross_dmps_of_quad_diff(quad_diff_order, tangent_dmp)
    if len(polys) == 0:
        return np.array([], dtype='float')
    if _VERBOSE_GENERATE_QUAD_DIFF:
        time0 = perf_counter()

    _sparse = isinstance(polys[0], PolyElement)
    if _sparse:
        deg = lambda p: max(map(sum, p.keys()), default=0)
        nvars = polys[0].ring.ngens
    else:
        deg = lambda p: p.total_degree()
        nvars = (polys[0].rep if isinstance(polys[0], Poly) else polys[0]).lev + 1


    mat = []
    nvars_of_steps = [step] * nvars
    symmetry = MonomialManager(nvars, symmetry)
    symmetry_base = symmetry.base() # initialize once to use cached properties
    for p in polys:
        degree_comb_mat = generate_partitions(nvars_of_steps, degree - deg(p), equal=True, descending=False)
        degree_comb_mat = np.array(degree_comb_mat, dtype='int32') * step

        submat = _get_matrix_of_lifted_degrees(p, degree_comb_mat, symmetry, symmetry_base, degree)
        if submat.shape[0]:
            mat.append(submat)

    mat = np.vstack(mat) if len(mat) > 0 else np.array([], dtype='float')

    if _VERBOSE_GENERATE_QUAD_DIFF:
        print('>> Time for converting bases to matrix:', perf_counter() - time0)

    return mat


def _get_matrix_of_lifted_degrees(
    poly: Union[DMP, Poly, PolyElement],
    degree_comb_mat: np.ndarray,
    symmetry: MonomialManager,
    symmetry_base: MonomialManager,
    degree: int
) -> np.ndarray:
    """
    Low-level function to convert bases to matrix representation efficiently.

    Parameters
    -----------
    poly : Union[DMP, Poly, PolyElement]
        The sympy polynomial or poly.rep.
    degree_comb_mat : np.ndarray
        A numpy matrix indicating the combinations of powers
        in each variable.
    symmetry : MonomialManager
        The monomial manager for reduced monomials.
    symmetry_base : MonomialManager
        Should be symmetry.base().
    degree : int
        The total degree of the generated bases.
    """

    if degree_comb_mat.shape[0] == 0:
        return np.array([], dtype='float')

    if isinstance(poly, PolyElement):
        nvars = poly.ring.ngens
        deg = lambda p: max(map(sum, p.keys()), default=0)
    else:
        nvars = (poly.rep if isinstance(poly, Poly) else poly).lev + 1
        deg = lambda p: p.total_degree()

    # # This a naive implementation
    # def _naive_implementation():
    #     symbols = [Symbol(f'x{i}') for i in range(nvars)]
    #     mat = [None] * degree_comb_mat.shape[0]
    #     poly_from_dict = Poly.from_dict
    #     p2dict = poly.as_dict()
    #     for mat_ind, power in enumerate(degree_comb_mat):
    #         new_p_dict = dict((tuple_sum(power, k), v) for k, v in p2dict.items())
    #         new_p = poly_from_dict(new_p_dict, symbols)
    #         mat[mat_ind] = symmetry.arraylize_np(new_p, expand_cyc=True)
    #     return np.vstack(mat) if len(mat) > 0 else np.array([], dtype='float')
    # return _naive_implementation()

    # Below is a faster, low-level implementation
    # But it is not equivalent to the naive implementation

    # encoding is a vector dot operation that maps monomials to a single integer
    # e.g. (4,3,2,1) -> 4 + 3*5 + 2*5^2 + 1*5^3 = 4 + 15 + 50 + 125 = 194
    # i.e. (4,3,2,1) * (1,5,25,125) = 194
    # Since it is linear, encoding of sum of monomials is the sum of encodings.
    # Don't worry about overflow, if there are too many monomials, then it is
    # impossible to solve the problem anyway.
    _DTYPE = 'int32'
    encoding = np.array([(degree + 1)**i for i in range(nvars)], dtype=_DTYPE)

    source_monoms = symmetry_base.inv_monoms(deg(poly))  # a list of monomials
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
    inv_target_monoms, multiplicity = _get_reduced_indices(symmetry, symmetry_base, degree)

    new_monoms = inv_target_monoms[new_monoms] # map to the indices of target monoms

    poly_vec = symmetry_base.arraylize_np(poly) #, expand_cyc=True)

    new_mat = _count_contribution_of_monoms(new_monoms, poly_vec, length_of_target)
    new_mat = new_mat * multiplicity.reshape((1, -1))

    return new_mat
