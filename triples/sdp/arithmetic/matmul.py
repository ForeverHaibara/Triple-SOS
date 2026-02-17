"""
This module provides an implementation of rational number matrices and their
arithmetic operations, aiming to provide a more efficient interface to
using sympy.Rational or numpy for matrix computations.
"""

from time import perf_counter
from typing import List, Tuple, Union, Optional, Callable, overload

from numpy import ndarray, int64, isnan, inf
from numpy import iinfo as np_iinfo
from numpy import any as np_any
from numpy import where as np_where
from sympy.matrices import MutableDenseMatrix as Matrix
from sympy.matrices.repmatrix import RepMatrix
from sympy import MatrixBase

from .matop import (
    ArithmeticTimeout,
    SDM, DDM, DFM,
    is_zz_qq_mat, vec2mat, reshape, primitive,
    rep_matrix_to_numpy, rep_matrix_from_numpy
)

_INT32_MAX = np_iinfo('int32').max # 2147483647
_INT64_MAX = np_iinfo('int64').max # 9223372036854775807

# for dev purpose only
_VERBOSE_MATMUL_MULTIPLE = False
_IS_STANDARD_INT64 = (_INT64_MAX == 9223372036854775807)

@overload
def matadd(A: MatrixBase, B: MatrixBase) -> MatrixBase: ...
@overload
def matadd(A: ndarray, B: ndarray) -> ndarray: ...
@overload
def matadd(A: MatrixBase, B: ndarray) -> MatrixBase: ...
@overload
def matadd(A: ndarray, B: MatrixBase) -> MatrixBase: ...

def matadd(A, B):
    """
    Compute A + B with proper data types casting.
    """
    if isinstance(A, ndarray) and isinstance(B, ndarray):
        return A + B
    if isinstance(A, ndarray):
        A = rep_matrix_from_numpy(A)
    if isinstance(B, ndarray):
        B = rep_matrix_from_numpy(B)
    return A + B

def matlshift(A: Matrix, B: int) -> Matrix:
    """
    Left shift the matrix A by B bits.

    Parameters
    ----------
    A: Matrix
        Matrix A
    B: int
        Number of bits to shift

    Returns
    -------
    Matrix:
        Result of A << B.

    Examples
    --------
    >>> A = Matrix([[1, 2], [3, 4]])
    >>> matlshift(A, 1)
    Matrix([
    [2, 4],
    [6, 8]])
    """
    if not isinstance(A, RepMatrix):
        return A * (2**B)
    rep = A._rep.rep
    dom = rep.domain
    if not dom.is_ZZ:
        return A * (2**B)
    if isinstance(rep, SDM):
        rep = SDM({i: {j: v << B for j, v in row.items()}
                   for i, row in rep.items()}, A.shape, dom)
    elif isinstance(rep, DDM):
        rep = DDM([[v << B for v in row] for row in rep], A.shape, dom)
    elif isinstance(rep, DFM):
        rep = rep.mul(2**B)
    else:
        rep = rep * (2**B)
    return A._fromrep(A._rep.from_rep(rep))


@overload
def matmul(A: MatrixBase, B: MatrixBase, return_shape: Optional[Tuple[int, int]] = None,
        time_limit: Optional[Union[Callable, float]] = None) -> Matrix: ...
@overload
def matmul(A: ndarray, B: ndarray, return_shape: Optional[Tuple[int, int]] = None,
        time_limit: Optional[Union[Callable, float]] = None) -> ndarray: ...
@overload
def matmul(A: MatrixBase, B: ndarray, return_shape: Optional[Tuple[int, int]] = None,
        time_limit: Optional[Union[Callable, float]] = None) -> Matrix: ...
@overload
def matmul(A: ndarray, B: MatrixBase, return_shape: Optional[Tuple[int, int]] = None,
        time_limit: Optional[Union[Callable, float]] = None) -> Matrix: ...

def matmul(A, B, return_shape=None, time_limit=None):
    """
    Fast, low-level implementation of symbolic matrix multiplication.
    When A and B are both rational matrices, it calls NumPy to compute the result.
    Otherwise, it falls back to the default method.

    Parameters
    ----------
    A: Matrix or ndarray
        Matrix A
    B: Matrix or ndarray
        Matrix B
    return_shape: Tuple[int, int]
        Shape of the result. If not specified, it will be inferred.
        If provided, it will be reshaped.

    Returns
    -------
    Matrix:
        Result of A @ B.

    Examples
    --------
    >>> A = Matrix([[1, 2], [3, 4]])
    >>> B = Matrix([[5, 6], [7, 8]])
    >>> matmul(A, B)
    Matrix([
    [19, 22],
    [43, 50]])
    >>> matmul(A, B, return_shape=(1, 4))
    Matrix([[19, 22, 43, 50]])

    If both A and B are numpy arrays, the return type will be numpy array.
    Overflows and other numerical issues are currently unhandled.
    >>> import numpy as np
    >>> A = np.array([[1, 2], [3, 4]])
    >>> B = np.array([[5, 6], [7, 8]])
    >>> matmul(A, B)
    array([[19, 22],
           [43, 50]])

    If one of A and B is a numpy array and the other is a sympy matrix,
    the numpy array will be casted to a sympy matrix.
    >>> B = Matrix([[5, 6], [7, 8]])
    >>> matmul(A, B)
    Matrix([
    [19, 22],
    [43, 50]])
    >>> matmul(A.astype(float), B)
    Matrix([
    [19.0, 22.0],
    [43.0, 50.0]])
    """
    if isinstance(A, ndarray) and isinstance(B, ndarray):
        C = A @ B
        if return_shape is not None:
            C = C.reshape(return_shape)
        return C
    time_limit = ArithmeticTimeout.make_checker(time_limit)
    if isinstance(A, ndarray):
        A = rep_matrix_from_numpy(A)
    if isinstance(B, ndarray):
        B = rep_matrix_from_numpy(B)
    time_limit()

    return_shape = return_shape or (A.shape[0], B.shape[-1])
    if A.shape[0] == 0 or B.shape[0] == 0 or B.shape[-1] == 0:
        return Matrix.zeros(*return_shape)
    A0, B0 = A, B

    def default(A0, B0):
        return reshape(A0 @ B0, return_shape)

    if not (is_zz_qq_mat(A) and is_zz_qq_mat(B)):
        return default(A0, B0)

    try:
        q1, q2, _MAXA, _MAXB = 0, 0, 0, 0
        q1, A = primitive(A._rep)
        A = rep_matrix_to_numpy(A, dtype=int64)
        _MAXA = abs(A).max()
        if isnan(_MAXA) or _MAXA == inf or _MAXA > _INT64_MAX:
            raise OverflowError
        time_limit()

        q2, B = primitive(B._rep)
        B = rep_matrix_to_numpy(B, dtype=int64)
        _MAXB = abs(B).max()
        if isnan(_MAXB) or _MAXB == inf or _MAXB > _INT64_MAX or int(_MAXA) * int(_MAXB) * B.shape[0] > _INT64_MAX:
            raise OverflowError
        time_limit()
    except OverflowError:
        # print(f'Default {A0.shape} * {B0.shape}, q1 = {q1}, q2 = {q2}, MAXA = {_MAXA}, MAXB = {_MAXB}')
        return default(A0, B0)

    q1q2 = q1 * q2
    C = (A @ B).reshape(return_shape)
    time_limit()
    C = rep_matrix_from_numpy(C) * q1q2
    return C


@overload
def matmul_multiple(A: MatrixBase, B: MatrixBase,
    time_limit: Optional[Union[Callable, float]] = None) -> Matrix: ...
@overload
def matmul_multiple(A: ndarray, B: ndarray,
    time_limit: Optional[Union[Callable, float]] = None) -> ndarray: ...
@overload
def matmul_multiple(A: MatrixBase, B: ndarray,
    time_limit: Optional[Union[Callable, float]] = None) -> Matrix: ...
@overload
def matmul_multiple(A: ndarray, B: MatrixBase,
    time_limit: Optional[Union[Callable, float]] = None) -> Matrix: ...

def matmul_multiple(A, B, time_limit=None):
    """
    Perform multiple matrix multiplications. This can be regarded as a 3-dim tensor multiplication.
    Assume A has shape N x (n^2) and B has shape n x m, then the result has shape N x (n*m).

    Parameters
    ----------
    A: Matrix or ndarray
        Matrix A
    B: Matrix or ndarray
        Matrix B

    Examples
    --------
    >>> from sympy import Matrix
    >>> A1, A2, B = Matrix([1,2,3,4]), Matrix([5,6,7,8]), Matrix([[9,10],[11,12]])
    >>> A = Matrix.vstack(A1.T, A2.T)
    >>> matmul_multiple(A, B)
    Matrix([
    [ 31,  34,  71,  78],
    [111, 122, 151, 166]])

    If both A and B are numpy arrays, the return type will be numpy array.
    Overflows and other numerical issues are currently unhandled.
    >>> import numpy as np
    >>> A = np.array([[1,2,3,4],[5,6,7,8]])
    >>> B = np.array([[9,10],[11,12]])
    >>> matmul_multiple(A, B)
    array([[ 31,  34,  71,  78],
           [111, 122, 151, 166]])

    If one of A and B is a numpy array and the other is a sympy matrix,
    the numpy array will be casted to a sympy matrix.
    >>> B = Matrix([[9,10],[11,12]])
    >>> matmul_multiple(A, B)
    Matrix([
    [ 31,  34,  71,  78],
    [111, 122, 151, 166]])
    >>> matmul_multiple(A.astype(float), B)
    Matrix([
    [ 31.0,  34.0,  71.0,  78.0],
    [111.0, 122.0, 151.0, 166.0]])
    """
    if isinstance(A, ndarray) and isinstance(B, ndarray):
        N, n = A.shape[0], B.shape[0]
        return (A.reshape(N, n, n) @ B).reshape(N, n*B.shape[1])
    time_limit = ArithmeticTimeout.make_checker(time_limit)
    if isinstance(A, ndarray):
        A = rep_matrix_from_numpy(A)
    if isinstance(B, ndarray):
        B = rep_matrix_from_numpy(B)
    time_limit()

    if A.shape[0] == 0 or B.shape[0] == 0 or B.shape[1] == 0:
        return Matrix.zeros(A.shape[0], B.shape[0]*B.shape[1])

    A0, B0 = A, B
    def default(A, B):
        eq_mat = []
        for i in range(A.shape[0]):
            Aij = vec2mat(A[i,:])
            eq = matmul(Aij, B, return_shape = (1, Aij.shape[0]*B.shape[1]), time_limit=time_limit)
            eq_mat.append(eq)
        eq_mat = Matrix.vstack(*eq_mat)
        return eq_mat

    if _VERBOSE_MATMUL_MULTIPLE:
        print('MatmulMultiple A B shape =', A.shape, B.shape)
        sparsity = lambda x: len(list(x.iter_values())) / (x.shape[0] * x.shape[1])
        print('>> Sparsity A B =', sparsity(A), sparsity(B))
        time0 = perf_counter()


    if not (is_zz_qq_mat(A) and is_zz_qq_mat(B)):
        # fallback to defaulted method
        return default(A0, B0)

    N = A.shape[0]
    n, m = B.shape

    try:
        q1, A = primitive(A._rep)
        A = rep_matrix_to_numpy(A, dtype=int64)
        _MAXA = abs(A).max()
        if isnan(_MAXA) or _MAXA == inf or _MAXA > _INT64_MAX:
            raise OverflowError
        time_limit()

        q2, B = primitive(B._rep)
        B = rep_matrix_to_numpy(B, dtype=int64)
        _MAXB = abs(B).max()
        if isnan(_MAXB) or _MAXB == inf or _MAXB > _INT64_MAX \
                or int(_MAXA) * int(_MAXB) * n > _INT64_MAX:
            raise OverflowError
        time_limit()
    except OverflowError:
        return default(A0, B0)

    A = A.reshape((N, n, n))

    if _VERBOSE_MATMUL_MULTIPLE:
        print('>> Time for casting to numpy:', perf_counter() - time0)
        time0 = perf_counter()

    C = (A @ B).reshape((N, n*m))
    time_limit()

    if _VERBOSE_MATMUL_MULTIPLE:
        print('>> Time for numpy matmul:', perf_counter() - time0) # very fast (can be ignored)
        time0 = perf_counter()

    q1q2 = q1 * q2
    C = rep_matrix_from_numpy(C) * q1q2

    if _VERBOSE_MATMUL_MULTIPLE:
        print('>> Time for casting to sympy:', perf_counter() - time0) # < 1 sec over (800*8000)
        time0 = perf_counter()

    return C

@overload
def symmetric_bilinear(U: MatrixBase, A: MatrixBase, is_A_vec: bool = False,
    return_shape: Optional[Tuple[int, int]] = None,
    time_limit: Optional[Union[Callable, float]] = None) -> Matrix: ...
@overload
def symmetric_bilinear(U: ndarray, A: ndarray, is_A_vec: bool = False,
    return_shape: Optional[Tuple[int, int]] = None,
    time_limit: Optional[Union[Callable, float]] = None) -> ndarray: ...
@overload
def symmetric_bilinear(U: MatrixBase, A: ndarray, is_A_vec: bool = False,
    return_shape: Optional[Tuple[int, int]] = None,
    time_limit: Optional[Union[Callable, float]] = None) -> Matrix: ...
@overload
def symmetric_bilinear(U: ndarray, A: MatrixBase, is_A_vec: bool = False,
    return_shape: Optional[Tuple[int, int]] = None,
    time_limit: Optional[Union[Callable, float]] = None) -> Matrix: ...

def symmetric_bilinear(U, A, is_A_vec=False, return_shape=None, time_limit=None):
    """
    Compute U.T * A * U efficiently.
    Assume U is n x m, U.T is m x n and A is n x n. The result is m x m.

    Complexity: Direct matmul is O(mn(n+m)).
    When A is sparse with k nonzeros, the complexity is theoretically O(2m^2k).

    Parameters
    ----------
    U: Matrix or ndarray
        Matrix U
    A: Matrix or ndarray
        Matrix A
    is_A_vec: bool
        Whether A is stored as a vector. If True, it first converts A to a symmetric matrix.
    return_shape: Optional[Tuple[int, int]]
        Shape of the returned matrix.

    Examples
    --------
    >>> from sympy import Matrix
    >>> U, A = Matrix([[1, 2], [3, 4]]), Matrix([[5, 6], [7, 8]])
    >>> symmetric_bilinear(U, A)
    Matrix([
    [116, 172],
    [170, 252]])
    >>> U.T @ A @ U
    Matrix([
    [116, 172],
    [170, 252]])

    If both U and A are numpy arrays, the return type will be numpy array.
    >>> import numpy as np
    >>> U, A = np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])
    >>> symmetric_bilinear(U, A)
    array([[116, 172],
           [170, 252]])

    If one of U and A is a numpy array and the other is a sympy matrix,
    the numpy array will be casted to a sympy matrix.
    >>> U, A = Matrix([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])
    >>> symmetric_bilinear(U, A)
    Matrix([
    [116, 172],
    [170, 252]])
    >>> symmetric_bilinear(U, A.astype(float))
    Matrix([
    [116.0, 172.0],
    [170.0, 252.0]])
    """
    if is_A_vec:
        A = vec2mat(A)
    M = matmul(U.T, matmul(A, U, time_limit=time_limit), time_limit=time_limit)
    if return_shape is not None:
        return reshape(M, return_shape)
    return M


@overload
def symmetric_bilinear_multiple(U: MatrixBase, A: MatrixBase,
    time_limit: Optional[Union[Callable, float]] = None) -> Matrix: ...
@overload
def symmetric_bilinear_multiple(U: ndarray, A: ndarray,
    time_limit: Optional[Union[Callable, float]] = None) -> ndarray: ...
@overload
def symmetric_bilinear_multiple(U: MatrixBase, A: ndarray,
    time_limit: Optional[Union[Callable, float]] = None) -> Matrix: ...
@overload
def symmetric_bilinear_multiple(U: ndarray, A: MatrixBase,
    time_limit: Optional[Union[Callable, float]] = None) -> Matrix: ...

def symmetric_bilinear_multiple(U, A, time_limit=None):
    """
    Perform multiple symmetric bilinear products U^T * Ai * U.
    Assume U has shape n x m and A has shape N x (n^2), then the result has shape N x m^2.

    Parameters
    ----------
    U: Matrix or ndarray
        Matrix U
    A: Matrix or ndarray
        Matrix A
    """
    if isinstance(A, ndarray) and isinstance(U, ndarray):
        N, n = A.shape[0], U.shape[0]
        return (U.T @ A.reshape(N, n, n) @ U).reshape(N, U.shape[1]**2)
    time_limit = ArithmeticTimeout.make_checker(time_limit)
    if isinstance(A, ndarray):
        A = rep_matrix_from_numpy(A)
    if isinstance(U, ndarray):
        U = rep_matrix_from_numpy(U)

    time_limit()

    A0, U0 = A, U
    def default(A, U):
        if _VERBOSE_MATMUL_MULTIPLE:
            time0 = perf_counter()
        eq_mat = [0] * A.shape[0]
        for i in range(A.shape[0]):
            # Aij = vec2mat(space[i,:])
            # eq = U.T * Aij * U
            eq = symmetric_bilinear(U, A[i,:], is_A_vec = True,
                    return_shape = (1, U.shape[1]**2), time_limit = time_limit)
            eq_mat[i] = eq
        eq_mat = Matrix.vstack(*eq_mat)
        if _VERBOSE_MATMUL_MULTIPLE:
            print(f">>> Default Symmetric Bilinear {U.shape}.T * {A.shape} * {U.shape}"\
                  + f", time = {perf_counter() - time0}")
        return eq_mat

    N = A.shape[0]
    n, m = U.shape
    if N == 0 or n == 0 or m == 0:
        return Matrix.zeros(N, m**2)

    if not (is_zz_qq_mat(A) and is_zz_qq_mat(U)):
        return default(A0, U0)
    # return default(A0, U0)

    try:
        q1, q2, _MAXA, _MAXU = 0, 0, 0, 0
        q1, A = primitive(A._rep)
        A = rep_matrix_to_numpy(A, dtype=int64)
        _MAXA = abs(A).max()
        if isnan(_MAXA) or _MAXA == inf or _MAXA > _INT64_MAX:
            raise OverflowError
        time_limit()

        q2, U = primitive(U._rep)
        U = rep_matrix_to_numpy(U, dtype=int64)
        _MAXU = abs(U).max()
        if isnan(_MAXU) or _MAXU == inf or _MAXU > _INT64_MAX:
            raise OverflowError
        time_limit()


        if _VERBOSE_MATMUL_MULTIPLE:
            time0 = perf_counter()
        if int(_MAXA) * int(_MAXU)**2 * n**2 <= _INT64_MAX:
            # direct numpy multiplication

            A = A.reshape((N, n, n))
            C = (U.T @ A @ U).reshape((N, m**2))
            time_limit()
            C = rep_matrix_from_numpy(C)

            if _VERBOSE_MATMUL_MULTIPLE:
                print(f">> Time for symmetric bilinear multiple {U.shape}.T * {A.shape} * {U.shape}"\
                      + f", time = {perf_counter() - time0}")
        else:
            C = _symmetric_bilinear_multiple_by_level(U, A)
            if _VERBOSE_MATMUL_MULTIPLE:
                print(f">> Time for symmetric bilinear multiple {U.shape}.T * {A.shape} * {U.shape} (L)"\
                      + f", time = {perf_counter() - time0}")

        q1q22 = q1 * q2**2
        C = C * q1q22

        return C
    except OverflowError:
        # print(f"Default {U0.shape}.T * {A0.shape} * {U0.shape}"\
        #       + f", q1 = {q1}, q2 = {q2}, MAXA = {_MAXA}, MAXU = {_MAXU}")
        return default(A0, U0)


def _decompose_int64_to_level_digits(arr: ndarray, level: int) -> List[ndarray]:
    """
    Split an int64 array into a list of arrays such that
    `arr = sum(a[i] * 2**(i*level))`

    Parameters
    ----------
    arr: ndarray
        Array to be decomposed.
    level: int
        Level of decomposition.

    Returns
    -------
    List[ndarray]
        List of arrays such that `arr = sum(a[i] * 2**(i*level))`

    Examples
    --------
    >>> import numpy as np
    >>> A = np.random.randint(-2**31, 2**31, size=(4,)).astype(np.int64)*2**31
    >>> B = _decompose_int64_to_level_digits(A, 16)
    >>> bool(np.all(A == B[0] + B[1]*2**16 + B[2]*2**32 + B[3]*2**48))
    True
    """
    assert arr.dtype == int64
    assert 1 <= level <= 63
    max_total_bits = 64
    digits = []
    shift = 0

    shift_info = []
    while shift < max_total_bits:
        remaining_bits = max_total_bits - shift
        current_level = min(level, remaining_bits)
        shift_info.append((shift, current_level))
        shift += level

    for idx, (shift, current_level) in enumerate(shift_info):
        is_highest_segment = (idx == len(shift_info) - 1)
        mask = (1 << current_level) - 1

        digit = (arr >> shift) & mask

        if is_highest_segment:
            sign_bit = 1 << (current_level - 1)
            digit = np_where(digit & sign_bit, digit - (1 << current_level), digit)

        digits.append(digit.astype(int64))

    return digits


def _symmetric_bilinear_multiple_by_level(U: ndarray, A: ndarray) -> Matrix:
    """
    Perform multiple symmetric bilinear products U^T * Ai * U
    where U and A are numpy int64 matrices.
    """
    if not _IS_STANDARD_INT64:
        raise OverflowError

    N = A.shape[0]
    n, m = U.shape

    level = 0
    for n_chunks in range(4, 9):
        level = 63//n_chunks + 1
        if 2**(3*level) * n**2 <= _INT64_MAX:
            break
    else:
        # when n is EXTREMELY large
        raise OverflowError

    A = A.reshape((N, n, n))

    level_u = level
    level_a = level

    levels_u = [level_u*i for i in range(63//level_u + 1)]
    levels_a = [level_a*i for i in range(63//level_a + 1)]

    parts_u = _decompose_int64_to_level_digits(U, level_u)
    parts_a = _decompose_int64_to_level_digits(A, level_a)

    parts_u = [(u if np_any(u) else None) for u in parts_u]
    parts_a = [(a if np_any(a) else None) for a in parts_a]

    shifts = [[] for _ in range(2*max(levels_u) + max(levels_a) + 1)]
    for lu1, u1 in zip(levels_u, parts_u):
        if u1 is None:
            continue
        for lu2, u2 in zip(levels_u, parts_u):
            if u2 is None:
                continue
            for la, a in zip(levels_a, parts_a):
                if a is None:
                    continue
                C = (u1.T @ a @ u2).reshape((N, m**2))
                shifts[lu1 + lu2 + la].append(C)

    result = None
    for shift, C_list in enumerate(shifts):
        if not C_list:
            continue
        if 2**(3*level) * n**2 * len(C_list) > _INT64_MAX:
            C_list = [rep_matrix_from_numpy(C) for C in C_list]
            C = sum(C_list[1:], start=C_list[0])
        else:
            C = sum(C_list[1:], start=C_list[0])
            C = rep_matrix_from_numpy(C)

        C = matlshift(C, shift)
        if result is None:
            result = C
        else:
            result = result + C

    if result is None:
        return Matrix.zeros(N, m**2)

    return result
