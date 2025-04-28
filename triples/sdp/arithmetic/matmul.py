"""
This module provides an implementation of rational number matrices and their  
arithmetic operations, aiming to provide a more efficient interface to  
using sympy.Rational or numpy for matrix computations.
"""

from time import time
from typing import List, Tuple, Union, Optional

from numpy import iinfo as np_iinfo
from numpy import ndarray, int64, isnan, inf, unique
import sympy as sp
from sympy.matrices import MutableDenseMatrix as Matrix

from .matop import (
    is_zz_qq_mat, vec2mat, reshape, primitive,
    rep_matrix_to_numpy, rep_matrix_from_numpy
)

_INT32_MAX = np_iinfo('int32').max # 2147483647
_INT64_MAX = np_iinfo('int64').max # 9223372036854775807

# for dev purpose only
_VERBOSE_MATMUL_MULTIPLE = False


def matadd(A: Union[Matrix, ndarray], B: Union[Matrix, ndarray]):
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


def matmul(A: Union[Matrix, ndarray], B: Union[Matrix, ndarray],
        return_shape: Optional[Tuple[int, int]] = None) -> Union[Matrix, ndarray]:
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
    if isinstance(A, ndarray):
        A = rep_matrix_from_numpy(A)
    if isinstance(B, ndarray):
        B = rep_matrix_from_numpy(B)

    return_shape = return_shape or (A.shape[0], B.shape[-1])
    if A.shape[0] == 0 or B.shape[0] == 0 or B.shape[-1] == 0:
        return sp.zeros(*return_shape)
    A0, B0 = A, B

    def default(A0, B0):
        return reshape(A0 @ B0, return_shape)
    
    if not (is_zz_qq_mat(A) and is_zz_qq_mat(B)):
        return default(A0, B0)

    try:
        q1, A = primitive(A._rep)
        A = rep_matrix_to_numpy(A, dtype=int64)
        _MAXA = abs(A).max()
        if isnan(_MAXA) or _MAXA == inf or _MAXA > _INT64_MAX:
            raise OverflowError

        q2, B = primitive(B._rep)
        B = rep_matrix_to_numpy(B, dtype=int64)
        _MAXB = abs(B).max()
        if isnan(_MAXB) or _MAXB == inf or _MAXB > _INT64_MAX or int(_MAXA) * int(_MAXB) * B.shape[0] > _INT64_MAX:
            raise OverflowError
    except OverflowError:
        return default(A0, B0)

    q1q2 = q1 * q2
    C = (A @ B).reshape(return_shape)
    C = rep_matrix_from_numpy(C) * q1q2
    return C


def matmul_multiple(A: Union[Matrix, ndarray], B: Union[Matrix, ndarray]) -> Union[Matrix, ndarray]:
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
    if isinstance(A, ndarray):
        A = rep_matrix_from_numpy(A)
    if isinstance(B, ndarray):
        B = rep_matrix_from_numpy(B)

    if A.shape[0] == 0 or B.shape[0] == 0 or B.shape[1] == 0:
        return sp.zeros(A.shape[0], B.shape[0]*B.shape[1])

    A0, B0 = A, B
    def default(A, B):
        eq_mat = []
        for i in range(A.shape[0]):
            Aij = vec2mat(A[i,:])
            eq = matmul(Aij, B, return_shape = (1, Aij.shape[0]*B.shape[1]))
            eq_mat.append(eq)
        eq_mat = Matrix.vstack(*eq_mat)
        return eq_mat

    if _VERBOSE_MATMUL_MULTIPLE:
        print('MatmulMultiple A B shape =', A.shape, B.shape)
        sparsity = lambda x: len(list(x.iter_values())) / (x.shape[0] * x.shape[1])
        print('>> Sparsity A B =', sparsity(A), sparsity(B))
        time0 = time()


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

        q2, B = primitive(B._rep)
        B = rep_matrix_to_numpy(B, dtype=int64)
        _MAXB = abs(B).max()
        if isnan(_MAXB) or _MAXB == inf or _MAXB > _INT64_MAX or int(_MAXA) * int(_MAXB) * n > _INT64_MAX:
            raise OverflowError
    except OverflowError:
        return default(A0, B0)

    A = A.reshape((N, n, n))

    if _VERBOSE_MATMUL_MULTIPLE:
        print('>> Time for casting to numpy:', time() - time0)
        time0 = time()

    C = (A @ B).reshape((N, n*m))

    if _VERBOSE_MATMUL_MULTIPLE:
        print('>> Time for numpy matmul:', time() - time0) # very fast (can be ignored)
        time0 = time()

    q1q2 = q1 * q2
    C = rep_matrix_from_numpy(C) * q1q2

    if _VERBOSE_MATMUL_MULTIPLE:
        print('>> Time for casting to sympy:', time() - time0) # < 1 sec over (800*8000)
        time0 = time()

    return C


def symmetric_bilinear(U: Union[Matrix, ndarray], A: Union[Matrix, ndarray], is_A_vec: bool = False,
        return_shape: Tuple[int, int] = None) -> Union[Matrix, ndarray]:
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
    M = matmul(U.T, matmul(A, U))
    if return_shape is not None:
        return reshape(M, return_shape)
    return M


def symmetric_bilinear_multiple(U: Union[Matrix, ndarray], A: Union[Matrix, ndarray]) -> Union[Matrix, ndarray]:
    """
    Perform multiple symmetric bilinear products.
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
    if isinstance(A, ndarray):
        A = rep_matrix_from_numpy(A)
    if isinstance(U, ndarray):
        U = rep_matrix_from_numpy(U)

    A0, U0 = A, U
    def default(A, U):
        eq_mat = [0] * A.shape[0]
        for i in range(A.shape[0]):
            # Aij = vec2mat(space[i,:])
            # eq = U.T * Aij * U
            eq = symmetric_bilinear(U, A[i,:], is_A_vec = True, return_shape = (1, U.shape[1]**2))
            eq_mat[i] = eq
        eq_mat = Matrix.vstack(*eq_mat)
        return eq_mat

    N = A.shape[0]
    n, m = U.shape
    if N == 0 or n == 0 or m == 0:
        return Matrix.zeros(N, m**2)

    if not (is_zz_qq_mat(A) and is_zz_qq_mat(U)):
        return default(A0, U0)
    # return default(A0, U0)

    try:
        q1, A = primitive(A._rep)
        A = rep_matrix_to_numpy(A, dtype=int64)
        _MAXA = abs(A).max()
        if isnan(_MAXA) or _MAXA == inf or _MAXA > _INT64_MAX:
            raise OverflowError

        q2, U = primitive(U._rep)
        U = rep_matrix_to_numpy(U, dtype=int64)
        _MAXU = abs(U).max()
        if isnan(_MAXU) or _MAXU == inf or _MAXU > _INT64_MAX or int(_MAXA) * int(_MAXU)**2 * n**2 > _INT64_MAX:
            raise OverflowError
    except OverflowError:
        return default(A0, U0)

    A = A.reshape((N, n, n))
    C = (U.T @ A @ U).reshape((N, m**2))

    if _VERBOSE_MATMUL_MULTIPLE:
        time0 = time()

    q1q22 = q1 * q2**2
    C = rep_matrix_from_numpy(C) * q1q22

    if _VERBOSE_MATMUL_MULTIPLE:
        print('>> Time for casting to sympy Bilinear:', time() - time0)

    return C