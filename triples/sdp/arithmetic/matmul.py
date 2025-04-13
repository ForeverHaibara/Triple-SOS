# arithmetic.py  
#  
# This module provides an implementation of rational number matrices and their  
# arithmetic operations, aiming to provide a more efficient alternative to  
# using sympy.Rational for matrix computations.  
#
# NOTE: Using sympy.Matrix to create a new object is very slow.
#       Always build from rep (SDM, DDM, DFM classes) instead.
#       SDM -> DomainMatrix -> MutableDenseMatrix
# TODO: compare the performance of the two implementations

from time import time
from typing import List, Tuple, Union, Optional

from numpy import iinfo as np_iinfo
from numpy import isnan, inf, unique
import sympy as sp
from sympy.matrices import MutableDenseMatrix as Matrix

from .matop import (
    is_zz_qq_mat, vec2mat, primitive, _cast_sympy_matrix_to_numpy,
    rep_matrix_from_numpy
)

_INT32_MAX = np_iinfo('int32').max # 2147483647
_INT64_MAX = np_iinfo('int64').max # 9223372036854775807

# for dev purpose only
_VERBOSE_MATMUL_MULTIPLE = False


def matmul(A: Matrix, B: Matrix, return_shape = None) -> Matrix:
    """
    Fast, low-level implementation of symbolic matrix multiplication.
    When A and B are both rational matrices, it calls NumPy to compute the result.
    Otherwise, it falls back to the default method.

    Parameters
    ----------
    A: Matrix
        Matrix A
    B: Matrix
        Matrix B
    return_shape: Tuple[int, int]
        Shape of the result. If not specified, it will be inferred.
    """
    if A.shape[0] == 0 or B.shape[0] == 0 or B.shape[1] == 0:
        return_shape = return_shape or (A.shape[0], B.shape[1])
        return sp.zeros(*return_shape)
    A0, B0 = A, B

    def default(A0, B0):
        AB = A0 * B0
        if return_shape: AB = AB.reshape(*return_shape)
        return AB
    
    if not (is_zz_qq_mat(A) and is_zz_qq_mat(B)):
        return default(A0, B0)

    q1, A = primitive(A._rep)
    try:
        A = _cast_sympy_matrix_to_numpy(A, dtype = 'int64')
    except OverflowError:
        return default(A0, B0)
    _MAXA = abs(A).max()
    if isnan(_MAXA) or _MAXA == inf or _MAXA > _INT64_MAX:
        return default(A0, B0)

    q2, B = primitive(B._rep)
    try:
        B = _cast_sympy_matrix_to_numpy(B, dtype = 'int64')
    except OverflowError:
        return default(A0, B0)
    _MAXB = abs(B).max()
    if isnan(_MAXB) or _MAXB == inf or _MAXB > _INT64_MAX or int(_MAXA) * int(_MAXB) * B.shape[0] > _INT64_MAX:
        return default(A0, B0)

    q1q2 = q1 * q2
    return_shape = return_shape or (A0.shape[0], B0.shape[1])
    C = (A @ B).reshape(return_shape)
    C = rep_matrix_from_numpy(C) * q1q2
    return C


def matmul_multiple(A: Matrix, B: Matrix) -> Matrix:
    """
    Perform multiple matrix multiplications. This can be regarded as a 3-dim tensor multiplication.
    Assume A has shape N x (n^2) and B has shape n x m, then the result has shape N x (n*m).

    Parameters
    ----------
    A: Matrix
        Matrix A
    B: Matrix
        Matrix B
    """
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

    q1, A = primitive(A._rep)
    try:
        A = _cast_sympy_matrix_to_numpy(A, dtype = 'int64')
    except OverflowError:
        return default(A0, B0)
    _MAXA = abs(A).max()
    if isnan(_MAXA) or _MAXA == inf or _MAXA > _INT64_MAX:
        return default(A0, B0)

    q2, B = primitive(B._rep)
    try:
        B = _cast_sympy_matrix_to_numpy(B, dtype = 'int64')
    except OverflowError:
        return default(A0, B0)
    _MAXB = abs(B).max()
    if isnan(_MAXB) or _MAXB == inf or _MAXB > _INT64_MAX or int(_MAXA) * int(_MAXB) * n > _INT64_MAX:
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


def symmetric_bilinear(U: Matrix, A: Matrix, is_A_vec: bool = False, return_shape = None):
    """
    Compute U.T * A * U efficiently.
    Assume U is n x m, U.T is m x n and A is n x n. The result is m x m.

    Complexity: Kronecker product is O(m^2n^2), and using direct matmul is O(mn(n+m)).
    When A is sparse with k nonzeros, the complexity is O(2m^2k).

    Parameters
    ----------
    U: Matrix
        Matrix U
    A: Matrix
        Matrix A
    is_A_vec: bool
        Whether A is stored as a vector. If True, it first converts A to a matrix.
    return_shape: Optional[Tuple[int, int]]
        Shape of the returned matrix.
    """
    # n, m = U.shape
    
    if is_A_vec:
        A = vec2mat(A)
    M = matmul(U.T, matmul(A, U))
    # M = U.T * A * U

    if return_shape is not None:
        return M.reshape(return_shape[0], return_shape[1])
    return M


def symmetric_bilinear_multiple(U: Matrix, A: Matrix) -> Matrix:
    """
    Perform multiple symmetric bilinear products.
    Assume U has shape n x m and A has shape N x (n^2), then the result has shape N x m^2.

    Parameters
    ----------
    U: Matrix
        Matrix U
    A: Matrix
        Matrix A
    """
    A0, U0 = A, U
    def default(A, U):
        eq_mat = []
        for i in range(A.shape[0]):
            # Aij = vec2mat(space[i,:])
            # eq = U.T * Aij * U
            eq = symmetric_bilinear(U, A[i,:], is_A_vec = True, return_shape = (1, U.shape[1]**2))
            eq_mat.append(eq)
        eq_mat = Matrix.vstack(*eq_mat)
        return eq_mat

    N = A.shape[0]
    if N == 0:
        return Matrix.zeros(0, U.shape[1]**2)

    if not (is_zz_qq_mat(A) and is_zz_qq_mat(U)):
        return default(A0, U0)
    # return default(A0, U0)
    n, m = U.shape


    q1, A = primitive(A._rep)
    try:
        A = _cast_sympy_matrix_to_numpy(A, dtype = 'int64')
    except OverflowError:
        return default(A0, U0)
    _MAXA = abs(A).max()
    if isnan(_MAXA) or _MAXA == inf or _MAXA > _INT64_MAX:
        return default(A0, U0)

    q2, U = primitive(U._rep)
    try:
        U = _cast_sympy_matrix_to_numpy(U, dtype = 'int64')
    except OverflowError:
        return default(A0, U0)
    _MAXU = abs(U).max()
    if isnan(_MAXU) or _MAXU == inf or _MAXU > _INT64_MAX or int(_MAXA) * int(_MAXU)**2 * n**2 > _INT64_MAX:
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