from typing import List, Optional, Tuple, Union, Callable
from contextlib import contextmanager

import numpy as np
import sympy as sp

from ...utils import congruence


def congruence_with_perturbation(M: sp.Matrix, perturb: bool = False):
    """
    Perform congruence decomposition on M. This is a wrapper of congruence.
    Write it as M = U.T @ S @ U where U is upper triangular and S is a diagonal matrix.

    Parameters
    ----------
    M : sp.Matrix
        Symmetric matrix to be decomposed.
    perturb : bool
        If perturb == True, make a slight perturbation to force M to be positive semidefinite.
        This is useful when there exists numerical errors in the matrix.

    Returns
    ---------
    U : sp.Matrix
        Upper triangular matrix.
    S : sp.Matrix
        Diagonal vector (1D array).
    """
    if not perturb:
        return congruence(M)
    else:
        min_eig = min([sp.re(v) for v in M.n(20).eigenvals()])
        if min_eig < 0:
            eps = 1e-15
            for i in range(10):
                cong = congruence(M + (-min_eig + eps) * sp.eye(M.shape[0]))
                if cong is not None:
                    return cong
                eps *= 10
        return congruence(M)
    return None


def is_numer_matrix(M: sp.Matrix) -> bool:
    """
    Check whether a matrix contains sp.Float.
    """
    for v in M:
        if not isinstance(v, sp.Rational):
            return True
    return False


def upper_vec_of_symmetric_matrix(
        S: Union[sp.Matrix, np.ndarray, int], 
        return_inds: bool = False,
        mul2: bool = False,
        check: Callable = None,
    ) -> List[Union[sp.Expr, Tuple[int, int]]]:
    """
    Gather the upper part of a symmetric matrix by rows as a vector.

    Parameters
    ----------
    S : np.ndarray | sp.Matrix | int
        Symmetric matrix or the shape[0] of the matrix.
    return_inds : bool
        If True, return (i,j) instead of S[i,j].
    mul2 : bool
        If True, multiply off-diagonal elements by 2.
    check : function
        Function to filter rows and columns.

    Yields
    ------
    S[i,j] :
        Upper part of S.
    """
    k = S.shape[0] if hasattr(S, 'shape') else S

    if return_inds:
        ret = lambda i, j: (i,j)
    elif mul2:
        ret = lambda i, j: 2 * S[i,j] if i != j else S[i,j]
    else:
        ret = lambda i, j: S[i,j]

    if check is not None:
        for i in filter(check, range(k)):
            for j in filter(check, range(i, k)):
                yield ret(i,j)
    else:
        for i in range(k):
            for j in range(i, k):
                yield ret(i,j)


def symmetric_matrix_from_upper_vec(upper_vec: Union[sp.Matrix, np.ndarray]) -> sp.Matrix:
    """
    Construct a symmetric matrix from its upper part.
    When upper_vec is a sympy matrix, it is assumed to be in the shape of a vector.
    When upper_vec is a numpy matrix, it can be 1D or 2D. If it is 1D as a vector,
    the function has the same behavior as the sympy version. If it is 2D, it returns
    a 3D tensor with the last dimension being the last dimension of upper_vec.

    Parameters
    ----------
    upper_vec : sp.Matrix | np.ndarray
        Upper part of a symmetric matrix.

    Returns
    -------
    S : sp.Matrix | np.ndarray
        Symmetric matrix.
    """
    n = round((2 * upper_vec.shape[0] + .25) ** 0.5 - .5)
    if isinstance(upper_vec, sp.MatrixBase):
        S = sp.zeros(n)
    elif isinstance(upper_vec, np.ndarray):
        S = np.zeros((n,n,upper_vec.shape[1])) if upper_vec.ndim == 2 else np.zeros((n,n))

    for (i,j), v in zip(upper_vec_of_symmetric_matrix(n, return_inds = True), upper_vec):
        S[i,j] = v
        S[j,i] = v
    return S


def solve_undetermined_linear(M: sp.Matrix, B: sp.Matrix) -> Tuple[sp.Matrix, sp.Matrix]:
    """
    Solve an undetermined linear system Mx = B with LU decomposition.
    See details at sympy.Matrix.gauss_jordan_solve.

    Returns
    -------
    x0: array
        One solution of Mx = B.
    space: Matrix
        All solution x is in the form of x0 + space @ y.
    """
    aug      = M.hstack(M.copy(), B.copy())
    B_cols   = B.cols
    row, col = aug[:, :-B_cols].shape

    # solve by reduced row echelon form
    A, pivots = aug.rref(normalize_last = False, simplify = False)
    A, v      = A[:, :-B_cols], A[:, -B_cols:]
    pivots    = list(filter(lambda p: p < col, pivots))
    rank      = len(pivots)

    # Get index of free symbols (free parameters)
    # non-pivots columns are free variables
    free_var_index = [c for c in range(A.cols) if c not in pivots]

    # Bring to block form
    permutation = sp.Matrix(pivots + free_var_index).T

    # check for existence of solutions
    # rank of aug Matrix should be equal to rank of coefficient matrix
    if not v[rank:, :].is_zero_matrix:
        raise ValueError("Linear system has no solution")

    # Full parametric solution
    V        = A[:rank, free_var_index]
    V        = V.col_join(-sp.eye(V.cols))
    vt       = v[:rank, :]
    vt       = vt.col_join(sp.zeros(V.cols, vt.cols))

    # Undo permutation
    V2       = sp.zeros(*V.shape)
    vt2      = sp.zeros(*vt.shape)
    for k in range(col):
        V2[permutation[k], :] = V[k, :]
        vt2[permutation[k]] = vt[k]
    return vt2, V2


def split_vector(chunks: List[Union[int, List, sp.Matrix]]) -> List[slice]:
    """
    It is common for a single array to store information of multiple
    matrices. And we need to extract each matrix from the array by 
    chunks of indices.

    This function takes in a list of ints / lists / matrices, and returns a list of
    slices with lengths equal to shape[1] of each matrix.

    Parameters
    ----------
    chunks : List[int | List | sp.Matrix]
        List of ints / lists / matrices.
    
    Returns
    ----------
    splits : List[slice]
        List of slices.
    """
    splits = []
    split_start = 0

    def length(c):
        if c is None:
            return 0
        elif isinstance(c, sp.MatrixBase):
            return c.shape[1]
        elif isinstance(c, int):
            return c
        elif isinstance(c, list):
            return len(c)

    for chunk in chunks:
        l = length(chunk)
        if l > 0:
            l = l * (l + 1) // 2
            splits.append(slice(split_start, split_start + l))
            split_start += l
    return splits


def S_from_y(y: sp.Matrix, x0: sp.Matrix, space: sp.Matrix, splits: List[slice]) -> List[sp.Matrix]:
    """
    Return the symmetric matrices S from the vector y.
    """
    if not isinstance(y, sp.MatrixBase):
        y = sp.Matrix(y)

    vecS = x0 + space * y
    Ss = []
    for split in splits:
        S = symmetric_matrix_from_upper_vec(sp.Matrix(vecS[split]))
        Ss.append(S)
    return Ss


@contextmanager
def indented_print(indent = 4, indent_fill = ' ', verbose = True):
    """
    Every print in this context will be indented by some spaces.
    """
    if not verbose:
        yield
        return

    import builtins
    old_print = print
    indent_str = indent_fill * indent
    def new_print(*args, **kwargs):
        old_print(indent_str, end = '')
        old_print(*args, **kwargs)

    builtins.print = new_print
    yield
    builtins.print = old_print