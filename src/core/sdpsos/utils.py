from typing import List, Optional
from contextlib import contextmanager

import numpy as np
import sympy as sp

from ...utils.roots.rationalize import rationalize
from ...utils import congruence

def _rationalize_matrix_mask(M, tolerance = 1e-10):
    M_abs = np.abs(M)
    mask_tolerance = M_abs.mean() * tolerance
    mask = M_abs > mask_tolerance
    return mask

def rationalize_matrix(M, mask_func = None, symmetric = True):
    """
    Rationalize a matrix with continued fraction.
    """
    if mask_func is None:
        mask_func = _rationalize_matrix_mask
    if not hasattr(mask_func, '__call__'):
        mask_copy = mask_func
        mask_func = lambda x: np.abs(x) > mask_copy
    mask = mask_func(M)

    M = np.where(mask, M, 0)
    M_ = sp.Matrix.zeros(M.shape[0], M.shape[1])
    if symmetric:
        for i in range(M.shape[0]):
            for j in range(i, M.shape[1]):
                M_[j,i] = M_[i,j] = rationalize(M[i,j], reliable = True)
    else:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                M_[i,j] = rationalize(M[i,j], reliable = True)

    return M_

def congruence_with_perturbation(M, allow_numer = False):
    """
    Perform congruence decomposition on M. 
    If allow_numer == True, make a slight perturbation
    so that M is positive semidefinite.
    """
    if not allow_numer:
        return congruence(M)
    else:
        min_eig = min([v.n(20) if not isinstance(v, sp.Float) else v for v in M.eigenvals()])
        if min_eig < 0:
            perturbation = -min_eig + 1e-15
            return congruence(M + perturbation * sp.eye(M.shape[0]))
        return congruence(M)
    return None


def is_numer_matrix(M):
    """
    Check whether a matrix contains sp.Float.
    """
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if isinstance(M[i,j], sp.Float):
                return True
    return False


def _discard_zero_rows(A, b, rank_tolerance = 1e-10):
    A_rowmax = np.abs(A).max(axis = 1)
    nonvanish =  A_rowmax > rank_tolerance * A_rowmax.max()
    rows = np.extract(nonvanish, np.arange(len(nonvanish)))
    return A[rows], b[rows]


def upper_vec_of_symmetric_matrix(S, return_inds = False, check = None):
    """
    Gather the upper part of a symmetric matrix by rows as a vector.

    Parameters
    ----------
    S : np.ndarray | int
        Symmetric matrix.
    return_inds : bool
        If True, return (i,j) instead of S[i,j].
    check : function
        Function to filter rows and columns.
    """
    k = S.shape[0] if hasattr(S, 'shape') else S

    if return_inds:
        ret = lambda i, j: (i,j)
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


# def solve_undetermined_linear(A, b, rank_tolerance = 1e-10):
#     """
#     Solve an undetermined linear system Ax = b with SVD.

#     The solution is in the form x = x0 + V[rank:].T @ y
#     where x0 is the least square solution and y is an arbitrary vector.

#     Returns
#     -------
#     solution : dict
#     """
#     U, D, V = np.linalg.svd(A, full_matrices = False)
#     rank_tolerance = 1e-11
#     rank = np.sum(D > rank_tolerance)
#     x0 = V.T @ np.linalg.lstsq(U * D.reshape((1, -1)), b, rcond = None)[0]

#     solution = {
#         'U': U,
#         'D': D,
#         'V': V,
#         'rank': rank,
#         'x0': x0
#     }
#     return solution


def solve_undetermined_linear(M: sp.Matrix, B: sp.Matrix):
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


def split_vector(constraints: List[sp.Matrix]):
    """
    It is common for a single array to store information of multiple
    matrices. And we need to extract each matrix from the array by 
    chunks of indices.

    This function takes in a list of matrices, and returns a list of
    slices with lengths equal to shape[1] of each matrix.

    Parameters
    ----------
    constraints : List[sp.Matrix]
        List of matrices.
    
    Returns
    ----------
    splits : List[slice]
        List of slices.
    """
    splits = []
    split_start = 0
    for constraint in constraints:
        if constraint is not None:
            splits.append(slice(split_start, split_start + constraint.shape[1]))
            split_start += constraint.shape[1]
    return splits


@contextmanager
def indented_print(indent = 4, indent_fill = ' '):
    """
    Every print in this context will be indented by some spaces.
    """
    import builtins
    old_print = print
    indent_str = indent_fill * indent
    def new_print(*args, **kwargs):
        old_print(indent_str, end = '')
        old_print(*args, **kwargs)

    builtins.print = new_print
    yield
    builtins.print = old_print