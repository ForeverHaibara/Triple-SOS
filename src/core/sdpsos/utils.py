from typing import List, Optional, Tuple
from contextlib import contextmanager, nullcontext

import sympy as sp

from ...utils import congruence


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


def is_numer_matrix(M: sp.Matrix) -> bool:
    """
    Check whether a matrix contains sp.Float.
    """
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if isinstance(M[i,j], sp.Float):
                return True
    return False


def upper_vec_of_symmetric_matrix(S: sp.Matrix, return_inds = False, check = None):
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

    Yields
    ------
    S[i,j] :
        Upper part of S.
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


def split_vector(constraints: List[sp.Matrix]) -> List[slice]:
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