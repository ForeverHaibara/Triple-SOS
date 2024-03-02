# arithmetic.py  
#  
# This module provides an implementation of rational number matrices and their  
# arithmetic operations, aiming to provide a more efficient alternative to  
# using sympy.Rational for matrix computations.  
#
# NOTE: sympy has supported DomainMatrix on ZZ/QQ to compute rref faster
# with version > 1.12 (exclusive)
# TODO: compare the performance of the two implementations

from fractions import Fraction
from time import time
from typing import List, Tuple

import sympy as sp
from sympy import Rational

def _find_reasonable_pivot_naive(col):
    """
    Find a reasonable pivot candidate in a column.
    See also in sympy.matrices.determinant._find_reasonable_pivot_naive
    """
    for i, col_val in enumerate(col):
        if col_val != 0:
            # This pivot candidate is non-zero.
            return i, col_val

    return None, None

def _row_reduce_list(mat, rows, cols, normalize_last=True, normalize=True, zero_above=True):
    """
    See also in sympy.matrices.reductions._row_reduce_list
    """
    one = Fraction(1, 1)

    def get_col(i):
        return mat[i::cols]

    def row_swap(i, j):
        mat[i*cols:(i + 1)*cols], mat[j*cols:(j + 1)*cols] = \
            mat[j*cols:(j + 1)*cols], mat[i*cols:(i + 1)*cols]

    def cross_cancel(a, i, b, j):
        """Does the row op row[i] = a*row[i] - b*row[j]"""
        q = (j - i)*cols
        for p in range(i*cols, (i + 1)*cols):
            mat[p] = (a*mat[p] - b*mat[p + q])

    piv_row, piv_col = 0, 0
    pivot_cols = []
    swaps = []

    # use a fraction free method to zero above and below each pivot
    while piv_col < cols and piv_row < rows:
        pivot_offset, pivot_val = _find_reasonable_pivot_naive(
                get_col(piv_col)[piv_row:])

        if pivot_offset is None:
            piv_col += 1
            continue

        pivot_cols.append(piv_col)
        if pivot_offset != 0:
            row_swap(piv_row, pivot_offset + piv_row)
            swaps.append((piv_row, pivot_offset + piv_row))

        # if we aren't normalizing last, we normalize
        # before we zero the other rows
        if normalize_last is False:
            i, j = piv_row, piv_col
            mat[i*cols + j] = one
            for p in range(i*cols + j + 1, (i + 1)*cols):
                mat[p] = (mat[p] / pivot_val)
            # after normalizing, the pivot value is 1
            pivot_val = one

        # zero above and below the pivot
        for row in range(rows):
            # don't zero our current row
            if row == piv_row:
                continue
            # don't zero above the pivot unless we're told.
            if zero_above is False and row < piv_row:
                continue
            # if we're already a zero, don't do anything
            val = mat[row*cols + piv_col]
            if val == 0:
                continue

            cross_cancel(pivot_val, row, val, piv_row)
        piv_row += 1

    # normalize each row
    if normalize_last is True and normalize is True:
        for piv_i, piv_j in enumerate(pivot_cols):
            pivot_val = mat[piv_i*cols + piv_j]
            mat[piv_i*cols + piv_j] = one
            for p in range(piv_i*cols + piv_j + 1, (piv_i + 1)*cols):
                mat[p] = (mat[p] / pivot_val)

    return mat, tuple(pivot_cols), tuple(swaps)

def _row_reduce(M, normalize_last=True,
                normalize=True, zero_above=True):
    """
    See also in sympy.matrices.reductions._row_reduce
    It converts sympy Rational matrix to Fraction without checking.
    """
    M_frac_list = [Fraction(x.p, x.q, _normalize=False) for x in M]

    mat, pivot_cols, swaps = _row_reduce_list(M_frac_list, M.rows, M.cols,
            normalize_last=normalize_last, normalize=normalize, zero_above=zero_above)

    mat = [Rational(x.numerator, x.denominator, gcd = 1) for x in mat]

    return sp.Matrix(M.rows, M.cols, mat), pivot_cols, swaps

def _rref(M, pivots=True, normalize_last=True):
    """
    See also in sympy.matrices.reductions.rref
    """
    if not all(_.is_Rational for _ in M):
        return M.rref(pivots=pivots, normalize_last=normalize_last)

    mat, pivot_cols, _ = _row_reduce(M, normalize_last=normalize_last, normalize=True, zero_above=True)

    if pivots:
        mat = (mat, pivot_cols)
    return mat


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

    # time0 = time()

    # solve by reduced row echelon form
    A, pivots = _rref(aug, normalize_last = False)

    # print("Time for rref: ", time() - time0, 'Shape =', M.shape)
    # time0 = time()

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

    # print("Time for solving: ", time() - time0)

    return vt2, V2


def solve_column_separated_linear(A: List[List[Tuple[int, Rational]]], b: sp.Matrix, x0_equal_indices: List[List[int]] = [], _cols: int = -1):
    """
    This is a function that solves a special linear system Ax = b => x = x_0 + C * y
    where each column of A has at most 1 nonzero element.

    Further, we could require some of entries of x to be equal.

    Parameters
    ----------
    A: List[List[Tuple[int, Rational]]]
        Sparse representation of A. If A[i][..] = (j, v), then A[i, j] = v.
        Also, it must guarantee that A[i', j] == 0 for all i' != i.
        Also, be sure that v != 0.
    b: sp.Matrix
        Right-hand side
    x0_equal_indices: List[List[int]]
        Each sublist contains indices of equal elements.
    _cols: int
        Number of columns of A. If not specified, it will be inferred from A.

    Returns
    ---------
    x0: Matrix
        One solution of Ax = b.
    space: Matrix
        All solution x is in the form of x0 + space @ y.
    """
    # count the number of columns of A
    cols = _cols if _cols != -1 else max(max(k[0] for k in row) for row in A) + 1

    # form the equal indices as a UFS
    ufs = list(range(cols))
    groups = {}
    for group in x0_equal_indices:
        for i in group:
            ufs[i] = group[0]
        groups[group[0]] = group
    for i in range(cols):
        if ufs[i] == i:
            group = groups.get(i)
            if group is None:
                groups[i] = [i]

    x0 = [0] * cols
    spaces = []
    for i, row in enumerate(A):
        if len(row):
            pivot = row[0]
            head = ufs[pivot[0]]
            group = groups[head]
            w = len(group) * pivot[1]
            v = b[i] / w
            for k in group:
                x0[k] = v

            for j in range(1, len(row)):
                pivot2 = row[j]
                head2 = ufs[pivot2[0]]
                if pivot2[0] != head2 or head2 == head:
                    continue
                # only handle the case that pivot is the head of the group
                group2 = groups[head2]
                w2 = len(group2) * pivot2[1]

                space = [0] * cols
                for k in group:
                    space[k] = w2
                for k in group2:
                    space[k] = -w
                spaces.append(space)

        else:
            if b[i] != 0:
                raise ValueError("Linear system has no solution")

            for j in range(len(row)):
                if ufs[row[j][0]] == row[j][0]:
                    group = groups[row[j][0]]

                    space = [0] * cols
                    for k in group:
                        space[k] = 1
                    spaces.append(space)

    return sp.Matrix(x0), sp.Matrix(spaces).T