# arithmetic.py  
#  
# This module provides an implementation of rational number matrices and their  
# arithmetic operations, aiming to provide a more efficient alternative to  
# using sympy.Rational for matrix computations.  
#
# NOTE: sympy has supported DomainMatrix on ZZ/QQ to compute rref faster
# with version > 1.12 (exclusive)
# TODO: compare the performance of the two implementations
# TODO: use numpy for matrix computation

# from fractions import Fraction
from math import gcd
from time import time
from typing import List, Tuple, Union, Optional

from numpy import array as np_array, around as np_round, zeros as np_zeros, iinfo as np_iinfo
from numpy import isnan, inf, unique
import sympy as sp
from sympy import __version__ as _SYMPY_VERSION
from sympy import Rational
from sympy.external.gmpy import MPQ # >= 1.9
from sympy.matrices.repmatrix import RepMatrix
from sympy.polys.domains import ZZ, QQ, EX, EXRAW # EXRAW >= 1.9
from sympy.polys.matrices.domainmatrix import DomainMatrix # polys.matrices >= 1.8
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.sdm import SDM

if _SYMPY_VERSION >= '1.13':
    from sympy.polys.matrices.dfm import DFM
else:
    class _DFM_dummy: ...
    DFM = _DFM_dummy

from .utils import Mat2Vec, is_empty_matrix

_INT32_MAX = np_iinfo('int32').max # 2147483647
_INT64_MAX = np_iinfo('int64').max # 9223372036854775807

# for dev purpose only
_VERBOSE_MATMUL_MULTIPLE = False
_VERBOSE_SOLVE_UNDETERMINED_LINEAR = False
_VERBOSE_SOLVE_CSR_LINEAR = False


def _lcm(x, y):
    """
    LCM func for python internal integers. DO NOT USE SYMPY BECAUSE IT IS SLOW
    WHEN CONVERTING TO SYMPY INTEGERS.
    """
    return x * y // gcd(x, y)

def is_zz_qq_mat(mat):
    """
    Judge whether a matrix is a ZZ/QQ RepMatrix.
    """
    return isinstance(mat, RepMatrix) and mat._rep.domain in (ZZ, QQ)

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
    one = MPQ.__new__(MPQ, 1, 1)

    def get_col(i):
        return mat[i::cols]

    def row_swap(i, j):
        mat[i*cols:(i + 1)*cols], mat[j*cols:(j + 1)*cols] = \
            mat[j*cols:(j + 1)*cols], mat[i*cols:(i + 1)*cols]

    def cross_cancel(a, i, b, j):
        """Does the row op row[i] = a*row[i] - b*row[j]"""
        q = (j - i)*cols
        gcdab = MPQ(gcd(a.numerator, b.numerator), gcd(a.denominator, b.denominator))
        a, b = a / gcdab, b / gcdab
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


def sdm_rref_den(A, K):
    """
    Return the reduced row echelon form (RREF) of A with denominator.
    The RREF is computed using fraction-free Gauss-Jordan elimination.

    The algorithm used is the fraction-free version of Gauss-Jordan elimination
    described as FFGJ. Here it is modified to handle zero or missing
    pivots and to avoid redundant arithmetic. This implementation is also
    optimized for sparse matrices. See also sympy.polys.matrices.sdm (version >= 1.13).
    """
    if _VERBOSE_SOLVE_UNDETERMINED_LINEAR:
        time0 = time()
    if not A:
        return ({}, K.one, [])
    elif len(A) == 1:
        Ai, = A.values()
        j = min(Ai)
        Aij = Ai[j]
        return ({0: Ai.copy()}, Aij, [j])

    # For inexact domains like RR[x] we use quo and discard the remainder.
    # Maybe it would be better for K.exquo to do this automatically.
    if K.is_Exact:
        exquo = K.exquo
    else:
        exquo = K.quo

    # Make sure we have the rows in order to make this deterministic from the
    # outset.
    _, rows_in_order = zip(*sorted(A.items()))

    col_to_row_reduced = {}
    col_to_row_unreduced = {}
    reduced = col_to_row_reduced.keys()
    unreduced = col_to_row_unreduced.keys()

    # Our representation of the RREF so far.
    A_rref_rows = []
    denom = None
    divisor = None

    # The rows that remain to be added to the RREF. These are sorted by the
    # column index of their leading entry. Note that sorted() is stable so the
    # previous sort by unique row index is still needed to make this
    # deterministic (there may be multiple rows with the same leading column).
    A_rows = sorted(rows_in_order, key=min)

    for Ai in A_rows:

        # All fully reduced columns can be immediately discarded.
        Ai = {j: Aij for j, Aij in Ai.items() if j not in reduced}
        Ai_cancel = {}

        for j in unreduced & Ai.keys():
            # Remove the pivot column from the new row since it would become
            # zero anyway.
            Aij = Ai.pop(j)

            Aj = A_rref_rows[col_to_row_unreduced[j]]

            for k, Ajk in Aj.items():
                Aik_cancel = Ai_cancel.get(k)
                if Aik_cancel is None:
                    Ai_cancel[k] = Aij * Ajk
                else:
                    Aik_cancel = Aik_cancel + Aij * Ajk
                    if Aik_cancel:
                        Ai_cancel[k] = Aik_cancel
                    else:
                        Ai_cancel.pop(k)

        # Multiply the new row by the current denominator and subtract.
        Ai_nz = set(Ai)
        Ai_cancel_nz = set(Ai_cancel)

        d = denom or K.one

        for k in Ai_cancel_nz - Ai_nz:
            Ai[k] = -Ai_cancel[k]

        for k in Ai_nz - Ai_cancel_nz:
            Ai[k] = Ai[k] * d

        for k in Ai_cancel_nz & Ai_nz:
            Aik = Ai[k] * d - Ai_cancel[k]
            if Aik:
                Ai[k] = Aik
            else:
                Ai.pop(k)

        # Now Ai has the same scale as the other rows and is reduced wrt the
        # unreduced rows.

        # If the row is reduced to zero then discard it.
        if not Ai:
            continue

        # Choose a pivot for this row.
        j = min(Ai)
        Aij = Ai.pop(j)

        # Cross cancel the unreduced rows by the new row.
        #     a[k][l] = (a[i][j]*a[k][l] - a[k][j]*a[i][l]) / divisor
        for pk, k in list(col_to_row_unreduced.items()):

            Ak = A_rref_rows[k]

            if j not in Ak:
                # This row is already reduced wrt the new row but we need to
                # bring it to the same scale as the new denominator. This step
                # is not needed in sdm_irref.
                for l, Akl in Ak.items():
                    Akl = Akl * Aij
                    if divisor is not None:
                        Akl = exquo(Akl, divisor)
                    Ak[l] = Akl
                continue

            Akj = Ak.pop(j)
            Ai_nz = set(Ai)
            Ak_nz = set(Ak)

            for l in Ai_nz - Ak_nz:
                Ak[l] = - Akj * Ai[l]
                if divisor is not None:
                    Ak[l] = exquo(Ak[l], divisor)

            # This loop also not needed in sdm_irref.
            for l in Ak_nz - Ai_nz:
                Ak[l] = Aij * Ak[l]
                if divisor is not None:
                    Ak[l] = exquo(Ak[l], divisor)

            for l in Ai_nz & Ak_nz:
                Akl = Aij * Ak[l] - Akj * Ai[l]
                if Akl:
                    if divisor is not None:
                        Akl = exquo(Akl, divisor)
                    Ak[l] = Akl
                else:
                    Ak.pop(l)

            if not Ak:
                col_to_row_unreduced.pop(pk)
                col_to_row_reduced[pk] = k

        i = len(A_rref_rows)
        A_rref_rows.append(Ai)
        if Ai:
            col_to_row_unreduced[j] = i
        else:
            col_to_row_reduced[j] = i

        # Update the denominator.
        if not K.is_one(Aij):
            if denom is None:
                denom = Aij
            else:
                denom *= Aij

        if divisor is not None:
            denom = exquo(denom, divisor)

        # Update the divisor.
        divisor = denom

    if denom is None:
        denom = K.one

    # Sort the rows by their leading column index.
    col_to_row = {**col_to_row_reduced, **col_to_row_unreduced}
    row_to_col = {i: j for j, i in col_to_row.items()}
    A_rref_rows_col = [(row_to_col[i], Ai) for i, Ai in enumerate(A_rref_rows)]
    pivots, A_rref = zip(*sorted(A_rref_rows_col))
    pivots = list(pivots)

    # Insert the pivot values
    for i, Ai in enumerate(A_rref):
        Ai[pivots[i]] = denom

    A_rref_sdm = dict(enumerate(A_rref))

    if _VERBOSE_SOLVE_UNDETERMINED_LINEAR:
        print(">> Time for sdm_rref_den:", time() - time0)
    return A_rref_sdm, denom, pivots


def _row_reduce(M, normalize_last=True,
                normalize=True, zero_above=True):
    """
    See also in sympy.matrices.reductions._row_reduce
    It converts sympy Rational matrix to MPQ without checking.

    M should be a RepMatrix with domain ZZ or QQ.
    """
    if _VERBOSE_SOLVE_UNDETERMINED_LINEAR:
        time0 = time()

    M_frac_list = [MPQ.__new__(MPQ, x.p, x.q) for x in M]

    if _VERBOSE_SOLVE_UNDETERMINED_LINEAR:
        print(">> Time for converting to MPQ:", time() - time0)
        time0 = time()

    mat, pivot_cols, swaps = _row_reduce_list(M_frac_list, M.rows, M.cols,
            normalize_last=normalize_last, normalize=normalize, zero_above=zero_above)

    if _VERBOSE_SOLVE_UNDETERMINED_LINEAR:
        print(">> Time for row reduce list:", time() - time0)
        time0 = time()

    mat = [Rational(x.numerator, x.denominator, gcd = 1) for x in mat]

    return sp.Matrix(M.rows, M.cols, mat), pivot_cols, swaps


def _rref(M, pivots=True, normalize_last=True):
    """
    See also in sympy.matrices.reductions.rref
    """
    if not is_zz_qq_mat(M):
        return M.rref(pivots=pivots, normalize_last=normalize_last)

    sdm = M._rep.to_sdm()
    K = sdm.domain

    sdm_rref_dict, den, pivots = sdm_rref_den(sdm, K)
    sdm_rref = sdm.new(sdm_rref_dict, sdm.shape, sdm.domain)
    dM = DomainMatrix.from_rep(sdm_rref)
    if den != K.one:
        dM = dM.to_field()
        dM = dM / den

    mat = M.__class__._fromrep(dM)
    # dM_rref = M.new(dM_rref, M.shape, M.domain)
    # mat = 
    # if pivots:
    #     return dM_rref, pivots
    # return dM_rref
    # mat, pivot_cols, _ = _row_reduce(M, normalize_last=normalize_last, normalize=True, zero_above=True)

    if pivots:
        mat = (mat, pivots)
    return mat

def _permute_matrix_rows(matrix, permutation):
    rep = matrix._rep.rep if isinstance(matrix, RepMatrix) else None

    if isinstance(rep, SDM):
        new_rep = {}
        for orig_row, cols_dict in rep.items():
            new_rep[permutation[orig_row]] = cols_dict #.copy()
        new_rep = SDM(new_rep, rep.shape, rep.domain)
        return matrix.__class__._fromrep(DomainMatrix.from_rep(new_rep))

    elif isinstance(rep, DDM):
        new_rep = [None for _ in range(rep.shape[0])]
        for orig_row in range(rep.shape[0]):
            new_rep[permutation[orig_row]] = rep[orig_row]#[:]
        new_rep = DDM(new_rep, rep.shape, rep.domain)
        return matrix.__class__._fromrep(DomainMatrix.from_rep(new_rep))

    elif isinstance(rep, DFM):
        new_rep = [None for _ in range(rep.shape[0])]
        rep2 = rep.rep.tolist()
        for orig_row in range(rep.shape[0]):
            new_rep[permutation[orig_row]] = rep2[orig_row]
        new_rep = DFM(new_rep, rep.shape, rep.domain)
        return matrix.__class__._fromrep(DomainMatrix.from_rep(new_rep))

    # else:
    new_mat = matrix.copy()
    for orig_row in range(matrix.shape[0]):
        new_mat[permutation[orig_row], :] = matrix[orig_row, :]
    return new_mat


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

    if _VERBOSE_SOLVE_UNDETERMINED_LINEAR:
        sparsity = lambda x: len(list(x.iter_values())) / (x.shape[0] * x.shape[1])
        print('SolveUndeterminedLinear', M.shape)
        print('>> Sparsity M =', sparsity(M))
        time0 = time()

    # solve by reduced row echelon form
    A, pivots = _rref(aug, normalize_last = False)

    
    if _VERBOSE_SOLVE_UNDETERMINED_LINEAR:
        print(">> Time for rref:", time() - time0) # main bottleneck
        time0 = time()

    A, v      = A[:, :-B_cols], A[:, -B_cols:]
    pivots    = list(filter(lambda p: p < col, pivots))
    rank      = len(pivots)

    # Get index of free symbols (free parameters)
    # non-pivots columns are free variables
    free_var_index = [c for c in range(A.cols) if c not in pivots]

    # Bring to block form
    permutation = pivots + free_var_index

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
    V2 = _permute_matrix_rows(V, permutation)
    vt2 = _permute_matrix_rows(vt, permutation)
    # V2       = sp.zeros(V.shape[0], V.shape[1])
    # vt2      = sp.zeros(vt.shape[0], vt.shape[1])
    # for k in range(col):
    #     V2[permutation[k], :] = V[k, :]
    #     vt2[permutation[k]] = vt[k]

    if _VERBOSE_SOLVE_UNDETERMINED_LINEAR:
        print(">> Time for undo permutation:", time() - time0,
              'V2, vt2 shape =', V2.shape, vt2.shape)

    return vt2, V2

def solve_nullspace(A: sp.Matrix) -> sp.Matrix:
    """
    Compute the null space of a matrix A.
    If A is full-rank and has shape m x n (m > n), then the null space has shape m x (m-n).
    """
    # try:
    #     x0, space = solve_undetermined_linear(A.T, sp.zeros(A.cols, 1))
    # except ValueError:
    #     return sp.Matrix.zeros(A.rows, 0)
    # return space
    m = sp.Matrix.hstack(*A.T.nullspace())
    if is_empty_matrix(m):
        return sp.zeros(A.shape[0], 0)
    return m



def solve_column_separated_linear(A: List[List[Tuple[int, Rational]]], b: sp.Matrix, x0_equal_indices: List[List[int]] = [],
                                  _cols: int = -1, domain = None):
    """
    This is a function that solves a special linear system Ax = b => x = x_0 + C * y
    where each column of A has at most 1 nonzero element. For more general cases, use solve_csr_linear.
    Further, we could require some of entries of x to be equal.

    Parameters
    ----------
    A: List[List[Tuple[int, Rational]]]
        Sparse row representation of A. If A[i][..] = (j, v), then A[i, j] = v.
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
    cols = _cols if _cols != -1 else (max(max(k[0] for k in row) for row in A) + 1)

    if not isinstance(b, RepMatrix):
        domain = EXRAW
    elif domain is None:
        domain = QQ # EXRAW
    else:
        domain = domain.unify(b._rep.domain).get_field()

    if _VERBOSE_SOLVE_CSR_LINEAR:
        time0 = time()

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

    if _VERBOSE_SOLVE_CSR_LINEAR:
        print('>> UFS construction time', time() - time0) # 0 sec, can be ignored
        time0 = time()

    toK = domain.from_sympy
    one, zero = domain.one, domain.zero
    b = b._rep.convert_to(domain).to_sdm() # DomainMatrix

    x0 = []
    spaces = []
    for i, row in enumerate(A):
        if len(row):
            pivot = row[0]
            head = ufs[pivot[0]]
            group = groups[head]
            w = len(group) * toK(pivot[1])
            bi = b.get(i, 0)
            if bi:
                bi = bi.get(0, zero)
                v = bi / w
                for k in group:
                    x0.append((k, {0: v}))

            for j in range(1, len(row)):
                pivot2 = row[j]
                head2 = ufs[pivot2[0]]
                if pivot2[0] != head2 or head2 == head:
                    continue
                # only handle the case that pivot is the head of the group
                group2 = groups[head2]
                w2 = len(group2) * toK(pivot2[1])

                space = {}
                if w2:
                    for k in group:
                        space[k] = w2
                if w:
                    for k in group2:
                        space[k] = -w
                spaces.append(space)

        else:
            bi = b.get(i, 0)
            print('bi', bi, bi.get(0, zero), zero)
            if bi != 0 and bi.get(0, zero) != zero:
                raise ValueError("Linear system has no solution")

            for j in range(len(row)):
                if ufs[row[j][0]] == row[j][0]:
                    group = groups[row[j][0]]

                    space = {}
                    for k in group:
                        space[k] = one
                    spaces.append(space)

    x0 = dict(x0)
    spaces = dict(enumerate(spaces))
    if _VERBOSE_SOLVE_CSR_LINEAR:
        print('>> Solve separated system time:', time() - time0) # fast, < 1 sec over (100 x 10000)
        time0 = time()

    to_mat = lambda x, shape: sp.Matrix._fromrep(DomainMatrix.from_rep(SDM(x, shape, domain)))
    # x0, space = sp.Matrix(x0), sp.Matrix(spaces).T
    x0 = to_mat(x0, (cols, 1))
    spaces = to_mat(spaces, (len(spaces), cols)).T

    if _VERBOSE_SOLVE_CSR_LINEAR:
        print('>> Matrix restoration time:', time() - time0, 'space shape =', spaces.shape)

    return x0, spaces


def solve_csr_linear(A: List[List[Tuple[int, Rational]]], b: sp.Matrix, x0_equal_indices: List[List[int]] = [], _cols: int = -1):
    """
    Solve a linear system Ax = b where A is stored in CSR format.
    Further, we could require some of entries of x to be equal.

    Parameters
    ----------
    A: List[List[Tuple[int, Rational]]]
        Sparse row representation of A. If A[i][..] = (j, v), then A[i, j] = v.
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
    cols = _cols if _cols != -1 else (max(max(k[0] for k in row) for row in A) + 1)
    if _VERBOSE_SOLVE_CSR_LINEAR:
        print('SolveCsrLinear A shape', (len(A), cols))
        time0 = time()

    # check whether there is at most one nonzero element in each column
    _column_separated = True
    seen_cols = set()
    for row in A:
        for col, _ in row:
            if col in seen_cols:
                _column_separated = False
                break
            seen_cols.add(col)
        if not _column_separated:
            break

    if _column_separated:
        if _VERBOSE_SOLVE_CSR_LINEAR:
            print('>> Column Separated System recognized', time() - time0)
        return solve_column_separated_linear(A, b, x0_equal_indices, _cols=cols)

    # convert to dense matrix

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
    group_keys = list(groups.keys())
    group_inds = {k: i for i, k in enumerate(group_keys)}

    # compress the columns
    cols2 = len(groups)
    A2 = [[0] * cols2 for _ in range(len(A))]
    for i, row in enumerate(A):
        for j, v in row:
            A2[i][group_inds[ufs[j]]] += v

    A2 = sp.Matrix(A2)
    x0_compressed, space_compressed = solve_undetermined_linear(A2, b)
    space_compressed = space_compressed.tolist()
    x0, space = [0] * cols, [None for _ in range(cols)]

    # restore the solution
    for i in range(cols):
        x0[i] = x0_compressed[group_inds[ufs[i]]]
        space[i] = space_compressed[group_inds[ufs[i]]]
    x0, space = sp.Matrix(x0), sp.Matrix(space)
    return x0, space


def _common_denoms(A: Union[List, sp.Matrix], bound: int = 4096) -> Optional[int]:
    """
    Compute the common denominator of a list of rational numbers.
    This might be slow when the list is large. (e.g. 30000 entries -> 0.2 sec)
    """
    if isinstance(A, RepMatrix):
        rep = A._rep
        if rep.domain is ZZ:
            return 1
        elif rep.domain is QQ:
            return int(rep.content().denominator)
        elif rep.domain in (EX, EXRAW):
            return None

    # fallback
    q = 1
    for v in A:
        if not v.is_Rational:
            return None
        q = _lcm(q, v.q)
        if q > bound:
            return None
    return int(q)
    # try:
    #     vec = unique(np_array([_.q for _ in A]).astype('int'))
    #     if len(vec) > 20 or abs(vec).max() > bound:
    #         return None

    #     q = 1
    #     for v in vec:
    #         q = _lcm(q, v)
    #         if q > bound:
    #             return None
    #     return int(q)
    # except:
    #     return None # not all elements are Rational

def _cast_sympy_matrix_to_numpy(sympy_mat_rep, dtype: str = 'int64'):
    rows, cols = sympy_mat_rep.shape
    items = list(sympy_mat_rep.iter_items())
    n = len(items)

    row_indices = [0] * n
    col_indices = [0] * n
    data_list = [0] * n

    for k in range(n):
        (i, j), v = items[k]
        row_indices[k] = i
        col_indices[k] = j
        data_list[k] = v.__int__()

    arr = np_zeros((rows, cols), dtype=dtype)
    if data_list:
        arr[row_indices, col_indices] = data_list
    return arr



def matmul(A: sp.Matrix, B: sp.Matrix, q1: Optional[int] = None, q2: Optional[int] = None) -> sp.Matrix:
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
    q1: int
        Common denominator of A. If not specified, it will be inferred.
    q2: int
        Common denominator of B. If not specified, it will be inferred.
    """
    if A.shape[0] == 0 or B.shape[0] == 0 or B.shape[1] == 0:
        return sp.zeros(A.shape[0], B.shape[1])
    A0, B0 = A, B

    
    if not (is_zz_qq_mat(A) and is_zz_qq_mat(B)):
        return A0 * B0

    q1, A = A._rep.primitive()
    try:
        A = _cast_sympy_matrix_to_numpy(A, dtype = 'int64')
    except OverflowError:
        return A0 * B0
    _MAXA = abs(A).max()
    if isnan(_MAXA) or _MAXA == inf or _MAXA > _INT64_MAX:
        return A0 * B0

    q2, B = B._rep.primitive()
    try:
        B = _cast_sympy_matrix_to_numpy(B, dtype = 'int64')
    except OverflowError:
        return A0 * B0
    _MAXB = abs(B).max()
    if isnan(_MAXB) or _MAXB == inf or _MAXB > _INT64_MAX or int(_MAXA) * int(_MAXB) * B.shape[0] > _INT64_MAX:
        return A0 * B0

    # CAST_TYPE = 'int' if int(_MAXA) * int(_MAXB) * n**2 <= _INT32_MAX else 'int64'
    # CAST_TYPE = 'int64'
    # A = np_round(A).astype(CAST_TYPE)
    # B = np_round(B).astype(CAST_TYPE)

    C = (A @ B).flatten().tolist()
    q1q2 = q1 * q2
    C = [i*q1q2 for i in C]
    C = sp.Matrix(A0.shape[0], B0.shape[1], [Rational(i.numerator, i.denominator, gcd = 1) for i in C])
    return C

def matmul_multiple(A: sp.Matrix, B: sp.Matrix, q1: Optional[int] = None, q2: Optional[int] = None) -> sp.Matrix:
    """
    Perform multiple matrix multiplications. This can be regarded as a 3-dim tensor multiplication.
    Assume A has shape N x (n^2) and B has shape n x m, then the result has shape N x (n*m).

    Parameters
    ----------
    A: Matrix
        Matrix A
    B: Matrix
        Matrix B
    q1: int
        Common denominator of A. If not specified, it will be inferred.
    q2: int
        Common denominator of B. If not specified, it will be inferred.
    """
    if A.shape[0] == 0 or B.shape[0] == 0 or B.shape[1] == 0:
        return sp.zeros(A.shape[0], B.shape[0]*B.shape[1])

    A0, B0 = A, B
    def default(A, B):
        eq_mat = []
        for i in range(A.shape[0]):
            Aij = Mat2Vec.vec2mat(A[i,:])
            eq = list(matmul(Aij, B))
            eq_mat.append(eq)
        eq_mat = sp.Matrix(eq_mat)
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

    q1, A = A._rep.primitive()
    try:
        A = _cast_sympy_matrix_to_numpy(A, dtype = 'int64')
    except OverflowError:
        return default(A0, B0)
    _MAXA = abs(A).max()
    if isnan(_MAXA) or _MAXA == inf or _MAXA > _INT64_MAX:
        return default(A0, B0)

    q2, B = B._rep.primitive()
    try:
        B = _cast_sympy_matrix_to_numpy(B, dtype = 'int64')
    except OverflowError:
        return default(A0, B0)
    _MAXB = abs(B).max()
    if isnan(_MAXB) or _MAXB == inf or _MAXB > _INT64_MAX or int(_MAXA) * int(_MAXB) * n > _INT64_MAX:
        return default(A0, B0)

    # CAST_TYPE = 'int' if int(_MAXA) * int(_MAXB) * n**2 <= _INT32_MAX else 'int64'
    # CAST_TYPE = 'int64'
    # A = np_round(A).astype(CAST_TYPE)
    # B = np_round(B).astype(CAST_TYPE)
    A = A.reshape((N, n, n))

    if _VERBOSE_MATMUL_MULTIPLE:
        print('>> Time for casting to numpy:', time() - time0)
        time0 = time()

    C = (A @ B).flatten().tolist()

    if _VERBOSE_MATMUL_MULTIPLE:
        print('>> Time for numpy matmul:', time() - time0) # very fast (can be ignored)
        time0 = time()

    q1q2 = q1 * q2
    C = [i*q1q2 for i in C]
    C = sp.Matrix(N, n*m, [Rational(i.numerator, i.denominator, gcd = 1) for i in C])

    if _VERBOSE_MATMUL_MULTIPLE:
        print('>> Time for casting to sympy:', time() - time0) # < 1 sec over (800*8000)
        time0 = time()

    return C


def symmetric_bilinear(U: sp.Matrix, A: sp.Matrix, is_A_vec: bool = False, return_vec: bool = False):
    """
    Compute U.T * A * U efficiently.
    Assume U.T is m x n and A is n x n. Then Kronecker product is O(m^2n^2),
    and using direct matmul is O(mn(n+m)).

    When A is sparse with k nonzeros, the complexity is O(2m^2k).

    Parameters
    ----------
    U: Matrix
        Matrix U
    A: Matrix
        Matrix A
    is_A_vec: bool
        Whether A is stored as a vector. If True, it first converts A to a matrix.
    return_vec: bool
        Whether to return the result as a vector list.
    """
    nonzeros = []
    for i, val in enumerate(A):
        if val != 0:
            nonzeros.append((i, val))

    n, m = U.shape
    if len(nonzeros) * 2 >= n**2:
    # if True:
        if is_A_vec:
            A = Mat2Vec.vec2mat(A)
        M = matmul(U.T, matmul(A, U))
        # M = U.T * A * U

        if return_vec:
            return list(M)
        return M
    else:
        # A is sparse
        M = [sp.S.Zero] * (m * m)
        for l, val in nonzeros:
            p = 0
            l1, l2 = divmod(l, n)
            for i in range(m):
                for j in range(m):
                    M[p] += U[l1, i] * U[l2, j] * val
                    p += 1

    if return_vec:
        return M

    return sp.Matrix(m, m, M)


def symmetric_bilinear_multiple(U: sp.Matrix, A: sp.Matrix) -> sp.Matrix:
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
            # Aij = Mat2Vec.vec2mat(space[i,:])
            # eq = U.T * Aij * U
            eq = symmetric_bilinear(U, A[i,:], is_A_vec = True, return_vec = True)
            eq_mat.append(eq)
        eq_mat = sp.Matrix(eq_mat)
        return eq_mat

    N = A.shape[0]
    if N == 0:
        return sp.Matrix.zeros(0, U.shape[1]**2)

    if not (is_zz_qq_mat(A) and is_zz_qq_mat(U)):
        return default(A0, U0)

    n, m = U.shape


    q1, A = A._rep.primitive()
    try:
        A = _cast_sympy_matrix_to_numpy(A, dtype = 'int64')
    except OverflowError:
        return default(A0, U0)
    _MAXA = abs(A).max()
    if isnan(_MAXA) or _MAXA == inf or _MAXA > _INT64_MAX:
        return default(A0, U0)

    q2, U = U._rep.primitive()
    try:
        U = _cast_sympy_matrix_to_numpy(U, dtype = 'int64')
    except OverflowError:
        return default(A0, U0)
    _MAXU = abs(U).max()
    if isnan(_MAXU) or _MAXU == inf or _MAXU > _INT64_MAX or int(_MAXA) * int(_MAXU) * n**2 > _INT64_MAX:
        return default(A0, U0)

    # CAST_TYPE = 'int' if int(_MAXA) * int(_MAXU) * n**2 <= _INT32_MAX else 'int64'
    # CAST_TYPE = 'int64'
    # A = np_round(A).astype(CAST_TYPE)
    # U = np_round(U).astype(CAST_TYPE)
    A = A.reshape((N, n, n))

    C = (U.T @ A @ U).flatten().tolist()
    q1q22 = q1 * q2**2
    C = [i * q1q22 for i in C]
    C = sp.Matrix(N, m**2, [Rational(i.numerator, i.denominator, gcd = 1) for i in C])
    # C = sp.Matrix(C.reshape((-1, m*m)).tolist()) / (q1 * q2**2)
    return C