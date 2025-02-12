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

# from fractions import Fraction
from collections import defaultdict
from math import gcd
from time import time
from typing import List, Tuple, Union, Optional

from numpy import array as np_array, around as np_round, zeros as np_zeros, iinfo as np_iinfo
from numpy import isnan, inf, unique
import sympy as sp
from sympy import __version__ as _SYMPY_VERSION
from sympy import Rational
from sympy.external.gmpy import MPQ, MPZ # >= 1.9
from sympy.matrices.repmatrix import RepMatrix
from sympy.polys.domains import ZZ, QQ, EX, EXRAW # EXRAW >= 1.9
from sympy.polys.matrices.domainmatrix import DomainMatrix # polys.matrices >= 1.8
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.sdm import SDM

def _is_version_greater_than(v1, v2) -> bool:
    """Version comparison. No dependencies on distutils
    (which has been deprecated since Python 3.10)."""
    def _version(s):
        from re import findall
        parts = []
        for comp in s.split('.'):
            for elem in findall(r'\d+|\D+', comp):
                parts.append(int(elem) if elem.isdigit() else elem)
        return tuple(parts)
    for i, j in zip(_version(v1), _version(v2)):
        if isinstance(i, str) or isinstance(j, str): i, j = str(i), str(j)
        if i > j: return True
        if i < j: return False
    return bool(len(_version(v1)) >= len(_version(v2)))

if _is_version_greater_than(_SYMPY_VERSION, '1.13'):
    from sympy.polys.matrices.dfm import DFM

    primitive = lambda x: x.primitive()
else:
    class _DFM_dummy: ...
    DFM = _DFM_dummy

    from sympy.polys.densetools import dup_primitive

    def primitive(self: DomainMatrix):
        K = self.domain
        dok = self.rep.to_dok()
        elements, data = list(dok.values()), list(dok.keys())
        content, prims = dup_primitive(elements, K)
        sdm = defaultdict(dict)
        for (i, j), v in zip(data, prims):
            sdm[i][j] = v
        M_primitive = self.from_rep(SDM(sdm, self.shape, K))
        return content, M_primitive

from .utils import Mat2Vec, is_empty_matrix

_INT32_MAX = np_iinfo('int32').max # 2147483647
_INT64_MAX = np_iinfo('int64').max # 9223372036854775807

# for dev purpose only
_VERBOSE_MATMUL_MULTIPLE = False
_VERBOSE_SOLVE_UNDETERMINED_LINEAR = False
_VERBOSE_SOLVE_CSR_LINEAR = False
_USE_SDM_RREF_DEN = False # has bug in low sympy versions due to behaviour of quo


def _lcm(x, y):
    """
    LCM func for python internal integers. DO NOT USE SYMPY BECAUSE IT IS SLOW
    WHEN CONVERTING TO SYMPY INTEGERS.
    """
    return x * y // gcd(x, y)

def is_zz_qq_mat(mat):
    """Judge whether a matrix is a ZZ/QQ RepMatrix."""
    return isinstance(mat, RepMatrix) and mat._rep.domain in (ZZ, QQ)

def _row_reduce_dict(mat, rows, cols, normalize_last=True, normalize=True, zero_above=True):
    """
    See also in sympy.matrices.reductions._row_reduce_list
    """
    one, zero = QQ.one, QQ.zero

    def row_swap(i, j):
        # mat[i*cols:(i + 1)*cols], mat[j*cols:(j + 1)*cols] = \
        #     mat[j*cols:(j + 1)*cols], mat[i*cols:(i + 1)*cols]
        mati = mat.get(i, None)
        matj = mat.get(j, None)
        if mati is not None and matj is not None:
            mat[i], mat[j] = matj, mati
        elif mati is not None:
            mat[j] = mat.pop(i)
        elif matj is not None:
            mat[i] = mat.pop(j)

    def cross_cancel(a, i, b, j):
        """Does the row op row[i] = a*row[i] - b*row[j]"""
        # q = (j - i)*cols
        gcdab = MPQ(gcd(a.numerator, b.numerator), gcd(a.denominator, b.denominator))
        a, b = a / gcdab, b / gcdab
        mati = mat.get(i, {})
        if mati:
            if a:
                mati = {k: a*v for k, v in mati.items()}
            else:
                mati = {}

        if b:
            matj = mat.get(j, {})
            for k, v in matj.items():
                new_val = mati.get(k, zero) - b*v
                if new_val:
                    mati[k] = new_val
                elif v: # mati[k] == b*v != 0
                    del mati[k]
        if mati:
            mat[i] = mati
        elif i in mat:
            del mat[i]

    def _find_reasonable_pivot_naive(piv_row, piv_col):
        """Find mat[piv_row + piv_offset, piv_col] != 0."""
        for row in range(piv_row, rows):
            val = mat.get(row, {}).get(piv_col, zero)
            if val != zero:
                return row - piv_row, val
        return None, None

    piv_row, piv_col = 0, 0
    pivot_cols = []
    swaps = []

    # use a fraction free method to zero above and below each pivot
    while piv_col < cols and piv_row < rows:
        pivot_offset, pivot_val = _find_reasonable_pivot_naive(piv_row, piv_col)

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
            if mat.get(i, None) is None:
                mat[i] = {j: one}
            else:
                mat[i][j] = one
            # for p in range(i*cols + j + 1, (i + 1)*cols):
            #     mat[p] = (mat[p] / pivot_val)
            mati = mat[i]
            for k, v in mati.items():
                if k > j:
                    mati[k] = v / pivot_val
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
            mat_row = mat.get(row, None)
            if not mat_row:
                continue

            val = mat_row.get(piv_col, zero)
            if val == zero:
                continue

            cross_cancel(pivot_val, row, val, piv_row)
        piv_row += 1

    # normalize each row
    if normalize_last is True and normalize is True:
        for piv_i, piv_j in enumerate(pivot_cols):
            # pivot_val = mat[piv_i*cols + piv_j]
            mat_piv_i = mat.get(piv_i, None)
            if mat_piv_i is None:
                mat_piv_i = {}
                mat[piv_i] = mat_piv_i
            pivot_val = mat_piv_i.get(piv_j, zero)
            mat_piv_i[piv_j] = one
            # for p in range(piv_i*cols + piv_j + 1, (piv_i + 1)*cols):
            #     mat[p] = (mat[p] / pivot_val)
            for k, v in mat_piv_i.items():
                if k > piv_j:
                    mat_piv_i[k] = v / pivot_val

    return mat, tuple(pivot_cols), tuple(swaps)


def _row_reduce(M, normalize_last=True,
                normalize=True, zero_above=True):
    """
    See also in sympy.matrices.reductions._row_reduce
    It converts sympy Rational matrix to MPQ without checking.

    M should be a RepMatrix with domain ZZ or QQ.
    """
    if _VERBOSE_SOLVE_UNDETERMINED_LINEAR:
        time0 = time()

    sdm = M._rep.to_field().rep.to_sdm()
    sdm = {k: {k2: i for k2, i in v.items()} for k, v in sdm.items()}

    if _VERBOSE_SOLVE_UNDETERMINED_LINEAR:
        print(">> Time for converting to MPQ:", time() - time0)
        time0 = time()

    mat, pivot_cols, swaps = _row_reduce_dict(sdm, M.shape[0], M.shape[1],
            normalize_last=normalize_last, normalize=normalize, zero_above=zero_above)

    if _VERBOSE_SOLVE_UNDETERMINED_LINEAR:
        print(">> Time for row reduce list:", time() - time0)
        time0 = time()

    mat = M._fromrep(DomainMatrix.from_rep(SDM(mat, M.shape, M._rep.domain.get_field())))

    return mat, pivot_cols, swaps


def _rref(M, pivots=True, normalize_last=True):
    """
    See also in sympy.matrices.reductions.rref
    """
    if not is_zz_qq_mat(M):
        return M.rref(pivots=pivots, normalize_last=normalize_last)

    if _USE_SDM_RREF_DEN:
        # sdm_rref_den is implemented in sympy since 1.13
        # even if we copy the code of sdm_rref_den,
        # it still has bugs on low sympy versions
        # due to the behaviour of domain.quo on Rings (e.g. ZZ)
        from sympy.polys.matrices.sdm import sdm_rref_den
        sdm = M._rep.rep.to_sdm()
        K = sdm.domain

        sdm_rref_dict, den, pivots = sdm_rref_den(sdm, K)
        sdm_rref = sdm.new(sdm_rref_dict, sdm.shape, sdm.domain)
        dM = DomainMatrix.from_rep(sdm_rref)
        if den != 1:
            dM = dM.to_field()
            dM = dM * K.get_field().quo(K.one, den)

        mat = M.__class__._fromrep(dM)
    else:
        mat, pivots, _ = _row_reduce(M, normalize_last=normalize_last, normalize=True, zero_above=True)

    if pivots:
        mat = (mat, pivots)
    return mat

def _permute_matrix_rows(matrix, permutation):
    """Fast operation of matrix rowswap."""
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
    A, pivots = _rref(aug, normalize_last=False)

    
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



def solve_column_separated_linear(A: sp.Matrix, b: sp.Matrix, x0_equal_indices: List[List[int]] = []):
    """
    This is a function that solves a special linear system Ax = b => x = x_0 + C * y
    where each column of A has at most 1 nonzero element. For more general cases, use solve_csr_linear.
    Further, we could require some of entries of x to be equal.

    Parameters
    ----------
    A: sp.Matrix
        Sympy matrix that satisfies the condition.
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
    cols = A.shape[1]
    domain = A._rep.domain.unify(b._rep.domain).get_field()

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

    toK = lambda x: x # domain.from_sympy
    one, zero = domain.one, domain.zero
    A = A._rep.convert_to(domain).rep.to_sdm() # SDM
    b = b._rep.convert_to(domain).rep.to_sdm() # SDM

    x0 = []
    spaces = []
    for i, row in A.items():
        row = list(row.items())
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


def solve_csr_linear(A: sp.Matrix, b: sp.Matrix, x0_equal_indices: List[List[int]] = []):
    """
    Solve a linear system Ax = b where A is stored in SDM (CSR) format.
    Further, we could require some of entries of x to be equal.

    Parameters
    ----------
    A: sp.Matrix
        Sympy matrix (with preferably SDM format).
    b: sp.Matrix
        Right-hand side
    x0_equal_indices: List[List[int]]
        Each sublist contains indices of equal elements.

    Returns
    ---------
    x0: Matrix
        One solution of Ax = b.
    space: Matrix
        All solution x is in the form of x0 + space @ y.
    """
    cols = A.shape[1]
    if _VERBOSE_SOLVE_CSR_LINEAR:
        print('SolveCsrLinear A shape', A.shape)
        time0 = time()

    Arep = A._rep.rep.to_sdm()

    # check whether there is at most one nonzero element in each column
    if isinstance(A, RepMatrix) and isinstance(b, RepMatrix):
        _column_separated = True
        seen_cols = set()
        for row in Arep.values():
            for col, _ in row.items():
                if col in seen_cols:
                    _column_separated = False
                    break
                seen_cols.add(col)
            if not _column_separated:
                break

        if _column_separated:
            if _VERBOSE_SOLVE_CSR_LINEAR:
                print('>> Column Separated System recognized', time() - time0)
            return solve_column_separated_linear(A, b, x0_equal_indices)

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
    domain = Arep.domain
    zero = domain.zero
    A2 = {}
    for i, row in Arep.items():
        for j, v in row.items():
            A2i = A2.get(i)
            if A2i is None:
                A2i = defaultdict(lambda : zero)
                A2[i] = A2i
            A2i[group_inds[ufs[j]]] += v

    A2 = A._fromrep(DomainMatrix.from_rep(SDM(A2, (A.shape[0], cols2), domain)))
    x0_compressed, space_compressed = solve_undetermined_linear(A2, b)

    # restore the solution: row[i] = row_compressed[mapping[i]]
    mapping = [group_inds[ufs[i]] for i in range(cols)]
    def _restore_from_compressed(mat, mapping):
        # Set new_mat[i] = mat[mapping[i]]
        rep = mat._rep.rep.to_sdm()
        new_rep = []
        for i, j in enumerate(mapping):
            repj = rep.get(j)
            if repj:
                new_rep.append((i, repj)) # Shall we make a copy?
        new_rep = dict(new_rep)
        sdm = SDM(new_rep, (cols, rep.shape[1]), rep.domain)
        return sp.Matrix._fromrep(DomainMatrix.from_rep(sdm))

    x0 = _restore_from_compressed(x0_compressed, mapping)
    space = _restore_from_compressed(space_compressed, mapping)
    return x0, space

def _cast_sympy_matrix_to_numpy(sympy_mat_rep, dtype: str = 'int64'):
    """Cast a sympy RepMatrix on ZZ to numpy int64 matrix."""
    rows, cols = sympy_mat_rep.shape
    items = list(sympy_mat_rep.rep.to_dok().items()) # avoid .iter_items() for version compatibility
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

def _cast_list_to_sympy_matrix(rows: int, cols: int, lst: List[int]) -> sp.Matrix:
    """Convert a list of INTEGER values to a sympy matrix efficiently."""
    sdm = {}
    for i in range(rows):
        row = {}
        for j, v in enumerate(lst[i*cols:(i+1)*cols]):
            if v:
                row[j] = MPZ(v)
        if row:
            sdm[i] = row
    return sp.Matrix._fromrep(DomainMatrix.from_rep(SDM(sdm, (rows, cols), ZZ)))



def matmul(A: sp.Matrix, B: sp.Matrix, return_shape = None) -> sp.Matrix:
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
        return sp.zeros(A.shape[0], B.shape[1])
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

    C = (A @ B).flatten().tolist()
    q1q2 = q1 * q2
    return_shape = return_shape or (A0.shape[0], B0.shape[1])
    C = _cast_list_to_sympy_matrix(*return_shape, C) * q1q2
    return C


def matmul_multiple(A: sp.Matrix, B: sp.Matrix) -> sp.Matrix:
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
            Aij = Mat2Vec.vec2mat(A[i,:])
            eq = matmul(Aij, B, return_shape = (1, Aij.shape[0]*B.shape[1]))
            eq_mat.append(eq)
        eq_mat = sp.Matrix.vstack(*eq_mat)
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

    C = (A @ B).flatten().tolist()

    if _VERBOSE_MATMUL_MULTIPLE:
        print('>> Time for numpy matmul:', time() - time0) # very fast (can be ignored)
        time0 = time()

    q1q2 = q1 * q2
    C = _cast_list_to_sympy_matrix(N, n*m, C) * q1q2

    if _VERBOSE_MATMUL_MULTIPLE:
        print('>> Time for casting to sympy:', time() - time0) # < 1 sec over (800*8000)
        time0 = time()

    return C


def symmetric_bilinear(U: sp.Matrix, A: sp.Matrix, is_A_vec: bool = False, return_shape = None):
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
        A = Mat2Vec.vec2mat(A)
    M = matmul(U.T, matmul(A, U))
    # M = U.T * A * U

    if return_shape is not None:
        return M.reshape(return_shape[0], return_shape[1])
    return M


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
            eq = symmetric_bilinear(U, A[i,:], is_A_vec = True, return_shape = (1, U.shape[1]**2))
            eq_mat.append(eq)
        eq_mat = sp.Matrix.vstack(*eq_mat)
        return eq_mat

    N = A.shape[0]
    if N == 0:
        return sp.Matrix.zeros(0, U.shape[1]**2)

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

    C = (U.T @ A @ U).flatten().tolist()

    if _VERBOSE_MATMUL_MULTIPLE:
        time0 = time()

    q1q22 = q1 * q2**2
    C = _cast_list_to_sympy_matrix(N, m**2, C) * q1q22

    if _VERBOSE_MATMUL_MULTIPLE:
        print('>> Time for casting to sympy Bilinear:', time() - time0)

    return C