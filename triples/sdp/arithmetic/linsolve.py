from collections import defaultdict
from math import gcd
from time import time
from typing import List, Tuple

from numpy import argsort
from sympy.external.gmpy import MPQ, MPZ # >= 1.9
from sympy.matrices import MutableDenseMatrix as Matrix
from sympy.matrices.repmatrix import RepMatrix
from sympy.polys.domains import ZZ, QQ, EX, EXRAW # EXRAW >= 1.9
from sympy.polys.matrices.domainmatrix import DomainMatrix # polys.matrices >= 1.8
from sympy.polys.matrices.sdm import SDM

from .matop import is_zz_qq_mat, is_empty_matrix, permute_matrix_rows

_VERBOSE_SOLVE_UNDETERMINED_LINEAR = False
_VERBOSE_SOLVE_CSR_LINEAR = False
_USE_SDM_RREF_DEN = False # has bug in low sympy versions due to behaviour of quo

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
    return_pivots = pivots
    if not is_zz_qq_mat(M):
        return M.rref(pivots=return_pivots, normalize_last=normalize_last)

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

    if return_pivots:
        mat = (mat, pivots)
    return mat



def solve_undetermined_linear(M: Matrix, B: Matrix) -> Tuple[Matrix, Matrix]:
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
    V        = V.col_join(-A.eye(V.cols))
    vt       = v[:rank, :]
    vt       = vt.col_join(A.zeros(V.cols, vt.cols))

    # Undo permutation
    inv_permutation = argsort(permutation).tolist()
    V2 = permute_matrix_rows(V, inv_permutation)
    vt2 = permute_matrix_rows(vt, inv_permutation)
    # for k in range(col):
    #     V2[permutation[k], :] = V[k, :]
    #     vt2[permutation[k]] = vt[k]

    if _VERBOSE_SOLVE_UNDETERMINED_LINEAR:
        print(">> Time for undo permutation:", time() - time0,
              'V2, vt2 shape =', V2.shape, vt2.shape)

    return vt2, V2


def solve_nullspace(A: Matrix) -> Matrix:
    """
    Compute the null space of a matrix A.
    If A is full-rank and has shape m x n (m > n), then the null space has shape m x (m-n).
    """
    # try:
    #     x0, space = solve_undetermined_linear(A.T, sp.zeros(A.cols, 1))
    # except ValueError:
    #     return Matrix.zeros(A.rows, 0)
    # return space
    m = Matrix.hstack(*A.T.nullspace())
    if is_empty_matrix(m):
        return A.zeros(A.shape[0], 0)
    return m

def solve_columnspace(A: Matrix) -> Matrix:
    """
    Compute the column space of a matrix A.
    If A is full-rank and has shape m x n (m < n), then the column space has shape m x n.
    """
    m = Matrix.hstack(*A.columnspace())
    if is_empty_matrix(m):
        return A.zeros(A.shape[0], 0)
    return m

def solve_column_separated_linear(A: Matrix, b: Matrix, x0_equal_indices: List[List[int]] = []):
    """
    This is a function that solves a special linear system Ax = b => x = x_0 + C * y
    where each column of A has at most 1 nonzero element. For more general cases, use solve_csr_linear.
    Further, we could require some of entries of x to be equal.

    Parameters
    ----------
    A: Matrix
        Sympy matrix that satisfies the condition.
    b: Matrix
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

    to_mat = lambda x, shape: Matrix._fromrep(DomainMatrix.from_rep(SDM(x, shape, domain)))
    # x0, space = Matrix(x0), Matrix(spaces).T
    x0 = to_mat(x0, (cols, 1))
    spaces = to_mat(spaces, (len(spaces), cols)).T

    if _VERBOSE_SOLVE_CSR_LINEAR:
        print('>> Matrix restoration time:', time() - time0, 'space shape =', spaces.shape)

    return x0, spaces


def solve_csr_linear(A: Matrix, b: Matrix, x0_equal_indices: List[List[int]] = []):
    """
    Solve a linear system Ax = b where A is stored in SDM (CSR) format.
    Further, we could require some of entries of x to be equal.

    Parameters
    ----------
    A: Matrix
        Sympy matrix (with preferably SDM format).
    b: Matrix
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
        return Matrix._fromrep(DomainMatrix.from_rep(sdm))

    x0 = _restore_from_compressed(x0_compressed, mapping)
    space = _restore_from_compressed(space_compressed, mapping)
    return x0, space

