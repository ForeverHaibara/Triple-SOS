from collections import defaultdict
from math import gcd
from time import time
from typing import List, Tuple, Dict, Union, Optional

from numpy import argsort
from sympy.external.gmpy import MPQ, MPZ # >= 1.9
from sympy.matrices import MutableDenseMatrix as Matrix
from sympy.matrices.repmatrix import RepMatrix
from sympy.polys.domains import ZZ, QQ, EX, EXRAW # EXRAW >= 1.9
from sympy.polys.matrices.domainmatrix import DomainMatrix # polys.matrices >= 1.8
from sympy.polys.matrices.sdm import SDM

from .matop import is_zz_qq_mat, is_empty_matrix, permute_matrix_rows, rep_matrix_from_dict

_VERBOSE_SOLVE_UNDETERMINED_LINEAR = False
_VERBOSE_SOLVE_CSR_LINEAR = False
_USE_SDM_RREF_DEN = False # has bug in low sympy versions due to behaviour of quo

def _restore_from_compressed(mat: Matrix, mapping: Union[List[int], Dict[int, int]], rows: Optional[int]=None):
    """
    Set new_mat[i] = mat[mapping[i]]

    Parameters
    ----------
    mat : Matrix
        The matrix to be restored.
    mapping : list or dict
        A list or a dict so that new_mat[i] = mat[mapping[i]].
    rows : int, optional
        The number of rows of the new matrix. If None, it will be set to the length of mapping.

    Returns
    -------
    new_mat : Matrix
        The restored matrix.
    """
    mapping = enumerate(mapping) if isinstance(mapping, list) else mapping.items()
    if rows is None: rows = len(mapping)
    rep = mat._rep.rep.to_sdm()
    new_rep = []
    for i, j in mapping:
        repj = rep.get(j)
        if repj:
            new_rep.append((i, repj)) # Shall we make a copy?
    new_rep = dict(new_rep)
    return rep_matrix_from_dict(new_rep, (rows, rep.shape[1]), rep.domain)


def _row_reduce_dict(mat, rows, cols, domain=QQ, normalize_last=True, normalize=True, zero_above=True):
    """
    See also in sympy.matrices.reductions._row_reduce_list
    """
    one, zero = domain.one, domain.zero
    if domain.is_QQ or domain.is_ZZ:
        gcdab = lambda a, b: MPQ(gcd(a.numerator, b.numerator), gcd(a.denominator, b.denominator))
    else:
        gcdab = lambda a, b: one

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
        _gcd = gcdab(a, b)
        a, b = a / _gcd, b / _gcd
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

    sdm = M._rep.to_field().rep
    domain = sdm.domain
    sdm = {k: {k2: i for k2, i in v.items()} for k, v in sdm.to_sdm().items()}

    if _VERBOSE_SOLVE_UNDETERMINED_LINEAR:
        print(">> Time for converting to MPQ:", time() - time0)
        time0 = time()

    mat, pivot_cols, swaps = _row_reduce_dict(sdm, M.shape[0], M.shape[1], domain=domain,
            normalize_last=normalize_last, normalize=normalize, zero_above=zero_above)

    if _VERBOSE_SOLVE_UNDETERMINED_LINEAR:
        print(">> Time for row reduce list:", time() - time0)
        time0 = time()

    mat = rep_matrix_from_dict(mat, M.shape, M._rep.domain.get_field())
    return mat, pivot_cols, swaps


def _rref(M, pivots=True, normalize_last=True):
    """
    See also in sympy.matrices.reductions.rref
    """
    return_pivots = pivots
    if not M._rep.domain.is_Exact:
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
    if isinstance(M._rep.rep, SDM) and _is_column_separated(M):
        return solve_column_separated_linear(M, B)

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


def _is_column_separated(A: Matrix) -> bool:
    seen_cols = set()
    Arep = A._rep.rep.to_sdm()
    for row in Arep.values():
        for col in row.keys():
            if col in seen_cols:
                return False
            seen_cols.add(col)
    # if len(seen_cols) != A.shape[1]:
    #     return False
    return True


def solve_column_separated_linear(A: Matrix, b: Matrix):
    """
    This is a function that solves a special linear system Ax = b => x = x_0 + C * y
    where each column of A has at most 1 nonzero element. For more general cases, use solve_csr_linear.
    Further, we could require some of entries of x to be equal.

    WARNING: IT IS NOT SAFE

    Parameters
    ----------
    A: Matrix
        Sympy matrix that satisfies the condition.
    b: Matrix
        Right-hand side

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

    # toK = lambda x: x # domain.from_sympy
    one, zero = domain.one, domain.zero
    A = A._rep.convert_to(domain).rep.to_sdm() # SDM
    b = b._rep.convert_to(domain).rep.to_sdm() # SDM

    x0 = []
    spaces = []

    def _assert_empty(b, i):
        bi = b.get(i, 0)
        if bi != 0 and bi.get(0, zero) != zero:
            raise ValueError("Linear system has no solution")

    for i in set(range(A.shape[0])) - set(A.keys()):
        _assert_empty(b, i)

    for i, row in A.items():
        row = list(row.items())
        if len(row):
            head, w = row[0]
            bi = b.get(i, 0)
            if bi:
                bi = bi.get(0, zero)
                if bi:
                    v = bi / w
                    x0.append((head, {0: v}))

            if len(row) > 1:
                for head2, w2 in row[1:]:
                    spaces.append({head: w2, head2: -w})

        else:
            _assert_empty(b, i)

    if _VERBOSE_SOLVE_CSR_LINEAR:
        print('>> Solve separated system time:', time() - time0) # fast, < 1 sec over (100 x 10000)
        time0 = time()

    all_cols = []
    for row in A.values():
        all_cols.extend(list(row.keys()))
    all_cols = set(all_cols)

    if len(all_cols) != cols:
        # unseen cols are free
        unseen_cols = set(range(cols)) - set(all_cols)
        spaces.extend([{i: one} for i in unseen_cols])

    x0 = dict(x0)
    spaces = dict(enumerate(spaces))
    x0 = rep_matrix_from_dict(x0, (cols, 1), domain)
    spaces = rep_matrix_from_dict(spaces, (len(spaces), cols), domain).T

    if _VERBOSE_SOLVE_CSR_LINEAR:
        print('>> Matrix restoration time:', time() - time0, 'space shape =', spaces.shape)

    return x0, spaces


def solve_csr_linear(A: Matrix, b: Matrix,
        x0_equal_indices: List[List[int]] = [],
        nonnegative_indices: List[int] = [],
        force_zeros: Dict[int, List[int]] = {}
    ):
    """
    Solve a linear system Ax = b where A is stored in SDM (CSR) format.
    Further, we could require some other properties.

    Parameters
    ----------
    A: Matrix
        Sympy matrix (with preferably SDM format). Non-sdm format
        matrices will be automatically converted to SDM.
    b: Matrix
        Right-hand side.
    x0_equal_indices: List[List[int]]
        If given, it requires some entries of x to be equal.
        Each sublist contains indices of equal elements.
        For example, a symmetric matrix might require its entries to be equal in pairs
        (i, j) and (j, i) for all i, j.
    nonnegative_indices: List[int]
        If given, it requires some entries of x to be nonnegative.
        If there exists an equation c1*x1+...+cn*xn = 0 where x1,...,xn are nonnegative
        and c1,...,cn >= 0, then the solution is x1 = x2 = ... = xn = 0.
        This is useful for positive semidefinite matrices which have nonnegative diagonals.
    force_zeros: Dict[int, List[int]]
        If given, each indices in `force_zeros[i]` must be zero if index `i` is zero.
        This is useful when a positive semidefinite matrix has a zero diagonal entry,
        which implies that the corresponding row and column must be all zeros.

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


    # form the equal indices as a UFS
    ufs, groups = _build_ufs(x0_equal_indices, cols)
    group_keys = list(groups.keys())
    group_inds = {k: i for i, k in enumerate(group_keys)}

    # compress the columns
    cols2 = len(groups)
    domain = Arep.domain
    zero = domain.zero
    A2 = {}
    mapping = [group_inds[ufs[i]] for i in range(cols)]
    for i, row in Arep.items():
        for j, v in row.items():
            A2i = A2.get(i)
            if A2i is None:
                A2i = defaultdict(lambda : zero)
                A2[i] = A2i
            A2i[mapping[j]] += v

    A2 = rep_matrix_from_dict(A2, (A.shape[0], cols2), domain)
    # x0, space = solve_undetermined_linear(A2, b)

    # compress other kwargs
    nonnegative_indices = []#mapping[i] for i in nonnegative_indices]
    new_force_zeros = {}
    for i in force_zeros:
        k = mapping[i]
        if not (k in new_force_zeros):
            new_force_zeros[k] = set()
        new_force_zeros[k].update(set(mapping[j] for j in force_zeros[i]))

    x0, space = _solve_csr_linear_force_zeros(A2, b,
                    nonnegative_indices=nonnegative_indices, force_zeros=new_force_zeros)

    # restore the solution: row[i] = row_compressed[mapping[i]]
    x0 = _restore_from_compressed(x0, mapping, rows=cols)
    space = _restore_from_compressed(space, mapping, rows=cols)
    return x0, space


def _build_ufs(groups: List[List[int]], n: int) -> Tuple[List[int], Dict[int, List[int]]]:
    parent = list(range(n))
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    for group in groups:
        if group:
            base = group[0]
            for x in group[1:]:
                union(base, x)

    new_groups = {}
    findi = [find(i) for i in range(n)]
    for i, f in enumerate(findi):
        if i == f:
            new_groups[i] = []
    for i, f in enumerate(findi):
        new_groups[f].append(i)
    return parent, new_groups


def _solve_csr_linear_force_zeros(A, b, nonnegative_indices=[], force_zeros={}):
    all_zero_inds = set() # all found zeros indices in the loop
    zero_inds = set() # a dynamic queue
    rep = A._rep.rep.to_sdm() # TODO: deepcopy?
    domain = rep.domain
    zero = domain.zero
    signfunc = (lambda x: x >= zero) if domain.is_QQ or domain.is_ZZ or domain.is_RR else (lambda x: False)

    nonzero_rows = set(b._rep.rep.to_sdm().keys())
    zero_rows = set(range(A.shape[0])) - nonzero_rows
    nonnegative_indices = set(nonnegative_indices)

    col_to_rows = None

    def _build_col_to_rows(): # csr to csc:
        col_to_rows = defaultdict(list)
        for i, row in rep.items():
            for j in row.keys():
                col_to_rows[j].append(i)
        return col_to_rows

    def _del_A_col(i, col_to_rows):
        # for Ar in rep.values():
        #     if i in Ar:
        #         del Ar[i]
        coli = col_to_rows.get(i)
        if coli:
            for j in coli:
                repj = rep.get(j)
                if repj:
                    del repj[i]

    def _clear_zero_inds(zero_inds, all_zero_inds, col_to_rows):
        """Handle newly found zero indices by deleting the corresponding columns
        and explore new zero indices from `force_zeros`"""
        while zero_inds:
            # use while-loop but not for-loop since the set will be changed during iter
            i = zero_inds.pop()
            # new zeros are explored by the rule of force_zeros
            all_zero_inds.add(i)
            new_zeros = force_zeros.get(i)
            if new_zeros is not None:
                for j in new_zeros:
                    if not (j in all_zero_inds):
                        zero_inds.add(j)

            # remove the corresponding column of A
            _del_A_col(i, col_to_rows)

    found_new_zeros = True
    while found_new_zeros:
        found_new_zeros = False
        zero_rows_to_remove = []
        for i in zero_rows:
            Ai = rep.get(i)
            if (Ai is not None) and len(Ai) > 0:
                if len(Ai) == 1:
                    # only one entry left, it must be zero
                    zero_inds.add(Ai.popitem()[0])
                    zero_rows_to_remove.append(i) # the row is empty, skip it after handled once
                    del rep[i]
                elif all((signfunc(v) and v in nonnegative_indices) for v in Ai.values()):
                    # all coefficients in this row are nonnegative and all variables are nonnegative
                    # the corresponding indices must be 0
                    for x in Ai:
                        nonnegative_indices.remove(x)
                        zero_inds.add(x)
                    del rep[i]
            else:
                zero_rows_to_remove.append(i)
        for i in zero_rows_to_remove:
            zero_rows.remove(i)

        if len(zero_inds):
            found_new_zeros = True
            if col_to_rows is None:
                col_to_rows = _build_col_to_rows()
            _clear_zero_inds(zero_inds, all_zero_inds, col_to_rows)


    # extract columns associated with nonzero indices
    # print('All zero indices:', all_zero_inds)
    new_A = A
    if len(all_zero_inds):
        nonzero_inds = (list(set(range(A.shape[1])) - all_zero_inds))
        mapping = dict(zip(nonzero_inds, range(len(nonzero_inds))))
        new_rep = {}
        for i in rep:
            new_rep[i] = {mapping[k]: v for k, v in rep[i].items() if k in mapping}
        new_A = rep_matrix_from_dict(new_rep, (A.shape[0], len(nonzero_inds)), domain)

    x0, space = solve_undetermined_linear(new_A, b)
    if len(all_zero_inds):
        x0 = _restore_from_compressed(x0, mapping, rows=A.shape[1])
        space = _restore_from_compressed(space, mapping, rows=A.shape[1])
    return x0, space