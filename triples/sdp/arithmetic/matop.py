from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np
from numpy import ndarray
from sympy import __version__ as _SYMPY_VERSION
from sympy.external.gmpy import MPQ, MPZ # >= 1.9
from sympy.external.importtools import version_tuple
from sympy.matrices import MutableDenseMatrix as Matrix
from sympy.matrices.repmatrix import RepMatrix
from sympy.polys.domains import ZZ, QQ, EX, EXRAW # EXRAW >= 1.9
from sympy.polys.matrices.domainmatrix import DomainMatrix # polys.matrices >= 1.8
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.sdm import SDM

if tuple(version_tuple(_SYMPY_VERSION)) >= (1, 13):
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


def is_empty_matrix(M: Union[Matrix, ndarray], check_all_zeros: bool = False) -> bool:
    """
    Check whether a matrix is zero size. Set check_all_zeros == True to
    check whether all entries are zero.
    """
    if any(_ == 0 for _ in M.shape):
        return True
    if check_all_zeros and not any(M):
        return True
    return False

def size_of_mat(M: Union[Matrix, ndarray]) -> int:
    return int(np.prod(M.shape))

def sqrtsize_of_mat(M: Union[Matrix, ndarray]) -> int:
    return int(np.sqrt(size_of_mat(M)))

def vec2mat(v: Union[Matrix, ndarray]) -> Matrix:
    """
    Convert a vector to a matrix.
    """
    n = sqrtsize_of_mat(v)
    return v.reshape(n, n)

def mat2vec(M: Union[Matrix, ndarray]) -> Matrix:
    """
    Convert a matrix to a vector.
    """
    if isinstance(M, ndarray):
        return M.flatten()
    return M.reshape((size_of_mat(M), 1))



def is_zz_qq_mat(mat):
    """Judge whether a matrix is a ZZ/QQ RepMatrix."""
    return isinstance(mat, RepMatrix) and mat._rep.domain in (ZZ, QQ)

def _cast_sympy_matrix_to_numpy(sympy_mat_rep: RepMatrix, dtype: str = 'int64') -> ndarray:
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

    arr = np.zeros((rows, cols), dtype=dtype)
    if data_list:
        arr[row_indices, col_indices] = data_list
    return arr

def _cast_list_to_sympy_matrix(rows: int, cols: int, lst: List[int]) -> Matrix:
    """Convert a list of INTEGER values to a sympy matrix efficiently."""
    sdm = {}
    for i in range(rows):
        row = {}
        for j, v in enumerate(lst[i*cols:(i+1)*cols]):
            if v:
                row[j] = MPZ(v)
        if row:
            sdm[i] = row
    return Matrix._fromrep(DomainMatrix.from_rep(SDM(sdm, (rows, cols), ZZ)))


def permute_matrix_rows(matrix, permutation):
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