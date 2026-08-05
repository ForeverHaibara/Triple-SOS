"""
This module contains very low level apis for matrix operations.

In SymPy, Matrix([...]) initializer needs to convert each SymPy element to
a domain element, e.g. MPQ or MPZ for rational or integers. This conversion
is very slow for large matrices. This module provides a faster way for
basic matrix operations.
"""

from collections import defaultdict
from time import perf_counter
from typing import (List, Dict, Tuple, Union, Optional, Callable, Set,
                    Any, TYPE_CHECKING, TypeVar)

import numpy as np
from numpy import ndarray
from scipy.sparse import spmatrix, csr_matrix
from sympy import __version__ as _SYMPY_VERSION
from sympy.external.gmpy import MPZ # >= 1.9
from sympy.external.importtools import version_tuple
from sympy import Float
from sympy.matrices import MatrixBase
from sympy.matrices import MutableDenseMatrix as Matrix
from sympy.matrices.repmatrix import RepMatrix
from sympy.polys.domains import ZZ, RR, CC # EXRAW >= 1.9
from sympy.polys.matrices.domainmatrix import DomainMatrix # polys.matrices >= 1.8
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.sdm import SDM

if TYPE_CHECKING:
    from sympy import Basic
    from sympy.polys.domains import Domain

if tuple(version_tuple(_SYMPY_VERSION)) >= (1, 13):
    from sympy.polys.matrices.dfm import DFM

    primitive = lambda self: self.primitive()
else:
    class _DFM_dummy: ...
    DFM = _DFM_dummy # type: ignore

    from sympy.polys.densetools import dup_primitive

    def primitive(self: DomainMatrix):
        K = self.domain
        dok = self.rep.to_dok()
        elements, data = list(dok.values()), list(dok.keys())
        content, prims = dup_primitive(elements, K)
        sdm: Dict[int, Dict[int, Any]] = defaultdict(dict)
        for (i, j), v in zip(data, prims):
            sdm[i][j] = v
        M_primitive = self.from_rep(SDM(sdm, self.shape, K))
        return content, M_primitive

try:
    from flint import fmpq, fmpz
    FLINT_TYPE: Any = (fmpq, fmpz)
except ImportError:
    FLINT_TYPE = ()


MatrixLike = Union[MatrixBase, ndarray, spmatrix]
MatrixT = TypeVar('MatrixT', MatrixBase, ndarray, spmatrix)


class ArithmeticTimeout(Exception):
    @classmethod
    def make_checker(cls, time_limit: Optional[Union[Callable, float]] = None) -> Callable[[], None]:
        """Returns a callable that raises an Exception when called if time exceeds current_time + time_limit."""
        if time_limit is None:
            return lambda : None
        if callable(time_limit):
           return time_limit
        future = perf_counter() + time_limit
        def checker():
            if perf_counter() > future:
                raise cls()
        return checker

def is_empty_matrix(M: Union[Matrix, ndarray], check_all_zeros: bool = False) -> bool:
    """
    Check whether a matrix is zero. Set check_all_zeros == True to
    check whether all entries are zero.

    Parameters
    ----------
    M : Matrix or ndarray
        The matrix to be checked.
    check_all_zeros : bool, optional
        Whether to check whether all entries are zero. If False,
        only check whether the size of the matrix is zero. Default is False.

    Examples
    --------
    >>> from sympy import Matrix
    >>> M = Matrix([[0, 0], [0, 0]])
    >>> is_empty_matrix(M)
    False
    >>> is_empty_matrix(M, check_all_zeros=True)
    True
    >>> is_empty_matrix(Matrix.zeros(15, 0))
    True
    >>> is_empty_matrix(Matrix.eye(3))
    False
    """
    if any(_ == 0 for _ in M.shape):
        return True
    if check_all_zeros:
        if isinstance(M, ndarray):
            return bool(np.all(M == 0))
        if isinstance(M, RepMatrix):
            rep = M._rep
            zero = rep.domain.zero
            for _, v in rep.to_dok():
                if v != zero:
                    return False
            return True
        return not any(M)
    return False

def size_of_mat(M: MatrixLike) -> int:
    """
    Return the size of a matrix.

    Parameters
    ----------
    M : Matrix or ndarray
        The matrix to be computed.

    Examples
    ----------
    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2], [3, 4], [5, 6]])
    >>> size_of_mat(M)
    6
    >>> size_of_mat(Matrix.zeros(3, 0))
    0
    """
    if len(M.shape) == 0:
        return 0
    return int(np.prod(M.shape))

def sqrtsize_of_mat(M: Union[MatrixLike, int]) -> int:
    """
    Return the int square root of the size of a matrix. This is
    helpful to infer the size of a symmetric matrix from its vector form.

    Parameters
    ----------
    M : Matrix or ndarray or int
        The matrix to be computed. If M is an integer, return the int square root of M.

    Examples
    ----------
    >>> from sympy import Matrix
    >>> sqrtsize_of_mat(Matrix([1,2,3,4,5,6,7,8,9]))
    3
    >>> sqrtsize_of_mat(100)
    10
    """
    if isinstance(M, int):
        return int(np.round(np.sqrt(M)))
    return int(np.round(np.sqrt(size_of_mat(M))))

def reshape(A: MatrixT, shape: Tuple[int, int]) -> MatrixT:
    """
    Reshape a matrix to a new shape. This function maintains the domain
    of SymPy RepMatrix for low SymPy versions.

    Parameters
    ----------
    A : Matrix or ndarray
        The matrix to be reshaped.
    shape : Tuple[int, int]
        The new shape of the matrix.

    Examples
    ----------
    >>> from sympy import Matrix, sqrt, QQ
    >>> M = Matrix([[1+sqrt(2), 2], [3, 4-sqrt(2)]])
    >>> reshape(M, (1, 4))
    Matrix([[1 + sqrt(2), 2, 3, 4 - sqrt(2)]])

    >>> dom = QQ.algebraic_field(sqrt(2))
    >>> M = Matrix._fromrep(M._rep.convert_to(dom))
    >>> M._rep.domain
    QQ<sqrt(2)>
    >>> reshape(M, (1, 4))._rep.domain
    QQ<sqrt(2)>
    """
    if A.shape == shape:
        return A
    if isinstance(A, RepMatrix):
        rep = A._rep.rep
        n, m = A.shape
        n2, m2 = shape
        f = lambda row, col: divmod(row*m + col, m2)
        dt = {f(i, j): v for (i, j), v in rep.to_dok().items()}
        dt_by_row: Dict[int, Dict[int, Any]] = {}
        for (i, j), v in dt.items():
            if i not in dt_by_row:
                dt_by_row[i] = {}
            dt_by_row[i][j] = v
        return A._fromrep(DomainMatrix.from_rep(
            SDM(dt_by_row, shape, A._rep.domain)
        ))
    if isinstance(A, spmatrix):
        return A.__class__(A.reshape(shape))
    return A.reshape(*shape)


def vec2mat(v: MatrixT) -> MatrixT:
    """
    Convert a vector to a symmetric matrix.

    Parameters
    ----------
    v : Matrix or ndarray
        The vector to be converted.

    Examples
    ----------
    >>> from sympy import Matrix
    >>> v = Matrix([1, 2, 3, 2, 4, 5, 3, 5, 7])
    >>> vec2mat(v)
    Matrix([
    [1, 2, 3],
    [2, 4, 5],
    [3, 5, 7]])
    """
    n = sqrtsize_of_mat(v)
    return reshape(v, (n, n))

def mat2vec(M: MatrixT) -> MatrixT:
    """
    Convert a matrix to a vector.

    Parameters
    ----------
    M : Matrix or ndarray
        The matrix to be converted.

    Examples
    ----------
    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2, 3], [2, 4, 5], [3, 5, 7]])
    >>> print(mat2vec(M))
    Matrix([[1], [2], [3], [2], [4], [5], [3], [5], [7]])
    """
    if isinstance(M, ndarray):
        return M.flatten()
    return reshape(M, (size_of_mat(M), 1))


def rep_matrix_from_dict(x: Dict[int, Dict[int, Any]], shape: Tuple[int, int], domain: 'Domain') -> Matrix:
    """
    Create a SymPy RepMatrix from a dictionary of domain elements.

    Parameters
    ----------
    x : Dict[int, Dict[int, Any]]
        The x[row][col] = value dictionary where each value is a domain element.
        Zeros should not be stored.
    shape : Tuple[int, int]
        The shape of the matrix.
    domain : Domain
        The domain of the matrix.

    Examples
    ----------
    >>> from sympy import QQ
    >>> x = {0: {1: QQ(2,5)}, 1: {0: QQ(3,5), 1: QQ(4,5)}}
    >>> rep_matrix_from_dict(x, (3, 2), QQ)
    Matrix([
    [  0, 2/5],
    [3/5, 4/5],
    [  0,   0]])
    """
    return Matrix._fromrep(DomainMatrix.from_rep(SDM(x, shape, domain)))

def rep_matrix_from_list(x: Union[List, List[List]], shape: Union[int, Tuple[int, int]], domain: 'Domain') -> Matrix:
    """
    Create a SymPy RepMatrix from a list of domain elements.

    Parameters
    ----------
    x : List or List[List]
        If x is an 1D list, it is treated as a column vector,
        and the return is a matrix with shape (len(x), 1).
        Otherwise, x is treated as a dense matrix.
        Each value of x must be a domain element.
        Zeros are allowed but they will be filtered out by this function.
    shape: int or Tuple[int, int]
        The shape of the matrix. If x is an 1D list, shape must be an integer.
        Otherwise, shape must be a tuple of two integers.
    domain : Domain
        The domain of the matrix.

    Examples
    ----------
    >>> from sympy import QQ, ZZ
    >>> x = [QQ(2,5), QQ(3,5), QQ(4,5)]
    >>> rep_matrix_from_list(x, 3, QQ)
    Matrix([
    [2/5],
    [3/5],
    [4/5]])

    >>> x = [[ZZ(0), ZZ(1), ZZ(2)], [ZZ(3), ZZ(4), ZZ(5)]]
    >>> rep_matrix_from_list(x, (2, 3), ZZ)
    Matrix([
    [0, 1, 2],
    [3, 4, 5]])
    """
    # filter out zeros in the list, which should not be stored in a sparse rep
    zero = domain.zero
    if isinstance(shape, int):
        y = {i: {0: j} for i, j in enumerate(x) if j != zero}
        shape = (shape, 1)
    else:
        y = {}
        for i in range(shape[0]):
            xi = x[i]
            yi = {}
            for j in range(shape[1]):
                if xi[j] != zero:
                    yi[j] = xi[j]
            if len(yi):
                y[i] = yi
    return rep_matrix_from_dict(y, shape, domain)


def is_zz_qq_mat(mat) -> bool:
    """
    Judge whether a matrix is a SymPy ZZ/QQ RepMatrix.

    Parameters
    ----------
    mat : Matrix
        The matrix to be checked.

    Examples
    ----------
    >>> from sympy import Matrix
    >>> is_zz_qq_mat(Matrix([1, 2, 3, 4])/5)
    True
    >>> is_zz_qq_mat(Matrix([1, 2, 3, 4]) * 0.1234)
    False
    """
    return isinstance(mat, RepMatrix) and (mat._rep.domain.is_ZZ or mat._rep.domain.is_QQ)

def is_numerical_mat(mat: MatrixLike) -> bool:
    """
    Judge whether a matrix is numerical, including RR, EX(RAW) with Float and numpy arrays.
    """
    if isinstance(mat, RepMatrix):
        dom = mat._rep.domain
        if not dom.is_Exact:
            return True
        if (dom.is_EX or dom.is_EXRAW) and mat.has(Float):
            return True
    elif isinstance(mat, (ndarray, spmatrix)):
        return True
    elif isinstance(mat, MatrixBase) and mat.has(Float):
        return True
    return False

def free_symbols_of_mat(mat: MatrixLike) -> Set['Basic']:
    """
    Get the free symbols of a matrix.
    """
    if isinstance(mat, RepMatrix) and mat._rep.domain.is_Numerical:
        return set()
    if isinstance(mat, MatrixBase):
        return mat.free_symbols
    return set()


def _cast_list_to_sympy_matrix(rows: int, cols: int, lst: List[int]) -> Matrix:
    """Convert a list of INTEGER values to a sympy matrix efficiently. Internal."""
    sdm = {}
    for i in range(rows):
        row = {}
        for j, v in enumerate(lst[i*cols:(i+1)*cols]):
            if v:
                row[j] = MPZ(v)
        if row:
            sdm[i] = row
    return rep_matrix_from_dict(sdm, (rows, cols), ZZ)

def _csr_to_dict_of_dict(csr_mat: spmatrix) -> Dict[int, Dict[int, Any]]:
    """Convert a CSR matrix to a dictionary of dictionaries. Internal."""
    # csr_mat = csr_mat.tocsr()
    dod = {}
    for row in range(csr_mat.shape[0]):
        start = csr_mat.indptr[row]
        end = csr_mat.indptr[row + 1]
        cols = csr_mat.indices[start:end].tolist()
        values = csr_mat.data[start:end].tolist()
        if len(cols):  # Only include rows with non-zero elements
            dod[row] = dict(zip(cols, values))
    return dod

def rep_matrix_from_numpy(arr: Union[ndarray, spmatrix]) -> Matrix:
    """
    Cast a numpy matrix to a sympy Matrix by handling dtypes carefully.

    Parameters
    ----------
    arr : ndarray or spmatrix
        The numpy matrix to be casted, can be either 1D or 2D.

    Examples
    ----------
    >>> import numpy as np
    >>> arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    >>> rep_matrix_from_numpy(arr)
    Matrix([
    [1, 2, 3],
    [4, 5, 6]])

    >>> arr = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    >>> M = rep_matrix_from_numpy(arr); print(M)
    Matrix([[1.10000000000000], [2.20000000000000], [3.30000000000000]])
    >>> M._rep.domain
    RR
    """
    if isinstance(arr, Matrix):
        return arr
    if isinstance(arr, ndarray) and np.issubdtype(arr.dtype, np.integer):
        shape = arr.shape if len(arr.shape) == 2 else (arr.shape[0], 1)
        return _cast_list_to_sympy_matrix(shape[0], shape[1], arr.flatten().tolist())

    conv: Optional[Callable[[Any], Any]] = None
    domain: Any = RR
    if np.issubdtype(arr.dtype, np.integer):
        domain = ZZ
        conv = lambda x: MPZ(int(x))
    elif np.issubdtype(arr.dtype, np.floating):
        domain = RR
        conv = RR.convert
    elif np.issubdtype(arr.dtype, np.complexfloating):
        domain = CC
        conv = CC.convert

    if conv is not None:
        if isinstance(arr, ndarray):
            if len(arr.shape) == 2:
                lst = [[conv(_) for _ in row] for row in arr.tolist()]
                return rep_matrix_from_list(lst, arr.shape, domain)
            else:
                values = [conv(_) for _ in arr.tolist()]
                return rep_matrix_from_list(values, arr.shape[0], domain)
        elif isinstance(arr, spmatrix):
            dt = _csr_to_dict_of_dict(arr.tocsr())
            dt = {r: {c: conv(v) for c, v in dt[r].items()} for r in dt}
            return rep_matrix_from_dict(dt, arr.shape, domain)

    # fallback to default constructor
    if isinstance(arr, spmatrix):
        arr = arr.toarray()
    return Matrix(arr.tolist())


def _rep_matrix_to_data(M, dtype: Any = np.float64) -> Optional[Tuple[List, List, List]]:
    """
    Try to convert a RepMatrix to (data, row_indices, col_indices).
    """
    dtype = np.dtype(dtype)
    f = None
    if isinstance(M, (RepMatrix, DomainMatrix)):
        if isinstance(M, RepMatrix):
            dM = M._rep # it is domain matrix
        else:
            dM = M

        # for domains like QQ[x], QQ(x), elements cannot be converted
        # directly to int/float/complex, and should be carefully handled
        wrapper = lambda f: f
        dom = dM.domain
        if dom.is_PolynomialRing:
            def wrapper(f):
                def _f(x):
                    lt = x.LT
                    if any(lt[0]):
                        raise TypeError('Cannot convert PolyElement.')
                    return f(lt[1])
                return _f
        elif dom.is_FractionField:
            def wrapper(f):
                def _f(x):
                    x1, x2 = x.numer, x.denom
                    lt1, lt2 = x1.LT, x2.LT
                    if any(lt1[0]) or any(lt2[0]):
                        raise TypeError('Cannot convert FracElement.')
                    return f(lt1[1]) / f(lt2[1])
                return _f
        elif dom.is_AlgebraicField:
            gen = dom.ext.n() # type: ignore
            gen_pow = [gen**i for i in range(len(dom.mod.to_list()) - 1)] # type: ignore
            def wrapper(f):
                def _f(x):
                    return f(sum(c * v for c, v in zip(x.rep[::-1], gen_pow)))
                return _f

        if np.issubdtype(dtype, np.integer):
            f = wrapper(lambda x: x.__int__())
        elif np.issubdtype(dtype, np.floating):
            if isinstance(dom.one, FLINT_TYPE) or\
                (dom.is_Composite and isinstance(dom.domain.one, FLINT_TYPE)): # type: ignore
                f = wrapper(lambda x: x.numerator.__int__() / x.denominator.__int__())
            else:
                f = wrapper(lambda x: x.__float__())
        elif np.issubdtype(dtype, np.complexfloating):
            f = wrapper(lambda x: x.__complex__())

    if f is not None:
        rows, cols = dM.shape
        items = list(dM.rep.to_dok().items()) # avoid .iter_items() for version compatibility
        n = len(items)

        row_indices = [0] * n
        col_indices = [0] * n
        data_list = [0] * n

        for k in range(n):
            (i, j), v = items[k]
            row_indices[k] = i
            col_indices[k] = j
            data_list[k] = f(v)
        return data_list, row_indices, col_indices
    return None

def rep_matrix_to_numpy(M: Union[MatrixLike, DomainMatrix], dtype: Any = np.float64) -> ndarray:
    """
    Cast a sympy RepMatrix to a numpy matrix efficiently.

    Parameters
    ----------
    M : MatrixBase or DomainMatrix
        The sympy matrix to be casted.
    dtype : numpy.dtype
        The dtype of the numpy matrix. Default is np.float64.
    """
    dtype = np.dtype(dtype)
    if isinstance(M, ndarray):
        return M.astype(dtype, copy=False)
    if isinstance(M, spmatrix):
        return M.toarray().astype(dtype, copy=False)

    result = _rep_matrix_to_data(M, dtype)
    if result is None:
        # fallback to default constructor
        return np.array(M).astype(dtype)

    data_list, row_indices, col_indices = result
    arr = np.zeros(M.shape, dtype=dtype)
    arr[row_indices, col_indices] = data_list
    return arr

def rep_matrix_to_scipy(M: Union[MatrixLike, DomainMatrix], dtype = np.float64) -> spmatrix:
    """
    Cast a sympy RepMatrix to a scipy sparse matrix efficiently.

    Parameters
    ----------
    M : MatrixBase or DomainMatrix
        The sympy matrix to be casted.
    dtype : numpy.dtype
        The dtype of the numpy matrix. Default is np.float64.
    """
    dtype = np.dtype(dtype)
    if isinstance(M, spmatrix):
        return M.astype(dtype, copy=False)
    if isinstance(M, ndarray):
        return csr_matrix(M.astype(dtype, copy=False))

    result = _rep_matrix_to_data(M, dtype)
    if result is None:
        # fallback to default constructor
        return csr_matrix(np.array(M).astype(dtype))

    data_list, row_indices, col_indices = result
    arr = csr_matrix((data_list, (row_indices, col_indices)), shape=M.shape, dtype=dtype)
    return arr


def permute_matrix_rows(
    matrix: MatrixT, permutation: List[int]
) -> MatrixT:
    """
    Fast operation of matrix[permutation].

    Parameters
    ----------
    matrix : Matrix or ndarray
        The matrix to be permuted.
    permutation : List[int]
        The permutation of rows such that new_mat[i] = matrix[permutation[i]].

    Examples
    ----------
    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> permute_matrix_rows(M, [2, 0, 0])
    Matrix([
    [7, 8, 9],
    [1, 2, 3],
    [1, 2, 3]])
    """
    shape = (len(permutation), matrix.shape[1])

    if isinstance(matrix, RepMatrix):
        rep = matrix._rep.rep
        if isinstance(rep, SDM):
            new_rep = {}
            for r in range(len(permutation)):
                v = rep.get(permutation[r], None)
                if v is not None:
                    new_rep[r] = v #.copy()
            return matrix._fromrep(DomainMatrix.from_rep(
                SDM(new_rep, shape, rep.domain)
            ))

        elif isinstance(rep, DDM):
            new_rep_rows: List[Any] = [None for _ in range(len(permutation))]
            for r in range(len(permutation)):
                new_rep_rows[r] = rep[permutation[r]]#[:]
            ddm_rep = DDM(new_rep_rows, shape, rep.domain)
            return matrix._fromrep(DomainMatrix.from_rep(ddm_rep))

        elif isinstance(rep, DFM):
            rep2 = rep.rep.tolist() # type: ignore
            domain = rep.domain # type: ignore
            new_rep_rows_dfm: List[Any] = [None for _ in range(len(permutation))]
            for r in range(len(permutation)):
                new_rep_rows_dfm[r] = rep2[permutation[r]]
            dfm_rep = DFM(new_rep_rows_dfm, shape, domain)
            return matrix._fromrep(DomainMatrix.from_rep(dfm_rep))

    if isinstance(matrix, MatrixBase):
        rows = [matrix[permutation[r], :] for r in range(len(permutation))]
        return matrix.vstack(*rows)

    if isinstance(matrix, spmatrix):
        return matrix.__class__(matrix.tocsr()[permutation])

    return matrix[permutation]
