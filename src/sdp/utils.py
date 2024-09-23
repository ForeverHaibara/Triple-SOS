from contextlib import contextmanager
from math import sqrt
from typing import Union, Optional, Tuple, List, Dict, Callable, Generator

from numpy import zeros as np_zeros
from numpy import ndarray
from sympy import zeros as sp_zeros
from sympy import Matrix, MatrixBase, Expr, re
from sympy.core.singleton import S as singleton

def congruence(M: Union[Matrix, ndarray]) -> Union[None, Tuple[Matrix, Matrix]]:
    """
    Write a symmetric matrix as a sum of squares.
    M = U.T @ S @ U where U is upper triangular and S is diagonal.

    Returns
    -------
    U : Matrix | ndarray
        Upper triangular matrix.
    S : Matrix | ndarray
        Diagonal vector (1D array).

    Return None if M is not positive semidefinite.
    """
    M = M.copy()
    n = M.shape[0]
    if isinstance(M[0,0], Expr):
        U, S = Matrix.zeros(n), Matrix.zeros(n, 1)
        One = singleton.One
    else:
        U, S = np_zeros((n,n)), np_zeros(n)
        One = 1
    for i in range(n-1):
        if M[i,i] > 0:
            S[i] = M[i,i]
            U[i,i+1:] = M[i,i+1:] / (S[i])
            U[i,i] = One
            M[i+1:,i+1:] -= U[i:i+1,i+1:].T @ (U[i:i+1,i+1:] * S[i])
        elif M[i,i] < 0:
            return None
        elif M[i,i] == 0 and any(_ for _ in M[i+1:,i]):
            return None
    U[-1,-1] = One
    S[-1] = M[-1,-1]
    if S[-1] < 0:
        return None
    return U, S


def congruence_with_perturbation(
        M: Matrix,
        perturb: bool = False
    ) -> Optional[Tuple[Matrix, Matrix]]:
    """
    Perform congruence decomposition on M. This is a wrapper of congruence.
    Write it as M = U.T @ S @ U where U is upper triangular and S is a diagonal matrix.

    Parameters
    ----------
    M : Matrix
        Symmetric matrix to be decomposed.
    perturb : bool
        If perturb == True, make a slight perturbation to force M to be positive semidefinite.
        This is useful when there exists numerical errors in the matrix.

    Returns
    ---------
    U : Matrix
        Upper triangular matrix.
    S : Matrix
        Diagonal vector (1D array).
    """
    if M.shape[0] == 0 or M.shape[1] == 0:
        return Matrix(), Matrix()
    if not perturb:
        return congruence(M)
    else:
        min_eig = min([re(v) for v in M.n(20).eigenvals()])
        if min_eig < 0:
            eps = 1e-15
            for i in range(10):
                cong = congruence(M + (-min_eig + eps) * sp.eye(M.shape[0]))
                if cong is not None:
                    return cong
                eps *= 10
        return congruence(M)
    return None


def is_empty_matrix(M: Matrix, check_all_zeros: bool = False) -> bool:
    """
    Check whether a matrix is zero size. Set check_all_zeros == True to
    check whether all entries are zero.
    """
    if M.shape[0] == 0 or M.shape[1] == 0:
        return True
    if check_all_zeros and not any(M):
        return True
    return False



class Mat2Vec:
    """
    Conversion between symmetric matrices and vector forms.
    It supports multiple vector representations of symmetric matrices.
    """
    DIRECT = 0
    UPPER = 1
    ISOMETRIC = 2
    DEFAULT = DIRECT

    @classmethod
    def length_of_vec(cls, n: int, mode: int = DEFAULT) -> int:
        """
        Infer the length of output vector given the n*n symmetric matrix.
        """
        if mode == cls.DIRECT:
            return n ** 2
        return n * (n + 1) // 2

    @classmethod
    def length_of_mat(cls, m: int, mode: int = DEFAULT) -> int:
        """
        Infer the length of output matrix given the length of the vector.
        """
        if mode == cls.DIRECT:
            return round(sqrt(m))
        return round(sqrt(2 * m + .25) - .5)


    @classmethod
    def mat2vec(cls, S: Union[Matrix, ndarray], mode: int = DEFAULT) -> Matrix:
        """
        Convert a symmetric matrix to a vector.

        Parameters
        ----------
        S : Matrix | ndarray
            Symmetric matrix.
        mode : int
            Mode of conversion. 0: direct, 1: upper part, 2: isometric.

        Returns
        -------
        Matrix
            Vector representation of S.
        """
        if mode == cls.DIRECT:
            if isinstance(S, MatrixBase):
                return S.reshape(S.shape[0] * S.shape[1], 1)
            elif isinstance(S, ndarray):
                return S.flatten()
        elif mode == cls.UPPER:
            return cls.upper_vec_of_symmetric_matrix(S)
        elif mode == cls.ISOMETRIC:
            raise NotImplementedError("Isometric mode is not implemented yet.")

    @classmethod
    def vec2mat(cls, v: Union[Matrix, ndarray], mode: int = DEFAULT) -> Matrix:
        """
        Convert a vector to a symmetric matrix.

        Parameters
        ----------
        v : Matrix | ndarray
            Vector representation of a symmetric matrix.
        mode : int
            Mode of conversion. 0: direct, 1: upper part, 2: isometric.

        Returns
        -------
        Matrix
            Symmetric matrix.
        """
        if mode == cls.DIRECT:
            n = round((v.shape[0] * v.shape[1]) ** .5)
            return v.reshape(n, n)
        elif mode == cls.UPPER:
            return cls.symmetric_matrix_from_upper_vec(v)
        elif mode == cls.ISOMETRIC:
            raise NotImplementedError("Isometric mode is not implemented yet.")

    @classmethod
    def upper_vec_of_symmetric_matrix(
        cls,
        S: Union[Matrix, ndarray, int], 
        return_inds: bool = False,
        mul: Optional[float] = None,
        check: Callable = None,
    ) -> Generator[List[Union[Expr, Tuple[int, int]]], None, None]:
        """
        Gather the upper part of a symmetric matrix by rows as a vector.

        Parameters
        ----------
        S : ndarray | Matrix | int
            Symmetric matrix or the shape[0] of the matrix.
        return_inds : bool
            If True, return (i,j) instead of S[i,j].
        mul: float
            If not None, multiply off-diagonal elements by this number.
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
        elif mul is not None:
            ret = lambda i, j: mul * S[i,j] if i != j else S[i,j]
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

    @classmethod
    def symmetric_matrix_from_upper_vec(
        cls,
        upper_vec: Union[Matrix, ndarray]
    ) -> Matrix:
        """
        Construct a symmetric matrix from its upper part.
        When upper_vec is a sympy matrix, it is assumed to be in the shape of a vector.
        When upper_vec is a numpy matrix, it can be 1D or 2D. If it is 1D as a vector,
        the function has the same behavior as the sympy version. If it is 2D, it returns
        a 3D tensor with the last dimension being the last dimension of upper_vec.

        Parameters
        ----------
        upper_vec : Matrix | ndarray
            Upper part of a symmetric matrix.

        Returns
        -------
        S : Matrix | ndarray
            Symmetric matrix.
        """
        n = cls.length_of_mat(upper_vec.shape[0], mode = cls.UPPER)
        if isinstance(upper_vec, MatrixBase):
            S = sp_zeros(n)
        elif isinstance(upper_vec, ndarray):
            S = np_zeros((n,n,upper_vec.shape[1])) if upper_vec.ndim == 2 else np_zeros((n,n))

        for (i,j), v in zip(cls.upper_vec_of_symmetric_matrix(n, return_inds = True), upper_vec):
            S[i,j] = v
            S[j,i] = v
        return S

    @classmethod
    def split_vector(cls,
            chunks: List[Union[int, List, Matrix]], mode: int = DEFAULT
        ) -> List[slice]:
        """
        It is common for a single array to store information of multiple
        matrices. And we need to extract each matrix from the array by 
        chunks of indices.

        The method is now deprecated.

        This function takes in a list of ints / lists / matrices, and returns a list of
        slices with lengths equal to shape[1] of each matrix.

        Parameters
        ----------
        chunks : List[int | List | Matrix]
            List of ints / lists / matrices.
        mode : int
            Mode of conversion. 0: direct, 1: upper part, 2: isometric.
        
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
            elif isinstance(c, MatrixBase):
                return c.shape[1]
            elif isinstance(c, int):
                return c
            elif isinstance(c, list):
                return len(c)

        for chunk in chunks:
            l = length(chunk)
            if l > 0:
                l = cls.length_of_vec(l, mode)
                splits.append(slice(split_start, split_start + l))
                split_start += l
        return splits


def S_from_y(
        y: Matrix,
        x0_and_space: Dict[str, Tuple[Matrix, Matrix]],
        mode: int = Mat2Vec.DEFAULT,
    ) -> Dict[str, Matrix]:
    """
    Return the symmetric matrices S from the vector y.

    Parameters
    ----------
    y : Matrix
        The vector to be checked.
    x0_and_space : Dict[str, Tuple[Matrix, Matrix]]
        vec(S[key]) = x0[key] + space[key] @ y.
    splits : List[slice]
        The splits of the symmetric matrices. Each split is a slice object.
    mode : int
        Mode of conversion. 0: direct, 1: upper part, 2: isometric.

    Returns
    ----------
    S_dict : Matrix
        Each S[key] satisfies that vec(S[key]) = x0[key] + space[key] @ y.
    """
    if not isinstance(y, MatrixBase):
        y = Matrix(y)

    S_dict = {}
    for key, (x0, space) in x0_and_space.items():
        vecS = x0 + space * y
        S = Mat2Vec.vec2mat(vecS, mode=mode)
        S_dict[key] = S
    return S_dict


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