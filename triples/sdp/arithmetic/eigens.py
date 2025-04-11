from typing import List, Tuple, Union, Optional

from numpy import zeros as np_zeros
from numpy import ndarray

from sympy import Matrix, MatrixBase, Expr, Rational, Symbol, re, eye, collect
from sympy.core.singleton import S as singleton

def congruence(M: Union[Matrix, ndarray], signfunc = None) -> Union[None, Tuple[Matrix, Matrix]]:
    """
    Write a symmetric matrix as the decomposition
    M = U.T @ S @ U where U is upper triangular and S is diagonal.

    Parameters
    ----------
    M : Matrix | ndarray
        Symmetric matrix to be decomposed.
    signfunc : Callable
        Function to determine the sign of a value. It takes a value as input
        and returns 1, -1, or 0. If None, it uses the default comparison.
        This is useful when the matrix is symbolic and there 
        includes complicated expressions, e.g. nested square roots. 

    Returns
    -------
    U : Matrix | ndarray
        Upper triangular matrix.
    S : Matrix | ndarray
        Diagonal vector (1D array).

    Return None if M is not positive semidefinite.

    See also
    ---------
    sympy.matrices.dense.DenseMatrix.LDLdecomposition
    """
    if signfunc is None:
        signfunc = lambda x: 1 if x > 0 else (-1 if x < 0 else 0)

    M = M.copy()
    n = M.shape[0]
    if isinstance(M[0,0], Expr):
        U, S = Matrix.zeros(n), Matrix.zeros(n, 1)
        One = singleton.One
    else:
        U, S = np_zeros((n,n)), np_zeros(n)
        One = 1
    for i in range(n-1):
        sgn = signfunc(M[i,i])
        if sgn > 0:
            S[i] = M[i,i]
            U[i,i+1:] = M[i,i+1:] / (S[i])
            U[i,i] = One
            M[i+1:,i+1:] -= U[i:i+1,i+1:].T @ (U[i:i+1,i+1:] * S[i])
        elif sgn < 0:
            return None
        elif sgn == 0 and any(signfunc(_) != 0 for _ in M[i+1:,i]):
            return None
    U[-1,-1] = One
    S[-1] = M[-1,-1]
    if signfunc(S[-1]) < 0:
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
                cong = congruence(M + (-min_eig + eps) * eye(M.shape[0]))
                if cong is not None:
                    return cong
                eps *= 10
        return congruence(M)
    return None