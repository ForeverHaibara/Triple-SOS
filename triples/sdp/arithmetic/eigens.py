from typing import List, Tuple, Union, Optional

import numpy as np
from numpy import ndarray
from numpy.linalg import eigvalsh, cholesky, eigh, LinAlgError

from sympy import Matrix, MatrixBase, Expr, RR, re, eye, Float
from sympy.core.singleton import S as singleton
from sympy.matrices.repmatrix import RepMatrix

from .matop import rep_matrix_from_list

def congruence(M: Union[Matrix, ndarray], perturb: Union[bool, float]=False,
    upper: bool=True, signfunc = None) -> Optional[Tuple[Matrix, Matrix]]:
    """
    Compute the decomposition of a positive semidefinite matrix M:
    `M = U.T @ diag(S) @ U` where `S` is stored as a (column) vector.
    The input M must be symmetric and real, which users should ensure by themselves.
    If M is not positive semidefinite, return None.

    Parameters
    ----------
    M : Matrix | ndarray
        Symmetric matrix to be decomposed.
    perturb : bool | float
        If `perturb` is given and M is a numerical matrix, then add a small
        perturbation to the diagonal of M to make it positive semidefinite
        to avoid numerical issues. If `perturb` is a float, then it is used as
        the upper bound of the perturbation. If M is an exact matrix (e.g.
        integer or rational matrix), then the perturbation is ignored.
    upper : bool
        If True, force M to be upper triangular. If False, M is not forced to be
        upper triangular. Set to False to improve numerical stability if M is
        a numerical matrix.
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

    Examples
    --------
    >>> from sympy import Matrix, Rational
    >>> hilbert = Matrix([[Rational(1,i+j+1) for j in range(3)] for i in range(3)])
    >>> hilbert
    Matrix([
    [  1, 1/2, 1/3],
    [1/2, 1/3, 1/4],
    [1/3, 1/4, 1/5]])
    >>> U, S = congruence(hilbert); U, S
    (Matrix([
    [1, 1/2, 1/3],
    [0,   1,   1],
    [0,   0,   1]]), Matrix([
    [    1],
    [ 1/12],
    [1/180]]))
    >>> U.T @ Matrix.diag(*S) @ U == hilbert
    True

    The function also supports positive semidefinite matrices.
    >>> M = Matrix([[1, -2], [-2, 4]])
    >>> congruence(M)
    (Matrix([
    [1, -2],
    [0,  0]]), Matrix([
    [1],
    [0]]))

    For numerical matrices, set the `perturb` parameter to a small float value
    to avoid numerical issues.
    >>> M = Matrix([[1, 1.73205081], [1.73205081, 3]])
    >>> congruence(M) is None
    True
    >>> congruence(M, perturb=1e-8) # doctest: +SKIP
    (Matrix([
    [1.00000000210541,    1.73205080635332],
    [             0.0, 9.17695800206094e-5]]), Matrix([
    [1.0],
    [1.0]]))

    However, the `perturb` parameter is ignored if the matrix
    is an exact SymPy matrix, e.g. an integer or a rational matrix.
    >>> M2 = Matrix([[1, Rational(173205081,10**8)], [Rational(173205081,10**8), 3]])
    >>> congruence(M2, perturb=1e-8) is None
    True

    Set `upper=False` to allow the matrix U not to be upper triangular.
    This may be more numerically stable for numerical matrices. The
    relation M = U.T @ diag(S) @ U still holds up to a small numerical error.
    >>> U, S = congruence(M, perturb=1e-8, upper=False); U, S # doctest: +SKIP
    (Matrix([
    [-0.866025403632494, 0.500000000263177],
    [ 0.500000000263177, 0.866025403632494]]), Matrix([
    [             0.0],
    [4.00000000210541]]))

    If the input is a numpy array, the function returns numpy arrays.
    And the array is treated as a floating value matrix even if its dtype is int.
    >>> import numpy as np
    >>> congruence(np.array([[1, 2], [2, 5]]))
    (array([[1., 2.],
           [0., 1.]]), array([1., 1.]))


    See also
    ---------
    sympy.matrices.dense.DenseMatrix.LDLdecomposition
    """
    if isinstance(M, ndarray):
        return _congruence_numerically(M, perturb=perturb, upper=upper)
    if isinstance(M, RepMatrix):
        domain = M._rep.domain
        if domain.is_Exact and domain.is_Numerical:
            return _congruence_on_exact_domain(M)
        if not domain.is_Exact:
            return _congruence_numerically(M, perturb=perturb, upper=upper)
    if isinstance(M, MatrixBase):
        if M.has(Float):
            return _congruence_numerically(M, perturb=perturb, upper=upper)

    if signfunc is None:
        signfunc = lambda x: 1 if x > 0 else (-1 if x < 0 else 0)

    M = M.copy()
    n = M.shape[0]
    if isinstance(M[0,0], Expr):
        U, S = Matrix.zeros(n), Matrix.zeros(n, 1)
        One = singleton.One
    else:
        U, S = np.zeros((n,n)), np.zeros(n)
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


def _congruence_on_exact_domain(M: RepMatrix) -> Optional[Tuple[RepMatrix, RepMatrix]]:
    """
    Perform congruence decomposition on M fast if M is a SymPy RepMatrix.
    This functions exploits low-level arithmetic on domain elements, e.g. rational numbers
    or algebraic numbers, which accelerates the computation.

    The function should only be called when M is a SymPy RepMatrix and is on a
    exact, numerical domain. Because SymPy uses EXRAW as domains for matrices of irrational numbers
    by default, they should be converted to AlgebraicFields to accelerate if possible.
    """
    dM = M._rep.to_field()
    n, domain, M = dM.shape[0], dM.domain, dM.rep.to_list()
    one, zero = domain.one, domain.zero

    if domain.is_QQ or domain.is_ZZ:
        signfunc = lambda x: 0 if x == zero else (1 if x > 0 else -1)
    else:
        signfunc = lambda x: 0 if x == zero else (1 if re(domain.to_sympy(x)) > 0 else -1)

    S = [zero] * n
    U = [[zero for _ in range(n)] for __ in range(n)]
    for i in range(n):
        Mi, Ui = M[i], U[i]
        p = Mi[i]
        sgn = signfunc(p)
        if sgn > 0:
            S[i] = p
            Ui[i] = one

            # # The vectorized version:
            # U[i,i+1:] = M[i,i+1:] / p
            # M[i+1:,i+1:] -= U[i:i+1,i+1:].T @ (U[i:i+1,i+1:] * p)

            invp = one / p
            for j in range(i+1, n):
                Ui[j] = Mi[j] * invp
            for k in range(i+1, n):
                Mk = M[k]
                for j in range(k, n):
                    Mk[j] = Mk[j] - Ui[j] * Mi[k] # rank 1 update
        elif sgn < 0:
            return None
        elif sgn == 0:
            # # The vectorized version:
            # if any(signfunc(_) != 0 for _ in M[i+1:,i]):
            #     return None
            if any(Mi[j] != zero for j in range(i+1, n)):
                return None

    U = rep_matrix_from_list(U, (n, n), domain)
    S = rep_matrix_from_list(S, n, domain)
    return U, S

def _congruence_numerically(M: Union[MatrixBase, ndarray], perturb: Union[bool, float]=0,
        upper: bool=True) -> Optional[Tuple[Union[Matrix, ndarray], Union[Matrix, ndarray]]]:
    """
    Perform congruence decomposition on M if M is numerical. It converts the matrix
    to numpy and calls cholesky or ldl decomposition. If upper=False, it calls eigh.
    """
    # from scipy.linalg import ldl
    is_sympy = isinstance(M, MatrixBase)
    n = M.shape[0]
    if n == 0:
        if is_sympy:
            return M.reshape(0, 0), M.reshape(0, 1)
        return M.reshape(0, 0), M.reshape(0)
    if is_sympy:
        M = np.array(M.n(15))
    if isinstance(M, ndarray):
        try:
            M = M.astype(float)
        except TypeError:
            M = np.real(M.astype(complex))

    if upper:
        def silent_chol(M):
            U = None
            try:
                U = cholesky(M)
            except LinAlgError:
                pass
            return U

        U = silent_chol(M)

        if U is None and (not (perturb is False)):
            if perturb is True:
                perturb = 1
            mineig = float(np.min(eigvalsh(M)))
            if mineig < -perturb:
                return None

            npeye = np.eye(n)
            eps = min(perturb, abs(mineig)*2 if mineig < -1e-14 else 1e-14)
            U = silent_chol(M + npeye*eps)
            for i in range(13, 0, -1):
                eps = 10**-i
                if U is not None or eps > perturb:
                    break
                U = silent_chol(M + npeye*eps)
        S = np.ones((n,), dtype=float)
    else:
        if perturb is True or perturb is False:
            perturb = int(perturb)
        S, U = eigh(M)
        if np.min(S) < -perturb:
            return None
        S = np.where(S < 0, 0., S)

    if U is None:
        return None
    U = U.T
    if is_sympy:
        conv = RR.convert
        U = rep_matrix_from_list([[conv(_) for _ in row] for row in U.tolist()], (n, n), RR)
        S = rep_matrix_from_list([conv(_) for _ in S.flatten().tolist()], n, RR)
    return U, S