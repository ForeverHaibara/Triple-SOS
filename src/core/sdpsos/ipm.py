from typing import List, Optional, Tuple, Callable, Dict, Union, Any

import numpy as np
import sympy as sp
import mpmath as mp


class _functional:
    """
    Functional class that provides the basic operations for numpy, sympy and mpmath matrices.    
    """
    def __new__(cls, *args, **kwargs):
        """
        Parameters
        ----------
        args : List[Any]
            The arguments. It can be None, 'sp', 'mp', or 'np' or a matrix.
            The dtype will be inferred from the first non-None argument.
            If the dtype cannot be inferred, it defaults to numpy.
        """
        for x in args:
            if x is None:
                continue
            elif isinstance(x, sp.Matrix):
                return super().__new__(_functional_sp)
            elif isinstance(x, mp.matrix):
                return super().__new__(_functional_mp)
            elif isinstance(x, str):
                x = x.lower()
                if x == 'sp' or x == 'sympy':
                    return super().__new__(_functional_sp)
                elif x == 'mp' or x == 'mpmath':
                    return super().__new__(_functional_mp)
                elif x == 'np' or x == 'numpy':
                    return super().__new__(_functional_np)
        return super().__new__(_functional_np)

    @classmethod
    def inner(cls, x: np.ndarray, y: np.ndarray) -> np.float64:
        return cls.trace(x @ y)

    @classmethod
    def linsolve(cls, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.linalg.solve(A, b)

    @classmethod
    def as_array(cls, x: List) -> np.ndarray:
        return np.array(x, dtype = np.float64)

    @classmethod
    def vector_span(cls, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x @ y.T

    @classmethod
    def matrix_list_add(cls, x: List[np.ndarray]) -> np.ndarray:
        s = x[0]
        if len(x) > 1:
            for xi in x[1:]:
                s += xi
        return s        

    @classmethod
    def shape(cls, x: np.ndarray) -> Tuple[int, int]:
        return x.shape

    @classmethod
    def trace(cls, x: np.ndarray) -> np.float64:
        return x.trace()

    @classmethod
    def isnan(cls, x: np.float64) -> bool:
        return np.isnan(x)

    @classmethod
    def zeros(cls, rows: int, cols: int) -> np.ndarray:
        return np.zeros((rows, cols), dtype = np.float64)

    @classmethod
    def eye(cls, n: int) -> np.ndarray:
        return np.eye(n, dtype = np.float64)

class _functional_np(_functional):
    @classmethod
    def inner(cls, x: np.ndarray, y: np.ndarray) -> np.float64:
        if x.ndim == 1: # we assume y.ndim == 1 also
            return np.inner(x, y)
        return np.einsum("ij,ji->", x, y)

    @classmethod
    def as_array(cls, x: List) -> np.ndarray:
        return np.array(x, dtype = np.float64)

    @classmethod
    def vector_span(cls, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x.reshape(-1, 1) @ y.reshape(1, -1)

class _functional_sp(_functional):
    linsolve = staticmethod(sp.Matrix.LUsolve)
    as_array = staticmethod(sp.Matrix)
    zeros = staticmethod(sp.zeros)
    eye = staticmethod(sp.eye)

    @classmethod
    def inner(cls, x: sp.Matrix, y: sp.Matrix) -> sp.Float:
        return sum((x.T).multiply_elementwise(y))

    @classmethod
    def isnan(cls, x: sp.Float) -> bool:
        return x is sp.nan

class _functional_mp(_functional):
    linsolve = staticmethod(mp.lu_solve)
    as_array = staticmethod(mp.matrix)
    isnan = staticmethod(mp.isnan)
    zeros = staticmethod(mp.zeros)
    eye = staticmethod(mp.eye)

    @classmethod
    def inner(cls, x: mp.matrix, y: mp.matrix) -> mp.mpf:
        return mp.fdot(x.T, y)

    @classmethod
    def shape(cls, x: mp.matrix) -> Tuple[int, int]:
        return (x.rows, x.cols)

    @classmethod
    def trace(cls, x: mp.matrix) -> mp.mpf:
        s = mp.mpf(0)
        for i in range(x.rows):
            s += x[i,i]
        return s


class SDPError(Exception): ...

class SDPConvergenceError(SDPError, RuntimeError): ...

class SDPNumericalError(SDPError, RuntimeError): ...

class SDPInfeasibleError(SDPError, RuntimeError): ...

class SDPRationalizeError(SDPError, RuntimeError): ...


def _block_iterator(blocks: List[int]) -> Tuple[int, int]:
    """
    A generator that yields the start and end indices of the blocks.
    """
    i = 0
    for b in blocks:
        yield i, i + b
        i += b


def sdp_ipm(
        A: List[Union[np.ndarray, sp.Matrix, mp.matrix]],
        b: Union[np.ndarray, sp.Matrix, mp.matrix],
        C: Union[np.ndarray, sp.Matrix, mp.matrix],
        X0: Union[np.ndarray, sp.Matrix, mp.matrix],
        dtype: Optional[str] = None,
        rho: float = .5,
        epsilon: float = 1e-8,
        max_iter: int = 100,
        blocks: Optional[List[int]] = None,
        callback: Optional[Callable[[Dict[str, Any]], Any]] = None
    ) -> Tuple[Union[np.ndarray, sp.Matrix, mp.matrix], Union[np.ndarray, sp.Matrix, mp.matrix]]:
    """
    Solve the semidefinite program
    
            min  <C, X>
            s.t. <A[i], X> = b[i], i = 1, ..., m
                X >= 0

    using the interior point method. The initial point X0 must be feasible.
    The method supports numpy, sympy and mpmath matrices, which indicates arbitrary precision.
    All three libraries support @ for matrix multiplication.

    The algorithm is based on [1]. However, the Cholesky decomposition can be omitted by
    using the trick that tr(L'AL L'BL) = tr(A (LL') B (LL')), where LL' is the Cholesky
    decomposition of Xk.

    Parameters
    ----------
    A : List[matrix]
        The constraint matrices A[i].
    b : matrix
        The constraint vector b.
    C : matrix
        The objective matrix C.
    X0 : matrix
        The initial point. It must be feasible.
    dtype : Optional[str]
        The type of the matrices. It can be 'np' for numpy, 'sp' for sympy, or 'mp' for mpmath.
        If None, the type will be inferred from the type of the matrices in X0, A[0].
    rho : float
        The parameter rho for step size. It must lie in (0, 1). The default is .5.
        Larger step size might lead to numerical instability, or stops too early.
        Small step size might lead to slow convergence.
    epsilon : float
        The tolerance for convergence. When the difference of the objective function
        between two iterations is less than epsilon, the algorithm stops. The default is 1e-8.
    max_iter : int
        The maximum number of iterations. The default is 100. SDPConvergenceError will be
        raised if the algorithm does not converge within max_iter iterations.
    blocks : List[int]
        The block sizes of the matrices. For example, if all A and X0 are block diagonal matrices,
        then computation can be sped up by specifying blocks = [n1, n2, ...], where n1, n2, ...
        are the sizes of the blocks. It should satisfy sum(blocks) = n, where n is the size of X0.
        The default is None, which means all matrices are treated as dense matrices.
    callback : Optional[Callable[[Dict[str, Any], Any]]]
        A callback function that will be called after each iteration. It takes a dictionary
        as argument, containing k, X, diff, z, ..., where k is the iteration number,
        X is the current point, diff is the difference of the objective function between
        two iterations, z is the current objective function value. If the callback function
        returns True, the algorithm will stop. The default is None.

    Returns
    ----------
    Xk : matrix
        The optimal solution.
    lam : matrix
        The dual solution.

    References
    ----------
    [1] D. Benterki, J.-P. Crouzeix, and B. Merikhi, "A numerical feasible interior point method
      for linear semidefinite programs" RAIRO - Operations Research, vol. 41, no. 1, pp. 49-59, 2007.
    """
    F = _functional(dtype, X0, A[0])
    A = [F.as_array(Ai) for Ai in A]
    b, C, Xk = F.as_array(b), F.as_array(C), F.as_array(X0)
    bbT = F.vector_span(b, b)
    n = F.shape(Xk)[0]

    if blocks is not None:
        if not sum(blocks) == n:
            raise ValueError('The sum of block sizes must be equal to the size of X0.')
    else:
        blocks = [n]


    for k in range(max_iter):

        CX_blocks = [None] * len(blocks)
        AX_blocks = [None] * len(blocks)
        VX_blocks = [None] * len(blocks)
        M_blocks = [None] * len(blocks)
        cxax_blocks = [None] * len(blocks)

        z = 0
        cxcx = 0
        for i, (start, end) in enumerate(_block_iterator(blocks)):
            block = (slice(start, end), slice(start, end))
            C_block = C[block]
            Xk_block = Xk[block]
            CX_block = C_block @ Xk_block
            AX_block = [Ai[block] @ Xk_block for Ai in A]


            M_blocks[i] = F.as_array([[F.inner(AXi, AXj) for AXj in AX_block] for AXi in AX_block])
            cxax_blocks[i] = F.as_array([F.inner(CX_block, AXi) for AXi in AX_block])

            z += F.inner(C_block, Xk_block)
            cxcx += F.inner(CX_block, CX_block)

            AX_blocks[i] = AX_block
            CX_blocks[i] = CX_block


        M = F.matrix_list_add(M_blocks)
        cxax = F.matrix_list_add(cxax_blocks)

        # # Original 1-block version:
        # z = F.inner(C, Xk)
        # CX = C @ Xk

        # AX = [Ai @ Xk for Ai in A]
        # M = F.as_array([[F.inner(AXi, AXj) for AXj in AX] for AXi in AX])
        # cxax = F.as_array([F.inner(CX, AXi) for AXi in AX])
        # cxcx = F.inner(CX, CX)

        d = b * (-z) - cxax
        lam = F.linsolve(M + bbT, d)
        v = - F.inner(b.T, lam) - z


        Vk2 = 0
        eig_mean = 0
        for i, (start, end) in enumerate(_block_iterator(blocks)):
            block = (slice(start, end), slice(start, end))
            VX_block = CX_blocks[i] # .copy()
            for j in range(len(A)):
                VX_block += lam[j] * AX_blocks[i][j]

            Vk2 += F.inner(VX_block, VX_block)
            eig_mean += F.trace(VX_block)
            VX_blocks[i] = VX_block


        eig_mean = eig_mean / n
        eig_std = (Vk2 / n - eig_mean**2)**.5
        beta = rho / max(v, eig_mean + eig_std*(n - 1)**.5)

        # # If we use the cholesky decomposition (1-block version):
        # L = np.linalg.cholesky(Xk)
        # V = L.T @ C @ L
        # for i in range(len(A)):
        #     V += lam[i] * L.T @ A[i] @ L
        # tau = (v**2 + F.inner(V, V))**.5
        # eig_mean = F.trace(V) / n
        # eig_std = (F.inner(V, V) / n - eig_mean**2)**.5

        r = 1. / (1. - beta * v)

        for i, (start, end) in enumerate(_block_iterator(blocks)):
            block = (slice(start, end), slice(start, end))
            Xk[block] *= r
            Xk[block] -= beta * (Xk[block] @ VX_blocks[i])


        # diff is the difference of the objective function between two iterations
        # i.e. z_{k+1} = z_k - diff
        diff = beta * (cxcx + F.inner(cxax.T, lam) - z * v) * r

        if diff < epsilon:
            # print('Converged in', k, 'iterations. diff =', diff)
            break
        if F.isnan(v):
            raise SDPNumericalError('NaN encountered. This is probably due to numerical instability '
                                    'or a infeasible system.')

        if callback is not None:
            params = dict(k = k, X = Xk, diff = diff, z = z - diff, M = M, lam = lam,
                          eig_mean = eig_mean, eig_std = eig_std, beta = beta, v = v, r = r)
            if callback(params) is True:
                break

    else:
        raise SDPConvergenceError(f'Maximum number of {max_iter} iterations reached. '
                                    f'Final diff = {diff}')

    return Xk, lam


def sdp_ipm_feasible(
        A: List[Union[np.ndarray, sp.Matrix, mp.matrix]],
        b: Union[np.ndarray, sp.Matrix, mp.matrix],
        X0: Optional[Union[np.ndarray, sp.Matrix, mp.matrix]] = None,
        dtype: Optional[str] = None,
        rho: float = .5,
        epsilon: float = 1e-8,
        max_iter: int = 100,
        blocks: Optional[List[int]] = None,
        callback: Optional[Callable[[Dict[str, Any]], Any]] = None
    ) -> Union[np.ndarray, sp.Matrix, mp.matrix]:
    """
    Find a feasible point for the semidefinite program
            s.t. <A[i], X> = b[i], i = 1, ..., m
                X >= 0
                
    using the interior point method. The method supports numpy, sympy and mpmath matrices,
    which indicates arbitrary precision. All three libraries support @ for matrix multiplication.

    The algorithm is based on [1]. We first introduce a relaxation variable and then solve
    a semidefinite program with known feasible point.

    Parameters
    ----------
    A : List[matrix]
        The constraint matrices A[i].
    b : matrix
        The constraint vector b.
    X0 : Optional[matrix]
        Any positive definite n*n matrix. When None, defaults to an identity matrix.
    dtype : Optional[str]
        The type of the matrices. It can be 'np' for numpy, 'sp' for sympy, or 'mp' for mpmath.
        If None, the type will be inferred from the type of the matrices in X0, A[0].
    rho : float
        The parameter rho for step size. It must lie in (0, 1). The default is .5.
        Larger step size might lead to numerical instability, or stops too early.
        Small step size might lead to slow convergence.
    epsilon : float
        The tolerance for convergence. When the difference of the objective function
        between two iterations is less than epsilon, the algorithm stops. The default is 1e-8.
    max_iter : int
        The maximum number of iterations. The default is 100. SDPConvergenceError will be
        raised if the algorithm does not converge within max_iter iterations.
    blocks : List[int]
        The block sizes of the matrices. For example, if all A and X0 are block diagonal matrices,
        then computation can be sped up by specifying blocks = [n1, n2, ...], where n1, n2, ...
        are the sizes of the blocks. It should satisfy sum(blocks) = n, where n is the size of X0.
        The default is None, which means all matrices are treated as dense matrices.
    callback : Optional[Callable[[Dict[str, Any], Any]]]
        A callback function that will be called after each iteration. It takes a dictionary
        as argument, containing k, X, diff, z, where k is the iteration number, X is the current
        point, diff is the difference of the objective function between two iterations,
        z is the current objective function value. If the callback function returns True,
        the algorithm will stop. The default is None.

    Returns
    ----------
    Xk : matrix
        The feasible point.

    References
    ----------
    [1] D. Benterki, J.-P. Crouzeix, and B. Merikhi, "A numerical feasible interior point method
      for linear semidefinite programs" RAIRO - Operations Research, vol. 41, no. 1, pp. 49-59, 2007.
    """
    if len(A) == 0:
        raise ValueError('Argument A must be non-empty.')

    F = _functional(dtype, X0, A[0])
    A = [F.as_array(Ai) for Ai in A]
    b = F.as_array(b)
    n = F.shape(A[0])[0]

    C = F.zeros(n + 1, n + 1)
    C[n, n] = 1

    def _pad(X, v = 0):
        X2 = F.zeros(n + 1, n + 1)
        X2[:n, :n] = X
        X2[n, n] = v
        return X2

    if X0 is None:
        X0 = F.eye(n + 1)
        A = [_pad(Ai, bi - F.trace(Ai)) for Ai, bi in zip(A, b)]
    else:
        X0 = F.as_array(X0)
        A = [_pad(Ai, bi - F.inner(Ai, X0)) for Ai, bi in zip(A, b)]
        X0 = _pad(X0, 1)


    if blocks is None:
        blocks = [n, 1]
    else:
        blocks = blocks + [1]


    Xk, lam = sdp_ipm(A, b, C, X0, dtype=dtype, 
                rho=rho, epsilon=epsilon, max_iter=max_iter, blocks=blocks, callback=callback)
    return Xk[:n, :n]


def sdp_dual_feasible(
        C: Union[np.ndarray, sp.Matrix, mp.matrix],
        A: List[Union[np.ndarray, sp.Matrix, mp.matrix]],
        dtype: Optional[str] = None,
        rho = .5,
        epsilon = 1e-8,
        max_iter = 100,
        blocks = None,
        callback = None
    ) -> Tuple[Union[np.ndarray, sp.Matrix, mp.matrix], Union[np.ndarray, sp.Matrix, mp.matrix]]:
    """
    Solve the dual problem of the SDP feasibility problem
    C + A[0]*x[0] + ... + A[n]*x[n] >= 0.

    Parameters
    ----------
    C : matrix
        The objective matrix C.
    A : List[matrix]
        The constraint matrices A[i].
    dtype : Optional[str]
        The type of the matrices. It can be 'np' for numpy, 'sp' for sympy, or 'mp' for mpmath.
        If None, the type will be inferred from the type of the matrices in X0, A[0].
    rho : float
        The parameter rho for step size. It must lie in (0, 1). The default is .5.
        Larger step size might lead to numerical instability, or stops too early.
        Small step size might lead to slow convergence.
    epsilon : float
        The tolerance for convergence. When the difference of the objective function
        between two iterations is less than epsilon, the algorithm stops. The default is 1e-8.
    max_iter : int
        The maximum number of iterations. The default is 100. SDPConvergenceError will be
        raised if the algorithm does not converge within max_iter iterations.
    blocks : List[int]
        The block sizes of the matrices. For example, if all A and X0 are block diagonal matrices,
        then computation can be sped up by specifying blocks = [n1, n2, ...], where n1, n2, ...
        are the sizes of the blocks. It should satisfy sum(blocks) = n, where n is the size of X0.
        The default is None, which means all matrices are treated as dense matrices.
    callback : Optional[Callable[[Dict[str, Any], Any]]]
        A callback function that will be called after each iteration. It takes a dictionary
        as argument, containing k, X, diff, z, where k is the iteration number, X is the current
        point, diff is the difference of the objective function between two iterations,
        z is the current objective function value. If the callback function returns True,
        the algorithm will stop. The default is None.

    Returns
    ----------
    Xk : matrix
        The optimal solution.
    lam : matrix
        The dual solution.
    """
    if len(A) == 0:
        raise ValueError('Argument A must be non-empty.')

    F = _functional(dtype, C, A[0])
    A = [F.as_array(Ai) for Ai in A]
    C = F.as_array(C)

    b = F.as_array([F.trace(Ai) for Ai in A])
    X, lam = sdp_ipm(A, b, C, F.eye(F.shape(C)[0]), dtype=dtype,
                rho=rho, epsilon=epsilon, max_iter=max_iter, blocks=blocks, callback=callback)
    return X, lam