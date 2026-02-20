from time import perf_counter
from typing import Dict, List, Union, Optional, Tuple, Any, Callable

from numpy import ndarray
import numpy as np
from sympy import MutableDenseMatrix as Matrix
from sympy import MatrixBase, Symbol, Float, Expr, Dummy
from sympy.core.relational import Relational

from .arithmetic import ArithmeticTimeout, sqrtsize_of_mat, vec2mat, is_numerical_mat, rep_matrix_from_numpy, rep_matrix_to_numpy
from .backends import SDPResult, SDPTimeoutError, solve_numerical_primal_sdp
from .rationalize import RationalizeWithMask, RationalizeSimultaneously
from .transforms import TransformablePrimal
from .utils import exprs_to_arrays

class SDPPrimal(TransformablePrimal):
    """
    Class to solve primal SDP problems, which is in the form of

        sum_i trace(S_i*A_ij) = b_j, Si >> 0

    Primal form of SDP is not flexible to be used for symbolic purposes,
    but it gains better performance in numerical solvers because it avoids
    reformulating the problem to the dual form.

    ## Solving Primal SDP

    Here is a simple tutorial to use this class to solve SDPs. Consider the example from
    https://github.com/vsdp/SDPLIB/tree/master:

        min   tr(-F0 * Y)

        s.t.  tr(F1 * Y) = 10,
              tr(F2 * Y) = 20,
              Y >> 0.

    where:

        F0 = Matrix(4,4,[1,0,0,0,0,2,0,0,0,0,3,0,0,0,0,4])
        F1 = Matrix(4,4,[1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
        F2 = Matrix(4,4,[0,0,0,0,0,1,0,0,0,0,5,2,0,0,2,6])

    To solve this problem, we view tr(Fj * Y) = bj as a linear constraint A @ vec(Y) = b,
    where A is a matrix of shape 2x16 and b is a vector of length 2. We initialize the problem as:

        >>> from sympy import Matrix
        >>> F0 = Matrix(4,4,[1,0,0,0,0,2,0,0,0,0,3,0,0,0,0,4])
        >>> F1 = Matrix(4,4,[1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
        >>> F2 = Matrix(4,4,[0,0,0,0,0,1,0,0,0,0,5,2,0,0,2,6])
        >>> A = Matrix.vstack(F1.reshape(1, 16), F2.reshape(1, 16))
        >>> b = Matrix([10,20])
        >>> sdp = SDPPrimal((b, {'X': A}))
        >>> sdp
        <SDPPrimal dof=16 size={'X': 4}>

    We can take a look at the symbolic matrix by calling the `S_from_y` method:

        >>> sdp.S_from_y()
        {'X': Matrix([
        [X_{0,0}, X_{0,1}, X_{0,2}, X_{0,3}],
        [X_{0,1}, X_{1,1}, X_{1,2}, X_{1,3}],
        [X_{0,2}, X_{1,2}, X_{2,2}, X_{2,3}],
        [X_{0,3}, X_{1,3}, X_{2,3}, X_{3,3}]])}

    Then we can solve the problem by calling the `solve_obj` method
    by passing in the objective vector, and it is expected to return the solution vector.

        >>> sdp.solve_obj(-F0.reshape(16, 1)) # doctest: +SKIP
        Matrix([
        [ 6.07047722664935],
        [              0.0],
        [              0.0],
        [              0.0],
        [              0.0],
        [ 3.92952277335065],
        [              0.0],
        [              0.0],
        [              0.0],
        [              0.0],
        [ 2.29577068463861],
        [-2.29578098235181],
        [              0.0],
        [              0.0],
        [-2.29578098235181],
        [ 2.29579128881059]])

    After the solution is found, the solution can also be accessed by `sdp.y`, `sdp.S`
    and `sdp.decompositions`. As the solving process is numerical, the matrix could
    be slightly nonpositive semidefinite up to a small numerical error.

        >>> sdp.S # doctest: +SKIP
        {'X': Matrix([
        [6.07047722664935,              0.0,               0.0,               0.0],
        [             0.0, 3.92952277335065,               0.0,               0.0],
        [             0.0,              0.0,  2.29577068463861, -2.29578098235181],
        [             0.0,              0.0, -2.29578098235181,  2.29579128881059]])}
        >>> sdp.decompositions # doctest: +SKIP
        {'X': (Matrix([
        [0.0, 0.0, -0.707108367720007, -0.707105194649528],
        [0.0, 1.0,                0.0,                0.0],
        [0.0, 0.0, -0.707105194649528,  0.707108367720007],
        [1.0, 0.0,                0.0,                0.0]]), Matrix([
        [4.34967772910966e-9],
        [   3.92952277335065],
        [   4.59156196909952],
        [   6.07047722664935]]))}
        >>> sdp.y.T @ -F0.reshape(16,1) # doctest: +SKIP
        Matrix([[-29.9999999825088]])
    """
    is_dual = False
    is_primal = True
    def __init__(self,
        x0_and_space: Tuple[Matrix, Dict[str, Matrix]],
        gens: Optional[List[Symbol]] = None,
    ) -> None:
        super().__init__()

        if not isinstance(x0_and_space, tuple):
            raise TypeError('x0_and_space must be a tuple of (x0, space).')
        if isinstance(x0_and_space[1], list):
            x0_and_space = (x0_and_space[0], dict(enumerate(x0_and_space)))
        self._x0_and_space = x0_and_space

        # check every space has same number of rows as x0
        x0, spaces = x0_and_space
        free_symbols_in_domain = x0.free_symbols if hasattr(x0, 'free_symbols') else set()
        for key, space in spaces.items():
            if space.shape[0] != x0.shape[0]:
                raise ValueError(f'The number of rows of space[{key}] must be the same as x0,'
                                f' but got {space.shape[0]} and {x0.shape[0]}.')
            if hasattr(space, 'free_symbols'):
                free_symbols_in_domain.update(space.free_symbols)
        self._free_symbols_in_domain = list(free_symbols_in_domain)

        dof = sum(space.shape[1] for space in spaces.values())
        if gens is not None:
            if len(gens)!= dof:
                raise ValueError(f"Length of free_symbols and space should be the same, but got"
                                 f" {len(gens)} and {dof}.")
            self._gens = list(gens)
        else:
            gens = []
            keys = {key for key in spaces.keys() if isinstance(key, str)}
            cnt = 0
            for key, space in spaces.items():
                m = sqrtsize_of_mat(space.shape[1])
                if isinstance(key, str):
                    name = key + '_{'
                else:
                    for i in range(len(keys) + 1):
                        name = 'S_{%d'%cnt
                        if any(k.startswith(name) for k in keys):
                            cnt += 1
                        else:
                            break
                    name = 'S_{%d,'%cnt
                gens += [Symbol('%s%d,%d}'%(name, min(j,k), max(j,k))) for j in range(m) for k in range(m)]
                cnt += 1
            self._gens = gens

    @property
    def gens(self) -> List[Symbol]:
        return self._gens

    @property
    def free_symbols(self) -> List[Symbol]:
        return self.gens + self._free_symbols_in_domain

    def keys(self, filter_none: bool = False) -> List[str]:
        space = self._x0_and_space[1]
        keys = list(space.keys())
        if filter_none:
            _size = lambda key: space[key].shape[1] * space[key].shape[0]
            keys = [key for key in keys if _size(key) != 0]
        return keys

    @property
    def dof(self) -> int:
        """
        The degree of freedom of the SDP problem. A symmetric n*n matrix is
        assumed to have n^2 degrees of freedom.
        """
        return sum(space.shape[1] for space in self._x0_and_space[1].values())
        # return sum(n*(n+1)//2 for n in self.size.values())

    def get_size(self, key: str) -> int:
        return sqrtsize_of_mat(self._x0_and_space[1][key].shape[1])

    def S_from_y(self, y: Optional[Union[Matrix, ndarray]] = None) -> Dict[str, Matrix]:
        if y is None:
            y = Matrix(self.gens)
        else:
            m = sum(space.shape[1] for space in self._x0_and_space[1].values())
            if isinstance(y, MatrixBase):
                if y.shape != (m, 1):
                    raise ValueError(f"Vector y must be a matrix of shape ({m}, 1), but got {y.shape}.")
            elif isinstance(y, ndarray):
                if y.size != m:
                    raise ValueError(f"Vector y must be an array of shape ({m},) or ({m}, 1), but got {y.shape}.")
                y = rep_matrix_from_numpy(y)

        S = {}
        cnt = 0
        for key, m in self.size.items():
            S[key] = vec2mat(y[cnt: cnt+m**2,:] if len(y.shape)==2 else y[cnt: cnt+m**2])
            cnt += m**2
        return S

    @property
    def full_space(self) -> Matrix:
        """
        The full space of the SDP problem.
        """
        if self.dof == 0:
            return Matrix.zeros(self._x0_and_space[0].shape[0], 0)
        return Matrix.hstack(*[space for space in self._x0_and_space[1].values()])

    @classmethod
    def from_full_x0_and_space(
        cls,
        x0: Matrix,
        space: Matrix,
        splits: Union[Dict[str, int], List[int]]
    ) -> 'SDPPrimal':
        """
        Create a SDPPrimal object from the full x0 and space matrices.

        Parameters
        ----------
        x0 : Matrix
            The full x0 matrix.
        space : Matrix
            The full space matrix.
        splits : Dict[str, int] or List[int]
            The splits of the space matrix. If it is a dict, it should be a mapping from key to the number of columns.
        """
        keys = None
        if isinstance(splits, dict):
            keys = list(splits.keys())
            splits = list(splits.values())

        space_list = []
        start = 0
        for n in splits:
            space_ = space[:,start:start+n**2]
            space_list.append(space_)
            start += n**2
        if keys is not None:
            space_list = dict(zip(keys, space_list))
        return SDPPrimal((x0, space_list))

    def exprs_to_arrays(self, exprs, dtype=np.float64):
        gens = self.gens.copy()
        bias = 0
        for n in self.size.values():
            for i in range(n):
                for j in range(i+1,n):
                    if gens[bias+i*n+j] == gens[bias+j*n+i]:
                        gens[bias+j*n+i] = Dummy('_') # mask non-unique symbols
            bias += n**2
        arrs = exprs_to_arrays(exprs, gens, dtype=dtype)

        def _symmetrize(arr):
            if isinstance(arr, ndarray):
                bias = 0
                for n in self.size.values():
                    i, j = np.triu_indices(n)
                    col1, col2 = i*n+j, j*n+i
                    arr[:,bias+col1] = (arr[:,bias+col1] + arr[:,bias+col2])/2
                    arr[:,bias+col2] = arr[:,bias+col1]
                    bias += n**2
            elif isinstance(arr, Matrix):
                bias = 0
                for n in self.size.values():
                    i, j = np.triu_indices(n)
                    for i0, j0 in zip(i.tolist(), j.tolist()):
                        arr[:,bias+i0*n+j0] = (arr[:,bias+i0*n+j0] + arr[:,bias+j0*n+i0])/2
                        arr[:,bias+j0*n+i0] = arr[:,bias+i0*n+j0]
                    bias += n**2
            return arr

        # restore each array to be "symmetric"
        for i in range(len(arrs)):
            arrs[i] = (_symmetrize(arrs[i][0]),) + arrs[i][1:]
        return arrs


    def project(self, y: Union[Matrix, ndarray], allow_numerical_solver: bool = True) -> Matrix:
        """
        Project a vector y so that it satisfies the equality constraints.

        Mathematically, we find y' for `argmin_{y'} ||y - y'|| s.t. Ay' = b`. (The PSD constraint is ignored.)
        Note that it is equivalent to `A(y' - y) = b - Ay` and we solve the least square problem for `y' - y`.

        Parameters
        ----------
        y : Matrix or ndarray
            The vector to be projected.
        allow_numerical_solver : bool
            If True, use numerical solver (NumPy) for float numbers to accelerate the projection.
        """
        if isinstance(y, ndarray):
            y = rep_matrix_from_numpy(y)
        A = self.full_space
        r = self._x0_and_space[0] - A @ y
        if all(i == 0 for i in r):
            return y
        dy = None
        if A.rows >= A.cols:
            dy = A.LDLsolve(r)
        elif allow_numerical_solver and is_numerical_mat(r):
            # use numerical solver for float numbers
            A2 = rep_matrix_to_numpy(A)
            r2 = rep_matrix_to_numpy(r)
            dy2 = np.linalg.lstsq(A2, r2, rcond=None)[0]
            dy = rep_matrix_from_numpy(dy2)
        else:
            dy = A.pinv_solve(r, arbitrary_matrix=Matrix.zeros(A.cols, 1))
        return y + dy


    def _solve_numerical_sdp(self,
        objective: np.ndarray,
        constraints: List[Tuple[np.ndarray, np.ndarray, str]] = [],
        solver: Optional[str] = None,
        return_result: bool = False,
        kwargs: Dict[str, Any] = {}
    ) -> Optional[np.ndarray]:
        return solve_numerical_primal_sdp(
            self._x0_and_space, objective=objective, constraints=constraints,
            solver=solver, return_result=return_result, **kwargs
        )


    def solve_obj(self,
        objective: Union[Expr, Matrix, List],
        constraints: List[Union[Relational, Expr, Tuple[Matrix, Matrix, str]]] = [],
        solver: Optional[str] = None,
        solve_child: bool = True,
        propagate_to_parent: bool = True,
        verbose: bool = False,
        time_limit: Optional[float] = None,
        kwargs: Dict[Any, Any] = {}
    ) -> Optional[Matrix]:
        """
        Solve the SDP problem numerically with the given objective.

        Parameters
        ----------
        objective : Expr, Matrix, or list
            Objective to minimize. If it is a sympy expression, it must be
            affine with respect to the variables. If it is a matrix (a column vector) or a list,
            the objective is the inner product of the vector and the variable vector.

        constraints : List[Union[Relational, Expr, Tuple[Matrix, Matrix, str]]]
            Additional affine constraints over variables. Each element of the list
            must be one of the following:

            A sympy affine relational expression, e.g., `x > 0` or `Eq(x + y, 1)`.
              Note that equality constraints must use `sympy.Eq` class instead of `==` operator,
              because the latter `x + y == 1` will be evaluated to a boolean value.

            A sympy affine expression, e.g., `x + y - 1`, they are treated as equality constraints.

            A tuple of (lhs, rhs, operator), where lhs is a 2D matrix, rhs is a 1D vector, and
              operator is a string. It is considered as `lhs @ variables (operator) rhs`.
              The operator can be one of '>', '<' or '='.

        solver : Optional[str]
            Backend solver to the numerical SDP, e.g., 'mosek', 'clarabel', 'cvxopt'.
            Corresponding packages must be installed. If None, the solver will be
            automatically selected. For a full list of supported backends, see `sdp.backends.caller.py`.

        solve_child : bool
            If there is a transformation graph of the SDP, whether to solve the child node and
            then convert the solution back to the parent node. This reduces the degree of freedom.
            Defaults to True.

        propagate_to_parent : bool
            If there is a transformation graph of the SDP, whether to propagate the solution of
            the SDP to its parents. Defaults to True.

        verbose : bool
            Whether to allow the backend SDP solver to print the log. Defaults to False.
            This argument will be suppressed if `kwargs` contains a `verbose` key.

        time_limit : Optional[float]
            Time limit in seconds for the solver. If None, no time limit is set. Defaults to None.
            When time limit is reached, the solver will try to terminate the process and raise
            an Exception. Only a few solvers support time limit, e.g., 'mosek', 'clarabel' and 'qics',
            and other solvers will not check timeout during the solving process.

        kwargs : Dict
            Extra kwargs passed to `sdp.backends.solve_numerical_dual_sdp`. Accepted kwargs keys:
            `verbose`, `max_iters`, `tol_gap_abs`, `tol_gap_rel`, `tol_fsb_abs`, `tol_fsb_rel`, `solver_options`,
            etc.
        """
        return super().solve_obj(
            objective, constraints=constraints, solver=solver,
            solve_child=solve_child, propagate_to_parent=propagate_to_parent,
            verbose=verbose, time_limit=time_limit, kwargs=kwargs
        )

    def solve(self,
        solver: Optional[str] = None,
        solve_child: bool = True,
        propagate_to_parent: bool = True,
        verbose: bool = False,
        allow_numer: int = 0,
        time_limit: Optional[float] = None,
        kwargs: Dict[Any, Any] = {}
    ) -> Optional[Matrix]:
        """
        Solve a feasible SDP problem. If the SDPProblem is rational,
        it tries to find a rational solution. However, the search for
        rational solutions is heuristic and could fail for weakly feasible SDPs.

        Parameters
        ----------
        solver : Optional[str]
            Backend solver to the numerical SDP, e.g.,'mosek', 'clarabel', 'cvxopt'.
            Corresponding packages must be installed. If None, the solver will be
            automatically selected. For a full list of supported backends, see `sdp.backends.caller.py`.

        solve_child : bool
            If there is a transformation graph of the SDP, whether to solve the child node and
            then convert the solution back to the parent node. This reduces the degree of freedom.
            Defaults to True.

        propagate_to_parent : bool
            If there is a transformation graph of the SDP, whether to propagate the solution of
            the SDP to its parents. Defaults to True.

        allow_numer : bool
            Whether to allow inexact, numerical feasible solutions. This is useful when the
            SDP is weakly feasible and no rational solution is found successfully.

        time_limit : Optional[float]
            Time limit in seconds for the solver. If None, no time limit is set. Defaults to None.
            When time limit is reached, the solver will try to terminate the process and raise
            an Exception. Only a few solvers support time limit, e.g., 'mosek', 'clarabel' and 'qics',
            and other solvers will not check timeout during the solving process.

        kwargs : Dict
            Extra kwargs passed to `sdp.backends.solve_numerical_dual_sdp`. Accepted kwargs keys:
            `verbose`, `max_iters`, `tol_gap_abs`, `tol_gap_rel`, `tol_fsb_abs`, `tol_fsb_rel`, `solver_options`,
            etc.

        Returns
        ----------
        y : Matrix
            The solution of the SDP problem. If it fails, return None.
        """
        original_self = self
        end_time = perf_counter() + time_limit if isinstance(time_limit, (int, float)) else None
        _time_limit = ArithmeticTimeout.make_checker(time_limit)
        if solve_child:
            self = self.get_last_child()

        x0, spaces = self._x0_and_space
        spaces = list(spaces.values())

        # add a relaxation variable on the diagonal to maximize the eigenvalue
        # sum(tr(AiSi)) = a => sum(tr(Ai(Xi + x*I))) = a where Si = Xi + xI
        try:
            spaces = [rep_matrix_to_numpy(_) for _ in spaces]
            diag = np.zeros((x0.shape[0], ), dtype=np.float64)
            for space in spaces:
                n = sqrtsize_of_mat(space.shape[1])

                # get the contribution of diagonals, i.e. traces
                diag += space[:,np.arange(0,n**2,n+1)].sum(axis = 1)
            spaces.append(diag)
            objective = np.array([0]*self.dof + [-1], dtype=np.float64)
            constraints = [(objective, 5, '<')] # avoid unboundness
            _time_limit()
        except ArithmeticTimeout as e:
            raise SDPTimeoutError.from_kwargs() from e


        kwargs = kwargs.copy()
        if (not ('verbose' in kwargs)) and float(verbose) > 1:
            kwargs['verbose'] = verbose
        if end_time is not None and (not ('time_limit' in kwargs)):
            kwargs['time_limit'] = end_time - perf_counter()

        sol = solve_numerical_primal_sdp(
            (x0, spaces), objective=objective, constraints=constraints,
            solver=solver, return_result=True, **kwargs
        )
        assert isinstance(sol, SDPResult)

        success = False
        try:
            if sol.y is not None and (not sol.infeasible):
                y, eig = sol.y[:-1], sol.y[-1] # discard the eigenvalue relaxation

                # restore matrices by adding the eigenvalue relaxation
                bias = 0
                for n in self.size.values():
                    y[bias: bias+n**2][np.arange(0,n**2,n+1)] += eig
                    bias += n**2

                self._ys.append(y)
                _time_limit()
                solution = self.rationalize(y, verbose=verbose,
                    rationalizers=[RationalizeWithMask(), RationalizeSimultaneously([1,1260,1260**3])])
                if solution is not None:
                    self.y = solution[0]
                    self.S = dict((key, S[0]) for key, S in solution[1].items())
                    self.decompositions = dict((key, S[1:]) for key, S in solution[1].items())
                    success = True
                elif allow_numer:
                    self.register_y(y, perturb=True, propagate_to_parent=propagate_to_parent)
                    success = True
        except ArithmeticTimeout as e:
            raise SDPTimeoutError(sol) from e

        if propagate_to_parent:
            self.propagate_to_parent(recursive=True)

        return original_self.y if success else None
