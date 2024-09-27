from typing import Dict, List, Union, Optional, Tuple, Any, Callable

from numpy import ndarray
import numpy as np
from sympy import MutableDenseMatrix as Matrix
from sympy import MatrixBase, Symbol, Float

from .abstract import Decomp, Objective, Constraint, MinEigen
from .backend import solve_numerical_primal_sdp
from .ipm import SDPRationalizeError
from .rationalize import rationalize_and_decompose
from .transform import PrimalTransformMixin
from .utils import Mat2Vec

class SDPPrimal(PrimalTransformMixin):
    """
    Symbolic SDP feasible problem in the primal form.
    This is reprsented by Sum_i trace(S_i*A_j) = b_j, X_i >> 0.

    Primal form of SDP is not flexible to be used for symbolic purposes,
    but it gains better performance in numerical solvers because it avoids
    reformulating the problem to the dual form.

    """
    def __init__(self,
        space: Dict[str, Matrix],
        x0: Matrix
    ) -> None:
        super().__init__()

        self._space: Dict[str, Matrix] = None
        self._x0 = x0
        self._init_space(space, '_space')

        # check every space has same number of rows as x0
        for key, space in self._space.items():
            if space.shape[0] != x0.shape[0]:
                raise ValueError(f'The number of rows of space["{key}"] must be the same as x0, but got {space.shape[0]} and {x0.shape[0]}.')

        # self.free_symbols = list()
        # for i, space in enumerate(self._space.values()):
        #     m = Mat2Vec.length_of_mat(space.shape[1])
        #     self.free_symbols += [Symbol('x_{%d,%d,%d}'%(i, j, k)) for j in range(m) for k in range(j, m)]

    def keys(self, filter_none: bool = False) -> List[str]:
        space = self._space
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
        return sum(space.shape[1] for space in self._space.values())
        # return sum(n*(n+1)//2 for n in self.size.values())

    def get_size(self, key: str) -> int:
        return Mat2Vec.length_of_mat(self._space[key].shape[1])

    @property
    def _transforms(self) -> None:
        raise NotImplementedError("The transforms are not implemented for the primal form.")

    def S_from_y(self, y: Optional[Union[Matrix, ndarray]] = None) -> Dict[str, Matrix]:
        if y is None:
            y = []
            for i, m in enumerate(self.size.values()):
                y += [Symbol('x_{%d,%d,%d}'%(i, min(j,k), max(j,k))) for j in range(m) for k in range(j, m)]
            y = Matrix(y)
        else:
            m = sum(space.shape[1] for space in self._space.values())
            if isinstance(y, MatrixBase):
                if y.shape != (m, 1):
                    raise ValueError(f"Vector y must be a matrix of shape ({m}, 1).")
            elif isinstance(y, ndarray):
                if y.size != m:
                    raise ValueError(f"Vector y must be an array of shape ({m},) or ({m}, 1).")
                y = Matrix(y.flatten())

        S = {}
        cnt = 0
        for key, m in self.size.items():
            S[key] = Mat2Vec.vec2mat(y[cnt: cnt+m**2,:])
            cnt += m**2
        return S

    @property
    def full_space(self) -> Matrix:
        """
        The full space of the SDP problem.
        """
        if self.dof == 0:
            return Matrix.zeros(self._x0.shape[0], 0)
        return Matrix.hstack(*[space for space in self._space.values()])

    def project(self, y: Union[Matrix, ndarray], allow_numerical_solver: bool = True) -> Matrix:
        """
        Project a vector y so that it satisfies the equality constraints.

        Mathematically, we find y' for argmin_{y'} ||y - y'|| s.t. Ay' = b. (The PSD constraint is ignored.)
        Note that it is equivalent to A(y' - y) = b - Ay, we solve the least square problem for y' - y.

        Parameters
        ----------
        y : Matrix or ndarray
            The vector to be projected.
        allow_numerical_solver : bool
            If True, use numerical solver (NumPy) for float numbers to accelerate the projection.
        """
        if isinstance(y, ndarray):
            y = Matrix(y.flatten())
        A = self.full_space
        r = self._x0 - A * y
        if all(i == 0 for i in r):
            return y
        dy = None
        if A.rows >= A.cols:
            dy = A.LDLsolve(r)
        elif allow_numerical_solver and any(isinstance(_, Float) for _ in r):
            # use numerical solver for float numbers
            A2 = np.array(A).astype(np.float64)
            r2 = np.array(r).astype(np.float64)
            dy2 = np.linalg.lstsq(A2, r2, rcond=None)[0]
            dy = Matrix(dy2)
        else:
            dy = A.pinv_solve(r, arbitrary_matrix=Matrix.zeros(A.cols, 1))
        return y + dy
    
    def register_y(self, y: Union[Matrix, ndarray], project: bool = False, perturb: bool = False, propagate_to_parent: bool = True) -> None:
        y2 = self.project(y)
        if (not (y2 is y)) and not project:
            # FIXME for numpy array
            raise ValueError("The vector y is not feasible by the equality constraints. Use project=True to project to the feasible region.")
        return super().register_y(y2, perturb=perturb, propagate_to_parent=propagate_to_parent)

    def _get_defaulted_configs(self) -> List[List[Any]]:
        sum_trace = np.concatenate([np.eye(m).flatten() for m in self.size.values()])
        objectives_and_min_eigens = [
            (sum_trace, 0),
            (np.array([0] * sum_trace.size + [1]), (1, 0))
        ]
        objectives = [_[0] for _ in objectives_and_min_eigens]
        min_eigens = [_[1] for _ in objectives_and_min_eigens]
        constraints = [[] for _ in objectives_and_min_eigens]
        return objectives, constraints, min_eigens

    def _solve_from_multiple_configs(self,
            list_of_objective: List[Objective] = [],
            list_of_constraints: List[List[Constraint]] = [],
            list_of_min_eigen: List[MinEigen] = [],
            solver: Optional[str] = None,
            allow_numer: int = 0,
            rationalize_configs = {},
            verbose: bool = False,
            solver_options: Dict[str, Any] = {},
            raise_exception: bool = False
        ) -> Optional[Tuple[Matrix, Decomp]]:

        num_sol = len(self._ys)

        for obj, con, eig in zip(list_of_objective, list_of_constraints, list_of_min_eigen):
            if len(con):
                raise NotImplementedError("The primal form does not support constraints.")

            y = solve_numerical_primal_sdp(
                self._space, self._x0,
                obj, constraints=con, min_eigen=eig,
                solver=solver, solver_options=solver_options, raise_exception=raise_exception
            )
            if y is not None:
                self._ys.append(y)

                def _force_return(self: SDPPrimal, y):
                    self.register_y(y, project=True, perturb = True, propagate_to_parent = False)
                    _decomp = dict((key, (s, d)) for (key, s), d in zip(self.S.items(), self.decompositions.values()))
                    return y, _decomp

                if allow_numer >= 3:
                    # force to return the numerical solution
                    return _force_return(self, y)

                decomp = rationalize_and_decompose(y, self.S_from_y, projection=self.project, **rationalize_configs)
                if decomp is not None:
                    return decomp

                if allow_numer == 2:
                    # force to return the numerical solution if rationalization fails
                    return _force_return(self, y)

        if len(self._ys) > num_sol:
            # new numerical solution found
            decomp = self.rationalize_combine(self._ys, self.S_from_y, projection=self.project, verbose = verbose)
            if decomp is not None:
                return decomp

            if allow_numer == 1:
                y = self._ys[-1]
                return rationalize_and_decompose(y, self.S_from_y, projection=self.project,
                            try_rationalize_with_mask = False, times = 0, perturb = True, check_pretty = False)
            else:
                raise SDPRationalizeError(
                    "Failed to find a rational solution despite having a numerical solution."
                )

        return None

            # temporary usage
            # from cvxpy import Problem, Minimize, trace, Variable
            # constraints = []
            # s_list = []
            # eq = -np.array(self._x0).astype(np.float64).flatten()
            # obj_expr = 0.0
            # for key, m in self.size.items():
            #     space = np.array(self._space[key]).astype(np.float64)
            #     S = Variable((m, m), PSD=True)
            #     s_list.append(S)
            #     eq = eq + space @ S.reshape((m**2,))
            #     obj_expr = obj_expr + obj[key].flatten() @ S.reshape((m**2,))
            # constraints.append(eq == 0.)
            # obj = Minimize(obj_expr)
            # prob = Problem(obj, constraints)
            # sol = prob.solve(verbose=verbose, **solver_options)
            # y = np.concatenate([np.array(s.value).flatten() for s in s_list])

            # ra = rationalize_and_decompose(y, self.S_from_y, **rationalize_configs)
            # if ra is not None:
            #     return ra
