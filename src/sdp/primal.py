from typing import Dict, List, Union, Optional, Tuple, Any, Callable

from numpy import ndarray
import numpy as np
from sympy import MutableDenseMatrix as Matrix
from sympy import MatrixBase, Symbol, Float

from .abstract import Decomp, Objective, Constraint, MinEigen
from .backend import solve_numerical_primal_sdp
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
    is_dual = False
    is_primal = True
    def __init__(self,
        x0: Matrix,
        space: Dict[str, Matrix],
    ) -> None:
        super().__init__()

        self._x0 = x0
        self._space: Dict[str, Matrix] = None
        self._init_space(space, '_space')

        # check every space has same number of rows as x0
        for key, space in self._space.items():
            if space.shape[0] != x0.shape[0]:
                raise ValueError(f'The number of rows of space["{key}"] must be the same as x0, but got {space.shape[0]} and {x0.shape[0]}.')

        # self.free_symbols = list()
        # for i, space in enumerate(self._space.values()):
        #     m = Mat2Vec.length_of_mat(space.shape[1])
        #     self.free_symbols += [Symbol('x_{%d,%d,%d}'%(i, j, k)) for j in range(m) for k in range(j, m)]

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
        return SDPPrimal(x0, space_list)


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
                    raise ValueError(f"Vector y must be a matrix of shape ({m}, 1), but got {y.shape}.")
            elif isinstance(y, ndarray):
                if y.size != m:
                    raise ValueError(f"Vector y must be an array of shape ({m},) or ({m}, 1), but got {y.shape}.")
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

    def _get_defaulted_configs(self) -> List[List[Any]]:
        sum_trace = np.concatenate([np.eye(m).flatten() for m in self.size.values()])
        objectives_and_min_eigens = [
            (sum_trace, 0),
            (np.array([0] * sum_trace.size + [-1]), (1, 0))
        ]
        objectives = [_[0] for _ in objectives_and_min_eigens]
        min_eigens = [_[1] for _ in objectives_and_min_eigens]
        constraints = [[] for _ in objectives_and_min_eigens]
        return objectives, constraints, min_eigens

    def _solve_numerical_sdp(self,
            objective: Objective,
            constraints: List[Constraint] = [],
            min_eigen: MinEigen = 0,
            scaling: float = 6.,
            solver: Optional[str] = None,
            verbose: bool = False,
            solver_options: Dict[str, Any] = {},
            raise_exception: bool = False
        ):

        # if len(constraints):
        #     raise NotImplementedError("The primal form does not support constraints.")
        return solve_numerical_primal_sdp(
                self._space, self._x0, objective=objective, constraints=constraints, min_eigen=min_eigen, scaling=scaling,
                solver=solver, solver_options=solver_options, raise_exception=raise_exception
            )
