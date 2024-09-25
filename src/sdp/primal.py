from typing import Dict, List, Union, Optional, Tuple, Any

from numpy import ndarray
import numpy as np
from sympy import MutableDenseMatrix as Matrix
from sympy import MatrixBase, Symbol

from .abstract import SDPProblemBase, Decomp, Objective, Constraint, MinEigen
from .rationalize import rationalize_and_decompose
from .utils import Mat2Vec

class SDPPrimal(SDPProblemBase):
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
        # for i, (_, space) in enumerate(self._x0_and_space.values()):
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
        assumed to have n*(n+1)//2 degrees of freedom.
        """
        # return sum(space.shape[1] for _, space in self._x0_and_space.values())
        return sum(n*(n+1)//2 for n in self.size.values())

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

    def project(self, y: Union[Matrix, ndarray]) -> Matrix:
        """
        Project a vector y so that it satisfies the equality constraints.

        Mathematically, we find y' for argmin_{y'} ||y - y'|| s.t. Ay' = b. (The PSD constraint is ignored.)
        Note that it is equivalent to A(y' - y) = b - Ay, we solve the least square problem for y' - y.
        """
        if isinstance(y, ndarray):
            y = Matrix(y.flatten())
        A = self.full_space
        r = self._x0 - A * y
        if all(i == 0 for i in r):
            return y
        return y + A.LDLsolve(r)
    
    def register_y(self, y: Union[Matrix, ndarray], project: bool = False, perturb: bool = False, propagate_to_parent: bool = True) -> None:
        y2 = self.project(y)
        if (not (y2 is y)) and not project:
            # FIXME for numpy array
            raise ValueError("The vector y is not feasible by the equality constraints. Use project=True to project to the feasible region.")
        return super().register_y(y2, perturb=perturb, propagate_to_parent=propagate_to_parent)
 
    def _get_defaulted_configs(self) -> List[List[Any]]:
        return [], [[]], [0]

    def _solve_from_multiple_configs(self,
            list_of_objective: List[Objective] = [],
            list_of_constraints: List[List[Constraint]] = [],
            list_of_min_eigen: List[MinEigen] = [],
            solver: Optional[str] = None,
            allow_numer: int = 0,
            rationalize_configs = {},
            verbose: bool = False,
            solver_options: Dict[str, Any] = {}
        ) -> Optional[Tuple[Matrix, Decomp]]:

        num_sol = len(self._ys)

        for obj, con, eig in zip(list_of_objective, list_of_constraints, list_of_min_eigen):
            if len(con):
                raise NotImplementedError("The primal form does not support constraints.")
            if eig != 0:
                raise NotImplementedError("The primal form does not support nonzero eigenvalue lower bounds.")
            
            # temporary usage
            from cvxpy import Problem, Minimize, trace, Variable
            constraints = []
            s_list = []
            eq = -np.array(self._x0).astype(np.float64).flatten()
            obj_expr = 0.0
            for key, m in self.size.items():
                space = np.array(self._space[key]).astype(np.float64)
                S = Variable((m, m), PSD=True)
                s_list.append(S)
                eq = eq + space @ S.reshape((m**2,))
                obj_expr = obj_expr + obj[key].flatten() @ S.reshape((m**2,))
            constraints.append(eq == 0.)
            obj = Minimize(obj_expr)
            prob = Problem(obj, constraints)
            sol = prob.solve(verbose=verbose, **solver_options)
            y = np.concatenate([np.array(s.value).flatten() for s in s_list])

            ra = rationalize_and_decompose(y, self.S_from_y, **rationalize_configs)
            if ra is not None:
                return ra
