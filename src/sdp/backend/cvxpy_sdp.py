from math import sqrt
from typing import Any

import numpy as np

from .backend import DualBackend, PrimalBackend


class DualBackendCVXPY(DualBackend):
    """
    CVXPY backend for SDP problems.
    CVXPY is a Python-embedded modeling language for convex optimization problems.

    Warning: It seems that CVXPY cannot recognize dual SDP problems properly,
    which would lead to very slow performance.

    Installation:
    pip install cvxpy

    Reference:
    [1] https://www.cvxpy.org/api_reference/cvxpy.html
    """
    _dependencies = ('cvxpy',)
    def __init__(self, dof) -> None:
        super().__init__(dof)
        from cvxpy import Variable
        self.y = Variable(dof + 1)
        self._mats = []
        self._constraints = []
        self._objective = None
        self.problem = None

    def _add_linear_matrix_inequality(self, x0: np.ndarray, extended_space: np.ndarray) -> np.ndarray:
        from cvxpy import Variable, reshape
        # TIP: cvxpy instances do not support .reshape method in earlier versions (e.g. <= 1.3)
        # thus we need to call cvxpy.reshape instead of (...).reshape
        k = int(round(sqrt(x0.shape[0])))
        S = Variable((k, k), PSD=True)
        self._constraints.append(S == reshape(x0 + extended_space @ self.y, (k, k)))
        return S

    def _add_constraint(self, constraint: np.ndarray, rhs: float = 0, operator = '__ge__') -> None:
        self._constraints.append(getattr(constraint @ self.y, operator)(rhs))

    def _set_objective(self, objective: np.ndarray) -> None:
        from cvxpy import Minimize
        self._objective = Minimize(objective @ self.y)

    def solve(self, solver_options = {}) -> np.ndarray:
        from cvxpy import Problem
        problem = Problem(self._objective, self._constraints)
        self.problem = problem
        self.problem.solve(**solver_options)
        value = self.y.value
        if isinstance(value, float):
            value = [value]
        return np.array(value).flatten()[:-1]


class PrimalBackendCVXPY(PrimalBackend):
    _dependencies = ('cvxpy',)
    def __init__(self, x0: np.ndarray) -> None:
        super().__init__(x0)
        self.solution = None

    def _add_constraint(self, constraint: np.ndarray, rhs: float = 0, operator = '__ge__') -> None:
        # self._constraints.append(getattr(constraint @ self.x, operator)(rhs))
        raise NotImplementedError

    def _create_problem(self) -> Any:
        from cvxpy import Problem, Minimize, Variable, reshape
        # TIP: cvxpy instances do not support .reshape method in earlier versions (e.g. <= 1.3)
        # thus we need to call cvxpy.reshape instead of (...).reshape
        variables = []
        mat_size = self._mat_size + [1]
        for m in mat_size:
            variables.append(Variable((m, m), PSD=True))

        sumobj = 0
        for v, obj in zip(variables, self.split_vector(self._objective)):
            sumobj = sumobj + reshape(obj, (obj.size,)) @ reshape(v, (v.size,))

        eq_constraint = -self.x0
        for v, space in zip(variables, self._spaces + [self._min_eigen_space]):
            eq_constraint = eq_constraint + space @ reshape(v, (v.size,))

        constraints = [eq_constraint == 0]
        problem = Problem(Minimize(sumobj), constraints)
        return problem, variables

    def solve(self, solver_options = {}) -> np.ndarray:
        problem, variables = self._create_problem()
        self.solution = problem.solve(**solver_options)
        value = [v.value.flatten() for v in variables]
        return np.concatenate(value)[:-1]
        