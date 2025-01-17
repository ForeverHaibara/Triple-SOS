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

        self._eqs = []
        self._eq_bs = []
        self._leqs = []
        self._leq_bs = []
        self.solution = None

    def _add_constraint(self, constraint: np.ndarray, rhs: float = 0, operator = '__eq__') -> None:
        if operator == '__eq__':
            self._eqs.append(constraint)
            self._eq_bs.append(rhs)
        elif operator == '__le__':
            self._leqs.append(constraint)
            self._leq_bs.append(rhs)
        elif operator == '__ge__':
            self._leqs.append(-constraint)
            self._leq_bs.append(-rhs)

    def _create_problem(self) -> Any:
        from cvxpy import Problem, Minimize, Variable, reshape, sum
        # TIP: cvxpy instances do not support .reshape method in earlier versions (e.g. <= 1.3)
        # thus we need to call cvxpy.reshape instead of (...).reshape
        variables = []
        for m in self._mat_size:
            variables.append(Variable((m, m), PSD=True))
        variables.append(Variable((1, 1), symmetric=True))

        sumobj = sum([obj.flatten() @ reshape(v, (v.size,)) for v, obj in zip(variables, self.split_vector(self._objective))])

        eq_constraint = sum([space @ reshape(v, (v.size,)) for v, space in zip(variables, self._spaces + [self._min_eigen_space])])
        constraints = [eq_constraint == self.x0]

        dof = self.dof + 1
        if len(self._eqs):
            eqs = np.vstack([eq.reshape((-1, dof)) for eq in self._eqs])
            eqs = self.split_vector(eqs)
            eq_b = np.concatenate([np.array(_).flatten() for _ in self._eq_bs])
            constraints.append(sum([eq @ reshape(v, (v.size,)) for v, eq in zip(variables, eqs)]) == eq_b)

        if len(self._leqs):
            leqs = np.vstack([leq.reshape((-1, dof)) for leq in self._leqs])
            leqs = self.split_vector(leqs)
            leq_b = np.concatenate([np.array(_).flatten() for _ in self._leq_bs])
            constraints.append(sum([leq @ reshape(v, (v.size,)) for v, leq in zip(variables, leqs)]) <= leq_b)

        problem = Problem(Minimize(sumobj), constraints)
        return problem, variables

    def solve(self, solver_options = {}) -> np.ndarray:
        problem, variables = self._create_problem()
        self.solution = problem.solve(**solver_options)
        value = [v.value.flatten() for v in variables]
        return self.restore_eigen(value)
        