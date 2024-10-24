from math import sqrt
from typing import Any

import numpy as np

from .backend import DualBackend, PrimalBackend, random_name


class DualBackendPICOS(DualBackend):
    """
    PICOS backend for SDP problems.
    Picos is a Python interface to conic optimization solvers,
    and the default solver is CVXOPT.

    Installation:
    pip install picos

    Reference:
    [1] https://picos-api.gitlab.io/picos/index.html
    """
    _dependencies = ('picos',)
    def __init__(self, dof) -> None:
        super().__init__(dof)
        from picos import Problem, RealVariable
        self.problem = Problem()
        self.y = RealVariable(random_name(), shape = dof + 1)

    def _add_linear_matrix_inequality(self, x0: np.ndarray, extended_space: np.ndarray) -> np.ndarray:
        from picos import SymmetricVariable
        S = SymmetricVariable(random_name(), (sqrt(x0.shape[0]), sqrt(x0.shape[0])))
        self.problem.add_constraint(S.vec == x0 + extended_space * self.y)
        self.problem.add_constraint(S >> 0)
        return S

    def _add_constraint(self, constraint: np.ndarray, rhs: float = 0, operator = '__ge__') -> None:
        self.problem.add_constraint(getattr(constraint * self.y, operator)(rhs))

    def _set_objective(self, objective: np.ndarray) -> None:
        self.problem.set_objective('min', objective * self.y)

    def solve(self, solver_options = {}) -> np.ndarray:
        self.problem.solve(**solver_options)
        value = self.y.value
        if isinstance(value, float):
            value = [value]
        return np.array(value).flatten()[:-1]


class PrimalBackendPICOS(PrimalBackend):
    _dependencies = ('picos',)
    def __init__(self, x0) -> None:
        super().__init__(x0)

        self._eqs = []
        self._eq_bs = []
        self._leqs = []
        self._leq_bs = []

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
        from picos import Problem, SymmetricVariable, RealVariable, sum
        problem = Problem()
        variables = []
        for m in self._mat_size:
            S = SymmetricVariable(random_name(), (m, m))
            variables.append(S)
            problem.add_constraint(S >> 0)
        variables.append(RealVariable(random_name(), 1))

        eq_constraint = sum([space * v.vec for v, space in zip(variables, self._spaces + [self._min_eigen_space])])
        problem.add_constraint(eq_constraint == self.x0)

        dof = self.dof + 1
        if len(self._eqs):
            eqs = np.vstack([eq.reshape((-1, dof)) for eq in self._eqs])
            eqs = self.split_vector(eqs)
            eq_b = np.concatenate([np.array(_).flatten() for _ in self._eq_bs])
            problem.add_constraint(sum([eq * v.vec for v, eq in zip(variables, eqs)]) == eq_b)

        if len(self._leqs):
            leqs = np.vstack([leq.reshape((-1, dof)) for leq in self._leqs])
            leqs = self.split_vector(leqs)
            leq_b = np.concatenate([np.array(_).flatten() for _ in self._leq_bs])
            problem.add_constraint(sum([leq * v.vec for v, leq in zip(variables, leqs)]) <= leq_b)

        sumobj = sum([obj.flatten() * v.vec for v, obj in zip(variables, self.split_vector(self._objective))])
        problem.set_objective('min', sumobj)

        return problem, variables

    def solve(self, solver_options = {}) -> np.ndarray:
        problem, variables = self._create_problem()
        self.solution = problem.solve(**solver_options)
        value = [np.array(v.value).flatten() for v in variables]
        return self.restore_eigen(value)