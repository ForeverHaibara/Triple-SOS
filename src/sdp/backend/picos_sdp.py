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
    def __init__(self, size) -> None:
        super().__init__(size)

    def _add_constraint(self, constraint: np.ndarray, rhs: float = 0, operator = '__eq__') -> None:
        raise NotImplementedError

    def _create_problem(self) -> Any:
        from picos import Problem, SymmetricVariable
        problem = Problem()
        mat_size = self._mat_size + [1]
        variables = []
        for m in mat_size:
            S = SymmetricVariable(random_name(), (m, m))
            variables.append(S)
            problem.add_constraint(S >> 0)

        sumobj = 0
        for v, obj in zip(variables, self.split_vector(self._objective)):
            sumobj = sumobj + obj.reshape((obj.size,)) * v.vec

        eq_constraint = -self.x0
        for v, space in zip(variables, self._spaces + [self._min_eigen_space]):
            eq_constraint = eq_constraint + space * v.vec

        problem.add_constraint(eq_constraint == 0)
        problem.set_objective('min', sumobj)
        return problem, variables

    def solve(self, solver_options = {}) -> np.ndarray:
        problem, variables = self._create_problem()
        self.solution = problem.solve(**solver_options)
        value = [np.array(v.value).flatten() for v in variables]
        return np.concatenate(value)[:-1]