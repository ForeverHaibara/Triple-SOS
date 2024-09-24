from math import sqrt
from typing import Any

import numpy as np

from .backend import SDPBackend


class SDPBackendCVXPY(SDPBackend):
    """
    CVXPY backend for SDP problems.
    CVXPY is a Python-embedded modeling language for convex optimization problems.

    Installation:
    pip install cvxpy

    Reference:
    [1] https://www.cvxpy.org/api_reference/cvxpy.html
    """
    def __init__(self, dof) -> None:
        super().__init__(dof)
        from cvxpy import Variable
        self.y = Variable(dof + 1)
        self._mats = []
        self._constraints = []
        self._objective = None
        self.problem = None

    def _add_linear_matrix_inequality(self, name: str, x0: np.ndarray, extended_space: np.ndarray) -> np.ndarray:
        from cvxpy import Variable
        k = int(round(sqrt(x0.shape[0])))
        S = Variable((k, k), PSD=True)
        self._constraints.append(S == (x0 + extended_space @ self.y).reshape((k, k)))
        return S

    def _add_constraint(self, constraint: np.ndarray, rhs: float = 0, operator = '__ge__') -> None:
        self._constraints.append(getattr(constraint @ self.y, operator)(rhs))

    @classmethod
    def is_available(cls) -> bool:
        try:
            import cvxpy
            return True
        except ImportError:
            return False

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