from math import sqrt
from typing import Any

import numpy as np

from .backend import SDPBackend


class SDPBackendPICOS(SDPBackend):
    """
    PICOS backend for SDP problems.
    Picos is a Python interface to conic optimization solvers,
    and the default solver is CVXOPT.

    Installation:
    pip install picos

    Reference:
    [1] https://picos-api.gitlab.io/picos/index.html
    """
    def __init__(self, dof) -> None:
        super().__init__(dof)
        from picos import Problem
        self.problem = Problem()

    def _add_vector_variable(self, name: str, shape: int) -> Any:
        from picos import RealVariable
        return RealVariable(name, shape)

    def _add_linear_matrix_inequality(self, name: str, x0: np.ndarray, extended_space: np.ndarray) -> np.ndarray:
        from picos import SymmetricVariable
        S = SymmetricVariable(name, (sqrt(x0.shape[0]), sqrt(x0.shape[0])))
        self.problem.add_constraint(S.vec == x0 + extended_space * self.y)
        self.problem.add_constraint(S >> 0)
        return S

    def _add_constraint(self, constraint: np.ndarray, rhs: float = 0, operator = '__ge__') -> None:
        self.problem.add_constraint(getattr(constraint * self.y, operator)(rhs))

    @classmethod
    def is_available(cls) -> bool:
        try:
            import picos
            return True
        except ImportError:
            return False

    def _set_objective(self, objective: np.ndarray) -> None:
        self.problem.set_objective('min', objective * self.y)

    def solve(self) -> np.ndarray:
        self.problem.solve()
        value = self.y.value
        if isinstance(value, float):
            value = [value]
        return np.array(value).flatten()[:-1]