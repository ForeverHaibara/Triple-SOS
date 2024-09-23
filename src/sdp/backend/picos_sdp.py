import numpy as np
from typing import Any

from .backend import SDPBackend, RelaxationVariable


class SDPBackendPicos(SDPBackend):
    def __init__(self, dof) -> None:
        super().__init__(dof)
        from picos import Problem
        self.problem = Problem()

    def add_vector_variable(self, name: str, shape: int) -> Any:
        from picos import RealVariable
        self.variables[name] = RealVariable(name, shape)
        return self.variables[name]

    def add_psd_variable(self, name: str, shape: int, min_eigen: float = 0) -> Any:
        from picos import SymmetricVariable
        self.variables[name] = SymmetricVariable(name, (shape, shape))
        if isinstance(min_eigen, RelaxationVariable):
            relax_var = self.relax_var
            self.problem.add_constraint(self.variables[name] >> min_eigen.k * relax_var + min_eigen.b)
            self.problem.add_constraint(min_eigen.k * relax_var + min_eigen.b >= 0)
        elif min_eigen != 0:
            self.problem.add_constraint(self.variables[name] >> min_eigen * np.eye(shape))
        else:
            self.problem.add_constraint(self.variables[name] >> 0)
        return self.variables[name]

    def add_linear_matrix_inequality(self, S: str, x0: np.ndarray, space: np.ndarray, y: str) -> None:
        S = self.get_var(S)
        y = self.get_var(y)
        self.problem.add_constraint(S.vec == x0 + space * y)

    def add_constraint(self, constraint: Any) -> None:
        self.problem.add_constraint(constraint)

    @classmethod
    def is_available(cls) -> bool:
        try:
            import picos
            return True
        except ImportError:
            return False

    def set_objective(self, direction: str, objective: Any) -> None:
        self.problem.set_objective(direction, objective)

    def solve(self) -> np.ndarray:
        self.problem.solve()
        value = self.y.value
        if isinstance(value, float):
            return np.array([value])
        return np.array(value).flatten()