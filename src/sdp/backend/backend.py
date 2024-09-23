from abc import ABC, abstractmethod
from typing import Any, Tuple, Callable

import numpy as np

class SDPBackend(ABC):
    def __init__(self, dof, name_y: str='_y', name_relax_var:str='_relax_var') -> None:
        self.dof = dof
        self.variables = {}
        self.constraints = []
        self.objective = None
        self.y = self.add_vector_variable(name_y, dof)
        self.relax_var = self.add_vector_variable(name_relax_var, 1)
    def __new__(cls, dof, *args, **kwargs):
        if dof == 0 and cls is not DegeneratedBackend:
            return DegeneratedBackend(dof)
        return super().__new__(cls)
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dof={self.dof})"
    def __str__(self) -> str:
        return self.__repr__()

    @abstractmethod
    def add_vector_variable(self, name: str, shape: int) -> Any: ...

    @abstractmethod
    def add_psd_variable(self, name: str, shape: int, min_eigen: float = 0) -> Any: ...

    @abstractmethod
    def add_linear_matrix_inequality(self, S: Any, x0: Any, space: Any, y: Any) -> None: ...

    @abstractmethod
    def add_constraint(self, constraint: str) -> None: ...

    @abstractmethod
    def set_objective(self, objective: str) -> None: ...

    @abstractmethod
    def solve(self, *args, **kwargs) -> Any: ...

    def get_var(self, name: str) -> Any:
        if isinstance(name, str):
            if name in self.variables:
                return self.variables[name]
            return getattr(self, name)
        return name

    def trace(self, S: Any) -> Any:
        return self.get_var(S).tr

    def inner(self, S1: Any, S2: Any) -> Any:
        return self.get_var(S1) | self.get_var(S2)


class DegeneratedBackend(SDPBackend):
    """
    Return array([]) if there is no optimization variable.
    """
    def add_vector_variable(self, name: str, shape: int) -> Any: ...
    def add_psd_variable(self, name: str, shape: int, min_eigen: float = 0) -> Any: ...
    def add_linear_matrix_inequality(self, S: Any, x0: Any, space: Any, y: Any) -> None: ...
    def add_constraint(self, constraint: str) -> None: ...
    def set_objective(self, objective: str) -> None: ...

    def solve(self, *args, **kwargs):
        return np.array([], dtype=np.float64)


class RelaxationVariable():
    """
    A relaxation variable k*x + b.
    """
    __slots__ = ['k', 'b']
    def __init__(self, k: float = 1, b: float = 0) -> None:
        self.k = k
        self.b = b

def max_relax_var_objective() -> Tuple[str, Callable]:
    """
    Return the objective function maximizing the relaxation variables.
    """
    return 'max', lambda x: x.relax_var