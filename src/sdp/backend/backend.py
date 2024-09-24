from abc import ABC, abstractmethod
from typing import Any, Tuple, Callable, Union

import numpy as np

class RelaxationVariable():
    """
    A relaxation variable k*x + b.
    """
    __slots__ = ['k', 'b']
    def __init__(self, k: float = 1, b: float = 0) -> None:
        self.k = k
        self.b = b
    def __repr__(self) -> str:
        return f"RelaxationVariable(k={self.k}, b={self.b})"
    def __str__(self) -> str:
        return self.__repr__()

def np_array(vec, flatten=False):
    vec = np.array(vec).astype(np.float64)
    if flatten: vec = vec.flatten()
    return vec

_STANDARDIZED_OPERATORS = {
    '>': '__ge__',
    '<': '__le__',
    '=': '__eq__',
    '==': '__eq__',
    '>=': '__ge__',
    '<=': '__le__',
    '__gt__': '__ge__',
    '__lt__': '__le__',
    '__ge__': '__ge__',
    '__le__': '__le__',
    '__eq__': '__eq__'
}


class SDPBackend(ABC):
    def __init__(self, dof, name_y: str='_y') -> None:
        self.dof = dof
        self.y = self.add_vector_variable(name_y, dof + 1)
    def __new__(cls, dof, *args, **kwargs):
        if dof == 0 and cls is not DegeneratedBackend:
            return DegeneratedBackend(dof)
        return super().__new__(cls)
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dof={self.dof}+1)"
    def __str__(self) -> str:
        return self.__repr__()

    @abstractmethod
    def _add_vector_variable(self, name: str, shape: int) -> Any: ...

    def add_vector_variable(self, name: str, shape: int) -> Any:
        """
        Add a vector variable.
        """
        return self._add_vector_variable(name, shape)

    @abstractmethod
    def _add_linear_matrix_inequality(self, name: str, x0: np.ndarray, extended_space: np.ndarray) -> np.ndarray: ...

    def add_linear_matrix_inequality(self, name: str, x0: np.ndarray, extended_space: np.ndarray) -> np.ndarray:
        """
        Add x0 + extended_space @ y >= 0 where y is the variable to be optimized.
        """
        return self._add_linear_matrix_inequality(name, x0, extended_space)

    # @abstractmethod
    def _add_constraint(self, constraint: np.ndarray, rhs: float = 0, operator='__ge__') -> None:
        """
        Add a constraint constraint @ y (operator) rhs.
        It is defaulted to be converted to 1d PSD matrix inequality.
        """
        # if constraint.ndim == 1:
        #     constraint = constraint.reshape(1, -1)
        # constraint = constraint.T.copy()
        constraint = constraint.reshape((1, -1)).copy()

        rhs = np.array([-rhs], dtype=np.float64)
        name = 'constraint_%d'%(id(constraint))
    
        if operator == '__le__':
            constraint, rhs = -constraint, -rhs

        lmi = self._add_linear_matrix_inequality(name, rhs, constraint)

        if operator == '__eq__':
            constraint = -constraint
            name = 'constraint_%d'%(id(constraint))
            self._add_linear_matrix_inequality(name, -rhs, constraint)
        return lmi

    def add_constraint(self, constraint: np.ndarray, rhs: float = 0, operator='__ge__') -> None:
        """
        Add a constraint constraint @ y (operator) rhs.
        It is defaulted to be converted to 1d PSD matrix inequality.
        """
        operator = _STANDARDIZED_OPERATORS.get(operator, None)
        if operator is None:
            raise ValueError(f"Operator {operator} is not supported.")
        constraint = self.extend_vector(np_array(constraint, flatten=True))
        return self._add_constraint(constraint, float(rhs), operator)

    @abstractmethod
    def _set_objective(self, objective: np.ndarray) -> None: ...

    def set_objective(self, objective: np.ndarray) -> None:
        """
        Set the objective function.
        """
        objective = self.extend_vector(np_array(objective, flatten=True))
        return self._set_objective(objective)

    @abstractmethod
    def solve(self, *args, **kwargs) -> Any: ...

    @classmethod
    def extend_space(self, x0: np.ndarray, space: np.ndarray, min_eigen: Union[float, RelaxationVariable] = 0) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(min_eigen, RelaxationVariable):
            k, b = min_eigen.k, min_eigen.b
        else:
            k, b = 0, min_eigen
        x = x0.copy()
        space2 = np.hstack([space, np.full((space.shape[0], 1), -k)])
        m = space.shape[0]
        for i in range(m):
            if i**2 >= m:
                break
            x[i**2] -= b # the diagonal elements
        return x, space2

    def extend_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Extend the vector to match the size of the optimization variable (with a relaxation variable).
        """
        if vector.size == self.dof:
            return np.concatenate([vector, [0]])
        return vector

class DegeneratedBackend(SDPBackend):
    """
    Return array([]) if there is no optimization variable.
    """
    def add_vector_variable(self, name: str, shape: int) -> Any: ...
    def add_psd_variable(self, name: str, shape: int, min_eigen: float = 0) -> Any: ...
    def add_linear_matrix_inequality(self, name: str, x0: np.ndarray, extended_space: np.ndarray) -> np.ndarray: ...
    def add_constraint(self, constraint: str) -> None: ...
    def set_objective(self, objective: str) -> None: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dof=0)"

    def solve(self, *args, **kwargs):
        return np.array([], dtype=np.float64)


def max_relax_var_objective(dof: int) -> np.ndarray:
    """
    Return the objective function maximizing the relaxation variables.
    """
    vec = np.zeros(dof + 1)
    vec[-1] = -1
    return vec

def min_trace_objective(space: np.ndarray) -> np.ndarray:
    """
    Return the objective function minimizing the trace of the matrix.
    """
    space = np.array(space).astype(np.float64)
    m = round(np.sqrt(space.shape[0]))
    return space[::m+1, :].sum(axis=0)

def max_trace_objective(space: np.ndarray) -> np.ndarray:
    """
    Return the objective function maximizing the trace of the matrix.
    """
    space = np.array(space).astype(np.float64)
    m = round(np.sqrt(space.shape[0]))
    return -space[::m+1, :].sum(axis=0)

def min_inner_objective(space: np.ndarray, S: Union[float, np.ndarray]) -> np.ndarray:
    """
    Return the objective function maximizing the inner product of the matrix.
    """
    space = np.array(space).astype(np.float64)
    if isinstance(S, (float, int)):
        return space.sum(axis=0) * S
    S = np.array(S).astype(np.float64).flatten()
    return (S @ space).flatten()

def max_inner_objective(space: np.ndarray, S: Union[float, np.ndarray]) -> np.ndarray:
    """
    Return the objective function maximizing the inner product of the matrix.
    """
    space = np.array(space).astype(np.float64)
    if isinstance(S, (float, int)):
        return space.sum(axis=0) * -S
    S = np.array(S).astype(np.float64).flatten()
    return -(S @ space).flatten()