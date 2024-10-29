from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Union

import numpy as np

def np_array(vec, flatten=False):
    """
    Convert the input to a numpy array.    
    """
    vec = np.array(vec).astype(np.float64)
    if flatten: vec = vec.flatten()
    return vec

def random_name() -> str:
    """
    Get a random name for the variable. This is used for some backends.
    """
    import uuid
    return str(uuid.uuid4())

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
    '__eq__': '__eq__',
    '__leq__': '__le__',
    '__geq__': '__ge__',
}


class SDPBackend(ABC):
    _dependencies = tuple()
    def __init__(self, *args, **kwargs) -> None:
        pass

    @classmethod
    def is_available(cls) -> bool:
        for dep in cls._dependencies:
            try:
                __import__(dep)
            except ImportError:
                return False
        return True

    @abstractmethod
    def _add_constraint(self, constraint: np.ndarray, rhs: float = 0, operator='__ge__') -> None: ...

    def add_constraint(self, constraint: np.ndarray, rhs: float = 0, operator='__ge__') -> None:
        """
        Add a constraint constraint @ y (operator) rhs.
        It is defaulted to be converted to 1d PSD matrix inequality.
        """
        op = _STANDARDIZED_OPERATORS.get(operator, None)
        if op is None:
            raise ValueError(f"Operator {operator} is not supported.")
        constraint = self.extend_vector(constraint).flatten()
        return self._add_constraint(constraint, float(rhs), op)

    def extend_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Extend the vector to match the size of the optimization variable (with a relaxation variable).
        If the vector has ndim > 1, then the operation is performed on the final axis.
        """
        vector = np_array(vector) #, flatten=True)
        # if vector.size == self.dof:
        #     return np.concatenate([vector, [0]])
        if vector.shape[-1] == self.dof:
            return np.concatenate([vector, np.zeros(vector.shape[:-1] + (1,))], axis=-1)
        return vector

    def add_relax_var_nonnegative_inequality(self, rhs: float = 0, operator='__ge__') -> None:
        """
        Add a constraint r >= 0.
        """
        vec = np.zeros(self.dof + 1)
        vec[-1] = 1
        return self.add_constraint(vec, rhs, operator=operator)

    @abstractmethod
    def _set_objective(self, objective: np.ndarray) -> None: ...

    def set_objective(self, objective: np.ndarray) -> None:
        """
        Set the objective function.
        """
        objective = self.extend_vector(objective).flatten()
        return self._set_objective(objective)

    @abstractmethod
    def solve(self, *args, **kwargs) -> Any: ...

class DualBackend(SDPBackend, ABC):
    """
    DualBackend solves the dual problem of the SDP problem numerically.
    which is in the form of:
    
        Mat(x + space @ y) >> 0,
    
    where y is the variable to be optimized. All linear constraints / objectives
    should be represented as an inner product of a vector and y. We allow multiple
    linear matrix inequalities by calling `add_linear_matrix_inequality()` multiple times.

    In addition to the standard form, we also permit a relaxation on the eigenvalue:

        Mat(x + space @ y) >> (k * r + b)*I,

    where (k,b) is a tuple and r is a relaxation variable. The relaxation variable
    is appended to the end of the optimization variable, so our SDP dual problem
    has actually (dof + 1) variables. Constraints of such eigenvalue relaxation
    can be passed in by calling extend_space(x0, space, min_eigen=(k, b)).
    If `min_eigen` is only a float, the relaxation variable is ignored and it is interpreted
    as:

        Mat(x + space @ y) >> b*I.
    
    After the problem is created, the solution is returned by calling `solve()`.
    """
    def __init__(self, dof) -> None:
        self.dof = dof
        self.y = None

    def __new__(cls, dof, *args, **kwargs):
        if dof == 0 and cls is not DegeneratedDualBackend:
            return DegeneratedDualBackend(dof)
        return super().__new__(cls)
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dof={self.dof}+1)"
    def __str__(self) -> str:
        return self.__repr__()

    @abstractmethod
    def _add_linear_matrix_inequality(self, x0: np.ndarray, extended_space: np.ndarray) -> np.ndarray: ...

    def add_linear_matrix_inequality(self, x0: np.ndarray, space: np.ndarray, min_eigen: Union[float, tuple] = 0) -> np.ndarray:
        """
        Add x0 + extended_space @ y >= 0 where y is the variable to be optimized.
        """
        x0 = np_array(x0, flatten=True)
        space = np_array(space)
        x0, extended_space = self.extend_space(x0, space, min_eigen)
        return self._add_linear_matrix_inequality(x0, extended_space)

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
    
        if operator == '__le__':
            constraint, rhs = -constraint, -rhs

        lmi = self._add_linear_matrix_inequality(rhs, constraint)

        if operator == '__eq__':
            constraint = -constraint
            self._add_linear_matrix_inequality(-rhs, constraint)
        return lmi

    def extend_space(self, x0: np.ndarray, space: np.ndarray, min_eigen: Union[float, Tuple[float, float]] = 0) -> Tuple[np.ndarray, np.ndarray]:
        if space.shape[1] == self.dof + 1:
            if min_eigen != 0:
                raise ValueError("The space is already extended. Cannot set min_eigen.")
            return x0, space
        elif space.shape[1] != self.dof:
            raise ValueError(f"Space has {space.shape[1]} columns, but the optimization variable has {self.dof} columns.")

        if isinstance(min_eigen, tuple):
            k, b = min_eigen
        else:
            k, b = 0, min_eigen
        k, b = float(k), float(b)

        x = x0.copy()
        m = int(round(np.sqrt(space.shape[0])))
        # subtract the relaxation variable on the diagonal
        space2 = np.hstack([space, np.eye(m).reshape((m**2, 1)) * (-k)])
        for i in range(m):
            x[i**2] -= b # the diagonal elements
        return x, space2


class DegeneratedDualBackend(DualBackend):
    """
    Return array([]) if there is no optimization variable.
    """
    def _add_linear_matrix_inequality(self, x0: np.ndarray, extended_space: np.ndarray) -> np.ndarray: ...
    def _add_constraint(self, constraint: np.ndarray, rhs: float = 0, operator='__ge__') -> None: ...
    def _set_objective(self, objective: np.ndarray) -> None: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dof=0)"

    def solve(self, *args, **kwargs):
        return np.array([], dtype=np.float64)


class PrimalBackend(SDPBackend, ABC):
    """
    PrimalBackend solves the primal problem of the SDP problem numerically.
    which is in the form of:
    
        S1, ..., SN >> 0,
        space1 @ vec(S1) + ... + spaceN @ vec(SN) = x0,
    
    where S1, ..., SN are PSD variables to be optimized. All linear constraints / objectives
    should be reprsent as an inner product of a vector and the concatenated vector of the PSD variables.

    In addition to the standard form, we also permit a relaxation on the eigenvalue:
    
            Si >> (ki * r + bi)*I,
    
    where (k,b) is a tuple and r is a relaxation variable. The relaxation variable is a 1d matrix.
    Constraints of such eigenvalue relaxation can be passed in by calling add_linear_matrix_equality(space, min_eigen=(k, b)).
    If `min_eigen` is only a float, the relaxation variable is ignored and it is interpreted as:
    
        Si >> bi*I.

    We are then solving for:

        Si' >> 0,
        space1 @ vec(Si' + (ki * r + bi)*I) + ... + spaceN @ vec(SN' + (kN * r + bN)*I) = x0.

    After the problem is created, the solution is returned by calling `solve()`.
    """
    def __init__(self, x0: np.ndarray) -> None:
        self.x0 = np_array(x0, flatten=True)
        self.dof = 0
        self._min_eigen_space = np.zeros((self.x0.shape[0], 1)) # the equality constraints on the relaxation variable
        self._eigen_relaxations = []
        self._mat_size = []
        self._spaces: List[np.ndarray] = []

        self._objective = None
        self._ineq_spaces: List[np.ndarray] = []

    # @property
    # def eqnum(self) -> int:
    #     """
    #     Eqnum of the primal problem is the number of equality constraints.
    #     It is also defined to be the dof of its dual.
    #     """
    #     return self.x0.shape[0]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dof={self.dof}+1)"
    def __str__(self) -> str:
        return self.__repr__()

    # @abstractmethod
    def _add_linear_matrix_equality(self, space: np.ndarray) -> None:
        self._spaces.append(space)

    def add_linear_matrix_equality(self, space: np.ndarray, min_eigen: Union[float, Tuple[float, float]] = 0) -> None:
        if isinstance(min_eigen, tuple):
            k, b = min_eigen
        else:
            k, b = 0, min_eigen
        k, b = float(k), float(b)

        space = np_array(space)
        m = int(round(np.sqrt(space.shape[1])))

        if k != 0 or b != 0:
            diag = np.eye(m).flatten()
            x = space @ diag
            if k != 0:
                self._min_eigen_space += x.reshape((-1,1)) * k
            if b != 0:
                self.x0 -= x * b

        self.dof += space.shape[1]
        self._mat_size.append(m)
        self._eigen_relaxations.append((k, b))
        return self._add_linear_matrix_equality(space)

    def _set_objective(self, objective: np.ndarray) -> None:
        self._objective = objective

    @property
    def full_space(self) -> np.ndarray:
        space = np.hstack([space for space in self._spaces] + [self._min_eigen_space])
        return space

    def split_vector(self, vec: np.ndarray) -> List[np.ndarray]:
        """
        Split the vector into a list of vectors by the size of the matrices.
        If the vector has ndim > 1, then the operation is performed on the final axis.
        """
        vec = self.extend_vector(vec)
        return np.split(vec, np.cumsum(np.array(self._mat_size + [1])**2)[:-1], axis=-1)

    def restore_eigen(self, vec: np.ndarray) -> np.ndarray:
        """
        Restore the eigenvalue relaxation variable to the original space.
        """
        if not isinstance(vec, list):
            mats = self.split_vector(vec)
        else:
            mats = vec
        eigen = mats[-1].flatten()[0]
        for i in range(len(mats) - 1):
            m = self._mat_size[i]
            k, b = self._eigen_relaxations[i]
            mats[i][np.arange(0,m**2,m+1)] += k * eigen + b
        return np.concatenate(mats[:-1])

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