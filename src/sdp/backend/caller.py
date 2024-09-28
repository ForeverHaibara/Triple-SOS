from typing import List, Tuple, Dict, Union, Optional, Any

from numpy import ndarray
from sympy import MutableDenseMatrix as Matrix

from .backend import DualBackend, PrimalBackend, DegeneratedDualBackend
from .cvxopt_sdp import DualBackendCVXOPT
from .cvxpy_sdp import DualBackendCVXPY, PrimalBackendCVXPY
from .picos_sdp import DualBackendPICOS, PrimalBackendPICOS
from .sdpap_sdp import DualBackendSDPAP
from ..utils import Mat2Vec


_DUAL_BACKENDS: Dict[str, DualBackend] = {
    'cvxopt': DualBackendCVXOPT,
    'cvxpy': DualBackendCVXPY,
    'picos': DualBackendPICOS,
    'sdpa': DualBackendSDPAP,
    'sdpap': DualBackendSDPAP,
}

_PRIMAL_BACKENDS: Dict[str, PrimalBackend] = {
    'cvxpy': PrimalBackendCVXPY,
    'picos': PrimalBackendPICOS,
}

_RECOMMENDED_BACKENDS = [
    'cvxpy', 'picos', 'cvxopt', 'sdpa'
]

def get_default_sdp_backend(dual = True) -> str:
    pointer = _DUAL_BACKENDS if dual else _PRIMAL_BACKENDS
    for backend in _RECOMMENDED_BACKENDS:
        if backend in pointer and pointer[backend].is_available():
            return backend
    return 'cvxpy'
    # raise ImportError('No available SDP solver. Please install one of the following packages: ' + ', '.join(_RECOMMENDED_BACKENDS))


def _create_numerical_dual_sdp(
        x0_and_space: Dict[str, Tuple[Matrix, Matrix]],
        objective: ndarray,
        constraints: List[Tuple[ndarray, float, str]] = [],
        min_eigen: Union[float, tuple, Dict[str, Union[float, tuple]]] = 0,
        solver: Optional[str] = None,
        add_relax_var_nonnegative_inequality: bool = True,
    ) -> DualBackend:
    """
    Create a numerical dual SDP problem.
    """
    dof = next(iter(x0_and_space.values()))[1].shape[1]
    if dof == 0:
        # nothing to optimize
        return DegeneratedDualBackend(dof)

    if solver is None:
        solver = get_default_sdp_backend(dual=True)
    if solver not in _DUAL_BACKENDS:
        raise ValueError(f'Unknown solver "{solver}".')
    backend: DualBackend = _DUAL_BACKENDS[solver](dof)

    if isinstance(min_eigen, (float, int, tuple)):
        min_eigen = {key: min_eigen for key in x0_and_space.keys()}

    for key, (x0, space) in x0_and_space.items():
        k = Mat2Vec.length_of_mat(x0.shape[0])
        if k == 0:
            continue

        backend.add_linear_matrix_inequality(x0, space, min_eigen.get(key, 0))

    backend.set_objective(objective)
    for constraint, rhs, op in constraints:
        backend.add_constraint(constraint, rhs, op)

    if add_relax_var_nonnegative_inequality:
        backend.add_relax_var_nonnegative_inequality()

    return backend


def solve_numerical_dual_sdp(
        x0_and_space: Dict[str, Tuple[ndarray, ndarray]],
        objective: ndarray,
        constraints: List[Tuple[ndarray, float, str]] = [],
        min_eigen: Union[float, tuple, Dict[str, Union[float, tuple]]] = 0,
        solver: Optional[str] = None,
        solver_options: Dict[str, Any] = {},
        raise_exception: bool = False,
    ) -> Optional[ndarray]:
    """
    Solve for y such that all(Mat(x0 + space @ y) >> 0 for x0, space in x0_and_space.values()).
    This is the dual form of SDP problem.

    Parameters
    ----------
    x0_and_space : Dict[str, Tuple[ndarray, ndarray]]
        A dictionary of x0 and space matrices.
    objective : ndarray
        The objective function, which is a vector.
    constraints : List[Tuple[ndarray, float, str]]
        A list of constraints, each represented as a tuple of (constraint, rhs, operator).
    min_eigen : Union[float, tuple, Dict[str, Union[float, tuple]]]
        The minimum eigenvalue of each PSD matrices, defaults to 0. But perturbation is allowed.
    solver : str
        The solver to use, defaults to None (auto selected). Refer to _DUAL_BACKEND for all solvers,
        but users should install the corresponding packages.
    solver_options : Dict[str, Any]
        The options to pass to the solver.
    raise_exception : bool
        Whether to raise exception when the solver fails.
    """
    backend = _create_numerical_dual_sdp(x0_and_space, objective, constraints, min_eigen, solver)

    try:
        y = backend.solve(solver_options)
        if y.size != backend.dof:
            if y.size == 0: # no solution is found
                y = None
            else:
                raise ValueError(f"Solution y has wrong size {y.size}, expected {backend.dof}.")
    except Exception as e:
        if raise_exception:
            raise e
        return None
    # y = backend.solve()
    return y


def _create_numerical_primal_sdp(
        space: Dict[str, ndarray],
        x0: ndarray,
        objective: ndarray,
        constraints: List[Tuple[ndarray, float, str]] = [],
        min_eigen: Union[float, tuple, Dict[str, Union[float, tuple]]] = 0,
        solver: Optional[str] = None,
    ) -> PrimalBackend:
    """
    Create a numerical primal SDP problem.
    """
    if solver is None:
        solver = get_default_sdp_backend(dual=False)
    if solver not in _PRIMAL_BACKENDS:
        raise ValueError(f'Unknown solver "{solver}".')
    backend: PrimalBackend = _PRIMAL_BACKENDS[solver](x0)

    if isinstance(min_eigen, (float, int, tuple)):
        min_eigen = {key: min_eigen for key in space.keys()}
    for key, space_mat in space.items():
        backend.add_linear_matrix_equality(space_mat, min_eigen.get(key, 0))
    
    backend.set_objective(objective)
    for constraint, rhs, op in constraints:
        backend.add_constraint(constraint, rhs, op)

    return backend


def solve_numerical_primal_sdp(
        space: Dict[str, ndarray],
        x0: ndarray,
        objective: ndarray,
        constraints: List[Tuple[ndarray, float, str]] = [],
        min_eigen: Union[float, tuple, Dict[str, Union[float, tuple]]] = 0,
        solver: Optional[str] = None,
        solver_options: Dict[str, Any] = {},
        raise_exception: bool = False,
    ) -> Optional[ndarray]:
    """
    Solve for x such that Sum(space_i @ Si) = x0.
    This is the primal form of SDP problem.
    """
    backend = _create_numerical_primal_sdp(space, x0, objective, constraints, min_eigen, solver)

    try:
        y = backend.solve(solver_options)
        if y.size != backend.dof:
            if y.size == 0: # no solution is found
                y = None
            else:
                raise ValueError(f"Solution y has wrong size {y.size}, expected {backend.dof}.")
    except Exception as e:
        if raise_exception:
            raise e
        return None
    # y = backend.solve()
    return y