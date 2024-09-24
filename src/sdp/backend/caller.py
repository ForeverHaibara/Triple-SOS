from typing import List, Tuple, Dict, Union, Callable, Optional, Any

import numpy as np
from sympy import MutableDenseMatrix as Matrix

from .backend import SDPBackend, DegeneratedBackend, np_array
from .cvxopt_sdp import SDPBackendCVXOPT
from .cvxpy_sdp import SDPBackendCVXPY
from .picos_sdp import SDPBackendPICOS
from .sdpap_sdp import SDPBackendSDPAP
from ..utils import Mat2Vec


_BACKENDS = {
    'cvxopt': SDPBackendCVXOPT,
    'cvxpy': SDPBackendCVXPY,
    'picos': SDPBackendPICOS,
    'sdpa': SDPBackendSDPAP,
    'sdpap': SDPBackendSDPAP,
}

_RECOMMENDED_BACKENDS = [
    'cvxpy', 'picos', 'cvxopt', 'sdpa'
]

def get_default_sdp_backend() -> str:
    for backend in _RECOMMENDED_BACKENDS:
        if _BACKENDS[backend].is_available():
            return backend
    return 'cvxpy'
    # raise ImportError('No available SDP solver. Please install one of the following packages: ' + ', '.join(_RECOMMENDED_BACKENDS))


def _create_numerical_dual_sdp(
        x0_and_space: Dict[str, Tuple[Matrix, Matrix]],
        objective: np.ndarray,
        constraints: List[Tuple[np.ndarray, float, str]] = [],
        min_eigen: Union[float, tuple, Dict[str, Union[float, tuple]]] = 0,
        solver: Optional[str] = None,
    ) -> SDPBackend:
    """
    Create a numerical dual SDP problem.
    """
    dof = next(iter(x0_and_space.values()))[1].shape[1]
    if dof == 0:
        # nothing to optimize
        return DegeneratedBackend(dof)

    if solver is None:
        solver = get_default_sdp_backend()
    if solver not in _BACKENDS:
        raise ValueError(f'Unknown solver "{solver}".')
    backend: SDPBackend = _BACKENDS[solver](dof)

    if isinstance(min_eigen, (float, int, tuple)):
        min_eigen = {key: min_eigen for key in x0_and_space.keys()}

    for key, (x0, space) in x0_and_space.items():
        k = Mat2Vec.length_of_mat(x0.shape[0])
        if k == 0:
            continue

        x0_ = np_array(x0, flatten=True)
        space_ = np_array(space)

        x0_, space_ = SDPBackend.extend_space(x0_, space_, min_eigen.get(key, 0))

        backend.add_linear_matrix_inequality(key, x0_, space_)

    backend.set_objective(objective)
    for constraint, rhs, op in constraints:
        backend.add_constraint(constraint, rhs, op)

    return backend


def solve_numerical_dual_sdp(
        x0_and_space: Dict[str, Tuple[np.ndarray, np.ndarray]],
        objective: np.ndarray,
        constraints: List[Tuple[np.ndarray, float, str]] = [],
        min_eigen: Union[float, tuple, Dict[str, Union[float, tuple]]] = 0,
        solver: Optional[str] = None,
        solver_options: Dict[str, Any] = {},
    ) -> Optional[np.ndarray]:
    """
    Solve for y such that all(Mat(x0 + space @ y) >> 0 for x0, space in x0_and_space.values()).
    This is the dual form of SDP problem.

    Parameters
    ----------
    x0_and_space : Dict[str, Tuple[np.ndarray, np.ndarray]]
        A dictionary of x0 and space matrices.
    objective : np.ndarray
        The objective function, which is a vector.
    constraints : List[Tuple[np.ndarray, float, str]]
        A list of constraints, each represented as a tuple of (constraint, rhs, operator).
    min_eigen : Union[float, tuple, Dict[str, Union[float, tuple]]]
        The minimum eigenvalue of each PSD matrices, defaults to 0. But perturbation is allowed.
    solver : str
        The solver to use, defaults to None (auto selected). Refer to _BACKENDS for all solvers,
        but users should install the corresponding packages.
    """
    backend = _create_numerical_dual_sdp(x0_and_space, objective, constraints, min_eigen, solver)

    try:
        y = backend.solve(solver_options)
    except Exception as e:
        # raise e
        return None
    # y = backend.solve()
    return y