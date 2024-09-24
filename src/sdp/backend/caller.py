from typing import List, Tuple, Dict, Union, Callable, Optional

import numpy as np
from sympy import MutableDenseMatrix as Matrix

from .backend import SDPBackend, DegeneratedBackend, RelaxationVariable, np_array
from .picos_sdp import SDPBackendPICOS
from .cvxopt_sdp import SDPBackendCVXOPT
from .sdpap_sdp import SDPBackendSDPAP
from ..utils import Mat2Vec


_BACKENDS = {
    'picos': SDPBackendPICOS,
    'cvxopt': SDPBackendCVXOPT,
    'sdpa': SDPBackendSDPAP,
    'sdpap': SDPBackendSDPAP,
}


def _create_numerical_dual_sdp(
        x0_and_space: Dict[str, Tuple[Matrix, Matrix]],
        objective: np.ndarray,
        constraints: List[Tuple[np.ndarray, float, str]] = [],
        min_eigen: Union[float, RelaxationVariable, Dict[str, Union[float, RelaxationVariable]]] = 0,
        solver='picos',
    ) -> SDPBackend:
    """
    Create a numerical dual SDP problem.
    """
    dof = next(iter(x0_and_space.values()))[1].shape[1]
    if dof == 0:
        # nothing to optimize
        return DegeneratedBackend(dof)

    backend: SDPBackend = _BACKENDS[solver](dof)

    if isinstance(min_eigen, (float, int, RelaxationVariable)):
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
        x0_and_space: Dict[str, Tuple[Matrix, Matrix]],
        objective: np.ndarray,
        constraints: List[Tuple[np.ndarray, float, str]] = [],
        min_eigen: Union[float, RelaxationVariable, Dict[str, Union[float, RelaxationVariable]]] = 0,
        solver='picos',
        **kwargs
    ) -> Optional[np.ndarray]:
    """
    Solve for y such that all(Mat(x0 + space @ y) >> 0 for x0, space in x0_and_space.values()).
    """
    backend = _create_numerical_dual_sdp(x0_and_space, objective, constraints, min_eigen, solver)

    try:
        y = backend.solve()
    except Exception as e:
        # raise e
        return None
    # y = backend.solve()
    return y