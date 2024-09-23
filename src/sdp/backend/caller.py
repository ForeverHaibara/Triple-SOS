from typing import List, Tuple, Dict, Union, Callable, Optional

import numpy as np
from sympy import MutableDenseMatrix as Matrix

from .backend import SDPBackend, DegeneratedBackend, RelaxationVariable
from .picos_sdp import SDPBackendPicos
from ..utils import Mat2Vec


_BACKENDS = {
    'picos': SDPBackendPicos
}


def _create_numerical_dual_sdp(
        x0_and_space: Dict[str, Tuple[Matrix, Matrix]],
        objective: Tuple[str, Callable],
        constraints: List[Callable] = [],
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

    backend = _BACKENDS[solver](dof)

    if isinstance(min_eigen, (float, int, RelaxationVariable)):
        min_eigen = {key: min_eigen for key in x0_and_space.keys()}

    for key, (x0, space) in x0_and_space.items():
        k = Mat2Vec.length_of_mat(x0.shape[0])
        if k == 0:
            continue
        x0_ = np.array(x0).astype(np.float64).flatten()
        space_ = np.array(space).astype(np.float64)

        S = backend.add_psd_variable(key, k, min_eigen=min_eigen.get(key, 0))
        backend.add_linear_matrix_inequality(S, x0_, space_, backend.y)

    backend.set_objective(objective[0], objective[1](backend))
    for constraint in constraints:
        backend.add_constraint(constraint(backend))

    return backend


def solve_numerical_dual_sdp(
        x0_and_space: Dict[str, Tuple[Matrix, Matrix]],
        objective: Tuple[str, Callable],
        constraints: List[Callable] = [],
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
        return None
    # y = backend.solve()
    return y