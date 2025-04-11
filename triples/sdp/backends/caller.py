from typing import List, Tuple, Dict, Union, Optional, Any

import numpy as np
from numpy import ndarray
from sympy import MutableDenseMatrix as Matrix

from .backend import DualBackend
# from .cvxopt_sdp import DualBackendCVXOPT
from .cvxpy_sdp import DualBackendCVXPY
from .picos_sdp import DualBackendPICOS

from .settings import SDPError

_DUAL_BACKENDS: Dict[str, DualBackend] = {
    # 'clarabel': DualBackendCLARABEL,
    # 'cvxopt': DualBackendCVXOPT,
    'cvxpy': DualBackendCVXPY,
    # 'mosek': DualBackendMOSEK,
    'picos': DualBackendPICOS,
    # 'sdpa': DualBackendSDPAP,
    # 'sdpap': DualBackendSDPAP,
}

_PRIMAL_BACKENDS: Dict[str, Any] = {
    # 'clarabel': PrimalBackendCLARABEL,
    # 'cvxpy': PrimalBackendCVXPY,
    # 'mosek': PrimalBackendMOSEK,
    # 'picos': PrimalBackendPICOS,
}

_RECOMMENDED_BACKENDS = [
    'picos',
#     'mosek', 'clarabel', 'cvxopt', 'sdpa', 'picos', 'cvxpy',
]

def get_default_sdp_backend(dual = True) -> str:
    pointer = _DUAL_BACKENDS if dual else _PRIMAL_BACKENDS
    for backend in _RECOMMENDED_BACKENDS:
        if backend in pointer and pointer[backend].is_available():
            return backend
    return 'cvxpy'
    # raise ImportError('No available SDP solver. Please install one of the following packages: ' + ', '.join(_RECOMMENDED_BACKENDS))


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

def create_numerical_dual_sdp(
        x0_and_space: Dict[str, Tuple[Matrix, Matrix]],
        objective: ndarray,
        constraints: List[Tuple[ndarray, float, str]] = [],
        solver: Optional[str] = None,
    ) -> DualBackend:
    """
    Create a numerical dual SDP problem.
    """
#     dof = next(iter(x0_and_space.values()))[1].shape[1]
#     if dof == 0:
#         # nothing to optimize
#         return DegeneratedDualBackend(dof)

    if solver is None:
        solver = get_default_sdp_backend(dual=True)
    if isinstance(solver, str):
        if (solver not in _DUAL_BACKENDS):
            raise ValueError(f'Unknown solver "{solver}".')
        backend: DualBackend = _DUAL_BACKENDS[solver]
    elif issubclass(solver, DualBackend):
        backend = solver
    else:
        raise TypeError(f'Unknown solver type "{type(solver)}".')

    as_array = backend.as_array

    x0_and_space = [(as_array(x0).flatten(), as_array(space)) for key, (x0, space) in x0_and_space.items()]
    x0_and_space = [(x0, space) for x0, space in x0_and_space if x0.shape[0] > 0]

    As = [space for x0, space in x0_and_space]
    bs = [x0 for x0, space in x0_and_space]

    c = as_array(objective).flatten()

    ineq_lhs, ineq_rhs = [], []
    eq_lhs, eq_rhs = [], []
    for constraint, rhs, op in constraints:
        op = _STANDARDIZED_OPERATORS[op]
        if isinstance(rhs, (float, int)) or not hasattr(rhs, '__len__'):
            rhs = [rhs]

        constraint = as_array(constraint)
        if len(constraint.shape) == 1:
            constraint = constraint.reshape(1, c.shape[0])
        rhs = as_array(rhs).flatten()

        if op == '__le__':
            constraint, rhs, op = -constraint, -rhs, '__ge__'
        if op == '__ge__':
            ineq_lhs.append(constraint)
            ineq_rhs.append(rhs)
        else: # if op == '__eq__':
            eq_lhs.append(constraint)
            eq_rhs.append(rhs)

    if len(ineq_lhs):
        ineq_lhs, ineq_rhs = np.vstack(ineq_lhs), np.concatenate(ineq_rhs)
    else:
        ineq_lhs, ineq_rhs = np.zeros((0, c.shape[0])), np.zeros((0,))

    if len(eq_lhs):
        eq_lhs, eq_rhs = np.vstack(eq_lhs), np.concatenate(eq_rhs)
    else:
        eq_lhs, eq_rhs = np.zeros((0, c.shape[0])), np.zeros((0,))

    backend = backend(As, bs, ineq_lhs, ineq_rhs, eq_lhs, eq_rhs, c)
    return backend


def solve_numerical_dual_sdp(
        x0_and_space: Dict[str, Tuple[ndarray, ndarray]],
        objective: ndarray,
        constraints: List[Tuple[ndarray, float, str]] = [],
        solver: Optional[str] = None,
        raise_exception: bool = True,
        verbose: Union[bool, int] = 0,
        tol_fsb_abs: float = 1e-8,
        tol_fsb_rel: float = 1e-8,
        tol_gap_abs: float = 1e-8,
        tol_gap_rel: float = 1e-8,
        solver_options: Dict[str, Any] = {},
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
    solver : str
        The solver to use, defaults to None (auto selected). Refer to _DUAL_BACKEND for all solvers,
        but users should install the corresponding packages.
    raise_exception : bool
        Whether to raise exception when the solver fails.
    """
    backend = create_numerical_dual_sdp(x0_and_space, objective, constraints, solver=solver)

    try:
        y = backend.solve(
            verbose=verbose,
            tol_fsb_abs=tol_fsb_abs,
            tol_fsb_rel=tol_fsb_rel,
            tol_gap_abs=tol_gap_abs,
            tol_gap_rel=tol_gap_rel,
            solver_options=solver_options,
        )
    except SDPError as e:
        if raise_exception:
            raise e from None
        return None
    return y


# def _create_numerical_primal_sdp(
#         space: Dict[str, ndarray],
#         x0: ndarray,
#         objective: ndarray,
#         constraints: List[Tuple[ndarray, float, str]] = [],
#         min_eigen: Union[float, tuple, Dict[str, Union[float, tuple]]] = 0,
#         scaling: float = 6.,
#         solver: Optional[str] = None,
#         add_relax_var_nonnegative_inequality: bool = True,
#     ) -> PrimalBackend:
#     """
#     Create a numerical primal SDP problem.
#     """
#     if solver is None:
#         solver = get_default_sdp_backend(dual=False)
#     if solver not in _PRIMAL_BACKENDS:
#         raise ValueError(f'Unknown solver "{solver}".')

#     if isinstance(min_eigen, (float, int, tuple)):
#         min_eigen = {key: min_eigen for key in space.keys()}
#         min_eigen = {key: (0, b) if not isinstance(b, tuple) else b for key, b in min_eigen.items()}

#     x0 = np_array(x0, flatten=True)
#     space = {key: np_array(space_mat) for key, space_mat in space.items()}
#     if scaling > 0:
#         _max_entry = max(max(abs(space_mat).max() if space_mat.size > 0 else 0 for space_mat in space.values()), abs(x0).max())
#         scaling = scaling / _max_entry if _max_entry > 0 else 0
#         if scaling > 0:
#             for key, space_mat in space.items():
#                 space[key] = space_mat * scaling
#             x0 = x0 * scaling
#             # min_eigen = {key: (k, b) for key, (k, b) in min_eigen.items()}
#             # constraints = [(c, rhs, op) for c, rhs, op in constraints]


#     backend: PrimalBackend = _PRIMAL_BACKENDS[solver](x0)
#     for key, space_mat in space.items():
#         backend.add_linear_matrix_equality(space_mat, min_eigen.get(key, 0))

#     if add_relax_var_nonnegative_inequality:
#         backend.add_relax_var_nonnegative_inequality()

#     backend.set_objective(objective)
#     for constraint, rhs, op in constraints:
#         backend.add_constraint(constraint, rhs, op)

#     return backend


# def solve_numerical_primal_sdp(
#         space: Dict[str, ndarray],
#         x0: ndarray,
#         objective: ndarray,
#         constraints: List[Tuple[ndarray, float, str]] = [],
#         min_eigen: Union[float, tuple, Dict[str, Union[float, tuple]]] = 0,
#         # lower_bound: Optional[float] = None,
#         scaling: float = 6.,
#         solver: Optional[str] = None,
#         solver_options: Dict[str, Any] = {},
#         raise_exception: bool = False,
#     ) -> Optional[ndarray]:
#     """
#     Solve for x such that Sum(space_i @ Si) = x0.
#     This is the primal form of SDP problem.
#     """
#     backend = _create_numerical_primal_sdp(space, x0, objective, constraints, min_eigen, scaling=scaling, solver=solver)

#     try:
#         y = backend.solve(solver_options)
#         if y is not None and y.size != backend.dof:
#             if y.size == 0: # no solution is found
#                 y = None
#             else:
#                 raise ValueError(f"Solution y has wrong size {y.size}, expected {backend.dof}.")
#     except Exception as e:
#         if raise_exception:
#             raise e
#         return None
#     # y = backend.solve()
#     return y