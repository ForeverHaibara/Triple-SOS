from typing import List, Tuple, Dict, Union, Optional, Any, Type

import numpy as np
from numpy import ndarray
from sympy import MutableDenseMatrix as Matrix

from .backend import DualBackend
from .clarabel_sdp import DualBackendCLARABEL
from .cvxopt_sdp import DualBackendCVXOPT
from .cvxpy_sdp import DualBackendCVXPY
from .mosek_sdp import DualBackendMOSEK
from .picos_sdp import DualBackendPICOS
from .sdpap_sdp import DualBackendSDPAP

from .settings import SDPError, SDPResult
# from ..utils import collect_constraints

_DUAL_BACKENDS: Dict[str, DualBackend] = {
    'clarabel': DualBackendCLARABEL,
    'cvxopt': DualBackendCVXOPT,
    'cvxpy': DualBackendCVXPY,
    'mosek': DualBackendMOSEK,
    'picos': DualBackendPICOS,
    'sdpa': DualBackendSDPAP,
    # 'sdpap': DualBackendSDPAP,
}

_PRIMAL_BACKENDS: Dict[str, Any] = {
    # 'clarabel': PrimalBackendCLARABEL,
    # 'cvxpy': PrimalBackendCVXPY,
    # 'mosek': PrimalBackendMOSEK,
    # 'picos': PrimalBackendPICOS,
}

_RECOMMENDED_BACKENDS = [
    'mosek', 'clarabel', 'cvxopt', 'cvxpy', 'picos', 'sdpa',
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

def collect_constraints(constraints: List[Tuple[ndarray, float, str]], dof: int)\
        -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Collect constraints and separate them into inequality and equality constraints.
    """
    as_array = DualBackend.as_array # TODO

    ineq_lhs, ineq_rhs = [], []
    eq_lhs, eq_rhs = [], []
    for constraint, rhs, op in constraints:
        op = _STANDARDIZED_OPERATORS[op]
        if isinstance(rhs, (float, int)) or not hasattr(rhs, '__len__'):
            rhs = [rhs]

        constraint = as_array(constraint)
        if len(constraint.shape) == 1:
            constraint = constraint.reshape(1, dof)
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
        ineq_lhs, ineq_rhs = np.zeros((0, dof)), np.zeros((0,))

    if len(eq_lhs):
        eq_lhs, eq_rhs = np.vstack(eq_lhs), np.concatenate(eq_rhs)
    else:
        eq_lhs, eq_rhs = np.zeros((0, dof)), np.zeros((0,))
    return ineq_lhs, ineq_rhs, eq_lhs, eq_rhs


def create_numerical_dual_sdp(
        x0_and_space: Union[List[Tuple[Matrix, Matrix]], Dict[Any, Tuple[Matrix, Matrix]]],
        objective: ndarray,
        constraints: List[Tuple[ndarray, float, str]] = [],
        solver: Optional[Union[str, Type[DualBackend]]] = None,
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

    if not isinstance(x0_and_space, (dict, list)):
        raise TypeError(f'x0_and_space must be a dict or list, but got {type(x0_and_space)}.')
    elif isinstance(x0_and_space, dict):
        x0_and_space = list(x0_and_space.values())

    as_array = backend.as_array

    x0_and_space = [(as_array(x0).flatten(), as_array(space)) for x0, space in x0_and_space]
    x0_and_space = [(x0, space) for x0, space in x0_and_space if x0.shape[0] > 0]

    As = [space for x0, space in x0_and_space]
    bs = [x0 for x0, space in x0_and_space]

    c = as_array(objective).flatten()

    ineq_lhs, ineq_rhs, eq_lhs, eq_rhs = collect_constraints(constraints, c.size)
    backend = backend(As, bs, ineq_lhs, ineq_rhs, eq_lhs, eq_rhs, c)
    return backend


def solve_numerical_dual_sdp(
        x0_and_space: Union[List[Tuple[Matrix, Matrix]], Dict[Any, Tuple[Matrix, Matrix]]],
        objective: ndarray,
        constraints: List[Tuple[ndarray, float, str]] = [],
        solver: Optional[str] = None,
        return_result: bool = False,
        verbose: Union[bool, int] = 0,
        max_iters: int = 200,
        tol_fsb_abs: float = 1e-8,
        tol_fsb_rel: float = 1e-8,
        tol_gap_abs: float = 1e-8,
        tol_gap_rel: float = 1e-8,
        solver_options: Dict[str, Any] = {},
    ) -> Optional[Union[ndarray, SDPResult]]:
    """
    Solve for y such that all(Mat(x0 + space @ y) >> 0 for x0, space in x0_and_space.values()).
    This is the dual form of SDP problem.

    Parameters
    ----------
    x0_and_space : Tuple[List[Tuple[Matrix, Matrix]], Dict[str, Tuple[Matrix, Matrix]]]
        A list or a dictionary of x0 and space matrices.
    objective : ndarray
        The objective function, which is a vector.
    constraints : List[Tuple[ndarray, float, str]]
        A list of constraints, each represented as a tuple of (constraint, rhs, operator).
    solver : str
        The solver to use, defaults to None (auto selected). Refer to _DUAL_BACKEND for all solvers,
        but users should install the corresponding packages.
    return_result : bool
        Whether to return a SDPResult object. If True, the return value is a SDPResult object.
        Otherwise, the return value is an 1D numpy array.
    """
    backend = create_numerical_dual_sdp(x0_and_space, objective, constraints, solver=solver)

    result = backend.solve(
        verbose=verbose,
        max_iters=max_iters,
        tol_fsb_abs=tol_fsb_abs,
        tol_fsb_rel=tol_fsb_rel,
        tol_gap_abs=tol_gap_abs,
        tol_gap_rel=tol_gap_rel,
        solver_options=solver_options,
    )
    if return_result:
        return result
    return result.raises()


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


def _fill_space(space: ndarray, n: int, bias: int) -> ndarray:
    """Set space[k(i,j), bias+i*n+j] = space[k(i,j), bias+j*n+i] = 1 for 0 <= i <= j < n
    where space has n*(n+1)//2 rows and k(i,j) is the index of (i,j) in the sorted set (0 <= i <= j < n).
    The modification is in-place.
    """
    i, j = np.triu_indices(n)
    cols = np.arange(bias, bias + n*(n+1)//2)
    rows1 = i*n + j
    rows2 = j*n + i
    space[rows1, cols] = 1
    space[rows2, cols] = 1
    return space

def _extract_triu(space: ndarray, n: int) -> ndarray:
    """Assume space has shape m x N where N = n**2. Return a matrix of shape m * (n*(n+1)//2)
    where each column is the upper triangular part of the corresponding column of space."""
    i, j = np.triu_indices(n)
    return space.T.reshape(n, n, -1)[i, j, :].T

def solve_numerical_primal_sdp(
        x0_and_space: Tuple[ndarray, Union[List[ndarray], Dict[Any, ndarray]]],
        objective: ndarray,
        constraints: List[Tuple[ndarray, float, str]] = [],
        solver: Optional[str] = None,
        return_result: bool = False,
        verbose: Union[bool, int] = 0,
        max_iters: int = 200,
        tol_fsb_abs: float = 1e-8,
        tol_fsb_rel: float = 1e-8,
        tol_gap_abs: float = 1e-8,
        tol_gap_rel: float = 1e-8,
        solver_options: Dict[str, Any] = {},
    ) -> Optional[ndarray]:
    """
    Solve for x such that Sum(space_i @ Si) = x0.
    This is the primal form of SDP problem.

    Now the implementation converts the primal form to an exact dual form.
    TODO: shall we implement a primal backend class directly?

    Parameters
    ----------
    x0_and_space : Tuple[ndarray, Union[List[ndarray], Dict[Any, ndarray]]]
        Vector x0 and a list or a dictionary of space matrices.
    objective : ndarray
        The objective function, which is a vector.
    constraints : List[Tuple[ndarray, float, str]]
        A list of constraints, each represented as a tuple of (constraint, rhs, operator).
    solver : str
        The solver to use, defaults to None (auto selected). Refer to _DUAL_BACKEND for all solvers,
        but users should install the corresponding packages.
    return_result : bool
        Whether to return a SDPResult object. If True, the return value is a SDPResult object.
        Otherwise, the return value is an 1D numpy array.
    """
    x0, spaces = x0_and_space

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

    if not isinstance(spaces, (dict, list)):
        raise TypeError(f'spaces must be a dict or list, but got {type(spaces)}.')
    elif isinstance(spaces, dict):
        spaces = list(spaces.values())

    asarray = backend.as_array
    x0 = asarray(x0).flatten()
    objective = asarray(objective).flatten()

    if x0.size > 0:
        spaces = [asarray(space).reshape(x0.size, -1) for space in spaces]
    else:
        spaces = [asarray(space).reshape(0, space.size) for space in spaces]
    spaces = [space.copy() for space in spaces if space.shape[1] > 0]

    # Formulate the dual form (but not lagrangian dual) by creating
    # a dual SDP with sum(n*(n+1)//2) degrees of freedom.
    # Each entry of the vector represents the (i,j),(j,i) entries of a symmetric matrix.
    sizes = [int(round(np.sqrt(space.shape[1]))) for space in spaces]
    dof = sum(n*(n+1)//2 for n in sizes)

    ineq_lhs, ineq_rhs, eq_lhs, eq_rhs = collect_constraints(constraints, objective.size)
    As = [np.zeros((n**2, dof), dtype=np.float64) for n in sizes]
    bs = [np.zeros((n**2,)) for n in sizes]
    bias = 0
    for i, n in enumerate(sizes):
        _fill_space(As[i], n, bias)
        bias += n*(n+1)//2

    def _extract_triu_multiple(mat):
        target = np.zeros((mat.shape[0], dof), dtype=np.float64)
        bias, bias2 = 0, 0
        for n in sizes:
            space = _extract_triu(mat[:, bias2:bias2+n**2], n)
            target[:, bias:bias+n*(n+1)//2] = space
            bias += n*(n+1)//2
            bias2 += n**2
        return target


    # constraints at off-diagonals are doubled since only the upper triangular
    # contributes to the sum
    bias = 0
    for space, n in zip(spaces, sizes):
        space[:, np.arange(0, n**2, n+1)] /= 2.
        space *= 2.
        for mat in (objective, ineq_lhs, eq_lhs):
            mat[..., bias:bias+n**2][..., np.arange(0, n**2, n+1)] /= 2.
            mat[..., bias:bias+n**2] *= 2.
        bias += n**2

    c = _extract_triu_multiple(objective.reshape(1, objective.size)).flatten()
    eq_lhs = np.vstack([eq_lhs, np.hstack(spaces) if len(spaces)
                                 else np.zeros((x0.shape[0],0), dtype=np.float64)])
    eq_rhs = np.concatenate([eq_rhs, x0])

    ineq_lhs = _extract_triu_multiple(ineq_lhs)
    eq_lhs = _extract_triu_multiple(eq_lhs)

    backend = backend(As, bs, ineq_lhs, ineq_rhs, eq_lhs, eq_rhs, c)
    result = backend.solve(
        verbose=verbose,
        max_iters=max_iters,
        tol_fsb_abs=tol_fsb_abs,
        tol_fsb_rel=tol_fsb_rel,
        tol_gap_abs=tol_gap_abs,
        tol_gap_rel=tol_gap_rel,
        solver_options=solver_options,
    )
    if result.y is not None:
        # restore the triu vector representation to the original matrix representation
        def _triu_to_mat(vec: np.ndarray, n: int) -> np.ndarray:
            mat = np.zeros((n, n))
            triu = np.triu_indices(n)
            mat[triu] = vec
            return mat + mat.T - np.diag(np.diag(mat))
        new_y = np.zeros((sum(n**2 for n in sizes),), dtype=np.float64)
        bias, bias2 = 0, 0
        for n in sizes:
            new_y[bias2:bias2+n**2] = _triu_to_mat(result.y[bias:bias+n*(n+1)//2], n).flatten()
            bias += n*(n+1)//2
            bias2 += n**2
        result.y = new_y

    if return_result:
        return result
    return result.raises()