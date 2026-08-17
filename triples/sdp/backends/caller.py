from typing import List, Tuple, Dict, Union, Optional, Any, Type, TYPE_CHECKING

import numpy as np
from scipy import sparse

from .backend import DualBackend
from .clarabel_sdp import DualBackendCLARABEL
from .cvxopt_sdp import DualBackendCVXOPT
from .cvxpy_sdp import DualBackendCVXPY
from .mosek_sdp import DualBackendMOSEK
from .picos_sdp import DualBackendPICOS
from .qics_sdp import DualBackendQICS
from .sdpap_sdp import DualBackendSDPAP
from ..arithmetic.matop import USE_SCIPY_ARRAY, csr_array

if USE_SCIPY_ARRAY:
    from scipy.sparse import diags_array
else:
    from scipy.sparse import diags as diags_array

if TYPE_CHECKING:
    from numpy import ndarray
    from sympy import MutableDenseMatrix as Matrix
    from .settings import SDPResult
# from ..utils import collect_constraints

_DUAL_BACKENDS: Dict[str, DualBackend] = {
    'clarabel': DualBackendCLARABEL,
    'cvxopt': DualBackendCVXOPT,
    'cvxpy': DualBackendCVXPY,
    'mosek': DualBackendMOSEK,
    'picos': DualBackendPICOS,
    'qics': DualBackendQICS,
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
    'mosek', 'clarabel', 'qics', 'cvxopt', 'cvxpy', 'picos', 'sdpa',
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

class _SparseInputBackend(DualBackend):
    _opt_sparse = 'csr'


def _vstack(mats):
    if any(sparse.issparse(mat) for mat in mats):
        return sparse.vstack(mats, format='csr')
    return np.vstack(mats)


def _hstack(mats):
    if any(sparse.issparse(mat) for mat in mats):
        return sparse.hstack(mats, format='csr')
    return np.hstack(mats)


def _as_sparse_matrix(mat):
    if sparse.issparse(mat):
        return mat.astype(np.float64, copy=False).tocsr()
    return csr_array(np.array(mat).astype(np.float64))


def _sparse_zero(rows, cols):
    return csr_array((rows, cols), dtype=np.float64)


def _dense_vector(vec):
    if sparse.issparse(vec):
        vec = vec.toarray()
    return np.array(vec).astype(np.float64).flatten()


def _reshape_primal_space(space, rows):
    space = _as_sparse_matrix(space)
    if rows > 0:
        return space.reshape((rows, -1)).tocsr()
    cols = space.shape[1] if space.shape[0] == 0 else space.shape[0] * space.shape[1]
    return space.reshape((0, cols)).tocsr()


def collect_constraints(constraints: List[Tuple['ndarray', float, str]], dof: int,
        backend: Type[DualBackend] = DualBackend)\
        -> Tuple['ndarray', 'ndarray', 'ndarray', 'ndarray']:
    """
    Collect constraints and separate them into inequality and equality constraints.
    """
    as_matrix = backend.as_matrix
    as_vector = backend.as_vector

    ineq_lhs, ineq_rhs = [], []
    eq_lhs, eq_rhs = [], []
    for constraint, rhs, op in constraints:
        op = _STANDARDIZED_OPERATORS[op]
        if isinstance(rhs, (float, int)) or not hasattr(rhs, '__len__'):
            rhs = [rhs]

        constraint = as_matrix(constraint)
        if len(constraint.shape) == 1:
            constraint = constraint.reshape(1, dof)
        rhs = as_vector(rhs)

        if op == '__le__':
            constraint, rhs, op = -constraint, -rhs, '__ge__'
        if op == '__ge__':
            ineq_lhs.append(constraint)
            ineq_rhs.append(rhs)
        else: # if op == '__eq__':
            eq_lhs.append(constraint)
            eq_rhs.append(rhs)

    if len(ineq_lhs):
        ineq_lhs, ineq_rhs = _vstack(ineq_lhs), np.concatenate(ineq_rhs)
    else:
        ineq_lhs, ineq_rhs = as_matrix(np.zeros((0, dof))), np.zeros((0,))

    if len(eq_lhs):
        eq_lhs, eq_rhs = _vstack(eq_lhs), np.concatenate(eq_rhs)
    else:
        eq_lhs, eq_rhs = as_matrix(np.zeros((0, dof))), np.zeros((0,))
    return ineq_lhs, ineq_rhs, eq_lhs, eq_rhs


def create_numerical_dual_sdp(
    x0_and_space: Union[List[Tuple['Matrix', 'Matrix']], Dict[Any, Tuple['Matrix', 'Matrix']]],
    objective: 'ndarray',
    constraints: List[Tuple['ndarray', float, str]] = [],
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

    as_matrix = backend.as_matrix
    as_vector = backend.as_vector

    x0_and_space = [(as_vector(x0), as_matrix(space)) for x0, space in x0_and_space]
    x0_and_space = [(x0, space) for x0, space in x0_and_space if x0.shape[0] > 0]

    As = [space for x0, space in x0_and_space]
    bs = [x0 for x0, space in x0_and_space]

    c = as_vector(objective)

    ineq_lhs, ineq_rhs, eq_lhs, eq_rhs = collect_constraints(constraints, c.size, backend=backend)
    backend = backend(As, bs, ineq_lhs, ineq_rhs, eq_lhs, eq_rhs, c)
    return backend


def solve_numerical_dual_sdp(
    x0_and_space: Union[List[Tuple['Matrix', 'Matrix']], Dict[Any, Tuple['Matrix', 'Matrix']]],
    objective: 'ndarray',
    constraints: List[Tuple['ndarray', float, str]] = [],
    solver: Optional[str] = None,
    return_result: bool = False,
    verbose: Union[bool, int] = 0,
    max_iters: int = 200,
    time_limit: float = 1e10,
    tol_fsb_abs: float = 1e-8,
    tol_fsb_rel: float = 1e-8,
    tol_gap_abs: float = 1e-8,
    tol_gap_rel: float = 1e-8,
    solver_options: Dict[str, Any] = {},
) -> Optional[Union['ndarray', 'SDPResult']]:
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
        time_limit=time_limit,
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


def _fill_space(space: 'ndarray', n: int, bias: int) -> 'ndarray':
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


def _fill_space_sparse(n: int, bias: int, dof: int):
    i, j = np.triu_indices(n)
    cols = np.arange(bias, bias + n*(n+1)//2)
    rows = i*n + j
    data = np.ones((len(rows),), dtype=np.float64)
    offdiag = i != j
    if np.any(offdiag):
        rows = np.concatenate((rows, j[offdiag]*n + i[offdiag]))
        cols = np.concatenate((cols, cols[offdiag]))
        data = np.concatenate((data, np.ones((int(np.sum(offdiag)),), dtype=np.float64)))
    return csr_array((data, (rows, cols)), shape=(n**2, dof), dtype=np.float64)


def _extract_triu(space: 'ndarray', n: int) -> 'ndarray':
    """Assume space has shape m x N where N = n**2. Return a matrix of shape m * (n*(n+1)//2)
    where each column is the upper triangular part of the corresponding column of space."""
    i, j = np.triu_indices(n)
    if sparse.issparse(space):
        return space[:, i*n + j]
    return space.T.reshape(n, n, -1)[i, j, :].T


def _symmetry_column_weights(sizes):
    weights = []
    for n in sizes:
        block = np.full((n**2,), 2., dtype=np.float64)
        block[np.arange(0, n**2, n+1)] = 1.
        weights.append(block)
    if len(weights):
        return np.concatenate(weights)
    return np.zeros((0,), dtype=np.float64)


def _scale_columns(mat, weights):
    if mat.shape[1] == 0:
        return mat
    if sparse.issparse(mat):
        return mat.dot(diags_array(weights, format='csr')).tocsr()
    return mat * weights

def solve_numerical_primal_sdp(
    x0_and_space: Tuple['ndarray', Union[List['ndarray'], Dict[Any, 'ndarray']]],
    objective: 'ndarray',
    constraints: List[Tuple['ndarray', float, str]] = [],
    solver: Optional[str] = None,
    return_result: bool = False,
    verbose: Union[bool, int] = 0,
    max_iters: int = 200,
    time_limit: float = 1e10,
    tol_fsb_abs: float = 1e-8,
    tol_fsb_rel: float = 1e-8,
    tol_gap_abs: float = 1e-8,
    tol_gap_rel: float = 1e-8,
    solver_options: Dict[str, Any] = {},
) -> Optional['ndarray']:
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

    x0 = _dense_vector(x0)
    objective = _dense_vector(objective)

    if x0.size > 0:
        spaces = [_reshape_primal_space(space, x0.size) for space in spaces]
    else:
        spaces = [_reshape_primal_space(space, 0) for space in spaces]
    spaces = [space.copy() for space in spaces if space.shape[1] > 0]

    # Formulate the dual form (but not lagrangian dual) by creating
    # a dual SDP with sum(n*(n+1)//2) degrees of freedom.
    # Each entry of the vector represents the (i,j),(j,i) entries of a symmetric matrix.
    sizes = [int(round(np.sqrt(space.shape[1]))) for space in spaces]
    dof = sum(n*(n+1)//2 for n in sizes)

    ineq_lhs, ineq_rhs, eq_lhs, eq_rhs = collect_constraints(constraints, objective.size,
                                                             backend=_SparseInputBackend)
    As = [_fill_space_sparse(n, bias, dof) for n, bias in zip(sizes, np.cumsum([0] + [n*(n+1)//2 for n in sizes[:-1]]))]
    bs = [np.zeros((n**2,)) for n in sizes]

    def _extract_triu_multiple(mat):
        if dof == 0:
            return _sparse_zero(mat.shape[0], 0)
        parts = []
        bias, bias2 = 0, 0
        for n in sizes:
            space = _extract_triu(mat[:, bias2:bias2+n**2], n)
            parts.append(space)
            bias += n*(n+1)//2
            bias2 += n**2
        return _hstack(parts)


    # constraints at off-diagonals are doubled since only the upper triangular
    # contributes to the sum
    weights = _symmetry_column_weights(sizes)
    spaces = [_scale_columns(space, _symmetry_column_weights([n])) for space, n in zip(spaces, sizes)]
    objective_mat = _scale_columns(_as_sparse_matrix(objective.reshape(1, objective.size)), weights)
    ineq_lhs = _scale_columns(ineq_lhs, weights)
    eq_lhs = _scale_columns(eq_lhs, weights)

    c = _dense_vector(_extract_triu_multiple(objective_mat))
    eq_lhs = sparse.vstack([eq_lhs, _hstack(spaces) if len(spaces)
                            else _sparse_zero(x0.shape[0], 0)], format='csr')
    eq_rhs = np.concatenate([eq_rhs, x0])

    ineq_lhs = _extract_triu_multiple(ineq_lhs)
    eq_lhs = _extract_triu_multiple(eq_lhs)

    backend = backend(As, bs, ineq_lhs, ineq_rhs, eq_lhs, eq_rhs, c)
    result = backend.solve(
        verbose=verbose,
        max_iters=max_iters,
        time_limit=time_limit,
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
