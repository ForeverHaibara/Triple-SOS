"""Search for exact exposing certificates in PSD subcones."""

import warnings
from fractions import Fraction
from itertools import chain
from math import gcd
from typing import (
    Any, Dict, Iterator, List, Mapping,
    NamedTuple, Optional, Sequence, Tuple, cast, TYPE_CHECKING
)

import numpy as np
from scipy import __version__ as _SCIPY_VERSION
from scipy.linalg import qr
from sympy import MutableDenseMatrix as Matrix
from sympy.external.importtools import version_tuple

from .arithmetic import (
    ArithmeticTimeout, is_zz_qq_mat, matmul, rep_matrix_to_numpy,
    solve_undetermined_linear, sqrtsize_of_mat,
)

if TYPE_CHECKING:
    from .arithmetic import TimeChecker, TimeLimit

_HAS_HIGHS = tuple(version_tuple(_SCIPY_VERSION)) >= (1, 6)
_MAX_SUPPORT = 128
# Bound the dense coefficient arrays for searches using general directions.
_MAX_SPECTRAL_COEFFICIENTS = 128_000


class _RankOneTerm(NamedTuple):
    key: Any
    vector: Matrix
    form: Matrix  # Exact affine coefficients of vector.T * S_key(y) * vector.


class _Direction(NamedTuple):
    """
    The direction `e_i` or `e_i + sign * e_j` in one PSD block.

    Used for testing whether there exists `w >= 0` so that
    `sum (w_k * v_k^TSv_k) == 0` by LP for `v` in given directions.
    """

    key: Any
    i: int
    j: int
    sign: int


def _rank_one_term(
    key: Any, vector: Matrix, x0: Matrix, space: Matrix,
    *,
    check_time: 'TimeChecker',
) -> _RankOneTerm:
    row = (vector * vector.T).reshape(1, x0.rows)
    offset = cast(Matrix, matmul(row, x0, time_limit=check_time))
    linear = cast(Matrix, matmul(row, space, time_limit=check_time))
    return _RankOneTerm(key, vector, offset.row_join(linear))


def _round_vector(vector: np.ndarray, bound: int) -> Tuple[int, ...]:
    """Round a chart vector and clear denominators, up to sign and scale."""
    fractions = [Fraction(float(v)).limit_denominator(bound) for v in vector]
    common = 1
    for f in fractions:
        common = common * f.denominator // gcd(common, f.denominator)
    integers = [f.numerator * (common // f.denominator) for f in fractions]
    divisor = 0
    for i in integers:
        divisor = gcd(divisor, i)
    if not divisor:
        return ()
    if next(i for i in integers if i) < 0:
        divisor = -divisor
    return tuple(i // divisor for i in integers)


def _rounded_positive_eigendirections(
    matrix: np.ndarray,
    *,
    bounds: Sequence[int],
    eigenvalue_rtol: float = 0.,
    largest: Optional[int] = None,
    check_time: 'TimeChecker',
) -> Iterator[Tuple[int, ...]]:
    """Yield distinct rational directions from the positive eigenspace."""
    try:
        eigenvalues, vectors = np.linalg.eigh(matrix)
    except np.linalg.LinAlgError:
        return
    indices = np.flatnonzero(eigenvalues > max(eigenvalues[-1]*eigenvalue_rtol, 0.))
    if largest is not None:
        indices = indices[::-1][:largest]
    seen = set()
    for index in indices:
        vector = vectors[:, index]
        vector = vector / vector[np.argmax(np.abs(vector))]
        for bound in bounds:
            remaining = check_time.remaining()
            if remaining is not None and remaining <= 0:
                return
            direction = _round_vector(vector, bound)
            if direction and direction not in seen:
                seen.add(direction)
                yield direction


def _nonnegative_weights(
    equations: Matrix,
    rhs: Matrix,
    *,
    check_time: 'TimeChecker',
) -> Optional[Matrix]:
    """Reconstruct nonnegative weights and check every equation over QQ."""
    try:
        weights, _ = solve_undetermined_linear(equations, rhs, time_limit=check_time)
    except ValueError:
        return None
    if (any(w.is_nonnegative is not True for (w,) in weights.tolist())
            or matmul(equations, weights, time_limit=check_time) != rhs):
        return None
    return weights


def _lp_supports(
    coefficients: np.ndarray,
    objective: np.ndarray,
    *,
    check_time: 'TimeChecker',
    objective_row: Optional[int] = None,
) -> Iterator[np.ndarray]:
    """Propose nonnegative supports with zero pairing on the equality rows.

    If supplied, objective_row replaces the given cost with the indicated
    coefficient row and requires a negative pairing on that row.
    """
    if not np.isfinite(coefficients).all():
        return
    scale = np.max(np.abs(coefficients), axis=0)
    coefficients /= np.where(scale == 0, 1, scale)
    if objective_row is not None:
        objective = coefficients[objective_row]
        coefficients = np.delete(coefficients, objective_row, axis=0)
    scale = np.max(np.abs(coefficients), axis=1, keepdims=True)
    coefficients /= np.where(scale == 0, 1, scale)

    from scipy.optimize import linprog

    remaining = check_time.remaining()
    if remaining is not None and remaining <= 0:
        return
    check_lp_time = check_time.with_limit(0.25)
    callback = None
    if _HAS_HIGHS:
        method = 'highs'
        options = {'primal_feasibility_tolerance': 1e-9}
    else:
        method = 'simplex'
        options = {'tol': 1e-9, 'maxiter': 1000}

        def callback(_):
            check_lp_time()

    width, count = coefficients.shape
    try:
        if _HAS_HIGHS:
            options['time_limit'] = check_lp_time.remaining()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('once')
            result = linprog(objective, A_eq=np.vstack([coefficients, np.ones(count)]),
                b_eq=np.r_[np.zeros(width), 1.], bounds=(0, None), method=method,
                options=options, callback=callback)
    except ArithmeticTimeout:
        check_time()
        return
    check_time()
    if result.success and (objective_row is None or result.fun < -1e-9):
        support = np.flatnonzero(result.x > 1e-10)
        if 0 < len(support) <= _MAX_SUPPORT:
            yield support
        if not _HAS_HIGHS:
            # Simplex can leave small positive weights outside the true support.
            trimmed = support[result.x[support] > 1e-8]
            if 0 < len(trimmed) <= _MAX_SUPPORT and len(trimmed) != len(support):
                yield trimmed


def find_exposing_nullspace(
    blocks: Dict[Any, Tuple[Matrix, Matrix]],
    *,
    y: Optional[np.ndarray] = None,
    dual: Optional[Mapping[Any, Optional[np.ndarray]]] = None,
    directions: Optional[Sequence[Tuple[Any, Matrix]]] = None,
    coordinate_level: int = 2,
    time_limit: 'TimeLimit' = None,
) -> Dict[Any, Matrix]:
    """Return the first exact exposure from dual, joint or spectral supports.

    Supplied directions are tried before the coordinate search.
    The coordinate level is independent: 0 disables it, 1 tries the
    nonnegative diagonal cone, and 2 also tries the diagonally dominant cone.
    """
    if coordinate_level not in (0, 1, 2):
        raise ValueError("coordinate_level must be 0, 1 or 2")
    check_time = ArithmeticTimeout.make_checker(time_limit)
    supports = chain(
        _dual_supports(blocks, dual, check_time=check_time),
        _supplied_direction_supports(blocks, directions, check_time=check_time),
        _dual_supports(blocks, dual, joint=True, check_time=check_time),
        _coordinate_supports(blocks, level=coordinate_level, check_time=check_time),
        _spectral_supports(blocks, y, check_time=check_time),
    )
    return _expose_supports(supports, check_time=check_time)


def _dual_supports(
    blocks: Dict[Any, Tuple[Matrix, Matrix]],
    dual: Optional[Mapping[Any, Optional[np.ndarray]]],
    *,
    joint: bool = False,
    check_time: 'TimeChecker',
) -> Iterator[List[_RankOneTerm]]:
    """Try rounded dual eigenvectors individually or in a joint support."""
    if not dual:
        return
    if not joint and not all(is_zz_qq_mat(m) for pair in blocks.values() for m in pair):
        return

    directions = []
    for key, z in dual.items():
        remaining = check_time.remaining()
        if remaining is not None and remaining <= 0:
            return
        if key not in blocks or z is None:
            continue
        x0, space = blocks[key]
        n = sqrtsize_of_mat(x0.rows)
        if n == 0 or (joint and n > _MAX_SUPPORT) or z.shape != (n, n) or not np.isfinite(z).all():
            continue
        for vector in _rounded_positive_eigendirections(
            z, bounds=(1, 4, 16, 64), largest=4 if joint else 2,
            eigenvalue_rtol=1e-4 if joint else 0., check_time=check_time,
        ):
            if not joint:
                yield [_rank_one_term(key, Matrix(vector), x0, space, check_time=check_time)]
            elif max(map(abs, vector)) <= 512:
                directions.append((key, Matrix(vector)))
        if len(directions) > _MAX_SUPPORT:
            return
    if len(directions) > 1:
        yield from _supplied_direction_supports(blocks, directions, check_time=check_time)


def _coordinate_term(
    blocks: Dict[Any, Tuple[Matrix, Matrix]],
    sizes: Dict[Any, int],
    direction: _Direction,
) -> _RankOneTerm:
    """Form exact coefficients by indexing the few nonzero entries of each atom."""
    key, i, j, sign = direction
    x0, space = blocks[key]
    n = sizes[key]
    offset, linear = x0[i*(n+1), :], space[i*(n+1), :]
    vector = Matrix.zeros(n, 1)
    vector[i] = 1
    if sign:
        offset += x0[j*(n+1), :] + sign*(x0[i*n+j, :] + x0[j*n+i, :])
        linear += space[j*(n+1), :] + sign*(space[i*n+j, :] + space[j*n+i, :])
        vector[j] = sign
    return _RankOneTerm(key, vector, offset.row_join(linear))


def _coordinate_columns(
    blocks: Dict[Any, Tuple[Matrix, Matrix]],
    sizes: Dict[Any, int],
    pairs: bool,
    *,
    check_time: 'TimeChecker',
) -> Tuple[np.ndarray, List[_Direction]]:
    """
    Return floating-point coefficients of the rank-one test forms.
    """
    columns = []
    directions: List[_Direction] = []
    for key, n in sizes.items():
        x0, space = blocks[key]
        diagonal = rep_matrix_to_numpy(x0[::n+1, :].row_join(space[::n+1, :]))
        columns.append(diagonal)
        directions.extend(_Direction(key, i, i, 0) for i in range(n))
        if pairs and n > 1:
            i, j = np.triu_indices(n, 1)
            upper, lower = (i*n+j).tolist(), (j*n+i).tolist()
            offsets = x0.extract(upper, [0]) + x0.extract(lower, [0])
            linear = space.extract(upper, range(space.cols)) + space.extract(lower, range(space.cols))
            cross = rep_matrix_to_numpy(offsets.row_join(linear))
            diagonal_sum = diagonal[i] + diagonal[j]
            for sign in (1, -1):
                columns.append(diagonal_sum + sign*cross)
                directions.extend(_Direction(key, int(a), int(b), sign) for a, b in zip(i, j))
        check_time()
    return np.concatenate(columns, axis=0).T, directions


def _coordinate_supports(
    blocks: Dict[Any, Tuple[Matrix, Matrix]],
    *,
    level: int,
    check_time: 'TimeChecker',
) -> Iterator[List[_RankOneTerm]]:
    """Propose supports from the diagonal cone and diagonally dominant cone.

    See Permenter and Parrilo, arXiv:1408.4685v3, Section 3.2.
    """
    if level == 0:
        return
    sizes = {key: sqrtsize_of_mat(x0.rows) for key, (x0, _) in blocks.items() if x0.rows}
    if not sizes or not all(is_zz_qq_mat(m) for k in sizes for m in blocks[k]):
        return

    width = blocks[next(iter(sizes))][1].cols + 1

    for pairs in ((False, True) if level == 2 else (False,)):
        count = sum(n*n if pairs else n for n in sizes.values())
        # Bound the dense numerical search and the exact support solve.
        if count > 2048 or count*width > 1_000_000:
            break
        remaining = check_time.remaining()
        if remaining is not None and remaining <= 0:
            break
        coefficients, directions = _coordinate_columns(blocks, sizes, pairs, check_time=check_time)
        for indices in _lp_supports(coefficients, np.zeros(count), check_time=check_time):
            yield [_coordinate_term(blocks, sizes, directions[i]) for i in indices]


def _spectral_directions(
    matrix: np.ndarray,
    *,
    check_time: 'TimeChecker',
) -> List[Tuple[Tuple[int, ...], float]]:
    """Choose a few short directions near the numerical nullspace of a block."""
    if not np.isfinite(matrix).all():
        return []
    if matrix.shape == (1, 1):
        # A scalar has no spectral gap. It can still be required to balance
        # an affine form from another block in the joint certificate.
        return [((1,), float(matrix[0, 0] != 0))]
    try:
        eigs, vectors = np.linalg.eigh(matrix)
    except np.linalg.LinAlgError:
        return []
    n = len(eigs)
    order = np.argsort(np.abs(eigs))
    scale = max(np.abs(eigs).max(), 1e-30)
    near_zero = np.count_nonzero(np.abs(eigs) <= 0.015 * scale)
    if not near_zero:
        return []
    matrix = matrix / scale
    candidates: Dict[Tuple[int, ...], float] = {}

    def add(direction: Tuple[int, ...]) -> None:
        if not direction or direction in candidates or max(map(abs, direction)) > 2048:
            return
        v = np.asarray(direction, dtype=float)
        residual = float(np.linalg.norm(matrix @ v) / np.linalg.norm(v))
        if residual < 0.012:
            candidates[direction] = residual

    # Diagonal atoms can complement dense directions in the same certificate.
    for i in range(n):
        add(tuple(int(j == i) for j in range(n)))
    for rank in range(1, min(n, near_zero + 1)):
        check_time()
        W = vectors[:, order[:rank]]
        _, _, pivots = qr(W.T, mode='economic', pivoting=True)
        try:
            chart = np.linalg.solve(W[pivots[:rank]].T, W.T)
        except np.linalg.LinAlgError:
            continue
        for v in chart:
            if np.isfinite(v).all():
                for bound in (1, 2, 4, 8, 16, 32, 64):
                    add(_round_vector(v, bound))
    return sorted(candidates.items(), key=lambda pair: pair[1])[:32]


def _spectral_supports(
    blocks: Dict[Any, Tuple[Matrix, Matrix]],
    y: Optional[np.ndarray],
    *,
    check_time: 'TimeChecker',
) -> Iterator[List[_RankOneTerm]]:
    """Propose supports using rounded charts of the near-kernels at y."""
    if y is None or not np.isfinite(y).all():
        return
    # Bound work spent on dense coefficient arrays and chart enumeration.
    sizes = {k: sqrtsize_of_mat(a.rows) for k, (a, _) in blocks.items() if a.rows}
    sizes = {k: n for k, n in sizes.items() if n <= 24}
    if not sizes or not all(is_zz_qq_mat(m) for k in sizes for m in blocks[k]):
        return
    width = y.size + 1
    if sum(n*n for n in sizes.values()) * width > _MAX_SPECTRAL_COEFFICIENTS:
        return

    directions = []
    pencils = {}
    for key, n in sizes.items():
        remaining = check_time.remaining()
        if remaining is not None and remaining <= 0:
            return
        a, A = blocks[key]
        pencil = rep_matrix_to_numpy(a.row_join(A))
        if not np.isfinite(pencil).all():
            continue
        with np.errstate(over='ignore', invalid='ignore'):
            matrix = (pencil[:, 0] + pencil[:, 1:] @ y).reshape(n, n)
        for vector, residual in _spectral_directions(matrix, check_time=check_time):
            directions.append((key, Matrix(vector), residual))
        pencils[key] = pencil
    yield from _direction_supports(blocks, directions, pencils, check_time=check_time)


def _supplied_direction_supports(
    blocks: Dict[Any, Tuple[Matrix, Matrix]],
    directions: Optional[Sequence[Tuple[Any, Matrix]]],
    *,
    check_time: 'TimeChecker',
) -> Iterator[List[_RankOneTerm]]:
    """Propose supports using only the supplied rational directions."""
    if not directions or len(directions) > _MAX_SUPPORT:
        return
    keys = dict.fromkeys(k for k, _ in directions)
    if any(k not in blocks or v.shape != (sqrtsize_of_mat(blocks[k][0].rows), 1)
           or not is_zz_qq_mat(v) for k, v in directions):
        return
    if (sum(blocks[k][0].rows * (1 + blocks[k][1].cols) for k in keys)
            > _MAX_SPECTRAL_COEFFICIENTS):
        return
    if not all(is_zz_qq_mat(m) for k in keys for m in blocks[k]):
        return
    pencils = {}
    for k in keys:
        check_time()
        a, A = blocks[k]
        pencils[k] = rep_matrix_to_numpy(a.row_join(A))
    yield from _direction_supports(
        blocks, [(k, v, 0.) for k, v in directions], pencils, check_time=check_time,
    )


def _positive_support(
    forms: Sequence[Matrix],
    *,
    check_time: 'TimeChecker',
) -> List[int]:
    """Return the positive indices of a normalized QQ dependence, if verified."""
    if len(forms) == 1:
        return [0] if forms[0].is_zero_matrix else []
    equations = Matrix.vstack(*forms).T.col_join(Matrix.ones(1, len(forms)))
    rhs = Matrix.zeros(equations.rows, 1)
    rhs[-1] = 1
    weights = _nonnegative_weights(equations, rhs, check_time=check_time)
    return [] if weights is None else [i for i, (w,) in enumerate(weights.tolist()) if w.is_positive]


def _expose_supports(
    supports: Iterator[List[_RankOneTerm]],
    *,
    check_time: 'TimeChecker',
) -> Dict[Any, Matrix]:
    """Verify zero pairing before imposing the corresponding kernel vectors.

    At a feasible point each form is nonnegative. A zero sum with positive
    weights therefore forces every positively weighted direction into the kernel.
    """
    for terms in supports:
        nullspace: Dict[Any, List[Matrix]] = {}
        for index in _positive_support([term.form for term in terms],
                                       check_time=check_time):
            key, vector, _ = terms[index]
            nullspace.setdefault(key, []).append(vector)
        if nullspace:
            return {key: Matrix.hstack(*basis) for key, basis in nullspace.items()}
    return {}


def _direction_supports(
    blocks: Dict[Any, Tuple[Matrix, Matrix]],
    directions: Sequence[Tuple[Any, Matrix, float]],
    pencils: Dict[Any, np.ndarray],
    *,
    check_time: 'TimeChecker',
    objective_row: Optional[int] = None,
) -> Iterator[List[_RankOneTerm]]:
    """Propose atom supports numerically and form their exact affine coefficients."""
    if not directions:
        return
    columns, costs = [], []
    for key, vector, cost in directions:
        v = rep_matrix_to_numpy(vector).ravel()
        with np.errstate(over='ignore', invalid='ignore'):
            columns.append(np.outer(v, v).reshape(-1) @ pencils[key])
        costs.append(cost)
        check_time()

    for indices in _lp_supports(
        np.asarray(columns).T, np.asarray(costs), check_time=check_time,
        objective_row=objective_row,
    ):
        yield [
            _rank_one_term(key, vector, *blocks[key], check_time=check_time)
            for key, vector, _ in (directions[i] for i in indices)
        ]
