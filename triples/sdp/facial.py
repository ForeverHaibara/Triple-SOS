"""Search for exact facial reduction certificates in sparse PSD subcones."""

from fractions import Fraction
from math import gcd
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple, cast

import numpy as np
from scipy import __version__ as _SCIPY_VERSION
from scipy.linalg import qr
from scipy.optimize import linprog
from sympy import MutableDenseMatrix as Matrix
from sympy.external.importtools import version_tuple

from .arithmetic import (
    ArithmeticTimeout, is_zz_qq_mat, matmul, rep_matrix_to_numpy, solve_undetermined_linear,
    sqrtsize_of_mat,
)

_HAS_HIGHS = tuple(version_tuple(_SCIPY_VERSION)) >= (1, 6)
_MAX_SUPPORT = 128
# Dense directions can also make the subsequent exact reduction expensive.
_MAX_SPECTRAL_COEFFICIENTS = 128_000


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


def _direction_columns(
    blocks: Dict[Any, Tuple[Matrix, Matrix]],
    sizes: Dict[Any, int],
    pairs: bool,
    remaining_time: Callable[[], Optional[float]],
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
        remaining_time()
    return np.concatenate(columns, axis=0).T, directions


def _verify_support(
    blocks: Dict[Any, Tuple[Matrix, Matrix]],
    sizes: Dict[Any, int],
    directions: Sequence[_Direction],
    remaining_time: Callable[[], Optional[float]],
) -> Dict[Any, Matrix]:
    """Reconstruct weights over QQ and check the entire affine identity."""
    forms = []
    for key, i, j, sign in directions:
        x0, space = blocks[key]
        n = sizes[key]
        offset, linear = x0[i*(n+1), :], space[i*(n+1), :]
        if sign:
            offset += x0[j*(n+1), :] + sign*(x0[i*n+j, :] + x0[j*n+i, :])
            linear += space[j*(n+1), :] + sign*(space[i*n+j, :] + space[j*n+i, :])
        forms.append(offset.row_join(linear))
        remaining_time()

    vectors: Dict[Any, List[Matrix]] = {}
    for index in _positive_support(forms, remaining_time):
        key, i, j, sign = directions[index]
        vector = Matrix.zeros(sizes[key], 1)
        vector[i] = 1
        if sign:
            vector[j] = sign
        vectors.setdefault(key, []).append(vector)
    return {key: Matrix.hstack(*basis) for key, basis in vectors.items()}


def _positive_support(
    forms: Sequence[Matrix],
    remaining_time: Callable[[], Optional[float]],
) -> List[int]:
    """Return the positive indices of a normalized QQ dependence, if verified."""
    equations = Matrix.vstack(*forms).T.col_join(Matrix.ones(1, len(forms)))
    rhs = Matrix.zeros(equations.rows, 1)
    rhs[-1] = 1
    try:
        weights, _ = solve_undetermined_linear(equations, rhs, time_limit=remaining_time)
    except ValueError:
        return []
    values = [row[0] for row in weights.tolist()]
    if any(weight.is_nonnegative is not True for weight in values) or equations * weights != rhs:
        return []
    return [i for i, weight in enumerate(values) if weight.is_positive]


def _candidate_supports(
    coefficients: np.ndarray,
    objective: np.ndarray,
    remaining_time: Callable[[], Optional[float]],
) -> Iterator[np.ndarray]:
    """Propose small supports with nonnegative weights using a numerical LP."""
    if not np.isfinite(coefficients).all():
        return
    # Positive column and row scaling preserves the possible supports.
    for axis in (0, 1):
        scale = np.max(np.abs(coefficients), axis=axis, keepdims=True)
        scale[scale == 0] = 1
        coefficients /= scale
    remaining = remaining_time()
    if remaining is not None and remaining <= 0:
        return
    budget = min(0.25, remaining) if remaining is not None else 0.25
    callback = None
    if _HAS_HIGHS:
        method = 'highs'
        options = {'time_limit': budget, 'primal_feasibility_tolerance': 1e-9}
    else:
        method = 'simplex'
        options = {'tol': 1e-9, 'maxiter': 1000}
        check_lp_time = ArithmeticTimeout.make_checker(budget)

        def callback(_):
            check_lp_time()

    width, count = coefficients.shape
    try:
        result = linprog(objective, A_eq=np.vstack([coefficients, np.ones(count)]),
            b_eq=np.r_[np.zeros(width), 1.], bounds=(0, None), method=method,
            options=options, callback=callback)
    except ArithmeticTimeout:
        remaining_time()
        return
    remaining_time()
    if result.success:
        support = np.flatnonzero(result.x > 1e-10)
        if 0 < len(support) <= _MAX_SUPPORT:
            yield support
        if not _HAS_HIGHS:
            # Simplex can leave small positive weights outside the true support.
            trimmed = support[result.x[support] > 1e-8]
            if 0 < len(trimmed) <= _MAX_SUPPORT and len(trimmed) != len(support):
                yield trimmed


def find_joint_exposing_nullspace(
    blocks: Dict[Any, Tuple[Matrix, Matrix]],
    remaining_time: Callable[[], Optional[float]],
) -> Dict[Any, Matrix]:
    """Find necessary nullspaces of rational symmetric affine PSD blocks.

    Use the nonnegative diagonal cone, followed by the diagonally dominant
    cone, to search for a nonzero exposing certificate. The linear program
    proposes a support. Positive rational weights must certify that the sum
    of the selected quadratic forms vanishes on the whole affine space.
    Since the forms are individually nonnegative at feasible points, each
    supported direction is a necessary kernel vector in its block.

    See Permenter and Parrilo, arXiv:1408.4685v3, Section 3.2.
    """
    sizes = {key: sqrtsize_of_mat(x0.rows) for key, (x0, _) in blocks.items() if x0.rows}
    if not sizes or not all(is_zz_qq_mat(m) for k in sizes for m in blocks[k]):
        return {}

    width = blocks[next(iter(sizes))][1].cols + 1
    for pairs in (False, True):
        count = sum(n*n if pairs else n for n in sizes.values())
        # Bound the dense numerical search and the exact support solve.
        if count > 2048 or count*width > 1_000_000:
            break
        remaining = remaining_time()
        if remaining is not None and remaining <= 0:
            break
        coefficients, directions = _direction_columns(blocks, sizes, pairs, remaining_time)
        for indices in _candidate_supports(coefficients, np.zeros(count), remaining_time):
            nullspace = _verify_support(blocks, sizes, [directions[i] for i in indices], remaining_time)
            if nullspace:
                return nullspace
    return {}


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


def _spectral_directions(
    matrix: np.ndarray,
    remaining_time: Callable[[], Optional[float]],
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
        remaining_time()
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


def find_spectral_exposing_nullspace(
    blocks: Dict[Any, Tuple[Matrix, Matrix]],
    y: np.ndarray,
    remaining_time: Callable[[], Optional[float]],
) -> Dict[Any, Matrix]:
    """Search for a joint exposure using rounded charts of near-zero eigenspaces.

    The numerical point only proposes directions. A verified nonnegative
    dependence must hold in the original QQ pencil, including its constant
    term, before any direction can restrict the feasible set.
    """
    if not np.isfinite(y).all():
        return {}
    # Bound work spent on dense coefficient arrays and chart enumeration.
    sizes = {k: sqrtsize_of_mat(a.rows) for k, (a, _) in blocks.items() if a.rows}
    sizes = {k: n for k, n in sizes.items() if n <= 24}
    if not sizes or not all(is_zz_qq_mat(m) for k in sizes for m in blocks[k]):
        return {}
    width = y.size + 1
    if sum(n*n for n in sizes.values()) * width > _MAX_SPECTRAL_COEFFICIENTS:
        return {}

    directions: List[Tuple[Any, Tuple[int, ...]]] = []
    costs: List[float] = []
    columns = []
    for key, n in sizes.items():
        remaining = remaining_time()
        if remaining is not None and remaining <= 0:
            return {}
        a, A = blocks[key]
        pencil = rep_matrix_to_numpy(a.row_join(A))
        if not np.isfinite(pencil).all():
            continue
        with np.errstate(over='ignore', invalid='ignore'):
            matrix = (pencil[:, 0] + pencil[:, 1:] @ y).reshape(n, n)
        for vector, residual in _spectral_directions(matrix, remaining_time):
            v = np.asarray(vector, dtype=float)
            with np.errstate(over='ignore', invalid='ignore'):
                columns.append(np.outer(v, v).reshape(-1) @ pencil)
            directions.append((key, vector))
            costs.append(residual)
    if not directions:
        return {}

    for indices in _candidate_supports(np.asarray(columns).T, np.asarray(costs), remaining_time):
        forms: List[Matrix] = []
        vectors: List[Tuple[Any, Matrix]] = []
        for index in indices:
            key, vector = directions[index]
            direction = Matrix(vector)
            a, A = blocks[key]
            weights = (direction * direction.T).reshape(1, a.rows)
            forms.append(cast(Matrix, matmul(weights, a.row_join(A), time_limit=remaining_time)))
            vectors.append((key, direction))
            remaining_time()
        nullspace: Dict[Any, List[Matrix]] = {}
        for index in _positive_support(forms, remaining_time):
            key, direction = vectors[index]
            nullspace.setdefault(key, []).append(direction)
        if nullspace:
            return {key: Matrix.hstack(*basis) for key, basis in nullspace.items()}
    return {}
