"""Search for exact facial reduction certificates in sparse PSD subcones."""

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from scipy import __version__ as _SCIPY_VERSION
from scipy.optimize import linprog
from sympy import MutableDenseMatrix as Matrix
from sympy.external.importtools import version_tuple

from .arithmetic import (
    ArithmeticTimeout, is_zz_qq_mat, rep_matrix_to_numpy, solve_undetermined_linear,
    sqrtsize_of_mat,
)

_HAS_HIGHS = tuple(version_tuple(_SCIPY_VERSION)) >= (1, 6)


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

    equations = Matrix.vstack(*forms).T.col_join(Matrix.ones(1, len(forms)))
    rhs = Matrix.zeros(equations.rows, 1)
    rhs[-1] = 1
    try:
        weights, _ = solve_undetermined_linear(equations, rhs, time_limit=remaining_time)
    except ValueError:
        return {}
    values = [row[0] for row in weights.tolist()]
    if any(weight.is_nonnegative is not True for weight in values) or equations * weights != rhs:
        return {}

    vectors: Dict[Any, List[Matrix]] = {}
    for (key, i, j, sign), weight in zip(directions, values):
        if weight.is_positive:
            vector = Matrix.zeros(sizes[key], 1)
            vector[i] = 1
            if sign:
                vector[j] = sign
            vectors.setdefault(key, []).append(vector)
    return {key: Matrix.hstack(*basis) for key, basis in vectors.items()}


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
        if not np.isfinite(coefficients).all():
            continue
        # Positive column and row scaling preserves the support search.
        for axis in (0, 1):
            scale = np.max(np.abs(coefficients), axis=axis, keepdims=True)
            scale[scale == 0] = 1
            coefficients /= scale
        remaining = remaining_time()
        if remaining is not None and remaining <= 0:
            break
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

        try:
            result = linprog(np.zeros(count),
                A_eq=np.vstack([coefficients, np.ones(count)]),
                b_eq=np.r_[np.zeros(width), 1.], bounds=(0, None), method=method,
                options=options, callback=callback)
        except ArithmeticTimeout:
            remaining_time()
            break
        remaining_time()
        if result.success:
            support = np.flatnonzero(result.x > 1e-10)
            supports = [support]
            if not _HAS_HIGHS:
                # Simplex can leave small positive weights outside the true support.
                trimmed = support[result.x[support] > 1e-8]
                if len(trimmed) != len(support):
                    supports.append(trimmed)
            for indices in supports:
                if not 0 < len(indices) <= 128:
                    continue
                nullspace = _verify_support(blocks, sizes, [directions[i] for i in indices], remaining_time)
                if nullspace:
                    return nullspace
    return {}
