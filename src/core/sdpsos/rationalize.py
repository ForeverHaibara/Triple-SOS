from typing import List, Union, Optional, Callable

import numpy as np
import sympy as sp

from .manifold import LowRankHermitian
from ...utils.roots.rationalize import rationalize
from ...utils import congruence


def rationalize_with_mask(y: np.ndarray, zero_tolerance: float = 1e-7):
    """
    Rationalize a numpy vector. First set small entries to zero.

    Parameters
    ----------
    y : np.ndarray
        The vector to be rationalized.
    zero_tolerance : float
        Assume the largest (abs) entry is `v`. Then entries with value
        smaller than `zero_tolerance * v` will be set to zero.

    Returns
    -------
    y_rational : sp.Matrix
        The rationalized vector.
    """
    y_rational_mask = np.abs(y) > np.abs(y).max() * zero_tolerance
    y_rational = np.where(y_rational_mask, y, 0)
    y_rational = [rationalize(_, reliable = True) for _ in y_rational]
    y_rational = sp.Matrix(y_rational)
    return y_rational


def rationalize_simutaneously(y: np.ndarray, lcm: int, times = 3):
    """
    Rationalize a vector `y` with the same denominator `lcm ^ power` 
    where `power = 0, 1, ..., times - 1`. This keeps the denominators of
    `y` aligned.

    Parameters
    ----------
    y : np.ndarray
        The vector to be rationalized.
    lcm : int
        The denominator of `y`.
    times : int
        The number of times to perform retries.
    """
    lcm_ = sp.S(1)
    for power in range(times):
        y_rational = [round(v * lcm_) / lcm_ for v in y]
        y_rational = sp.Matrix(y_rational)
        yield y_rational

        lcm_ *= lcm


def verify_is_pretty(
        y: Union[List, sp.Matrix], 
        threshold: Optional[Callable] = None
    ):
    """
    Check whether the rationalization of `y` is pretty. Idea: in normal cases, 
    the denominators of `y` should be aligned. For example, 
    `[2/11, 56/33, 18/11, 2/3]` seems to be reasonable and great. However,
    `[2/3, 3/5, 4/7, 5/11]` is nonsense because the denominators are not aligned.

    We check whether the lcm of denominators of y exceeds certain threshold. If it
    exceeds, we return False. Otherwise, we return True.

    Parameters
    ----------
    y : Union[List, sp.Matrix]
        The vector to be checked.
    threshold : Optional[Callable]
        The threshold function. It should be a function of y and returns the 
        corresponding threshold. If None, we use `max(36, max(v.q for v in y)) ** 2`

    Returns
    -------
    bool
        Whether the rationalization of `y` is pretty.
    """
    lcm = 1
    if threshold is None:
        s = max(36, max(v.q for v in y)) ** 2
    else:
        s = threshold(y)
    for v in y:
        lcm = sp.lcm(lcm, v.q)
        if lcm > s:
            return False
    return True


def verify_is_positive(
        vecS: sp.Matrix,
        splits: List[slice]
    ):
    """
    Recover symmetric matrices from `x0 + space * y` and check whether they are
    positive semidefinite. See more details in `solver.py`.

    Parameters
    ----------
    vecS:
        The vectorized symmetric matrices.
    splits : List[slice]
        The splits of the space. Each split is a slice object.

    Returns
    -------
    decompositions : Optional[List[Tuple[sp.Matrix, sp.Matrix List[sp.Rational]]]]
        If the matrices are positive semidefinite, return the congruence decompositions `(S, U, diag)`
        So that each `S = U.T * diag(diag) * U` where `U` is upper triangular.
        Otherwise, return None.
    """
    decompositions = []
    for split in splits:
        S = LowRankHermitian(None, sp.Matrix(vecS[split])).S
        U, diag = congruence(S)
        if any(_ < 0 for _ in diag):
            return None
        decompositions.append((S, U, diag))
    return decompositions


