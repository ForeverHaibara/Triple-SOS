from itertools import chain
from typing import List, Union, Optional, Callable, Tuple

import numpy as np
import sympy as sp

from .utils import congruence_with_perturbation, S_from_y


def rationalize(x, rounding = 1e-2, **kwargs):
    """
    Although there is a rationalize function in triples.src.roots,
    we implement a new one here to reduce the dependency. The
    function is a wrapper of sympy.nsimplify.

    Parameters
    ----------
    x : sp.Rational | sp.Float
        The number to be rationalized.
    rounding : float
        The rounding threshold.
    """
    rounding = max(rounding, 1e-15)
    return sp.nsimplify(x, tolerance = rounding, rational = True)


def rationalize_with_mask(y: np.ndarray, zero_tolerance: float = 1e-7) -> sp.Matrix:
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
    y_rational = [rationalize(v, rounding = abs(v) * 1e-4) for v in y_rational]
    y_rational = sp.Matrix(y_rational)
    return y_rational


def rationalize_simultaneously(y: np.ndarray, lcm: int = 1260, times: int = 3) -> sp.Matrix:
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

    Yields
    ------
    y_rational : sp.Matrix
        The rationalized vector.
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
    ) -> bool:
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


def rationalize_and_decompose(
        y: Union[np.ndarray, sp.Matrix],
        x0: sp.Matrix,
        space: sp.Matrix,
        splits: List[slice],
        try_rationalize_with_mask: bool = True,
        lcm: int = 1260,
        times: int = 3,
        reg: float = 0,
        perturb: bool = False,
        check_pretty: bool = True,
    ):
    """
    Recover symmetric matrices from `x0 + space * y` and check whether they are
    positive semidefinite.

    Parameters
    ----------
    y : np.ndarray
        The vector to be rationalized.
    x0 : sp.Matrix
        The constant part of the equation. Stands for a particular solution
        of the space of S.
    space : sp.Matrix
        The space of S. Stands for the space constraint of S.
    splits : List[slice]
        The splits of the symmetric matrices. Each split is a slice object.
    try_rationalize_with_mask: bool
        If True, function `rationalize_with_mask` will be called first.
    lcm: int
        The denominator used to rationalize `y`. Defaults to 1260.
    times : int
        The number of times to perform retries. Defaults to 3.
    reg : float
        We require `S[i]` to be positive semidefinite, but in practice
        we might want to add a small regularization term to make it
        positive definite >> reg * I.
    perturb : bool
        If perturb == True, it must return the result by adding a small
        perturbation * identity to the matrices.
    check_pretty : bool
        If True, we check whether the rationalization of `y` is pretty.
        See `verify_is_pretty` for more details.

    Returns
    -------
    y, decompositions : Optional[Tuple[sp.Matrix, List[Tuple[sp.Matrix, sp.Matrix, List[sp.Rational]]]]]
        If the matrices are positive semidefinite, return the congruence decompositions `y, [(S, U, diag)]`
        So that each `S = U.T * diag(diag) * U` where `U` is upper triangular.
        Otherwise, return None.
    """
    if isinstance(y, np.ndarray):
        ys = rationalize_simultaneously(y, lcm = lcm, times = times)
        if try_rationalize_with_mask:
            ys = chain([rationalize_with_mask(y)], ys)

    elif isinstance(y, sp.Matrix):
        ys = [y]


    for y_rational in ys:
        if check_pretty and not verify_is_pretty(y_rational):
            continue

        Ss = S_from_y(y_rational, x0, space, splits)
        decompositions = []
        for S in Ss:
            if reg != 0:
                S = S + reg * sp.eye(S.shape[0])

            congruence_decomp = congruence_with_perturbation(S, perturb = perturb)
            if congruence_decomp is None:
                break

            U, diag = congruence_decomp
            decompositions.append((S, U, diag))
        else:
            return y_rational, decompositions