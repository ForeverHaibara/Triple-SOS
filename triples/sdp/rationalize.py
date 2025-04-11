from abc import ABC, abstractmethod
from itertools import chain
from typing import Union, Optional, Tuple, List, Dict, Callable, Generator

import numpy as np
import sympy as sp

from .arithmetic import congruence_with_perturbation

Decomp = Dict[str, Tuple[sp.Matrix, sp.Matrix, List[sp.Rational]]]


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


class Rationalizer(ABC):
    @abstractmethod
    def __call__(self, y: np.ndarray) -> Generator[sp.Matrix, None, None]:
        raise NotImplementedError
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

class IdentityRationalizer(Rationalizer):
    def __call__(self, y: np.ndarray) -> Generator[sp.Matrix, None, None]:
        return (sp.Matrix(y),)

class EmptyRationalizer(Rationalizer):
    def __call__(self, y: np.ndarray) -> Generator[sp.Matrix, None, None]:
        return (sp.zeros(0, 1),)

class RationalizeWithMask(Rationalizer):
    """
    Rationalize a numpy vector by first setting entries with small values to zero.

    Parameters
    ----------
    zero_tolerance : float
        Assume the largest (abs) entry is `v`. Then entries with value
        smaller than `zero_tolerance * v` will be set to zero.
    """
    def __init__(self, zero_tolerance: float = 1e-7):
        self.zero_tolerance = zero_tolerance

    def __call__(self, y: np.ndarray) -> Generator[sp.Matrix, None, None]:
        tol = max(1, np.abs(y).max()) * self.zero_tolerance
        y_rational_mask = np.abs(y) > tol
        y_rational = np.where(y_rational_mask, y, 0).flatten().tolist()
        y_rational = [rationalize(v, rounding = abs(v) * 1e-4) for v in y_rational]
        y_rational = sp.Matrix(y_rational)
        return (y_rational,)


class RationalizeSimultaneously(Rationalizer):
    """
    Rationalize a vector `y` with the same denominator `lcm`.

    Parameters
    ----------
    lcms : List[int]
        The list of denominators.
    """
    def __init__(self, lcms: List[int] = (1, 1260, 1260**2, 1260**3)):
        self.lcms = lcms

    def __call__(self, y: np.ndarray) -> Generator[sp.Matrix, None, None]:
        for lcm in self.lcms:
            lcm = sp.Integer(lcm)
            y_rational = [round(v * lcm) / lcm for v in y]
            y_rational = sp.Matrix(y_rational)
            yield y_rational


def verify_is_pretty(
        y: Union[List, sp.Matrix], 
        threshold: Optional[Callable] = None
    ) -> bool:
    """
    A heuristic method to check whether the rationalization of `y` is pretty.
    Idea: in normal cases, the denominators of `y` should be aligned. For example, 
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
    if len(y) == 0: return True
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
        mat_func: Callable[[sp.Matrix], Dict[str, sp.Matrix]],
        projection: Optional[Callable[[sp.Matrix], sp.Matrix]] = None,
        rationalizers: List[Rationalizer] = [],
        reg: float = 0,
        perturb: bool = False,
        check_pretty: bool = True,
    ) -> Optional[Tuple[sp.Matrix, Decomp]]:
    """
    Recover symmetric matrices from `x0 + space * y` and check whether they are
    positive semidefinite.

    Parameters
    ----------
    y : np.ndarray
        The vector to be rationalized.
    mat_func : Callable[[sp.Matrix], Dict[str, sp.Matrix]]
        Given a rationalized vector `y`, return a dictionary of matrices
        that needs to be PSD.
    projection : Optional[Callable[[sp.Matrix], sp.Matrix]]
        The projection function. If not None, we project `y` to the feasible
        region before checking the PSD property.
    rationalizers : List[Rationalizer]
        The list of rationalizers. We will try to rationalize `y` with each
        rationalizer in the list.
    reg : float
        We require `S[i]` to be positive semidefinite, but in practice
        we might allow a small regularization term to make it
        positive definite >> reg * I.
    perturb : bool
        If perturb == True, it must return the result by adding a small
        perturbation * identity to the matrices.
    check_pretty : bool
        If True, we check whether the rationalization of `y` is pretty.
        See `verify_is_pretty` for more details.

    Returns
    -------
    y, decompositions : Optional[Tuple[sp.Matrix, Dict[str, Tuple[sp.Matrix, sp.Matrix, List[sp.Rational]]]]]
        If the matrices are positive semidefinite, return the congruence decompositions `y, [(S, U, diag)]`
        So that each `S = U.T * diag(diag) * U` where `U` is upper triangular.
        Otherwise, return None.
    """
    if isinstance(y, sp.MatrixBase):
        rationalizers = [IdentityRationalizer()]
    if len(y) == 0:
        rationalizers = [EmptyRationalizer()]
    if isinstance(y, np.ndarray):
        y = y.flatten().tolist()

    for rationalizer in rationalizers:
        # print(rationalizer, rationalizer(y))
        for y_rational in rationalizer(y):
            if check_pretty and not verify_is_pretty(y_rational):
                continue
            if projection is not None:
                y_rational = projection(y_rational)

            S_dict = mat_func(y_rational)
            decompositions = {}
            for key, S in S_dict.items():
                if reg != 0:
                    S = S + reg * sp.eye(S.shape[0])

                congruence_decomp = congruence_with_perturbation(S, perturb = perturb)
                if congruence_decomp is None:
                    break

                U, diag = congruence_decomp
                decompositions[key] = (S, U, diag)
            else:
                return y_rational, decompositions