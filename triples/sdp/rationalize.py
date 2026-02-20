from abc import ABC, abstractmethod
from itertools import chain
from typing import Union, Optional, Tuple, List, Dict, Callable, Generator, Any

import numpy as np
from sympy import Rational, MatrixBase, nsimplify
from sympy import MutableDenseMatrix as Matrix
try:
    from sympy.polys.matrices.exceptions import DMError
except ImportError:
    class DMError(Exception): ...

from .arithmetic import (
    ArithmeticTimeout, matadd, matmul, matmul_multiple, solve_undetermined_linear,
    sqrtsize_of_mat, congruence, rep_matrix_from_numpy, rep_matrix_to_numpy, lll
)
from .backends import SDPError

Decomp = Dict[str, Tuple[Matrix, Matrix, List[Rational]]]


class SDPRationalizeError(SDPError):
    @classmethod
    def from_sdp_error(cls, error: SDPError) -> "SDPRationalizeError":
        return cls(error.result)

def rationalize(x, rounding = 1e-2, **kwargs):
    """
    Although there is a rationalize function in triples.src.roots,
    we implement a new one here to reduce the dependency. The
    function is a wrapper of sympy.nsimplify.

    Parameters
    ----------
    x : Rational | Float
        The number to be rationalized.
    rounding : float
        The rounding threshold.
    """
    rounding = max(rounding, 1e-15)
    return nsimplify(x, tolerance = rounding, rational = True)


class DualRationalizer:
    LLL_WEIGHT = 10**19
    LLL_TRUNC  = 2000 # TODO
    LLL_EIG    = 1e-2 # do not call LLL if block eig > LLL_EIG
    ROUND_DENOMS = (1, 12, 240, 2520, 2304, 2970, 210600, 1260**3)

    def __init__(self, sdp):
        self._sdp = sdp
        self._x0_and_space_numer = None

    @property
    def x0_and_space(self) -> Dict[Any, Tuple[Matrix, Matrix]]:
        return self._sdp._x0_and_space

    @property
    def x0_and_space_numer(self) -> Dict[Any, Tuple[np.ndarray, np.ndarray]]:
        if self._x0_and_space_numer is None:
            self._x0_and_space_numer = {k:
                (rep_matrix_to_numpy(v[0]).flatten(), rep_matrix_to_numpy(v[1]))
                for k, v in self.x0_and_space.items()
            }
        return self._x0_and_space_numer

    def S_from_y(self, y):
        x0_and_space = self.x0_and_space
        if isinstance(y, np.ndarray):
            x0_and_space = self.x0_and_space_numer
        size = {key: sqrtsize_of_mat(x0.shape[0]) for key, (x0, space) in x0_and_space.items()}
        return {key: matadd(x0, matmul(space, y)).reshape(size[key], size[key])
                    for key, (x0, space) in x0_and_space.items()}

    def eigvalsh(self, y: np.ndarray) -> Dict[Any, np.ndarray]:
        return {key: np.linalg.eigvalsh(S) for key, S in self.S_from_y(y).items()}

    def mineigs(self, y: np.ndarray) -> Dict[Any, float]:
        return {key: float(np.min(eigs)) if len(eigs) else 0. for key, eigs in self.eigvalsh(y).items()}

    def mineig(self, y: np.ndarray) -> float:
        return min(self.mineigs(y).values()) if len(self.x0_and_space) else 0.

    def decompose(self, y: Matrix) -> Optional[Tuple[Matrix, Decomp]]:
        """
        Decompose a vector `y` into a sum of rational numbers.
        """
        decomps = {}
        S = self.S_from_y(y)
        for key, s in S.items():
            decomp = congruence(s, upper=False)
            if decomp is None:
                return None
            U, diag = decomp
            decomps[key] = (s, U, diag)
        return y, decomps

    def nullspaces_lll(self, y: np.ndarray) -> Dict[Any, Matrix]:
        """
        Infer the nullspaces of PSD matrices by the LLL algorithm, see [1].

        References
        ----------
        [1] David Monniaux. On using sums-of-squares for exact computations without
        strict feasibility. 2010. hal-00487279.
        """
        eigvalsh = self.eigvalsh(y)

        y = rep_matrix_from_numpy(y)
        S = self.S_from_y(y)
        V = {}
        trunc = self.LLL_TRUNC
        for key, s in S.items():
            eigs = eigvalsh[key]
            if len(eigs) == 0:
                # empty block
                V[key] = Matrix.zeros(0, 0)
                continue

            if float(np.min(eigs)) > self.LLL_EIG:
                # this block is strictly definite -> no need to compute lll
                V[key] = Matrix.zeros(s.shape[0], 0)
                continue

            nullrank = int(np.sum(eigs < self.LLL_EIG))

            aug = Matrix.hstack(
                (self.LLL_WEIGHT*s).applyfunc(round), s.eye(s.shape[0]))
            try:
                v = lll(aug)[:, s.shape[0]:]
                vrep = v._rep.rep.to_sdm()
                v = v[:nullrank, :]
                for row in range(nullrank):
                    if any(map(lambda x: abs(x) > trunc, vrep.get(row, {}).values())):
                        v = v[:row, :]
                        break
                v = v.T
            except (DMError, ValueError, ZeroDivisionError) as e: # LLL algorithm failed
                v = Matrix.zeros(s.shape[0], 0)

            V[key] = v
        return V

    def rationalize_lll(self, y: np.ndarray,
            time_limit: Optional[Union[Callable, float]] = None) -> Optional[Tuple[Matrix, Decomp]]:
        time_limit = ArithmeticTimeout.make_checker(time_limit)

        V = self.nullspaces_lll(y)
        eq_space = []
        eq_rhs = []
        for key, (x0, space) in self.x0_and_space.items():
            y_space = matmul_multiple(space.T, V[key], time_limit = time_limit)
            rhs = matmul_multiple(x0.T, V[key], time_limit = time_limit)
            eq_space.append(y_space.T)
            eq_rhs.append(-rhs.T)
            time_limit()

        eq_space = Matrix.vstack(*eq_space)
        eq_rhs = Matrix.vstack(*eq_rhs)
        try:
            # this is really slow, we will think how to make it faster
            x0, space = solve_undetermined_linear(eq_space, eq_rhs, time_limit = time_limit)
        except ValueError: # Linear system no solution
            return None

        x0_numer = rep_matrix_to_numpy(x0).flatten()
        space_numer = rep_matrix_to_numpy(space)
        v0 = np.linalg.lstsq(space_numer, y - x0_numer, rcond=None)[0]

        for denom in self.ROUND_DENOMS:
            v = np.round(v0 * denom) / denom
            y = x0_numer + space_numer @ v
            if self.mineig(y) >= -1e-10:
                v = rep_matrix_from_numpy(np.round(v0 * denom).astype(int)) / denom
                y = x0 + space @ v
                result = self.decompose(y)
                if result is not None:
                    return result
                time_limit()

    def rationalize(self, y: np.ndarray,
            time_limit: Optional[Union[Callable, float]] = None) -> Optional[Tuple[Matrix, Decomp]]:
        time_limit = ArithmeticTimeout.make_checker(time_limit)

        if not isinstance(y, np.ndarray):
            y = rep_matrix_to_numpy(y)
            time_limit()
        y = y.astype(float)
        y0 = y.copy()

        if y.size == 0:
            return self.decompose(Matrix.zeros(0, 1))
        # if y.size == 1: # A + xB -> generalized eigenvalue problem
        #     pass

        # eigvalsh = self.eigvalsh(y)
        mineig_y0 = self.mineig(y0)
        time_limit()

        for denom in self.ROUND_DENOMS:
            y = np.round(y0 * denom) / denom
            if self.mineig(y) >= -1e-10:
                y = rep_matrix_from_numpy(np.round(y0 * denom).astype(int)) / denom
                result = self.decompose(y)
                if result is not None:
                    return result
                time_limit()

        if abs(mineig_y0) < 1e-4:
            result = self.rationalize_lll(y0, time_limit = time_limit)
            if result is not None:
                return result

        return None


####################################################################
"""
Below is the old code for rationalizing the solution of a SDP problem. It is
still used internally by the SDPPrimal class. However, the SDPProblem (dual form)
class uses the latest DualRationalizer algorithms from above.
Currently, the SDPSOS solver for inequalities uses the SDPProblem (dual form) class
instead of the SDPPrimal class. So it does not affect the behaviour of the SDPSOS
solver although it is outdated.
In the future, the following code for rationalization will be deprecated and removed.
"""
####################################################################

class Rationalizer(ABC):
    """TODO: deprecation"""
    @abstractmethod
    def __call__(self, y: np.ndarray) -> Generator[Matrix, None, None]:
        raise NotImplementedError
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

class IdentityRationalizer(Rationalizer):
    def __call__(self, y: np.ndarray) -> Generator[Matrix, None, None]:
        return (Matrix(y),)

class EmptyRationalizer(Rationalizer):
    def __call__(self, y: np.ndarray) -> Generator[Matrix, None, None]:
        return (Matrix.zeros(0, 1),)

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

    def __call__(self, y: np.ndarray) -> Generator[Matrix, None, None]:
        tol = max(1, np.abs(y).max()) * self.zero_tolerance
        y_rational_mask = np.abs(y) > tol
        y_rational = np.where(y_rational_mask, y, 0).flatten().tolist()
        y_rational = [rationalize(v, rounding = abs(v) * 1e-4) for v in y_rational]
        y_rational = Matrix(y_rational)
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

    def __call__(self, y: np.ndarray) -> Generator[Matrix, None, None]:
        y = np.array(y).astype(float)
        for lcm in self.lcms:
            y_rational = rep_matrix_from_numpy(np.round(y*lcm).astype(np.int64)) / lcm
            yield y_rational


def verify_is_pretty(
    y: Union[List, Matrix],
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
    y : Union[List, Matrix]
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

    # use math.lcm over python int is faster than sympy.lcm
    from math import lcm as _lcm

    lcm = 1
    if threshold is None:
        s = max(36, max(v.q for v in y)) ** 2
    else:
        s = threshold(y)
    for v in y:
        lcm = _lcm(lcm, v.q)
        if lcm > s:
            return False
    return True


def rationalize_and_decompose(
    y: Union[np.ndarray, Matrix],
    mat_func: Callable[[Matrix], Dict[str, Matrix]],
    projection: Optional[Callable[[Matrix], Matrix]] = None,
    rationalizers: List[Rationalizer] = [],
    reg: float = 0,
    perturb: bool = False,
    check_pretty: bool = True,
) -> Optional[Tuple[Matrix, Decomp]]:
    """
    TODO: deprecation

    Recover symmetric matrices from `x0 + space * y` and check whether they are
    positive semidefinite.

    Parameters
    ----------
    y : np.ndarray
        The vector to be rationalized.
    mat_func : Callable[[Matrix], Dict[str, Matrix]]
        Given a rationalized vector `y`, return a dictionary of matrices
        that needs to be PSD.
    projection : Optional[Callable[[Matrix], Matrix]]
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
    y, decompositions : Optional[Tuple[Matrix, Dict[str, Tuple[Matrix, Matrix, List[Rational]]]]]
        If the matrices are positive semidefinite, return the congruence decompositions `y, [(S, U, diag)]`
        So that each `S = U.T * diag(diag) * U` where `U` is upper triangular.
        Otherwise, return None.
    """
    if isinstance(y, MatrixBase):
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
                    S = S + reg * Matrix.eye(S.shape[0])

                congruence_decomp = congruence(S, perturb=perturb, upper=False)
                if congruence_decomp is None:
                    break

                U, diag = congruence_decomp
                decompositions[key] = (S, U, diag)
            else:
                return y_rational, decompositions
