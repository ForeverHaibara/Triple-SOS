from typing import Union, Optional, Any, Tuple, List, Dict

import numpy as np
from sympy import Expr, Symbol, Rational, MatrixBase
from sympy import MutableDenseMatrix as Matrix

from .abstract import Decomp, Objective, Constraint, MinEigen
from .arithmetic import solve_undetermined_linear
from .backend import (
    SDPBackend, solve_numerical_dual_sdp,
    max_relax_var_objective, min_trace_objective, max_inner_objective
)
from .transform import DualTransformMixin

from .utils import S_from_y, decompose_matrix, exprs_to_arrays



def _infer_free_symbols(x0_and_space: Dict[str, Tuple[Matrix, Matrix]], free_symbols: List[Symbol]) -> List[Symbol]:
    """
    Get the free symbols from given and validate the dimension of the input matrices.

    Parameters
    ----------
    x0_and_space : Dict[str, Tuple[Matrix, Matrix]]
        The matrices to be decomposed.
    free_symbols : List[Symbol]
        The free symbols to be used. If None, it uses the default symbols.
    """
    keys = list(x0_and_space.keys())
    if len(keys):
        dof = x0_and_space[keys[0]][1].shape[1]
        for x0, space in x0_and_space.values():
            if space.shape[1] != dof:
                raise ValueError("The number of columns of spaces should be the same.")

        if free_symbols is not None:
            if len(free_symbols) != dof:
                raise ValueError("Length of free_symbols and space should be the same. But got %d and %d."%(len(free_symbols), dof))
            return list(free_symbols)
        else:
            return list(Symbol('y_{%d}'%i) for i in range(dof))
    return []


class SDPProblem(DualTransformMixin):
    """
    Class to solve rational dual SDP feasible problems, which is in the form of

        S_i = C_i + y_1 * A_i1 + y_2 * A_i2 + ... + y_n * A_in >> 0.
    
    where C, A_ij ... are known symmetric matrices, and y_i are free variables.

    It can be rewritten in the form of

        vec(S_i) = x_i + space_i @ y >> 0.

    And together they are vec([S_1, S_2, ...]) = [x_1, x_2, ...] + [space_1, space_2, ...] @ y
    where x_i and space_i are known. The problem is to find a rational solution y such that S_i >> 0.
    This is the standard form of our rational SDP feasible problem.
    """
    is_dual = True
    is_primal = False
    def __init__(
        self,
        x0_and_space: Union[Dict[str, Tuple[Matrix, Matrix]], List[Tuple[Matrix, Matrix]]],
        free_symbols = None
    ):
        """
        Initializing a SDPProblem object.
        """
        super().__init__()

        self._x0_and_space: Dict[str, Tuple[Matrix, Matrix]] = None
        self._init_space(x0_and_space, '_x0_and_space')

        self.free_symbols = _infer_free_symbols(self._x0_and_space, free_symbols)

    def keys(self, filter_none: bool = False) -> List[str]:
        space = self._x0_and_space
        keys = list(space.keys())
        if filter_none:
            _size = lambda key: space[key][1].shape[1] * space[key][1].shape[0]
            keys = [key for key in keys if _size(key) != 0]
        return keys

    @property
    def dof(self):
        """
        The degree of freedom of the SDP problem.
        """
        return len(self.free_symbols)

    @classmethod
    def from_full_x0_and_space(
        cls,
        x0: Matrix,
        space: Matrix,
        splits: Union[Dict[str, int], List[int]],
        constrain_symmetry: bool = True
    ) -> 'SDPProblem':
        keys = None
        if isinstance(splits, dict):
            keys = list(splits.keys())
            splits = list(splits.values())

        x0_and_space = []
        start = 0
        for n in splits:
            x0_ = x0[start:start+n**2,:]
            space_ = space[start:start+n**2,:]
            x0_and_space.append((x0_, space_))
            start += n**2

        if keys is not None:
            x0_and_space = dict(zip(keys, x0_and_space))
        sdp = SDPProblem(x0_and_space)

        if constrain_symmetry:
            sdp = sdp.constrain_symmetry()
            sdp._transforms.clear()
        return sdp

    @classmethod
    def from_equations(
        cls,
        eq: Matrix,
        rhs: Matrix,
        splits: Union[Dict[str, int], List[int]]
    ) -> 'SDPProblem':
        """
        Assume the SDP problem can be rewritten in the form of

            eq * [vec(S1); vec(S2); ...] = rhs
        
        where Si.shape[0] = splits[i].
        The function formulates the SDP problem from the given equations.
        This is also the primal form of the SDP problem.

        Parameters
        ----------
        eq : Matrix
            The matrix eq.
        rhs : Matrix
            The matrix rhs.
        splits : Union[Dict[str, int], List[int]]
            The splits of the size of each symmetric matrix.

        Returns
        ----------
        sdp : SDPProblem
            The SDP problem constructed.    
        """
        x0, space = solve_undetermined_linear(eq, rhs)
        return cls.from_full_x0_and_space(x0, space, splits)

    @classmethod
    def from_matrix(
        cls,
        S: Union[Matrix, List[Matrix], Dict[str, Matrix]],
    ) -> 'SDPProblem':
        """
        Construct a `SDPProblem` from symbolic symmetric matrices.
        The problem is to solve a parameter set such that all given
        symmetric matrices are positive semidefinite. The result can
        be obtained by `SDPProblem.as_params()`.

        Parameters
        ----------
        S : Union[Matrix, List[Matrix], Dict[str, Matrix]]
            The symmetric matrices that SDP requires to be positive semidefinite.
            Each entry of the matrix should be linear in the free symbols.

        Returns
        ----------
        sdp : SDPProblem
            The SDP problem constructed.
        """

        keys = None
        if isinstance(S, dict):
            keys = list(S.keys())
            S = list(S.values())

        if isinstance(S, Matrix):
            S = [S]

        free_symbols = set()
        for s in S:
            if not isinstance(s, Matrix):
                raise ValueError("S must be a list of Matrix or dict of Matrix.")
            free_symbols |= set(s.free_symbols)

        free_symbols = list(free_symbols)
        free_symbols = sorted(free_symbols, key = lambda x: x.name)

        x0_and_space = []
        for s in S:
            x0, space, _ = decompose_matrix(s, free_symbols)
            x0_and_space.append((x0, space))

        if keys is not None:
            x0_and_space = dict(zip(keys, x0_and_space))

        return SDPProblem(x0_and_space, free_symbols = free_symbols)

    def S_from_y(self, y: Optional[Union[Matrix, np.ndarray, Dict]] = None) -> Dict[str, Matrix]:
        m = self.dof
        if y is None:
            y = Matrix(self.free_symbols).reshape(m, 1)
        elif isinstance(y, MatrixBase):
            if m == 0 and y.shape[0] * y.shape[1] == 0:
                y = Matrix.zeros(0, 1)
            elif y.shape == (1, m):
                y = y.T
            elif y.shape != (m, 1):
                raise ValueError(f"Vector y must be a matrix of shape ({m}, 1), but got {y.shape}.")
        elif isinstance(y, np.ndarray):
            if y.size != m:
                raise ValueError(f"Vector y must be an array of shape ({m},) or ({m}, 1), but got {y.shape}.")
            y = Matrix(y.flatten())
        elif isinstance(y, dict):
            y = Matrix([y.get(v, v) for v in self.free_symbols]).reshape(m, 1)

        return S_from_y(y, self._x0_and_space)

    def as_params(self) -> Dict[Symbol, Rational]:
        """
        Return the free symbols and their values.
        """
        return dict(zip(self.free_symbols, self.y))

    def _get_defaulted_configs(self) -> List[List[Any]]:
        """
        Get the default configurations of the SDP problem.
        """
        if self.dof == 0:
            objective_and_min_eigens = [(0, 0)]
        else:
            obj_key = self.keys(filter_none = True)[0]
            min_trace = min_trace_objective(self._x0_and_space[obj_key][1])
            objective_and_min_eigens = [
                (min_trace, 0),
                (np.zeros(self.dof), 0), # feasible solution
                # (-min_trace, 0),
                # (max_inner_objective(self._x0_and_space[obj_key][1], 1.), 0),
                (max_relax_var_objective(self.dof), (1, 0)),
            ]

        objectives = [_[0] for _ in objective_and_min_eigens]
        min_eigens = [_[1] for _ in objective_and_min_eigens]
        # x = np.random.randn(*sdp.variables[obj_key].shape)
        # objectives.append(('max', lambda sdp: sdp.variables[obj_key]|x))
        constraints = [[] for _ in range(len(objectives))]
        return [objectives, constraints, min_eigens]

    def _solve_numerical_sdp(self,
            objective: Objective,
            constraints: List[Constraint] = [],
            min_eigen: MinEigen = 0,
            scaling: float = 6.,
            solver: Optional[str] = None,
            verbose: bool = False,
            solver_options: Dict[str, Any] = {},
            raise_exception: bool = False
        ):
        _locals = None

        if callable(objective) or any(callable(_) for _  in constraints):
            _locals = self.S_from_y()
            _locals['y'] = self.y

        con = exprs_to_arrays(_locals, self.free_symbols, constraints)
        obj = exprs_to_arrays(_locals, self.free_symbols, [objective])[0][0]
        return solve_numerical_dual_sdp(
                self._x0_and_space, objective=obj, constraints=con, min_eigen=min_eigen, scaling=scaling,
                solver=solver, solver_options=solver_options, raise_exception=raise_exception
            )