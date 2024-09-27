from typing import Union, Optional, Any, Tuple, List, Dict, Callable

import numpy as np
import sympy as sp
from sympy import Expr, Symbol, Rational, MatrixBase
from sympy import MutableDenseMatrix as Matrix
from sympy.core.relational import Relational

from .abstract import Decomp, Objective, Constraint, MinEigen
from .arithmetic import solve_undetermined_linear
from .backend import (
    SDPBackend, solve_numerical_dual_sdp,
    max_relax_var_objective, min_trace_objective, max_inner_objective
)
from .rationalize import rationalize, rationalize_and_decompose
from .transform import DualTransformMixin
from .ipm import SDPRationalizeError

from .utils import S_from_y

_RELATIONAL_TO_OPERATOR = {
    sp.GreaterThan: (1, '__ge__'),
    sp.StrictGreaterThan: (1, '__ge__'),
    sp.LessThan: (-1, '__ge__'),
    sp.StrictLessThan: (-1, '__ge__'),
    sp.Equality: (1, '__eq__')
}

def _decompose_matrix(
        M: Matrix,
        variables: Optional[List[Symbol]] = None
    ) -> Tuple[Matrix, Matrix, Matrix]:
    """
    Decomposes a symbolic matrix into the form vec(M) = x + A @ v
    where x is a constant vector, A is a constant matrix, and v is a vector of variables.

    See also in `sympy.solvers.solveset.linear_eq_to_matrix`.

    Parameters
    ----------
    M : Matrix
        The matrix to be decomposed.
    variables : List[Symbol]
        The variables to be used in the decomposition. If None, it uses M.free_symbols.

    Returns
    ----------
    x : Matrix
        The constant vector.
    A : Matrix
        The constant matrix.
    v : Matrix
        The vector of variables.
    """
    rows, cols = M.shape
    if variables is None:
        variables = list(M.free_symbols)
        variables = sorted(variables, key = lambda x: x.name)
    variable_index = {var: idx for idx, var in enumerate(variables)}

    v = Matrix(variables)
    x = sp.zeros(rows * cols, 1)
    A = sp.zeros(rows * cols, len(variables))

    for i in range(rows):
        for j in range(cols):
            expr = M[i, j]
            terms = sp.collect(expr, variables, evaluate=False)

            constant_term = terms.pop(sp.S.One, 0)  # Extract and remove constant term for x
            x[i * cols + j] = constant_term

            for term, coeff in terms.items():
                A[i * cols + j, variable_index[term]] = coeff  # Extract coefficients for A

    return x, A, v

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

def _exprs_to_arrays(locals: Dict[str, Any], symbols: List[Symbol],
        exprs: List[Union[Callable, Expr, Relational, Union[Tuple[Matrix, Rational], Tuple[Matrix, Rational, str]]]]
    ) -> List[Union[Tuple[Matrix, Rational], Tuple[Matrix, Rational, str]]]:
    """
    Convert expressions to arrays with respect to the free symbols.

    Parameters
    ----------
    locals : Dict[str, Any]
        The local variables.
    symbols : List[Symbol]
        The free symbols.
    exprs : List[Union[Callable, Expr, Relational, Matrix]]
        For each expression, it can be a Callable, Expr, Relational, or matrix.
        If it is a Callable, it should be a function that calls on the locals and returns Expr/Relational/Matrix.
        If it is a Expr, it should be with respect to the free symbols.
        If it is a Relational, it should be with respect to the free symbols.

    Returns
    ----------
    Matrix, Rational, [, operator] : Union[Tuple[Matrix, Rational], Tuple[Matrix, Rational, str]]
        The coefficient vector with respect to the free symbols and the Rational of RHS (constant).
        If it is a Relational, it returns the operator also.
    """
    op_list = []
    vec_list = []
    index_list = []
    result = [None for _ in range(len(exprs))]
    for i, expr in enumerate(exprs):
        c, op = 0, None
        if callable(expr):
            expr = expr(locals)
        if isinstance(expr, tuple):
            if len(expr) == 3:
                expr, c, op = expr
            else:
                expr, c = expr
        if isinstance(expr, Relational):
            sign, op = _RELATIONAL_TO_OPERATOR[expr.__class__]
            expr = expr.lhs - expr.rhs if sign == 1 else expr.rhs - expr.lhs
            c = -c if sign == -1 else c
        if isinstance(expr, (Expr, int, float)):
            vec_list.append(expr)
            op_list.append(op)
            index_list.append(i)
        else:
            if op is not None:
                result[i] = (expr, c, op)
            else:
                result[i] = (expr, c)

    const, A, _ = _decompose_matrix(sp.Matrix(vec_list), symbols)

    for j in range(len(index_list)):
        i = index_list[j]
        if op_list[j] is not None:
            result[i] = (A[j,:], -const[j], op_list[j])
        else:
            result[i] = (A[j,:], -const[j])
    return result


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
        splits: Union[Dict[str, int], List[int]]
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
        return SDPProblem(x0_and_space)
        

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
            x0, space, _ = _decompose_matrix(s, free_symbols)
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
                (-min_trace, 0),
                (max_inner_objective(self._x0_and_space[obj_key][1], 1.), 0),
                (max_relax_var_objective(self.dof), (1, 0)),
            ]

        objectives = [_[0] for _ in objective_and_min_eigens]
        min_eigens = [_[1] for _ in objective_and_min_eigens]
        # x = np.random.randn(*sdp.variables[obj_key].shape)
        # objectives.append(('max', lambda sdp: sdp.variables[obj_key]|x))
        constraints = [[] for _ in range(len(objectives))]
        return [objectives, constraints, min_eigens]

    def rationalize(
            self,
            y: np.ndarray,
            try_rationalize_with_mask: bool = True,
            times: int = 1,
            check_pretty: bool = True,
        ) -> Optional[Tuple[Matrix, Decomp]]:
        """
        Rationalize a numerical vector y so that it produces a rational solution to SDP.

        Parameters
        ----------
        y : np.ndarray
            Numerical solution y.
        kwargs : Any
            Arguments that passed into rationalize_and_decompose.

        Returns
        ----------
        y, decompositions : Optional[Tuple[Matrix, Decomp]]
            If the problem is solved, return the congruence decompositions `y, [(S, U, diag)]`
            So that each `S = U.T * diag(diag) * U` where `U` is upper triangular.
            Otherwise, return None.
        """
        decomp = rationalize_and_decompose(y, self.S_from_y,
            try_rationalize_with_mask=try_rationalize_with_mask, times=times, check_pretty=check_pretty
        )
        return decomp

    def _solve_from_multiple_configs(self,
            list_of_objective: List[Objective] = [],
            list_of_constraints: List[List[Constraint]] = [],
            list_of_min_eigen: List[MinEigen] = [],
            solver: Optional[str] = None,
            allow_numer: int = 0,
            rationalize_configs = {},
            verbose: bool = False,
            solver_options: Dict[str, Any] = {},
            raise_exception: bool = False
        ):

        num_sol = len(self._ys)
        _locals = None

        if any(callable(_) for _ in list_of_objective) or any(any(callable(_) for _ in _) for _ in list_of_constraints):
            _locals = self.S_from_y()
            _locals['y'] = self.y

        for obj, con, eig in zip(list_of_objective, list_of_constraints, list_of_min_eigen):
            # iterate through the configurations
            con = _exprs_to_arrays(_locals, self.free_symbols, con)
            y = solve_numerical_dual_sdp(
                self._x0_and_space,
                _exprs_to_arrays(_locals, self.free_symbols, [obj])[0][0],
                constraints=con,
                min_eigen=eig,
                solver=solver, solver_options=solver_options, raise_exception=raise_exception
            )
            if y is not None:
                self._ys.append(y)

                def _force_return(self: SDPProblem, y):
                    self.register_y(y, perturb = True, propagate_to_parent = False)
                    _decomp = dict((key, (s, d)) for (key, s), d in zip(self.S.items(), self.decompositions.values()))
                    return y, _decomp

                if allow_numer >= 3:
                    # force to return the numerical solution
                    return _force_return(self, y)

                decomp = rationalize_and_decompose(y, self.S_from_y, **rationalize_configs)
                if decomp is not None:
                    return decomp

                if allow_numer == 2:
                    # force to return the numerical solution if rationalization fails
                    return _force_return(self, y)

        if len(self._ys) > num_sol:
            # new numerical solution found
            decomp = self.rationalize_combine(self._ys, self.S_from_y, verbose = verbose)
            if decomp is not None:
                return decomp

            if allow_numer == 1:
                y = self._ys[-1]
                return rationalize_and_decompose(y, self.S_from_y,
                            try_rationalize_with_mask = False, times = 0, perturb = True, check_pretty = False)
            else:
                raise SDPRationalizeError(
                    "Failed to find a rational solution despite having a numerical solution."
                )

        return None