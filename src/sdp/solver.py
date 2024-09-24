from contextlib import contextmanager, nullcontext, AbstractContextManager
from typing import Union, Optional, Any, Tuple, List, Dict, Callable, Generator

import numpy as np
import sympy as sp
from sympy import Expr, Symbol, Rational, MatrixBase
from sympy import MutableDenseMatrix as Matrix
from sympy.core.relational import Relational

from .arithmetic import solve_undetermined_linear
from .backend import (
    SDPBackend, RelaxationVariable, solve_numerical_dual_sdp,
    max_relax_var_objective, min_trace_objective, max_inner_objective
)
from .rationalize import rationalize, rationalize_and_decompose
from .transform import SDPTransformMixin
from .ipm import (
    SDPConvergenceError, SDPNumericalError, SDPInfeasibleError, SDPRationalizeError
)
from .utils import (
    S_from_y, Mat2Vec, congruence_with_perturbation,
    is_empty_matrix
)


Decomp = Dict[str, Tuple[Matrix, Matrix, List[Rational]]]
Objective = Tuple[str, Union[Expr, Callable[[SDPBackend], Any]]]
Constraint = Callable[[SDPBackend], Any]
MinEigen = Union[float, RelaxationVariable, Dict[str, Union[float, RelaxationVariable]]]
PicosExpression = Any

_RELATIONAL_TO_OPERATOR = {
    sp.GreaterThan: (1, '__ge__'),
    sp.StrictGreaterThan: (1, '__ge__'),
    sp.LessThan: (-1, '__ge__'),
    sp.StrictLessThan: (-1, '__ge__'),
    sp.Equality: (1, '__eq__')
}

def _check_picos(verbose = False):
    """
    Check whether PICOS is installed.
    """
    try:
        import picos
    except ImportError:
        if verbose:
            print('Cannot import picos, please use command "pip install picos" to install it.')
        return False
    return True


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

def _align_iters(
        iters: List[Union[Any, List[Any]]],
        default_types: List[Union[List[Any], Callable[[Any], bool]]]
    ) -> List[List[Any]]:
    """
    Align the iterators with the default types.
    """
    check_tp = lambda i, tp: (callable(tp) and not isinstance(tp, type) and tp(i)) or isinstance(i, tp)
    aligned_iters = []
    for i, tp in zip(iters, default_types):
        if isinstance(i, list):
            if len(i) == 0 and not check_tp(i, tp):
                return [[] for _ in range(len(iters))]
            if len(i) and check_tp(i[0], tp):
                aligned_iters.append(i)
                continue
        aligned_iters.append(None)
    lengths = [len(i) if i is not None else 0 for i in aligned_iters]
    max_len = max(lengths) if lengths else 0
    if max_len == 0:
        # everything iterator is a single value
        return [[i] for i in iters]
    return [is_single if is_single is not None else [i] * max_len for is_single, i in zip(aligned_iters, iters)]


class SDPProblem(SDPTransformMixin):
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
    _has_picos = _check_picos(verbose = True)
    def __init__(
        self,
        x0_and_space: Union[Dict[str, Tuple[Matrix, Matrix]], List[Tuple[Matrix, Matrix]]],
        free_symbols = None
    ):
        """
        Initializing a SDPProblem object.
        """
        if isinstance(x0_and_space, list):
            keys = ['S_%d'%i for i in range(len(x0_and_space))]
            x0_and_space = dict(zip(keys, x0_and_space))
        elif isinstance(x0_and_space, dict):
            keys = list(x0_and_space.keys())
        else:
            raise TypeError("x0_and_space should be a dict or a list containing tuples.")

        self._x0_and_space = x0_and_space

        self.y = None
        self.S = None
        self.decompositions = None
        self.free_symbols = _infer_free_symbols(x0_and_space, free_symbols)

        self.sdp = None

        # record the numerical solutions
        self._ys = []

        super().__init__()

    @property
    def dof(self):
        """
        The degree of freedom of the SDP problem.
        """
        return len(self.free_symbols)

    def keys(self, filter_none: bool = False) -> List[str]:
        keys = list(self._x0_and_space.keys())
        if filter_none:
            _size = lambda key: self._x0_and_space[key][1].shape[1] * self._x0_and_space[key][1].shape[0]
            keys = [key for key in keys if _size(key) != 0]
        return keys

    def get_size(self, key: str) -> int:
        return Mat2Vec.length_of_mat(self._x0_and_space[key][1].shape[0])

    @property
    def size(self) -> Dict[str, int]:
        return {key: self.get_size(key) for key in self.keys()}

    def __repr__(self):
        return "<SDPProblem dof=%d size=%s>"%(self.dof, self.size)

    def _standardize_mat_dict(self, mat_dict: Dict[str, Matrix]) -> Dict[str, Matrix]:
        """
        Standardize the matrix dictionary.
        """
        if not set(mat_dict.keys()) == set(self.keys()):
            raise ValueError("The keys of the matrix dictionary should be the same as the keys of the SDP problem.")
        for key, X in mat_dict.items():
            if not isinstance(X, MatrixBase):
                raise ValueError("The values of the matrix dictionary should be sympy MatrixBase.")
            if is_empty_matrix(X):
                n = self.get_size(key)
                mat_dict[key] = sp.zeros(n, 0)
        return mat_dict

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

    def S_from_y(self, 
            y: Optional[Union[Matrix, np.ndarray, Dict]] = None
        ) -> Dict[str, Matrix]:
        """
        Given y, compute the symmetric matrices. This is useful when we want to see the
        symbolic representation of the SDP problem.

        This function does not register the result to self.S.

        Parameters
        ----------
        y : Optional[Union[Matrix, np.ndarray]]
            The generating vector. If None, it uses a symbolic vector.

        Returns
        ----------
        S : Dict[str, Matrix]
            The symmetric matrices that SDP requires to be positive semidefinite.
        """
        m = self.dof
        if y is None:
            y = Matrix(self.free_symbols).reshape(m, 1)
        elif isinstance(y, MatrixBase):
            if y.shape != (m, 1):
                raise ValueError(f"Vector y must be a matrix of shape ({m}, 1).")
        elif isinstance(y, np.ndarray):
            if y.size != m:
                raise ValueError(f"Vector y must be an array of shape ({m},) or ({m}, 1).")
            y = Matrix(y.flatten())
        elif isinstance(y, dict):
            y = Matrix([y.get(v, v) for v in self.free_symbols]).reshape(m, 1)

        return S_from_y(y, self._x0_and_space)

    def register_y(self,
            y: Union[Matrix, np.ndarray, Dict],
            perturb: bool = False,
            propagate_to_parent: bool = True
        ) -> None:
        """
        Manually register a solution y to the SDP problem.

        Parameters
        ----------
        y : Union[Matrix, np.ndarray, Dict]
            The solution to the SDP problem.
        perturb : bool
            If perturb == True, it must return the result by adding a small perturbation * identity to the matrices.
            This is useful when the given y is numerical.
        propagate_to_parent : bool
            If True, propagate the solution to the parent SDP problem.
        """
        S = self.S_from_y(y)
        decomps = {}
        for key, s in S.items():
            decomp = congruence_with_perturbation(s, perturb = perturb)
            if decomp is None:
                raise ValueError(f"Matrix {key} is not positive semidefinite given y.")
            decomps[key] = decomp
        self.y = y
        self.S = S
        self.decompositions = decomps
        if propagate_to_parent:
            self.propagate_to_parent(recursive = True)


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
                (max_relax_var_objective(self.dof), RelaxationVariable(1, 0)),
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
        decomp = rationalize_and_decompose(y, self._x0_and_space,
            try_rationalize_with_mask=try_rationalize_with_mask, times=times, check_pretty=check_pretty
        )
        return decomp

    def rationalize_combine(
            self,
            ys: List[np.ndarray] = None,
            verbose: bool = False,
        ) ->  Optional[Tuple[Matrix, Decomp]]:
        """
        Linearly combine all numerical solutions [y] to produce a rational solution.

        Parameters
        ----------
        y : np.ndarray
            Numerical solution y.
        verbose : bool
            Whether to print out the eigenvalues of the combined matrix. Defaults
            to False.

        Returns
        ----------
        y, decompositions : Optional[Tuple[Matrix, Decomp]]
            If the problem is solved, return the congruence decompositions `y, [(S, U, diag)]`
            So that each `S = U.T * diag(diag) * U` where `U` is upper triangular.
            Otherwise, return None.
        """
        if ys is None:
            ys = self._ys

        if len(ys) == 0:
            return None

        y = np.array(ys).mean(axis = 0)

        S_numer = S_from_y(y, self._x0_and_space)
        if all(_.is_positive_definite for _ in S_numer.values()):
            lcm, times = 1260, 5
        else:
            # spaces = [space for x0, space in self._x0_and_space.values()]
            # lcm = max(1260, sp.prod(set.union(*[set(sp.primefactors(_.q)) for _ in spaces if isinstance(_, Rational)])))
            # times = int(10 / sp.log(lcm, 10).n(15) + 3)
            times = 5

        if verbose:
            Svalues = [np.array(_).astype(np.float64) for _ in S_numer.values()]
            mineigs = [min(np.linalg.eigvalsh(S)) for S in Svalues if S.size > 0]
            print('Minimum Eigenvals = %s'%mineigs)

        decomp = rationalize_and_decompose(y, self._x0_and_space,
            try_rationalize_with_mask = False, lcm = 1260, times = times
        )
        return decomp


    def _solve_from_multiple_configs(self,
            list_of_objective: List[Objective] = [],
            list_of_constraints: List[List[Constraint]] = [],
            list_of_min_eigen: List[MinEigen] = [],
            solver: str = 'picos',
            allow_numer: int = 0,
            rationalize_configs = {},
            verbose: bool = False,
            **kwargs
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
                solver=solver,
                **kwargs
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

                decomp = rationalize_and_decompose(y, self._x0_and_space, **rationalize_configs)
                if decomp is not None:
                    return decomp

                if allow_numer == 2:
                    # force to return the numerical solution if rationalization fails
                    return _force_return(self, y)

        if len(self._ys) > num_sol:
            # new numerical solution found
            decomp = self.rationalize_combine(verbose = verbose)
            if decomp is not None:
                return decomp

            if allow_numer == 1:
                y = self._ys[-1]
                return rationalize_and_decompose(y, self._x0_and_space,
                            try_rationalize_with_mask = False, times = 0, perturb = True, check_pretty = False)
            else:
                raise SDPRationalizeError(
                    "Failed to find a rational solution despite having a numerical solution."
                )

        return None

    def solve(self,
            objectives: Union[Objective, List[Objective]] = [],
            constraints: Union[List[Constraint], List[List[Constraint]]] = [],
            min_eigen: Union[MinEigen, List[MinEigen]] = [],
            solver: str = 'picos',
            use_default_configs: bool = True,
            allow_numer: int = 0,
            verbose: bool = False,
            solve_child: bool = True,
            propagate_to_parent: bool = True,
            **kwargs
        ) -> bool:
        """
        Interface for solving the SDP problem.

        Parameters
        ----------
        use_default_configs : bool
            Whether to use the default configurations of objectives+constraints+min_eigen.
            If True, it appends the default configurations of SDP to the given configurations.
            If False, it only uses the given configurations.
        allow_numer : int
            Whether to accept numerical solution. 
            If 0, then it claims failure if the rational feasible solution does not exist.
            If 1, then it accepts a numerical solution if the rational feasible solution does not exist.
            If 2, then it accepts the first numerical solution if rationalization fails.
            If 3, then it accepts the first numerical solution directly. Defaults to 0.
        verbose : bool
            If True, print the information of the solving process.
        solve_child : bool
            Whether to solve the problem from the child node. Defaults to True. If True,
            it only uses the newest child node to solve the problem. If no child node is found,
            it defaults to solve the problem by itself.
        propagate_to_parent : bool
            Whether to propagate the result to the parent node. Defaults to True.

        Returns
        ----------
        bool
            Whether the problem is solved. If True, the result can be accessed by
            SDPProblem.y and SDPProblem.S and SDPProblem.decompositions.
        """
        if solve_child:
            child: SDPProblem = self.get_last_child()
            if child is not self:
                return child.solve(
                    objectives = objectives,
                    constraints = constraints,
                    min_eigen = min_eigen,
                    solver = solver,
                    allow_numer = allow_numer,
                    verbose = verbose,
                    solve_child = solve_child,
                    propagate_to_parent = propagate_to_parent,
                    **kwargs
                )

        configs = _align_iters(
            [objectives, constraints, min_eigen],
            [tuple, list, (float, int, RelaxationVariable, dict)]
        )
        if use_default_configs:
            default_configs = self._get_defaulted_configs()
            for i in range(len(configs)):
                configs[i] += default_configs[i]
        if self.dof == 0:
            # trim the configs to the first one
            if len(configs[0]) > 1:
                configs = [[_[0]] for _ in configs]


        #################################################
        #            Solve the SDP problem
        #################################################
        solution = self._solve_from_multiple_configs(
            *configs, solver=solver, allow_numer = allow_numer, verbose = verbose, **kwargs
        )

        if solution is not None:
            # register the solution
            self.y = solution[0]
            self.S = dict((key, S[0]) for key, S in solution[1].items())
            self.decompositions = dict((key, S[1:]) for key, S in solution[1].items())

        if propagate_to_parent:
            self.propagate_to_parent()

        return (solution is not None)
