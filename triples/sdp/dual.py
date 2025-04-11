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
from .backends import SDPError, SDPProblemError, SDPInfeasibleError
from .rationalize import RationalizeWithMask, RationalizeSimultaneously
from .transforms import TransformableDual

from .utils import S_from_y, decompose_matrix, exprs_to_arrays


def _get_unique_symbols(_x0_and_space, dof: int, xname: str = 'y'):
    """
    Generate `dof` unique symbols that differ from the existing symbols in `_x0_and_space`.

    Parameters
    ----------
    _x0_and_space : Dict[str, Tuple[Matrix, Matrix]]
        The given matrices.
    dof : int
        The number of symbols to generate.
    xname : str
        The prefix of the symbol name.
    """
    used_symbols = set()
    for x0, space in _x0_and_space.values():
        if hasattr(x0, 'free_symbols'):
            used_symbols.update(set(_.name for _ in x0.free_symbols))
        if hasattr(space, 'free_symbols'):
            used_symbols.update(set(_.name for _ in space.free_symbols))
    xname = xname + '_{'
    n = len(xname)
    used_symbols = set(s[n:-1] for s in used_symbols if s.startswith(xname) and s[-1] == '}')
    digits = '0123456789'
    used_digits = list(map(int, filter(lambda x: all(d in digits for d in x), used_symbols)))
    max_digit = max(used_digits, default=-1) + 1
    return [Symbol(xname + str(i) + '}') for i in range(max_digit, max_digit + dof)]

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
        dof = 0
        for x0, space in x0_and_space.values():
            if space.shape[1] != 0:
                dof = space.shape[1]
                break
        for key, (x0, space) in x0_and_space.items():
            if space.shape[0] * space.shape[1] == 0:
                x0_and_space[key] = (x0, Matrix.zeros(x0.shape[0], dof))

        for x0, space in x0_and_space.values():
            if space.shape[1] != dof:
                raise ValueError("The number of columns of spaces should be the same.")

        if free_symbols is not None:
            if len(free_symbols) != dof:
                raise ValueError("Length of free_symbols and space should be the same. But got %d and %d."%(len(free_symbols), dof))
            return list(free_symbols)
        else:
            return _get_unique_symbols(x0_and_space, dof, xname='y')
            # return list(Symbol('y_{%d}'%i) for i in range(dof))
    return []


class SDPProblem(TransformableDual):
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
        constrain_symmetry: bool = False
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
        return cls.from_full_x0_and_space(x0, space, splits, constrain_symmetry = True)

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
            y = Matrix(m, 1, y.flatten().tolist())
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


    def _old_solve(self,
            objective: Union[Objective, List[Objective]] = None,
            constraints: Union[List[Constraint], List[List[Constraint]]] = None,
            min_eigen: Union[MinEigen, List[MinEigen]] = None,
            scaling: float = 6.,
            solver: Optional[str] = None,
            use_default_configs: int = 1,
            allow_numer: int = 0,
            verbose: bool = False,
            solve_child: bool = True,
            propagate_to_parent: bool = True,
            solver_options: Dict[str, Any] = {},
            raise_exception: bool = False
        ) -> bool:
        """
        Interface for solving the SDP problem.

        Parameters
        ----------
        objectives : Expr, ndarray or list of objectives
            Objective to minimize. It can be linear sympy expressions using
            the symbols from this SDPProblem.free_symbols. If it is a numpy
            array, it should align the length of dof and will be treated
            as a inner product. If it is a list, it should contain
            the objectives for multiple runs.
        constraints : list of Relational or list of list of Relational
            Constraints to satisfy. Each constraint should be a linear
            sympy relational expression using the symbols from this SDPProblem.free_symbols.
            If it is a list of list, it should contain the constraints for multiple runs.
        min_eigen : int, tuple, dict or list of int, tuple, dict
            The minimum eigenvalue of the solution. If it is a tuple (k, b), it will be
            inferred as S >= k*x + b where x >= 0 is a slack variable. It can also be a dict
            to control the eigenvalue bound of each matrix.
            If it is a list, it should contain for multiple runs.
        scaling: float
            When the entries of the matrices are too large, it will be scaled so that
            the maximum entry does not exceed this value. Defaults to 6. Set to zero
            to disable scaling. This argument is set only for nmerical stability.

        use_default_configs : int
            Whether to use the default configurations of objectives+constraints+min_eigen.
            Defaults to 1.
            * If 0, it only uses the given configurations.
            * If 1, it appends the default configurations to the given configurations if 
            no configurations are given. But when any configuration is given,
            it only uses the given configurations.
            * If 2, it appends the default configurations to the given configurations.

        allow_numer : int
            Whether to accept numerical solution. Defaults to 0.
            * If 0, then it claims failure if the rational feasible solution does not exist.
            * If 1, then it accepts a numerical solution if the rational feasible solution does not exist.
            * If 2, then it accepts the first numerical solution if rationalization fails.
            * If 3, then it accepts the first numerical solution directly.

        verbose : bool
            If True, print the information of the solving process.
        solve_child : bool
            Whether to solve the problem from the child node. Defaults to True. If True,
            it only uses the newest child node to solve the problem. If no child node is found,
            it defaults to solve the problem by itself.
        propagate_to_parent : bool
            Whether to propagate the result to the parent node. Defaults to True.
        solver_options : Dict[str, Any]
            The options passed to the SDP backend solver.
        raise_exception : bool
            If True, raise an exception if an error occurs in the backend. It is for 
            debugging purpose.

        Returns
        ----------
        bool
            Whether the problem is solved. If True, the result can be accessed by
            SDPProblem.y and SDPProblem.S and SDPProblem.decompositions.

        Examples
        ----------
        Here is a simple tutorial of using the SDPProblem class
        to solve a symbolic dual SDP feasible problem. The SDPProblem class
        can be initialized using .from_matrix() method given one or multiple
        symmetric symbolic linear sympy matrices. For example:

            >>> import sympy as sp
            >>> a, b, c = sp.symbols('a b c')
            >>> S1 = sp.Matrix([[a+1,0,b],[0,2,-a-1],[b,-a-1,2]])
            >>> sdp = SDPProblem.from_matrix(S1)
            >>> sdp.solve()
            True
        
        The code above will try to find a feasible solution for S1 >> 0.
        The result can be accessed by sdp.y and sdp.S. For example,

            >>> print(sdp.y)
            Matrix([[-1], [0]])
            >>> print(sdp.S)
            {'S_0': Matrix([
            [0, 0, 0],
            [0, 2, 0],
            [0, 0, 2]])}

        Pass in a dictionary of matrices to require multiple matrices to be positive semidefinite:

            >>> S2 = sp.Matrix([[a,b+1],[b+1,a]])
            >>> sdp = SDPProblem.from_matrix({'S1': S1, 'S2': S2})
            >>> sdp.solve()
            True
            >>> print(sdp.S)
            {'S1': Matrix([
            [ 1,  0, -1],
            [ 0,  2, -1],
            [-1, -1,  2]]), 'S2': Matrix([
            [0, 0],
            [0, 0]])}

        When the SDPProblem is a leaf node, it is possible to specify objectives /
        constraints / minimum eigenvalues of the matrices for the numerical solver.
        However, the solution would be rationalized so it is not guaranteed to get the truly
        optimal solution. Here is an example that solves the problem with objective (minimizing) (-a)
        and a linear constraint b <= -1:

            >>> sdp.solve(objective=-a, constraints=[b <= -1], min_eigen=0)
            True
            >>> sdp.as_params()
            {a: 8641/12799, b: -1}

        The truly optimal solution should be a = 0.675130870566... and b = -1, involving
        a cubic root of -a**3 - 3*a**2 + a + 1 = 0.
        To solve the truly optimal (numerical) solution and avoid rationalization, set allow_numer to 3.
        However, this does not guarantee strict feasibility. The matrices are PSD up to some rounding error:

            >>> sdp.solve(objective=-a, constraints=[b <= -1], min_eigen=0, allow_numer=3)
            True
            >>> sdp.as_params()
            {a: 0.6751308702458965, b: -1.0000000002876923}
        """
        return super().solve(
            objective=objective, constraints=constraints, min_eigen=min_eigen,
            scaling=scaling, solver=solver, use_default_configs=use_default_configs,
            allow_numer=allow_numer, verbose=verbose, solve_child=solve_child,
            propagate_to_parent=propagate_to_parent, solver_options=solver_options,
            raise_exception=raise_exception
        )

    def solve(self,
            solver: Optional[str] = None,
            verbose: bool = False,
            solve_child: bool = True,
            propagate_to_parent: bool = True,
        ) -> bool:
        if solve_child:
            self = self.get_last_child()
        if self.dof == 0:
            y = Matrix.zeros(0, 1)
            try:
                self.register_y(y)
                return True
            except ValueError:
                raise SDPInfeasibleError
            return False

        success = False

        from .backends import solve_numerical_dual_sdp
        x0_and_space = {}
        size = self.size
        for key, (x0, space) in self._x0_and_space.items():
            n = size[key]
            diagonals = Matrix([-i%(n+1) for i in range(n**2)])
            x0_and_space[key] = (x0, Matrix.hstack(space, diagonals))
        objective = np.zeros(self.dof + 1)
        objective[-1] = -1
        constraints = [(objective, 0, '<'), (objective, -10, '>')]
        sol = None
        try:
            sol = solve_numerical_dual_sdp(
                x0_and_space, objective=objective, constraints=constraints,
                solver=solver
            )
        except SDPError as e:
            if isinstance(e, SDPProblemError):
                raise SDPInfeasibleError from None
        if sol is not None:
            sol = sol[:-1]
            self._ys.append(sol)
            solution = self.rationalize(sol, verbose=verbose,
                rationalizers=[RationalizeWithMask(), RationalizeSimultaneously([1,1260,1260**3])])
            if solution is not None:
                self.y = solution[0]
                self.S = dict((key, S[0]) for key, S in solution[1].items())
                self.decompositions = dict((key, S[1:]) for key, S in solution[1].items())
                success = True

        if propagate_to_parent:
            self.propagate_to_parent(recursive=True)

        return success