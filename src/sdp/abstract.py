from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Union, Callable, Optional, Any

from numpy import ndarray
import numpy as np
from sympy import MutableDenseMatrix as Matrix
from sympy import MatrixBase, Expr, Rational
import sympy as sp

from .backend import SDPBackend
from .rationalize import rationalize_and_decompose
from .utils import is_empty_matrix, Mat2Vec, congruence_with_perturbation

Decomp = Dict[str, Tuple[Matrix, Matrix, List[Rational]]]
Objective = Tuple[str, Union[Expr, Callable[[SDPBackend], Any]]]
Constraint = Callable[[SDPBackend], Any]
MinEigen = Union[float, int, tuple, Dict[str, Union[float, int, tuple]]]


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


class SDPProblemBase(ABC):
    """
    Common class for SDP problems in both primal and dual forms.
    """
    is_dual = False
    is_primal = False
    def __init__(self, *args, **kwargs) -> None:
        # associated with the PSD matrices
        self.y = None
        self.S = None
        self.decompositions = None

        # record the numerical solutions
        self._ys = []

    def _init_space(self, arg: Union[list, Dict], argname: str) -> Dict[str, Matrix]:
        """
        Init self from a list or a dict and write the result to self.argname.
        """
        if isinstance(arg, list):
            keys = ['S_%d'%i for i in range(len(arg))]
            arg = dict(zip(keys, arg))
        elif isinstance(arg, dict):
            keys = list(arg.keys())
        else:
            raise TypeError(f"The {argname} should be a dict or a list of parameters.")
        setattr(self, argname, arg)
        return arg

    @abstractmethod
    def keys(self, filter_none: bool = False) -> List[str]: ...

    @property
    @abstractmethod
    def dof(self) -> int: ...

    def get_size(self, key: str) -> int:
        return Mat2Vec.length_of_mat(self._x0_and_space[key][1].shape[0])

    @property
    def size(self) -> Dict[str, int]:
        return {key: self.get_size(key) for key in self.keys()}

    def __repr__(self) -> str:
        return "<%s dof=%d size=%s>"%(self.__class__.__name__, self.dof, self.size)

    def __str__(self) -> str:
        return self.__repr__()

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

    @abstractmethod
    def S_from_y(self, y: Optional[Union[Matrix, ndarray, Dict]] = None) -> Dict[str, Matrix]:
        
        """
        Given y, compute the symmetric matrices. This is useful when we want to see the
        symbolic representation of the SDP problem.

        This function does not register the result to self.S and PSD is not checked.

        Parameters
        ----------
        y : Optional[Union[Matrix, ndarray]]
            The generating vector. If None, it uses a symbolic vector.

        Returns
        ----------
        S : Dict[str, Matrix]
            The symmetric matrices that SDP requires to be positive semidefinite.
        """
        ...

    def mats(self, *args, **kwargs) -> Dict[str, Matrix]:
        """
        Alias of `S_from_y`.
        """
        return self.S_from_y(*args, **kwargs)

    def register_y(self, y: Union[Matrix, ndarray, Dict], perturb: bool = False, propagate_to_parent: bool = True) -> None:
        """
        Manually register a solution y to the SDP problem.

        Parameters
        ----------
        y : Union[Matrix, ndarray, Dict]
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
    
    def rationalize_combine(
            self,
            ys: List[ndarray],
            mat_func: Callable[[Matrix], Dict[str, Matrix]],
            projection: Optional[Callable[[Matrix], Matrix]] = None,
            verbose: bool = False,
        ) ->  Optional[Tuple[Matrix, Decomp]]:
        """
        Linearly combine all numerical solutions [y] to produce a rational solution.

        Parameters
        ----------
        y : ndarray
            Numerical solution y.
        mat_func : Callable[[Matrix], Dict[str, Matrix]]
            Given a rationalized vector `y`, return a dictionary of matrices
            that needs to be PSD.
        projection : Optional[Callable[[Matrix], Matrix]]
            The projection function. If not None, we project `y` to the feasible
            region before checking the PSD property.
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

        S_numer = mat_func(Matrix(y))
        if all(_.is_positive_semidefinite for _ in S_numer.values()):
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

        decomp = rationalize_and_decompose(y, mat_func=mat_func, projection=projection,
            try_rationalize_with_mask = False, lcm = 1260, times = times
        )
        return decomp

    def propagate_to_parent(self, *args, **kwargs) -> None:
        # this method should be implemented in the TransformMixin
        ...

    def get_last_child(self) -> 'SDPProblemBase':
        # this method should be implemented in the TransformMixin
        return self

    @abstractmethod
    def _get_defaulted_configs(self) -> List[List[Any]]:
        """
        Get the default configurations of the SDP problem.
        """
        ...

    @abstractmethod
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
        ) -> Optional[Tuple[Matrix, Decomp]]:
        """
        Attempt multiple numerical SDP trials given multiple solver configurations.

        Parameters
        ----------
        list_of_objective : List[Objective]
            The list of objectives.
        list_of_constraints : List[List[Constraint]]
            The list of constraints.
        list_of_min_eigen : List[MinEigen]
            The list of minimum eigenvalues for the symmetric matrices.
        solver : Optional[str]
            The name of the solver. Refer to sdp.backend.caller._BACKENDS for the list of available solvers.
        allow_numer : int
            Whether to accept numerical solution. 
            If 0, then it claims failure if the rational feasible solution does not exist.
            If 1, then it accepts a numerical solution if the rational feasible solution does not exist.
            If 2, then it accepts the first numerical solution if rationalization fails.
            If 3, then it accepts the first numerical solution directly. Defaults to 0.
        rationalize_configs : Dict
            The configurations passed to the rationalization function.
        verbose : bool
            If True, print the information of the solving process.
        solver_options : Dict[str, Any]
            The options passed to the SDP backend solver.
        raise_exception : bool
            If True, raise an exception if an error occurs.
        """
        ...

    def solve(self,
            objectives: Union[Objective, List[Objective]] = [],
            constraints: Union[List[Constraint], List[List[Constraint]]] = [],
            min_eigen: Union[MinEigen, List[MinEigen]] = [],
            solver: Optional[str] = None,
            use_default_configs: bool = True,
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
        solver_options : Dict[str, Any]
            The options passed to the SDP backend solver.
        raise_exception : bool
            If True, raise an exception if an error occurs.

        Returns
        ----------
        bool
            Whether the problem is solved. If True, the result can be accessed by
            SDPProblem.y and SDPProblem.S and SDPProblem.decompositions.
        """
        
        if solve_child:
            child: SDPProblemBase = self.get_last_child()
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
                    solver_options = solver_options,
                    raise_exception = raise_exception
                )

        configs = _align_iters(
            [objectives, constraints, min_eigen],
            [tuple, list, (float, int, tuple, dict)]
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
            *configs, solver=solver, allow_numer = allow_numer, verbose = verbose,
            solver_options = solver_options, raise_exception = raise_exception
        )

        if solution is not None:
            # register the solution
            self.y = solution[0]
            self.S = dict((key, S[0]) for key, S in solution[1].items())
            self.decompositions = dict((key, S[1:]) for key, S in solution[1].items())

        if propagate_to_parent:
            self.propagate_to_parent()

        return (solution is not None)