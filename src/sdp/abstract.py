from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Union, Callable, Optional, Any

from numpy import ndarray
import numpy as np
from sympy import MutableDenseMatrix as Matrix
from sympy import MatrixBase, Expr, Rational
import sympy as sp

from .backend import SDPBackend
from .ipm import SDPRationalizeError
from .rationalize import (
    rationalize_and_decompose, RationalizeWithMask, RationalizeSimultaneously, IdentityRationalizer
)
from .utils import (
    is_empty_matrix, Mat2Vec, congruence_with_perturbation, align_iters, IteratorAlignmentError
)

Decomp = Dict[str, Tuple[Matrix, Matrix, List[Rational]]]
Objective = Union[Expr, ndarray, Callable[[SDPBackend], Any]]
Constraint = Callable[[SDPBackend], Any]
MinEigen = Union[float, int, tuple, Dict[str, Union[float, int, tuple]]]



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
        return Mat2Vec.length_of_mat(self._x0_and_space[key][0].shape[0])

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

    def mats(self, *args, **kwargs) -> Dict[str, Matrix]:
        """
        Alias of `S_from_y`.
        """
        return self.S_from_y(*args, **kwargs)

    def project(self, y: Matrix) -> Matrix:
        """
        Project the vector `y` to the feasible region.
        """
        return y

    def register_y(self,
            y: Union[Matrix, ndarray, Dict],
            project: bool = True,
            perturb: bool = False,
            propagate_to_parent: bool = True
        ) -> None:
        """
        Manually register a solution y to the SDP problem.

        Parameters
        ----------
        y : Union[Matrix, ndarray, Dict]
            The solution to the SDP problem.
        project : bool
            For primal forms, if project == True, it must project the solution to the feasible
            region of linear constraints. But PSD constraints are not checked.
            This is ignored for dual problems.
        perturb : bool
            If perturb == True, it must return the result by adding a small perturbation * identity
            to the matrices. This is useful when the given y is numerical.
        propagate_to_parent : bool
            If True, propagate the solution to the parent SDP problem.
        """
        if len(y) == 0 and self.dof != 0:
            raise ValueError("No solution is given to be registered.")
        y2 = self.project(y)
        if (not (y2 is y)) and not project:
            # FIXME for numpy array
            raise ValueError("The vector y is not feasible by the equality constraints."
                             "Use project=True to project the approximated solution to the feasible region.")
        y = y2
  
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

    def rationalize(self, y: ndarray, verbose = False, **kwargs) -> Optional[Tuple[Matrix, Decomp]]:
        """
        Rationalize a NumPy vector `y`. If verbose == True, display the numerical eigenvalues
        before rationalization.
        """
        if y is None: return None
        if len(y) == 0 and self.dof != 0:
            # rationalize an empty vector
            return None
        if verbose:
            S = self.S_from_y(y)
            S_numer = [np.array(mat).astype('float64') for mat in S.values()]
            S_eigen = [np.min(np.linalg.eigvalsh(mat)) if mat.size else 0 for mat in S_numer]
            print(f'Minimum Eigenvalues = {S_eigen}')
        return rationalize_and_decompose(y, mat_func=self.S_from_y, projection=self.project, **kwargs)


    @abstractmethod
    def _solve_numerical_sdp(self,
            objective: Objective,
            constraints: List[Constraint] = [],
            min_eigen: MinEigen = 0,
            scaling: float = 6.,
            solver: Optional[str] = None,
            verbose: bool = False,
            solver_options: Dict[str, Any] = {},
            raise_exception: bool = False
        ) -> Optional[ndarray]:
        """
        Solve a single numerical SDP.
        """

    def _solve_from_multiple_configs(self,
            list_of_objective: List[Objective] = [],
            list_of_constraints: List[List[Constraint]] = [],
            list_of_min_eigen: List[MinEigen] = [],
            scaling: float = 6.,
            solver: Optional[str] = None,
            allow_numer: int = 0,
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
        verbose : bool
            If True, print the information of the solving process.
        solver_options : Dict[str, Any]
            The options passed to the SDP backend solver.
        raise_exception : bool
            If True, raise an exception if an error occurs.
        """
        num_sol = len(self._ys)

        for obj, con, eig in zip(list_of_objective, list_of_constraints, list_of_min_eigen):
            # iterate through the configurations
            y = self._solve_numerical_sdp(objective=obj, constraints=con, min_eigen=eig, scaling=scaling,
                solver=solver, solver_options=solver_options, raise_exception=raise_exception
            )
            if y is not None:
                self._ys.append(y)

                def _force_return(self: SDPProblemBase, y):
                    self.register_y(y, perturb = True, propagate_to_parent = False)
                    _decomp = dict((key, (s, d)) for (key, s), d in zip(self.S.items(), self.decompositions.values()))
                    return y, _decomp

                if allow_numer >= 3:
                    # force to return the numerical solution
                    return _force_return(self, y)

                decomp = self.rationalize(y, verbose=verbose, #check_pretty=False,
                            rationalizers=[RationalizeWithMask(), RationalizeSimultaneously([1,1260,1260**3])])
                if decomp is not None:
                    return decomp

                if allow_numer == 2:
                    # force to return the numerical solution if rationalization fails
                    return _force_return(self, y)

        if len(self._ys) > num_sol:
            # new numerical solution found
            decomp = self.rationalize(np.array(self._ys).mean(axis=0), verbose=verbose, check_pretty=False,
                        rationalizers=[RationalizeWithMask(), RationalizeSimultaneously()])
            if decomp is not None:
                return decomp

            if allow_numer == 1:
                y = self._ys[-1]
                return self.rationalize(y, verbose=verbose,
                            perturb=True, check_pretty=False, rationalizers=[IdentityRationalizer()])
            else:
                raise SDPRationalizeError(
                    "Failed to find a rational solution despite having a numerical solution."
                )

        return None

    def solve(self,
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
        use_default_configs : int
            Whether to use the default configurations of objective+constraints+min_eigen.
            Defaults to 1.
            * If 0, it only uses the given configurations.
            * If 1, it appends the default configurations to the given configurations if 
            no configurations are given. But when any configuration is given,
            it only uses the given configurations.
            * If 2, it appends the default configurations to the given configurations.
            Defaults to 1.

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
        """
        
        if solve_child:
            child: SDPProblemBase = self.get_last_child()
            if child is not self:
                return child.solve(
                    objective = objective,
                    constraints = constraints,
                    min_eigen = min_eigen,
                    scaling = scaling,
                    solver = solver,
                    allow_numer = allow_numer,
                    verbose = verbose,
                    solve_child = solve_child,
                    propagate_to_parent = propagate_to_parent,
                    solver_options = solver_options,
                    raise_exception = raise_exception
                )

        try:
            configs = align_iters(
                [objective, constraints, min_eigen],
                [(ndarray, Expr, float, int), list, (float, int, tuple, dict)],
                raise_exception = True
            )
        except IteratorAlignmentError as e:
            raise IteratorAlignmentError("Incompatible lengths of configurations: objectives {}, constraints {}, min_eigen {}.".format(*e.args[1]))

        if use_default_configs:
            if use_default_configs > 1 or (use_default_configs == 1 and len(configs[0]) == 0):
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
            *configs, scaling=scaling, solver=solver, allow_numer = allow_numer, verbose = verbose,
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