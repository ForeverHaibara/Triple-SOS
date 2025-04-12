from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Union, Callable, Optional, Any

from numpy import ndarray
import numpy as np
from sympy import MutableDenseMatrix as Matrix
from sympy import MatrixBase, Expr, Rational
import sympy as sp

from .arithmetic import sqrtsize_of_mat, is_empty_matrix, congruence_with_perturbation
from .backend import SDPBackend
from .ipm import SDPRationalizeError
from .rationalize import (
    rationalize_and_decompose, RationalizeWithMask, RationalizeSimultaneously, IdentityRationalizer
)
from .utils import exprs_to_arrays, merge_constraints

Decomp = Dict[Any, Tuple[Matrix, Matrix, List[Rational]]]
Objective = Union[Expr, ndarray, Callable[[SDPBackend], Any]]
Constraint = Callable[[SDPBackend], Any]
MinEigen = Union[float, int, tuple, Dict[Any, Union[float, int, tuple]]]



class SDPProblemBase(ABC):
    """
    Common class for SDP problems in both primal and dual forms.
    """
    is_dual = False
    is_primal = False
    y = None
    S = None
    decompositions = None
    def __init__(self, *args, **kwargs) -> None:
        # record the numerical solutions
        self._ys = []

    def _init_space(self, arg: Union[list, Dict], argname: str) -> Dict[Any, Matrix]:
        """
        Init self from a list or a dict and write the result to self.argname.
        """
        if isinstance(arg, list):
            # keys = ['S_%d'%i for i in range(len(arg))]
            keys = list(range(len(arg)))
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
        return sqrtsize_of_mat(self._x0_and_space[key][0])

    @property
    def size(self) -> Dict[Any, int]:
        return {key: self.get_size(key) for key in self.keys()}

    def __repr__(self) -> str:
        return "<%s dof=%d size=%s>"%(self.__class__.__name__, self.dof, self.size)

    def __str__(self) -> str:
        return self.__repr__()

    def _standardize_mat_dict(self, mat_dict: Dict[Any, Matrix]) -> Dict[Any, Matrix]:
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
    def S_from_y(self, y: Optional[Union[Matrix, ndarray, Dict]] = None) -> Dict[Any, Matrix]:
        
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
        S : Dict[Any, Matrix]
            The symmetric matrices that SDP requires to be positive semidefinite.
        """

    def mats(self, *args, **kwargs) -> Dict[Any, Matrix]:
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
            S_numer = [np.array(mat).astype(np.float64) for mat in S.values()]
            S_eigen = [np.min(np.linalg.eigvalsh(mat)) if mat.size else 0 for mat in S_numer]
            print(f'Minimum Eigenvalues = {S_eigen}')
        return rationalize_and_decompose(y, mat_func=self.S_from_y, projection=self.project, **kwargs)

    @abstractmethod
    def _solve_numerical_sdp(self,
        objective: Matrix,
        constraints: List[Tuple[Matrix, Matrix, str]] = [],
        solver: Optional[str] = None,
        kwargs: Dict[Any, Any] = {}
    ) -> Optional[ndarray]:
        """
        Internal interface to solve a single numerical SDP by calling backends.
        """

    def solve_obj(self,
        objective: Objective,
        constraints: List[Constraint] = [],
        solver: Optional[str] = None,
        solve_child: bool = True,
        propagate_to_parent: bool = True,
        kwargs: Dict[Any, Any] = {}
    ) -> Optional[Matrix]:
        """
        Solve the SDP problem with a given objective and constraints.
        """
        obj = exprs_to_arrays(None, self.free_symbols, [objective], dtype=np.float64)[0]
        cons = exprs_to_arrays(None, self.free_symbols, constraints, dtype=np.float64)

        cons = [(_[0], _[1], '>') if len(_) == 2 else _ for _ in cons]
        ineq_lhs, ineq_rhs, eq_lhs, eq_rhs = merge_constraints(cons, self.dof)

        if solve_child:
            self = self.get_last_child()
            obj = self.propagate_affine_to_child(obj[0], obj[1])
            ineq_lhs, ineq_rhs = self.propagate_affine_to_child(ineq_lhs, ineq_rhs)
            eq_lhs, eq_rhs = self.propagate_affine_to_child(eq_lhs, eq_rhs)

        cons = [(ineq_lhs, ineq_rhs, '>'), (eq_lhs, eq_rhs, '==')]

        y = self._solve_numerical_sdp(objective=obj[0], constraints=cons, solver=solver, kwargs=kwargs)
        if y is not None and propagate_to_parent:
            self.propagate_to_parent(recursive = True)
        return y

    @abstractmethod
    def solve(self,
        solver: Optional[str] = None,
        verbose: bool = False,
        solve_child: bool = True,
        propagate_to_parent: bool = True,
    ) -> bool:
        """
        Interface for solving the SDP problem.

        Parameters
        ----------
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

        Returns
        ----------
        bool
            Whether the problem is solved. If True, the result can be accessed by
            SDPProblem.y and SDPProblem.S and SDPProblem.decompositions.
        """