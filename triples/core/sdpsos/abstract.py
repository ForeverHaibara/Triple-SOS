from typing import List, Tuple, Dict, Any, Callable, Optional, Union
from time import time

import numpy as np
from sympy import Expr, Poly, Symbol, Domain, QQ
from sympy.matrices import MutableDenseMatrix as Matrix
from sympy.core.relational import Relational

from .algebra import StateAlgebra, SOSBasis
from ...sdp import SDPProblem
from ...sdp.arithmetic import sqrtsize_of_mat, matmul, matadd, solve_csr_linear
from ...sdp.utils import exprs_to_arrays, collect_constraints


class SOSEquationSystem:
    rhs: Matrix
    psd_eqs: Dict[Any, Matrix]
    linear_eqs: Dict[Any, Matrix]
    parameter_eq: Matrix
    # force_zeros: List[Tuple[int, int]]
    equal_entries: List[List[int]]

    def __init__(self, rhs, psd_eqs, linear_eqs, parameter_eq, equal_entries=[]):
        self.rhs = rhs
        self.psd_eqs = psd_eqs
        self.linear_eqs = linear_eqs
        self.parameter_eq = parameter_eq
        self.equal_entries = equal_entries

    def vstack(*args):
        ...

class SOSElement:
    """
    Base class for SOS elements.
    """

    TERM_SPARSITY = 1

    _sdp = None
    _parameters = None
    _parameter_space = None

    @property
    def sdp(self) -> SDPProblem:
        """
        Return the root node of the constructed SDP problem.
        """
        return self._sdp

    @property
    def sdpp(self) -> SDPProblem:
        """
        Return the child node of the SDP problem.
        """
        return self.sdp.get_last_child() if self.sdp is not None else None

    def as_params(self) -> Dict[Symbol, Expr]:
        if self._parameters is None or len(self._parameters) == 0:
            return {}
        x0, space = self._parameter_space
        y = x0 + space @ self.sdp.y
        return dict(zip(self._parameters, y))

    def solve_obj(self,
        objective: Union[Expr, Matrix, List],
        constraints: List[Union[Relational, Expr, Tuple[Matrix, Matrix, str]]] = [],
        solver: Optional[str] = None,
        verbose: bool = False,
        kwargs: Dict[Any, Any] = {}
    ) -> Optional[Matrix]:
        """
        Solve the SOS problem. Arguments are passed to SDPProblem.solve.
        """
        if self._sdp is None:
            self.construct(verbose=verbose)
        obj = exprs_to_arrays([objective], self._parameters, dtype=np.float64)[0][0]
        cons = exprs_to_arrays(constraints, self._parameters, dtype=np.float64)
        ineq_lhs, ineq_rhs, eq_lhs, eq_rhs = collect_constraints(cons, len(self._parameters))
        x0, space = self._parameter_space

        # ineq_lhs * (x0 + space * y) >= ineq_rhs
        obj      = matmul(obj, space)
        ineq_lhs = matmul(ineq_lhs, space)
        ineq_rhs = matadd(ineq_rhs, -matmul(ineq_lhs, x0))
        eq_lhs   = matmul(eq_lhs, space)
        eq_rhs   = matadd(eq_rhs, -matmul(eq_lhs, x0))
        return self.sdp.solve_obj(obj, [(ineq_lhs, ineq_rhs, '>'), (eq_lhs, eq_rhs, '==')],
                solver=solver, verbose=verbose, kwargs=kwargs)

    def solve(self, **kwargs) -> Optional[Matrix]:
        """
        Solve the SOS problem. Arguments are passed to SDPProblem.solve.
        """
        if self._sdp is None:
            self.construct(verbose=kwargs.get('verbose', False))
        return self.sdp.solve(**kwargs)

    def as_solution(self, *args, **kwargs):
        raise NotImplementedError

    def _post_construct(self, verbose: bool = False):
        return

    def construct(self, *args, **kwargs):
        raise NotImplementedError


class AtomSOSElement(SOSElement):
    poly: Poly
    algebra: StateAlgebra
    _qmodule_bases: Dict[Any, SOSBasis] = None
    _ideal_bases: Dict[Any, SOSBasis] = None
    _qmodule: Dict[Any, Poly] = None
    _ideal: Dict[Any, Poly] = None
    _ideal_space: Dict[Any, Tuple[Matrix, Matrix]] = None

    def _get_parameters(self) -> List[Symbol]:
        return list(self.poly.free_symbols - set(self.poly.gens))

    def _get_parameter_eq(self, parameters: List[Symbol]=[]) -> Matrix:
        arraylize = self.algebra.arraylize
        if len(parameters) == 0:
            rhs = arraylize(self.poly)
            return rhs, Matrix.zeros(rhs.shape[0], 0)

        domain, gens = self.poly.domain, self.poly.gens
        poly = self.poly.as_poly(*parameters)
        if poly.total_degree() > 1:
            raise ValueError(f"Unable to handle nonlinear terms {poly.LM()} in the polynomial."
                                " Set parameters to False or to a tuple of linear parameters.")

        onehot = [0]*len(parameters)
        coeff_of_params = [0]*len(parameters)
        for i in range(len(parameters)):
            onehot[i] = 1
            coeff_of_params[i] = poly.coeff_monomial(tuple(onehot)).as_poly(gens)
            onehot[i] = 0
        poly = poly.coeff_monomial(tuple(onehot)).as_poly(gens)

        rhs = arraylize(poly)

        # it is important to flip the sign
        eq = -Matrix.hstack(*[arraylize(coeff) for coeff in coeff_of_params])
        return rhs, eq


    def _get_equation_system(self, parameters: List[Symbol]=[]) -> SOSEquationSystem:
        if self._qmodule_bases is None or self._ideal_bases is None:
            self._qmodule_bases, self._ideal_bases = self.algebra.infer_bases(
                self.poly, self._qmodule, self._ideal)

        rhs, parameter_eq = self._get_parameter_eq(parameters)
        qmodule_eqs = {k: b.localizing_matrix(QQ) for k, b in self._qmodule_bases.items()}
        ideal_eqs = {k: b.localizing_matrix(QQ) for k, b in self._ideal_bases.items()}

        equal_entries = []
        offset = 0
        for k, b in self._qmodule_bases.items():
            equal_entries.extend([[_+offset for _ in group] for group in b.get_equal_entries()])
            offset += len(b)**2
        for k, b in self._ideal_bases.items():
            equal_entries.extend([[_+offset for _ in group] for group in b.get_equal_entries()])
            offset += len(b)

        return SOSEquationSystem(rhs, qmodule_eqs, ideal_eqs, parameter_eq=parameter_eq, equal_entries=equal_entries)

    
    def construct(self,
            parameters: Union[List[Symbol], bool] = True,
            verbose: bool = False
        ) -> SDPProblem:
        # collect the equation system and solve it
        # set the _sdp attribute, parameter_spaces
        if parameters is True:
            parameters = self._get_parameters()
        elif parameters is False:
            parameters = []

        time0 = time()
        eqs = self._get_equation_system(parameters=parameters)

        stack_psd = Matrix.hstack(*eqs.psd_eqs.values())
        A = Matrix.hstack(stack_psd, *eqs.linear_eqs.values(), eqs.parameter_eq)
        rhs = eqs.rhs

        if verbose:
            print(f"Time for building coefficient equations : {time() - time0:.6f} seconds.")


        time0 = time()
        sdp, (x0, space) = parameter_space = SDPProblem.from_equations(A, rhs,
            splits = {key: sqrtsize_of_mat(value.shape[1]) for key, value in eqs.psd_eqs.items()},
            add_force_zeros = True,
            equal_entries = eqs.equal_entries, add_equal_entries = True,
        )
        if verbose:
            print(f"Time for solving coefficient equations  : {time() - time0:.6f} seconds. Dof = {sdp.dof}")


        self._sdp = sdp

        offset = 0
        self._ideal_space = {}
        for key, b in self._ideal_bases.items():
            self._ideal_space[key] = (x0[offset:offset+len(b),:], space[offset:offset+len(b),:])
            offset += len(b)
        self._parameters = list(parameters)
        self._parameter_space = (x0[offset:,:], space[offset:,:])
        self._post_construct(verbose=verbose)

        return sdp


