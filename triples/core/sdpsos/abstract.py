from collections import OrderedDict
from typing import List, Tuple, Dict, Any, Callable, Optional, Union
from time import time, perf_counter

import numpy as np
from sympy import Expr, Poly, Symbol, Domain, QQ
from sympy.matrices import MutableDenseMatrix as Matrix
from sympy.core.relational import Relational

from .algebra import StateAlgebra, SOSBasis
from ...sdp import SDPProblem
from ...sdp.arithmetic import ArithmeticTimeout, sqrtsize_of_mat, matmul, matadd, solve_csr_linear, rep_matrix_from_dict
from ...sdp.utils import exprs_to_arrays, collect_constraints
from ...sdp.wedderburn import symmetry_adapted_basis

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

    def insert_prefix(self, prefix: Any) -> 'SOSEquationSystem':
        """
        Rename each key by (prefix, key). This is modified in-place.
        """
        self.psd_eqs = OrderedDict(((prefix, key), value) for key, value in self.psd_eqs.items())
        self.linear_eqs = OrderedDict(((prefix, key), value) for key, value in self.linear_eqs.items())
        return self

    def dof(self) -> int:
        return sum(x.shape[1] for x in self.psd_eqs.values()) +\
            sum(x.shape[1] for x in self.linear_eqs.values()) +\
            self.parameter_eq.shape[1]

    def vstack(*args) -> 'SOSEquationSystem':
        """
        Stack multiple SOSEquationSystem vertically. Be sure no keys collide.
        """
        if len(args) == 0:
            raise ValueError('At least one argument is required.')

        rhs = Matrix.vstack(*[_.rhs for _ in args])
        parameter_eq = Matrix.vstack(*[_.parameter_eq for _ in args])


        psd_eqs = OrderedDict()
        linear_eqs = OrderedDict()
        offset = 0
        def _add_offset_to_row(m: Matrix, offset: int) -> Matrix:
            # add an offset to each row index of m
            rep = m._rep.rep.to_sdm()
            if offset:
                rep = {k+offset: v for k, v in rep.items()}
            return rep_matrix_from_dict(rep, (rhs.shape[0], m.shape[1]), m._rep.domain)
        for arg in args:
            psd_eqs.update(OrderedDict(
                ((key, _add_offset_to_row(value, offset)) for key, value in arg.psd_eqs.items())))
            linear_eqs.update(OrderedDict(
                ((key, _add_offset_to_row(value, offset)) for key, value in arg.linear_eqs.items())))
            offset += arg.rhs.shape[0]


        psd_offset = 0
        lin_offset = 0
        equal_entries = []
        for i in range(len(args)):
            psd_size = sum(value.shape[1] for key, value in args[i].psd_eqs.items())
            lin_size = sum(value.shape[1] for key, value in args[i].linear_eqs.items())
            equal_entries += [[x + psd_offset if x < psd_size else x + lin_offset for x in y]
                                for y in args[i].equal_entries]
            psd_offset += psd_size
            lin_offset += lin_size

        return SOSEquationSystem(rhs, psd_eqs=psd_eqs, linear_eqs=linear_eqs,
                parameter_eq=parameter_eq, equal_entries=equal_entries)


class SOSElement:
    """
    Highly abstract base class for SOS elements. SOS elements must be associated with a
    SDPProblem instance and store linear parameters in the problem.
    SOSElement contains methods to construct and solve the SOS problem.
    """

    TERM_SPARSITY = 1

    _sdp = None
    _parameters = None
    _parameter_space = None

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} at {hex(id(self))}>'

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
        time_limit: Optional[float] = None,
        kwargs: Dict[Any, Any] = {}
    ) -> Optional[Matrix]:
        """
        Solve the SOS problem. Arguments are passed to `SDPProblem.solve`.
        """
        end_time = perf_counter() + time_limit if isinstance(time_limit, (int, float)) else None
        time_limit = ArithmeticTimeout.make_checker(time_limit)
        if self._sdp is None:
            self.construct(verbose=verbose, time_limit=time_limit)
        time_limit()
        rest_time = end_time - perf_counter() if end_time is not None else time_limit

        obj = exprs_to_arrays([objective], self._parameters, dtype=np.float64)[0][0]
        cons = exprs_to_arrays(constraints, self._parameters, dtype=np.float64)
        ineq_lhs, ineq_rhs, eq_lhs, eq_rhs = collect_constraints(cons, len(self._parameters))
        x0, space = self._parameter_space
        time_limit()

        # ineq_lhs * (x0 + space * y) >= ineq_rhs
        obj      = matmul(obj, space)
        ineq_lhs = matmul(ineq_lhs, space)
        ineq_rhs = matadd(ineq_rhs, -matmul(ineq_lhs, x0))
        eq_lhs   = matmul(eq_lhs, space)
        eq_rhs   = matadd(eq_rhs, -matmul(eq_lhs, x0))
        time_limit()
        return self.sdp.solve_obj(obj, [(ineq_lhs, ineq_rhs, '>'), (eq_lhs, eq_rhs, '==')],
                solver=solver, verbose=verbose, time_limit=rest_time, kwargs=kwargs)

    def solve(self,
        solver: Optional[str] = None,
        solve_child: bool = True,
        propagate_to_parent: bool = True,
        verbose: bool = False,
        time_limit: Optional[float] = None,
        allow_numer: int = 0,
        kwargs: Dict[Any, Any] = {},
    ) -> Optional[Matrix]:
        """
        Solve the SOS problem. Arguments are passed to `SDPProblem.solve`.
        """
        end_time = perf_counter() + time_limit if isinstance(time_limit, (int, float)) else None
        time_limit = ArithmeticTimeout.make_checker(time_limit)
        if self._sdp is None:
            self.construct(verbose=verbose, time_limit=time_limit)
        time_limit()
        rest_time = end_time - perf_counter() if end_time is not None else time_limit
        return self.sdp.solve(solver=solver, solve_child=solve_child,
                propagate_to_parent=propagate_to_parent, verbose=verbose,
                time_limit=rest_time, allow_numer=allow_numer, kwargs=kwargs)

    def as_solution(self, *args, **kwargs):
        raise NotImplementedError

    def _get_parameters(self) -> List[Symbol]:
        """
        Infer all default parameters in the problem. For example, if we assume the polynomial

            x^2-u*x*y+v*y^2 is a sum-of-squares in (x,y),

        then the parameters are [u,v]. For AtomSOSElement, it should be implemented
        as the free symbols of the polynomial (with generators removed). For JointSOSElement,
        it should be implemented as the union of parameters of all children.
        """
        raise NotImplementedError

    def _get_parameter_eq(self, parameters: List[Symbol]=[]) -> Matrix:
        """
        Get how each parameter contributes to the coefficients of the vector representation
        of the polynomial. For example, if we assume the polynomial

            x^2-u*x*y+v*y^2 is a sum-of-squares in (x,y),

        then it can also be written as:

            x^2 = SOS + u*(x*y) + v*(-y^2).

        The contributions of parameters u,v to the equation system are the vector representations
        of x*y and -y^2. For AtomSOSElement, it should be implemented as the negative
        vector representation of the coefficient of the parameter, and NONLINEAR PARAMETERS
        ARE EXPECTED TO THROW AN ERROR. For JointSOSElement, it should be implemented as the
        vstack of contributions of parameters of all children.
        """
        raise NotImplementedError

    def _get_equation_system(self, parameters: List[Symbol]=[], domain=None) -> SOSEquationSystem:
        raise NotImplementedError

    def _insert_prefix(self, prefix: Any):
        raise NotImplementedError

    def _post_construct(self, verbose: bool = False, time_limit: Optional[Union[Callable, float]] = None, **kwargs):
        """
        Post-construct the SDPProblem instance. It might involve any operations
        that reduce the size of the SDPProblem or might apply transformations
        on the SDPProblem.
        """
        return

    def construct(self,
        parameters: Union[List[Symbol], bool] = True,
        wedderburn: bool = True,
        verbose: bool = False,
        time_limit: Optional[Union[Callable, float]] = None
    ) -> SDPProblem:
        """
        Actually construct the SDPProblem instance from the problem data. It
        might take some time for large-scale problems, so it is not initialized
        in the __init__ method.

        The `construct` method should setup `_sdp`, `_parameters` and `_parameter_space`.
        If it is a JointSOSElement, it should also setup the attributes for its children.

        Parameters
        -----------
        parameters: Union[List[Symbol], bool]
            All linear parameters in the problem will be converted to variables
            when solving the SDPProblem. If True, all free symbols in the polynomial
            will be used as parameters.
        wedderburn: bool
            If True, use Wedderburn decomposition to reduce the size of the SDPProblem.
        verbose: bool
            If True, print the construction process.
        time_limit: Optional[Union[Callable, float]]
            Try to raise the ArithmeticTimeout Exception when timeout is detected.
            If callable, it should be a function to check timeout and raise the Exception.
        TODO: add a numer option
        """
        raise NotImplementedError


class AtomSOSElement(SOSElement):
    """
    Represent a single sum-of-squares instance given some state algebra.
    Multiple AtomSOSElements can be unioned by JointSOSElement based on
    shared parameters.

    General AtomSOSElements are solving problems in the following form:

        sum(s(Gi * ui^2) for i in range(n)) + sum(s(Hj * vj) for j in range(m)) = Poly

    Here Gi >= 0 are inequality constraints, also known as the generators of
    the quadratic module, and Hj == 0 are equality constraints, also
    known as the generators of the ideal. The operator "s" is a state operator,
    which is a positive, linear functional, e.g., an identity operator, a cyclic-sum
    operator, a moment (integral) operator or a trace operator.
    See more details about the state operator in `.algebra.state_algebra`.

    The problem is solved by converting to a SDP problem. More detailedly,
    sums of ui^2 are represented by positive semidefinite matrices, while vj are represented
    by vectors. We then compare the coefficients of the left and right-hand-sides of
    the equation, which is a problem to solve a system of linear equations.

    The linear equations of an AtomSOSElement is determined by the right-hand-side
    (the vector representation of `poly`), and also the localizing matrix of the bases
    (how each entry contributes to the coefficients of the vector representation). An
    AtomSOSElement is equipped with `._get_equation_system` method to build the
    equation system and the system will be solved in `.construct` method to form the SDP
    problem.
    """
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
        # no-parameter case should be handled separately
        if len(parameters) == 0:
            rhs = arraylize(self.poly)
            return rhs, Matrix.zeros(rhs.shape[0], 0)

        domain, gens = self.poly.domain, self.poly.gens
        poly = self.poly.as_poly(*parameters)
        if poly.total_degree() > 1:
            raise ValueError(f"Unable to handle nonlinear terms {poly.LM()} in the polynomial."
                                " Set parameters to False or to a tuple of linear parameters.")

        # get the coefficients of the polynomial in the parameters
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

    def _post_construct(self,
        wedderburn: bool = True,
        verbose = False,
        time_limit = None,
        **kwargs
    ):
        time_limit = ArithmeticTimeout.make_checker(time_limit)
        if wedderburn:
            sa_bases = {}
            for key, qb in self._qmodule_bases.items():
                G, act = qb.get_symmetry_representation()
                dr = symmetry_adapted_basis(G, act)
                if len(dr) > 1:
                    # only decompose when there are at least 2 blocks
                    sa_bases[(self, key)] = Matrix.hstack(*dr)

            if len(sa_bases):
                self.sdp.constrain_congruence(sa_bases, time_limit=time_limit)

            # print(self.sdpp.get_block_structures())
            self.sdpp.constrain_block_structures()

        self.sdpp.constrain_zero_diagonals(time_limit=time_limit)


    def _get_equation_system(self, parameters: List[Symbol]=[], domain=None) -> SOSEquationSystem:
        if self._qmodule_bases is None or self._ideal_bases is None:
            # infer default bases if not given
            self._qmodule_bases, self._ideal_bases = self.algebra.infer_bases(
                self.poly, self._qmodule, self._ideal)

        # call each basis for localizing matrix
        # for q in self._qmodule_bases.values():
        #     domain = domain.unify(q.qmodule.domain)
        # for i in self._ideal_bases.values():
        #     domain = domain.unify(i.ideal.domain)

        rhs, parameter_eq = self._get_parameter_eq(parameters)
        qmodule_eqs = {k: b.localizing_matrix(domain) for k, b in self._qmodule_bases.items()}
        ideal_eqs = {k: b.localizing_matrix(domain) for k, b in self._ideal_bases.items()}

        equal_entries = []
        offset = 0
        for k, b in self._qmodule_bases.items():
            equal_entries.extend([[_+offset for _ in group] for group in b.get_equal_entries()])
            offset += len(b)**2
        for k, b in self._ideal_bases.items():
            equal_entries.extend([[_+offset for _ in group] for group in b.get_equal_entries()])
            offset += len(b)

        eq = SOSEquationSystem(rhs, qmodule_eqs, ideal_eqs, parameter_eq=parameter_eq, equal_entries=equal_entries)
        return eq.insert_prefix(self)

    def construct(self,
        parameters: Union[List[Symbol], bool] = True,
        wedderburn: bool = True,
        verbose: bool = False,
        time_limit: Optional[Union[Callable, float]] = None,
    ) -> SDPProblem:
        time_limit = ArithmeticTimeout.make_checker(time_limit)

        # infer linear parameters if not given
        if parameters is True:
            parameters = self._get_parameters()
        elif parameters is False:
            parameters = []


        ######################################################################
        #                    Form the equation system
        ######################################################################
        time0 = time()
        eqs = self._get_equation_system(parameters=parameters)

        stack_psd = Matrix.hstack(*eqs.psd_eqs.values())
        A = Matrix.hstack(stack_psd, *eqs.linear_eqs.values(), eqs.parameter_eq)
        rhs = eqs.rhs

        if verbose:
            print(f"Time for building coefficient equations : {time() - time0:.6f} seconds.")
        time_limit()

        ######################################################################
        #           Solve the equation system to build SDPProblem
        ######################################################################
        time0 = time()
        sdp, (x0, space) = SDPProblem.from_equations(A, rhs,
            splits = {key: sqrtsize_of_mat(value.shape[1]) for key, value in eqs.psd_eqs.items()},
            add_force_zeros = True,
            equal_entries = eqs.equal_entries, add_equal_entries = True,
            time_limit = time_limit
        )
        if verbose:
            print(f"Time for solving coefficient equations  : {time() - time0:.6f} seconds. Dof = {sdp.dof}")
        time_limit()


        ######################################################################
        #            Write variables to corresponding attributes
        ######################################################################
        self._sdp = sdp

        offset = 0
        self._ideal_space = {}
        for key, b in self._ideal_bases.items():
            self._ideal_space[key] = (x0[offset:offset+len(b),:], space[offset:offset+len(b),:])
            offset += len(b)
        self._parameters = list(parameters)
        self._parameter_space = (x0[offset:,:], space[offset:,:])
        time_limit()
        self._post_construct(wedderburn=wedderburn, verbose=verbose, time_limit=time_limit)

        if verbose:
            self._sdp.print_graph(short=2)

        return sdp


class JointSOSElement(SOSElement):
    """
    JointSOSElement collects mutiple SOSElements by shared parameters, and
    solves them together.

    Examples
    --------

    ### Multiple sum-of-squares

    Consider the problem from the documentation of Yalmip,
    https://yalmip.github.io/tutorial/sumofsquaresprogramming/,
    min {t} s.t. t*(1+x*y)^2-x*y+(1-y)^2 and (1-x*y)^2+x*y+t*(1+y)^2 are both sum-of-squares.
    To solve the problem, we first initialize them separately by SOSPoly and then
    union them by JointSOSElement.

        >>> from triples.core.sdpsos import SOSPoly, JointSOSElement
        >>> from sympy.abc import x, y, t
        >>> p1 = SOSPoly(t*(1+x*y)**2-x*y+(1-y)**2, (x,y), [1])
        >>> p2 = SOSPoly((1-x*y)**2+x*y+t*(1+y)**2, (x,y), [1])
        >>> p = JointSOSElement([p1, p2])

    After unioning, we can solve the problem by calling `.solve_obj` method on the JointSOSElement.

        >>> _ = p.solve_obj(t)
        >>> p.as_params()  # doctest: +SKIP
        {t: 0.249999997333379}
        >>> p1.as_solution().solution.n(3)  # doctest: +SKIP
        2.15*(0.0988*x*y + 0.652*y - 0.751)**2 + 0.349*(0.811*x*y - 0.491*y - 0.32)**2
        >>> p2.as_solution().solution.n(3)  # doctest: +SKIP
        0.17*(0.181*x*y - 0.936*y + 0.3)**2 + 1.67*(0.593*x*y - 0.14*y - 0.793)**2 + 0.662*(0.784*x*y + 0.322*y + 0.53)**2


    ### Finding Lyapunov functions

    We next illustrate an example of finding Lyapunov functions from
    "Structured Semidefinite Programs and Semialgebraic Geometry Methods
    in Robustness and Optimization" by Pablo A. Parrilo. A system of two variables
    (x, y) satisfies:

        dx / dt = -x - 2*y^2
        dy / dt = -y - x*y - 2*y^3

    We wish to find V(x,y) such that V(0,0)=0, V(x,y)>=0 and dV/dt <= 0. We also assume V(x,y)
    to be quadratic. We can assume V does not have linear terms to ensure its nonnegativity
    around (0,0). Also, we can assume the x^2 term has coefficient 1 up to a scaling. To solve
    for such V, we define the V as a symbolic-coefficient polynomial in (x,y), and assume
    V and -dV/dt are both sum-of-squares polynomials using `SOSPoly`. We then union them by
    `JointSOSElement` and solve for a set of parameters.

        >>> from sympy import Symbol
        >>> V = sum([Symbol('c_{%d,%d}'%(i,j))*x**i*y**j for i in range(3) for j in range(3) if 1<i+j<3])
        >>> V = V.subs({Symbol('c_{2,0}'):1})
        >>> V
        c_{0,2}*y**2 + c_{1,1}*x*y + x**2
        >>> dV = V.diff(x)*(-x-2*y**2)+V.diff(y)*(-y-x*y-2*y**3)
        >>> p1 = SOSPoly(V, (x,y), [1])
        >>> p2 = SOSPoly(-dV, (x,y), [1])
        >>> p = JointSOSElement([p1,p2])
        >>> _ = p.solve()
        >>> p.as_params()  # doctest: +SKIP
        {c_{1,1}: 0, c_{0,2}: 2}
        >>> p1.as_solution().solution  # doctest: +SKIP
        x**2 + 2*y**2
        >>> p2.as_solution().solution  # doctest: +SKIP
        4*y**2 + 2*(x + 2*y**2)**2
    """
    sos_elements: List[SOSElement] = None
    def __init__(self, sos_elements: List[SOSElement]):
        self.sos_elements = sos_elements

    def _get_parameters(self) -> List[Symbol]:
        parameters = set()
        for sos in self.sos_elements:
            parameters.update(set(sos._get_parameters()))
        return list(parameters)

    def _get_parameter_eq(self, parameters: List[Symbol]=[]) -> Matrix:
        parameter_eq = []
        for sos in self.sos_elements:
            parameter_eq.append(sos._get_parameter_eq(parameters))
        return Matrix.vstack(*parameter_eq)

    def _get_equation_system(self, parameters: List[Symbol]=[], domain=None) -> SOSEquationSystem:
        eqs = []
        for sos in self.sos_elements:
            eqs.append(sos._get_equation_system(parameters, domain))
        if len(eqs) == 0:
            raise ValueError("SOS elements cannot be empty.")
        return SOSEquationSystem.vstack(*eqs)

    def _post_construct(self, verbose: bool = False, time_limit: Optional[Union[Callable, float]] = None, **kwargs):
        time_limit = ArithmeticTimeout.make_checker(time_limit)
        for sos in self.sos_elements:
            sos._post_construct(verbose=verbose, time_limit=time_limit, **kwargs)

    def construct(self,
        parameters: Union[List[Symbol], bool] = True,
        wedderburn: bool = True,
        verbose: bool = False,
        time_limit: Optional[Union[Callable, float]] = None
    ) -> SDPProblem:

        # infer linear parameters if not given
        if parameters is True:
            parameters = self._get_parameters()
        elif parameters is False:
            parameters = []


        ######################################################################
        #                    Form the equation system
        ######################################################################
        time0 = time()
        eqs = self._get_equation_system(parameters=parameters)

        stack_psd = Matrix.hstack(*eqs.psd_eqs.values())
        A = Matrix.hstack(stack_psd, *eqs.linear_eqs.values(), eqs.parameter_eq)
        rhs = eqs.rhs

        if verbose:
            print(f"Time for building coefficient equations : {time() - time0:.6f} seconds.")


        ######################################################################
        #           Solve the equation system to build SDPProblem
        ######################################################################
        time0 = time()
        sdp, (x0, space) = SDPProblem.from_equations(A, rhs,
            splits = {key: sqrtsize_of_mat(value.shape[1]) for key, value in eqs.psd_eqs.items()},
            add_force_zeros = True,
            equal_entries = eqs.equal_entries, add_equal_entries = True,
        )
        if verbose:
            print(f"Time for solving coefficient equations  : {time() - time0:.6f} seconds. Dof = {sdp.dof}")


        ######################################################################
        #            Write variables to corresponding attributes
        ######################################################################
        self._sdp = sdp

        offset = 0
        for _, b in eqs.linear_eqs.items():
            offset += len(b)
        self._parameters = list(parameters)
        self._parameter_space = (x0[offset:,:], space[offset:,:])


        # also write the info to children
        def recur_write(sos):
            sos._sdp = sdp
            sos._parameters = self._parameters
            sos._parameter_space = self._parameter_space
            if isinstance(sos, JointSOSElement):
                for child in sos.sos_elements:
                    recur_write(child)
            elif isinstance(sos, AtomSOSElement):
                sos._ideal_space = {}
        recur_write(self)

        offset = 0
        for (sos, key), b in eqs.linear_eqs.items():
            sos._ideal_space[key] = (x0[offset:offset+len(b),:], space[offset:offset+len(b),:])

        self._post_construct(wedderburn=wedderburn, verbose=verbose, time_limit=time_limit)
        if verbose:
            self._sdp.print_graph()

        return sdp
