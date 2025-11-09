from typing import Optional, Tuple, List, Dict, Union, Any
from time import perf_counter
import warnings

import numpy as np
from sympy import Poly, Expr, Symbol
from sympy.combinatorics import PermutationGroup
from sympy.external.importtools import version_tuple
from scipy import __version__ as _SCIPY_VERSION

from .basis import LinearBasis, LinearBasisTangent, LinearBasisTangentEven
from .tangents import prepare_tangents, prepare_inexact_tangents, get_qmodule_list
from .correction import linear_correction, odd_basis_to_even
from .updegree import lift_degree
from .solution import create_linear_sol_from_y_basis
from ..preprocess import ProofNode, SolvePolynomial
from ..shared import homogenize_expr_list
from ...sdp.arithmetic import ArithmeticTimeout
from ...utils import Root, Solution, MonomialManager, clear_polys_by_symmetry


LINPROG_OPTIONS = {
    'method': 'highs-ds' if tuple(version_tuple(_SCIPY_VERSION)) >= (1, 6) else 'simplex',
    'options': {
        'presolve': False,
    }
}

if _SCIPY_VERSION in ('1.15.0','1.15.1','1.15.2'):
    warnings.warn('SciPy 1.15.0-1.15.2 has a bug in linprog that gets stuck with large scale problems.\n'
        'Please upgrade to 1.15.3 or higher or downgrade to 1.14.x.\n'
        'See https://github.com/scipy/scipy/issues/22655 for details.')



class _basis_limit_exceeded(Exception): ...


def _check_basis_limit(basis: List, limit: int = 15000) -> None:
    if len(basis) > limit:
        raise _basis_limit_exceeded


class LinearSOSSolver(ProofNode):
    """
    Solve a constrained polynomial inequality problem using linear programming (LP).

    Linear programming constructs a list of nonegative polynomials as bases
    and tries to find a linear combination of the bases that adds to the target polynomial.
    Linear programming provides a smaller cone than the semidefinite programming (SDP) relaxation,
    but often supports a larger scale of problems.
    """
    default_configs = {
        'verbose': False,
        'linprog_options': LINPROG_OPTIONS,
        'allow_numer': 0,
        'centralize': True,
        'augment_tangents': True,
        'preordering': 'quadratic',
        'quad_diff_order': 8,
        'basis_limit': 15000,
        'lift_degree_limit': 4,
    }


    _transformed_problem = None
    _tangents = None
    _decentralizer = None
    _complexity_models = True
    def _centralize(self, configs):
        """
        Apply an auto scaling on the variables so that one of the roots is (1,1,...,1).
        The new transformed problem overwrites `self._transformed_problem` and `self._tangents`.

        After solving the centralized problem, the solution to the original problem can be obtained by
        `solution.xreplace(self._decentralizer)`.
        """
        problem = self._transformed_problem
        if problem.roots is None:
            return

        poly = problem.expr
        gens = poly.gens 
        roots = [r for r in problem.roots if not r.is_zero]
        symmetry = MonomialManager(len(gens), problem.identify_symmetry())
        if not (len(roots) == 1 and roots[0].is_Rational and symmetry.is_trivial\
                    and any(ri != 1 and ri != 0 for ri in roots[0])):
            return

        centralizer = {gens[i]: roots[0][i]*gens[i] for i in range(len(gens)) if roots[0][i] != 0 and roots[0][i] != 1}
        def ct(x):
            return x.as_expr().xreplace(centralizer).as_poly(gens)
        new_poly = ct(poly)
        new_ineqs = {ct(k): v.xreplace(centralizer) for k, v in problem.ineq_constraints.items()}
        new_eqs = {ct(k): v.xreplace(centralizer) for k, v in problem.eq_constraints.items()}
        new_roots = [Root(tuple(1 if roots[0][i] != 0 else 0 for i in range(len(gens))))]
        new_tangents = [ct(t) for t in self._tangents]

        self._transformed_problem = self.new_problem(new_poly, new_ineqs, new_eqs)
        self._transformed_problem.roots = new_roots
        self._tangents = new_tangents
        self._decentralizer = {gens[i]: gens[i]/roots[0][i] for i in range(len(gens)) if roots[0][i] != 0 and roots[0][i] != 1}

        if configs.get('verbose', False):
            print(f'LinearSOS centralizing {centralizer}')
            # print('Goal         :', new_poly)
            # print('Inequalities :', new_ineqs)
            # print('Equalities   :', new_eqs)
            # print('Roots        :', new_roots)

    def _prepare_qmodule(self, configs: Dict[str, Any]) -> Dict[Poly, Expr]:
        problem = (self._transformed_problem if self._transformed_problem is not None else self.problem)
        poly = problem.expr
        ineq_constraints = problem.ineq_constraints
        qmodule = get_qmodule_list(poly, ineq_constraints,
            all_nonnegative=configs['all_nonnegative'], preordering=configs['preordering'])
        qmodule = clear_polys_by_symmetry(qmodule, poly.gens, configs['symmetry'])
        ArithmeticTimeout.make_checker(configs['time_limit'])()
        return dict(qmodule)

    def _prepare_ideal(self, configs: Dict[str, Any]) -> Dict[Poly, Expr]:
        problem = (self._transformed_problem if self._transformed_problem is not None else self.problem)
        eq_constraints = list(problem.eq_constraints.items())
        ideal = clear_polys_by_symmetry(eq_constraints, problem.expr.gens, configs['symmetry'])
        ArithmeticTimeout.make_checker(configs['time_limit'])()
        return dict(ideal)

    def _prepare_tangents(self, configs):
        problem = (self._transformed_problem if self._transformed_problem is not None else self.problem)
        time_limit = ArithmeticTimeout.make_checker(configs['time_limit'])

        qmodule = configs.get('qmodule', problem.ineq_constraints)
        tangents = list(prepare_tangents(problem, qmodule=qmodule,
            additional_tangents=configs['tangents']).items())
        time_limit()

        if configs['augment_tangents']:
            tangents += list(prepare_inexact_tangents(problem,
                monomial_manager=configs['symmetry'],
                all_nonnegative=configs['all_nonnegative']).items())
            time_limit()

        tangents = clear_polys_by_symmetry(tangents, problem.expr.gens, configs['symmetry'])
        time_limit()
        return tangents

    def _prepare_basis(self, degree: int, tangents: Dict[Poly, Expr],
            configs: Dict[str, Any]) -> Tuple[List[LinearBasis], np.ndarray]:
        """
        Prepare basis for linear programming.

        Parameters
        -----------
        degree: int
            The working degree to generate bases.
        tangents: Dict[Poly, Expr]
            Tangents to generate the bases. Each (key, value) pair is the polynomial
            form and the expression form of the tangent.
        configs: Dict[str, Any]
            A dictionary of configuration parameters containing the following keys:
            * all_nonnegative: bool
                Whether all variables are nonnegative.
            * quad_diff_order: int
                The maximum degree of the form (xi - xj)^(2k)*... in the basis. Defaults to 6.
            * basis_limit: int
                Limit of the basis. When the basis exceeds the limit, raise an error to kill the solver.
            * time_limit: int
                Limit of time in seconds. When the time exceeds the limit, raise an error to kill the solver.
            * basis: list
                Additional bases to be added to the basis.
            * symmetry: PermutationGroup
                The symmetry of the polynomial.

        Returns
        -------
        basis: list
            Basis for linear programming.
        arrays: np.array
            Array representation of the basis. A numpy or scipy matrix.
        """
        problem = (self._transformed_problem if self._transformed_problem is not None else self.problem)
        symbols = problem.expr.gens
        symmetry = configs['symmetry']
        quad_diff_order = configs['quad_diff_order']
        basis_limit = configs['basis_limit']
        time_limit = ArithmeticTimeout.make_checker(configs['time_limit'])

        all_basis = []
        all_arrays = []

        def append_basis(all_basis, all_arrays, basis, symmetry = symmetry):
            all_basis += basis
            all_arrays += np.vstack([x.as_array_np(expand_cyc=True, symmetry=symmetry) for x in basis])
            _check_basis_limit(all_basis, basis_limit)

        basis = configs.get('basis', None)
        if basis is not None:
            append_basis(all_basis, all_arrays, basis, symmetry=symmetry)

        if configs['all_nonnegative']:
            cls = LinearBasisTangent
        else:
            cls = LinearBasisTangentEven

        for tangent_p, tangent in tangents: # .items():
            basis, mat = cls.generate_quad_diff(tangent, symbols, degree,
                symmetry=symmetry, tangent_p=tangent_p, quad_diff_order=quad_diff_order)
            all_basis += basis
            _check_basis_limit(all_basis, basis_limit)
            time_limit()
            all_arrays.append(mat)

        eq_constraints = configs.get('ideal', problem.eq_constraints)
        for eq_p, eq in eq_constraints.items():
            basis, mat = LinearBasisTangent.generate_quad_diff(eq, symbols, degree,
                symmetry=symmetry, tangent_p=eq_p, quad_diff_order=0)
            all_basis += basis
            all_arrays.append(mat)
            basis = [b.__neg__() for b in basis]
            all_basis += basis
            _check_basis_limit(all_basis, basis_limit)
            time_limit()
            all_arrays.append(-mat)

        all_arrays = [non_zero_array for non_zero_array in all_arrays if non_zero_array.size > 0]
        if len(all_arrays) <= 0:
            return [], np.array([])
        all_arrays = np.vstack(all_arrays)
        time_limit()
        # print('Time for converting basis to arrays:', perf_counter() - time0)
        return all_basis, all_arrays

    def _lift_degree(self, configs: Dict[str, Any]) -> Any:
        problem = (self._transformed_problem if self._transformed_problem is not None else self.problem)
        return lift_degree(problem.expr, ineq_constraints=problem.ineq_constraints,
                    symmetry=configs['symmetry'],
                    lift_degree_limit=configs['lift_degree_limit'])

    def explore(self, configs):
        if self.status != 0:
            return

        verbose = configs['verbose']
        time_limit = configs['time_limit']
        end_time = perf_counter() + time_limit
        time_limit = ArithmeticTimeout.make_checker(time_limit)

        if self.problem.roots is None:
            domain = self.problem.expr.domain
            if domain.is_QQ or domain.is_ZZ:
                time1 = perf_counter()
                roots = self.problem.find_roots()
                if verbose:
                    print(f"Time for finding roots num = {len(roots):<6d}     : {perf_counter() - time1:.6f} seconds.")
                time_limit()

        problem, _homogenizer = self.problem.homogenize()
        self._transformed_problem = problem
        tangents = []

        if _homogenizer is not None: 
            # homogenize the tangents
            tangents = homogenize_expr_list(tangents, _homogenizer)
        else:
            # homogeneous polynomial does not accept non-homogeneous tangents
            tagents_poly = [t.as_poly(poly.gens) for t in tangents]
            tangents = [t for t, p in zip(tangents, tagents_poly) if len(p.free_symbols_in_domain)==0 and p.is_homogeneous]
        self._tangents = tangents


        ####################################################################
        #            Centralize the polynomial and symmetry
        ####################################################################
        if configs['centralize']:
            self._centralize(configs)
            problem = self._transformed_problem

        poly = problem.expr
        symmetry = MonomialManager(len(poly.gens), problem.identify_symmetry())
        problem, cons_restoration = problem.wrap_constraints(symmetry.perm_group)
        self._transformed_problem = problem


        ####################################################################
        #            Prepare tangents to form linear bases
        ####################################################################
        symbol_signs = problem.get_symbol_signs()
        all_nonnegative = all(s is not None and s > 0 for s, e in symbol_signs.values())

        internal_configs = configs.copy()
        internal_configs.update({
            'all_nonnegative': all_nonnegative,
            'symmetry': symmetry,
            'time_limit': time_limit,
            'tangents': self._tangents,
        })

        internal_configs['qmodule'] = self._prepare_qmodule(internal_configs)
        internal_configs['ideal']   = self._prepare_ideal(internal_configs)
        tangents = self._prepare_tangents(internal_configs)
        # print('Tangents after removing duplicates:', tangents)

        solution = None
        try:
            # prepare to lift the degree in an iterative way
            for lift_degree_info in self._lift_degree(internal_configs):
                time_limit()

                # RHS
                time0 = perf_counter()
                degree = lift_degree_info['degree']
                basis, arrays = self._prepare_basis(degree, tangents, internal_configs)

                if len(basis) <= 0:
                    continue
                time_limit()

                # LHS (from multipliers * poly)
                basis += lift_degree_info['basis']
                arrays = np.vstack([arrays,
                    np.array([x.as_array_np(expand_cyc=True, symmetry=symmetry) for x in lift_degree_info['basis']])])

                # sum of coefficients of the multipliers should be 1
                regularizer = np.zeros(arrays.shape[0])
                regularizer[-len(lift_degree_info['basis']):] = 1
                arrays = np.hstack([arrays, regularizer[:, None]])

                if verbose:
                    time1 = perf_counter()
                    print('Linear Programming Shape = (%d, %d)'%(arrays.shape[0], arrays.shape[1]),
                        '\tPreparation Time: %.3f s'%(time1 - time0), end = '')

                b = np.zeros(arrays.shape[1])
                b[-1] = 1
                optimized = np.ones(arrays.shape[0])


                ##############################################
                # main function: linear programming
                ##############################################
                linear_sos = None
                with warnings.catch_warnings(record=True) as __warns:
                    warnings.simplefilter('once')
                    from scipy.optimize import linprog
                    try:
                        linprog_options = configs['linprog_options']
                        if linprog_options.get('method', '').startswith('highs'):
                            linprog_options = linprog_options.copy()
                            linprog_options['options'] = linprog_options.get('options', {}).copy()
                            linprog_options['options']['time_limit'] = end_time - perf_counter()

                        linear_sos = linprog(optimized, A_eq=arrays.T, b_eq=b, **linprog_options)
                    except:
                        pass
                if linear_sos is None or not linear_sos.success:
                    # lift the degree up and retry
                    if verbose:
                        print('\tLP failed.')
                    continue
                if verbose:
                    print('\t\033[92mLP succeeded.\033[0m')

                y, basis, is_equal = linear_correction(
                    linear_sos.x,
                    basis,
                    num_multipliers = len(lift_degree_info['basis']),
                    symmetry = symmetry
                )
                if is_equal or configs['allow_numer'] > 0:
                    basis = odd_basis_to_even(basis, poly.gens, symbol_signs)
                    solution = create_linear_sol_from_y_basis(
                        problem=poly, y=y, basis=basis, symmetry=symmetry,
                        ineq_constraints=problem.ineq_constraints,
                        eq_constraints=problem.eq_constraints
                    )
                    solution = cons_restoration(solution)
                    break
                else:
                    if verbose:
                        print('\tRationalization failed.')

        except Exception as e:
            solution = None
            if isinstance(e, _basis_limit_exceeded):
                if verbose:
                    print(f"Basis limit {configs['basis_limit']} exceeded. LinearSOS aborted.")
            elif isinstance(e, ArithmeticTimeout):
                if verbose:
                    print(f"Arithmetic timeout. LinearSOS aborted.")
                raise e
            elif isinstance(e, MemoryError):
                if verbose:
                    print(f"Memory error: {e}. LinearSOS aborted.")
            else:
                raise e

        if solution is not None:
            if self._decentralizer is not None:
                solution = solution.xreplace(self._decentralizer)
            if _homogenizer is not None:
                solution = Solution.dehomogenize(solution, _homogenizer)
            self.problem.solution = solution

        self.status = -1
        self.finished = True



def LinearSOS(
        poly: Poly,
        ineq_constraints: Union[List[Expr], Dict[Expr, Expr]] = {},
        eq_constraints: Union[List[Expr], Dict[Expr, Expr]] = {},
        symmetry: Optional[Union[PermutationGroup, MonomialManager]] = None,
        roots: Optional[List[Root]] = None,
        tangents: List[Expr] = [],
        augment_tangents: bool = True,
        centralize: bool = True,
        preordering: str = 'quadratic',
        verbose: bool = False,
        quad_diff_order: int = 8,
        basis_limit: int = 15000,
        lift_degree_limit: int = 4,
        time_limit: float = 86400.,
        linprog_options: Dict = LINPROG_OPTIONS,
        allow_numer: int = 0
    ) -> Optional[Solution]:
    """
    Main function for linear programming SOS.

    Parameters
    -----------
    poly: Poly
        The polynomial to perform SOS on.
    ineq_constraints: Union[List[Expr], Dict[Expr, Expr]]
        Inequality constraints to the problem. This assumes g_1(x) >= 0, g_2(x) >= 0, ...
    eq_constraints: Union[List[Expr], Dict[Expr, Expr]]
        Equality constraints to the problem. This assumes h_1(x) = 0, h_2(x) = 0, ...
    symmetry: PermutationGroup or MonomialManager
        The symmetry of the polynomial. When it is None, it will be automatically generated. 
        If we want to skip the symmetry generation algorithm, please pass in a MonomialManager object.
    roots: list
        Equality cases of the inequality. If None, it will be searched automatically. To disable auto
        search, pass in an empty list.
    tangents: list
        Additional tangents to form the bases. Each tangent is a sympy nonnegative expression.
    augment_tangents: bool
        Whether to augment the tangents using heuristic methods. Defaults to True.
    preordering: str
        The preordering method for extending the basis. It can be 'none', 'linear', 'quadratic' or 'full'.
        Defaults to 'quadratic'.
    verbose: bool
        Whether to print the information of the linear programming problem. Defaults to False.
    quad_diff_order: int
        The maximum degree of the form (xi - xj)^(2k)*... in the basis. Defaults to 8.
    basis_limit: int
        The limit of the basis. When the basis exceeds the limit, the solver stops and returns None.
        Defaults to 15000.
    lift_degree_limit: int
        The maximum degree to lift the polynomial. Defaults to 4.
    time_limit: float
        The time limit in seconds for the solver.
    linprog_options: dict
        Options for scipy.optimize.linprog. Defaultedly use `{'method': 'highs-ds', 'options': {'presolve': False}}`. 
        Note that interiorpoint oftentimes does not provide exact rational solution. Both 'highs-ds' or 'simplex' are 
        recommended, yet the former is slightly faster.
        
        Moreover, using `presolve == True` has bug solving s((b2-a2+3c2+ab+7bc-5ca)(a2-b2-ab+2bc-ca)2):
        Assertion failed: abs_value < pivot_tolerance, file ../../scipy/_lib/highs/src/util/HFactor.cpp, line 1474
        Thus, for stability, we use `presolve == False` by default. However, setting it to True could be slightly faster.
    allow_numer: int
        Whether to allow numerical solution. When it is 0, the solution must be exact. When > 0, the solution can be numerical,
        this might be useful for large scale problems or irrational problems. TODO: Allow tolerance?

    Returns
    -------
    solution: Optional[Solution]
        The solution of the linear programming SOS. When solution is None, it means that the linear
        programming SOS fails.
    """
    problem = ProofNode.new_problem(poly, ineq_constraints, eq_constraints)
    configs = {
        SolvePolynomial: {
            'solvers': [LinearSOSSolver],
        },
        LinearSOSSolver: {
            'verbose': verbose,
            'linprog_options': linprog_options,
            'allow_numer': allow_numer,
            # 'symmetry': symmetry,
            # 'roots': roots,
            # 'tangents': tangents,
            'augment_tangents': augment_tangents,
            'centralize': centralize,
            'preordering': preordering,
            'quad_diff_order': quad_diff_order,
            'basis_limit': basis_limit,
            'lift_degree_limit': lift_degree_limit,
        }
    }
    return problem.sum_of_squares(configs, time_limit=time_limit)
