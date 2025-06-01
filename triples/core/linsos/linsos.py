from itertools import combinations
from typing import Optional, Tuple, List, Dict, Union
from time import time
import warnings

import numpy as np
import sympy as sp
from sympy import Poly, Expr, Symbol
from sympy.core.singleton import S
from sympy.combinatorics import PermutationGroup
from sympy.external.importtools import version_tuple
from scipy.optimize import linprog
from scipy import __version__ as _SCIPY_VERSION

from .basis import LinearBasis, LinearBasisTangent, LinearBasisTangentEven
from .tangents import prepare_tangents, prepare_inexact_tangents, get_qmodule_list
from .correction import linear_correction
from .updegree import lift_degree
from .solution import SolutionLinear
from ..preprocess import sanitize
from ..shared import homogenize_expr_list, clear_polys_by_symmetry
from ...utils import Root, optimize_poly
from ...utils.monomials import MonomialManager


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

def _prepare_basis(
        symbols: List[Symbol],
        all_nonnegative: bool,
        degree: int,
        tangents: List[Tuple[Poly, Expr]] = {},
        eq_constraints: List[Tuple[Poly, Expr]] = {},
        basis: Optional[List[LinearBasis]] = None,
        symmetry: Union[MonomialManager, PermutationGroup] = PermutationGroup(),
        quad_diff_order: int = 8,
        basis_limit: int = 15000,
    ) -> Tuple[List[LinearBasis], np.ndarray]:
    """
    Prepare basis for linear programming.

    Parameters
    -------
    poly: Poly
        The polynomial to be optimized.
    all_nonnegative: bool
        Whether all variables are nonnegative.
    degree: int
        The degree of the polynomial to be optimized.
    tangents: Dict[Poly, Expr]
        Tangents to be added to the basis. Each (key, value) pair is the polynomial
        form and the expression form of the tangent.
    eq_constraints: Dict[Poly, Expr]
        Equality constraints to be added to the basis. Each (key, value) pair is the polynomial
        form and the expression form of the equality constraint.
    rootsinfo: RootsInfo
        Roots information of the polynomial to be optimized. When it has nontrivial roots,
        we skip the loading of normal basis like AMGM.
    basis: list
        Additional basis to be added to the basis.
    symmetry: MonomialManager or PermutationGroup
        The symmetry of the polynomial. When it is None, it will be automatically generated.
    quad_diff_order: int
        The maximum degree of the form (xi - xj)^(2k)*... in the basis. Defaults to 8.
    basis_limit: int
        Limit of the basis. When the basis exceeds the limit, raise an error to kill the solver.

    Returns
    -------
    basis: list
        Basis for linear programming.
    arrays: np.array
        Array representation of the basis. A matrix.
    """
    # time0 = time()
    all_basis = []
    all_arrays = []

    def append_basis(all_basis, all_arrays, basis, symmetry = symmetry):
        all_basis += basis
        all_arrays += np.vstack([x.as_array_np(expand_cyc=True, symmetry=symmetry) for x in basis])
        _check_basis_limit(all_basis, basis_limit)

    if basis is not None:
        append_basis(all_basis, all_arrays, basis, symmetry = symmetry)

    if all_nonnegative:
        cls = LinearBasisTangent
    else:
        cls = LinearBasisTangentEven

    for tangent_p, tangent in tangents: # .items():
        basis, mat = cls.generate_quad_diff(tangent, symbols, degree, symmetry=symmetry, tangent_p=tangent_p, quad_diff_order=quad_diff_order)
        all_basis += basis
        _check_basis_limit(all_basis, basis_limit)
        all_arrays.append(mat)

    for eq_p, eq in eq_constraints: # .items():
        basis, mat = LinearBasisTangent.generate_quad_diff(eq, symbols, degree, symmetry=symmetry, tangent_p=eq_p, quad_diff_order=0)
        all_basis += basis
        all_arrays.append(mat)
        basis = [b.__neg__() for b in basis]
        all_basis += basis
        _check_basis_limit(all_basis, basis_limit)
        all_arrays.append(-mat)

    all_arrays = [non_zero_array for non_zero_array in all_arrays if non_zero_array.size > 0]
    if len(all_arrays) <= 0:
        return [], np.array([])
    all_arrays = np.vstack(all_arrays)
    # print('Time for converting basis to arrays:', time() - time0)
    return all_basis, all_arrays


def _get_signs_of_vars(ineq_constraints: Dict[Poly, Expr], symbols: Tuple[Symbol, ...]) -> Dict[Symbol, int]:
    """
    Infer the signs of each variable from the inequality constraints.
    """
    signs = dict.fromkeys(symbols, 0)
    for ineq in ineq_constraints:
        if ineq.is_monomial and ineq.total_degree() == 1:
            sgn = 1 if ineq.LC() > 0 else (-1 if ineq.LC() < 0 else 0)
            signs[ineq.free_symbols.pop()] = sgn
    return signs

def _odd_basis_to_even(basis: List[LinearBasis], symbols: Tuple[Symbol, ...], ineq_constraints: Dict[Poly, Expr]) -> List[LinearBasis]:
    mapping = dict((s, s) for s in symbols)
    for ineq, e in ineq_constraints.items():
        if ineq.is_monomial and ineq.total_degree() == 1 and ineq.LC() > 0:
            mapping[ineq.free_symbols.pop()] = e / ineq.LC()
    mapping = [mapping[s] for s in symbols]

    new_basis = []
    for b in basis:
        if isinstance(b, LinearBasisTangentEven) or not isinstance(b, LinearBasisTangent):
            new_basis.append(b)
        else:
            new_basis.append(b.to_even(mapping))
    return new_basis


@sanitize(homogenize=True, infer_symmetry=True, wrap_constraints=True,
    disable_denom_finding_roots=True)
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
        linprog_options: Dict = LINPROG_OPTIONS,
        allow_numer: int = 0,
        _homogenizer: Optional[Symbol] = None
    ) -> Optional[SolutionLinear]:
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
    _homogenizer: Optional[Symbol]
        Indicates which symbol to be used as the homogenizer. The symbol will be used to homogenize the tangents.
        This is expected to autogenerated by the function decorator and should not be used by the user.

    Returns
    -------
    solution: Optional[Solution]
        The solution of the linear programming SOS. When solution is None, it means that the linear
        programming SOS fails.
    """
    return _LinearSOS(poly, ineq_constraints=ineq_constraints, eq_constraints=eq_constraints,
                symmetry=symmetry,roots=roots,tangents=tangents,augment_tangents=augment_tangents,
                centralize=centralize,preordering=preordering,verbose=verbose,quad_diff_order=quad_diff_order,
                basis_limit=basis_limit,lift_degree_limit=lift_degree_limit,
                linprog_options=linprog_options,allow_numer=allow_numer,_homogenizer=_homogenizer)


def _LinearSOS(
        poly: Poly,
        ineq_constraints: Dict[Poly, Expr] = {},
        eq_constraints: Dict[Poly, Expr] = {},
        symmetry: MonomialManager = None,
        roots: Optional[List[Root]] = None,
        tangents: List[Poly] = [],
        augment_tangents: bool = True,
        centralize: bool = True,
        preordering: str = 'quadratic',
        verbose: bool = False,
        quad_diff_order: int = 8,
        basis_limit: int = 15000,
        lift_degree_limit: int = 4,
        linprog_options: Dict = LINPROG_OPTIONS,
        allow_numer: int = 0,
        _homogenizer: Optional[Symbol] = None
    ) -> Optional[SolutionLinear]:

    if _homogenizer is not None: 
        # homogenize the tangents
        tangents = homogenize_expr_list(tangents, _homogenizer)
        if roots is not None:
            roots = [Root(r) if not isinstance(r, Root) else r for r in roots]
            roots = [Root(r.root + (sp.Integer(1),), r.domain, r.rep + (r.domain.one,)) for r in roots]
    else:
        # homogeneous polynomial does not accept non-homogeneous tangents
        tagents_poly = [t.as_poly(poly.gens) for t in tangents]
        tangents = [t for t, p in zip(tangents, tagents_poly) if len(p.free_symbols_in_domain)==0 and p.is_homogeneous]

    if roots is None:
        roots = []
        if poly.domain.is_QQ or poly.domain.is_ZZ:
            time1 = time()
            roots = optimize_poly(poly, list(ineq_constraints),
                ([poly] + list(eq_constraints) + [_homogenizer - 1] if _homogenizer is not None else []), poly.gens)
            if verbose:
                print(f"Time for finding roots num = {len(roots):<6d}     : {time() - time1:.6f} seconds.")

        roots = [Root(r) for r in roots]
    roots = [Root(r) if not isinstance(r, Root) else r for r in roots]
    roots = [r for r in roots if not r.is_zero]



    ####################################################################
    #            Centralize the polynomial and symmetry
    ####################################################################
    if centralize and len(roots) == 1 and roots[0].is_Rational and symmetry.is_trivial and any(ri != 1 and ri != 0 for ri in roots[0]):
        gens = poly.gens
        centralizer = {gens[i]: roots[0][i]*gens[i] for i in range(len(gens)) if roots[0][i] != 0 and roots[0][i] != 1}
        def ct(x):
            return x.as_expr().xreplace(centralizer).as_poly(gens)
        new_poly = ct(poly)
        new_ineqs = {ct(k): v.xreplace(centralizer) for k, v in ineq_constraints.items()}
        new_eqs = {ct(k): v.xreplace(centralizer) for k, v in eq_constraints.items()}
        new_tangents = [ct(t) for t in tangents]
        new_roots = [Root(tuple(1 if roots[0][i] != 0 else 0 for i in range(len(gens))))]
        if verbose:
            print(f'LinearSOS centralizing {centralizer}')
            # print('Goal         :', new_poly)
            # print('Inequalities :', new_ineqs)
            # print('Equalities   :', new_eqs)
        sol = _LinearSOS(new_poly, ineq_constraints=new_ineqs, eq_constraints=new_eqs,
                symmetry=symmetry,roots=new_roots,tangents=new_tangents,augment_tangents=augment_tangents,
                centralize=False,preordering=preordering,verbose=verbose,quad_diff_order=quad_diff_order,
                basis_limit=basis_limit,lift_degree_limit=lift_degree_limit,
                linprog_options=linprog_options,allow_numer=allow_numer)
        if sol is not None:
            sol.problem = poly
            sol.solution = sol.solution.xreplace({
                gens[i]: gens[i]/roots[0][i] for i in range(len(gens)) if roots[0][i] != 0 and roots[0][i] != 1
            })
        return sol


    ####################################################################
    #            Prepare tangents to form linear bases
    ####################################################################
    signs = _get_signs_of_vars(ineq_constraints, poly.gens)
    all_nonnegative = all(s > 0 for s in signs.values())

    qmodule = get_qmodule_list(poly, ineq_constraints, all_nonnegative=all_nonnegative, preordering=preordering)
    qmodule = clear_polys_by_symmetry(qmodule, poly.gens, symmetry)

    tangents = list(prepare_tangents(poly, qmodule, eq_constraints, roots=roots, additional_tangents=tangents).items())
    if augment_tangents:
        tangents += list(prepare_inexact_tangents(poly, ineq_constraints, eq_constraints,
            monomial_manager=symmetry, roots=roots, all_nonnegative=all_nonnegative).items())

    tangents = clear_polys_by_symmetry(tangents, poly.gens, symmetry)
    eq_constraints = clear_polys_by_symmetry(eq_constraints.items(), poly.gens, symmetry)
    # print('Tangents after removing duplicates:', tangents)


    try:
        # prepare to lift the degree in an iterative way
        for lift_degree_info in lift_degree(poly, ineq_constraints=ineq_constraints, symmetry=symmetry, lift_degree_limit=lift_degree_limit):
            # RHS
            time0 = time()
            degree = lift_degree_info['degree']
            basis, arrays = _prepare_basis(poly.gens, all_nonnegative=all_nonnegative, degree=degree, tangents=tangents,
                                            eq_constraints=eq_constraints, symmetry=symmetry, 
                                            quad_diff_order=quad_diff_order, basis_limit=basis_limit)
            if len(basis) <= 0:
                continue

            # LHS (from multipliers * poly)
            basis += lift_degree_info['basis']
            arrays = np.vstack([arrays, np.array([x.as_array_np(expand_cyc=True, symmetry=symmetry) for x in lift_degree_info['basis']])])

            # sum of coefficients of the multipliers should be 1
            regularizer = np.zeros(arrays.shape[0])
            regularizer[-len(lift_degree_info['basis']):] = 1
            arrays = np.hstack([arrays, regularizer[:, None]])

            if verbose:
                time1 = time()
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
                try:
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
            if is_equal or allow_numer > 0:
                basis = _odd_basis_to_even(basis, poly.gens, ineq_constraints)
                solution = SolutionLinear._from_y_basis(
                    problem=poly, y=y, basis=basis, symmetry=symmetry,
                    ineq_constraints=ineq_constraints, eq_constraints=dict(eq_constraints),
                    is_equal=is_equal
                )
                return solution
            else:
                if verbose:
                    print('\tRationalization failed.')

    except _basis_limit_exceeded:
        if verbose:
            print(f'Basis limit {basis_limit} exceeded. LinearSOS aborted.')
    return None