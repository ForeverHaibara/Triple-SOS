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
from .tangents import root_tangents
from .correction import linear_correction
from .updegree import lift_degree
from .solution import SolutionLinear
from ..shared import homogenize_expr_list, clear_polys_by_symmetry, sanitize_input, sanitize_output
from ...utils import findroot, RootsInfo, RootTangent
from ...utils.monomials import MonomialManager


LINPROG_OPTIONS = {
    'method': 'highs-ds' if version_tuple(_SCIPY_VERSION) >= (1, 6) else 'simplex',
    'options': {
        'presolve': False,
    }
}

class _basis_limit_exceeded(Exception): ...

def _prepare_tangents(symbols, prepared_tangents = [], rootsinfo: RootsInfo = None) -> Dict[Poly, Expr]:
    """
    Combine appointed tangents and tangents generated from rootsinfo.
    Roots that do not vanish at strict roots will be filtered out.
    """
    if rootsinfo is not None:
        # filter out tangents that do not vanish at strict roots
        prepared_tangents = rootsinfo.filter_tangents([
            RootTangent(expr, symbols) if isinstance(expr, Expr) else expr for expr in prepared_tangents
        ])
        tangents = [t.as_expr() for t in prepared_tangents] + [t.as_expr()**2 for t in rootsinfo.tangents]
    else:
        tangents = prepared_tangents.copy()
        
    if rootsinfo is None or not rootsinfo.has_nontrivial_roots():
        if S.One not in tangents:
            tangents.append(S.One)

        if len(symbols) == 3:
            a, b, c = symbols
            tangents += [
                (a**2 - b*c)**2, (b**2 - a*c)**2, (c**2 - a*b)**2,
                (a**3 - b*c**2)**2, (a**3 - b**2*c)**2, (b**3 - a*c**2)**2,
                (b**3 - a**2*c)**2, (c**3 - a*b**2)**2, (c**3 - a**2*b)**2,
            ]
        elif len(symbols) == 4:
            a, b, c, d = symbols
            tangents += [
                (a*b - c*d)**2, (a*c - b*d)**2, (a*d - b*c)**2,
            ]

    return dict((Poly(t, symbols), t) for t in tangents)


def _check_basis_limit(basis: List, limit: int = 15000) -> None:
    if len(basis) > limit:
        raise _basis_limit_exceeded

def _prepare_basis(
        symbols: List[Symbol],
        all_nonnegative: bool,
        degree: int,
        tangents: List[Tuple[Poly, Expr]] = {},
        eq_constraints: List[Tuple[Poly, Expr]] = {},
        rootsinfo = None,
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


def _get_qmodule_list(poly: Poly, ineq_constraints: Dict[Poly, Expr],
        all_nonnegative: bool = False, preordering: str = 'linear') -> List[Tuple[Poly, Expr]]:
    _ACCEPTED_PREORDERINGS = ['none', 'linear']
    if not preordering in _ACCEPTED_PREORDERINGS:
        raise ValueError("Invalid preordering method, expected one of %s, received %s." % (str(_ACCEPTED_PREORDERINGS), preordering))

    degree = poly.homogeneous_order()
    poly_one = Poly(1, *poly.gens)

    monomials = []
    linear_ineqs = []
    nonlin_ineqs = [(poly_one, S.One)]
    for ineq, e in ineq_constraints.items():
        if ineq.is_monomial and len(ineq.free_symbols) == 1 and ineq.total_degree() == 1 and ineq.LC() >= 0:
            monomials.append((ineq, e))
        elif ineq.is_linear:
            linear_ineqs.append((ineq, e))
        else:
            nonlin_ineqs.append((ineq, e))

    if all_nonnegative:
        # in this case we generate basis by LinaerBasisTangent rather than LinearBasisTangentEven
        # we discard all monomials
        pass
    else:
        linear_ineqs = monomials + linear_ineqs

    if preordering == 'none':
        return linear_ineqs + nonlin_ineqs

    qmodule = nonlin_ineqs.copy()
    for n in range(1, len(linear_ineqs) + 1):
        for comb in combinations(linear_ineqs, n):
            mul = poly_one
            for c in comb:
                mul = mul * c[0]
            d = mul.homogeneous_order()
            if d > degree:
                continue
            mul_expr = sp.Mul(*(c[1] for c in comb))
            for ineq, e in nonlin_ineqs:
                new_d = d + ineq.homogeneous_order()
                if new_d <= degree:
                    qmodule.append((mul * ineq, mul_expr * e))

    return (qmodule)


@sanitize_output()
@sanitize_input(homogenize=True, infer_symmetry=True, wrap_constraints=True)
def LinearSOS(
        poly: Poly,
        ineq_constraints: Union[List[Expr], Dict[Expr, Expr]] = {},
        eq_constraints: Union[List[Expr], Dict[Expr, Expr]] = {},
        symmetry: Optional[Union[PermutationGroup, MonomialManager]] = None,
        tangents: List[Expr] = [],
        rootsinfo: Optional[RootsInfo] = None,
        preordering: str = 'linear',
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
    tangents: list
        Additional tangents to be added to the basis.
    rootsinfo: RootsInfo
        Roots information of the polynomial to be optimized. When it is None, it will be automatically
        generated. If we want to skip the root finding algorithm or the tangent generation algorithm,
        please pass in a RootsInfo object. TODO: Shall we deprecate it?
    preordering: str
        The preordering method for extending the basis. It can be 'none' or 'linear'. Defaults to 'linear'.
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
                symmetry=symmetry,tangents=tangents,rootsinfo=rootsinfo,preordering=preordering,
                verbose=verbose,quad_diff_order=quad_diff_order,
                basis_limit=basis_limit,lift_degree_limit=lift_degree_limit,
                linprog_options=linprog_options,allow_numer=allow_numer,_homogenizer=_homogenizer)


def _LinearSOS(
        poly: Poly,
        ineq_constraints: Dict[Poly, Expr] = {},
        eq_constraints: Dict[Poly, Expr] = {},
        symmetry: MonomialManager = None,
        tangents: List[Poly] = [],
        rootsinfo: Optional[RootsInfo] = None,
        preordering: str = 'linear',
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
    else:
        # homogeneous polynomial does not accept non-homogeneous tangents
        tagents_poly = [t.as_poly(poly.gens) for t in tangents]
        tangents = [t for t, p in zip(tangents, tagents_poly) if len(p.free_symbols_in_domain)==0 and p.is_homogeneous]

    if rootsinfo is None:
        rootsinfo = findroot(poly, with_tangents = root_tangents)

    signs = _get_signs_of_vars(ineq_constraints, poly.gens)
    all_nonnegative = all(s > 0 for s in signs.values())

    tangents = _prepare_tangents(poly.gens, tangents, rootsinfo)
    tangents.update(ineq_constraints)
    tangents = _get_qmodule_list(poly, tangents, all_nonnegative=all_nonnegative, preordering=preordering)

    # remove duplicate tangents by symmetry
    # print('Tangents before removing duplicates:', tangents)
    tangents = clear_polys_by_symmetry(tangents, poly.gens, symmetry)
    eq_constraints = clear_polys_by_symmetry(eq_constraints.items(), poly.gens, symmetry)
    # print('Tangents after  removing duplicates:', tangents)

    try:
        # prepare to lift the degree in an iterative way
        for lift_degree_info in lift_degree(poly, ineq_constraints=ineq_constraints, symmetry=symmetry, lift_degree_limit=lift_degree_limit):
            # RHS
            time0 = time()
            degree = lift_degree_info['degree']
            basis, arrays = _prepare_basis(poly.gens, all_nonnegative=all_nonnegative, degree=degree, tangents=tangents,
                                            eq_constraints=eq_constraints, rootsinfo=rootsinfo, symmetry=symmetry, 
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
                print('Linear Programming Shape = (%d, %d)'%(arrays.shape[0], arrays.shape[1]), '\tPreparation Time: %.3f s'%(time1 - time0), end = '')

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
    except _basis_limit_exceeded:
        if verbose:
            print(f'Basis limit {basis_limit} exceeded. LinearSOS aborted.')
    return None