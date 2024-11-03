from itertools import combinations
from typing import Optional, Tuple, List, Dict, Union
from time import time
import warnings

import numpy as np
import sympy as sp
from sympy.core.singleton import S
from sympy.combinatorics import PermutationGroup
from scipy.optimize import linprog
from scipy import __version__ as SCIPY_VERSION

from .basis import LinearBasis, LinearBasisTangent, LinearBasisTangentEven
from .tangents import root_tangents
from .correction import linear_correction
from .updegree import lift_degree
from .solution import SolutionLinear
from ..shared import homogenize_expr_list, identify_symmetry_from_lists, clear_polys_by_symmetry, sanitize_input
from ...utils import findroot, RootsInfo, RootTangent
from ...utils.basis_generator import MonomialPerm, MonomialReduction

LINPROG_OPTIONS = {
    'method': 'highs-ds' if SCIPY_VERSION >= '1.6.0' else 'simplex',
    'options': {
        'presolve': False,
    }
}

def _prepare_tangents(symbols, prepared_tangents = [], rootsinfo: RootsInfo = None):
    """
    Combine appointed tangents and tangents generated from rootsinfo.
    Roots that do not vanish at strict roots will be filtered out.
    """
    if rootsinfo is not None:
        # filter out tangents that do not vanish at strict roots
        prepared_tangents = rootsinfo.filter_tangents([
            RootTangent(expr, symbols) if isinstance(expr, sp.Expr) else expr for expr in prepared_tangents
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

    return tangents



def _prepare_basis(
        symbols: List[sp.Symbol],
        all_nonnegative: bool,
        degree: int,
        tangents: List[sp.Expr],
        eq_constraints: List[sp.Expr] = [],
        rootsinfo = None,
        basis: Optional[List[LinearBasis]] = None,
        symmetry: Union[MonomialReduction, PermutationGroup] = PermutationGroup()
    ) -> Tuple[List[LinearBasis], np.ndarray]:
    """
    Prepare basis for linear programming.

    Parameters
    -------
    poly: sp.Poly
        The polynomial to be optimized.
    all_nonnegative: bool
        Whether all variables are nonnegative.
    degree: int
        The degree of the polynomial to be optimized.
    tangents: list
        Tangents to be added to the basis.
    rootsinfo: RootsInfo
        Roots information of the polynomial to be optimized. When it has nontrivial roots,
        we skip the loading of normal basis like AMGM.
    basis: list
        Additional basis to be added to the basis.

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

    if basis is not None:
        append_basis(all_basis, all_arrays, basis, symmetry = symmetry)

    if all_nonnegative:
        cls = LinearBasisTangent
    else:
        cls = LinearBasisTangentEven
        # nonneg_vars = [s for s, nonneg in zip(symbols, nonnegativity) if nonneg]
        # if len(nonneg_vars):
        #     tangents2 = tangents.copy()
        #     for s in nonneg_vars:
        #         tangents2.extend([t * s for t in tangents])
        #     tangents = tangents2

    for tangent in tangents:
        basis, mat = cls.generate_quad_diff(tangent, symbols, degree, symmetry = symmetry)
        all_basis += basis
        all_arrays.append(mat)

    for eq in eq_constraints:
        basis, mat = LinearBasisTangent.generate_quad_diff(eq, symbols, degree, symmetry = symmetry, quad_diff = False)
        all_basis += basis
        all_arrays.append(mat)
        basis = [b.__neg__() for b in basis]
        all_basis += basis
        all_arrays.append(-mat)

    # if is_cyc:
    #     for tangent in tangents:
    #         basis += LinearBasisTangentCyclic.generate(degree, tangent = tangent)

    #     if not rootsinfo.has_nontrivial_roots():
    #         basis += CachedCommonLinearBasisTangentCyclic.generate(degree)
    #         basis += LinearBasisAMGMCyclic.generate(degree)
    #         basis += CachedCommonLinearBasisSpecialCyclic.generate(degree)
    # else:
    #     for tangent in tangents:
    #         basis += LinearBasisTangent.generate(degree, tangent = tangent)
    #     basis += CachedCommonLinearBasisTangent.generate(degree)

    all_arrays = [non_zero_array for non_zero_array in all_arrays if non_zero_array.size > 0]
    if len(all_arrays) <= 0:
        return [], np.array([])
    all_arrays = np.vstack(all_arrays)
    # print('Time for converting basis to arrays:', time() - time0)
    return all_basis, all_arrays


def _get_signs_of_vars(ineq_constraints: List[sp.Poly], symbols: Tuple[sp.Symbol, ...]) -> Dict[sp.Symbol, int]:
    """
    Infer the signs of each variable from the inequality constraints.
    """
    signs = dict.fromkeys(symbols, 0)
    for ineq in ineq_constraints:
        if ineq.is_monomial and ineq.total_degree() == 1:
            sgn = 1 if ineq.LC() > 0 else (-1 if ineq.LC() < 0 else 0)
            signs[ineq.free_symbols.pop()] = sgn
    return signs


def _get_qmodule_list(poly: sp.Poly, ineq_constraints: List[sp.Poly], all_nonnegative: bool = False, preordering: str = 'linear') -> List[sp.Poly]:
    _ACCEPTED_PREORDERINGS = ['none', 'linear']
    if not preordering in _ACCEPTED_PREORDERINGS:
        raise ValueError("Invalid preordering method, expected one of %s, received %s." % (str(_ACCEPTED_PREORDERINGS), preordering))

    degree = poly.homogeneous_order()
    poly_one = sp.Poly(1, *poly.gens)

    if preordering == 'none':
        return ineq_constraints

    monomials = []
    linear_ineqs = []
    nonlin_ineqs = [poly_one]
    for ineq in ineq_constraints:
        if ineq.is_monomial and len(ineq.free_symbols) == 1 and ineq.total_degree() == 1 and ineq.LC() >= 0:
            monomials.append(ineq)
        elif ineq.is_linear:
            linear_ineqs.append(ineq)
        else:
            nonlin_ineqs.append(ineq)

    if all_nonnegative:
        # in this case we generate basis by LinaerBasisTangent rather than LinearBasisTangentEven
        # we discard all monomials
        pass
    else:
        linear_ineqs = monomials + linear_ineqs

    qmodule = nonlin_ineqs.copy()
    for n in range(1, len(linear_ineqs) + 1):
        for comb in combinations(linear_ineqs, n):
            mul = poly_one
            for c in comb:
                mul = mul * c
            d = mul.homogeneous_order()
            for ineq in nonlin_ineqs:
                new_d = d + ineq.homogeneous_order()
                if new_d <= degree:
                    qmodule.append(mul * ineq)

    return qmodule


@sanitize_input(homogenize=True, infer_symmetry=True)
def LinearSOS(
        poly: sp.Poly,
        ineq_constraints: List[sp.Poly] = [],
        eq_constraints: List[sp.Poly] = [],
        symmetry: Optional[Union[PermutationGroup, MonomialReduction]] = None,
        tangents: List[sp.Poly] = [],
        rootsinfo: Optional[RootsInfo] = None,
        preordering: str = 'linear',
        verbose: bool = False,
        degree_limit: int = 12,
        linprog_options: Dict = LINPROG_OPTIONS,
        _homogenizer: Optional[sp.Symbol] = None
    ) -> Optional[SolutionLinear]:
    """
    Main function for linear programming SOS.

    Parameters
    -------
    poly: sp.Poly
        The polynomial to perform SOS on. It should be a homogeneous, cyclic polynomial of a, b, c.
        There is no check for this.
    tangents: list
        Additional tangents to be added to the basis.
    symmetry: MonomialReduction or PermutationGroup
        The symmetry of the polynomial. When it is None, it will be automatically generated. 
        If we want to skip the symmetry generation algorithm, please pass in a MonomialReduction object.
    rootsinfo: RootsInfo
        Roots information of the polynomial to be optimized. When it is None, it will be automatically
        generated. If we want to skip the root finding algorithm or the tangent generation algorithm,
        please pass in a RootsInfo object.
    verbose: bool
        Whether to print the information of the linear programming problem. Defaults to False.
    degree_limit: int
        The degree limit of the polynomial to be optimized. When the degree exceeds the degree limit, 
        the SOS is forced to shutdown. Defaults to 12.
    linprog_options: dict
        Options for scipy.optimize.linprog. Defaultedly use `{'method': 'highs-ds', 'options': {'presolve': False}}`. 
        Note that interiorpoint oftentimes does not provide exact rational solution. Both 'highs-ds' or 'simplex' are 
        recommended, yet the former is slightly faster.
        
        Moreover, using `presolve == True` has bug solving s((b2-a2+3c2+ab+7bc-5ca)(a2-b2-ab+2bc-ca)2):
        Assertion failed: abs_value < pivot_tolerance, file ../../scipy/_lib/highs/src/util/HFactor.cpp, line 1474
        Thus, for stability, we use `presolve == False` by default. However, setting it to True is slightly faster (0.03 sec).
    _homogenizer: Optional[sp.Symbol]
        Indicates which symbol to be used as the homogenizer. The symbol will be used to homogenize the tangents.
        This is expected to autogenerated by the function decorator and should not be used by the user.

    Returns
    -------
    solution: Optional[Solution]
        The solution of the linear programming SOS. When solution is None, it means that the linear
        programming SOS fails.
    """
    if _homogenizer is not None: 
        # homogenize the tangents
        tangents = homogenize_expr_list(tangents, _homogenizer)
    else:
        # homogeneous polynomial does not accept non-homogeneous tangents
        tagents_poly = [t.as_poly(poly.gens) for t in tangents]
        tangents = [t for t, p in zip(tangents, tagents_poly) if p.domain.is_Numerical and p.is_homogeneous]

    if rootsinfo is None:
        rootsinfo = findroot(poly, with_tangents = root_tangents)

    signs = _get_signs_of_vars(ineq_constraints, poly.gens)
    all_nonnegative = all(s > 0 for s in signs.values())
    var_signs = [signs[s] for s in poly.gens]

    tangents = _prepare_tangents(poly.gens, tangents, rootsinfo)
    tangents = [sp.Poly(t, *poly.gens) for t in tangents] + ineq_constraints
    tangents = _get_qmodule_list(poly, tangents, all_nonnegative=all_nonnegative, preordering = preordering)

    # remove duplicate tangents by symmetry
    # print('Tangents before removing duplicates:', tangents)
    tangents = clear_polys_by_symmetry(tangents, poly.gens, symmetry)
    eq_constraints = clear_polys_by_symmetry(eq_constraints, poly.gens, symmetry)
    # print('Tangents after  removing duplicates:', tangents)

    # following lines are equvialent to: [t.as_expr().factor() for t in ...]
    _prod_factors = lambda s: sp.Mul(s[0], *(i.as_expr()**d for i, d in s[1]))
    tangents = list(map(_prod_factors, map(lambda t: t.factor_list(), tangents)))
    eq_constraints = list(map(_prod_factors, map(lambda t: t.factor_list(), eq_constraints)))


    # prepare to lift the degree in an iterative way
    for lift_degree_info in lift_degree(poly, var_signs, symmetry=symmetry, degree_limit = degree_limit):
        # RHS
        degree = lift_degree_info['degree']
        basis, arrays = _prepare_basis(poly.gens, all_nonnegative, degree, tangents,
                                        eq_constraints=eq_constraints, rootsinfo=rootsinfo, symmetry=symmetry)
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
            print('Linear Programming Shape = (%d, %d)'%(arrays.shape[0], arrays.shape[1]))

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
            continue

        solution = linear_correction(
            poly, 
            linear_sos.x,
            basis,
            num_multipliers = len(lift_degree_info['basis']),
            symmetry = symmetry
        )
        return solution