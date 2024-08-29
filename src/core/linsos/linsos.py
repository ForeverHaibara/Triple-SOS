from typing import Optional, List, Dict, Union
import warnings

import numpy as np
import sympy as sp
from sympy.combinatorics import PermutationGroup
from scipy.optimize import linprog

from .basis import (
    LinearBasis, LinearBasisTangent,
    cross_exprs, diff_tangents
)
from .tangents import root_tangents
from .correction import linear_correction
from .updegree import lift_degree
from .solution import SolutionLinear
from ..solver import homogenize
from ...utils import arraylize, findroot, RootsInfo, identify_symmetry, MonomialPerm, MonomialReduction


LINPROG_OPTIONS = {
    'method': 'highs-ds',
    'options': {
        'presolve': False,
    }
}

def _prepare_tangents(poly, prepared_tangents = [], rootsinfo = None):
    """
    Combine appointed tangents and tangents generated from rootsinfo.
    Roots that do not vanish at strict roots will be filtered out.
    """
    if rootsinfo is not None:
        # filter out tangents that do not vanish at strict roots
        prepared_tangents = rootsinfo.filter_tangents(prepared_tangents)
        tangents = prepared_tangents + [t.as_expr()**2 for t in rootsinfo.tangents]
        return tangents
    return prepared_tangents

def _prepare_basis(
        symbols: List[sp.Symbol],
        degree: int,
        tangents: List[sp.Expr],
        rootsinfo = None,
        basis: Optional[List[LinearBasis]] = None,
        symmetry: PermutationGroup = PermutationGroup()
    ):
    """
    Prepare basis for linear programming.

    Parameters
    -------
    poly: sp.Poly
        The polynomial to be optimized.
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
    if basis is None:
        basis = []
    _diff_tangents = diff_tangents(symbols)
    if sp.S(1) not in tangents:
        tangents.append(sp.S(1))
    for tangent in tangents:
        d = tangent.as_poly(symbols).total_degree()
        if degree >= d:
            cross_tangents = cross_exprs(_diff_tangents, symbols, degree - d)
            for t in cross_tangents:
                basis += LinearBasisTangent.generate(t * tangent, symbols, degree)

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

    arrays = np.array([x.as_array_np(expand_cyc=True, symmetry=symmetry) for x in basis])
    return basis, arrays


def LinearSOS(
        poly: sp.polys.Poly,
        tangents: List = [],
        rootsinfo: Optional[RootsInfo] = None,
        symmetry: Optional[Union[PermutationGroup, MonomialReduction]] = None,
        verbose: bool = 1,#False,
        degree_limit: int = 12,
        linprog_options: Dict = LINPROG_OPTIONS,
    ) -> Optional[SolutionLinear]:
    """
    Main function for linear programming SOS.

    Parameters
    -------
    poly: sp.polys.Poly
        The polynomial to perform SOS on. It should be a homogeneous, cyclic polynomial of a, b, c.
        There is no check for this.
    tangents: list
        Additional tangents to be added to the basis.
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

    Returns
    -------
    solution: Optional[Solution]
        The solution of the linear programming SOS. When solution is None, it means that the linear
        programming SOS fails.
    """
    if not poly.domain.is_Numerical:
        return
    poly, homogenizer = homogenize(poly)

    if homogenizer is not None: 
        # homogenize the tangents
        symbols = poly.gens[:-1]
        translation = dict(zip(symbols, (i / homogenizer for i in symbols)))
        tangents = [t.subs(translation).together() for t in tangents]
        tangent_d = [sp.fraction(t)[1].as_poly(homogenizer).degree() for t in tangents]
        tangents = [t * homogenizer**d for t, d in zip(tangents, tangent_d)]

    if rootsinfo is None:
        rootsinfo = findroot(poly, with_tangents = root_tangents)

    tangents = _prepare_tangents(poly, tangents, rootsinfo)

    if symmetry is None:
        symmetry = identify_symmetry(poly, homogenizer)
    if isinstance(symmetry, PermutationGroup):
        symmetry = MonomialPerm(symmetry)

    # prepare to lift the degree in an iterative way
    for lift_degree_info in lift_degree(poly, symmetry=symmetry, degree_limit = degree_limit):
        # RHS
        degree = lift_degree_info['degree']
        basis, arrays = _prepare_basis(poly.gens, degree, tangents, rootsinfo = rootsinfo, symmetry=symmetry)
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
        if solution is not None:
            solution = solution.dehomogenize(homogenizer)
        return solution