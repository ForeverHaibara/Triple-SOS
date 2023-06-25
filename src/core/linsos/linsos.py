from typing import Optional, List, Dict
import warnings

import numpy as np
import sympy as sp
from scipy.optimize import linprog

from .basis import (
    LinearBasisSquare, 
    CachedCommonLinearBasisSquare, 
    LinearBasisAMGM, 
    CachedCommonLinearBasisSpecial
)
from .correction import linear_correction
from .updegree import higher_degree
from .solution import SolutionLinear
from ...utils.polytools import deg
from ...utils.basis_generator import arraylize
from ...utils.roots.findroot import findroot
from ...utils.roots.rootsinfo import RootsInfo


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

def _prepare_basis(degree, tangents, rootsinfo = None, basis = []):
    """
    Prepare basis for linear programming.

    Parameters
    -------
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
    for tangent in tangents:
        basis += LinearBasisSquare.generate(degree, tangent = tangent)

    if not rootsinfo.has_nontrivial_roots:
        basis += CachedCommonLinearBasisSquare.generate(degree)
        basis += LinearBasisAMGM.generate(degree)
        basis += CachedCommonLinearBasisSpecial.generate(degree)

    arrays = np.array([x.array for x in basis])
    return basis, arrays


def LinearSOS(
        poly: sp.polys.Poly,
        tangents: List = [],
        rootsinfo: Optional[RootsInfo] = None,
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

    n = deg(poly)

    if rootsinfo is None:
        rootsinfo = findroot(poly, with_tangents = True)

    tangents = _prepare_tangents(poly, tangents, rootsinfo)

    # prepare to higher the degree in an iterative way
    for higher_degree_info in higher_degree(poly, degree_limit = degree_limit):
        n = higher_degree_info['degree']
        basis, arrays = _prepare_basis(n, tangents, rootsinfo = rootsinfo, basis = higher_degree_info['basis'])
        if len(basis) <= 0:
            continue

        if verbose:
            print('Linear Programming Shape = (%d, %d)'%(arrays.shape[0], arrays.shape[1]))

        b = arraylize(higher_degree_info['poly'])
        optimized = np.ones(arrays.shape[0])
        # optimized[:len(higher_degree_info['basis'])] = -2 # these are multiplier adjustment


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
            # raise the degree up and retry
            continue

        solution = linear_correction(
            poly, 
            linear_sos.x, 
            basis, 
            multiplier = higher_degree_info['multiplier']
        )
        return solution