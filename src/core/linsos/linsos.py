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
from ...utils.polytools import deg
from ...utils.basis_generator import arraylize
from ...utils.roots.findroot import findroot

def _prepare_tangents(poly, prepared_tangents = [], rootsinfo = None):    
    prepared_tangents = rootsinfo.filter_tangents(prepared_tangents)
    tangents = prepared_tangents + [t.as_expr()**2 for t in rootsinfo.tangents]

    return tangents

def _prepare_basis(degree, tangents, rootsinfo = None):
    basis = []
    for tangent in tangents:
        basis += LinearBasisSquare.generate(degree, tangent = tangent)

    if not rootsinfo.has_nontrivial_roots:
        basis += CachedCommonLinearBasisSquare.generate(degree)
        basis += LinearBasisAMGM.generate(degree)
        basis += CachedCommonLinearBasisSpecial.generate(degree)

    arrays = np.array([x.array for x in basis])
    return basis, arrays

def LinearSOS(
        poly,
        tangents = [],
        rootsinfo = None,
        verbose = False,
        degree_limit = 12,
        linprog_options = {},
    ):

    n = deg(poly)

    if rootsinfo is None:
        rootsinfo = findroot(poly, with_tangents = True)

    tangents = _prepare_tangents(poly, tangents, rootsinfo)

    for higher_degree_info in higher_degree(poly, degree_limit = degree_limit):
        n = higher_degree_info['degree']
        basis, arrays = _prepare_basis(n, tangents, rootsinfo)
        if len(basis) <= 0:
            continue

        if verbose:
            print('Linear Programming Shape = (%d, %d)'%(arrays.shape[0], arrays.shape[1]))

        b = arraylize(higher_degree_info['poly'])
        linear_sos = None
        with warnings.catch_warnings(record=True) as __warns:
            warnings.simplefilter('once')
            try:
                # options = {
                #     method: 'simplex',
                #     tol: 1e-9,
                # }
                linear_sos = linprog(np.ones(arrays.shape[0]), A_eq=arrays.T, b_eq=b, **linprog_options)
            except:
                pass

        if linear_sos is None or not linear_sos.success:
            continue

        solution = linear_correction(
            poly, 
            linear_sos.x, 
            basis, 
            multiplier = higher_degree_info['multiplier']
        )
        return solution