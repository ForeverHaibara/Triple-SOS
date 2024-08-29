from typing import List, Optional, Tuple

import numpy as np
import sympy as sp
from sympy.core.singleton import S
from sympy.combinatorics import PermutationGroup

from .basis import LinearBasis
from .solution import SolutionLinear
from ...utils.basis_generator import arraylize_sp
from ...utils.roots.rationalize import rationalize_array

def _filter_zero_y(y, basis):
    """
    Filter out the zero coefficients.
    """
    reduced_y, reduced_basis = [], []
    for v, b in zip(y, basis):
        if v != 0:
            reduced_y.append(v)
            reduced_basis.append(b)

    return reduced_y, reduced_basis

def _basis_as_matrix(basis: List[LinearBasis], symmetry: PermutationGroup = PermutationGroup()) -> sp.Matrix:
    """
    Extract the array representations of each basis and stack them into a matrix.
    """
    mat = [b.as_array_sp(expand_cyc=True, symmetry=symmetry) for b in basis]
    mat = sp.Matrix(mat).reshape(len(mat), mat[0].shape[0]).T
    return mat

def linear_correction(
        poly: sp.Poly,
        y: List[float] = [],
        basis: List[LinearBasis] = [],
        multiplier: sp.Expr = S.One,
        symmetry: PermutationGroup = PermutationGroup(),
        zero_tol: float = 1e-6,
    ) -> SolutionLinear:
    """
    Linear programming is a numerical way to solve the SOS problem. However, we require
    the solution to be exact. This function is used to correct the numerical error. 

    Firstly it tries to approximate each of the coefficients in the solution by continued fraction
    so that the coefficients are rational numbers. If it still fails, it will try to solve a rational
    linear system to find the exact solution.

    Parameters
    -------
    poly: sp.polys.Poly
        The target polynomial.
    y: List[float]
        The coefficients of the basis.
    basis: List[LinearBasis]
        The collection of basis.
    multiplier: sp.Expr
        The multiplier such that poly * multiplier = sum(y_i * basis_i).
    symmetry: PermutationGroup
        Every term will be wrapped by a cyclic sum of symmetryutation group.
    homogenizer: Optional[sp.Symbol]
        The homogenizer of the polynomial.
    """

    # first use the continued fraction to approximate the coefficients
    y_mask = np.abs(y).max() * zero_tol
    y = rationalize_array(y, y_mask, reliable = True)
    y, basis = _filter_zero_y(y, basis)

    reduced_arrays = _basis_as_matrix(basis)

    target = arraylize_sp(poly * multiplier.doit(), symmetry=symmetry, expand_cyc=True)
    obtained = reduced_arrays * sp.Matrix(y)

    is_equal = False
    if target == obtained:
        is_equal = True
    else:
        # second try to solve the linear system
        try:
            reduced_basis = basis
            reduced_y = reduced_arrays.LUsolve(target)

            if all(_ >= 0 for _ in reduced_y):
                reduced_y, reduced_basis = _filter_zero_y(reduced_y, reduced_basis)
                reduced_arrays = _basis_as_matrix(reduced_basis)
                obtained = reduced_arrays * sp.Matrix(reduced_y)
                if target == obtained:
                    is_equal = True
                    y, basis = reduced_y, reduced_basis
        except Exception as e:
            # print(e)
            is_equal = False

    solution = SolutionLinear(
        problem = poly,
        y = y,
        basis = basis,
        multiplier = multiplier,
        symmetry = symmetry,
        is_equal = is_equal,
    )
    return solution