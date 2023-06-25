from typing import List

import numpy as np
import sympy as sp
from sympy.core.singleton import S

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

def _basis_as_matrix(basis):
    """
    Extract the array representations of each basis and stack them into a matrix.
    """
    mat = [b.array_sp for b in basis]
    mat = sp.Matrix(mat).reshape(len(mat), mat[0].shape[0]).T
    return mat

def linear_correction(
        poly: sp.polys.Poly,
        y: List[float] = [],
        basis: List[LinearBasis] = [],
        multiplier: sp.Expr = S.One,
        cyc: bool = True,
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
    cyc: bool
        Whether the problem is cyclic. Now we only support cyclic problems.
    """

    # first use the continued fraction to approximate the coefficients
    y_mask = np.abs(y).max() * 1e-6
    y = rationalize_array(y, y_mask, reliable = True)
    y, basis = _filter_zero_y(y, basis)

    reduced_arrays = _basis_as_matrix(basis)

    target = arraylize_sp(poly * multiplier.doit(), cyc = cyc)
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
        is_equal = is_equal,
    )
    return solution