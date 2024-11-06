from typing import List, Optional, Tuple

import numpy as np
import sympy as sp
from sympy.combinatorics import PermutationGroup

from .basis import LinearBasis
from .solution import SolutionLinear
from ...utils.roots.rationalize import rationalize_array

def _filter_zero_y(y: List[float], basis: List[LinearBasis], num_multipliers: int) -> Tuple[List[float], List[LinearBasis], int]:
    """
    Filter out the zero coefficients.
    """
    reduced_y, reduced_basis = [], []
    for v, b in zip(y, basis):
        if v != 0:
            reduced_y.append(v)
            reduced_basis.append(b)
    reduced_num = sum(v != 0 for v in y[-num_multipliers:])

    return reduced_y, reduced_basis, reduced_num

def _basis_as_matrix(basis: List[LinearBasis], symmetry: PermutationGroup) -> sp.Matrix:
    """
    Extract the array representations of each basis and stack them into a matrix.
    """
    mat = [b.as_array_sp(expand_cyc=True, symmetry=symmetry) for b in basis]
    mat = sp.Matrix(mat).reshape(len(mat), mat[0].shape[0]).T
    return mat

def _add_regularizer(mat: sp.Matrix, num_multipliers: int) -> sp.Matrix:
    """
    Add a regularizer row to the matrix.
    """
    regularizer = sp.Matrix([[0] * (mat.shape[1] - num_multipliers) + [1] * num_multipliers])
    mat = sp.Matrix.vstack(mat, regularizer)
    return mat

def linear_correction(
        y: List[float] = [],
        basis: List[LinearBasis] = [],
        num_multipliers: int = 1,
        symmetry: PermutationGroup = PermutationGroup(),
        zero_tol: float = 1e-6,
    ) -> Tuple[sp.Matrix, sp.Matrix, bool]:
    """
    Linear programming is a numerical way to solve the SOS problem. However, we require
    the solution to be exact. This function is used to correct the numerical error. 

    Firstly it tries to approximate each of the coefficients in the solution by continued fraction
    so that the coefficients are rational numbers. If it still fails, it will try to solve a rational
    linear system to find the exact solution.

    Parameters
    -----------
    y: List[float]
        The coefficients of the basis.
    basis: List[LinearBasis]
        The collection of basis.
    num_multipliers: int
        The number of multipliers.
    symmetry: PermutationGroup
        Every term will be wrapped by a cyclic sum of symmetryutation group.
    homogenizer: Optional[sp.Symbol]
        The homogenizer of the polynomial.
    """

    # first use the continued fraction to approximate the coefficients
    y_mask = np.abs(y).max() * zero_tol
    y = rationalize_array(y, y_mask, reliable = True)
    y, basis, num_multipliers = _filter_zero_y(y, basis, num_multipliers)
    reduced_arrays = _basis_as_matrix(basis, symmetry=symmetry)

    # add a regularizer row
    # regularizer = sp.Matrix([[0] * (reduced_arrays.shape[1] - num_multipliers) + [1] * num_multipliers])
    # reduced_arrays = sp.Matrix.vstack(reduced_arrays, regularizer)
    reduced_arrays = _add_regularizer(reduced_arrays, num_multipliers)

    target = sp.zeros(reduced_arrays.shape[0], 1)
    target[-1, 0] = 1 # sum of coefficients of the multipliers should be 1
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
                reduced_y, reduced_basis, num_multipliers = _filter_zero_y(reduced_y, reduced_basis, num_multipliers)
                reduced_arrays = _basis_as_matrix(reduced_basis, symmetry=symmetry)
                reduced_arrays = _add_regularizer(reduced_arrays, num_multipliers)
                obtained = reduced_arrays * sp.Matrix(reduced_y)
                if target == obtained:
                    is_equal = True
                    y, basis = reduced_y, reduced_basis
        except Exception as e:
            # raise e
            is_equal = False

    return y, basis, is_equal