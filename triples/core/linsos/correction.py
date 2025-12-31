from typing import List, Tuple, Dict, Optional

import numpy as np
from sympy import Poly, Expr, Symbol, Rational
from sympy import MutableDenseMatrix as Matrix
from sympy.combinatorics import PermutationGroup
from sympy.polys.matrices import DomainMatrix

from .basis import LinearBasis, LinearBasisTangent, LinearBasisTangentEven
from ...sdp.arithmetic import reshape
from ...utils.roots.rationalize import rationalize_array


def odd_basis_to_even(basis: List[LinearBasis], symbols: Tuple[Symbol, ...],
        signs: Dict[Symbol, Tuple[Optional[int], Expr]]) -> List[LinearBasis]:
    mapping = [signs[s][1] if signs[s][0] == 1 else s for s in symbols]
    new_basis = []
    for b in basis:
        if isinstance(b, LinearBasisTangentEven) or not isinstance(b, LinearBasisTangent):
            # already converted / no need to convert
            new_basis.append(b)
        else:
            new_basis.append(b.to_even(mapping))
    return new_basis


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

def _basis_as_matrix(basis: List[LinearBasis], symmetry: PermutationGroup) -> Matrix:
    """
    Extract the array representations of each basis and stack them into a matrix.
    """
    mat = [b.as_array_sp(expand_cyc=True, symmetry=symmetry) for b in basis]
    mat = reshape(Matrix(mat), (len(mat), mat[0].shape[0])).T
    return mat

def _add_regularizer(mat: Matrix, num_multipliers: int) -> Matrix:
    """
    Add a regularizer row to the matrix.
    """
    regularizer = Matrix([[0] * (mat.shape[1] - num_multipliers) + [1] * num_multipliers])
    mat = Matrix.vstack(mat, regularizer)
    return mat

def linear_correction(
    y: List[float] = [],
    basis: List[LinearBasis] = [],
    num_multipliers: int = 1,
    symmetry: PermutationGroup = PermutationGroup(),
    zero_tol: float = 1e-6,
) -> Tuple[Matrix, Matrix, bool]:
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
    homogenizer: Optional[Symbol]
        The homogenizer of the polynomial.
    """

    # first use the continued fraction to approximate the coefficients
    y_mask = np.abs(y).max() * zero_tol
    y = rationalize_array(y, y_mask, reliable = True)
    y, basis, num_multipliers = _filter_zero_y(y, basis, num_multipliers)
    reduced_arrays = _basis_as_matrix(basis, symmetry=symmetry)

    # add a regularizer row
    # regularizer = Matrix([[0] * (reduced_arrays.shape[1] - num_multipliers) + [1] * num_multipliers])
    # reduced_arrays = Matrix.vstack(reduced_arrays, regularizer)
    reduced_arrays = _add_regularizer(reduced_arrays, num_multipliers)

    target = Matrix.zeros(reduced_arrays.shape[0], 1)
    target[-1, 0] = 1 # sum of coefficients of the multipliers should be 1
    obtained = reduced_arrays * Matrix(y)

    is_equal = False
    if target == obtained:
        is_equal = True
    else:
        # second try to solve the linear system
        try:
            reduced_basis = basis
            reduced_y = LUsolve(reduced_arrays, target)

            if all(_ >= 0 for _ in reduced_y):
                reduced_y, reduced_basis, num_multipliers = _filter_zero_y(reduced_y, reduced_basis, num_multipliers)
                reduced_arrays = _basis_as_matrix(reduced_basis, symmetry=symmetry)
                reduced_arrays = _add_regularizer(reduced_arrays, num_multipliers)
                reduced_y = Matrix(reduced_y)
                if _is_Ax_equal_to_b(reduced_arrays, reduced_y, target):
                    is_equal = True
                    y, basis = reduced_y, reduced_basis
        except Exception as e:
            # raise e
            is_equal = False

    return y, basis, is_equal

def LUsolve(A: Matrix, b: Matrix) -> Matrix:
    """
    Solve the linear system Ax = b. Handle irrational cases cleverly.
    When the matrix contains algebraic numbers like sqrt(2)+1,
    regular routines like A.LUsolve(b) will lead to nested fractions,
    which will be extremely slow. We need to convert them to DomainMatrix
    and solve the linear system in the extension field.
    """
    # if all(isinstance(_, Rational) for _ in A) and all(isinstance(_, Rational) for _ in b):
    #     return A.LUsolve(b)

    A2 = None
    try:
        A2 = DomainMatrix.from_Matrix(A, extension=True)
        b2 = DomainMatrix.from_Matrix(b, extension=True)

        new_domain = A2.domain.unify(b2.domain).get_field()
        A2 = A2.convert_to(new_domain)
        b2 = b2.convert_to(new_domain)
    except Exception as e:
        A2 = None
    if A2 is None: # fallback to default
        return A.LUsolve(b)

    x = A2.lu_solve(b2)
    x = x.to_Matrix()
    return x

def _is_Ax_equal_to_b(A: Matrix, x: Matrix, b: Matrix) -> bool:
    """
    Check if Ax = b. For rational cases, using A * x == b is enough. However,
    for irrational cases, we need to handle more carefully. For example,
    we need to check equations between algebraic numbers: 1/(sqrt(2)+1) == sqrt(2)-1.
    This will not be True if we use A * x == b directly, so we convert them to DomainMatrix
    given an extension field.
    """
    # return A * x == b
    try:
        A2 = DomainMatrix.from_Matrix(A, extension=True)
        x2 = DomainMatrix.from_Matrix(x, extension=True)
        b2 = DomainMatrix.from_Matrix(b, extension=True)

        new_domain = A2.domain.unify(x2.domain).unify(b2.domain)#.get_field()
        A2 = A2.convert_to(new_domain)
        x2 = x2.convert_to(new_domain)
        b2 = b2.convert_to(new_domain)
    except Exception as e:
        A2 = None

    if A2 is None: # fallback to default
        return A * x == b

    return A2 * x2 == b2
