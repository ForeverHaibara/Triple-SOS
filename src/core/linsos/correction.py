import numpy as np
import sympy as sp
from sympy.core.singleton import S

from .solution import SolutionLinear
from ...utils.basis_generator import arraylize, arraylize_sp
from ...utils.roots.rationalize import rationalize_array

def _filter_zero_y(y, basis):
    reduced_y, reduced_basis = [], []
    for v, b in zip(y, basis):
        if v != 0:
            reduced_y.append(v)
            reduced_basis.append(b)

    return reduced_y, reduced_basis

def _basis_as_matrix(basis):
    mat = [b.array_sp for b in basis]
    mat = sp.Matrix(mat).reshape(len(mat), mat[0].shape[0]).T
    return mat

def linear_correction(
        poly,
        y = None,
        basis = [],
        multiplier = S.One,
        cyc = True,
    ):

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