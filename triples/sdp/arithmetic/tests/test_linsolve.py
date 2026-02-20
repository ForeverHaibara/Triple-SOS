from sympy.matrices import MutableDenseMatrix as Matrix

from ..linsolve import (
    solve_csr_linear, solve_column_separated_linear, solve_undetermined_linear
)

def test_solve_csr_linear():
    A = Matrix([[0,1,3,0,0,2], [1,0,-1,0,0,3], [0,0,2,3,4,5], [0,0,0,5,7,0]])
    b = Matrix([7,11,23,97])

    v, M = solve_csr_linear(A, b)
    assert M.shape == (6, 2)
    assert v.shape == (6, 1)
    assert (A * M).is_zero_matrix
    assert (A * v - b).is_zero_matrix

    v2, M2 = solve_undetermined_linear(A, b)
    assert (v2 - v).is_zero_matrix
    assert (M2 - M).is_zero_matrix

    # test x0_equal_indices
    cases = [
        [[0, 1, 3]],
        [[1, 5], [2, 4]]
    ]
    for i, x0_equal_indices in enumerate(cases):
        v, M = solve_csr_linear(A, b, x0_equal_indices=x0_equal_indices)
        assert (A * M).is_zero_matrix, f'wrong case {i}'
        assert (A * v - b).is_zero_matrix, f'wrong case {i}'

        for group in x0_equal_indices:
            for j in group[1:]:
                assert M[j, :] == M[group[0], :], f'wrong case {i}'
                assert v[j] == v[group[0]], f'wrong case {i}'

    try:
        v, M = solve_csr_linear(A, b, x0_equal_indices=[[1, 4, 5], [2, 3]])
        assert False, "should raise ValueError"
    except ValueError:
        # okay
        pass

    # TODO: test other domains


def test_solve_column_separated_linear():
    A = Matrix([[0,1,0,0,1,0,0], [2,0,0,4,0,0,-3], [0,0,3,0,0,0,0]])
    b = Matrix([7,11,23])

    v, M = solve_column_separated_linear(A, b)
    assert M.shape == (7, 4)
    assert v.shape == (7, 1)
    assert (A * M).is_zero_matrix
    assert (A * v - b).is_zero_matrix

    v2, M2 = solve_csr_linear(A, b)
    assert (v2 - v).is_zero_matrix
    assert (M2 - M).is_zero_matrix

    v2, M2 = solve_undetermined_linear(A, b)
    assert (v2 - v).is_zero_matrix
    assert (M2 - M).is_zero_matrix
