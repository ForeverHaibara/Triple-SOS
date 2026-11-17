"""
This file contains SDP facial reduction test examples
taken from references.

References
----------
[1] Permenter F. & Parrilo P.A. (2014), Partial facial reduction: simplified,
    equivalent SDPs via approximations of the PSD cone, Math. Program., 171, 1-54.
"""
import pytest
from sympy import symbols, linear_eq_to_matrix
from sympy import MutableDenseMatrix as Matrix
from sympy.abc import a, b, c, x, t

from ...dual import SDPProblem

def fd_example2():
    # [1] Section 4.3.2
    y1, y2, y3 = a, b, c
    A = Matrix(
        [[  1,      -y1,        0,  -y3],
         [-y1, 2*y2 - 1,       y3,    0],
         [  0,       y3, 2*y1 - 1,  -y2],
         [-y3,        0,       -y2,   1]]
    )
    sdp = SDPProblem.from_matrix({'A': A}, gens=(y1, y2, y3))
    return sdp

def fd_example4():
    # [1] Section 5.1.2 Example 2
    y1, y2 = a, b
    A = Matrix(
        [[y1,   0, y2,   0],
         [ 0, -y1,  0,   0],
         [y2,   0, y2,   0],
         [0,    0,  0, -y2]]
    )
    sdp = SDPProblem.from_matrix({'A': A}, gens=(y1, y2))
    return sdp

def fd_example5():
    # [1] Section 5.1.2 Example 3
    y1, y2, y3 = a, b, c
    A = Matrix(
        [[  1, -y1,  0,   0],
         [-y1,  y3,  0,   0],
         [  0,   0, y2,  y3],
         [  0,   0, y3, -y2]]
    )
    sdp = SDPProblem.from_matrix({'A': A}, gens=(y1, y2, y3))
    return sdp


def test_dual_linear_transform():
    sdp = fd_example2()
    # y1, y2, y3 = a, b, c

    # ensure y1 == y2
    for i in range(2):
        sdp._transforms.clear()
        if i == 0:
            A = Matrix([[1, 0], [1, 0], [0, 2]])
            B = Matrix([0, 0, -1])
            sdpp = sdp.constrain_affine(A, B)
        else:
            A = Matrix([[1, -1, 0]])
            B = Matrix([0])
            sdpp = sdp.constrain_equations(A, B)

        assert sdpp.dof == 2
        sdp.solve()

        if i == 0:
            assert sdpp.y == Matrix([2, 1])/2

        assert sdp.y == Matrix([1, 1, 0])
        assert sdp.S == sdp.S_from_y([1, 1, 0])
        U, S = sdp.decompositions['A']
        assert (U.T @ Matrix.diag(*S) @ U == sdp.S['A'])


def test_dual_matrix_transform():
    sdp = fd_example2()
    y1, y2, y3 = a, b, c

    U = Matrix([[1, 0], [-1, 0], [0, 1], [0, -1]])
    V = Matrix([[1, 0], [1, 0], [0, 1], [0, 1]])

    # V^T * A * V
    # == Matrix([[-2*y1 + 2*y2, 0], [0, 2*y1 - 2*y2]])

    for i in range(2):
        sdp._transforms.clear()
        if i == 0:
            sdpp = sdp.constrain_columnspace({'A': U})
        else:
            sdpp = sdp.constrain_nullspace({'A': V})
        assert sdpp.dof == 0 and sdpp.size == {'A': 2}
        sdp.solve()
        assert sdp.as_params() == {y1: 1, y2: 1, y3: 0}


def test_constrain_coordinate_faces():
    sdp = fd_example2()
    sdpp = sdp.constrain_coordinate_faces(level=2)
    assert sdpp.dof == 0 and sdpp.size == {'A': 2}
    sdpp.solve()
    assert sdp.as_params() == {a: 1, b: 1, c: 0}
    assert sdp.S['A'] == Matrix.diag(Matrix([[1,-1],[-1,1]]), Matrix([[1,-1],[-1,1]]))

    sdp = fd_example5()
    sdpp = sdp.constrain_coordinate_faces(level=1)
    assert sdpp.dof == 0 and sdpp.size == {'A': 1}
    sdpp.solve()
    assert sdp.as_params() == {a: 0, b: 0, c: 0}
    assert sdp.S['A'] == Matrix.diag(1,0,0,0)


def test_dual_congruence():
    sdp = fd_example2()

    Q = Matrix.diag(Matrix([[1, 1], [1, 2]]), Matrix([[1, -1], [-1, 2]]))
    V = Matrix([[1, 0], [1, 0], [0, 1], [0, 1]])

    sdpp = sdp.constrain_congruence({'A': Q})
    sdppp = sdp.constrain_nullspace({'A': V}, to_child=True)
    assert sdppp.dof == 0

    sdp.solve()
    assert sdpp.y == Matrix([1, 1, 0])
    assert sdp.y == Matrix([1, 1, 0])
    assert sdpp.S['A'] == Q.T * sdp.S['A'] * Q
    U, S = sdp.decompositions['A']
    assert (U.T @ Matrix.diag(*S) @ U == sdp.S['A'])


def test_linear_transform_solve_obj():
    # compute maximum t that `x^6 - 6x + 5 >= t*(x^4 - 4x + 3)`
    A = Matrix(4, 4, list(symbols('a:z')[:16]))
    sdp = SDPProblem.from_matrix({'A': A}, gens=symbols('a:z')[:16] + (t,))
    vec = Matrix([1, x, x**2, x**3])

    poly = (vec.T@ A @ vec)[0, 0].as_poly(x)
    poly = poly - (x**6 - 6*x + 5 - t*(x**4 - 4*x + 3)).as_poly(x)

    eqs, rhs = linear_eq_to_matrix(poly.coeffs(), sdp.gens)

    sdpp = sdp.constrain_symmetry()
    assert sdpp.constrain_symmetry() is sdpp
    sdp.constrain_equations(eqs, rhs, to_child=True)
    sdp.constrain_nullspace({'A': Matrix.ones(4, 1)}, to_child=True)
    assert sdp.get_last_child().size == {'A': 3}

    Q = Matrix([[1, 1, 1], [1, 2, 3], [2, 3, 6]]) # det == 2
    sdp.get_last_child().constrain_congruence({'A': Q})

    sdp.solve_obj(-t)
    assert abs(sdp.as_params()[t] - 1.5) < 1e-5


@pytest.mark.parametrize('principal', [True, False])
def test_dual_matrix_transform_nullspace_propagation(principal):
    W = Matrix([[1, 0], [0, 1], [2, -1]])
    S = W * Matrix([[a, b], [b, 1]]) * W.T
    sdp = SDPProblem.from_matrix({'A': S, 'B': Matrix([[2]]), 'zero': Matrix([[0]])})
    V = Matrix([-2, 1, 1])
    if principal:
        first = sdp.constrain_nullspace({'A': V, 'zero': Matrix([[1]])})
    else:
        first = sdp.constrain_columnspace({'A': W, 'zero': Matrix.zeros(1, 0)}, to_child=False)
    assert first.size == {'A': 2, 'B': 1}
    y = Matrix([3, 1])
    A, offset = sdp.common_transform(first).get_y_transform_from_child()
    parent_block = sdp.S_from_y(A*y + offset)['A']
    expected = parent_block.extract([1, 2], [1, 2]) if principal else W.T * parent_block * W
    assert first.S_from_y(y)['A'] == expected
    assert sdp.constrain_nullspace({'A': V, 'zero': Matrix([[1]])}) is first

    # The direction has zero entries on the retained indices, but its image
    # under the transpose of the lift is nonzero.
    second = sdp.constrain_nullspace({'A': Matrix([1, 0, 0])})
    assert second.size == {'A': 1, 'B': 1}
    assert second.dof == 0
    sdp.solve()
    assert sdp.y == Matrix([0, 0])
    assert sdp.S['A'] == W[:, 1] * W[:, 1].T
    U, diag = sdp.decompositions['A']
    assert U.T * Matrix.diag(*diag) * U == sdp.S['A']
