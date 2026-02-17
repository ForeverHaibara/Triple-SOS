"""
This file contains SDP facial reduction test examples
taken from references.

References
----------
[1] Permenter F. & Parrilo P.A. (2014), Partial facial reduction: simplified,
    equivalent SDPs via approximations of the PSD cone, Math. Program., 171, 1-54.
"""
from sympy import MutableDenseMatrix as Matrix
from sympy.abc import a, b, c

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
    A = Matrix.diag(
        [[  1, -y1,  0,   0],
         [-y1,  y3,  0,   0],
         [  0,   0, y2,  y3],
         [  0,   0, y3, -y2]]
    )
    sdp = SDPProblem.from_matrix({'A': A}, gens=(y1, y2, y3))
    return sdp


def test_dual_linear_transform():
    sdp = fd_example2()
    y1, y2, y3 = a, b, c

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
