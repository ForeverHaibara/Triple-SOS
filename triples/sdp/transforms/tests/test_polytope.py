from sympy.matrices import MutableDenseMatrix as Matrix

from .test_linear import fd_example2, fd_example5

def test_sdp_row_extraction():
    sdp = fd_example5()

    sdpp = sdp.constrain_zero_diagonals()\
              .constrain_zero_diagonals()
    assert sdpp.dof == 0 and sdpp.size == {'A': 1}

    sdp.solve()
    assert sdpp.S == {'A': Matrix([[1]])}
    assert sdp.y == Matrix([0, 0, 0])
    assert sdp.S == sdp.S_from_y([0, 0, 0])
    U, S = sdp.decompositions['A']
    assert (U.T @ Matrix.diag(*S) @ U == sdp.S['A'])


def test_composed_row_extraction():
    sdp = fd_example2()

    U = Matrix([[1, 0, 1, 0], [1, 0, -1, 0], [0, 1, 0, 1], [0, 1, 0, -1]])
    sdpp = sdp.constrain_congruence({'A': U})\
            .constrain_zero_diagonals(masks={'A': [0, 1]})

    assert sdpp.dof == 0 and sdpp.size == {'A': 2}

    sdp.solve()
    assert sdp.y == Matrix([1, 1, 0])
    assert sdp.S == sdp.S_from_y([1, 1, 0])
    U, S = sdp.decompositions['A']
    assert (U.T @ Matrix.diag(*S) @ U == sdp.S['A'])
