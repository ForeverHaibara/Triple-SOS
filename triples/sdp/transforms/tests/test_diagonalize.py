from sympy import Matrix

from ...dual import SDPProblem

def test_diagonalize():
    # sdplib
    F0 = Matrix(4,4,[1,0,0,0,0,2,0,0,0,0,3,0,0,0,0,4])
    F1 = Matrix(4,4,[1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
    F2 = Matrix(4,4,[0,0,0,0,0,1,0,0,0,0,5,2,0,0,2,6])
    A = Matrix.hstack(F1.reshape(16, 1), F2.reshape(16, 1))
    b = -F0.reshape(16, 1)
    sdp = SDPProblem({'X': (b, A)})

    assert sdp.get_block_structures() == \
        {'X': [[0], [1], [2, 3]]}

    sdpp = sdp.constrain_block_structures()
    assert sorted(list(sdpp.size.values())) == [1, 1, 2]

    # test restoration of solution
    y = sdp.solve_obj([10, 20])
    assert y is not None and sdpp.y is not None
    assert abs(y[0] - 1) < 1e-4 and abs(y[1] - 1) < 1e-4
    assert max((sdp.S_from_y([1, 1])['X'] - sdp.S['X']).applyfunc(abs)) < 1e-4
    U, S = sdp.decompositions['X']
    assert max((U.T @ Matrix.diag(*S) @ U - sdp.S['X']).applyfunc(abs)) < 1e-4
