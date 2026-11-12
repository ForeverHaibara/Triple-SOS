import pytest
from sympy import Matrix, Rational, symbols

from ..dual import SDPProblem
from ..facial import find_joint_exposing_nullspace


@pytest.mark.parametrize('sign', [0, 1, -1])
def test_joint_exposure(sign):
    # a-p >= 0 and p-a >= 0 enforce a-p == 0
    # rotate the gram matrix so that the face is nontrivial
    a, b = symbols('a b')
    p, q = Rational(1, 13), Rational(1, 17)
    gram = Matrix([[a-p, b-q], [b-q, 1]])
    T = Matrix([[1, sign], [1, -sign]]) if sign else Matrix.eye(2)
    sdp = SDPProblem.from_matrix({'gram': T.T * gram * T, 'bound': Matrix([[p-a]])})

    directions = find_joint_exposing_nullspace(sdp._x0_and_space, lambda: None)
    assert directions == {'gram': Matrix([1, sign]), 'bound': Matrix([1])}
    child = sdp.constrain_nullspace(directions)
    assert child.dof == 0
    assert child.solve() == Matrix.zeros(0, 1)
    assert sdp.y == Matrix([p, q])
    assert sdp.S['gram'] == T.T * Matrix.diag(0, 1) * T


def test_joint_exposure_on_strictly_feasible_pencil():
    a = symbols('a')
    sdp = SDPProblem.from_matrix([Matrix([[a]]), Matrix([[1-a]])])
    assert find_joint_exposing_nullspace(sdp._x0_and_space, lambda: None) == {}
