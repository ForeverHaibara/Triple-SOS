import pytest
import numpy as np
from sympy import Matrix, Rational
from sympy.abc import a, b

from ..dual import SDPProblem
from ..facial import find_exposing_nullspace


@pytest.mark.parametrize('sign', [0, 1, -1])
def test_joint_exposure(sign):
    # a-p >= 0 and p-a >= 0 enforce a-p == 0
    # rotate the gram matrix so that the face is nontrivial
    p, q = Rational(1, 13), Rational(1, 17)
    gram = Matrix([[a-p, b-q], [b-q, 1]])
    T = Matrix([[1, sign], [1, -sign]]) if sign else Matrix.eye(2)
    sdp = SDPProblem.from_matrix({'gram': T.T * gram * T, 'bound': Matrix([[p-a]])})

    directions = find_exposing_nullspace(sdp._x0_and_space)
    assert directions == {'gram': Matrix([1, sign]), 'bound': Matrix([1])}
    # The supplied bound direction needs a second atom to expose the face.
    for level in (0, 1, 2):
        result = find_exposing_nullspace(
            sdp._x0_and_space, directions=[('bound', Matrix([1]))],
            coordinate_level=level,
        )
        assert result == (directions if level == 2 or (level == 1 and not sign) else {})
    child = sdp.constrain_nullspace(directions)
    assert child.dof == 0
    assert child.solve() == Matrix.zeros(0, 1)
    assert sdp.y == Matrix([p, q])
    assert sdp.S['gram'] == T.T * Matrix.diag(0, 1) * T


def test_strictly_feasibility():
    # Strictly feasible SDP has no face.
    sdp = SDPProblem.from_matrix([Matrix([[a]]), Matrix([[1-a]])])
    assert find_exposing_nullspace(sdp._x0_and_space) == {}


def test_dual_exposure():
    gram = (3 * Matrix.eye(3) - Matrix.ones(3)) * a
    sdp = SDPProblem.from_matrix({'gram': gram})
    blocks = sdp._x0_and_space
    assert find_exposing_nullspace(
        blocks, dual={'gram': np.ones((3, 3))},
    ) == {'gram': Matrix.ones(3, 1)}


@pytest.mark.parametrize('bound', [0, Rational(1, 10**12)])
def test_spectral_joint_exposure(bound):
    # The only exposed direction in the Gram block has support three.
    gram = (3 + a) * Matrix.eye(3) - Matrix.ones(3)
    sdp = SDPProblem.from_matrix({
        'gram': gram, 'bound': Matrix([[bound-a]]), 'constant': Matrix([[1]]),
    })
    blocks = sdp._x0_and_space
    assert find_exposing_nullspace(blocks) == {}

    # This dual hint proposes a direction whose form is a, rather than
    # identically zero. The shared verifier must proceed to the joint search.
    directions = find_exposing_nullspace(
        blocks, y=np.array([1e-7]), dual={'gram': np.ones((3, 3))},
    )
    if bound:
        # The same approximate kernel is present, but the pencil has interior.
        assert not directions
    else:
        assert directions == {'gram': Matrix.ones(3, 1), 'bound': Matrix([[1]])}
        child = sdp.constrain_nullspace(directions)
        assert child.dof == 0
        child.solve()
        assert sdp.y == Matrix([0])
