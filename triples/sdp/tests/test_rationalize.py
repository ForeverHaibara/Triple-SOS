import numpy as np
import pytest
from sympy import Matrix, Rational
from sympy.abc import a

from .. import SDPInfeasibleError
from ..arithmetic import ArithmeticTimeout
from ..backends import SDPError
from ..dual import SDPProblem
from ..rationalize import DualRationalizer


def _check_strong_separator(problem, certificate, decompositions):
    assert certificate
    assert certificate.keys() == decompositions.keys()
    dof = problem.dof
    pairing = Matrix.zeros(1, dof + 1)
    for key, Z in certificate.items():
        block, U, D = decompositions[key]
        assert all(x.is_Rational for x in U)
        assert all(w.is_Rational and w > 0 for w in D)
        assert Z == block == U.T @ Matrix.diag(*D) @ U
        a, A = problem._x0_and_space[key]
        pairing += Z.reshape(1, a.rows) * a.row_join(A)
    assert pairing == Matrix([[-1] + [0]*dof])


@pytest.mark.parametrize('bound', [-Rational(1, 13), 0, Rational(1, 13)])
def test_rational_separator(bound):
    gram = (3 + a)*Matrix.eye(3) - Matrix.ones(3)
    sdp = SDPProblem.from_matrix({'gram': gram, 'bound': Matrix([[bound-a]])})
    # A slightly perturbed dual direction also proposes atoms for the
    # feasible models, including the one on the boundary.
    dual = {'gram': np.ones((3, 3)) + np.diag([0., 0., 1e-7]), 'bound': np.ones((1, 1))}
    certificate = DualRationalizer(sdp).rationalize_infeasibility(dual)
    if bound < 0:
        _check_strong_separator(sdp, *certificate)
    else:
        assert certificate is None


@pytest.mark.parametrize('retry', [False, True])
def test_solve_infeasibility_certification(retry):
    pytest.importorskip('clarabel')
    # A 25x25 block exceeds the separator search limit. Its explicit zero
    # exposes a 24x24 face on which the same contradiction can be proved.
    blocks = ([Matrix.diag(a, -2-5*a, 0, *([a]*22))] if retry
              else [Matrix([[a]]), Matrix([[-2-5*a]])])
    sdp = SDPProblem.from_matrix(blocks)
    recovery_rounds = int(retry)
    with pytest.raises(SDPInfeasibleError) as caught:
        sdp.solve(solver='clarabel', recovery_rounds=recovery_rounds)
    error = caught.value
    assert (error.problem is not sdp) == retry
    _check_strong_separator(error.problem, error.certificate, error.decompositions)

    with pytest.raises(SDPError) as caught:
        sdp.solve(solver='clarabel', recovery_rounds=recovery_rounds, certify_infeasibility=False)
    assert not isinstance(caught.value, SDPInfeasibleError)


def test_weak_infeasibility():
    # The dual block exposing e2 has zero pairing; imposing S*e2 = 0
    # would be inconsistent, but it cannot be normalized to pairing -1.
    sdp = SDPProblem.from_matrix({'gram': Matrix([[a, 1], [1, 0]])})
    dual = {'gram': np.diag([0., 1.])}
    assert DualRationalizer(sdp).rationalize_infeasibility(dual) is None


def test_rational_infeasibility():
    # The SDP is weakly feasible but there exists no rational solution.
    sdp = SDPProblem.from_matrix({
        'A': Matrix([[2, a], [a, 1]]),
        'B': Matrix([[a, 2], [2, 2*a]])
    }) # only solution: a = 2**.5
    s = 2**.5
    dual = {
        'A': np.array([[s, -2], [-2, 2*s]]),
        'B': np.array([[2, -s], [-s, 1]])
    }
    assert DualRationalizer(sdp).rationalize_infeasibility(dual) is None


def test_infeasibility_solve_timeout():
    sdp = SDPProblem.from_matrix([Matrix([[a]]), Matrix([[-2-5*a]])])
    dual = dict.fromkeys(sdp.size, np.ones((1, 1)))

    def remaining_time():
        raise ArithmeticTimeout()

    with pytest.raises(ArithmeticTimeout):
        DualRationalizer(sdp).rationalize_infeasibility(
            dual, time_limit=remaining_time)
