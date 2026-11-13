from sympy import Matrix, QQ
from sympy.polys.rings import ring
from sympy.abc import x

from ..manifold import _parametrization_jets
from ..sos import SOSPoly


def test_gram_jets_require_the_line_to_satisfy_the_ideal():
    # Along (0, t), F = t**5 and the weight is t. The coefficients
    # of 1 and t must be in the kernel, while the t**2 term can survive.
    R, t = ring('t', QQ)
    target = {(0, 5): QQ.one}
    y = {(0, 1): QQ.one}
    basis = [(0, 0), (1, 0), (0, 1), (0, 2)]
    blocks = [(0, y, basis)]
    parametrization = (R.zero, t)
    assert list(_parametrization_jets(target, [], blocks, parametrization)) == [
        (0, (1, 0, 0, 0)), (0, (0, 0, 1, 0))]

    assert not list(_parametrization_jets(target, [y], blocks, parametrization))
    assert list(_parametrization_jets(target, [{(1, 0): QQ.one}], blocks,
                                    parametrization)) == [(0, (1, 0, 0, 0)), (0, (0, 0, 1, 0))]


def test_gram_jets_on_a_line_of_zeros():
    R, t = ring('t', QQ)
    basis = [(2, 0), (1, 1), (0, 2)]
    target = {(0, 4): QQ.one}
    blocks = [(0, {(0, 0): QQ.one}, basis), (1, {(0, 1): QQ.one}, basis)]
    assert list(_parametrization_jets(target, [], blocks, (1+t, R.zero))) == [
        (0, (1, 0, 0)), (0, (2, 0, 0)), (0, (1, 0, 0))]


def test_gram_jets_stop_at_first_possible_cancellation():
    _, t = ring('t', QQ)
    # 0 = t*M0 - t**3*M1. M0 can equal t**2 and M1 can equal 1.
    blocks = [
        (0, {(1,): QQ.one}, [(0,), (1,), (2,)]),
        (1, {(3,): -QQ.one}, [(0,)]),
    ]
    assert list(_parametrization_jets({}, [], blocks, (t,))) == [(0, (1, 0, 0))]


def test_gram_jets_include_opposite_signs_in_the_same_orbit():
    _, t = ring('t', QQ)
    # The weight x-y and its swapped copy cancel at every point for M=1.
    blocks = [(0, {(1, 0): QQ.one, (0, 1): -QQ.one}, [(0, 0)])]
    assert not list(_parametrization_jets({}, [], blocks, (t.ring.one, t),
                                         [(0, 1), (1, 0)]))


def test_gram_jets_on_a_nonlinear_parametrization():
    _, t = ring('t', QQ)
    # On y=x**2 the basis y has a t**2 coefficient. It must vanish
    # when the target y**3 has order six.
    target = {(0, 3): QQ.one}
    ideal = {(0, 1): QQ.one, (2, 0): -QQ.one}
    blocks = [(0, {(0, 0): QQ.one}, [(0, 0), (1, 0), (0, 1)])]
    assert list(_parametrization_jets(target, [ideal], blocks, (t, t**2))) == [
        (0, (1, 0, 0)), (0, (0, 1, 0)), (0, (0, 0, 1))]


def test_sospoly_signed_generators_preserve_feasible_gram_points():
    # 0 = x*M0 - x**3*M1 admits M0=x**2 and M1=1.
    # The only necessary numerator direction in the mixed basis (1, x) is 1.
    sos = SOSPoly(0, (x,), qmodule=[x, -x**3], degree=3, roots=[(0,), (1,)])
    sos.construct()
    assert sos.sdpp.dof == 1
    value = Matrix([1])
    if any(s[0, 0] < 0 for s in sos.sdpp.S_from_y(value).values() if s.rows):
        value = -value
    assert sos.sdpp.register_y(value, propagate_to_parent=True)
    assert sos.sdp.S[(sos, 1)] == Matrix([[1]])
