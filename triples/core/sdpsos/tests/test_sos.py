from sympy.abc import x, y, z

from ..sos import SOSPoly

import pytest


@pytest.mark.parametrize('ideal, homogeneous, degrees', [
    ([x-y], True, {2}),
    ([x-y], None, {0, 1, 2}),
    ([x-y-1], False, {0, 1, 2}),
])
def test_sos_zero_target_basis(ideal, homogeneous, degrees):
    sos = SOSPoly(0, (x, y), qmodule=[1, -x**2], ideal=ideal,
                  degree=4, is_homogeneous=homogeneous)
    equations = sos._get_equation_system()
    assert {sum(m) for m in sos._qmodule_bases[0]._basis} == degrees
    assert {sum(m) for m in sos._ideal_bases[0]._basis} == (
        {3} if homogeneous else {0, 1, 2, 3})
    assert equations.rhs.is_zero_matrix


def test_sos_recovery_with_face_equalities(capsys):
    pytest.importorskip('clarabel')
    f = ((x-z)**2 + (y-z)**2) * (x**2 + y**2 + z**2) / 13
    sos = SOSPoly(f.as_poly(x, y, z), (x, y, z), qmodule=[1], roots=[])
    sos.construct()
    assert sos.solve(solver='clarabel', verbose=True, normalize=False,
        recovery_with_equalities=True, kwargs={'max_iters': 3}) is not None
    assert 'Recovered SDP with explicit face equations' in capsys.readouterr().out
    assert (sos.as_solution().solution.doit() - f).expand() == 0
