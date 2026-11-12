from sympy.abc import x, y, z

from ..sos import SOSPoly

import pytest


def test_sos_recovery_with_face_equalities(capsys):
    pytest.importorskip('clarabel')
    f = ((x-z)**2 + (y-z)**2) * (x**2 + y**2 + z**2) / 13
    sos = SOSPoly(f.as_poly(x, y, z), (x, y, z), qmodule=[1], roots=[])
    sos.construct()
    assert sos.solve(solver='clarabel', verbose=True, normalize=False,
        recovery_with_equalities=True, kwargs={'max_iters': 3}) is not None
    assert 'Recovered SDP with explicit face equations' in capsys.readouterr().out
    assert (sos.as_solution().solution.doit() - f).expand() == 0
