from ..polysolve import univar_realroots
from sympy.abc import b
from sympy import Integer, sqrt

def test_univar_realroots():
    p1 = b**4 - sqrt(2)*b**3 + (Integer(55)/6 - 6*sqrt(2))*b**2 \
        + (-Integer(76)/9 + 53*sqrt(2)/9)*b + Integer(17)/9 - 4*sqrt(2)/3
    p1 = p1.as_poly(b, extension = sqrt(2))
    rts = univar_realroots(p1, p1.gen)
    assert len(rts) == 2
    for rt in rts:
        assert rt.args[1] in (Integer(0), Integer(1))
