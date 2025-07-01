from sympy.abc import a, b, c, d, x, y, t
from sympy.combinatorics import SymmetricGroup

import pytest

from ..sos import SOSPoly

class SOSObjProblems:
    """
    Each of the problem must return a tuple of (sdp, obj, constraints, val)
    """
    @classmethod
    def collect(cls):
        return {k: getattr(cls, k) for k in dir(cls) if k.startswith('problem')}

    @classmethod
    def problem_sostools_constrained(cls):
        # https://arxiv.org/pdf/1310.4716 page 40,
        sos = SOSPoly(a+b+t, (a,b), [a,2*b-1], [a**2+b**2-1,2*b-2*a**2-1], degree=4)
        return sos, t, [], (1-7**.5)/2 - (7**.5/2 - 1)**.5

    @classmethod
    def problem_gloptipoly_six_hump_camel(cls):
        # https://arxiv.org/pdf/0709.2559 page 3,
        sos = SOSPoly(4*a**2+a*b-4*b**2-2.1*a**4+4*b**4+a**6/3+t, (a,b), [1])
        return sos, t, [], 1.03162845348988

    @classmethod
    def problem_schur(cls):
        # (Σa**4*(a-b)*(a-c)) + 1/3*(a-b)**2*(b-c)**2*(c-a)**2
        # = (Σ(a - b)**2*(6*a**2 + 4*a*b - 4*a*c + 6*b**2 - 4*b*c - 2*c**2)**2)/72
        sos = SOSPoly(a**4*(a-b)*(a-c)+b**4*(b-c)*(b-a)+c**4*(c-a)*(c-b)-t*(a-b)**2*(b-c)**2*(c-a)**2, (a,b,c),
                [1], symmetry=SymmetricGroup(3), roots=[(1,1,1),(1,1,0),(0,1,1),(1,0,1)])
        return sos, -t, [], 1/3


@pytest.mark.parametrize("problem", SOSObjProblems.collect().values(),
    ids=SOSObjProblems.collect().keys())
def test_sos_obj(problem, tol=1e-5):
    sdp, obj, constraints, val = problem()
    y = sdp.solve_obj(obj, constraints=constraints)
    val2 = obj.xreplace(sdp.as_params())
    assert abs(val - val2) < tol