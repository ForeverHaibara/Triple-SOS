from sympy.abc import a, b, c, d, x, y, z

from ..sdpsos import SDPSOS
from ....testing.doctest_parser import solution_checker

import pytest


class SDPSOSProblems:
    """
    Each of the problem must return a tuple of (ineq_constraints, eq_constraints, signs)
    """
    @classmethod
    def collect(cls):
        return {k: getattr(cls, k) for k in dir(cls) if k.startswith('problem')}

    @classmethod
    def problem_large_coeffs(cls):
        # Xia, Bican, and Lu Yang. "Automated inequality proving and discovering".
        # World Scientific, 2016. Example 10.10.
        return 2572755344*x**4 - 20000000*x**3*y - 6426888360*x**3*z + 30000000*x**2*y**2 \
            + 5315682897*x**2*z**2 - 20000000*x*y**3 - 1621722090*x*z**3 + 170172209*y**4 \
            - 1301377672*y**3*z + 3553788598*y**2*z**2 - 3864133016*y*z**3 + 1611722090*z**4, [], []

    @classmethod
    def problem_lax_lax(cls):
        # Lax-lax form
        # test degree-lifting and rationalization
        return a*(a-b)*(a-c)*(a-d)+b*(b-c)*(b-d)*(b-a)\
            +c*(c-d)*(c-a)*(c-b)+d*(d-a)*(d-b)*(d-c)+a*b*c*d, [], []


@pytest.mark.slow
@pytest.mark.parametrize("problem", SDPSOSProblems.collect().values(),
    ids=SDPSOSProblems.collect().keys())
def test_sdpsos(problem):
    sol = SDPSOS(*problem())
    assert solution_checker(sol)
