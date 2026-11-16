from sympy.abc import a, b, c, d, r, s, x, y, z

from ..sdpsos import SDPSOS
from ....sdp.backends.clarabel_sdp import DualBackendCLARABEL
from ....testing.doctest_parser import solution_checker
from ....utils import pl

import pytest


# Available when python >= 3.7
HAS_CLARABEL = DualBackendCLARABEL.is_available()


def depends_on_clarabel(func):
    if not HAS_CLARABEL:
        def wrapper(*args, **kwargs):
            return x**2, [], []
        return wrapper
    return func


class SDPSOSProblems:
    """
    Each of the problem must return a tuple of (ineq_constraints, eq_constraints, signs)
    """
    @classmethod
    def collect(cls):
        return {k: getattr(cls, k) for k in dir(cls) if k.startswith('problem')}

    @classmethod
    @depends_on_clarabel
    def problem_large_coeffs(cls):
        # Xia, Bican, and Lu Yang. "Automated inequality proving and discovering".
        # World Scientific, 2016. Example 10.10.

        # cvxopt may fail
        return 2572755344*x**4 - 20000000*x**3*y - 6426888360*x**3*z + 30000000*x**2*y**2 \
            + 5315682897*x**2*z**2 - 20000000*x*y**3 - 1621722090*x*z**3 + 170172209*y**4 \
            - 1301377672*y**3*z + 3553788598*y**2*z**2 - 3864133016*y*z**3 + 1611722090*z**4, [], []

    @classmethod
    def problem_lax_lax(cls):
        # Lax-lax form
        # test degree-lifting and rationalization
        return a*(a-b)*(a-c)*(a-d)+b*(b-c)*(b-d)*(b-a)\
            +c*(c-d)*(c-a)*(c-b)+d*(d-a)*(d-b)*(d-c)+a*b*c*d, [], []

    @classmethod
    def problem_joint_face_recovery(cls):
        # vasile 32012
        return 4 - a*b**2*c**3, [a, b, c, b-a, c-b], [a*b + b*c + c*a - 3]

    @classmethod
    @depends_on_clarabel
    def problem_joint_dual_face_recovery(cls):
        g, h = r**2 - 2, s**2 - 2
        return (g*h*(r**2 - 1)*(s**2 - 1) + (r*s + r + s + 2)**2).expand(), [g, h], []

    @classmethod
    def problem_schur6_enhanced(cls):
        return pl('s(a4(a-b)(a-c))-5p(a-b)2'), [a,b,c], [], {'lift_degree_limit': 2}

    @classmethod
    def problem_schur6_enhanced_scaled(cls):
        return pl('s(a4(a-b)(a-c))-5p(a-b)2').as_expr().xreplace({
            b: 2*b, c: 3*c
        }), [a,b,c], [], {'lift_degree_limit': 2}

    @classmethod
    def problem_isosceles_right(cls):
        return pl('s(a3(a-b)(a-c)(a2-b2-c2)2)'), [a,b,c], [], {'lift_degree_limit': 2}

    # @classmethod
    # def problem_symmetric_quaternary_degree6(cls):
    #     # Slow. Need not test.
    #     return pl('s(((((a+b-c)(a+b-d)-2ab)2+2(ab)2)-(a+b-c-d)2(c-d)2)(a-b)2)',
    #         (a, b, c, d), 'sym'), [], [], {'lift_degree_limit': 4}


@pytest.mark.slow
@pytest.mark.parametrize("problem", SDPSOSProblems.collect().values(),
    ids=SDPSOSProblems.collect().keys())
def test_sdpsos(problem):
    solver = 'clarabel' if HAS_CLARABEL else None
    data = problem()
    kwargs = {}
    if len(data) >= 4:
        data, kwargs = data[:3], data[3]
    sol = SDPSOS(*data, solver=solver, **kwargs)
    assert solution_checker(sol)
