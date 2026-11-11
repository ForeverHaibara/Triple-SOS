import numpy as np
import sympy as sp
from sympy.abc import a, x, t
from sympy.matrices import MutableDenseMatrix as Matrix

import pytest

from ..backends import SDPError
from ..dual import SDPProblem

class DualObjProblems:
    """
    Each of the problem must return a tuple of (sdp, obj, constraints, val)
    """
    @classmethod
    def collect(cls):
        return {k: getattr(cls, k) for k in dir(cls) if k.startswith('problem')}

    @classmethod
    def problem_conic_opt(cls):
        """Jiawang Nie, Moment and Polynomial Optimization, Example 1.6.2"""
        y1, y2, y3 = sp.symbols('y1 y2 y3')
        sdp = SDPProblem.from_matrix([
            Matrix([[1,y1,y2],[y1,2,y3],[y2,y3,3]]),
            Matrix([[y1+y3,0,0,y1-1], [0,y1+y3,0,y2], [0,0,y1+y3,y3-1], [y1-1,y2,y3-1,y1+y3]]) # SOC
        ])
        return sdp, y1-y3, [y1+y2+y3-3, -1>-y1-y2, -1>-y2-y3], -2.

    @classmethod
    def problem_quartic_opt(cls):
        """Minimize the parameter t for a nonnegative univariate quartic polynomial."""
        poly = ((x-2)**2 * (x**2+(1+t)*x+5-t)).as_poly(x)
        c4, c3, c2, c1, c0 = poly.all_coeffs()
        sdp = SDPProblem.from_matrix(
            {0: Matrix([[c0, c1/2, a], [c1/2, c2-2*a, c3/2], [a, c3/2, c4]])})
        sdp.constrain_nullspace({0: Matrix([1,2,4])}) # [1,x,x^2] is in its nullspace
        return sdp, t+3, [], -2*7**.5 # optimal t = -3 - 2*7**.5

@pytest.mark.parametrize('problem', DualObjProblems.collect().values(),
    ids=DualObjProblems.collect().keys())
def test_dual_solve_obj(problem, tol=1e-5):
    sdp, obj, constraints, val = problem()
    assert sdp.solve_obj(obj, constraints)
    val2 = obj.xreplace(sdp.as_params())
    assert abs(val2 - val) < tol


def test_dual_numerical_history():
    pytest.importorskip('clarabel')
    sdp = SDPProblem.from_matrix({'S': Matrix([[a, 1], [1, a]]),
                                  'bound': Matrix([[2-a]])})
    with pytest.raises(SDPError) as exc:
        sdp.solve_obj(a, solver='clarabel', kwargs={'max_iters': 1})
    first, = sdp.numerical_results
    assert first.result is exc.value.result
    assert first.relaxed == 0
    assert first.result.inaccurate
    assert np.allclose(sdp._ys[0], first.result.y)

    assert sdp.solve(solver='clarabel')
    attempt = sdp.numerical_results[-1]
    result = attempt.result
    assert attempt.relaxed == 1
    assert tuple(key for key, _ in attempt.block_scales) == ('S', 'bound')
    y = sdp._ys[-1]
    assert np.allclose(y, result.y[:-1])
    matrices = sdp.S_from_y(y)
    for (key, scale), slack in zip(attempt.block_scales, result.slack):
        restored = scale * (slack + result.y[-1]*np.eye(slack.shape[0]))
        assert np.allclose(restored, np.array(matrices[key], dtype=float), atol=1e-6)
    dual = [matrix/scale for matrix, (_, scale) in zip(result.dual, attempt.block_scales)]
    assert abs(np.trace(dual[0]) - dual[1][0, 0]) < 1e-6
