import numpy as np
import sympy as sp
from sympy import MutableDenseMatrix as Matrix
from sympy import Symbol

import pytest

from ..primal import SDPPrimal

class PrimalObjProblems:
    """
    Each of the problem must return a tuple of (sdp, obj, constraints, val)
    """
    @classmethod
    def collect(cls):
        return {k: getattr(cls, k) for k in dir(cls) if k.startswith('problem')}

    @classmethod
    def problem_trace(cls):
        # min(-3a-b-1) s.t. [[a, 1-a], [1-a, b]] >> 0 and a + 2b <= 5
        x0_and_space = (Matrix([2]), {0: Matrix([[2,1,1,0]])})
        sdp = SDPPrimal(x0_and_space)
        a, na, na2, b = sdp.gens
        return sdp, -2*a + na*.2+na2*.8 - b - 2, [2*a + 2*b <= 6-.7*na-.3*na2], -29/4-5*57**.5/12

@pytest.mark.parametrize('problem', PrimalObjProblems.collect().values(),
    ids=PrimalObjProblems.collect().keys())
def test_primal_solve_obj(problem, tol=1e-5):
    sdp, obj, constraints, val = problem()
    assert sdp.solve_obj(obj, constraints)
    val2 = obj.xreplace(sdp.as_params())
    assert abs(val2 - val) < tol


def test_primal_exprs_to_arrays():
    _gens = [Symbol('x%d'%i) for i in range(13)]
    for gens in (None, _gens):
        sdp = SDPPrimal(
            (Matrix([1,2]), {
                'A': Matrix([[1,2,2,4],[2,0,0,1]]) /4,
                'B': Matrix.ones(2, 9)
            }),
            gens = gens
        )
        expr = sum((i+1)*sdp.gens[i] for i in range(sdp.dof))
        array = sdp.exprs_to_arrays([expr])[0][0]

        assert np.abs(array.flatten() - np.array([1,2.5,2.5,4, 5,7,9,7,9,11,9,11,13])).max() < 1e-9