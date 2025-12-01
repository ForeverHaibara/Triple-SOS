from sympy.abc import a, b, c, d, e, u, v, w, x, y, z

from sympy import Poly, Function, ZZ, ring
from sympy.combinatorics import CyclicGroup
from sympy.polys.rings import PolyElement
from sympy.polys.fields import FracElement

from ..problem import InequalityProblem
from ..dispatch import _fracelement_init


def test_is_homogeneous():
    rng = ring((a, b, c, u, v), ZZ)[0]
    fld = rng.to_field()
    ineq = InequalityProblem(
        a*x**2 + b*x*y + (c + a)*((y + 2*x)**2 + z**2),
        [
            Poly(b*x + 2*w**2/3 + z*c, a, b, x, c, w, z),
            rng(-c**10 - 8*(u**7 + b*c**6)*(a**2 + u*c)*v),
            fld((a**2 + u*v)/(b**2 + u*c) + 2 - 4*(b - 2*c)/3/(a**2 + b*v)*(a + 5*u)),
            Poly(0, z),
            rng.zero,
            _fracelement_init(fld.domain, rng.zero, rng(a**2 + b)),
            a/(b + 3*x)**2 - 1/(y - 2*z)
        ],
    )
    for k in ineq.ineq_constraints.keys():
        assert ineq._dtype_is_homogeneous(k), f"{k} is homogeneous, but asserted not"
    assert ineq.is_homogeneous


def test_wrap_constraints():
    pro = InequalityProblem(a+b+c, [2*a+b, 2*b+c, 2*c+a])
    pro1 = pro.wrap_constraints()[0]
    assert set(pro1.ineq_constraints) == {2*a + b, 2*b + c, a + 2*c}
    assert len(set([_.find(Function).pop().func for _ in pro1.ineq_constraints.values()])) == 3
    assert [len(_.free_symbols) for _ in pro1.ineq_constraints.values()] == [2, 2, 2]

    pro2 = pro.wrap_constraints(CyclicGroup(3))[0]
    assert set(pro2.ineq_constraints) == {2*a + b, 2*b + c, a + 2*c}
    assert len(set([_.find(Function).pop().func for _ in pro2.ineq_constraints.values()])) == 1
    assert [len(_.free_symbols) for _ in pro2.ineq_constraints.values()] == [2, 2, 2]
