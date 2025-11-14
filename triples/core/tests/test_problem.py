from sympy.abc import a, b, c, d, e, u, v, w, x, y, z

from sympy import Poly, ZZ, ring
from sympy.polys.rings import PolyElement
from sympy.polys.fields import FracElement

from ..problem import InequalityProblem


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
            FracElement(fld.domain, rng.zero, rng(a**2 + b)),
            a/(b + 3*x)**2 - 1/(y - 2*z)
        ],
    )
    for k in ineq.ineq_constraints.keys():
        assert ineq._dtype_is_homogeneous(k), f"failed to check homogeneity of {k}"
    assert ineq.is_homogeneous