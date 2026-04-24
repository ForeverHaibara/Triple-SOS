from sympy import FiniteField as FF
from sympy.polys import ring

from ..polytools import dmp_gf_factor

def _R_dmp_factor_list(f):
    c, factors = dmp_gf_factor(f.to_dense(), f.ring.ngens - 1, f.ring.domain)
    return (c, [(f.ring.from_dense(g), k) for g, k in factors])

def test_dmp_gf_factor():
    R, x, y = ring("x,y", FF(2))
    assert _R_dmp_factor_list(x**2 + y**2) == (1, [(x + y, 2)])

    R, x, y, z = ring("x,y,z", FF(3))
    assert _R_dmp_factor_list(5*(x + y + z)**3) == (2, [(x + y + z, 3)])

    R, x, y = ring("x,y", FF(5))

    assert _R_dmp_factor_list(x**2 - 1) == (1, [
        (x + 1, 1),
        (x - 1, 1),
    ])

    R, x, y = ring("x,y", FF(7))

    f = (x + y + 1)*(x + y + 2)
    assert _R_dmp_factor_list(f) == (1, [
        (x + y + 1, 1),
        (x + y + 2, 1),
    ])

    R, x, y = ring("x,y", FF(5))

    f = (x + y + 1)**2*(x + 2*y + 1)*(x**2 + y**2 + 1)
    coeff, factors = _R_dmp_factor_list(f)

    assert coeff == 1
    assert factors == [
        (x + y + 1, 2),
        (x + 2*y + 1, 1),
        (x**2 + y**2 + 1, 1),
    ] or [
        (x + 2*y + 1, 1),
        (x + y + 1, 2),
        (x**2 + y**2 + 1, 1),
    ]

    R, a, b = ring("a,b", FF(7))
    c = 1

    f = -2*(-a + c)**4*(a - b)**4*(b - c)**4*(a + b + c)**4 + \
        9*(a**4*(a - b)**2*(a - c)**2 + b**4*(-a + b)**2*(b - c)**2 +
           c**4*(-a + c)**2*(-b + c)**2)**2

    coeff, factors = _R_dmp_factor_list(f)

    assert coeff != 0
    assert len(factors) == 4

    g = coeff

    for h, k in factors:
        g *= h**k

    assert g == f
