from sympy import Function, ZZ, expand, symbols
from sympy.combinatorics import Permutation

from ..moment_algebra import MixedMomentAlgebra


def test_moment_algebra_monom_standardization():
    x, y, z = symbols('x y z')
    state = Function('s')
    cycle = Permutation([1, 2, 0])
    algebra = MixedMomentAlgebra(3, 3, 2, is_homogeneous=False)
    p = algebra.as_poly(x*state(x)*state(y)**2, (x, y, z), state_operator=state)
    monom = p.monoms()[0]
    permuted = algebra.permute_monom(monom, cycle)
    result = p.per(p.rep.per({permuted: p.domain.one}))
    assert result.as_expr(state_operator=state) == z*state(z)*state(x)**2
    assert list(permuted[1:]) == sorted(permuted[1:])
    assert algebra.permute_monom(algebra.s((monom, ZZ.one))[0], cycle) == algebra.s((permuted, ZZ.one))[0]

    # state(x) == state(y), and the terms should be merged
    algebra = MixedMomentAlgebra(3, 3, 2, is_homogeneous=False,
        standard_monom=lambda m: tuple(sorted(m)))
    p = algebra.as_poly(x*state(x)*state(y), (x, y, z), state_operator=state)
    permuted = algebra.permute_monom(p.monoms()[0], cycle)
    assert permuted == ((0, 0, 1), ((0, 0, 1), 2))


    algebra = MixedMomentAlgebra(2, 3, 3, is_homogeneous=False)

    def convert(expr):
        return algebra.as_poly(expr, (x, y), state_operator=state)

    expr = x*state(x*y) + 2*state(x)**2*state(y) + state(1)
    p = convert(expr)
    assert p.as_expr(state_operator=state) == expr
    assert p.total_degree() == 3 and not p.is_homogeneous
    assert convert(state(x + y)).as_expr(state_operator=state) == state(x) + state(y)
    assert convert(state(x*state(y))).as_expr(state_operator=state) == state(x)*state(y)
    assert convert(state(state(x))).as_expr(state_operator=state) == state(1)*state(x)
    assert convert(state(0)).is_zero
    assert convert(x*state(x)).state().as_expr(state_operator=state) == state(x)**2
    product = convert(state(x)*state(y)) * convert(state(x) + state(x*y))
    assert expand(product.as_expr(state_operator=state)) == expand(state(x)*state(y)*(state(x) + state(x*y)))

    # state(x) == state(y), and the terms should be merged
    symmetric = MixedMomentAlgebra(2, 2, 2, is_homogeneous=False,
        standard_monom=lambda monom: tuple(sorted(monom)))
    q = symmetric.as_poly(x + 2*y, (x, y)).state()
    assert q.rep == {((0, 0), ((0, 1), 1)): ZZ(3)}
    assert symmetric.as_poly(x - y, (x, y)).state().is_zero
