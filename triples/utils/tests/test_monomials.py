from sympy import symbols
from sympy.combinatorics import CyclicGroup, DihedralGroup, SymmetricGroup

from ..monomials import (
    _identify_symmetry_from_action,
    identify_symmetry_from_lists,
    verify_symmetry,
)


def test_identify_symmetry_from_lists():
    # S3
    a, b, c, d, z1, z2, z3, z4, z5 = symbols("a b c d z1 z2 z3 z4 z5")
    gens = (a, b, c, z1, z2, z3, z4, z5)
    to_poly = lambda expression: expression.as_poly(*gens)
    polys = [
        [z1 - z2 - z3 - z4 + 2*z5],
        [
            a*b + a*c + b*c,
            a**2 + a*b + b**2,
            z2,
            a**2 + a*c + c**2,
            z3,
            b**2 + b*c + c**2,
            z4,
            a**2 + b**2 + c**2,
            z5,
            a, b, c
        ],
        [
            a*b + a*c + b*c - z1**2,
            a**2 + a*b + b**2 - z2**2,
            a**2 + a*c + c**2 - z3**2,
            b**2 + b*c + c**2 - z4**2,
            a**2 + b**2 + c**2 - z5**2,
        ],
    ]
    polys = [[to_poly(poly) for poly in group] for group in polys]
    symmetry = identify_symmetry_from_lists(polys)

    assert symmetry.order() == 6
    assert verify_symmetry(polys[1], symmetry)
    assert verify_symmetry(polys[2], symmetry)


    # test the function is independent of the variable name or order
    polys = [[
        (2*a + b).as_poly(a, b, c),
        (2*b + c).as_poly(a, b, c),
        (2*c + a).as_poly(a, b, c),
    ]]
    permuted_polys = [[poly.reorder(b, c, a) for poly in polys[0]]]

    assert identify_symmetry_from_lists(polys).order() == 3
    assert identify_symmetry_from_lists(permuted_polys).order() == 3


    # respect to the ambient group
    a, b, c, d = symbols("a b c d")
    polys = [[(a*b).as_poly(a, b, c, d)]]

    assert identify_symmetry_from_lists(polys, DihedralGroup(4)).order() == 2
    assert identify_symmetry_from_lists(polys, CyclicGroup(4)).order() == 1


def test_identify_symmetry_from_action():
    objects = [[
        (1, 2, 3),
        (2, 3, 1),
        (3, 1, 2),
    ]]

    def action(obj, permutation):
        return tuple(obj[permutation(i)] for i in range(3))

    symmetry = _identify_symmetry_from_action(
        objects, SymmetricGroup(3), action
    )

    assert symmetry.order() == 3
