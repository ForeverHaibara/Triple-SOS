from sympy import Matrix, sqrt, I
from sympy.combinatorics import (
    Permutation, PermutationGroup, SymmetricGroup, AlternatingGroup
)

from ..dixon import dixon_character_table

def _cmp_tbl(tbl1, tbl2):
    # compare character tables numerically up to permutations of rows
    tbl1 = Matrix(tbl1).n(15).tolist()
    tbl2 = Matrix(tbl2).n(15).tolist()
    return {tuple(row) for row in tbl1} == {tuple(row) for row in tbl2}

def test_dixon_character_table():
    # trivial group
    G = PermutationGroup(Permutation(size=2))
    tbl = dixon_character_table(G)
    assert not tbl._rep.domain.is_AlgebraicField
    assert _cmp_tbl(tbl, [[1]])

    # S4
    G = SymmetricGroup(4)
    cc = [
        Permutation([[3]]),
        Permutation([[0,1],[3]]),
        Permutation([[0,1],[2,3]]),
        Permutation([[0,1,2],[3]]),
        Permutation([[0,1,2,3]])
    ]
    cc = [G.conjugacy_class(c) for c in cc]
    tbl = dixon_character_table(cc)
    expected = [
        [1, 1, 1, 1, 1],
        [1, -1, 1, 1, -1],
        [2, 0, 2, -1, 0],
        [3, 1, -1, 0, -1],
        [3, -1, -1, 0, 1],
    ]
    assert not tbl._rep.domain.is_AlgebraicField
    assert _cmp_tbl(tbl, expected)

    # A4
    G = AlternatingGroup(4)
    cc = [
        Permutation([[3]]),
        Permutation([[1,2,3]]),
        Permutation([[1,3,2]]),
        Permutation([[0,1],[2,3]])
    ]
    cc = [G.conjugacy_class(c) for c in cc]
    tbl = dixon_character_table(cc)
    dom = tbl._rep.domain
    assert dom.is_AlgebraicField and dom.mod.to_list() == [1, 1, 1]
    w = (-1 + sqrt(3)*I)/2
    expected = [
        [1, 1, 1, 1],
        [1, w, w**2, 1],
        [1, w**2, w, 1],
        [3, 0, 0, -1],
    ]

    # C₅ ⋊ C₄
    G = PermutationGroup(
        Permutation([[0,1,3,4,2]]),
        Permutation([[1,4,2,3]])
    )
    cc = [
        Permutation([[4]]),
        Permutation([[1,2],[3,4]]),
        Permutation([[1,3,2,4]]),
        Permutation([[1,4,2,3]]),
        Permutation([[0,1,3,4,2]])
    ]
    cc = [G.conjugacy_class(c) for c in cc]
    tbl = dixon_character_table(cc)
    dom = tbl._rep.domain
    assert dom.is_AlgebraicField and dom.mod.to_list() == [1, 0, 1]
    expected =  [
        [1, 1, 1, 1, 1],
        [1, 1, -1, -1, 1],
        [1, -1, -I, I, 1],
        [1, -1, I, -I, 1],
        [4, 0, 0, 0, -1],
    ]
    assert _cmp_tbl(tbl, expected)
