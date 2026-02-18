from sympy import Matrix, sqrt, I
from sympy.polys.polyclasses import ANP
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
    assert isinstance(tbl, Matrix)
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
    assert isinstance(tbl, Matrix)
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
    assert isinstance(tbl, Matrix)
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


def _conj_cyclotomic_mat(m: Matrix) -> Matrix:
    """Efficiently compute the conjugate of a matrix
    over a cyclotomic field."""
    dom = m._rep.domain
    if not dom.is_AlgebraicField:
        return m

    # tricky way to get the order of the cyclotomic field
    n = int(dom.ext.alias.name.lstrip('zeta'))
    def conj(a: ANP) -> ANP:
        rep = a.rep
        if not rep: # a == 0
            return a
        rep = rep[:-1][::-1] + [0] * (n - len(rep)) + [rep[-1]]
        return ANP(rep, a.mod, a.dom)
    rows = m._rep.to_list()
    rows = [[conj(v) for v in row] for row in rows]
    dm = m._rep.__class__(rows, (len(rows), len(rows)), dom)
    return m._fromrep(dm)


def _check_generic_character_table(cc, tbl: Matrix, name: str='') -> bool:
    order = sum(len(c) for c in cc)
    assert tbl.shape[0] == len(cc) and tbl.shape[1] == len(cc), name
    assert all(v == 1 for v in tbl[0, :]), name
    assert tbl[:, 0].dot(tbl[:, 0]) == order, name

    tblh = _conj_cyclotomic_mat(tbl).T
    x = tblh * tbl
    assert x.is_diagonal(), name
    assert list(x.diagonal()) == [order//len(c) for c in cc], name


def test_generic_dixon_character_table():
    try:
        from sympy.combinatorics.galois import (
            S1TransitiveSubgroups,
            S2TransitiveSubgroups,
            S3TransitiveSubgroups,
            S4TransitiveSubgroups,
            S5TransitiveSubgroups,
            S6TransitiveSubgroups
        )
    except ImportError:
        return
    classes = [
        S1TransitiveSubgroups,
        S2TransitiveSubgroups,
        S3TransitiveSubgroups,
        S4TransitiveSubgroups,
        S5TransitiveSubgroups,
        S6TransitiveSubgroups,
    ]
    for cls in classes:
        for key, value in cls.__members__.items():
            G = value.get_perm_group()
            tbl = dixon_character_table(G)
            cc = G.conjugacy_classes()
            _check_generic_character_table(cc, tbl, name=key)
