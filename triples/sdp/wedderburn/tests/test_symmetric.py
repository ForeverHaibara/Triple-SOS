from collections import defaultdict

from sympy import factorial
from sympy.combinatorics import SymmetricGroup, Permutation
from ..symmetric import murnaghan_nakayama_character_table, young_symmetrizers
from ..dixon import dixon_character_table

def test_murnaghan_nakayama_character_table():
    a000041 = [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77, 101]
    for n in range(1, 9):
        m = a000041[n]
        X = murnaghan_nakayama_character_table(n)
        assert X.shape == (m, m)
        assert all(i == 1 for i in X[0, :].tolist()[0])
        assert all(X[i+1, 0] >= X[i, 0] for i in range(m - 1))
        assert (X.T * X).is_diagonal()
        assert sum(X[:, 0].applyfunc(lambda x: x**2)) == factorial(n)

    for n in range(1, 6):
        G = SymmetricGroup(n)
        cc = G.conjugacy_classes()
        X = murnaghan_nakayama_character_table(n, cc=cc).tolist()
        Y = dixon_character_table(cc).tolist()
        assert sorted(X) == sorted(Y)


def test_young_symmetrizers():
    def _mul(a, b):
        d = defaultdict(int)
        for p, v1 in a.items():
            for q, v2 in b.items():
                d[p * q] += v1 * v2
        return {p: v for p, v in d.items() if v}

    for n in range(1, 6):
        dims = {}
        young = list(young_symmetrizers(n, action=Permutation).items())
        for i in range(len(young)):
            # idempotent
            y = young[i][1]
            y2 = _mul(y, y)
            c = y2[Permutation(size=n)]
            dims[young[i][0]] = c
            assert c != 0 and len(y2) == len(y) and all(y2[k] == c * y[k] for k in y)

            # orthogonality between different shapes
            for j in range(i+1, len(young)):
                assert not _mul(y, young[j][1])

        dims = {k:factorial(n)//v for k, v in dims.items()}
        expected = murnaghan_nakayama_character_table(n)[:, 0]
        assert sorted(dims.values()) == sorted(expected)
