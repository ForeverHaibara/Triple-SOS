from sympy import factorial
from sympy.combinatorics import SymmetricGroup
from ..symmetric import murnaghan_nakayama_character_table
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
