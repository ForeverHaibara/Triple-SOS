import numpy as np
from sympy import MutableDenseMatrix as Matrix
from sympy import Rational, primerange, re, exp, pi

from ..eigens import congruence

def test_congruence():
    hilbert = Matrix([[Rational(1,i+j+1) for j in range(4)] for i in range(4)])
    U, S = congruence(hilbert)
    assert U.T @ Matrix.diag(*S) @ U == hilbert

    hilbert_perturb = hilbert - Matrix.eye(4) / 1000
    assert congruence(hilbert_perturb) is None

    U = Matrix(4,4,list(primerange(56)))
    A = U.T @ Matrix.diag(*[2,0,1,3]) @ U
    U1, S1 = congruence(A)
    assert U1.T @ Matrix.diag(*S1) @ U1 == A

    U2, S2 = congruence((A/11).n(15), perturb=True, upper=False)
    assert U2._rep.domain.is_RR and S2._rep.domain.is_RR
    assert max((U2.T @ Matrix.diag(*S2) @ U2 - A/11).applyfunc(abs)) < 1e-10

    U2, S2 = congruence((A/7).n(15), perturb=True, upper=True)
    assert U2._rep.domain.is_RR and S2._rep.domain.is_RR
    assert max((U2.T @ Matrix.diag(*S2) @ U2 - A/7).applyfunc(abs)) < 1e-10
    assert congruence((A/7).n(8), perturb=1e-15, upper=False) is None
    assert congruence((A/7).n(8), perturb=1e-15, upper=True) is None


    assert congruence(U.T @ Matrix.diag(*[0,0,1,2])/7 @ U) is not None
    assert congruence(U.T @ Matrix.diag(*[3,0,2,0]) @ U) is not None

    assert congruence(Matrix([[1,1,0,0],[1,1,1,1],[0,1,0,0],[0,1,0,2]])) is None

    # test matrix to be converted to real
    z = exp(2j*pi/3.0)
    A = Matrix([[1,1,1],[1,z,z**2],[1,z**2,z**3]])
    A = A.T.conjugate() @ A
    U, S = congruence(A)
    assert U._rep.domain.is_RR and S._rep.domain.is_RR
    assert max((U.T @ Matrix.diag(*S) @ U - A.applyfunc(re)).applyfunc(abs)) < 1e-10