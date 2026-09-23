import sys

import numpy as np
from scipy import sparse
from sympy import Rational, primerange
from sympy import MutableDenseMatrix as Matrix
from ..matmul import (
    kronecker_product, matadd, matlshift, matmul, matmul_multiple,
    symmetric_bilinear, symmetric_bilinear_multiple,
    _decompose_int64_to_level_digits
)
from ..matop import FLINT_TYPE, USE_SCIPY_ARRAY


if USE_SCIPY_ARRAY:
    csr_array = sparse.csr_array
else:
    csr_array = sparse.csr_matrix


def npeq(A, B):
    return A.shape == B.shape and not np.any(A - B)

def sparse_dense(A):
    return A.toarray() if sparse.issparse(A) else np.asarray(A)

def test_matlshift():
    A = Matrix([1, 2**63 - 1, -2**63, 2**64, -2**64, -1234, 12345678901234])
    func = ['to_sdm', 'to_ddm', 'to_dfm']
    if not FLINT_TYPE:
        func = func[:2]
    for f in func:
        dm = A._rep
        if f == 'to_dfm' and (not hasattr(dm.rep, f)):
            continue
        B = A._fromrep(dm.from_rep(getattr(dm.rep, f)()))
        assert matlshift(B, 2) == B * 4
        assert matlshift(B, 10) == B * 1024


def test_matmul():
    # int64 overflow
    A = Matrix([[922337203685, 36854775807], [3372036854, 77580792233]])
    B = Matrix([[847288609447, 26843545767], [486905010818, 60944792233]])
    C, C2 = matmul(A, B), matmul(A/7, B/9)
    assert C._rep.domain.is_ZZ and C == A @ B
    assert C2._rep.domain.is_QQ and C2 == (A @ B)/63
    assert matmul(A, B, return_shape=(1,4)) == C.reshape(1,4)

    A = Matrix([[2**63, 1], [3**15, 2**30]])
    assert matmul(A, B) == A @ B
    assert matmul(B, A/144) == B @ A / 144


    # int32 overflow
    A = Matrix([[-1654615998, 1806341205], [ 173879092, 1112038970]])
    B = Matrix([[ 577090037, 271041745], [-1095513148, 506456969]])
    C, C2 = matmul(A, B), matmul(A/7, B/9)
    assert C._rep.domain.is_ZZ and C == A @ B
    assert C2._rep.domain.is_QQ and C2 == (A @ B)/63
    assert matmul(A, B, return_shape=(4,1)) == C.reshape(4,1)

    # test empty products
    assert matmul(Matrix.zeros(0, 0), Matrix.zeros(0, 0)) == Matrix.zeros(0, 0)
    assert matmul(Matrix.zeros(0, 0), Matrix.zeros(0, 5)) == Matrix.zeros(0, 5)
    assert matmul(Matrix.zeros(7, 0), Matrix.zeros(0, 3)) == Matrix.zeros(7, 3)
    assert matmul(Matrix(7, 2, list(range(2,16))), Matrix.zeros(2, 0)) == Matrix.zeros(7, 0)
    assert matmul(Matrix(7, 3, list(range(5,26))), Matrix.zeros(3, 0), return_shape=(0,7)) == Matrix.zeros(0, 7)

    # test numpy matmuls
    A, B = np.array([[1, -2], [3, 4]]), np.array([[15, -6], [37, 48]])
    assert npeq(matmul(A, B), A @ B)
    assert npeq(matmul(A, B, return_shape=(1,4)), (A @ B).reshape(1,4))
    assert npeq(matmul(A, B[:,0]), A @ B[:,0])
    assert npeq(matmul(A, B[:,:1]), A @ B[:,:1])
    assert npeq(matmul(A, B[:,1], return_shape=(1,2)), (A @ B[:,1]).reshape(1,2))
    assert npeq(matmul(A/3, np.zeros((2,0)), return_shape=(0,2)), np.zeros((0,2)))

    # test sympy mul numpy
    A, B = np.array([[5, -2], [3, -7]]), Matrix([[15, -6], [37, 48]])
    assert matmul(A, B) == Matrix(A.tolist()) @ B
    C, C2 = matmul(A/7, 2*B/3), matmul(2*B/3, A/7, return_shape=(4,1))
    assert C._rep.domain.is_RR and max((C - 2*A @ B/21).applyfunc(abs)) < 1e-10
    assert C2._rep.domain.is_RR and max((C2 - 2*(B @ A).reshape(4,1)/21).applyfunc(abs)) < 1e-10


def test_sparse_matadd_and_matmul():
    A = csr_array([[0., 2., 0.], [3., 0., 4.]])
    B = csr_array([[5., 0.], [0., 7.], [11., 0.]])
    D = np.array([[1., 2.], [3., 4.], [5., 6.]])

    C1 = matmul(A, B)
    C2 = matmul(D.T, B)
    S = matadd(A, csr_array([[1., 0., -2.], [0., 5., 0.]]))

    assert sparse.issparse(C1)
    assert np.allclose(C1.toarray(), A.toarray() @ B.toarray())
    assert np.allclose(sparse_dense(C2), D.T @ B.toarray())
    assert sparse.issparse(S)
    assert np.allclose(S.toarray(), np.array([[1., 2., -2.], [3., 5., 4.]]))

    A = csr_array([[0, 2, 0], [3, 0, 4]])
    B = Matrix([[5], [7], [11]])
    C = matmul(A, B)
    assert C == Matrix([[14], [59]])
    assert C._rep.domain.is_ZZ


def test_matmul_multiple():
    # test empty products
    assert matmul_multiple(Matrix.zeros(5, 0), Matrix.zeros(0, 3)) == Matrix.zeros(5, 0)
    assert matmul_multiple(Matrix.zeros(4, 0), np.zeros((0, 3))) == Matrix.zeros(4, 0)
    assert matmul_multiple(Matrix.zeros(0, 4), Matrix.zeros(2, 3)) == Matrix.zeros(0, 6)
    assert npeq(matmul_multiple(np.zeros((2, 0)), np.zeros((0, 1))), np.zeros((2, 0)))
    assert npeq(matmul_multiple(np.ones((2, 9)), np.zeros((3, 0))), np.zeros((2, 0)))
    assert npeq(matmul_multiple(np.zeros((0, 9)), np.zeros((3, 4))), np.zeros((0, 12)))

    # test matmul_multiple
    A = [[3,1,4,1,5,9,2,6,5], [2,6,5,3,5,8,9,7,9], [2,7,1,8,2,8,1,8,2], [8,4,5,9,0,4,5,2,3]]
    B = Matrix([[1,2], [5,7], [13,17]])
    C = Matrix.vstack(*[(Matrix(_).reshape(3,3) @ B).reshape(1,6) for _ in A])
    assert matmul_multiple(Matrix(A), B) == C
    assert matmul_multiple(np.array(A), B) == C
    assert matmul_multiple(Matrix(A)/3, -B/7) == -C/21

    # test int64 overflow
    B = (2**50*B+Matrix.ones(*B.shape))/3
    C = Matrix.vstack(*[((2**50*Matrix(_).reshape(3,3)+Matrix.ones(3,3))/7 @ B).reshape(1,6) for _ in A])
    assert matmul_multiple((2**50*Matrix(A)+Matrix.ones(4,9))/7, B) == C

    # test sparsity
    A = csr_array([
        [1., 0., 2., 0.],
        [0., -3., 0., 4.],
        [5., 0., 0., 6.],
    ])
    B = csr_array([[7., 0., 11.], [0., 13., 17.]])
    expected = (A.toarray().reshape(3, 2, 2) @ B.toarray()).reshape(3, 6)

    for n in range(4):
        A1 = A if n//2 else A.toarray()
        B1 = B if n%2 else B.toarray()
        C = matmul_multiple(A1, B1)
        assert not isinstance(C, Matrix)
        if sparse.issparse(A1) and sparse.issparse(B1):
            assert sparse.issparse(C)
        assert np.allclose(sparse_dense(C), expected)


def test_matmul_multiple_sympy_sparse_dispatch(monkeypatch):
    module = sys.modules[matmul_multiple.__module__]
    # A = Matrix([
    #     [0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # ])
    # B = Matrix([[0, 11], [13, 0], [0, 0], [-17, 0]])
    # expected = Matrix.vstack(*[(Matrix(_).reshape(4, 4) @ B).reshape(1, 8) for _ in A.tolist()])

    # def unexpected_dense_conversion(M, dtype=np.float64):
    #     raise AssertionError('sparse dispatch should not densify SymPy inputs')

    # func = getattr(module, 'rep_matrix_to_numpy')
    # monkeypatch.setattr(module, 'rep_matrix_to_numpy', unexpected_dense_conversion)
    # assert matmul_multiple(A, B) == expected
    # monkeypatch.setattr(module, 'rep_matrix_to_numpy', func)

    A = Matrix([[1, 2, 3, 4], [5, 6, 7, 8]])
    B = Matrix([[9, 10], [11, 12]])
    expected = Matrix.vstack(*[(Matrix(_).reshape(2, 2) @ B).reshape(1, 4) for _ in A.tolist()])

    def unexpected_sparse_kernel(A, B):
        raise AssertionError('dense dispatch should not call sparse kernel')

    monkeypatch.setattr(module, '_matmul_multiple_sparray', unexpected_sparse_kernel)
    assert matmul_multiple(A, B) == expected


def test_symmetric_bilinear():
    # test empty products
    assert symmetric_bilinear(Matrix.zeros(3, 0), Matrix.ones(3, 3)) == Matrix.zeros(0, 0)
    assert symmetric_bilinear(Matrix.zeros(0, 5), Matrix.zeros(0, 0)) == Matrix.zeros(5, 5)
    assert symmetric_bilinear(Matrix.zeros(0, 5), Matrix.zeros(0, 0), return_shape=(25,1)) == Matrix.zeros(25, 1)
    assert npeq(symmetric_bilinear(np.zeros((3,0)), np.ones((3,3))), np.zeros((0,0)))
    assert npeq(symmetric_bilinear(np.zeros((0,5)), np.zeros((0,0))), np.zeros((5,5)))
    assert npeq(symmetric_bilinear(np.zeros((0,5)), np.zeros((0,0)), return_shape=(25,1)), np.zeros((25,1)))

    # test int64 overflow
    U = 2**24*Matrix(3,2,[-2,-5,-7,-6,-8,-9]) + 7*Matrix.ones(3,2)
    A = 2**24*Matrix(3,3,list(range(9))) + Matrix.ones(3,3)
    C = U.T @ A @ U
    assert symmetric_bilinear(U, A) == C
    assert symmetric_bilinear(U, A.reshape(9, 1), is_A_vec=True) == C
    assert symmetric_bilinear(U, A, return_shape=(4,1)) == C.reshape(4, 1)

    U = Matrix(3,2,[-2,-5,-7,-6,-8,-9])
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) / 7.
    assert symmetric_bilinear(U, A)._rep.domain.is_RR
    assert symmetric_bilinear(U, A, return_shape=(4,1))._rep.domain.is_RR

    # test sparsity
    U = csr_array([[1., 0.], [2., -1.], [0., 3.]])
    A = csr_array([[5., 0., 7.], [0., 11., 0.], [13., 0., 17.]])
    expected = U.toarray().T @ A.toarray() @ U.toarray()

    for n in range(4):
        U1 = U if n//2 else U.toarray()
        A1 = A if n%2 else A.toarray()
        C = symmetric_bilinear(U1, A1)
        assert not isinstance(C, Matrix)
        if sparse.issparse(U1) and sparse.issparse(A1):
            assert sparse.issparse(C)
        assert np.allclose(sparse_dense(C), expected)


def test_symmetric_bilinear_multiple():
    # test empty products
    assert symmetric_bilinear_multiple(Matrix.zeros(3, 0), Matrix.ones(2, 9)/5) == Matrix.zeros(2, 0)
    assert symmetric_bilinear_multiple(Matrix.ones(3, 2), Matrix.zeros(0, 9)) == Matrix.zeros(0, 4)
    assert npeq(symmetric_bilinear_multiple(np.zeros((3,0)), np.ones((2,9))/5), np.zeros((2,0)))
    assert npeq(symmetric_bilinear_multiple(np.ones((3,2)), np.zeros((0,9))), np.zeros((0,4)))

    A = [[3,1,4,1,5,9,2,6,5], [2,6,5,3,5,8,9,7,9], [2,7,1,8,2,8,1,8,2], [8,4,5,9,0,4,5,2,3]]
    B = Matrix([[1,2], [5,7], [13,17]])
    C = Matrix.vstack(*[(B.T @ Matrix(_).reshape(3,3) @ B).reshape(1,4) for _ in A])
    assert symmetric_bilinear_multiple(B, Matrix(A)) == C

    # test int64 overflow
    B = -2**25*B + 13*Matrix.ones(*B.shape)
    C = Matrix.vstack(*[(B.T @ (2**25*Matrix(_).reshape(3,3) - Matrix.ones(3,3)) @ B).reshape(1,4) for _ in A])
    assert symmetric_bilinear_multiple(B, 2**25*Matrix(A) - Matrix.ones(4,9)) == C

    A = Matrix(A) - Matrix(4,9,[Rational(1, _) for _ in primerange(156)]) # there are 36 primes within 156
    # B = Matrix([[1,2], [5,7], [13,17]])
    C = Matrix.vstack(*[(B.T @ Matrix(_).reshape(3,3) @ B).reshape(1,4) for _ in A.tolist()])
    assert symmetric_bilinear_multiple(B, A) == C

    # test sparsity
    U = csr_array([[1., 0.], [2., -1.], [0., 3.]])
    A = csr_array([
        [5., 0., 7., 0., 11., 0., 13., 0., 17.],
        [0., 19., 0., 23., 0., 29., 0., 31., 0.],
    ])
    expected = (U.toarray().T @ A.toarray().reshape(2, 3, 3) @ U.toarray()).reshape(2, 4)

    for n in range(4):
        U1 = U if n//2 else U.toarray()
        A1 = A if n%2 else A.toarray()
        C = symmetric_bilinear_multiple(U1, A1)
        assert not isinstance(C, Matrix)
        if sparse.issparse(U1) and sparse.issparse(A1):
            assert sparse.issparse(C)
        assert np.allclose(sparse_dense(C), expected)


def test_symmetric_bilinear_multiple_sympy_sparse_dispatch(monkeypatch):
    # module = sys.modules[symmetric_bilinear_multiple.__module__]
    # U = Matrix([[3, 0], [0, 0], [0, 0], [0, -5]])
    # A = Matrix([
    #     [7, 0, 0, 0, 0, 0, -8, 0, 2, 0, 6, 0, 0, 0, 0, 11],
    #     [0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0],
    # ])
    # expected = Matrix.vstack(*[(U.T @ Matrix(_).reshape(4, 4) @ U).reshape(1, 4) for _ in A.tolist()])

    # def unexpected_dense_conversion(M, dtype=np.float64):
    #     raise AssertionError('sparse dispatch should not densify SymPy inputs')

    # monkeypatch.setattr(module, 'rep_matrix_to_numpy', unexpected_dense_conversion)
    # assert symmetric_bilinear_multiple(U, A) == expected
    pass

def test_kronecker_product():
    A = csr_array([[1., 0.], [2., 3.]])
    B = np.array([[0., 5.], [7., 0.]])

    K = kronecker_product(A, B)

    assert sparse.issparse(K)
    assert np.allclose(K.toarray(), np.kron(A.toarray(), B))


def test_decompose_int64_to_level_digits():
    seed = np.random.randint(0, 2**31)
    for level in range(8, 13):
        np.random.seed(seed)
        A = [2**63 - 1, -2**63, 2**63 - 2, -2**63 + 1, -2**63 + 2]
        # generate numbers of different binary length
        for l in range(1, 64):
            for t in range(3):
                s = ''.join([str(np.random.randint(0, 2)) for _ in range(l)])
                A.append(int(s, 2))
                A.append(-A[-1])
        A = np.array(A, dtype=np.int64)
        B = _decompose_int64_to_level_digits(A, level)
        assert B[0].shape == A.shape
        assert all(_.dtype == np.int64 for _ in B), f'wrong seed = {seed}'

        C = [_.astype(object)*2**(level*i) for i,_ in enumerate(B)]
        assert np.all(sum(C) == A.astype(object)), f'wrong seed = {seed}'
