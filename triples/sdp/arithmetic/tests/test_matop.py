import numpy as np
from scipy import sparse
from sympy import MutableDenseMatrix as Matrix
from sympy import Rational

from ..matop import (
    FLINT_TYPE, is_zz_qq_mat, permute_matrix_rows,
    rep_matrix_to_numpy, rep_matrix_to_scipy
)

def test_permute_matrix_rows():
    mats = [
        Matrix([[1,2,3],[4,5,6],[7,8,9]]),
        Matrix([[1,2,3],[4,5,6],[7,8,9]]) / 5,
        3.5 * Matrix([[1,2,3],[4,5,6],[7,8,9]])
    ]
    p = [2,0,0,1]

    for M in mats:
        funcs = ['to_sdm', 'to_ddm', 'to_dfm']
        if not FLINT_TYPE:
            funcs = funcs[:2]
        for func in funcs:
            rep = M._rep.rep
            if func == 'to_dfm' and (not is_zz_qq_mat(M) or not hasattr(rep, func)):
                continue
            rep2 = getattr(rep, func)()
            M2 = M._fromrep(M._rep.from_rep(rep2))
            M_p = permute_matrix_rows(M2, p)
            assert (M_p.n(15) - Matrix.vstack(*[M[i, :] for i in p])).is_zero_matrix, f"func={func}"


def test_rep_matrix_to_scipy_sparse_rep_matrix():
    M = Matrix.zeros(7, 9)
    M[1, 3] = Rational(2, 5)
    M[4, 0] = -3
    M[6, 8] = Rational(7, 11)

    S = rep_matrix_to_scipy(M)

    assert sparse.isspmatrix_csr(S)
    assert S.shape == M.shape
    assert S.nnz == 3
    assert np.allclose(S.toarray(), rep_matrix_to_numpy(M))


def test_rep_matrix_to_scipy_passthrough_and_dtype():
    S0 = sparse.coo_matrix(([1, -2, 3], ([0, 2, 2], [1, 0, 3])), shape=(4, 5))

    S = rep_matrix_to_scipy(S0, dtype=np.float32)
    A = rep_matrix_to_numpy(S, dtype=np.float64)

    assert sparse.issparse(S)
    assert S.dtype == np.float32
    assert A.dtype == np.float64
    assert np.allclose(A, S0.toarray())


def test_rep_matrix_to_numpy_keeps_vector_dense():
    v = np.array([1, 0, -2, 5], dtype=np.int64)

    out = rep_matrix_to_numpy(v, dtype=np.int64)

    assert isinstance(out, np.ndarray)
    assert out.shape == (4,)
    assert np.array_equal(out, v)
