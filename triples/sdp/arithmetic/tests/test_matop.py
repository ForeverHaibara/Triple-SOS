import numpy as np
from scipy import sparse
from sympy import MutableDenseMatrix as Matrix
from sympy import Poly, RR, QQ
from sympy.abc import z

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


def test_rep_matrix_to_numpy_scipy():
    mats = [
        Matrix([[1,2,3],[4,0,0],[0,8,-9]]),
        Matrix([[1,2,3],[4,0,0],[0,8,-9]]) / 7,
        Matrix._fromrep(Matrix([[1,2.23,3.45],[4,0,0],[0,-8.231,9.6]])\
                        ._rep.convert_to(RR)),
    ]
    for mat in mats:
        expected = np.array(mat.n().tolist(), dtype=np.float64)

        # also cast to composite domains
        dMs = [
            mat._rep,
            mat._rep.convert_to(mat._rep.domain[z]),
            # mat._rep.convert_to(mat._rep.domain[z].get_field())
        ]
        for dM in dMs:
            mat = Matrix._fromrep(dM)
            A = rep_matrix_to_numpy(mat)
            B = rep_matrix_to_scipy(mat)

            assert np.allclose(A, expected)
            assert np.allclose(B.toarray(), expected)
            assert sparse.issparse(B)
            assert B.nnz == 6


    # 1d vector
    v = np.array([1, 0, -2, 5], dtype=np.int64)
    out = rep_matrix_to_numpy(v, dtype=np.int64)
    assert isinstance(out, np.ndarray)
    assert out.shape == (4,)
    assert out.tolist() == v.tolist()

    # real algebraic fields
    r = Poly(z**3 - 4*z + 1, z).all_roots()[1]
    alg = QQ.algebraic_field(r)
    mat = Matrix([[r, 1/r, 0], [0, 3*r**2 - r/3 + 2, (r - 4*r**2)/5]])
    mat = Matrix._fromrep(mat._rep.convert_to(alg))
    A = rep_matrix_to_numpy(mat)
    B = rep_matrix_to_scipy(mat)
    expected = np.array(mat.n().tolist(), dtype=np.float64)
    assert np.allclose(A, expected)
    assert np.allclose(B.toarray(), expected)
    assert sparse.issparse(B)
    assert B.nnz == 4
