from sympy import MutableDenseMatrix as Matrix

from ..matop import FLINT_TYPE, is_zz_qq_mat, permute_matrix_rows

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
