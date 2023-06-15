"""
Deprecated
"""

import numpy as np
import sympy as sp

from .utils import _upper_vec_of_symmetric_matrix, rationalize_matrix, congruence, solve_undetermined_linear
from .eigen_roots import _as_vec, _get_standard_roots
from ...utils.text_process import deg
from ...utils.basis_generator import generate_expr, arraylize
from ...utils.root_guess import cancel_denominator

def eigen_as_sos(Ms, n, cancel = True):
    a, b, c = sp.symbols('a b c')
    y, names = [], []
    for minor in range(len(Ms)):
        M = Ms[minor]
        if not (n % 2) ^ (minor == 1):
            M = M / 3
        U, S = congruence(M)

        monoms = generate_expr(n//2 - minor, cyc = 0)[1]
        multiplier = {
            (0, 0): 1,
            (0, 1): a * b,
            (1, 0): a,
            (1, 1): a * b * c,
        }[(n % 2, minor)]
        for i, s in enumerate(S):
            if s == 0:
                continue
            val = sp.S(0)
            if cancel:
                r = cancel_denominator(U[i,i:])
            for j in range(i, len(monoms)):
                monom = monoms[j]
                val += U[i,j] / r * a**monom[0] * b**monom[1] * c**monom[2]
            names.append(multiplier * val ** 2)
            y.append(s * r**2)
    return {'y': y, 'names': names}


def _discard_zero_rows(A, b, rank_tolerance = 1e-10):
    A_rowmax = np.abs(A).max(axis = 1)
    nonvanish =  A_rowmax > rank_tolerance * A_rowmax.max()
    rows = np.extract(nonvanish, np.arange(len(nonvanish)))
    return A[rows], b[rows]


def _add_eq(sos, x0, space, y, suffix = '_0'):
    import picos

    # k(k+1)/2 = len(x0)
    k = round(np.sqrt(2 * len(x0) + .25) - .5)
    S = picos.SymmetricVariable('S%s'%suffix, (k,k))
    sos.add_constraint(S >> 0)

    target = space * y + x0.reshape((-1,1))
    for entry, targeti in zip(_upper_vec_of_symmetric_matrix(S), target):
        sos.add_constraint(entry == targeti)
    return S


def _construct_QQvecS(n, roots, minor = False):
    roots_filter = {
        (0, 0): lambda x: True,
        (0, 1): lambda x: x[0] != 0 and x[1] != 0, # a * b
        (1, 0): lambda x: x[0] != 0, # a
        (1, 1): lambda x: x[0] != 0 and x[1] != 0 and x[2] != 0, # a * b * c
    }[(n % 2, minor)]
    roots = [root for root in roots if roots_filter(root)]

    X = np.vstack([_as_vec(n//2 - minor, root) for root in roots]).T
    Xqr = np.linalg.qr(X, 'complete')
    Xqr_R = np.abs(Xqr[1]).max(axis = 1)
    rank = (Xqr_R > (Xqr_R.max() * 1e-10)).sum()
    Q = Xqr[0][:, rank:]

    inv_monoms_n, dict_monoms_n = generate_expr(n, cyc = 0)
    dict_monoms_half = generate_expr(n//2 - minor, cyc = 0)[1]

    # the original matrix M = Q @ S @ Q^T where S is positive symmetric
    # we have got more constraints than the number of variables
    # so we can solve the problem by a linear system

    # vec(M) = kron(Q,Q) @ vec(S)
    QQ = np.kron(Q, Q)
    QQvecS = np.zeros((len(inv_monoms_n), Q.shape[1]**2))


    bias = {
        (0, 0): (0, 0, 0), # g(a,b,c)^2
        (0, 1): (1, 1, 0), # ab * g(a,b,c)^2
        (1, 0): (1, 0, 0), # a * g(a,b,c)^2
        (1, 1): (1, 1, 1), # abc * g(a,b,c)^2
    }[(n % 2, minor)]

    def monom_add(i, j):
        a1, a2, a3 = dict_monoms_half[i]
        b1, b2, b3 = dict_monoms_half[j]
        c1, c2, c3 = bias
        return (a1+b1+c1, a2+b2+c2, a3+b3+c3)

    if (n % 2) ^ (minor == 1):
        permute = lambda x: [x, (x[1], x[2], x[0]), (x[2], x[0], x[1])]
    else:
        permute = lambda x: [x]

    m, k = len(dict_monoms_half), Q.shape[1]
    for i in range(m):
        for j in range(m):
            monom = monom_add(i, j)
            for monom2 in permute(monom):
                QQvecS[inv_monoms_n[monom2]] += QQ[i*m+j]

    # cancel symmetric entries
    QQvecS2 = []
    for i, j in _upper_vec_of_symmetric_matrix(k, return_inds = True):
        if i != j:
            QQvecS2.append(QQvecS[:,i*k+j] + QQvecS[:,j*k+i])
        else:
            QQvecS2.append(QQvecS[:,i*k+i])
    QQvecS = np.array(QQvecS2).T
    return Q, QQvecS



def eigen_sos(poly, roots = None, minor = False):
    import picos

    n = deg(poly)

    if roots is None:
        roots = poly
    roots = _get_standard_roots(roots, positive = (n % 2 == 0) and (minor == 0))

    Qs, QQvecSs = [], []
    for i in range(int(minor) + 1):
        Q, QQvecS = _construct_QQvecS(n, roots, minor = i)
        Qs.append(Q)
        QQvecSs.append(QQvecS)
    QQvecS = np.hstack(QQvecSs)

    vecM = arraylize(poly, cyc = 0)
    QQvecS, vecM = _discard_zero_rows(QQvecS, vecM, rank_tolerance = 1e-10)

    # we have QQvecS @ vecS = vecM
    linsol = solve_undetermined_linear(QQvecS, vecM, rank_tolerance = 1e-10)
    rank, x0, QQD, QQV = linsol['rank'], linsol['x0'], linsol['D'], linsol['V']

    # x0 is a base solution, and we can add any vector in the null space of QQvecS
    # null vector should be in form of QQV[:,rank:] @ x1
    # we should guarantee that (x0 + QQV[:,rank:] @ x1) makes S positive definite
    split = QQvecSs[0].shape[1]
    sos = picos.Problem()
    Ss_ = []
    if QQV.shape[0] != rank:
        y = picos.RealVariable('y', int(QQV.shape[0] - rank))
        for i in range(minor + 1):
            slicing = slice(None, split) if i == 0 else slice(split, None)
            S = _add_eq(sos, x0[slicing], QQV[rank:, slicing].T, y, suffix = '_%d'%i)
            Ss_.append(S)

        for i in range(minor + 1):
            S = Ss_[i]
            if hasattr(S, 'tr'):
                sos.set_objective('max', S.tr)
                sos.solve()
                break
    else:
        # degenerated case
        print('DEGEN')
        for x_part in (x0[:split], x0[split:]):
            k = round(np.sqrt(2 * len(x_part) + .25) - .5)
            S = np.zeros((k,k))
            for (i,j), v in zip(_upper_vec_of_symmetric_matrix(S, return_inds = True), x0):
                S[i,j] = S[j,i] = v
            Ss_.append(S)

    Ss = [np.array(candidate) for candidate in Ss_]

    Ms = [Q @ S @ Q.T for Q, S in zip(Qs, Ss)]
    mask = max(np.abs(M).max() for M in Ms) * 1e-7
    M_rationals = [rationalize_matrix(M, mask_func = mask) for M in Ms]
    for M in Ms:
        print(np.linalg.eigvalsh(M))

    result = eigen_as_sos(M_rationals, n)
    for key in ['sos', 'Q', 'QQvecS', 'Ms', 'M_rationals']:
        result[key] = locals()[key]
    return result