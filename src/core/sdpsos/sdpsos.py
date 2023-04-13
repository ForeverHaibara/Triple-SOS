from itertools import product

import picos
import sympy as sp
import numpy as np

from ...utils.text_process import PreprocessText, deg
from ...utils.basis_generator import generate_expr
from ...utils.root_guess import rationalize


def SDPConvertor(n, dict_monom: dict):
    # e.g. when n=6
    # first find monomials for degree=3: a^3, a^2b, a^2c, ab^2, abc, ac^2, b^3, b^2c, bc^2, c^3
    # then match pairwise and put them into UFS (union-find-set), e.g. a^2b * ac^2 = a^3bc^2 = a^3 * bc^2
    # for cyclic ones, only

    conversion = dict((i, []) for i in dict_monom.keys())
    def sum_monoms(x, y):
        return (x[0]+y[0], x[1]+y[1], n-x[0]-y[0]-x[1]-y[1])

    for m1, m2 in product(product(range(n//2+1), repeat = 2), repeat = 2):
        if sum(m1) <= n//2 and sum(m2) <= n//2 and m1 != m2:
            s = conversion.get(sum_monoms(m1, m2))
            if s is not None:
                s.append((m1, m2))
     
    for m1 in product(range(n//2+1), repeat=2):
        # cases when m1 = m2
        s = conversion.get(sum_monoms(m1, m1))
        if s is not None:
            s.append((m1, m1))

    # number of variables
    print(sum([len(i) for i in conversion.values()])-len(conversion.keys()))

    return conversion


def SDPPerturbation(X, poly, inv_vec, conversion, next_monom):
    X = np.array(X)
    S = sp.zeros(X.shape[0])

    coeffs = {}
    for coeff, monom in zip(poly.coeffs(), poly.monoms()):
        if not isinstance(coeff, sp.Rational):
            coeff = sp.Rational(*rationalize(coeff, reliable = True))
        coeffs[monom] = coeff

    def coeff(x):
        t = coeffs.get(x)
        return t if t is not None else 0

    def perm_monom(x, times = 1):
        for t in range(times):
            x = next_monom(x)
        return x

    rounding = 1
    for i in range(1):
        rounding *= 1e-8
        for monom, alterns in conversion.items():
            permute = 1 if (monom[0] == monom[1] and monom[1] == monom[2]) else 3
            for j in range(permute):
                s = 0
                for altern in alterns:
                    x , y = inv_vec[perm_monom(altern[0],j)], inv_vec[perm_monom(altern[1],j)]
                    if abs(X[x,y]) < rounding * 1e-3:
                        S[x,y] = 0
                    else:
                        S[x,y] = sp.Rational(*rationalize(X[x,y], rounding, reliable=False))
                    
                    s += S[x,y]
                
                altern = alterns[0]
                x , y = inv_vec[perm_monom(altern[0],j)], inv_vec[perm_monom(altern[1],j)]
                if x != y: # then S[x,y], S[y,x] are already registered
                    S[x,y] += (coeff(monom) - s) / 2
                    S[y,x] = S[x,y]


        # S = np.array(S).astype('float')
        # print(np.linalg.eigvalsh(S))
        # print(np.linalg.norm(S - X))
        for j in range(X.shape[0]):
            print(sp.Matrix.det(S[:j,:j]),end = ' ')
        print('\n',S)


def RationalCongruence(A):
    """
    Given a matrix A, find rational 
    """
    return A


def SOS_SDP(poly, dict_monom: dict):
    n = deg(poly)
    assert (n & 1) == 0 and n > 0
    conversion = SDPConvertor(n, dict_monom)
    sos :picos.Problem = picos.Problem()

    d = (n//2+1)*(n//2+2)//2
    X = picos.SymmetricVariable('X', (d, d))

    vec = []
    for i in range(n//2+1):
        for j in range(n//2+1-i):
            vec.append((i,j))
    inv_vec = dict([(vec[i], i) for i in range(len(vec))])

    coeffs = {}
    for coeff, monom in zip(poly.coeffs(), poly.monoms()):
        coeffs[monom] = float(coeff)
    def coeff(monom):
        t = coeffs.get(monom)
        return t if t is not None else 0
    def next_monom(monom):
        return (n//2-sum(monom), monom[0])
    def prev_monom(monom):
        return (monom[1], n//2-sum(monom))

    for monom, alterns in conversion.items():
        # main = inv_vec[alternatives[0]]
        c = coeff(monom)
        sos.add_constraint(
            sum(X[inv_vec[altern[0]], inv_vec[altern[1]]] for altern in alterns) == c
        )
        # cyclize
        if not (monom[0] == monom[1] and monom[1] == monom[2]):
            sos.add_constraint(
                sum(X[inv_vec[next_monom(altern[0])], inv_vec[next_monom(altern[1])]] for altern in alterns) == c
            )
            sos.add_constraint(
                sum(X[inv_vec[prev_monom(altern[0])], inv_vec[prev_monom(altern[1])]] for altern in alterns) == c
            )
        
    sos.add_constraint(X >> 0)
    sos.solve()

    for S in SDPPerturbation(X.value, poly, inv_vec, conversion, next_monom):
        S = RationalCongruence(S)
    print(np.linalg.eigvalsh(S))


if __name__ == '__main__':
    txt = 's(a(a-b)(a-c))2+0p(a-b)2'
    txt = 's(a2)2-3s(a3b)'
    txt = PreprocessText(txt)
    SOS_SDP(txt, generate_expr(deg(txt))[0])
