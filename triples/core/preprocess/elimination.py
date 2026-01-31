from typing import Tuple, List, Set, Optional

from sympy import Poly, Expr, Symbol, Integer, Mul, QQ, ZZ
from sympy import MutableDenseMatrix as Matrix

from ..problem import InequalityProblem

def _identify_matrix_symmetry(S: Matrix) -> List[List[int]]:
    """
    Given a symmetric matrix `S`, indices `i, j` are
    equivalent if exchanging `i, j` does not change `S`.
    Identify the clusters of equivalent indices.
    """
    n = S.shape[0]
    parent = list(range(n))

    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    ddm = S._rep.rep.to_ddm()
    _S = lambda i, j: ddm[i][j]

    # this can be optimized, e.g. checking the elements
    # in the first row
    for i in range(n):
        for j in range(i + 1, n):
            if _S(i, i) != _S(j, j):
                continue

            is_equivalent = True
            for k in range(n):
                if k == i or k == j:
                    continue
                if _S(i, k) != _S(j, k):
                    is_equivalent = False
                    break

            if is_equivalent:
                union(i, j)

    cluster_dict = {}
    for i in range(n):
        root = find(i)
        if root not in cluster_dict:
            cluster_dict[root] = []
        cluster_dict[root].append(i)

    clusters = sorted(list(cluster_dict.values()), key=lambda x: min(x))
    return clusters

def _rowwise_primitive(A: Matrix) -> Matrix:
    from sympy.polys.densetools import dup_primitive
    rows = A._rep.rep.to_dod()
    dom = A._rep.domain
    for row, terms in rows.items():
        prim_terms = dup_primitive(list(terms.values()), dom)[1]
        rows[row] = {i: v for i, v in zip(terms.keys(), prim_terms)}
    return Matrix._fromrep(A._rep.from_dod(rows, A.shape, dom).convert_to(ZZ))


def _symmetry_adapted_nullspace(A: Matrix) -> Tuple[Matrix, List[List[int]]]:
    m, n = A.shape
    A = (-Matrix.eye(m, m)).row_join(A)

    P2 = A.T * (A * A.T).inv() * A
    P = Matrix.eye(*P2.shape) - P2

    # rearrange indices by clustered symmetry
    clusters = _identify_matrix_symmetry(P[m:, m:])
    concatenated_clusters = list(range(m)) + [i+m for group in clusters for i in group]
    P = P[concatenated_clusters, concatenated_clusters]

    basis, pivots = P.rref()
    rank = max(pivots) + 1
    basis = basis[:rank, m:].T

    return basis, clusters

def _inv_integer_matrix(X: Matrix) -> Matrix:
    """
    Given an integer, full-column-rank matrix `X`, find an integer matrix
    `A` such that `AX = I` by Smith normal form.

    If such integer matrix exists, returns `A`. If not, returns a rational
    matrix `A` such that `AX = I` but `A._rep.domain = QQ`.
    """
    if X.shape[0] < X.shape[1]:
        raise ValueError("X should have more rows than columns.")
    from sympy.polys.matrices.normalforms import smith_normal_decomp
    smith, left, right = smith_normal_decomp(X._rep)

    # left * X * right == smith
    # =>  right * smith.pinv() * left * X == I

    diag = smith.rep.diagonal()
    if any(v == 0 for v in diag):
        raise ValueError("X should be full rank.")
    if not all(v == 1 for v in diag):
        # cast pinv to QQ
        pinv = smith.from_dok({(i, i): QQ(1, v) for i, v in enumerate(diag)},
                (smith.shape[1], smith.shape[0]), QQ)
        left = pinv * left
    return Matrix._fromrep(right * left)


def _get_free_symbols(symbols: Set[Symbol], n: int, prefix: str="x") -> List[Symbol]:
    counter = 0
    m = len(prefix)
    for symbol in symbols:
        name = symbol.name
        if name.startswith(prefix) and name[m:].isdecimal():
            counter = max(counter, int(name[m:]))
    counter += 1
    return [Symbol(prefix+str(i)) for i in range(counter, counter+n)]


def eliminate_power_constraints(problem: InequalityProblem):
    ineq_constraints = problem.ineq_constraints
    eq_constraints   = problem.eq_constraints

    mat = []
    exprs = []
    for eq, e in eq_constraints.items():
        terms = eq.terms()
        if len(terms) != 2:
            continue
        m0, c0 = terms[0]
        m1, c1 = terms[1]
        c = (-c0/c1)
        m = tuple(i - j for i, j in zip(m0, m1))
        if c != 1:
            continue
        exprs.append(e/Mul(-c1, *[g**p for g, p in zip(eq.gens, m1)]))
        mat.append(list(m))
    if len(mat) == 0:
        return problem, lambda x: x
    mat = Matrix(mat)

    basis, clusters = _symmetry_adapted_nullspace(mat)

    # use an integer basis
    basis = _rowwise_primitive(basis.T).T

    inv_basis = _inv_integer_matrix(basis)
    # if not inv_basis._rep.domain.is_ZZ:
    #     return problem, lambda x: x

    new_gens = _get_free_symbols(problem.free_symbols, basis.shape[1])
    transform = {
        g0: Mul(*[g**p for g, p in zip(new_gens, row)])
            for g0, row in zip(problem.gens, basis.tolist())
    }
    inv_transform = {
        g: Mul(*[g0**p for g0, p in zip(problem.gens, row)])
            for g, row in zip(new_gens, inv_basis.tolist())
    }
    # print('transform =', transform, '\ninv_transform =', inv_transform)

    problem, restore_transform = problem.transform(transform, inv_transform)

    problem, restore_marginalize = problem.marginalize(
        {g: Integer(1) for g in new_gens[:mat.shape[0]]},
        {g: g - 1 for g, e in zip(new_gens[:mat.shape[0]], exprs)})

    def composed(x):
        y = restore_transform(restore_marginalize(x))
        if y is None:
            return None
        return y.xreplace(
            {g: (e + 1).together()**p for g, e, p in zip(
                new_gens[:mat.shape[0]], exprs, inv_basis.diagonal()[:mat.shape[0]])})

    return problem, composed
