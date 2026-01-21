from typing import List, Optional

from sympy import Expr, Add, factorial
from sympy import MutableDenseMatrix as Matrix
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.ddm import DDM
from sympy.combinatorics import SymmetricGroup

from ....sdp import congruence
from ....utils import Coeff, CyclicSum

def make_mat_from_coeff(coeff: Coeff) -> Optional[Matrix]:
    nvars = len(coeff.gens)
    zero = coeff.domain.zero
    mat = [[zero for _ in range(nvars)] for __ in range(nvars)]
    for k, v in coeff.items():
        if v == 0:
            continue
        inds = []
        for i in range(nvars):
            if k[i] > 2:
                return None
            elif k[i] == 2:
                inds = (i, i)
                break
            elif k[i] == 1:
                if len(inds) == 2:
                    return None
                inds.append(i)
        if len(inds) == 1:
            # nonhomogeneous
            return None
            inds.append(nvars)
        elif len(inds) == 0:
            # nonhomogeneous
            return None
            inds = (nvars, nvars)
        if inds[0] == inds[1]:
            mat[inds[0]][inds[0]] = v
        else:
            mat[inds[0]][inds[1]] = v/2
            mat[inds[1]][inds[0]] = v/2

    mat = coeff.as_matrix(mat, (nvars, nvars))
    return mat


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

    clusters = sorted(list(cluster_dict.values()), key=lambda x: x[0])
    return clusters

def _isotopic_decomposition(S: Matrix, clusters: List[List[int]]):
    """
    Given a symmetric matrix `S` and its symmetric clusters `s1, ..., sm`,
    find an `m * m` matrix `M` and a length-`m` vector `c` such that:
    ```
    v^TSv == Σ_{i=1}^m Σ_{j=1}^m (
                sum(v[_] for _ in clusters[i]) * M[i,j] * sum(v[_] for _ in clusters[j]))
            + Σ_{i=1}^m c[i] * Σ_{1<=j<k<=len(clusters[i])} (
                (v[clusters[i][j]] - v[clusters[i][k]])**2)
    ```
    """
    m = len(clusters)

    M = [[0] * m for _ in range(m)]
    c = [0] * m

    ddm = S._rep.rep.to_ddm()
    _S = lambda i, j: ddm[i][j]

    for i in range(m):
        cluster_i = clusters[i]
        d_i = len(cluster_i)
        idx_i = cluster_i[0] # representative of cluster i

        # compute c[i], the internal fluctuation coefficient
        if d_i > 1:
            # two different elements in the cluster: p, q
            p, q = cluster_i[0], cluster_i[1]
            val_diag = _S(p, p)
            val_off = _S(p, q)
            # fomula: c = (a - b) / d
            c[i] = (val_diag - val_off) / d_i
        else:
            # c[i] = 0
            pass

        # compute M[i, i], the diagonal block mean coefficient
        # formula: M_ii = (a + (d-1)b) / d
        # equivalent to: the sum of the elements in the cluster / d
        row_sum_internal = sum(_S(idx_i, k) for k in cluster_i)
        M[i][i] = row_sum_internal / d_i

        # compute M[i, j], the cross block mean coefficient
        for j in range(i + 1, m):
            cluster_j = clusters[j]
            idx_j = cluster_j[0]
            # the elements between the clusters are constants
            val_cross = _S(idx_i, idx_j)
            M[i][j] = M[j][i] = val_cross

    return M, c

def _isotopic_congruence(S: Matrix, vec: List[Expr]) -> Optional[Expr]:
    """
    Given a PSD matrix `S` and a vector, compute a nice SOS
    decomposition of `vec.T * S * vec` by exploiting its symmetric clusters.
    """
    clusters = _identify_matrix_symmetry(S)
    M, c = _isotopic_decomposition(S, clusters)

    domain = S._rep.domain
    if any(v != 0 and domain.to_sympy(v) < 0 for v in c):
        return None

    m = len(clusters)
    M = Matrix._fromrep(DomainMatrix.from_rep(DDM(M, (m, m), domain)))

    cong = congruence(M)
    if cong is None:
        return None
    U, S = cong

    cluster_gens = [[vec[j] for j in cluster] for cluster in clusters]
    cluster_sums = [Add(*cluster_gens[i]) for i in range(m)]

    cross = [S[i] * Add(*[U[i, j] * cluster_sums[j] for j in range(m)])**2 for i in range(m)]
    fluct = []
    for i in range(m):
        gens = cluster_gens[i]
        d = len(clusters[i])
        if d == 1:
            v = gens[0]
        elif d == 2:
            v = (gens[0] - gens[1])**2
        else:
            # translate Σ_{i<j}(g[i]-g[j])**2 to the complete
            # symmetric sum with d! terms:
            v = 1/factorial(d-2)/2 * CyclicSum(
                (gens[0] - gens[1])**2, gens, SymmetricGroup(d), evaluate=False)
        fluct.append(c[i] * v)
    return Add(*cross, *fluct)


def sos_struct_nvars_quadratic(coeff: Coeff, **kwargs):
    """
    Solve a quadratic inequality on real numbers. Coeff
    must be homogeneous.
    """
    mat = make_mat_from_coeff(coeff)
    if mat is None:
        return None
    solution = sos_struct_nvars_quadratic_real(coeff, mat)
    if solution is not None:
        return solution
    return sos_struct_nvars_quadratic_copositive(coeff, mat)


def sos_struct_nvars_quadratic_real(
    coeff: Coeff, mat: Optional[Matrix]=None, **kwargs
):
    """
    Solve a (homogeneous) quadratic form over real numbers by
    LDL decomposition.

    Examples
    --------
    :: ineqs = [], gens = [a,b,c,d,e,f]

    => s((b-a)2)-(f-a)2+a2+f2-2(1-cos(pi/7))s(a2) # Fan-Taussky-Todd ineq

    => 11/2s(a2)+10(s(ab)-fa)+8(s(ac)-fb-ea)+6(s(ad)-fc-eb-da)+4(ae+bf)+2af # 2023 CMO

    => 5a2-6ab-8ac-8ad+2ae-8af+5b2-8bc-8bd+2be-8bf+19c2+32cd-14ce+32cf+19d2-14de+32df+10e2-14ef+19f2
    """
    if mat is None:
        return None

    # although we can use `congruence` directly,
    # isotopic congruence exploits the symmetry of the matrix
    # and yields nicer results when decomposing the matrix
    return _isotopic_congruence(mat, coeff.gens)


def sos_struct_nvars_quadratic_copositive(
    coeff: Coeff, mat: Optional[Matrix]=None, **kwargs
):
    """TODO: Implement copositive cases, e.g., Motzkin-Straus theorem."""
    return None
