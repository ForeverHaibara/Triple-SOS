from typing import List, Callable, Optional

from sympy import QQ, totient, mobius
from sympy import MutableDenseMatrix as Matrix
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.combinatorics import PermutationGroup, Permutation
try:
    from sympy.external.gmpy import gcd
except ImportError:
    from math import gcd

from .character_table import character_table
from .symmetric import symmetry_adapted_basis_sn
from ..arithmetic import solve_columnspace

def _ramanujan_sum(K: int):
    """
    Compute [sum(z**(j*a) for a in range(K) if gcd(a, K) == 1) for j in range(K)]
    """
    gcds = [gcd(a, K) for a in range(K)]
    _totient = lambda x: int(totient(x))
    _mobius = lambda x: int(mobius(x))
    phi = _totient(K)
    ramanujan = [phi*_mobius(K//g)//(_totient(K//g)) for g in gcds]
    return ramanujan


def symmetry_adapted_basis(
    G: PermutationGroup,
    representation: Optional[Callable[[Permutation], List[int]]]=None
) -> List[Matrix]:
    """
    Compute the symmetry-adapted basis of the representation of G.

    The returned basis is a list of matrices so that
    `Q^TAQ = diag([Qi.T * A * Qi for Qi in Qs])`
    holds if A is symmetric and commutative with the representation matrices.
    The size of each block matches the total dimension (including multiplicity)
    of an irreducible representation.

    Each Q is not ensured to be an orthogonal matrix.

    Parameters
    ----------
    G: PermutationGroup
        The permutation group.
    representation: Optional[Callable[[Permutation], List[int]]], optional]
        A permutation representation of G. If not provided,
        it uses the default representation.

    Returns
    -------
    List[Matrix]
        A list of matrices `Qs`, so that
        `Q^TAQ = diag([Qi.T * A * Qi for Qi in Qs])`
        if A is symmetric and commutative with the representation matrices.
    """
    if G.is_symmetric:
        return symmetry_adapted_basis_sn(G.degree, representation)
    # TODO: test whether it is isomorphic to Sn
    return _symmetry_adapted_basis(G, representation)

def _symmetry_adapted_basis(
    G: PermutationGroup,
    representation: Optional[Callable[[Permutation], List[int]]]=None
) -> List[Matrix]:
    """Internal function to compute the symmetry-adapted_basis
    for a general permutation group G."""
    if representation is None:
        representation = lambda g: g.array_form
    n = len(representation(G.identity))

    cc = G.conjugacy_classes()
    tbl = character_table(G, cc=cc)

    # fixed = []
    # for c in cc:
    #     r = next(iter(c))
    #     fixed.append(sum(i == j for i, j in enumerate(representation(r))))
    # mul = list(tbl.T.LUsolve(Matrix(fixed)).n(5).applyfunc(round))

    dm = tbl._rep
    if dm.domain.is_ZZ:
        dm = dm.convert_to(QQ)
    dom = dm.domain
    ctab = dm.to_list()

    cols = []
    zero = dom.zero
    m = len(ctab)
    # order = sum(len(c) for c in cc)

    K = int(dom.ext.alias.name.lstrip('zeta')) if dom.is_Algebraic else 1 # type: ignore
    ramanujan = _ramanujan_sum(K)
    # QQzero = QQ.zero
    galsum = lambda anp: QQ(sum(
        (r*v for r, v in zip(ramanujan, anp.rep[::-1]))))

    seen = []
    # perms = []
    for chi in range(m):
        P = [[zero] * n for _ in range(n)]
        for c in range(m):
            chi_g = ctab[chi][c]
            for g in cc[c]:
                perm = representation(g)
                # perms.append(perm)
                for i, j in enumerate(perm):
                    P[j][i] += chi_g # no need to conjugate

        if K > 1:
            P = [[galsum(anp) for anp in row] for row in P]
        else:
            P = [[QQ(v) for v in row] for row in P]

        # dim = ctab[chi][0] if K == 1 else ctab[chi][0].rep[0]
        # scale = dim / order
        # P = [[i*scale for i in row] for row in P]

        proj = DomainMatrix(P, (n, n), QQ)
        if K > 1:
            if proj in seen:
                continue
            seen.append(proj)

        # DomainMatrix.columnspace has version compatibility issues
        # so we use our own version
        ns = solve_columnspace(proj)
        cols.append(ns)

    return [Matrix._fromrep(ns) for ns in cols]
