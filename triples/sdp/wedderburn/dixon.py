"""
Implements the Dixon's algorithm for computing the character
table in Python. Adapted from:
https://github.com/kalmarek/SymbolicWedderburn.jl.
"""

from typing import List, Set, Union

from sympy import (Poly, Symbol, FiniteField,
    sqrt_mod, nextprime, primitive_root,
    primefactors, ZZ, QQ
)
from sympy import MutableDenseMatrix as Matrix
from sympy.external.gmpy import gcd, sqrt as isqrt
from sympy.polys.domainmatrix import DomainMatrix
from sympy.polys.polyclasses import ANP
from sympy.combinatorics import Permutation, PermutationGroup

class CMMatrix:
    """
    Internal class to represent the class multiplication matrix.
    """
    def __init__(self,
        cc: List[Set[Permutation]],
        r: int
    ):
        self.cc = cc
        self.r = r
        self.m = [[-1]*len(cc) for _ in range(len(cc))]
    @property
    def shape(self):
        return len(self.cc), len(self.cc)
    def __getitem__(self, idx):
        """Supports only `cmm[int,int]` or `cmm[slice, slice]`"""
        i, j = idx
        if isinstance(i, int) and isinstance(j, int):
            return self.getitem(i,j)
        if isinstance(i, slice):
            i = list(range(*i.indices(len(self.cc))))
        if isinstance(j, slice):
            j = list(range(*j.indices(len(self.cc))))
        return [[self.getitem(i_, j_) for j_ in j] for i_ in i]
    def getitem(self, i, j):
        m = self.m
        if m[i][j] != -1:
            return m[i][j]
        cc = self.cc
        n = len(cc)
        m[i] = [0] * n
        rmul = Permutation.rmul_with_af
        for g in cc[self.r]:
            for t in range(n):
                out = next(iter(cc[t]))
                # out = g^{-1} * h
                h = rmul(out, g)
                if h in cc[i]:
                    m[i][t] += 1
        return m[i][j]

def _group_exponent_from_cc(cc: List[Set[Permutation]]) -> int:
    from sympy.external.gmpy import lcm
    # TODO: old versions did not have lcm
    orders = [int(next(iter(c)).order()) for c in cc]
    exponent = lcm(*orders)
    return exponent

def dixon_prime(order: int, exponent: int) -> int:
    """Find a prime p so that p > 2*sqrt(order) and p%exponent == 1."""
    if order == 1:
        # trivial group => exponent == 1
        return 3
    p = 2*isqrt(order)
    while True:
        p = nextprime(p + 1)
        if p % exponent == 1:
            break
    return p

def eigenspace_decomposition(A: DomainMatrix) -> List[DomainMatrix]:
    """Compute eigenspaces of A on the ground domain."""
    A = A.transpose()
    Fp = A.domain
    charpoly = Poly(A.charpoly(), Symbol('x'), domain=Fp)
    eigens = []
    for z in charpoly.ground_roots():
        z = Fp(z)
        B = A - A.diag([z] * A.shape[0], Fp)
        basis = B.nullspace()
        basis, pivots = basis.rref()
        eigens.append(basis)
    return eigens

def refine_spaces(spaces: List[DomainMatrix], N: CMMatrix, Fp) -> List[DomainMatrix]:
    new_spaces = []
    dM = DomainMatrix.from_list(N[:, :], Fp)
    for S in spaces:
        if S.shape[0] <= 1:
            new_spaces.append(S)
            continue
        _, pivots = S.rref()
        N0 = dM.extract(range(S.shape[1]), pivots)
        sub_decomp = eigenspace_decomposition(S * N0)
        for sub_S in sub_decomp:
            # sub_S: (sub_dim x dim), S: (dim x n) -> (sub_dim x n)
            new_spaces.append(sub_S * S)
    return new_spaces

def common_esd(Ns: List[CMMatrix], Fp: FiniteField) -> DomainMatrix:
    """
    Compute the common eigenspace decomposition of a list of CMMatrix
    over Fp. Returns Z such that Z * N * Z.inv() is diagonal for all
    N in Ns (over Fp).
    """
    N0 = DomainMatrix.from_list(Ns[0][:, :], Fp)
    spaces = eigenspace_decomposition(N0)
    n = len(Ns)
    for i in range(1, n):
        if len(spaces) == n:
            break
        spaces = refine_spaces(spaces, Ns[i], Fp)
    return DomainMatrix.vstack(*spaces)

def get_inv_map(cc: List[Set[Permutation]]) -> List[int]:
    """Compute inv_map[i] = j such that class[i]**-1 == class[j]."""
    inv_map = [0] * len(cc)
    reps = [next(iter(c)) for c in cc]
    inv_reps = [~g for g in reps]
    for i, ig in enumerate(inv_reps):
        for j, c in enumerate(cc):
            if ig in c:
                inv_map[i] = j
                break
    return inv_map

def get_powermap(cc: List[Set[Permutation]], exponent: int) -> List[List[int]]:
    """Compute pm[i][pow] = k such that class[i]**pow == k."""
    n = len(cc)
    pm = [[0] * exponent for _ in range(n)]
    reps = [next(iter(c)) for c in cc]
    rmul = Permutation.rmul_with_af
    identity = Permutation(size=reps[0].size)
    for i in range(n):
        g = reps[i]
        gk = identity
        for k in range(exponent):
            for t in range(n):
                if gk in cc[t]:
                    pm[i][k] = t
                    break
            gk = rmul(gk, g)
    return pm

def normalize_fp(cc: List[Set[Permutation]], esd: DomainMatrix):
    Fp = esd.domain
    p = Fp.mod
    rows = esd.to_list()
    n = len(cc)
    cc_sizes = [Fp(len(c)) for c in cc]
    G_order = sum(len(cc[t]) for t in range(len(cc)))

    inv_map = get_inv_map(cc)
    normalized_rows = []
    for row in rows:
        # 1. normalize so the first class is 1
        scale = 1/Fp(row[0])
        row = [Fp(x) * scale for x in row]

        # 2. chi(1)^2 * sum |Ci| * row[i] * row[inv_i] = |G|
        dot = sum(cc_sizes[k] * row[k] * row[inv_map[k]] for k in range(n))
        
        # chi1_sq = |G| / dot
        chi1_sq = Fp(G_order) / dot

        root = sqrt_mod(int(chi1_sq), p)
        if root is None:
            raise ValueError(f"Failed to compute sqrt({chi1_sq}) on {Fp}")
        chi1 = Fp(root)

        normalized_rows.append([x * chi1 for x in row])

    return normalized_rows


def _compute_multiplicities(
    pm: List[List[int]],
    normalized_rows: List[List[int]],
    Fp: FiniteField,
    exponent: int,
    sort: bool = True
) -> List[List[int]]:
    n = len(normalized_rows)

    zeta = Fp(primitive_root(Fp.mod))**(Fp.mod//exponent)
    inv_zeta = 1/zeta
    inv_e = 1/Fp(exponent)

    multiplicities = [] # [char_idx][class_idx][root_pow]

    for i in range(n):
        char_mults = []
        for j in range(n):
            class_mults = []
            # a_k = (1/e) * sum_{l=0}^{e-1} chi(g^l) * zeta^(-kl)
            for k in range(exponent):
                val = sum(normalized_rows[i][pm[j][l]] \
                          * (inv_zeta**(k * l)) for l in range(exponent))
                class_mults.append(int(val * inv_e))
            char_mults.append(class_mults)
        multiplicities.append(char_mults)

    if sort:
        multiplicities.sort()

        # move the trivial character to the first row
        for k, row in enumerate(multiplicities):
            if all(v[0] == 1 and not any(v[1:]) for v in row):
                multiplicities[0], multiplicities[k] =\
                    multiplicities[k], multiplicities[0]
                break

    return multiplicities


def dixon_character_table(
    G_or_cc: Union[PermutationGroup, List[Set[Permutation]]]
) -> Matrix:
    if isinstance(G_or_cc, PermutationGroup):
        cc = G_or_cc.conjugacy_classes()
    else:
        cc = G_or_cc

    order = sum(len(cc[t]) for t in range(len(cc)))
    exponent = _group_exponent_from_cc(cc)
    p = dixon_prime(order, exponent)
    Fp = FiniteField(p)

    Ns = [CMMatrix(cc, i) for i in range(len(cc))]
    esd = common_esd(Ns, Fp)

    normalized = normalize_fp(cc, esd)
    pm = get_powermap(cc, exponent)
    conductor = _get_global_conductor(normalized, pm, exponent)

    mults = _compute_multiplicities(pm, normalized, Fp, exponent)

    dom = QQ.cyclotomic_field(exponent, ss=True)
    mod = dom.mod
    mults =  [[ANP(rep[::-1], mod, QQ) for rep in row] for row in mults]
    dm = DomainMatrix(mults, (len(mults), len(mults)), dom)

    if conductor > 1:
        # TODO: this should be made faster
        dm = dm.convert_to(QQ.cyclotomic_field(conductor, ss=True))
    else:
        dm = dm.convert_to(ZZ)

    tbl = Matrix._fromrep(dm)
    return tbl


def _get_global_conductor(normalized_rows, pm: List[List[int]], m: int):
    n = len(normalized_rows)
    gal_m = [a for a in range(m) if gcd(a, m) == 1]
    p_factors = primefactors(m)[::-1]

    for p in p_factors:
        while m % p == 0:
            m = m//p
            r = 1 % m

            fixed = True
            for a in gal_m:
                if a % m != r:
                    continue
                for i in range(n):
                    row = normalized_rows[i]
                    for j in range(n):
                        if row[pm[j][a]] != row[j]:
                            fixed = False
                            break
                    if not fixed:
                        break
                if not fixed:
                    break

            if not fixed:
                m = m * p # restore
                break
    return m
