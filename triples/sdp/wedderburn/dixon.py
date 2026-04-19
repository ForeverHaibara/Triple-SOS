from typing import List, Set, Union, Sequence, Generator, Iterable

from sympy import (FiniteField,
    sqrt_mod, nextprime, primitive_root,
    primefactors, ZZ, QQ
)
from sympy import MutableDenseMatrix as Matrix
from sympy.external.gmpy import sqrt as isqrt
from sympy.external.gmpy import MPZ
try:
    from sympy.external.gmpy import gcd, lcm
except ImportError:
    from math import gcd
    from functools import reduce
    lcm = lambda *args: reduce(lambda x, y: x*y//gcd(x, y), args, 1)
from sympy.polys.domains.domain import Domain
from sympy.polys.domainmatrix import DomainMatrix
from sympy.polys.factortools import dup_factor_list_include
from sympy.polys.polyclasses import ANP
from sympy.combinatorics import Permutation, PermutationGroup

from sympy import __version__ as SYMPY_VERSION
from sympy.external.importtools import version_tuple

CC = List[Set[Permutation]]

if tuple(version_tuple(SYMPY_VERSION)) >= (1, 11):
    def cyclotomic_field(n: int, alias: str = "zeta"):
        return QQ.cyclotomic_field(n, ss=True, alias=alias)
else:
    def cyclotomic_field(n: int, alias: str = "zeta"):
        """Implement QQ.cyclotomic_field to supports low
        versions of SymPy."""
        from sympy import Symbol
        from sympy.polys.specialpolys import cyclotomic_poly
        from sympy.polys.rootoftools import CRootOf
        from sympy.core.numbers import AlgebraicNumber
        root = CRootOf(cyclotomic_poly(n, polys=True), -1)
        alpha = AlgebraicNumber(root, alias=f"{alias}{n}")
        field = QQ.algebraic_field(alpha)
        field.ext.alias = Symbol(f"{alias}{n}")
        return field


def _compute_cmmatrices(cc: Sequence[CC], dom: Domain) -> Generator[DomainMatrix, None, None]:
    n = len(cc)
    rmul = Permutation.rmul_with_af
    for ind in range(n):
        m = [[-1] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if m[i][j] != -1:
                    continue
                m[i] = [0] * n
                for g in cc[ind]:
                    for t in range(n):
                        out = next(iter(cc[t]))
                        # out = g^{-1} * h
                        h = rmul(out, g)
                        if h in cc[i]:
                            m[i][t] += 1
        m = [[dom(_) for _ in row] for row in m]
        yield DomainMatrix(m, (n, n), dom)

def _group_exponent_from_cc(cc: Sequence[CC]) -> MPZ:
    orders = [int(next(iter(c)).order()) for c in cc]
    exponent = lcm(*orders)
    return exponent

def dixon_prime(order: Union[int, MPZ], exponent: Union[int, MPZ]) -> Union[int, MPZ]:
    """Find a prime p so that p > 2*sqrt(order) and p%exponent == 1."""
    if order == 1:
        # trivial group => exponent == 1
        return 3
    p = 2 * isqrt(order)
    while True:
        p = nextprime(p + 1)
        if p % exponent == 1:
            break
    return p

def _eigenspace_decomp(A: DomainMatrix) -> List[DomainMatrix]:
    """Compute eigenspaces of A on the ground domain."""
    charpoly = A.charpoly()
    factors = dup_factor_list_include(charpoly, A.domain)
    eigens = []
    for p, _ in factors:
        if len(p) == 2:
            # linear factor
            z = -p[1]/p[0]
            B = A - A.diag([z] * A.shape[0], A.domain)
            basis = B.nullspace()
            basis, pivots = basis.rref()
            eigens.append(basis)
    return eigens

def _eigenspace_split(spaces: List[DomainMatrix], N: DomainMatrix) -> List[DomainMatrix]:
    new_spaces = []
    for S in spaces:
        if S.shape[0] <= 1:
            new_spaces.append(S)
            continue
        _, pivots = S.rref()
        N0 = N.extract(range(S.shape[1]), pivots)
        sub_decomp = _eigenspace_decomp((S * N0).transpose())
        for sub_S in sub_decomp:
            # sub_S: (sub_dim x dim), S: (dim x n) -> (sub_dim x n)
            new_spaces.append(sub_S * S)
    return new_spaces

def common_esd(mats: Iterable[DomainMatrix], check_dim: bool=True) -> DomainMatrix:
    """
    Compute the common eigenspace decomposition of a list of DomainMatrices.
    Returns Z such that Z * N * Z.inv() is diagonal for all N in mats
    and Z is on the same domain as mats (assuming Z exists).
    """
    it = iter(mats)
    N0 = next(it)
    spaces = _eigenspace_decomp(N0)
    n = N0.shape[0]
    for N in it:
        if len(spaces) == n:
            break
        spaces = _eigenspace_split(spaces, N)
    if check_dim and len(spaces) != n:
        # not expected to happen
        raise ValueError("Failed to compute the common eigenspace decomposition.")
    return DomainMatrix.vstack(*spaces)

def _get_invmap(cc: Sequence[CC]) -> List[int]:
    """Compute inv_map[i] = j such that class[i]**-1 == class[j]."""
    inv_map = [-1] * len(cc)
    reps = [next(iter(c)) for c in cc]
    inv_reps = [~g for g in reps]
    for i, ig in enumerate(inv_reps):
        if inv_map[i] != -1:
            continue
        for j, c in enumerate(cc):
            if ig in c:
                inv_map[i] = j
                inv_map[j] = i
                break
    return inv_map

def _get_powermap(cc: Sequence[CC], exponent: Union[int, MPZ]) -> List[List[int]]:
    """Compute pm[i][pow] = k such that class[i]**pow == k."""
    n = len(cc)
    pm = [[-1] * exponent for _ in range(n)]
    reps = [next(iter(c)) for c in cc]
    rmul = Permutation.rmul_with_af
    identity = rmul(reps[0], ~reps[0])
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

def normalize_fp(cc: Sequence[CC], esd: DomainMatrix) -> List[List]:
    Fp = esd.domain
    p = Fp.mod # type: ignore
    n = len(cc)
    rows = esd.to_list()
    cc_sizes = [Fp(len(c)) for c in cc]
    G_order = sum(len(cc[t]) for t in range(len(cc)))

    inv_map = _get_invmap(cc)
    normalized_rows = []
    for row in rows:
        # 1. normalize so the first class is 1
        scale = 1/Fp(row[0])
        row = [x * scale for x in row]

        # 2. chi(1)^2 * sum |Ci| * row[i] * row[inv_i] = |G|
        dot = sum(cc_sizes[k] * row[k] * row[inv_map[k]] for k in range(n))

        # chi1_sq = |G| / dot
        chi1_sq = Fp(G_order) / dot

        root = sqrt_mod(int(chi1_sq), p)
        if root is None:
            # not expected to happen
            raise ValueError(f"Failed to compute sqrt({chi1_sq}) on {Fp}")
        chi1 = Fp(root)

        normalized_rows.append([x * chi1 for x in row])

    return normalized_rows


def _get_global_conductor(rows: List[List], pm: List[List[int]], m: Union[int, MPZ]) -> Union[int, MPZ]:
    """
    Find the smallest natural number k such that all values of the
    character table can be embedded in the k-th cyclotomic field.
    """
    n = len(rows)
    gal_m = [a for a in range(m) if gcd(a, m) == 1]
    p_factors = primefactors(m)[::-1]

    for p in p_factors:
        while m % p == 0:
            # test invariance under m//p
            m = m//p
            r = 1 % m

            fixed = True
            for a in gal_m:
                if a % m != r:
                    continue
                for i in range(n):
                    row = rows[i]
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

def _lift_to_minimal_field(normalized_rows, pm, k, e, Fp):
    n = len(normalized_rows)
    p = Fp.mod
    half_p = p // 2

    if k == 1:
        # integer character table
        int_rows = []
        for i in range(n):
            # abs(every entry) <= sqrt(|G|) <= p//2
            row = [ZZ(int(v)) for v in normalized_rows[i]]
            row = [v if v <= half_p else v - p for v in row]
            int_rows.append(row)
        _sort_characters(int_rows, ZZ)
        return DomainMatrix(int_rows, (n, n), ZZ)

    # dom = QQ.cyclotomic_field(k, ss=True)
    dom = cyclotomic_field(k)

    x = Fp(primitive_root(p))**((p - 1) // k)

    gal = [a for a in range(k) if gcd(a, k) == 1]
    phi = len(gal)

    # V[a, i] = (x**a)**i
    V = []
    for a in gal:
        xa = x**a
        row = [xa**i for i in range(phi)]
        V.append(row)

    V_mat = DomainMatrix(V, (phi, phi), Fp)
    V_inv = V_mat.inv()

    # for a in (Z/k)*, find A in (Z/exp)* that A = a (mod k)
    gal_lifted = []
    for a in gal:
        A = a
        while gcd(A, e) != 1:
            A += k
        gal_lifted.append(A)

    dM = []
    for i in range(n):
        char_row_fp = normalized_rows[i]
        row_anps = []

        for j in range(n):
            # [chi(g^A1), chi(g^A2), ...] mod p
            b_vec_data = []
            for A in gal_lifted:
                class_of_gA = pm[j][A % e]
                b_vec_data.append([char_row_fp[class_of_gA]])

            b_vec = DomainMatrix(b_vec_data, (phi, 1), Fp)

            # c = V^-1 * b
            c_vec = (V_inv * b_vec).to_list_flat()

            c_ints = [int(v) for v in c_vec]
            c_ints = [v if v <= half_p else v - p for v in c_ints]

            row_anps.append(ANP(c_ints[::-1], dom.mod, QQ))

        dM.append(row_anps)

    _sort_characters(dM, dom)
    return DomainMatrix(dM, (n, n), dom)

def _sort_characters(rows: List[List], dom: Domain):
    """
    Sort the character table and move the trivial
    character to the first row. Done in-place.
    """
    one = dom.one
    rows.sort()
    for i in range(len(rows)):
        if all(v == one for v in rows[i]):
            rows[0], rows[i] = rows[i], rows[0]
            break


def dixon_character_table(conjugacy_classes: Union[Sequence[CC], PermutationGroup]) -> Matrix:
    """
    Computes the character table of a permutation group
    given its character table, using Dixon's algorithm.

    References
    ----------
    [1] John D. Dixon. High speed computation of group characters. Numerische
    Mathematik, 10:446-450, 1967.

    [2] Gerhard J. A. Schneider. Dixon's character table algorithm revisited. Journal of
    Symbolic Computation, 9(5-6):601-606, 1990.

    [3] Jean Michel. Handbook of computational group theory. Math. Comput., 75, 01 2006.
    """
    if isinstance(conjugacy_classes, PermutationGroup):
        cc = conjugacy_classes.conjugacy_classes()
    else:
        cc = conjugacy_classes
    order = sum(len(cc[t]) for t in range(len(cc)))
    exponent = _group_exponent_from_cc(cc)
    p = dixon_prime(order, exponent)
    Fp = FiniteField(p)

    mats = _compute_cmmatrices(cc, Fp)
    esd = common_esd(mats)

    normalized = normalize_fp(cc, esd)
    pm = _get_powermap(cc, exponent)

    conductor = _get_global_conductor(normalized, pm, exponent)
    dm = _lift_to_minimal_field(normalized, pm, conductor, exponent, Fp)

    return Matrix._fromrep(dm)
