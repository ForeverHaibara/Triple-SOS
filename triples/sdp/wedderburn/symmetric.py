from typing import Tuple, List, Dict, Set, Callable, Optional, Any
from collections import defaultdict
from itertools import permutations, product

from sympy import MutableDenseMatrix as Matrix
from sympy.combinatorics.permutations import Permutation, _af_parity
from sympy.utilities.iterables import ordered_partitions

def sign_of(perm: Tuple[int, ...]) -> int:
    """
    Compute the sign of a permutation of 0, 1, ..., n-1.

    Parameters
    ----------
    perm: Tuple[int, ...]
        The permutation to compute the sign of.

    Returns
    -------
    int
        The sign of the permutation.

    Examples
    --------
    >>> sign_of((1,2,0,5,4,3))
    -1
    """
    return -1 if _af_parity(perm) % 2 else 1


def murnaghan_nakayama_character_table(n: int, cc: Optional[List[Set[Permutation]]]=None) -> Matrix:
    """
    Compute the character table of a symmetric group Sn
    using the Murnaghan-Nakayama formula.

    Parameters
    ----------
    n: int
        The size of the symmetric group.
    cc: Optional[List[Set[Permutation]]], optional
        The conjugacy classes of Sn symmetric group.
        If provided, columns are sorted to match the conjugacy classes.

    Returns
    -------
    Matrix
        The character table of Sn.
    """
    if n < 0:
        raise ValueError("n must be nonnegative")
    if n == 0:
        return Matrix([[1]])

    def normalize(p):
        return tuple(reversed(p))

    parts = [normalize(p) for p in ordered_partitions(n)]
    by_size = {0: [()]}
    for k in range(1, n + 1):
        by_size[k] = [normalize(p) for p in ordered_partitions(k)]

    hook_memo = {}
    char_memo = {}

    def rim_hooks(lam, k):
        key = (lam, k)
        if key in hook_memo:
            return hook_memo[key]

        target = sum(lam) - k
        if target < 0:
            hook_memo[key] = ()
            return ()

        res = []
        for mu in by_size[target]:
            m = max(len(lam), len(mu))
            cells = []
            ok = True

            for i in range(m):
                a = lam[i] if i < len(lam) else 0
                b = mu[i] if i < len(mu) else 0
                if b > a:
                    ok = False
                    break
                for j in range(b + 1, a + 1):
                    cells.append((i, j))

            if not ok or len(cells) != k:
                continue

            s = set(cells)
            stack = [cells[0]]
            seen = {cells[0]}

            while stack:
                i, j = stack.pop()
                for nb in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                    if nb in s and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)

            if len(seen) != k:
                continue

            rows = {i for i, _ in s}
            cols = {j for _, j in s}
            ribbon = True

            for i in rows:
                for j in cols:
                    if ((i, j) in s and (i + 1, j) in s and
                            (i, j + 1) in s and (i + 1, j + 1) in s):
                        ribbon = False
                        break
                if not ribbon:
                    break

            if ribbon:
                res.append((mu, len(rows) - 1))

        hook_memo[key] = tuple(res)
        return hook_memo[key]

    def chi(lam, mu):
        key = (lam, mu)
        if key in char_memo:
            return char_memo[key]

        if sum(lam) != sum(mu):
            return 0
        if not mu:
            return 1 if not lam else 0
        if not lam:
            return 0

        total = 0
        first = mu[0]
        rest = mu[1:]

        for nu, height in rim_hooks(lam, first):
            total += (-1) ** height * chi(nu, rest)

        char_memo[key] = total
        return total

    raw_rows = [[chi(lam, mu) for mu in parts] for lam in parts]

    rows = sorted(
        zip(parts, raw_rows),
        key=lambda t: (t[1][0], 0 if t[0] == (n,) else 1, t[0]),
    )

    table = [row for _, row in rows]

    if cc is not None:
        part_to_col = {part: i for i, part in enumerate(parts)}

        cc_parts = []
        for cls in cc:
            if not cls:
                raise ValueError("each conjugacy class set must be nonempty")
            perm = next(iter(cls))
            lengths = [len(c) for c in perm.full_cyclic_form]
            part = tuple(sorted(lengths, reverse=True))
            if sum(part) != n:
                raise ValueError("permutations in cc must lie in S_n")
            cc_parts.append(part)

        if len(cc_parts) != len(parts):
            raise ValueError("cc must contain exactly p(n) conjugacy classes")

        try:
            col_order = [part_to_col[p] for p in cc_parts]
        except KeyError as exc:
            raise ValueError("cc contains an invalid cycle type for S_n") from exc

        table = [[row[j] for j in col_order] for row in table]

    return Matrix(table)


def young_symmetrizers(
    n: int,
    action: Optional[Callable[[Tuple[int, ...]], Any]]=None
) -> Dict[Tuple[int, ...], Dict[Any, int]]:
    """
    Compute a dict of young symmetrizers of degree n. The returned dict
    has partitions as keys and group elements as values.
    The time and space complexity is at least O(n!).

    Parameters
    ----------
    n: int
        The size of the symmetric group.

    action: Optional[Callable[[Tuple[int, ...]], Any], optional]
        A homomorphism from Permutations to some hashable values.

    Returns
    -------
    Dict[Tuple[int, ...], Dict[Any, int]]
        A dict of young symmetrizers of degree n.

    Examples
    --------
    >>> young_symmetrizers(3) # doctest: +SKIP
    {(3,): {(0, 1, 2): 1,
      (0, 2, 1): 1,
      (1, 0, 2): 1,
      (1, 2, 0): 1,
      (2, 0, 1): 1,
      (2, 1, 0): 1},
     (2, 1): {(0, 1, 2): 1, (2, 1, 0): -1, (1, 0, 2): 1, (2, 0, 1): -1},
     (1, 1, 1): {(0, 1, 2): 1,
      (0, 2, 1): -1,
      (1, 0, 2): -1,
      (1, 2, 0): 1,
      (2, 0, 1): 1,
      (2, 1, 0): -1}}
    >>> from sympy.combinatorics import Permutation
    >>> young_symmetrizers(3, action=Permutation) # doctest: +SKIP
    {(3,): {Permutation(2): 1,
      Permutation(1, 2): 1,
      Permutation(2)(0, 1): 1,
      Permutation(0, 1, 2): 1,
      Permutation(0, 2, 1): 1,
      Permutation(0, 2): 1},
     (2, 1): {Permutation(2): 1,
      Permutation(0, 2): -1,
      Permutation(2)(0, 1): 1,
      Permutation(0, 2, 1): -1},
     (1, 1, 1): {Permutation(2): 1,
      Permutation(1, 2): -1,
      Permutation(2)(0, 1): -1,
      Permutation(0, 1, 2): 1,
      Permutation(0, 2, 1): 1,
      Permutation(0, 2): -1}}
    """
    if action is None:
        action = lambda x: x
    if n < 0:
        raise ValueError("n must be nonnegative")
    if n == 0:
        return {(): {action(()): 1}}

    shapes = [tuple(p[::-1]) for p in reversed(list(ordered_partitions(n)))]

    perm_by_len = {0: [()]}
    sign_by_len = {0: {(): 1}}
    for k in range(1, n + 1):
        perms_k = [tuple(p) for p in permutations(range(k))]
        perm_by_len[k] = perms_k
        sign_by_len[k] = {p: sign_of(p) for p in perms_k}

    def build_perm(blocks, choice, signed=False):
        arr = list(range(n))
        sgn = 1
        for block, rel in zip(blocks, choice):
            if signed:
                sgn *= sign_by_len[len(block)][rel]
            for pos, idx in zip(block, rel):
                arr[pos] = block[idx]
        return tuple(arr), sgn

    out = {}
    for shape in shapes:
        rows = []
        start = 0
        for ln in shape:
            rows.append(tuple(range(start, start + ln)))
            start += ln

        col_blocks = [
            tuple(row[j] for row in rows if j < len(row))
            for j in range(shape[0])
        ]

        col_terms = []
        for choice in product(*(perm_by_len[len(block)] for block in col_blocks)):
            col_terms.append(build_perm(col_blocks, choice, signed=True))

        coeffs = defaultdict(int)
        for choice in product(*(perm_by_len[len(block)] for block in rows)):
            r, _ = build_perm(rows, choice, signed=False)
            for c, sgn in col_terms:
                coeffs[action(tuple(r[i] for i in c))] += sgn

        out[shape] = {k: v for k, v in coeffs.items() if v}

    return out
