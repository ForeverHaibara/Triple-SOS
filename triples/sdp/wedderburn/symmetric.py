from typing import List, Set, Optional

from sympy import MutableDenseMatrix as Matrix
from sympy.combinatorics import Permutation
from sympy.utilities.iterables import ordered_partitions


def murnaghan_nakayama_character_table(n: int, cc: Optional[List[Set[Permutation]]]=None) -> Matrix:
    """
    Compute the character table of a symmetric group Sn
    using the Murnaghan-Nakayama formula.

    Parameters
    ----------
    n : int
        The size of the symmetric group.

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
