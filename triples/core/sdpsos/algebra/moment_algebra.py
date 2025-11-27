from itertools import combinations
from typing import List, Dict, Any, Optional, Callable

from sympy import Expr, Poly, Symbol, Add, Mul

from .state_algebra import CommutativeStateAlgebra, TERM, MONOM
from .basis import QmoduleBasis, IdealBasis

from ....utils.monomials import generate_partitions

def generate_monoms_mm(nvars, degree, num_moments, hom=True, standard_monom=None):
    monoms = generate_partitions(nvars, degree, equal=False, descending=True)

    reduced = []
    visited = set()
    for i in range(len(monoms) - 1):
        # 1, len(monoms) because the constant term must be removed
        m = standard_monom(monoms[i])
        if m not in visited:
            visited.add(m)
            reduced.append(i)

    d_list = list(map(sum, monoms))

    rest = {i: [] for i in range(degree + 1)}
    for m in monoms:
        rest[sum(m)].append(m)
    if not hom:
        # containing monomials with total degree less than `degree`
        for i in range(1, degree + 1):
            rest[i].extend(rest[i-1])

    new_monoms = []
    for k in range(min(num_moments, len(reduced)) + 1):
        # traverse all combinations with exactly k nonzeros

        for comb in combinations(reduced, k):
            d_comb = [d_list[i] for i in comb]
            d_rem = degree - sum(d_comb)
            if d_rem < 0:
                continue
            for extra in generate_partitions(d_comb, d_rem):
                k_rem = (num_moments - k) - sum(extra)
                if k_rem < 0:
                    continue
                d_rem2 = d_rem - sum([ei * di for ei, di in zip(extra, d_comb)])
                tail = tuple([(monoms[ci], di + 1) for ci, di in zip(comb, extra)][::-1])

                for r in rest[d_rem2]:
                    new_monoms.append((r,) + tail)

                for num_zero in range(1, k_rem+1):
                    for r in rest[d_rem2]:
                        new_monoms.append((r,) + (((0,)*nvars, num_zero),) + tail)

    new_monoms = set(new_monoms)
    dict_monoms = {m: i for i, m in enumerate(new_monoms)}
    return dict_monoms, list(new_monoms)


class MixedMomentAlgebra(CommutativeStateAlgebra):
    """
    Monomials are in the form of:

    ```
        (m0, (m1, d1), (m2, d2), ...)
    ```

    where `m0, m1, m2, ...` are tuples, `d1, d2, ...` are strictly positive integers.
    Moreover, `m1 < m2 < ...` are sorted ascendingly.
    It represents:

    ```
        x**m0 * (Σ x**m1)**d1 * (Σ x**m2)**d2 * ...
    ```

    The moment operator sends a monomial to the pure moment:

    ```
        1 * (Σ x**m0) * (Σ x**m1)**d1 * (Σ x**m2)**d2 * ...
    ```

    Note that the `Σ x**m0` should be inserted to the correct position in implementation.
    """
    def __init__(self, nvars, degree, num_moments, is_homogeneous=True, standard_monom=None):
        self.nvars = nvars
        self.degree = degree
        self.num_moments = num_moments
        self.is_homogeneous = is_homogeneous
        if standard_monom is None:
            standard_monom = lambda x: x
        self.standard_monom = standard_monom

        self._dict_monoms, self._inv_monoms = generate_monoms_mm(
            nvars, degree, num_moments, hom=is_homogeneous, standard_monom=standard_monom)

    def gen_monom(self, i: Optional[int]) -> MONOM:
        if i is None:
            return ((0,) * self.nvars,)
        return ((0,) * i + (1,) + (0,) * (self.nvars - i - 1),)

    def total_degree(self, monom: MONOM) -> int:
        return sum(monom[0]) + sum([sum(m) * mul for m, mul in monom[1:]])

    def s(self, term: TERM) -> TERM:
        m, c = term
        m0 = self.standard_monom(m[0])
        for i in range(1, len(m)):
            if m[i][0] == m0:
                return ((0,) * len(m0),) + m[1:i] + ((m0, m[i][1] + 1),) + m[i+1:], c
            if m[i][0] > m0:
                return ((0,) * len(m0),) + m[1:i] + ((m0, 1),) + m[i:], c
        return ((0,) * len(m0),) + m[1:] + ((m0, 1),), c

    def as_expr(self, poly: Poly, state_operator: Optional[Callable[[Expr], Expr]]=None) -> Expr:
        """Convert a polynomial in this algebra to sympy Expr."""
        if state_operator is None:
            state_operator = lambda x: x
        gens = poly.gens
        me = lambda m: Mul(*[gen**i for gen, i in zip(gens, m)])
        med = lambda md: state_operator(me(md[0]))**md[1]
        return Add(*[Mul(c, me(m[0]), *[med(_) for _ in m[1:]]) for m, c in poly.terms()])

    def mul(self, term1: TERM, term2: TERM) -> TERM:
        m1, c1 = term1
        m2, c2 = term2
        m01, m02 = m1[0], m2[0]
        m0 = tuple(i + j for i, j in zip(m01, m02))
        concat = []

        l1, l2 = len(m1), len(m2)
        p1 = 1
        p2 = 1
        while p1 < l1 and p2 < l2:
            if m1[p1][0] < m2[p2][0]:
                concat.append(m1[p1])
                p1 += 1
            elif m1[p1][0] > m2[p2][0]:
                concat.append(m2[p2])
                p2 += 1
            else: # m1[p1][0] == m2[p2][0]
                concat.append((m1[p1][0], m1[p1][1] + m2[p2][1]))
                p1 += 1
                p2 += 1
        concat = tuple(concat)
        if p1 < l1:
            concat = concat + m1[p1:]
        if p2 < l2:
            concat = concat + m2[p2:]
        return (m0,) + concat, c1 * c2

    def infer_bases(self, poly: Poly, qmodule: Dict[Any, Poly], ideal: Dict[Any, Poly]):
        is_homogeneous = self.is_homogeneous
        degree = self.degree

        _nm = lambda x: sum([mul for m, mul in x[1:]])
        nm = lambda x: max(map(_nm, x.monoms()), default=0)

        qmodule_bases = {}
        for key, q in qmodule.items():
            d = q.total_degree()
            if is_homogeneous and ((degree - d)%2 != 0 or (not q.is_homogeneous)):
                qmodule_bases[key] = QmoduleBasis(self, q, basis=[])
                continue
            if d > degree:
                qmodule_bases[key] = QmoduleBasis(self, q, basis=[])
                continue
            m = nm(q)
            if self.num_moments - 1 - m < 0:
                continue
            dict_basis, basis = generate_monoms_mm(
                self.nvars, (degree - d)//2, num_moments=(self.num_moments - 1 - m)//2, hom=is_homogeneous,
                standard_monom=self.standard_monom)
            qmodule_bases[key] = QmoduleBasis(self, q, basis=basis, dict_basis=dict_basis)

        ideal_bases = {}
        for key, i in ideal.items():
            d = i.total_degree()
            if is_homogeneous and (not i.is_homogeneous):
                ideal_bases[key] = IdealBasis(self, i, basis=[])
                continue
            if d > degree:
                ideal_bases[key] = IdealBasis(self, i, basis=[])
                continue
            m = nm(i)
            if self.num_moments - 1 - m < 0:
                continue
            dict_basis, basis = generate_monoms_mm(
                self.nvars, (degree - d), num_moments=self.num_moments - 1 - m, hom=is_homogeneous,
                standard_monom=self.standard_monom)
            ideal_bases[key] = IdealBasis(self, i, basis=basis, dict_basis=dict_basis)

        return qmodule_bases, ideal_bases
