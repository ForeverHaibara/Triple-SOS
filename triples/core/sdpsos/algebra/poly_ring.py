from typing import List, Dict, Any

from sympy import Expr, Poly
from sympy.combinatorics import PermutationGroup, Permutation

from .state_algebra import CommutativeStateAlgebra, TERM, MONOM
from .basis import QmoduleBasis, IdealBasis
from ....utils import generate_monoms

class PolyRing(CommutativeStateAlgebra):
    def __init__(self, nvars, degree, is_homogeneous=True, symmetry=None):
        self.nvars = nvars
        self.degree = degree
        self.is_homogeneous = is_homogeneous
        self.symmetry = symmetry

        self._dict_monoms, self._inv_monoms = generate_monoms(
            nvars, degree, hom=is_homogeneous, symmetry=symmetry)

        if symmetry is None:
            setattr(self, 's', lambda term: term)
            setattr(self, 'permute', lambda monom: [monom])
        else:
            if not isinstance(symmetry, PermutationGroup):
                raise TypeError(f"Symmetry should be a PermutationGroup or None, but got {type(symmetry)}.")

            def state(term: TERM) -> TERM:
                cnt = 0
                m0 = term[0]
                std_m = m0
                for p in self.symmetry.elements:
                    m = tuple(p(m0))
                    if m in self._dict_monoms:
                        cnt += 1
                        std_m = m
                return (std_m, term[1] * cnt)
            def permute(monom: MONOM) -> List[MONOM]:
                return [tuple(p(monom)) for p in self.symmetry.elements]
            setattr(self, 's', state)
            setattr(self, 'permute', permute)

    # def s(self, term: TERM) -> TERM:
        # return term

    def permute_monom(self, monom: MONOM, perm: Permutation) -> MONOM:
        return tuple(perm(monom))

    def total_degree(self, monom: MONOM) -> int:
        return sum(monom)

    def terms(self, poly: Poly) -> List[TERM]:
        return poly.rep.terms()

    def as_expr(self, poly: Poly) -> Expr:
        return poly.as_expr()

    def mul(self, term1: TERM, term2: TERM) -> TERM:
        (t1, v1), (t2, v2) = term1, term2
        return (tuple(t1[i] + t2[i] for i in range(len(t1))), v1 * v2)

    def infer_bases(self, poly: Poly, qmodule: Dict[Any, Poly], ideal: Dict[Any, Poly]):
        is_homogeneous = self.is_homogeneous
        degree = self.degree

        qmodule_bases = {}
        for key, q in qmodule.items():
            d = q.total_degree()
            if is_homogeneous and ((degree - d)%2 != 0 or (not q.is_homogeneous)):
                qmodule_bases[key] = QmoduleBasis(self, q, basis=[])
                continue
            if d > degree:
                qmodule_bases[key] = QmoduleBasis(self, q, basis=[])
                continue
            dict_basis, basis = generate_monoms(
                self.nvars, (degree - d)//2, hom=is_homogeneous)
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
            dict_basis, basis = generate_monoms(
                self.nvars, (degree - d), hom=is_homogeneous)
            ideal_bases[key] = IdealBasis(self, i, basis=basis, dict_basis=dict_basis)

        return qmodule_bases, ideal_bases