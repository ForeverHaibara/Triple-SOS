from itertools import product
from typing import Dict, Any

from sympy import Poly, Expr, Add, Mul

from .state_algebra import StateAlgebra, TERM, MONOM
from .basis import QmoduleBasis, IdealBasis
from .pseudo_poly import PseudoPoly

def generate_monoms_nc(nvars, degree, hom=True):
    def generate_monom_nc_hom(nvars, degree):
        return list(product(range(nvars), repeat=degree))

    monoms = generate_monom_nc_hom(nvars, degree)
    if not hom:
        for i in range(degree - 1, -1, -1):
            monoms += generate_monom_nc_hom(nvars, i)

    def compress(monom):
        m = []
        pre = -1
        cnt = 0
        for x in monom:
            if x == pre:
                cnt += 1
            else:
                m.append((pre, cnt))
                pre = x
                cnt = 1
        m.append((pre, cnt))
        return tuple(m[1:]) # remove the first element (which is -1)

    monoms = [compress(monom) for monom in monoms]
    dict_monoms = {monom: i for i, monom in enumerate(monoms)}
    inv_monoms = monoms
    return dict_monoms, inv_monoms

      

class NCPolyRing(StateAlgebra):
    """
    Basic noncommutative polynomial ring (free algebra). A monomial
    `a1**d1*a2**d2*...*an**dn` is represented by `((a1, d1), (a2, d2), ..., (an, dn))`.
    Here `a1, a2, ..., an` are hermitian variables and might include duplicated variables.
    However, it is important that `a_{i}!=a_{i+1}`. The degrees `d1, ..., dn` are nonnegative integers.
    
    To allow inverses like `a^{-1}`, one must add `a1^{-1}` to the variables
    and define the relation `a*a^{-1}=a^{-1}*a=1`.
    """
    def __init__(self, nvars, degree, is_homogeneous=True, symmetry=None):
        self.nvars = nvars
        self.degree = degree
        self.is_homogeneous = is_homogeneous
        self.symmetry = symmetry

        self._dict_monoms, self._inv_monoms = generate_monoms_nc(
            nvars, degree, hom=is_homogeneous
        )

        if symmetry is not None:
            raise NotImplementedError

    def s(self, term: TERM) -> TERM:
        return term

    def total_degree(self, monom: MONOM) -> int:
        return sum(_[1] for _ in monom) if len(monom) else 0

    def as_expr(self, poly: Poly) -> Expr:
        if isinstance(poly, PseudoPoly):
            gens = poly.gens
            return Add(*[Mul(coeff, *[gens[i]**v for i, v in monom]) for monom, coeff in poly.terms()])
        return poly.as_expr()

    def mul(self, term1: TERM, term2: TERM) -> TERM:
        (t1, v1), (t2, v2) = term1, term2
        if len(t1) and len(t2) and t1[-1][0] == t2[0][0]: # same alphabet
            t = t1[:-1] + ((t1[-1][0], t1[-1][1] + t2[0][1]),) + t2[1:]
        else:
            t = t1 + t2 # tuple concat
        return (t, v1 * v2)

    def adjoint(self, term: TERM) -> TERM:
        return (term[0][::-1], term[1])

    def infer_bases(self, poly: Poly, qmodule: Dict[Any, Poly], ideal: Dict[Any, Poly]):
        is_homogeneous = self.is_homogeneous
        degree = self.degree
        if len(ideal):
            raise NotImplementedError
            
        qmodule_bases = {}
        for key, q in qmodule.items():
            d = q.total_degree()
            if is_homogeneous and ((degree - d)%2 != 0 or (not q.is_homogeneous)):
                qmodule_bases[key] = QmoduleBasis(self, q, basis=[])
                continue
            if d > degree:
                qmodule_bases[key] = QmoduleBasis(self, q, basis=[])
                continue
            dict_basis, basis = generate_monoms_nc(
                self.nvars, (degree - d)//2, hom=is_homogeneous)
            qmodule_bases[key] = QmoduleBasis(self, q, basis=basis, dict_basis=dict_basis)
    
        ideal_bases = {}
        # for key, i in ideal.items():
        #     d = i.total_degree()
        #     if is_homogeneous and (not i.is_homogeneous):
        #         ideal_bases[key] = IdealBasis(self, i, basis=[])
        #         continue
        #     if d > degree:
        #         ideal_bases[key] = IdealBasis(self, i, basis=[])
        #         continue
        #     dict_basis, basis = generate_monoms(
        #         self.nvars, (degree - d), hom=is_homogeneous)
        #     ideal_bases[key] = IdealBasis(self, i, basis=basis, dict_basis=dict_basis)

        return qmodule_bases, ideal_bases