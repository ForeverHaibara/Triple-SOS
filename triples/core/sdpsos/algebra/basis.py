from collections import defaultdict
from typing import List, Tuple, Dict, Any, Callable, Optional

from sympy import Expr, Poly, Domain
from sympy.matrices import MutableDenseMatrix as Matrix
from scipy.sparse import csr_matrix

from .state_algebra import StateAlgebra, MONOM, TERM
from ....sdp.arithmetic import rep_matrix_from_dict


class SOSBasis:
    """
    Customized SOSBasis MUST implement `localizing_matrix`.
    """
    algebra: StateAlgebra
    _dict_basis: Dict[Any, int] # basis: index
    _basis: List[Any]

    def __init__(self, algebra: StateAlgebra, basis: List[Any]):
        self.algebra = algebra
        self._basis = basis

    @property
    def is_commutative(self) -> bool:
        return self.algebra.is_commutative

    @property
    def is_cyclic_equivalent(self) -> bool:
        return self.algebra.is_cyclic_equivalent

    def __len__(self):
        return len(self._basis)

    def _localizing_mapping(self, domain: Any) -> Callable[[int, int], Dict[MONOM, Expr]]:
        raise NotImplementedError

    def localizing_matrix(self, domain: Any):
        raise NotImplementedError

    def get_equal_entries(self) -> List[List[int]]:
        return []

    def as_expr(self, coeff, vec, expr=None, adjoint_operator=None, state_operator=None) -> Expr:
        raise NotImplementedError

class QmoduleBasis(SOSBasis):
    qmodule: Poly
    def __init__(self, algebra: StateAlgebra, qmodule: Poly, basis: List[MONOM]=[], dict_basis: Dict[MONOM, int]=None):
        self.qmodule = qmodule
        self.algebra = algebra
        self._basis = basis
        self._dict_basis = dict_basis
        if dict_basis is None:
            self._dict_basis = {b: i for i, b in enumerate(basis)}

    def _nc_localizing_mapping(self, domain: Any=None) -> Callable[[int, int], List[TERM]]:
        if domain is None:
            domain = self.qmodule.domain
        algebra = self.algebra
        qmodule = algebra.terms(self.qmodule.set_domain(domain))
        one, zero = domain.one, domain.zero
        mul, state, adjoint, index = algebra.mul, algebra.s, algebra.adjoint, algebra.index
        basis = self._basis
        basis_star = [adjoint((m, one)) for m in basis]
        def _mapping(i, j):
            vec = defaultdict(lambda : zero)
            m1, m2 = basis_star[i], (basis[j], one)
            for t in qmodule:
                t2 = mul(m1, mul(t, m2))
                st2 = state(t2)
                vec[index(st2[0])] += st2[1]
            return {k: v for k, v in vec.items() if v != zero}
        return _mapping

    def _comm_localizing_mapping(self, domain: Any=None) -> Callable[[int, int], Dict[MONOM, Expr]]:
        if domain is None:
            domain = self.qmodule.domain
        algebra = self.algebra
        qmodule = algebra.terms(self.qmodule.set_domain(domain))
        one, zero = domain.one, domain.zero
        mul, state, adjoint, index = algebra.mul, algebra.s, algebra.adjoint, algebra.index
        basis = self._basis
        def _mapping(i, j):
            vec = defaultdict(lambda : zero)
            m1m2 = mul((basis[i], one), (basis[j], one))
            for t in qmodule:
                t2 = mul(t, m1m2)
                st2 = state(t2)
                vec[index(st2[0])] += st2[1]
            return {k: v for k, v in vec.items() if v != zero}
        return _mapping

    def _localizing_mapping(self, domain: Any=None) -> Callable[[int, int], Dict[MONOM, Expr]]:
        if self.is_commutative:
            return self._comm_localizing_mapping(domain)
        return self._nc_localizing_mapping(domain)

    def localizing_matrix(self, domain: Any=None) -> Matrix:
        if domain is None:
            domain = self.qmodule.domain
        mapping = self._localizing_mapping(domain)
        N, n = len(self.algebra), len(self)

        if isinstance(domain, Domain):
            rows = defaultdict(lambda : dict())
            if self.is_commutative:
                for i in range(n):
                    for t, v in mapping(i, i).items():
                        rows[t][i*(n+1)] = v
                    for j in range(i+1, n):
                        for t, v in mapping(i, j).items():
                            rows[t][i*n+j] = 2*v
            else:
                for i in range(n):
                    # for t, v in mapping(i, i).items():
                    #     rows[t][i*(n+1)] = v
                    for j in range(n):
                        for t, v in mapping(i, j).items():
                            rows[t][i*n+j] = v
            return rep_matrix_from_dict(rows, (N, n**2), domain)
        raise NotImplementedError

    def get_equal_entries(self) -> List[List[int]]:
        # This might be temporary, in the future there might be more symmetries,
        # e.g. sign symmetries, finite matrix group. And there might be a Wedderburn decomposition.
        symmetry = self.algebra.symmetry
        if symmetry is None:
            return []

        # TODO: it does not need to be fully symmetric,
        # partially symmetric is also acceptable
        if not self.algebra.is_symmetric(self.qmodule):
            return []

        pm = self.algebra.permute_monom
        dict_basis, basis = self._dict_basis, self._basis
        n = len(self)
        equal_entries = []
        visited = set()
        for i in range(n):
            m1 = basis[i]
            for j in range(i, n):
                if i*n+j in visited:
                    continue

                m2 = basis[j]
                s = set((i*n+j, j*n+i))
                for p in symmetry.elements:
                    m3, m4 = pm(m1, p), pm(m2, p)
                    i2, j2 = dict_basis.get(m3), dict_basis.get(m4)
                    if i2 is not None and j2 is not None:
                        s.add(i2*n+j2)
                visited.update(s)
                equal_entries.append(list(s))
        return equal_entries

    def as_expr(self, coeff, vec, expr=None, adjoint_operator=None, state_operator=None) -> Expr:
        raise NotImplementedError


class TwoSidedIdealBasis(SOSBasis):
    ...


class IdealBasis(SOSBasis):
    ideal: Poly
    def __init__(self, algebra: StateAlgebra, ideal: Poly, basis: List[MONOM]=[], dict_basis: Dict[MONOM, int]=None):
        self.ideal = ideal
        self.algebra = algebra
        self._basis = basis
        self._dict_basis = dict_basis
        if dict_basis is None:
            self._dict_basis = {b: i for i, b in enumerate(basis)}

    def _localizing_mapping(self, domain: Any=None) -> Callable[[int], List[TERM]]:
        # be careful that it should be a two-sided ideal for non-commutative algebra,
        # let's think more about it.
        if domain is None:
            domain = self.ideal.domain
        algebra = self.algebra
        ideal = algebra.terms(self.ideal.set_domain(domain))
        one, zero = domain.one, domain.zero
        mul, state, adjoint, index = algebra.mul, algebra.s, algebra.adjoint, algebra.index
        basis = self._basis
        def _mapping(i):
            vec = defaultdict(lambda : zero)
            m1 = (basis[i], one)
            for t in ideal:
                t2 = mul(t, m1)
                st2 = state(t2)
                vec[index(st2[0])] += st2[1]
            return {k: v for k, v in vec.items() if v != zero}
        return _mapping

    def localizing_matrix(self, domain: Any=None) -> Matrix:
        if domain is None:
            domain = self.ideal.domain
        mapping = self._localizing_mapping(domain)
        N, n = len(self.algebra), len(self)

        if isinstance(domain, Domain):
            rows = defaultdict(lambda : dict())
            for i in range(n):
                for t, v in mapping(i).items():
                    rows[t][i] = v
            return rep_matrix_from_dict(rows, (N, n), domain)
        raise NotImplementedError

    
    def get_equal_entries(self) -> List[List[int]]:
        # This might be temporary, in the future there might be more symmetries,
        # e.g. sign symmetries, finite matrix group. And there might be a Wedderburn decomposition.
        symmetry = self.algebra.symmetry
        if symmetry is None:
            return []

        # TODO: it does not need to be fully symmetric,
        # partially symmetric is also acceptable
        if not self.algebra.is_symmetric(self.ideal):
            return []

        pm = self.algebra.permute_monom
        dict_basis, basis = self._dict_basis, self._basis
        n = len(self)
        equal_entries = []
        visited = set()
        for i in range(n):
            m1 = basis[i]
            if i in visited:
                continue

            s = {i}
            for p in symmetry.elements:
                m3 = pm(m1, p)
                i2 = dict_basis.get(m3)
                if i2 is not None:
                    s.add(i2)
            visited.update(s)
            equal_entries.append(list(s))
        return equal_entries