"""
This module contains heuristic, experimental, imcomplete facial reduction algorithm for SDP algorithm
via computing the equality cases of the original SOS problem.
"""
from collections import defaultdict, deque
from itertools import product
from time import time
from typing import Dict, List, Tuple, Union, Optional, Any

from sympy import Poly, Expr, MatrixBase, prod
from sympy.matrices import MutableDenseMatrix as Matrix
from sympy.combinatorics import PermutationGroup

from .algebra import SOSBasis
from ...sdp import SDPProblem
from ...sdp.arithmetic import solve_columnspace, rep_matrix_from_list, rep_matrix_to_numpy
from ...utils import optimize_poly, Root, MonomialManager
from ...utils.roots.roots import _algebraic_extension, _derv


def _permute_root(perm_group: PermutationGroup, root: Root) -> List[Root]:
    """Permute a Root object efficiently given a permutation group."""
    perms = list(perm_group.elements)
    roots = [None] * len(perms)
    root0, rep, domain = root.root, root.rep, root.domain
    for i, perm in enumerate(perms):
        roots[i] = Root(perm(root0), domain=domain, rep=perm(rep))
    return roots

def _is_binary_root(root: Root) -> bool:
    # return isinstance(root, RootRational) and len(set(root.root)) <= 2
    return root.is_Rational and len(set(root.root)) <= 2

def _root_span(root: Root, basis: Any, degree: int = 0, diff: Tuple[int,...] = None) -> Matrix:
    if isinstance(basis, MonomialManager):
        return root.span(degree, diff, symmetry=basis)
    elif isinstance(basis, SOSBasis):
        monoms = basis._basis # should be a list of tuple

        vec = [None] * len(monoms)
        _single_power = root._single_power_monomial

        zero = root.domain.zero
        if diff is None:
            for ind, monom in enumerate(monoms):
                vec[ind] = _single_power(monom)
        else:
            for ind, monom in enumerate(monoms):
                if any(order_m < order_diff for order_m, order_diff in zip(monom, diff)):
                    vec[ind] = zero
                else:
                    dervs = [_derv(order_m, order_diff) for order_m, order_diff in zip(monom, diff)]
                    powers = [order_m - order_diff for order_m, order_diff in zip(monom, diff)]
                    vec[ind] = int(prod(dervs)) * _single_power(powers)

        if not root.is_Rational:
            vec = _algebraic_extension(vec, root.domain)
        else:
            vec = rep_matrix_from_list(vec, len(vec), domain=root.domain)
        return vec

 
    raise TypeError(f"Unknown basis type {type(basis)}")
    

class _bilinear():
    """
    Represent Sum(m * u' * M * v) where M >> 0 and m is a monomial,
    represented by a tuple of orders. u and v are also tuples,
    recording the differential orders of the variables.

    For example, for 3-var poly, a * v' * M * v is

        _bilinear({(1, 0, 0): {(0, 0, 0), (0, 0, 0)}})

    Taking the differential with respect to the first variable, we get
    the new poly v' * M * v + 2a * v' * M * (dv/da). It has two different terms
    and is represented by

        _bilinear({(0, 0, 0): {(0, 0, 0), (0, 0, 0)}, (1, 0, 0): {(0, 0, 0), (1, 0, 0)}})

    Identical terms are merged, coefficients are not stored.
    """
    __slots__ = ('terms',)
    @classmethod
    def _reg(cls, u, v):
        return (u, v) if u < v else (v, u)

    def __init__(self, terms: Dict[Tuple[int, ...], set]):
        self.terms = terms
        for m in terms:
            terms[m] = set(map(lambda uv: self._reg(*uv), terms[m]))

    def diff(self, i: int) -> '_bilinear':
        def increase_tuple(t, i, v):
            return t[:i] + (t[i]+v,) + t[i+1:]
        _reg = self._reg
        new_terms = defaultdict(set)
        for m, uvs in self.terms.items():
            if m[i]:
                m2 = increase_tuple(m, i, -1)
                new_terms[m2] |= uvs
            for u, v in uvs:
                u2 = increase_tuple(u, i, 1)
                v2 = increase_tuple(v, i, 1)
                new_terms[m].add(_reg(u2, v))
                new_terms[m].add(_reg(u, v2))
        return _bilinear(new_terms)

    def diff_monoimal(self, m: Tuple[int, ...]) -> '_bilinear':
        b = self
        for i in range(len(m)):
            for times in range(m[i]):
                b = b.diff(i)
        return b

    def __str__(self) -> str:
        return f'_bilinear({str(self.terms)})'
    def __repr__(self) -> str:
        return self.__str__()


def _compute_diff_orders(poly: Poly, root: Root, mixed=False, only_binary_roots=True) -> List[Tuple[int, ...]]:
    """
    Compute tuples (a1, ..., an) such that d^a1/dx1^a1 ... d^an/dxn^an f = 0 at the root.
    The root object is assumed to be indeed a root of the polynomial, which will not be checked.

    Parameters
    ----------
    poly : Poly
        The polynomial to compute the differential orders for.
    root : Root
        The root to compute the differential orders for.
    mixed : bool
        If False, only differentiate with respect to one variable at a time.
        If True, differentiate with respect to all variables at the same time. However,
        it is not correct to use mixed=True.
    only_binary_roots : bool
        If True, only "binary" roots are computed. This is incomplete but
        is fast and sufficient for most cases.

    Examples
    --------
    >>> from sympy.abc import a, b, c
    >>> from triples.utils import pl
    >>> _compute_diff_orders((a**3*(a-b)*(a-c)+b**3*(b-c)*(b-a)+c**3*(c-a)*(c-b)).as_poly(a,b,c), Root((1,1,0)))
    [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 2)]

    >>> _compute_diff_orders(pl('16p(2a-b-c)2-27p(a)s((a-b)2(13a-5b-17c))'), Root((1,2,0)), only_binary_roots=False)
    [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    """
    gens = poly.gens
    nvars = len(gens)
    if (only_binary_roots and not _is_binary_root(root)):
        return [(0,) * nvars]

    if mixed:
        def dfs(poly: Poly, order: Tuple[int, ...]) -> List[Tuple[int, ...]]:
            orders = [order]
            for i in range(nvars):
                # take one more derivative
                poly2 = poly.diff(gens[i])
                if poly2.total_degree() and root.eval(poly2) == 0:
                    new_order = order[:i] + (order[i] + 1,) + order[i+1:]
                    orders.extend(dfs(poly2, new_order))
            return orders
        return dfs(poly, (0,) * nvars)

    else:
        orders = [(0,) * nvars]
        for i in range(nvars):
            poly2 = poly.diff(gens[i])
            j = 1
            while poly2.total_degree() and root.eval(poly2) == 0:
                orders.append((0,) * i + (j,) + (0,) * (nvars - i - 1))
                poly2 = poly2.diff(gens[i])
                j += 1
        return orders


def _compute_nonvanishing_diff_orders(poly: Poly, root: Root, monomial: Tuple[int, ...],
                                        only_binary_roots=True) -> List[Tuple[int, ...]]:
    """
    Compute the differential orders that the root do not vanish at
    the given quadratic module (monomial). TODO: extend to non-monomial cases
    """
    orders = _compute_diff_orders(poly, root, only_binary_roots=only_binary_roots)
    # if len(orders) <= 0 or (len(orders) == 1 and not any(orders[0])):
    #     return orders
    nvars = len(poly.gens)
    b0 = _bilinear({monomial: [((0,)*nvars, (0,)*nvars)]})

    def _make_vanish_checker(root):
        nonzeros = [1 if _ != 0 else 0 for _ in root.root]
        if all(nonzeros):
            nonvanish = lambda _nonzeros: True
        else:
            def nonvanish(monomial):
                return all(i != 0 for i, j in zip(nonzeros, monomial) if j != 0)
        return nonvanish
    nonvanish = _make_vanish_checker(root)

    sets = []
    for m in orders:
        b = b0.diff_monoimal(m)
        for key in filter(nonvanish, b.terms):
            sets.append(b.terms[key])

    zeros = deque([(-1,) * nvars]) # starting sentinel
    handled_zeros = set()
    while len(zeros):
        zero = zeros.popleft()
        if zero in handled_zeros:
            continue
        for set_ in sets:
            for i, j in list(set_):
                if i == zero or j == zero:
                    # one of the vectors of the bilinear form is proven to vanish
                    # and we no longer needs to consider this term
                    set_.remove((i, j))
            if len(set_) == 1:
                p = set_.pop()
                if p[0] == p[1]:
                    # in this case the bilinear form is PSD and it vanishes at this derivative
                    # this imposes a nullspace constraint on the PSD matrix
                    zeros.append(p[0])
        if zero[0] != -1:
            handled_zeros.add(zero)
    return handled_zeros


def _root_space(root: Root, poly: Poly, qmodule: Poly, codegree: int, basis: MonomialManager) -> Matrix:
    """
    Compute the constrained nullspace spanned by a given root. Consider:

        `poly = CyclicSum(qmodule * SOS) + SOS_OTHERS`

    where the poly vanishes at the given root. If the qmodule does not vanish,
    then the associated SOS term must vanish.

    For normal case, it is simply the span of the root. However, things become
    nontrivial if the derivative is also zero. Imagine
    f = a * (x'Mx) and f|_r = 0, (df/da)|_r = 0 where r = (a,b,c) and x = [a^3,a^2b,...].
    When M >> 0 and a == 0, we still require Mx = 0 because
    (df/da) = (x'Mx) + a * ((dx/da)'Mx + x'M(dx/da)) = x'Mx.
    """
    spans = []

    if qmodule.is_monomial and not qmodule.is_zero and all(_ != 0 for _ in root.root):
        vanish = lambda _: False
    else:
        vanish = lambda r: r.eval(qmodule) == 0

    if qmodule.is_zero:
        pass
    elif root.is_Rational and qmodule.is_monomial:
        # also compute high order dervs
        monomial = tuple(qmodule.monoms()[0])
        orders = _compute_nonvanishing_diff_orders(poly, root, monomial)
        for order in orders:
            # spans.append(root.span(codegree, order, symmetry=basis))
            spans.append(_root_span(root, basis, codegree, order))
    else:
        # this is an incomplete (but fast) implementation
        # we do not consider higher order derivatives
        # TODO: consider higher order nonvanishing derivatives for irrational roots
        if not vanish(root):
            # spans.append(root.span(codegree, symmetry=basis))
            spans.append(_root_span(root, basis, codegree))

        # for j, permed_root in enumerate(_permute_root(perm_group, root)):
        #     if not vanish(permed_root):
        #         for i in range(span.shape[1]):
        #             span2 = symmetry.permute_vec(span[:,i], codegree)[:,j]
        #             spans.append(span2)

    if len(spans):
        return Matrix.hstack(*spans)
    n = len(basis.inv_monoms(codegree)) if isinstance(basis, MonomialManager) else len(basis)
    return Matrix.zeros(n, 0)


def _findroot_binary(poly: Poly, symmetry: PermutationGroup = None) -> List[Root]:
    """
    Very easy implementation to find binary roots of the polynomial.
    """
    symmetry = MonomialManager(len(poly.gens), perm_group=symmetry)
    roots = set()
    for root in product([0, 1], repeat=len(poly.gens)):
        # root = RootRational(root)
        # if root.subs(poly, poly.gens) == 0:
        #     roots.append(root)
        if poly(*root) == 0:
            # roots.append(RootRational(root))
            roots.add(symmetry.standard_monom(root))
    roots = set(roots)
    all_zero = tuple([0] * len(poly.gens))
    if all_zero in roots:
        roots.remove(all_zero)
    roots = [Root(root) for root in roots]
    return roots


def get_nullspace(poly: Poly, ineq_constraints: Dict[Any, Poly], eq_constraints: Dict[Any, Poly],
        ineq_bases: Dict[Any, SOSBasis], eq_bases: Dict[Any, SOSBasis],
        degree: Optional[int]=None, roots: List[Root] = [], perm_group: Optional[PermutationGroup] = None) -> Dict[Any, Matrix]:
    """
    In the current, all roots must satisfy poly(roots) == 0 and ineq_constraints(roots) >= 0,
    and eq_constraints(roots) == 0, and this property will not be checked.
    """
    if degree is None:
        degree = poly.total_degree()
    # for root in roots:
    #     is_feasible = True
    #     for permed_root in _permute_root(perm_group, root):
    #         if any(permed_root.eval(ineq) < 0 for ineq in ineq_constraints) or\
    #                 any(permed_root.eval(eq) != 0 for eq in eq_constraints):
    #             is_feasible = False
    #             break
    #     # print('root =', permed_root, 'is_feasible =', is_feasible)
    #     if not is_feasible:
    #         continue

    nullspaces = {}

    for key, ineq in ineq_constraints.items():
        spans = []
        for root in roots:
            span = _root_space(root, poly, qmodule=ineq,
                        codegree=(degree-ineq.total_degree())//2, basis=ineq_bases[key])
            if span.shape[1] > 0:
                spans.append(solve_columnspace(span))

        if len(spans):
            nullspaces[key] = Matrix.hstack(*spans)

    return nullspaces



def constrain_root_nullspace(sdp: SDPProblem, poly: Poly, ineq_constraints: Dict, eq_constraints: Dict,
        ineq_bases: Dict[Any, Any], eq_bases: Dict[Any, Any], degree: int,
        roots: Optional[List[Root]]=None, symmetry: Optional[PermutationGroup]=None, verbose: bool = False
    ) -> Tuple[SDPProblem, List[Root]]:
    """
    Internal helper function to constrain the nullspace of the SDP problem. It will be called
    in `SOSProblem.construct`.
    """
    def _symmetry_expand(polys):
        """Add in the permuted polynomoials given a symmetry group."""
        if symmetry is None or len(polys) == 0 or symmetry.is_trivial:
            return list(polys)
        rep_set = set()
        polylize, gens = polys[0].__class__.new, polys[0].gens
        for poly in polys:
            rep = poly.rep
            rep_set.add(rep)
            for perm in symmetry.elements:
                reorder = poly.reorder(*perm(gens)).rep
                if rep == reorder:
                    continue
                rep_set.add(reorder)
        return [polylize(rep, *gens) for rep in rep_set] 

    time0 = time()
    if roots is None:
        # find roots automatically
        all_polys = list(ineq_constraints.values()) + list(eq_constraints.values()) + [poly]
        if all(p.domain.is_ZZ or p.domain.is_QQ for p in all_polys):
            ineqs = _symmetry_expand(list(ineq_constraints.values()))
            eqs   = _symmetry_expand(list(eq_constraints.values()))
            roots = optimize_poly(poly, ineqs, eqs + [poly], return_type='root')
        else:
            # TODO: clean this
            roots = _findroot_binary(poly)# symmetry=self._symmetry)
        if verbose:
            print(f"Time for finding roots num = {len(roots):<6d}     : {time() - time0:.6f} seconds.")
            time0 = time()
    else:
        roots = [Root(_) if not isinstance(_, Root) else _ for _ in roots]


    time0 = time()
    nullspaces = get_nullspace(poly, ineq_constraints, eq_constraints, ineq_bases, eq_bases,
                        degree=degree, roots=roots)
    if verbose:
        print(f"Time for computing nullspace            : {time() - time0:.6f} seconds.")
        time0 = time()

    new_sdp = sdp.constrain_nullspace(nullspaces, to_child=True)

    if verbose:
        print(f"Time for constraining nullspace         : {time() - time0:.6f} seconds. Dof = {sdp.get_last_child().dof}")
        time0 = time()

    return new_sdp, roots
