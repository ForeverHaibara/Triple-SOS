from collections import defaultdict, deque
from itertools import product
from typing import Optional, Dict, List, Tuple

import sympy as sp
from sympy import Poly

from ...utils import (
    convex_hull_poly, findroot_resultant, Root, RootAlgebraic, RootRational,
    MonomialReduction, generate_expr, arraylize_sp
)


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


def _is_binary_root(root: Root) -> bool:
    return isinstance(root, RootRational) and len(set(root.root)) <= 2


def _hull_space(
        nvars: int,
        degree: int,
        convex_hull: Dict[Tuple[int, ...], bool],
        monomial: Tuple[int, ...],
        symmetry: MonomialReduction
    ) -> Optional[sp.Matrix]:
    """
    For example, s(ab(a-b)2(a+b-3c)2) does not have a^6,
    so in the positive semidefinite representation, the entry (a^3,a^3) of M is zero.
    This requires that Me_i = 0 where i is the index of a^3.

    Deprecated: This is not used anymore. Use SDPProblem.constrain_zero_diagonals() instead.
    """
    if convex_hull is None:
        return None

    half_degree = (degree - sum(monomial)) // 2
    dict_monoms = generate_expr(nvars, half_degree, symmetry=symmetry.base())[0]

    def onehot(i: int) -> List[int]:
        v = [0] * len(dict_monoms)
        v[i] = 1
        return v

    space = []
    for key, value in convex_hull.items():
        if value:
            continue

        # value == False: not in convex hull
        rest_monom = tuple(key[i] - monomial[i] for i in range(nvars))
        if any(r % 2 for r in rest_monom):
            continue
        rest_monom = tuple(r // 2 for r in rest_monom)

        space.append(onehot(dict_monoms[rest_monom]))

    return sp.Matrix(space).T


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
    >>> _compute_diff_orders((a**3*(a-b)*(a-c)+b**3*(b-c)*(b-a)+c**3*(c-a)*(c-b)).as_poly(a,b,c), Root((1,1,0)))
    [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 2)]

    >>>  _compute_diff_orders(pl('16p(2a-b-c)2-27p(a)s((a-b)2(13a-5b-17c))'), Root((1,2,0)), only_binary_roots=False)
    [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    """
    gens = poly.gens
    nvars = len(gens)
    if only_binary_roots and not _is_binary_root(root):
        return [(0,) * nvars]

    if mixed:
        def dfs(poly: Poly, order: Tuple[int, ...]) -> List[Tuple[int, ...]]:
            orders = [order]
            for i in range(nvars):
                # take one more derivative
                poly2 = poly.diff(gens[i])
                if poly2.total_degree() and root.subs(poly2, gens) == 0:
                    new_order = order[:i] + (order[i] + 1,) + order[i+1:]
                    orders.extend(dfs(poly2, new_order))
            return orders
        return dfs(poly, (0,) * nvars)

    else:
        orders = [(0,) * nvars]
        for i in range(nvars):
            poly2 = poly.diff(gens[i])
            j = 1
            while poly2.total_degree() and root.subs(poly2, gens) == 0:
                orders.append((0,) * i + (j,) + (0,) * (nvars - i - 1))
                poly2 = poly2.diff(gens[i])
                j += 1
        return orders


def _compute_nonvanishing_diff_orders(poly: Poly, root: Root, monomial: Tuple[int, ...],
                                        only_binary_roots=True) -> List[Tuple[int, ...]]:
    """
    Compute the differential orders that the root do not vanish at
    the given quadratic module (monomial).    
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


def _root_space(manifold: 'RootSubspace', root: RootAlgebraic, constraint: Poly) -> sp.Matrix:
    """
    Compute the constraint nullspace spaned by a given root.

    For normal case, it is simply the span of the root. However, things become
    nontrivial if the derivative is also zero. Imagine
    f = a * (x'Mx) and f|_r = 0, (df/da)|_r = 0 where r = (a,b,c) and x = [a^3,a^2b,...].
    When M >> 0 and a == 0, we still require Mx = 0 because
    (df/da) = (x'Mx) + a * ((dx/da)'Mx + x'M(dx/da)) = x'Mx.
    """
    d = (manifold._degree - constraint.homogeneous_order()) // 2
    nvars = len(constraint.gens)

    symmetry = manifold._symmetry
    base_symmetry = symmetry.base()
    spans = []
    if isinstance(root, RootRational) and constraint.is_monomial:
        monomial = tuple(constraint.monoms()[0])
        for r_ in symmetry.permute(root.root):
            new_r = RootRational(r_)
            orders = _compute_nonvanishing_diff_orders(manifold.poly, new_r, monomial)
            for order in orders:
                spans.append(new_r.span(d, order, symmetry=base_symmetry))

    else:
        if constraint.is_monomial and not constraint.is_zero and all(_ != 0 for _ in root.root):
            vanish = lambda _: False
        else:
            vanish = lambda r: r.subs(constraint, constraint.gens) == 0

        # this is an incomplete (but fast) implementation
        # we do not consider higher order derivatives
        # TODO: consider higher order nonvanishing derivatives for irrational roots
        span = root.span(d, symmetry=base_symmetry)
        for j, permed_root in enumerate(symmetry.permute(root.root)):
            if not vanish(Root(permed_root)):
                for i in range(span.shape[1]):
                    span2 = symmetry.permute_vec(nvars, span[:,i])[:,j]
                    spans.append(span2)        
        # for i in range(span.shape[1]):
        #     span2 = symmetry.permute_vec(nvars, span[:,i])
        #     for j, perm in zip(range(span2.shape[1]), symmetry.permute(root.root)):
        #         if not vanish(Root(perm)):
        #             spans.append(span2[:,j])

    return sp.Matrix.hstack(*spans)


def _findroot_binary(poly: Poly, symmetry: MonomialReduction = None) -> List[Root]:
    """
    Very easy implementation to find binary roots of the polynomial.
    """
    roots = set()
    for root in product([0, 1], repeat=len(poly.gens)):
        # root = RootRational(root)
        # if root.subs(poly, poly.gens) == 0:
        #     roots.append(root)
        if poly(*root) == 0:
            # roots.append(RootRational(root))
            roots.add(symmetry._standard_monom(root))
    roots = set(roots)
    all_zero = tuple([0] * len(poly.gens))
    if all_zero in roots:
        roots.remove(all_zero)
    roots = [RootRational(root) for root in roots]
    return roots


class RootSubspace():
    def __init__(self, poly: Poly, symmetry: MonomialReduction) -> None:
        self.poly = poly
        self._degree: int = poly.total_degree()
        self._nvars : int = len(poly.gens)
        self._symmetry: MonomialReduction = symmetry

        self.convex_hull = None # deprecated
        self.roots = []

        if self._nvars == 3 and poly.domain.is_Numerical:
            self.roots = findroot_resultant(poly)
        elif self._nvars <= 10:
            self.roots = _findroot_binary(poly, symmetry)

        # self.roots = [r for r in self.roots if not r.is_corner]
        self._additional_nullspace = {}

    @property
    def additional_nullspace(self) -> Dict[Tuple[int, ...], sp.Matrix]:
        return self._additional_nullspace

    def set_nullspace(self, constraint: Poly, nullspace: sp.Matrix) -> None:
        """
        Set extra nullspace for given constraint.
        """
        ns = self._additional_nullspace
        ns[constraint] = nullspace

    def append_nullspace(self, constraint: Poly, nullspace: sp.Matrix) -> None:
        """
        Append extra nullspace for given constraint.
        """
        ns = self._additional_nullspace
        if constraint in ns:
            ns[constraint] = sp.Matrix.hstack(ns[constraint], nullspace)
        else:
            ns[constraint] = nullspace

    def nullspace(self, constraint: Poly, ineq_constraints: List[Poly] = [], eq_constraints: List[Poly] = []) -> sp.Matrix:
        """
        Compute the nullspace for SDP of the polynomial.

        Parameters
        ----------
        constraint : Poly
            The specific constraint, a generator of the quadratic module. Should be homogeneous.
        ineq_constraints : List[Poly]
            List of all inequality constraints, used to test whether a root is in the feasible region.
        eq_constraints : List[Poly]
            List of all equality constraints, used to test whether a root is in the feasible region.
        """
        half_degree = (self._degree - constraint.homogeneous_order()) # // 2
        if half_degree % 2 != 0:
            raise ValueError(f"Degree of the polynomial ({self._degree}) minus the degree of the constraint {constraint} must be even.")
        half_degree //= 2

        funcs = [
            self._nullspace_hull,
            lambda *args, **kwargs: self._nullspace_from_roots(*args, **kwargs),
            self._nullspace_extra,
        ]

        nullspaces = []
        for func in funcs:
            n_ = func(constraint, ineq_constraints, eq_constraints)
            if isinstance(n_, list) and len(n_) and isinstance(n_[0], sp.MatrixBase):
                nullspaces.extend(n_)
            elif isinstance(n_, sp.MatrixBase) and n_.shape[0] * n_.shape[1] > 0:
                nullspaces.append(n_)

        nullspaces = list(filter(lambda x: x.shape[0] * x.shape[1] > 0, nullspaces))
        return sp.Matrix.hstack(*nullspaces)

    def _nullspace_hull(self, constraint: Poly, ineq_constraints: List[Poly] = [], eq_constraints: List[Poly] = []) -> sp.Matrix:
        # This is deprecated, use SDPProblem.constrain_zero_diagonals() instead.
        # We do not need to compute the convex hull (Newton polytope) of the polynomial now.
        return None
        # return _hull_space(self._nvars, self._degree, self.convex_hull, constraint, self._symmetry)

    def _nullspace_from_roots(self, constraint: Poly, ineq_constraints: List[Poly] = [], eq_constraints: List[Poly] = [], roots: List[RootAlgebraic] = None) -> List[sp.Matrix]:
        nullspaces = []
        if roots is None: roots = self.roots
        for root in roots:
            is_feasible = True
            for permed_root in self._symmetry.permute(root.root):
                permed_root = Root(permed_root)
                if any(permed_root.subs(ineq, self.poly.gens) < 0 for ineq in ineq_constraints) or\
                        any(permed_root.subs(eq, self.poly.gens) != 0 for eq in eq_constraints):
                    is_feasible = False
                    break
            # print('root =', permed_root, 'is_feasible =', is_feasible)
            if not is_feasible:
                continue

            span = _root_space(self, root, constraint)

            if span.shape[1] > 0:
                span = sp.Matrix.hstack(*span.columnspace())

            nullspaces.append(span)

        return nullspaces

    def _nullspace_extra(self, constraint: Poly, ineq_constraints: List[Poly] = [], eq_constraints: List[Poly] = []):
        ns = self._additional_nullspace
        return ns.get(constraint, None)


    def __str__(self) -> str:
        return f"RootSubspace(poly={self.poly})"

    def __repr__(self) -> str:
        return f"<RootSubspace nvars={self._nvars} degree={self._degree} roots={self.roots}>"