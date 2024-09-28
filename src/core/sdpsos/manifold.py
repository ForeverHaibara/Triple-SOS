from collections import defaultdict, deque
from itertools import product
from typing import Optional, Dict, List, Tuple

import sympy as sp
from sympy.combinatorics import PermutationGroup

from ...utils import (
    convex_hull_poly, findroot_resultant, Root, RootAlgebraic, RootRational,
    MonomialReduction, generate_expr
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


def _compute_diff_orders(poly: sp.Poly, root: Root, mixed=False, only_binary_roots=True) -> List[Tuple[int, ...]]:
    """
    Compute tuples (a1, ..., an) such that d^a1/dx1^a1 ... d^an/dxn^an f = 0 at the root.

    Parameters
    ----------
    poly : sp.Poly
        The polynomial to compute the differential orders for.
    root : Root
        The root to compute the differential orders for.
    mixed : bool
        If False, only differentiate with respect to one variable at a time.
        If True, differentiate with respect to all variables at the same time. However,
        it is not correct to use mixed=True.
    only_binary_roots : bool
        If True, only "binary" roots are computed. This is incomplete but
        is sufficient for most cases.
    """
    gens = poly.gens
    nvars = len(gens)
    if only_binary_roots and not _is_binary_root(root):
        return [(0,) * nvars]

    if mixed:
        def dfs(poly: sp.Poly, order: Tuple[int, ...]) -> List[Tuple[int, ...]]:
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


def _compute_nonvanishing_diff_orders(poly: sp.Poly, root: Root, monomial: Tuple[int, ...],
                                        only_binary_roots=True) -> List[Tuple[int, ...]]:
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
                    set_.remove((i, j))
            if len(set_) == 1:
                p = set_.pop()
                if p[0] == p[1]:
                    zeros.append(p[0])
        if zero[0] != -1:
            handled_zeros.add(zero)
    return handled_zeros


def _root_space(manifold: 'RootSubspace', root: RootAlgebraic, monomial: Tuple[int, ...]) -> sp.Matrix:
    """
    Compute the constraint nullspace spaned by a given root.

    For normal case, it is simply the span of the root. However, things become
    nontrivial if the derivative is also zero. Imagine
    f = a * (x'Mx) and f|_r = 0, (df/da)|_r = 0 where r = (a,b,c) and x = [a^3,a^2b,...].
    When M >> 0 and a == 0, we still require Mx = 0 because
    (df/da) = (x'Mx) + a * ((dx/da)'Mx + x'M(dx/da)) = x'Mx.
    """
    d = (manifold._degree - sum(monomial)) // 2
    nvars = len(monomial)
    nonzeros = [1 if _ != 0 else 0 for _ in root.root]

    if all(nonzeros):
        vanish = lambda _nonzeros: False
    else:
        def vanish(_nonzeros):
            return all(i > 0 for i, j in zip(monomial, _nonzeros) if j == 0)

    symmetry = manifold._symmetry
    spans = []

    # this is an incomplete (but fast) implementation
    if isinstance(root, RootRational):
        base_symmetry = symmetry.base()
        for r_ in symmetry.permute(root.root):
            new_r = RootRational(r_)
            orders = _compute_nonvanishing_diff_orders(manifold.poly, new_r, monomial)
            for order in orders:
                spans.append(new_r.span(d, order, symmetry=base_symmetry))

    else:
        span = root.span(d, symmetry=symmetry.base())
        for i in range(span.shape[1]):
            span2 = symmetry.permute_vec(nvars, span[:,i])
            for j, perm in zip(range(span2.shape[1]), symmetry.permute(nonzeros)):
                if not vanish(perm):
                    spans.append(span2[:,j])

    return sp.Matrix.hstack(*spans)


def _findroot_binary(poly: sp.Poly, symmetry: MonomialReduction = None) -> List[Root]:
    """
    Find binary roots of the polynomial.
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
    def __init__(self, poly: sp.Poly, symmetry: MonomialReduction) -> None:
        self.poly = poly
        self._degree: int = poly.total_degree()
        self._nvars : int = len(poly.gens)
        self._symmetry: MonomialReduction = symmetry

        self.convex_hull = None
        self.roots = []

        if self._nvars == 3: # and self._symmetry.is_cyc:
            # if self._symmetry.is_cyc:
            #     self.convex_hull = convex_hull_poly(poly)[0]
            self.roots = findroot_resultant(poly)
        elif self._nvars <= 10:
            self.roots = _findroot_binary(poly, symmetry)

        # self.roots = [r for r in self.roots if not r.is_corner]
        self._additional_nullspace = {}

    @property
    def additional_nullspace(self) -> Dict[Tuple[int, ...], sp.Matrix]:
        return self._additional_nullspace

    def set_nullspace(self, monomial: Tuple[int, ...], nullspace: sp.Matrix) -> None:
        """
        Set extra nullspace for given monomial.
        """
        self._additional_nullspace[monomial] = nullspace

    def append_nullspace(self, monomial: Tuple[int, ...], nullspace: sp.Matrix) -> None:
        """
        Append extra nullspace for given monomial.
        """
        ns = self._additional_nullspace
        if monomial in ns:
            ns[monomial] = sp.Matrix.hstack(ns[monomial], nullspace)
        else:
            ns[monomial] = nullspace

    def nullspace(self, monomial, real: bool = False) -> sp.Matrix:
        """
        Compute the nullspace of the polynomial.

        Parameters
        ----------
        monomial : Tuple[int, ...]
            The monomial to compute the nullspace for.

        real : bool
            Whether prove the inequality on R^n or R+^n. If R+^n, it ignores nonpositive roots.
        """
        half_degree = (self._degree - sum(monomial))
        if half_degree % 2 != 0:
            raise ValueError(f"Degree of the polynomial ({self._degree}) minus the degree of the monomial {monomial} must be even.")

        funcs = [
            self._nullspace_hull,
            lambda *args, **kwargs: self._nullspace_roots(*args, **kwargs, real = real),
            self._nullspace_extra,
        ]

        nullspaces = []
        for func in funcs:
            n_ = func(monomial)
            if isinstance(n_, list) and len(n_) and isinstance(n_[0], sp.MatrixBase):
                nullspaces.extend(n_)
            elif isinstance(n_, sp.MatrixBase) and n_.shape[0] * n_.shape[1] > 0:
                nullspaces.append(n_)

        nullspaces = list(filter(lambda x: x.shape[0] * x.shape[1] > 0, nullspaces))
        return sp.Matrix.hstack(*nullspaces)


    def _nullspace_hull(self, monomial):
        symmetry = self._symmetry
        return _hull_space(self._nvars, self._degree, self.convex_hull, monomial, symmetry)

    def _nullspace_roots(self, monomial, real: bool = False):
        nullspaces = []
        for root in self.roots:
            if not real and any(_ < 0 for _ in root.root):
                continue

            span = _root_space(self, root, monomial)

            if span.shape[1] > 0:
                span = sp.Matrix.hstack(*span.columnspace())

            nullspaces.append(span)

        return nullspaces

    def _nullspace_extra(self, monomial):
        return self._additional_nullspace.get(monomial, None)


    def __str__(self) -> str:
        return f"RootSubspace(poly={self.poly})"

    def __repr__(self) -> str:
        return f"<RootSubspace nvars={self._nvars} degree={self._degree} roots={self.roots}>"