from typing import Tuple, List, Dict, Optional
from itertools import combinations

import numpy as np
import sympy as sp
from sympy import Poly, Expr, Matrix
from sympy.combinatorics import PermutationGroup
from sympy.polys.polyclasses import DMP

from ...utils import Root, MonomialManager
from ...utils.roots.num_extrema import numeric_optimize_skew_symmetry
from ...sdp.arithmetic import _permute_matrix_rows

DEFAULT_TANGENTS = {
    3: (lambda a, b, c: [
            (a**2 - b*c)**2, (b**2 - a*c)**2, (c**2 - a*b)**2,
            (a**3 - b*c**2)**2, (a**3 - b**2*c)**2, (b**3 - a*c**2)**2,
            (b**3 - a**2*c)**2, (c**3 - a*b**2)**2, (c**3 - a**2*b)**2,
        ]),
    4: (lambda a, b, c, d: [
            (a*b - c*d)**2, (a*c - b*d)**2, (a*d - b*c)**2,
        ])
}


def _get_sorted_nullspace_by_weights(mat: Matrix, weights: List[int]) -> List[Matrix]:
    inds = np.argsort(weights)
    invinds = np.argsort(inds)

    # new_mat[i] = mat[inds[i]]
    new_mat = _permute_matrix_rows(mat, invinds)

    vecs = new_mat.T.nullspace()
    vecs = [_permute_matrix_rows(v, inds) for v in vecs]
    return vecs

def _get_sorted_nullspace(monomial_manager: MonomialManager, mat: Matrix, degree: int) -> List[Matrix]:
    """
    Compute a list of sympy vectors (column matrices) that are perpendicular to the given matrix
    by heuristic sorting.    
    """
    # return mat.T.nullspace() # <- no sorting version

    weights = (np.array(monomial_manager.inv_monoms(degree), dtype=int)**2).sum(axis = 1)
    vecs = _get_sorted_nullspace_by_weights(mat, weights)
    # if len(vecs) > 1:
    #     vecs.extend(_get_sorted_nullspace_by_weights(mat, -weights))
    return vecs

def _canonicalize(p):
    """Extract trivial factors from a polynomial p,
    e.g. a*b*(a+b-c) -> a+b-c."""
    monoms = np.array(p.monoms(), dtype=int)
    if monoms.shape[0] <= 1: # monomials, just discard
        return None

    d = np.min(monoms, axis=0)
    if np.any(d):
        monoms = monoms - d.reshape((1, -1))

    if monoms.shape[0] == 2: # check whether it is a^n - b^n
        if np.max(monoms[0]) == np.max(monoms[1]) and \
            (monoms != 0).sum() == 2 and \
            sum(p.rep.coeffs()) == p.rep.dom.zero:
            return None

    if not np.any(d):
        return p
    new_monoms = [tuple(_) for _ in monoms.tolist()]
    rep = dict(zip(new_monoms, p.rep.coeffs()))
    rep = DMP.from_dict(rep, p.rep.lev, p.rep.dom)
    return Poly.new(rep, *p.gens)

def _get_ineq_constrained_tangents(ineq: Poly, ineq_expr: Expr, roots: List[Root] = [], monomial_manager: MonomialManager = None) -> Dict[Poly, Expr]:
    """
    ...
    """
    roots = [r for r in roots if r.eval(ineq) != 0]

    symbols = ineq.gens
    tangents = {}
    if ineq.is_monomial and sum(_ != 0 for _ in tuple(ineq.LM())) == 1:
        ineq, ineq_expr = Poly(1, *symbols), sp.Integer(1)
    if roots:
        for degree in range(1, 8): # TODO: avoid magic numbers
            mat = Matrix.hstack(*[r.span(degree) for r in roots])
            vecs = _get_sorted_nullspace(monomial_manager, mat, degree)
            if not vecs:
                continue

            new_polys = [monomial_manager.invarraylize(p, symbols, degree) for p in vecs]
            # new_polys = [p for p in new_polys if not p.is_monomial]
            new_polys = [_canonicalize(p) for p in new_polys]
            new_polys = set([p for p in new_polys if p is not None])
            tangents.update(dict([(ineq*p**2, ineq_expr*p.as_expr()**2) for p in new_polys]))
            if len(tangents) > 5:
                break

    if all(not r.is_nontrivial for r in roots):
        tangents[ineq] = ineq_expr

    # print('Roots of', ineq_expr, ':', roots)
    # print('Tangents of', ineq_expr, ':', [e for t, e in tangents.items()])
    return tangents


def prepare_tangents(poly: Poly, ineq_constraints: Dict[Poly, Expr] = {}, eq_constraints: Dict[Poly, Expr] = {},
        roots: List[Root] = [], additional_tangents: List[Expr] = [],
    ) -> Dict[Poly, Expr]:
    """
    Prepare tangents for LinearSOS given a list of roots (equality cases). The tangents should
    vanish in the given roots. The function returns a dictionary of {tangent: tangent_expr}.
    The "tangent" does not have a definition of canonical forms, so the behaviour of this
    function might change in the future.

    Suppose the original polynomial can be written in the SOS-form:
    
        F = sum(ineq * SOS for ineq in ineq_constraints) + sum(eq * p for eq in eq_constraints).

    Suppose `root` satisfies that F(*root) == 0, ineq(*root) >= 0 and eq(*root) == 0. Then if
    ineq_constraints[i](*root) > 0, then there must be the complementary: SOS[i](*root) == 0.
    This constrains SOS[i] to lie in a subspace where the root vanishes. The
    function generates a linear bases of this subspace for each inequality constraint.

    In practice, the `ineq_constraints` are actually generators of a quadratic module. For instance:

        (a^2+b^2-c^2 >= 0, a^2+c^2-b^2 >= 0) => ((a^2+b^2-c^2)*(a^2+c^2-b^2) >= 0).

    Thus new ineq_constraints are obtained by taking the product of possible combinations of
    the given ineq_constraints.

    TODO: Handle high-order roots.
    """
    symbols = poly.gens

    tangents = [t.as_expr() for t in additional_tangents]
    if all(not r.is_nontrivial for r in roots):
        if sp.S.One not in tangents:
            tangents.append(sp.S.One)

        if len(symbols) in DEFAULT_TANGENTS:
            tangents.extend(DEFAULT_TANGENTS[len(symbols)](*symbols))
    tangents = dict((Poly(t, symbols), t) for t in tangents)

    # remove very trivial roots
    roots = [r for r in roots if (not r.is_corner)] # and len(set(r.rep)) > 1]

    # TODO: remove it???
    # tangents.extend([_.as_expr()**2 for _ in root_tangents(poly, roots)])

    monomial_manager = MonomialManager(len(symbols))
    ineq_constraints = ineq_constraints.items() if isinstance(ineq_constraints, dict) else ineq_constraints
    for ineq, ineq_expr in ineq_constraints:
        new_tangents = _get_ineq_constrained_tangents(ineq, ineq_expr,
                roots=roots, monomial_manager=monomial_manager)
        tangents.update(new_tangents)
    # print(tangents)
    return tangents


def get_qmodule_list(poly: Poly, ineq_constraints: Dict[Poly, Expr],
        all_nonnegative: bool = False, preordering: str = 'linear') -> List[Tuple[Poly, Expr]]:
    _ACCEPTED_PREORDERINGS = ['none', 'linear']
    if not preordering in _ACCEPTED_PREORDERINGS:
        raise ValueError("Invalid preordering method, expected one of %s, received %s." % (str(_ACCEPTED_PREORDERINGS), preordering))

    degree = poly.homogeneous_order()
    poly_one = Poly(1, *poly.gens)

    monomials = []
    linear_ineqs = []
    nonlin_ineqs = [(poly_one, sp.S.One)]
    for ineq, e in ineq_constraints.items():
        if ineq.is_monomial and len(ineq.free_symbols) == 1 and ineq.total_degree() == 1 and ineq.LC() >= 0:
            monomials.append((ineq, e))
        elif ineq.is_linear:
            linear_ineqs.append((ineq, e))
        else:
            nonlin_ineqs.append((ineq, e))

    if all_nonnegative:
        # in this case we generate basis by LinaerBasisTangent rather than LinearBasisTangentEven
        # we discard all monomials
        pass
    else:
        linear_ineqs = monomials + linear_ineqs

    if preordering == 'none':
        return linear_ineqs + nonlin_ineqs

    qmodule = nonlin_ineqs.copy()
    for n in range(1, len(linear_ineqs) + 1):
        for comb in combinations(linear_ineqs, n):
            mul = poly_one
            for c in comb:
                mul = mul * c[0]
            d = mul.homogeneous_order()
            if d > degree:
                continue
            mul_expr = sp.Mul(*(c[1] for c in comb))
            for ineq, e in nonlin_ineqs:
                new_d = d + ineq.homogeneous_order()
                if new_d <= degree:
                    qmodule.append((mul * ineq, mul_expr * e))

    return qmodule


def prepare_inexact_tangents(poly: Poly, ineq_constraints: Dict[Poly, Expr] = {}, eq_constraints: Dict[Poly, Expr] = {},
    monomial_manager: MonomialManager = None, roots: List[Root] = [], all_nonnegative: bool = False) -> Dict[Poly, Expr]:
    """
    """
    nvars = len(poly.gens)
    if nvars <= 1 or len(eq_constraints) or monomial_manager.is_symmetric or any(r.is_nontrivial for r in roots):
        return dict()

    perms = [[1,0] + list(range(2, nvars))] # reflection
    if nvars >= 3:
        perms.append(list(range(1, nvars)) + [0])

    new_roots = []
    try:
        new_roots = numeric_optimize_skew_symmetry(poly, poly.gens, perms, num=5)
        new_roots = [r for r in new_roots if all(ineq(*r) >= 0 for ineq in ineq_constraints)]
        new_roots = [Root(_, domain=sp.RR) for _ in new_roots] # convert to RR
    except Exception as e: # shall we handle this?
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise e
        pass

    if len(new_roots) == 0:
        return dict()

    new_tangents = []
    monomial_base = monomial_manager.base() # no symmetry version
    for root in new_roots:
        value = poly(*root)
        mean_value = sum([poly(*root[perm]) for perm in perms])/len(perms)
        if value < 0 or value > mean_value/2:
            continue

        for degree in range(2, 5): # TODO: avoid magic numbers
            mat = monomial_manager.permute_vec(root.span(degree), degree)
            mat = sp.Matrix.hstack(mat, sp.Matrix.ones(mat.shape[0], 1)) # TODO
            vecs = _get_sorted_nullspace(monomial_base, mat, degree)
            if not vecs:
                continue

            vecs = [(vec * (1260/max(vec))).applyfunc(round)/1260 for vec in vecs] # TODO
            new_tangents.extend([
                _canonicalize(monomial_base.invarraylize(p, poly.gens, degree).retract()) for p in vecs])
            break

    new_tangents = dict([(t**2, t.as_expr()**2) for t in new_tangents if t is not None])
    # print('New tangents:\n', list(new_tangents.values()))

    return new_tangents