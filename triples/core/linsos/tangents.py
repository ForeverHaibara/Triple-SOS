from typing import Tuple, List, Dict, Optional
from itertools import combinations

import numpy as np
from sympy import Poly, Expr, Integer, Mul, RR
from sympy import MutableDenseMatrix as Matrix
from sympy.combinatorics import PermutationGroup
from sympy.polys.polyclasses import DMP

from ..problem import InequalityProblem
from ...utils import Root, MonomialManager
from ...utils.roots.num_extrema import numeric_optimize_skew_symmetry
from ...sdp.arithmetic import permute_matrix_rows

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


def _get_sorted_nullspace_by_weights(mat: Matrix, weights: Optional[List[int]]=None) -> List[Matrix]:
    """Compute a left nullspace of a matrix by first ordering the rows by weights."""
    if weights is not None:
        inds = np.argsort(weights)
        invinds = np.argsort(inds)

        # new_mat[i] = mat[inds[i]]
        mat = permute_matrix_rows(mat, inds)

    vecs = mat.T.nullspace()
    if weights is not None:
        vecs = [permute_matrix_rows(v, invinds) for v in vecs]
    return vecs

def _get_sorted_nullspace(monomial_manager: MonomialManager, mat: Matrix, degree: int) -> List[Matrix]:
    """
    Compute a list of sympy vectors (column matrices) that are perpendicular to the given matrix
    by heuristic sorting. Recall that the computation of the left nullspace of a matrix depends on
    the order of rows, the function uses heuristic orderings of the rows.

    Given a monomial `(x1,...,xn)`, consider the monomial `p = a1**x1*...*an**xn`. We study the behaviour
    of `p` around `(1,...,1)`, given by:

        sum(p(1,...,1+t,...,1-t,....1) - 1 for i, j in combinations(range(n), 2))
        = sum((1 + t)**xi * (1 - t)**xj - 1 for i, j in combinations(range(n), 2))
        = -sum(xi*xj for i, j in combinations(range(n), 2))*t^2 + o(t**2).

    For monomials with lower `-sum(xi*xj for i, j in combinations(range(n), 2))`, i.e.,
    lower values of `sum(xi**2 for i in range(n))`, they tend to be lower around `(1,...,1)`
    as they have lower `t^2` terms and produce tighter inequalities. Therefore, we sort the
    terms with respect to this ordering.
    """
    # return mat.T.nullspace() # <- no sorting version

    weights = (np.array(monomial_manager.inv_monoms(degree), dtype=int)**2).sum(axis = 1)
    vecs = _get_sorted_nullspace_by_weights(mat, weights)
    # if len(vecs):
    #     vecs.extend(_get_sorted_nullspace_by_weights(mat))
    #     vecs.extend(_get_sorted_nullspace_by_weights(mat, -weights))
    return vecs

def _canonicalize(p):
    """
    Extract trivial factors from a polynomial p. E.g. a*b*(a+b-c) -> a+b-c.
    It helps remove redundant or trivial tangents for forming the LinearSOS bases.
    """
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

def _get_ineq_constrained_tangents(
    ineq: Poly,
    ineq_expr: Expr,
    roots: List[Root] = [],
    monomial_manager: MonomialManager = None,
    num_tangents: int = 5,
    max_degree: int = 8
) -> Dict[Poly, Expr]:
    """
    Compute a dictionary of items `(ineq*poly**2, ineq_expr*expr**2)` such that
    `ineq * poly` vanishes at all given roots. As there are infinitely many polynomials
    that satisfy this condition, the function heuristically picks a few.

    Parameters
    ----------
    ineq : Poly
        The inequality constraint.
    ineq_expr : Expr
        The sympy expression of the inequality constraint.
    roots : List[Root]
        The list of roots at which `ineq*poly**2` is forced to vanish.
    monomial_manager : MonomialManager
        The monomial manager object for generating monomials.
    num_tangents : int
        Tangents are generated in a loop until it reaches the expected number of tangents.
    max_degree : int
        Tangents are generated in a loop until it reaches the maximum degree.

    Returns
    -------
    Dict[Poly, Expr]
        A dictionary of items `(ineq*poly**2, ineq_expr*expr**2)`.
    """
    roots = [r for r in roots if r.eval(ineq) != 0]

    symbols = ineq.gens
    tangents = {}
    if ineq.is_monomial and sum(_ != 0 for _ in tuple(ineq.LM())) == 1:
        ineq, ineq_expr = Poly(1, *symbols), Integer(1)
    if roots:
        for degree in range(1, max_degree):
            mat = Matrix.hstack(*[r.span(degree) for r in roots])
            vecs = _get_sorted_nullspace(monomial_manager, mat, degree)
            if not vecs:
                continue

            new_polys = [monomial_manager.invarraylize(p, symbols, degree) for p in vecs]
            # new_polys = [p for p in new_polys if not p.is_monomial]
            new_polys = [_canonicalize(p) for p in new_polys]
            new_polys = set([p for p in new_polys if p is not None])
            tangents.update(dict([(ineq*p**2, ineq_expr*p.as_expr()**2) for p in new_polys]))
            if len(tangents) > num_tangents:
                break

    if all(not r.is_nontrivial for r in roots):
        tangents[ineq] = ineq_expr

    # print('Roots of', ineq_expr, ':', roots)
    # print('Tangents of', ineq_expr, ':', [e for t, e in tangents.items()])
    return tangents


def prepare_tangents(
        problem: InequalityProblem,
    qmodule: Optional[Dict[Poly, Expr]] = None,
    default_tangents = DEFAULT_TANGENTS,
    additional_tangents: List[Expr] = [],
) -> Dict[Poly, Expr]:
    """
    Prepare tangents for LinearSOS given a list of roots (equality cases). The tangents should
    vanish in the given roots. The function returns a dictionary of {tangent: tangent_expr}.
    The "tangent" does not have a definition of canonical forms, so the behaviour of this
    function might change in the future.

    The Polya's Positivstellensatz claims that if a homogeneous polynomial `F(x1,...xn) > 0`
    strictly holds over the simplex `x1,..,xn>=0, x1+...+xn=1`, then there must be a positive integer
    `m` such that all coefficients of `(x1+..+xn)^m*F(x1,..,xn)` are nonnegative. This gives a
    sum-of-squares proof of `F > 0`. However, this property does not hold when the inequality is weak,
    i.e., when there exists `F(x1,...,xn)=0` in the domain. The principle is easy to understand,
    as `(x1+..+xn)^m*F(x1,..,xn)` must exist negative coefficients to attain the zero. Thus the
    zeros (roots) of an inequality should be carefully handled to avoid conflicts.

    Consider a polynomial inequality `F >= 0` over R, but `F(*root) == 0` attains the zero of F.
    Suppose F can be decomposed into SOS: `F = (...)^2 + (...)^2 + ...`. Then each of the squares
    should attain zero at the given root, which forms implicit linear constraints on the coefficients.

    In SDPSOS, the feasible set of the sum-of-squares implicity satisfies a linear constraint
    in this case and has no interior. It can be reduced to a low-rank problem by applying
    a linear transformation. For example, if `p1,...,pn` form a basis of the polys that
    vanish in the roots, then the SDPSOS is instead trying to find a PSD quadratic form
    of `p1,...,pn` that evaluates to F and each base of the squares is a linear combinations
    of `p1,...,pn`. Working on the vanishing ideal keeps the zeros of F. For LinearSOS,
    we may relax the problem to `F = p1*PSD[1] + p2*PSD[2] + ...`, where PSD[i] are
    linear combinations of nonnegative polynomials.


    More generally, consider an inequality on a generic semialgebra set.
    Suppose the original polynomial can be written in the SOS-form:

        `F = sum(ineq * SOS for ineq in ineq_constraints) + sum(eq * p for eq in eq_constraints)`.

    Suppose `root` satisfies that `F(*root) == 0`, `ineq(*root) >= 0` and `eq(*root) == 0`. Then if
    `ineq_constraints[i](*root) > 0`, then there must be the complementary: `SOS[i](*root) == 0`.
    This constrains `SOS[i]` to lie in a subspace where the root vanishes.

    The function generates a linear basis of this subspace for each inequality constraint.

    In practice, the `ineq_constraints` are actually generators of a quadratic module. For instance:

        `(a^2+b^2-c^2 >= 0, a^2+c^2-b^2 >= 0) => ((a^2+b^2-c^2)*(a^2+c^2-b^2) >= 0)`.

    Thus new ineq_constraints are obtained by taking the product of possible combinations of
    the given ineq_constraints.

    TODO: Handle high-order roots.
    """
    poly = problem.expr
    ineq_constraints = qmodule if qmodule is not None else problem.ineq_constraints
    roots = [r for r in problem.roots if not r.is_zero] if problem.roots is not None else []

    symbols = poly.gens

    tangents = [t.as_expr() for t in additional_tangents]
    if all(not r.is_nontrivial for r in roots):
        # When there is a nontrivial root, then there cannot be
        # terms like "a^i*b^j*c^k*(a-b)^(2l)*(b-c)^(2m)*(c-a)^(2n)" in the SOS form,
        # as it does not vanish at the root.
        if Integer(1) not in tangents:
            tangents.append(Integer(1))

        if len(symbols) in default_tangents:
            tangents.extend(default_tangents[len(symbols)](*symbols))
    tangents = dict((Poly(t, symbols), t) for t in tangents)

    # remove very trivial roots
    roots = [r for r in roots if (not r.is_corner)] # and len(set(r.rep)) > 1]

    monomial_manager = MonomialManager(len(symbols))
    ineq_constraints = ineq_constraints.items() if isinstance(ineq_constraints, dict) else ineq_constraints
    for ineq, ineq_expr in ineq_constraints:
        new_tangents = _get_ineq_constrained_tangents(ineq, ineq_expr,
                roots=roots, monomial_manager=monomial_manager)
        tangents.update(new_tangents)
    # print(tangents)
    return tangents


def get_qmodule_list(
    poly: Poly,
    ineq_constraints: Dict[Poly, Expr],
    all_nonnegative: bool = False,
    preordering: str = 'quadratic'
) -> List[Tuple[Poly, Expr]]:
    """
    Extend the generators of the quadratic module given `ineq_constraints` given a
    preordering rule. For instance, if `F >= 0` given assumptions: `G1, ..., Gn >= 0`,
    then we would possibly assume `F = sum_i G_{i1} * ... * G_{ik_i} * SOS[i]`.

    Parameters
    ----------
    poly : Poly
        The target polynomial.
    ineq_constraints : Dict[Poly, Expr]
        The dictionary of inequality constraints.
    all_nonnegative : bool, optional
        If True, then all monomials are known to be nonnegative and trivial `ineq_constraints`
        (e.g. a monomial) will be filtered out to reduce the number of generators.
    preordering : str, optional
        The preordering rule, one of 'none', 'linear', 'quadratic', 'full'.
    """
    _ACCEPTED_PREORDERINGS = ['none', 'linear', 'quadratic', 'full']
    if not preordering in _ACCEPTED_PREORDERINGS:
        raise ValueError("Invalid preordering method, expected one of %s, received %s." % (str(_ACCEPTED_PREORDERINGS), preordering))

    # degree = poly.total_degree()
    poly_one = Poly(1, *poly.gens)

    monomials = []
    linear_ineqs = []
    nonlin_ineqs = [(poly_one, Integer(1))]
    for ineq, e in ineq_constraints.items():
        if ineq.is_monomial and len(ineq.free_symbols) == 1 and ineq.total_degree() == 1 and ineq.LC() >= 0:
            monomials.append((ineq, e))
        elif ineq.is_linear:
            linear_ineqs.append((ineq, e))
        elif preordering == 'quadratic' and ineq.is_quadratic:
            # Although they are nonlinear, but they should be combined
            # as linear constraints, so we treat them together.
            linear_ineqs.append((ineq, e))
        else:
            nonlin_ineqs.append((ineq, e))

    if all_nonnegative:
        # In this case we generate basis by LinearBasisTangent rather than LinearBasisTangentEven
        # we discard all monomials
        pass
    else:
        linear_ineqs = monomials + linear_ineqs

    if preordering == 'none':
        return linear_ineqs + nonlin_ineqs
    if preordering == 'full':
        linear_ineqs = linear_ineqs + nonlin_ineqs
        nonlin_ineqs = [(poly_one, Integer(1))]

    qmodule = nonlin_ineqs.copy()
    for n in range(1, len(linear_ineqs) + 1):
        for comb in combinations(linear_ineqs, n):
            mul = poly_one
            for c in comb:
                mul = mul * c[0]
            # d = mul.total_degree()
            # if d > degree:
            #     continue
            mul_expr = Mul(*(c[1] for c in comb))
            for ineq, e in nonlin_ineqs:
                # new_d = d + ineq.total_degree()
                # if new_d <= degree:
                qmodule.append((mul * ineq, mul_expr * e))

    return qmodule


###################################################################
#
#          Various heuristics to approximate tangents
#
###################################################################

def prepare_inexact_tangents(
        problem: InequalityProblem,
    monomial_manager: MonomialManager = None,
    all_nonnegative: bool = False,
    threshold: float = 0.5,
    max_degree: int = 5
) -> Dict[Poly, Expr]:
    """
    The function `prepare_tangents` has highlighted the importance of handling the roots.
    However, even if there are no zeros, we need to pay attention to local minima that are very close
    to zeros. They can be handled by a slight numerical perturbation that makes them numerical zeros.
    """
    poly = problem.expr
    ineq_constraints = problem.ineq_constraints
    eq_constraints = problem.eq_constraints
    nvars = len(poly.gens)
    roots = [r for r in problem.roots if not r.is_zero] if problem.roots is not None else []
    if nvars <= 1 or len(eq_constraints) or monomial_manager.is_symmetric or any(r.is_nontrivial for r in roots):
        return dict()

    # parameters for numerical optimization
    perms = [[1,0] + list(range(2, nvars))] # reflection
    if nvars >= 3:
        perms.append(list(range(1, nvars)) + [0])

    new_roots = []
    try:
        # numerically find local extrema in the feasible set
        new_roots = numeric_optimize_skew_symmetry(poly, poly.gens, perms, num=5)
        new_roots = [r for r in new_roots if all(ineq(*r) >= 0 for ineq in ineq_constraints)]
        new_roots = [Root(_, domain=RR) for _ in new_roots] # convert to RR
    except Exception as e: # shall we handle this?
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise e
        pass

    if len(new_roots) == 0:
        return dict()

    new_tangents = []
    monomial_base = monomial_manager.base()
    for root in new_roots:
        value = poly(*root)
        mean_value = sum([poly(*root[perm]) for perm in perms])/len(perms)
        if value < 0 or value > mean_value * threshold:
            # If value > mean_value * threshold,
            # it is not very close to zero and we discard it.
            continue

        for degree in range(2, max_degree):
            # Pretend that we require the root to be a zero of poly,
            # which generates polynomials that have low values around the root.
            mat = monomial_manager.permute_vec(root.span(degree), degree)
            mat = Matrix.hstack(mat, Matrix.ones(mat.shape[0], 1)) # TODO
            vecs = _get_sorted_nullspace(monomial_base, mat, degree)
            if not vecs:
                continue

            # Convert numerical polys to rationals
            vecs = [(vec * (1260/max(vec))).applyfunc(round)/1260 for vec in vecs] # TODO
            new_tangents.extend([
                _canonicalize(monomial_base.invarraylize(p, poly.gens, degree).retract()) for p in vecs])
            break

    new_tangents = dict([(t**2, t.as_expr()**2) for t in new_tangents if t is not None])
    # print('New tangents:\n', list(new_tangents.values()))

    return new_tangents
