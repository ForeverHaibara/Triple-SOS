from typing import List, Dict, Union, Tuple, Callable, Optional

from sympy import Expr, Poly

from ..problem import InequalityProblem
from ..node import ProofNode, TransformNode
from ...utils import (
    Solution, MonomialManager, CyclicSum,
    identify_symmetry_from_lists, verify_symmetry, poly_reduce_by_symmetry
)


class SolvePolynomial(TransformNode):
    """
    Solve a dense polynomial inequality. The target expression and its constraints
    are all converted and stored as sympy (dense) Poly class. However, the process
    of converting expressions to dense polynomials is inefficient for very large inputs.
    """
    _dense_problem = None
    def explore(self, configs):
        if self.status == 0:
            # self._dense_problem, _restoration = self.problem.polylize(), lambda x: x
            self._dense_problem, _restoration = reduce_over_quotient_ring(self.problem.polylize())

            solvers = configs.get('solvers', None)
            if solvers is None:
                from ..structsos.structsos import StructuralSOSSolver
                from ..linsos.linsos import LinearSOSSolver
                from ..sdpsos.sdpsos import SDPSOSSolver
                from ..symsos.symsos import SymmetricSubstitution
                from .pivoting import Pivoting
                solvers = [
                    StructuralSOSSolver,
                    LinearSOSSolver,
                    SDPSOSSolver,
                    SymmetricSubstitution,
                    Pivoting
                ]
            # from ..structsos.structsos import StructuralSOSSolver
            # from ..linsos.linsos import LinearSOSSolver
            # from ..sdpsos.sdpsos import SDPSOSSolver
            # from ..symsos.symsos import SymmetricSubstitution
            # solvers = [
            #         SymmetricSubstitution,
            #         StructuralSOSSolver,
            #         LinearSOSSolver,
            #         SDPSOSSolver,
            # ]

            self.children = [solver(self._dense_problem) for solver in solvers]

            self.status = - 1

        self.restorations = {c: _restoration for c in self.children}
        if self.status > 0 and len(self.children) == 0:
            # all children failed
            self.finished = True


#########################################################
#
#                Bidegree Homogenization
#
#########################################################

def _is_bidegree(p: Poly) -> Optional[Tuple[Poly, Poly, int]]:
    """Check whether a multivariate polynomial has two different degrees.
    Returns l1, l2, sgn such that p = sgn * (l2 - l1). Also,
    l1, l2 are homogeneous, deg(l1) < deg(l2) and l1.LC() > 0. Returns
    None if p does not have the property.
    
    Examples
    ---------
    >>> from sympy.abc import a, b, c
    >>> _is_bidegree((a**2 + b**2 + c**2).as_poly(a,b,c)) is None
    True
    >>> _is_bidegree((2 - 3*(a+b+c) - a*b*c).as_poly(a,b,c)) is None
    True
    >>> _is_bidegree((3*(a+b+c) - a*b*c).as_poly(a,b,c))
    (Poly(3*a + 3*b + 3*c, a, b, c, domain='ZZ'), Poly(a*b*c, a, b, c, domain='ZZ'), -1)
    >>> _is_bidegree((-3*(a+b+c) + a*b*c).as_poly(a,b,c))
    (Poly(3*a + 3*b + 3*c, a, b, c, domain='ZZ'), Poly(a*b*c, a, b, c, domain='ZZ'), 1)
    """
    terms = {}
    for m, c in p.terms():
        d = sum(m)
        if d not in terms:
            if len(terms) == 2:
                return None
            terms[d] = [(m, c)]
        else:
            terms[d].append((m, c))
    if len(terms) != 2:
        return None
    d1, l1 = terms.popitem()
    d2, l2 = terms.popitem()
    if d1 > d2:
        d1, d2 = d2, d1
        l1, l2 = l2, l1
    # d1 < d2
    l1 = Poly(dict(l1), p.gens, domain=p.domain)
    l2 = Poly(dict(l2), p.gens, domain=p.domain)
    if l1.LC() < 0:
        l1 = -l1
        sgn = 1
    else:
        l2 = -l2
        sgn = -1
    return l1, l2, sgn

def _align_degree(p: Poly, p1: Poly, p2: Poly, accept_odd_degree: bool = False) -> Optional[Tuple[Poly, int, Poly]]:
    """
    Homogenize p given p1 == p2 and deg(p1) < deg(p2).
    Returns q, d, x such that q = p1**d * p + x(p2 - p1) where q, x are polynomials
    and q is homogeneous. Returns None if it fails.

    Parameters
    ----------
    p: Poly
        The polynomial to be homogenized.
    p1: Poly
        The first polynomial in the alignment.
    p2: Poly
        The second polynomial in the alignment.
    accept_odd_degree: bool
        If False, d is forced to be even.

    Examples
    --------
    >>> from sympy.abc import a, b, c
    >>> from sympy import Poly
    >>> p, p1, p2 = Poly(a**2+b**2+c**2 - 3, (a,b,c)), Poly(3, (a,b,c)), Poly(a+b+c, (a,b,c))
    >>> q, d, x = _align_degree(p, p1, p2); (q, d, x) # doctest: +NORMALIZE_WHITESPACE
    (Poly(6*a**2 - 6*a*b - 6*a*c + 6*b**2 - 6*b*c + 6*c**2, a, b, c, domain='ZZ'),
    2,
    Poly(-3*a - 3*b - 3*c - 9, a, b, c, domain='ZZ'))
    >>> (q - (p1**d * p + x*(p2 - p1)))
    Poly(0, a, b, c, domain='ZZ')
    """
    ddiff = p2.total_degree() - p1.total_degree()
    if ddiff == 0:
        return None
    terms = {}
    for m, c in p.terms():
        d = sum(m)
        if d not in terms:
            terms[d] = [(m, c)]
        else:
            terms[d].append((m, c))
    # p = sum(d * polys[d] for d in terms)
    keys = sorted(list(terms.keys()))
    for i in range(len(keys)-1):
        if (keys[i+1]-keys[i])%ddiff != 0:
            return None
    muldeg = (keys[-1] - keys[0])//ddiff
    if muldeg == 0:
        # nothing to do
        return p, 0, Poly({}, p.gens, domain=p.domain)
    if (not accept_odd_degree) and muldeg % 2 != 0:
        # d is not even, might multiply a negative term
        return None
    polys = [Poly(dict(terms[d]), p.gens, domain=p.domain) for d in keys]
    q = Poly({}, p.gens, domain=p.domain)
    for d, poly in zip(keys, polys):
        codeg = (keys[-1] - d)//ddiff
        # poly -> poly * p2**codeg * p1**muldeg
        q += poly * p2**codeg * p1**(muldeg - codeg)
        # x += poly * (p2**codeg - p1**codeg)/(p2 - p1) * p1**(muldeg - codeg)
    divrem = (q - p1**muldeg*p).div(p2 - p1)
    if not divrem[1].is_zero:
        return None
    # print(p, '- (hom) ->', q, muldeg, divrem[0])
    return q, muldeg, divrem[0]

def _bidegree_recover_expr(lst: List[Dict[Poly, Tuple[Poly, int, Poly]]], p1: Poly, sgn_expr: Expr) -> Expr:
    """
    Utility function for bidegree homogenization.
    Recover the original expressions from homogenization info, each represented
    by a tuple (mul_deg, expr, quo). After recovery, it would be:

        `p1.as_expr()**mul_deg * expr + sgn_expr * quo.as_expr()`

    The modification is done in-place. Returns only the expression
    form of `p1`.
    """
    symmetry = identify_symmetry_from_lists([list(d.keys()) for d in lst])
    if symmetry.is_trivial:
        symmetry = None

    def p2expr(p: Poly) -> Expr:
        # convert a polynomial to expr wisely by exploting the symmetry
        if (symmetry is not None) and verify_symmetry(p, symmetry):
            p = poly_reduce_by_symmetry(p, symmetry)
            return CyclicSum(p.as_expr(), p.gens, symmetry)
        return p.as_expr()
    p1_expr = p2expr(p1)
    for d in lst:
        for p, (mul_deg, expr, quo) in d.items():
            d[p] = p1_expr**mul_deg * expr + p2expr(quo)*sgn_expr
    return p1_expr

def _bidegree_attempt(problem: InequalityProblem, eq: Poly) -> Optional[Tuple[InequalityProblem, Callable]]:
    """
    Test whether the constraint `eq` can be use to homogenize the original problem.

    If yes, returns a new problem and a function to recover the solution.
    """
    poly, ineq_constraints, eq_constraints = problem.expr, problem.ineq_constraints, problem.eq_constraints

    bideg = _is_bidegree(eq)
    if bideg is None:
        return None
    p1, p2, sgn = bideg

    accept_odd = True if (p1.total_degree() == 0 and p1.LC() > 0) else False
    # print(f'Bidegree: {eq} == 0  <=>  {p1} == {p2}')

    # handle them universally
    dicts = [{poly: 0}, ineq_constraints, eq_constraints]
    new_dicts = [{}, {}, {}]

    for dict_i, src in enumerate(dicts):
        dst = new_dicts[dict_i]
        for key, expr in src.items():
            if dict_i == 2 and key == eq:
                continue
            alignment = _align_degree(key, p1, p2, accept_odd_degree=accept_odd)
            if alignment is None:
                return None
            # to avoid wasting time, we only store the homogenization info here
            new_poly, mul_deg, quo = alignment
            dst[new_poly] = (mul_deg, expr, quo)

    # All polynomials are homogenized,
    # now we restore the homogenization info (tuples) to exprs.
    mul_deg = next(iter(new_dicts[0].values()))[0]
    eq_expr = eq_constraints[eq]
    p1_expr = _bidegree_recover_expr(new_dicts, p1, sgn * eq_expr)

    new_poly, expr_shift = next(iter(new_dicts[0].items()))
    def _align_degree_restore(x):
        """Recover z such that `(p1_expr**mul_deg * z + expr_shift) == x`"""
        return (x - expr_shift) / p1_expr**mul_deg
    new_problem = InequalityProblem(new_poly, new_dicts[1], new_dicts[2])
    new_problem.roots = problem.roots
    return new_problem, _align_degree_restore

def bidegree_homogenization(problem: InequalityProblem) -> Tuple[InequalityProblem, Callable]:
    for eq in problem.eq_constraints:
        attempt = _bidegree_attempt(problem, eq)
        if attempt is not None:
            return attempt
    return problem, lambda x: x


def reduce_over_quotient_ring(problem: InequalityProblem):
    """
    Perform quotient ring reduction of the problem, including operations like
    homogenization.

    Given equality constraint f(x) = g(x) where f,g are nonzero homogeneous polynomials
    and deg(f) = deg(g), we obtain 1 = (f(x)/g(x)) and can be used for homogenization.

    TODO:
    1. Eliminate linear equality constraints, e.g. a+2b=3
    2. Eliminate linear inequality constraints, e.g. b+c-a, c+a-b, a+b-c>=0
    """
    poly, ineq_constraints, eq_constraints = problem.expr, problem.ineq_constraints, problem.eq_constraints
    restorations = []
    ################################################################
    #           Homogenize the polynomial and constraints
    ################################################################
    is_hom = poly.is_homogeneous and all(e.is_homogeneous for e in ineq_constraints)\
        and all(e.is_homogeneous for e in eq_constraints)
    if is_hom:
        # nothing to do
        return problem, lambda x: x

    ################################################################
    #         Homogenize using bidegree constraints
    ################################################################

    problem, restore = bidegree_homogenization(problem)
    restorations.append(restore)

    if len(restorations) == 0:
        restorations.append(lambda x: x)
    def restoration(sol):
        if sol is None:
            return None
        for rs in restorations[::-1]:
            sol = rs(sol)
        return sol

    return problem, restoration