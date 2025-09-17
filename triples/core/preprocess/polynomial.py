from typing import List, Dict, Union, Tuple

from sympy import Expr, Poly

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
#           Transformations on the Problem
#
#########################################################

def _is_bidegree(p):
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

def _align_degree(p, p1, p2, accept_odd_degree=False):
    """Homogenize p given p1 == p2 and deg(p1) < deg(p2).
    Returns q, d, x such that q = p1**d * p + x(p2 - p1) where q, x are polynomials
    and q is homogeneous. Returns None if it fails"""
    ddiff = p2.total_degree() - p1.total_degree()
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


def reduce_over_quotient_ring(problem):
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

    # tested_bidgree = 0
    for eq, expr in eq_constraints.items():
        bideg = _is_bidegree(eq)
        if bideg is None:
            continue
        p1, p2, sgn = bideg
        # diffdeg = p2.total_degree() - p1.total_degree()
        accept_odd = True if (p1.total_degree() == 0 and p1.LC() > 0) else False
        # print(f'Bidegree: {eq} == 0  <=>  {p1} == {p2}')

        new_poly = _align_degree(poly, p1, p2, accept_odd_degree=accept_odd)
        if new_poly is None:
            continue
        sgn_expr = sgn * expr
        new_ineqs = {}
        new_eqs = {}
        success = True
        for ineq, ineq_expr in ineq_constraints.items():
            new_ineq = _align_degree(ineq, p1, p2, accept_odd_degree=accept_odd)
            if new_ineq is None:
                success = False
                break
            new_ineq, muldeg, quo = new_ineq
            new_ineqs[new_ineq] = (muldeg, ineq_expr, quo)
        if not success:
            continue
        for eq2, eq_expr in eq_constraints.items():
            if eq2 == eq:
                continue
            new_eq = _align_degree(eq2, p1, p2, accept_odd_degree=accept_odd)
            if new_eq is None:
                success = False
                break
            new_eq, muldeg, quo = new_eq
            new_eqs[new_eq] = (muldeg, eq_expr, quo)
        if not success:
            continue

        # update the expression associated with each constraint after homogenization
        symmetry = identify_symmetry_from_lists(
                    [[new_poly[0]], list(new_ineqs.keys()), list(new_eqs.keys())])
        if symmetry.is_trivial:
            symmetry = None
        def p2expr(p: Poly) -> Expr:
            # convert a polynomial to expr wisely by exploting the symmetry
            if (symmetry is not None) and verify_symmetry(p, symmetry):
                p = poly_reduce_by_symmetry(p, symmetry)
                return CyclicSum(p.as_expr(), p.gens, symmetry)
            return p.as_expr()
        p1_expr = p2expr(p1)
        for new_ineq, (muldeg, ineq_expr, quo) in new_ineqs.items():
            new_ineqs[new_ineq] = p1_expr**muldeg * ineq_expr + p2expr(quo)*sgn_expr 
        for new_eq, (muldeg, eq_expr, quo) in new_eqs.items():
            new_eqs[new_eq] = p1_expr**muldeg * eq_expr + p2expr(quo)*sgn_expr

        # homogenize successfully
        poly = new_poly[0]
        is_hom = True
        ineq_constraints, eq_constraints = new_ineqs, new_eqs
        def _align_degree_restore(x):
            return (x - p2expr(new_poly[2])*sgn_expr) / p1_expr**new_poly[1]
        restorations.append(_align_degree_restore)
        break

    if len(restorations) == 0:
        restorations.append(lambda x: x)
    def restoration(sol):
        if sol is None:
            return None
        for rs in restorations[::-1]:
            sol = rs(sol)
        return sol

    new_problem = ProofNode.new_problem(
        poly, ineq_constraints, eq_constraints,
    )
    return new_problem, restoration