from typing import Dict, Any

from sympy import Poly, Expr, Integer, Dummy, Symbol

from ..node import ProofNode
from ..problem import InequalityProblem
from ...utils import PSatz

def _solver(expr, ineq_constraints, eq_constraints) -> ProofNode:
    from .polynomial import SolvePolynomial
    problem = InequalityProblem.new(expr, ineq_constraints, eq_constraints)
    return SolvePolynomial(problem)


def _dict_inject(x: Dict[Any, Any], y: Dict[Any, Any]) -> Dict[Any, Any]:
    x = x.copy()
    x.update(y)
    return x

class Pivoting(ProofNode):
    problem: InequalityProblem
    _constraints_wrapper = None
    def __init__(self, *args ,**kwargs):
        super().__init__(*args, **kwargs)
        self._pivots = []

    def explore(self, configs):
        if self.status != 0:
            return

        self.status = 2
        problem = self.problem
        poly = problem.expr

        self._constraints_wrapper = problem.wrap_constraints()

        if poly.total_degree() <= 2:
            # this should be handled by QCQP solvers
            self.status = -1
            self.finished = True

        gens = poly.gens
        for gen in gens:
            if poly.degree(gen) == 1:
                pass
            elif poly.degree(gen) == 2:
                _quadratic_pivoting(self, gen)

        self.status = -1

    def update(self, *args, **kwargs):
        deleted = False
        for ind, pivot in enumerate(self._pivots.copy()):
            if all(_.problem.solution is not None for _ in pivot['children']):
                self.register_solution(pivot['restoration']())
            if any(_.finished and _.problem.solution is None for _ in pivot['children']):
                del self._pivots[ind]
                deleted = True

        if deleted:
            for child in self.children.copy():
                if not any(child in pivot['children'] for pivot in self._pivots):
                    self.children.remove(child)




def _get_symbol_bounds(ineqs: Dict[Poly, Expr], eqs: Dict[Poly, Expr], x: Symbol):
    """Get (lb, x - lb), (ub, ub - x)"""
    lb, ub = (None, None), (None, None)
    if eqs:
        return
    if ineqs:
        if len(ineqs) == 1:
            ineq = list(ineqs.keys())[0]
            if ineq.is_monomial and \
                ineq.total_degree() == 1 and ineq.LC() > 0:
                lb = (Integer(0), ineqs[ineq]/ineq.LC())
        if lb is None and ub is None:
            return
    return lb, ub


def _quadratic_pivoting(self: Pivoting, x):
    """
    Pivoting on a quadratic expression.
    """
    problem = self._constraints_wrapper[0]
    poly = problem.expr
    p = poly.as_poly(x)

    ineqs, eqs = problem.ineq_constraints, problem.eq_constraints

    bounds = _get_symbol_bounds(*(problem.extract_constraints(x)[:2]), x)
    if bounds is None:
        return
    (lb, lb_expr), (ub, ub_expr) = bounds

    A, B, C = p.all_coeffs()
    if A == 0 or C == 0:
        return

    children = [
        _solver(A, ineqs, eqs)
    ]

    if lb is None and ub is None:
        children.append(_solver(4*A*C-B**2, ineqs, eqs))

        def restoration():
            a, ndisc = [_.problem.solution for _ in children]
            return ((2*A*x + B)**2 + ndisc) / (4*a)

    elif lb == 0 and ub is None:
        negB = -B.as_poly(*problem.expr.gens)
        db = Dummy('B')
        children.append(_solver(C, ineqs, eqs))
        children.append(_solver(4*A*C-B**2, _dict_inject(ineqs, {negB: db}), eqs))

        def restoration():
            a, c, ndisc = [_.problem.solution for _ in children]
            preorder = [db] + list(ineqs.values())
            ideal = list(eqs.values())
            p1 = a*x**2 + db*lb_expr + c
            p2 = ((2*A*x + B)**2 + ndisc) / (4*a)
            p1, p2 = [PSatz.from_sympy(preorder, ideal, _) for _ in [p1, p2]]
            if p1 is None or p2 is None:
                return None
            ps = p1.join(p2, 0)
            return ps.as_expr()

    def compose(r):
        def _composed():
            return self._constraints_wrapper[1](r())
        return _composed

    self.children.extend(children)
    self._pivots.append({
        'gen': x,
        'children': children,
        'restoration': compose(restoration)
    })
