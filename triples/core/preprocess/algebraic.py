from typing import List, Dict, Tuple, Union, Optional

import sympy as sp
from sympy import Expr, Poly, Rational, Integer, fraction

from .polynomial import SolvePolynomial
from ..node import ProofNode
from ...utils import Solution


class CancelDenominator(ProofNode):
    _numer = None
    _denom = None
    _numer_sol = None
    _denom_sol = None
    def explore(self, configs):
        problem = self.problem
        poly, ineq_constraints, eq_constraints = problem.expr, problem.ineq_constraints, problem.eq_constraints
        if self.status == 0:
            if isinstance(poly, Expr):
                numer, denom = fraction(poly.doit().together())
            elif isinstance(poly, Poly):
                numer, denom = poly, Integer(1)

            # handle constraints
            new_ineqs = {}
            new_eqs = {}
            for ineq, expr in ineq_constraints.items():
                if isinstance(ineq, Expr):
                    ineq = fraction(ineq.together())
                    new_ineqs[ineq[0]*ineq[1]] = expr * ineq[1]**2
                elif isinstance(ineq, Poly):
                    new_ineqs[ineq] = expr

            for eq, expr in eq_constraints.items():
                if isinstance(eq, Expr):
                    eq = fraction(eq.together())
                    new_eqs[eq[0]] = expr * eq[1]
                elif isinstance(eq, Poly):
                    new_eqs[eq] = expr

            self._numer = self.new_problem(numer, new_ineqs, new_eqs)
            self._denom = self.new_problem(denom, new_ineqs, new_eqs)

            self.children = [
                SolvePolynomial(self._denom)
            ]

            self.status = 1
            return

    def update(self, *args, **kwargs):
        if not self.children:
            return
        child = self.children[0]
        if child.finished:
            if child.problem is self._denom:
                if child.problem.solution is None:
                    self.finished = True
                    return
                self.status = 3
                self.children = [
                    SolvePolynomial(self._numer)
                ]
            elif child.problem is self._numer:
                if child.problem.solution is None:
                    return

                self.problem.solution = self._numer.solution / self._denom.solution
                self.finished = True

def prove_expr(expr: Expr,
        ineq_constraints: Dict[Expr, Expr],
        eq_constraints: Dict[Expr, Expr],
        solver,
    ) -> Optional[Expr]:
    """Prove the nonnegativity of a sympy expression without expanding
    it to a sympy polynomial. This is useful for trivially nonnegative expressions,
    e.g., the denominator of an expression."""
    def wrapped_solver(*args, **kwargs):
        sol = solver(*args, **kwargs)
        if isinstance(sol, Solution):
            sol = sol.solution
        return sol
    def _prove_by_recur(expr):
        if isinstance(expr, Rational):
            if expr.numerator >= 0:
                return expr
            return None
        elif expr.is_Pow:
            if isinstance(expr.args[1], Rational) and int(expr.args[1].numerator) % 2 == 0:
                return expr
            sol = _prove_by_recur(expr.args[0])
            if sol is not None:
                return sol ** expr.args[1]
        elif expr.is_Mul:
            # prove the nonnegativity of each term in the Mul object
            # NOTE: this does not hold for noncommutative problems
            proved = []
            unproved = []
            for arg in expr.args:
                arg_sol = _prove_by_recur(arg)
                if arg_sol is not None:
                    proved.append(arg_sol)
                else:
                    arg_sol_neg = _prove_by_recur(arg)
                    if arg_sol_neg is not None:
                        proved.append(arg_sol_neg)
                        unproved.append(-Rational(1,1))
                    else:
                        unproved.append(arg)

            proved_sol = expr.func(*proved)
            if len(unproved) == 0:
                return proved_sol
            unproved = expr.func(*unproved)
            if isinstance(unproved, Rational):
                if unproved >= 0:
                    return unproved * proved_sol
                else:
                    return None
            unproved_sol = wrapped_solver(unproved, ineq_constraints, eq_constraints)
            if unproved_sol is not None:
                return proved_sol * unproved_sol

        if len(expr.free_symbols) == 0:
            # e.g. (sqrt(2) - 1)
            sgn = (expr >= 0)
            if sgn in (sp.true, True):
                return expr
            return None

        # elif expr.is_Add:
        return wrapped_solver(expr, ineq_constraints, eq_constraints)
    return _prove_by_recur(expr)