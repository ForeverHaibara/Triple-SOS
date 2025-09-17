from typing import List, Dict, Tuple, Union, Optional

from sympy import Expr, Poly, Rational, Integer, Mul, fraction

from .polynomial import SolvePolynomial
from .signs import sign_sos
from ..node import ProofNode


class CancelDenominator(ProofNode):
    _numer = None
    _denom = None
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
                SolveMul(self._denom)
            ]

            self.status = 1
            return

    def update(self, *args, **kwargs):
        if self.status == 1:
            if self._denom.solution is not None:
                self.children = [
                    SolvePolynomial(self._numer)
                ]
                self.status = -1
            elif len(self.children) == 0 and self._denom.solution is None:
                self.status = -1
                self.finished = True
                return
            
        elif self.status == -1:
            if self._numer.solution is not None:
                self.register_solution(self._numer.solution / self._denom.solution)



class SolveMul(ProofNode):
    """
    Prove the nonnegativity of a Mul expression before it is expanded to a
    polynomial. Trivially nonnegative terms in the multiplication are removed from the expression.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proved = []
        self.unproved = []

    def explore(self, configs):
        """Prove the nonnegativity of a sympy expression without expanding
        it to a sympy polynomial. This is useful for trivially nonnegative expressions,
        e.g., the denominator of an expression."""
        if self.status != 0:
            return
        self.status = 1

        expr = self.problem.expr
        if isinstance(expr, Rational):
            if expr.numerator >= 0 and expr.denominator > 0:
                self.register_solution(expr)
            self.status = -1
            self.finished = True
            return

        args = Mul.make_args(expr)
        signs = self.problem.get_symbol_signs()

        for arg in args:
            # prove nonnegativity for each term
            sol = sign_sos(arg, signs)
            if sol is not None:
                self.proved.append(sol)
            else:
                if arg.is_Pow and isinstance(arg.exp, Integer) and arg.exp % 2 == 0:
                    self.unproved.append((arg.base, 1))
                    self.proved.append(arg.base ** (arg.exp - 1))
                else:
                    self.unproved.append((arg, 1))

        if self.unproved:
            rest = Mul(*[base for base, exp in self.unproved])
            rest_problem = self.new_problem(rest, self.problem.ineq_constraints, self.problem.eq_constraints)
            self.children = [
                SolvePolynomial(rest_problem)
            ]
            self.status = -1
        else:
            self.register_solution(Mul(*self.proved))
            self.finished = True
        self.status = -1


    def update(self, *args, **kwargs):
        if len(self.children) == 1:
            sol = self.children[0].problem.solution
            if sol is not None:
                self.register_solution(sol * Mul(*self.proved))