from typing import List, Dict, Tuple, Union, Optional

import sympy as sp
from sympy import Expr, Poly, Rational, Integer, Mul, Symbol, fraction

from .polynomial import SolvePolynomial
from ..node import ProofNode
from ...utils import CyclicExpr, Solution


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
                SolveMul(self._denom)
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
                self.problem.solution = expr
            self.status = 4
            self.finished = True
            return

        args = Mul.make_args(expr)
        signs = self.problem.get_symbol_signs()

        for arg in args:
            # prove nonnegativity for each term
            sol = prove_by_recur(arg, signs)
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
            self.status = 2
        else:
            self.problem.solution = Mul(*self.proved)
            self.status = 4
            self.finished = True


    def update(self, *args, **kwargs):
        if len(self.children) == 1:
            sol = self.children[0].problem.solution
            if sol is not None:
                self.problem.solution = sol * Mul(*self.proved)
            if self.children[0].finished:
                self.status = 2
                self.finished = True



def prove_by_recur(expr, signs: Dict[Symbol, Tuple[int, Expr]]):
    """
    Very fast and simple nonnegativity check for a SymPy (commutative, real)
    expression instance given signs of symbols.
    """
    if isinstance(expr, Rational):
        if expr.numerator >= 0 and expr.denominator > 0:
            return expr
        return None
    elif expr.is_Symbol:
        if signs.get(expr, (0, None))[0] == 1:
            return signs[expr][1]
        return None
    elif expr.is_Pow:
        if isinstance(expr.exp, Rational) and int(expr.exp.numerator) % 2 == 0:
            return expr
        sol = prove_by_recur(expr.base, signs)
        if sol is not None:
            return sol ** expr.exp
        return None
    elif expr.is_Add or expr.is_Mul:
        nonneg = []
        for arg in expr.args:
            nonneg.append(prove_by_recur(arg, signs))
            if nonneg[-1] is None:
                return None
        return expr.func(*nonneg)
    elif isinstance(expr, CyclicExpr):
        arg = expr.args[0]
        mulargs = []
        if arg.is_Pow:
            mulargs = [arg]
        elif arg.is_Mul:
            mulargs = arg.args
        def single(x):
            if x.is_Pow and isinstance(x.exp, Rational) and int(x.exp.numerator) % 2 == 0:
                return True
            if isinstance(x, Rational) and x >= 0:
                return True
            return False

        if all(single(_) for _ in mulargs):
            return expr

        # TODO: make it nicer
        return prove_by_recur(expr.doit(deep=False), signs)

    if len(expr.free_symbols) == 0:
        # e.g. (sqrt(2) - 1)
        sgn = (expr >= 0)
        if sgn in (sp.true, True):
            return expr
        return None