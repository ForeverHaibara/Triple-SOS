from functools import wraps
from inspect import signature
from typing import List, Dict, Tuple, Union, Optional

import sympy as sp
from sympy import Expr, Poly, Rational, Integer, fraction

from ...utils import Solution

def handle_rational(
    disable_denom_finding_roots = False,
):
    """
    Convert all rational expressions to polynomials.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(poly: Expr,
                ineq_constraints: Dict[Expr, Expr] = {},
                eq_constraints: Dict[Expr, Expr] = {}, *args, **kwargs):
            if isinstance(poly, Expr):
                numer, denom = fraction(poly.together())
            elif isinstance(poly, Poly):
                numer, denom = poly.as_expr(), Integer(1)
            else:
                raise TypeError("poly must be an Expr or Poly object") # not expected to happen

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

            numer_sol = None
            denom_sol = None
            denom_solver = func
            if not isinstance(denom, Rational):
                denom_kwargs = kwargs.copy()
                if disable_denom_finding_roots:
                    denom_kwargs['roots'] = []
                denom_solver = lambda *_args, **_kwargs: func(*_args,  **_kwargs, **denom_kwargs)
                if denom_sol is not None:
                    denom_sol = denom_sol.solution
            denom_sol = prove_expr(denom, new_ineqs, new_eqs, denom_solver)

            if denom_sol is not None:
                numer_sol = func(numer, new_ineqs, new_eqs, *args, **kwargs)
                if numer_sol is not None:
                    new_sol = Solution(
                        problem = poly,
                        solution = numer_sol.solution / denom_sol,
                        ineq_constraints = ineq_constraints,
                        eq_constraints = eq_constraints,
                    )
                    return new_sol
        return wrapper
    return decorator


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