from functools import wraps
from inspect import signature
from typing import List, Dict, Tuple, Union

import sympy as sp
from sympy import Expr, Poly, Rational, Integer, fraction

from ...utils import SolutionSimple as Solution

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
            if isinstance(denom, Rational) or len(denom.free_symbols) == 0:
                # denominator is a constant
                sgn = (denom > 0)
                if sgn in (sp.true, True):
                    denom_sol = denom
                elif sgn in (sp.false, False):
                    denom_sol = -denom
                    numer = -numer
                # else: the sign is not determined

            else:
                denom_kwargs = kwargs.copy()
                if disable_denom_finding_roots:
                    denom_kwargs['roots'] = []
                denom_sol = func(denom, new_ineqs, new_eqs, *args, **denom_kwargs)
                if denom_sol is not None:
                    denom_sol = denom_sol.solution

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