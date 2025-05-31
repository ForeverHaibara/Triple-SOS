from datetime import datetime
from functools import wraps
from typing import List, Dict, Tuple, Union

from sympy import Expr, Poly, sympify

from .algebraic import handle_rational
from .modeling import handle_general_expr
from .polynomial import handle_polynomial
from ...utils import Solution

def sanitize(
        homogenize: bool = False,
        ineq_constraint_sqf: bool = True,
        eq_constraint_sqf: bool = True,
        infer_symmetry: bool = False,
        wrap_constraints: bool = False
    ):
    """
    Decorator for sum of square functions. It sanitizes the input type before calling the solver function.
    Non-polynomial input will be converted to polynomials by applying suitable transformations or
    by introducing auxiliary variables.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(poly: Expr,
                ineq_constraints: Union[List[Expr], Dict[Expr, Expr]] = {},
                eq_constraints: Union[List[Expr], Dict[Expr, Expr]] = {}, *args, **kwargs):
            start_time = datetime.now()
            poly = sympify(poly)
            if not isinstance(ineq_constraints, dict):
                ineq_constraints = {e: e for e in ineq_constraints}
            if not isinstance(eq_constraints, dict):
                eq_constraints = {e: e for e in eq_constraints}
            ineq_constraints = dict((sympify(e), sympify(e2).as_expr()) for e, e2 in ineq_constraints.items())
            eq_constraints = dict((sympify(e), sympify(e2).as_expr()) for e, e2 in eq_constraints.items())

            new_func = handle_polynomial(
                homogenize=homogenize, ineq_constraint_sqf=ineq_constraint_sqf, eq_constraint_sqf=eq_constraint_sqf,
                infer_symmetry=infer_symmetry, wrap_constraints=wrap_constraints)(func)
            new_func = handle_rational()(new_func)
            new_func = handle_general_expr()(new_func)
            
            sol = new_func(poly, ineq_constraints, eq_constraints, *args, **kwargs)

            if isinstance(sol, Solution):
                end_time = datetime.now()
                sol._start_time = start_time
                sol._end_time = end_time
                sol.problem = poly
                sol.ineq_constraints = ineq_constraints
                sol.eq_constraints = eq_constraints
            return sol
        return wrapper
    return decorator