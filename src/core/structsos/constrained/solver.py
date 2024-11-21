from typing import Dict, Optional, Tuple, Callable

import sympy as sp

from .linear import elimination_linear
from .acute import constrained_acute


from ..utils import clear_free_symbols, has_gen
from ...shared import SS

_SOLVERS = [
    constrained_acute
]

def structural_sos_constrained(poly: sp.Poly, ineq_constraints: Dict[sp.Poly, sp.Expr] = {}, eq_constraints: Dict[sp.Poly, sp.Expr] = {}) -> Optional[sp.Expr]:
    """
    Solve general constrained polynomial inequalities by synthetic heuristics.
    """
    if len(ineq_constraints) == 0:
        return None

    for solver in _SOLVERS:
        solution = solver(poly, ineq_constraints, eq_constraints)
        if solution is not None:
            return solution


def structural_sos_constraints_elimination(poly: sp.Poly, ineq_constraints: Dict[sp.Poly, sp.Expr], eq_constraints: Dict[sp.Poly, sp.Expr]) -> Tuple[sp.Poly, Dict[sp.Poly, sp.Expr], Dict[sp.Poly, sp.Expr], Callable]:
    restore = lambda x: x
    funcs = [
        elimination_linear
    ]
    for func in funcs:
        poly, ineq_constraints, eq_constraints, restore = func(poly, ineq_constraints, eq_constraints, restore)
    return poly, ineq_constraints, eq_constraints, restore