from typing import Dict, Optional, Tuple, Callable

from sympy import Poly, Expr

from .linear import elimination_linear
from .acute import constrained_acute


from ..utils import clear_free_symbols, has_gen

_SOLVERS = [
    constrained_acute
]

def structural_sos_constrained(
    poly: Poly, ineq_constraints: Dict[Poly, Expr] = {}, eq_constraints: Dict[Poly, Expr] = {}
) -> Optional[Expr]:
    """
    Solve general constrained polynomial inequalities by synthetic heuristics.
    """
    if len(ineq_constraints) == 0:
        return None

    for solver in _SOLVERS:
        solution = solver(poly, ineq_constraints, eq_constraints)
        if solution is not None:
            return solution


def structural_sos_constraints_elimination(
    poly: Poly, ineq_constraints: Dict[Poly, Expr], eq_constraints: Dict[Poly, Expr]
) -> Tuple[Poly, Dict[Poly, Expr], Dict[Poly, Expr], Callable]:
    restore = lambda x: x
    funcs = [
        elimination_linear
    ]
    for func in funcs:
        poly, ineq_constraints, eq_constraints, restore = func(poly, ineq_constraints, eq_constraints, restore)
    return poly, ineq_constraints, eq_constraints, restore
