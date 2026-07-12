from typing import Optional, TYPE_CHECKING

from .acute import constrained_acute

if TYPE_CHECKING:
    from sympy import Expr
    from ...problem import InequalityProblem


_SOLVERS = [
    constrained_acute
]

def structural_sos_constrained(
    problem: "InequalityProblem"
) -> Optional["Expr"]:
    """
    Solve general constrained polynomial inequalities by synthetic heuristics.
    """
    if len(problem.ineq_constraints) == 0:
        return None

    for solver in _SOLVERS:
        solution = solver(problem)
        if solution is not None:
            return solution


# def structural_sos_constraints_elimination(
#     poly: "Poly", ineq_constraints: Dict["Poly", "Expr"], eq_constraints: Dict["Poly", "Expr"]
# ) -> Tuple["Poly", Dict["Poly", "Expr"], Dict["Poly", "Expr"], Callable]:
#     restore = lambda x: x
#     funcs = [
#         elimination_linear
#     ]
#     for func in funcs:
#         poly, ineq_constraints, eq_constraints, restore = func(poly, ineq_constraints, eq_constraints, restore)
#     return poly, ineq_constraints, eq_constraints, restore
