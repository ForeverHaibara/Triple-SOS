from typing import Union, List, Dict, Optional, Any

import sympy as sp

from .utils import Coeff, has_gen, clear_free_symbols
from .solution import SolutionStructural
from .nvars import sos_struct_nvars_quartic_symmetric
from .constrained import structural_sos_constrained
from .sparse import sos_struct_linear, sos_struct_quadratic
from .ternary import structural_sos_3vars
from .quarternary import structural_sos_4vars
from .pivoting import structural_sos_2vars
from ..preprocess import ProofNode, SolvePolynomial

from ..problem import ProblemComplexity
from ..solution import Solution

class StructuralSOSSolver(ProofNode):
    def explore(self, *args, **kwargs):
        if self.status == 0:
            problem, _homogenizer = self.problem.homogenize()

            solution = _structural_sos(problem.expr, problem.ineq_constraints, problem.eq_constraints)
            if solution is not None:
                problem.solution = solution

                if _homogenizer is not None:
                    self.problem.solution = Solution.dehomogenize(solution, _homogenizer)

        self.status = -1
        self.finished = True

    def _evaluate_complexity(self) -> ProblemComplexity:
        return ProblemComplexity(0.001, 1.)

# @sanitize(homogenize=True, infer_symmetry=False, wrap_constraints=False)
def StructuralSOS(
        poly: sp.Poly,
        ineq_constraints: Union[List[sp.Poly], Dict[sp.Poly, sp.Expr]] = {},
        eq_constraints: Union[List[sp.Poly], Dict[sp.Poly, sp.Expr]] = {},
        verbose: Union[bool, int] = False,
    ) -> SolutionStructural:
    """
    Main function of structural SOS. It solves polynomial inequalities by
    synthetic heuristics. For example, quartic 3-var cyclic polynomials have a complete
    algorithm, which can be solved directly and beautifully.

    Parameters
    -------
    poly: sp.Poly
        The polynomial to perform SOS on.
    ineq_constraints: List[sp.Poly]
        Inequality constraints to the problem. This assume g_1(x) >= 0, g_2(x) >= 0, ...
    eq_constraints: List[sp.Poly]
        Equality constraints to the problem. This assume h_1(x) = 0, h_2(x) = 0, ...

    Returns
    -------
    solution: SolutionStructuralSimple

    """
    problem = ProofNode.new_problem(poly, ineq_constraints, eq_constraints)
    configs = {
        SolvePolynomial: {'solvers': [StructuralSOSSolver]}
    }
    return problem.sum_of_squares(configs)


def _structural_sos(poly: sp.Poly, ineq_constraints: Dict[sp.Poly, sp.Expr] = {}, eq_constraints: Dict[sp.Poly, sp.Expr] = {}) -> sp.Expr:
    """
    Internal function of StructuralSOS, returns a sympy expression only.
    The polynomial must be homogeneous. TODO: remove the homogeneous requirement?
    """
    if poly.is_zero:
        return sp.Integer(0)
    d = poly.total_degree()
    nvars = len(poly.gens)
    if poly.is_monomial:
        if poly.LC() >= 0 and d % 2 == 0 and all(_ % 2 == 0 for _ in poly.degree_list()):
            # since the poly is homogeneous, it must be a monomial
            return poly.as_expr()
        return None

    poly, ineq_constraints, eq_constraints = clear_free_symbols(poly, ineq_constraints, eq_constraints)
    d = poly.total_degree()
    nvars = len(poly.gens)

    solution = None
    if nvars == 2:
        # homogeneous bivariate
        solution = structural_sos_2vars(poly, ineq_constraints, eq_constraints)
    elif nvars == 3:
        solution = structural_sos_3vars(poly, ineq_constraints, eq_constraints)
    elif nvars == 4:
        solution = structural_sos_4vars(poly, ineq_constraints, eq_constraints)

    if solution is None and nvars > 3:
        solution = sos_struct_nvars_quartic_symmetric(poly)
    if solution is None and nvars > 3 and d == 2:
        solution = sos_struct_quadratic(poly)

    if solution is None:
        solution = structural_sos_constrained(poly, ineq_constraints, eq_constraints)

    return solution
