from typing import List, Dict, Union, Optional, Any
import traceback

from sympy import Poly, Expr, Integer, construct_domain
from sympy.polys.polyerrors import BasePolynomialError

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
    default_configs = {
        'verbose': False,
        'raise_exception': False,
    }
    def explore(self, configs):
        if self.status == 0:
            problem, _homogenizer = self.problem.homogenize()

            try:
                solution = _structural_sos(problem.expr, problem.ineq_constraints, problem.eq_constraints)
            except Exception as e:
                # Internal errors are not expected, but could occur as the code is very complex
                # To prevent the program from crashing, we catch them here
                if configs['raise_exception']:
                    raise e
                if configs['verbose']:
                    traceback.print_exc()
                solution = None

            if solution is not None:
                problem.solution = solution

                if _homogenizer is not None:
                    self.problem.solution = Solution.dehomogenize(solution, _homogenizer)

        self.status = -1
        self.finished = True

    def _evaluate_complexity(self) -> ProblemComplexity:
        # Fast in most cases
        return ProblemComplexity(0.001, 1.)

# @sanitize(homogenize=True, infer_symmetry=False, wrap_constraints=False)
def StructuralSOS(
        poly: Poly,
        ineq_constraints: Union[List[Poly], Dict[Poly, Expr]] = {},
        eq_constraints: Union[List[Poly], Dict[Poly, Expr]] = {},
        verbose: Union[bool, int] = False,
        raise_exception: bool = False,
    ) -> Optional[SolutionStructural]:
    """
    A rule-based expert system to solve polynomial inequalities in specific structures.
    Most algorithms run in O(1) or linear time.

    Parameters
    -------
    poly: Poly
        The polynomial to perform SOS on.
    ineq_constraints: List[Poly]
        Inequality constraints to the problem. This assume g_1(x) >= 0, g_2(x) >= 0, ...
    eq_constraints: List[Poly]
        Equality constraints to the problem. This assume h_1(x) = 0, h_2(x) = 0, ...
    verbose: bool
        Whether to print verbose information.
    raise_exception: bool
        Whether to raise exception when an error occurs. Set to True for debug purpose.

    Returns
    -------
    solution: Solution

    """
    problem = ProofNode.new_problem(poly, ineq_constraints, eq_constraints)
    configs = {
        SolvePolynomial: {'solvers': [StructuralSOSSolver]},
        StructuralSOSSolver: {'verbose': verbose, 'raise_exception': raise_exception},
    }
    return problem.sum_of_squares(configs)


def _structural_sos(poly: Poly, ineq_constraints: Dict[Poly, Expr] = {}, eq_constraints: Dict[Poly, Expr] = {}) -> Expr:
    """
    Internal function of StructuralSOS, returns a sympy expression only.
    The polynomial must be homogeneous. TODO: remove the homogeneous requirement?
    """
    if poly.is_zero:
        return Integer(0)
    d = poly.total_degree()
    nvars = len(poly.gens)
    if poly.is_monomial:
        if poly.LC() >= 0 and d % 2 == 0 and all(_ % 2 == 0 for _ in poly.degree_list()):
            # since the poly is homogeneous, it must be a monomial
            return poly.as_expr()
        return None

    poly, ineq_constraints, eq_constraints = clear_free_symbols(poly, ineq_constraints, eq_constraints)

    if poly.domain.is_EX or poly.domain.is_EXRAW:
        # cast the polynomial to an extended domain
        try:
            dom, rep = construct_domain(poly.as_dict(zero=True), field=True, extension=True)
            poly = poly.from_dict(rep, poly.gens, domain=dom)
        except BasePolynomialError:
            return None
        if poly is None or poly.domain.is_EX or poly.domain.is_EXRAW:
            return None

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
