from typing import List, Dict, Union, Optional, TYPE_CHECKING

from sympy import construct_domain
from sympy.polys.polyerrors import BasePolynomialError

from .constrained import structural_sos_constrained
from .pivoting    import structural_sos_2vars
from .ternary     import structural_sos_3vars
from .quaternary  import structural_sos_4vars
from .nvars       import structural_sos_nvars
from ..preprocess import ProofNode, SolvePolynomial

from ..problem import ProblemComplexity
from ..solution import Solution

if TYPE_CHECKING:
    from sympy import Poly, Expr
    from ..problem import InequalityProblem

class StructuralSOSSolver(ProofNode):
    def explore(self, configs):
        if self.state == 0:
            problem, _homogenizer = self.problem.homogenize()

            solution = _structural_sos(problem)

            if solution is not None:
                if _homogenizer is not None:
                    self.solution = Solution.dehomogenize(solution, _homogenizer)
                else:
                    self.solution = solution

        self.state = -1
        self.finished = True

    def _evaluate_complexity(self) -> ProblemComplexity:
        # Fast in most cases
        return ProblemComplexity(0.001, 1.)


def StructuralSOS(
    expr: "Expr",
    ineq_constraints: Union[List["Expr"], Dict["Expr", "Expr"]] = {},
    eq_constraints: Union[List["Expr"], Dict["Expr", "Expr"]] = {},
    *,
    verbose: Union[bool, int] = False,
    raise_exception: bool = False,
) -> Optional["Solution"]:
    """
    A rule-based expert system to solve polynomial inequalities in specific structures.
    Most algorithms run in O(1) or linear time.

    Parameters
    ----------
    expr: Expr
        The expression to perform SOS on.
    ineq_constraints: Union[List[Expr], Dict[Expr, Expr]]
        Inequality constraints to the problem. This assumes g_1(x) >= 0, g_2(x) >= 0, ...
    eq_constraints: Union[List[Expr], Dict[Expr, Expr]]
        Equality constraints to the problem. This assumes h_1(x) = 0, h_2(x) = 0, ...
    verbose: bool
        Whether to print verbose information.
    raise_exception: bool
        Whether to raise exception when an error occurs. Set to True for debug purpose.
        Experimental.

    Returns
    -------
    solution: Solution

    Examples
    --------
    StructuralSOS uses an expert system to solve inequalities in specific structures. Many
    classical Olympiad-level ternary symmetric or cyclic inequalities are supported.

    >>> from triples import StructuralSOS, CyclicSum
    >>> from sympy.abc import a, b, c
    >>> sol = StructuralSOS(a**4*(a-b)*(a-c)+b**4*(b-c)*(b-a)+c**4*(c-a)*(c-b)
    ... -5*(a-b)**2*(b-c)**2*(c-a)**2, [a,b,c])
    >>> sol.solution # doctest:+SKIP
    4*((Σ(a**2*(a - b)*(a - c)))**2/4 + (Σ(a**2*(b - c)**2*(a**2 - 2*a*b - 2*a*c + b**2 + 2*b*c + c**2)**2))/8
    + (Σ(a*b*(a - b)**2*(a**2 - 2*a*b + 2*a*c + b**2 + 2*b*c - 3*c**2)**2))/4)/(Σ(a**2))

    StructuralSOS uses very fast (but incomplete) algorithms, and extends to high-degree
    or high-dimensional problems in some cases.

    >>> sol = StructuralSOS(a**30*(a-b)*(a-c)+b**30*(b-c)*(b-a)+c**30*(c-a)*(c-b), [a,b,c])
    >>> sol is not None
    True
    >>> sol.time # doctest:+SKIP
    0.191594

    Sometimes StructuralSOS better handles problems with ill-conditioned or irrational
    coefficients than other numerical algorithms.

    >>> from sympy import sqrt
    >>> sol = StructuralSOS(CyclicSum(a**3-a**2*b + (sqrt(13+16*sqrt(2))-1)/2*a*b*(b-a),
    ... (a,b,c)), [a,b,c])
    >>> sol.solution # doctest:+SKIP
    (2*(∏(a))*(Σ((a - b)**2)) + (Σ(a*(14*b**2 + b*(-a + c)*(-3*sqrt(13 + 16*sqrt(2))
    + 7 + sqrt(2)*sqrt(13 + 16*sqrt(2)) + 7*sqrt(2)) - 14*c**2 + c*(a - b)*(
    -sqrt(2)*sqrt(13 + 16*sqrt(2)) + 7 + 7*sqrt(2) + 3*sqrt(13 + 16*sqrt(2))))**2))/98)/(2*(Σ(a*b)))

    However, StructuralSOS is not a complete solver and it does not solve general inequality
    problems. It only provides a quick check to see whether a problem can be easily solved.
    """
    from ..node import ProofTree
    problem = ProofNode.new_problem(expr, ineq_constraints, eq_constraints)
    configs = {
        ProofTree: {"verbose": verbose},
        SolvePolynomial: {"solvers": [StructuralSOSSolver]},
        StructuralSOSSolver: {"verbose": verbose, "raise_exception": raise_exception},
    }
    return problem.sum_of_squares(configs)


def _structural_sos(problem: "InequalityProblem") -> "Expr":
    """
    Internal function of StructuralSOS, returns a sympy expression only.
    The polynomial must be homogeneous. TODO: remove the homogeneous requirement?
    """
    problem = problem.remove_redundancy()

    poly: "Poly" = problem.expr
    ineq_constraints = problem.ineq_constraints
    eq_constraints = problem.eq_constraints

    if poly.is_zero:
        return poly.as_expr()

    d = poly.total_degree()
    nvars = len(poly.gens)
    if poly.is_monomial:
        if poly.LC() >= 0 and d % 2 == 0 and all(_ % 2 == 0 for _ in poly.degree_list()):
            # since the poly is homogeneous, it must be a monomial
            return poly.as_expr()
        return None

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
        solution = structural_sos_2vars(problem)
    elif nvars == 3:
        solution = structural_sos_3vars(problem)
    elif nvars == 4:
        solution = structural_sos_4vars(problem)

    if solution is None and nvars > 3:
        solution = structural_sos_nvars(problem)

    if solution is None:
        solution = structural_sos_constrained(problem)

    return solution
