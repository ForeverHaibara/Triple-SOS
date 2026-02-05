from sympy.abc import a, b, c, d, e, t, x, y, z
from sympy import sqrt

from ..pivoting import Pivoting
from ...problem import InequalityProblem
from ...node import ProofTree
from ...structsos import StructuralSOSSolver
from ...linsos import LinearSOSSolver
from ...preprocess import SolvePolynomial
from ....utils import SymmetricSum, CyclicSum

import pytest

class PivotingProblems:
    """
    Each of the problem must return a tuple of (expr, ineq_constraints, eq_constraints)
    """
    @classmethod
    def collect(cls):
        return {k: getattr(cls, k) for k in dir(cls) if k.startswith('problem')}

    @classmethod
    def problem_quadratic_real(cls):
        return a*x**2 + b*x + c, [a, c, 4*a*c - b**2], []

    @classmethod
    def problem_quadratic_real2(cls):
        # https://artofproblemsolving.com/community/u861323h3638618p35731117
        p0 = a*d*(b - c)**2*(-(a - d)**2 + (a + d - x*(a - b - c + d))**2)
        return SymmetricSum(p0, (a,b,c,d)), [a, b, c, d], []

    @classmethod
    def problem_quadratic_real_irrational(cls):
        # https://artofproblemsolving.com/community/u861323h3622674p35508375
        return (a**2 + 2)*(b**2 + x**2) + 4*a*b - 4*x*(a + b), [x - sqrt(2)], []

    @classmethod
    def problem_quadratic_r(cls):
        # holds for t <= 3 as well, but t <= 1 will make the
        # degree-1 coefficient nonnegative
        return CyclicSum((a + b - c)*(a - b)**2*(a + b - t*c)**2, (a, b, c)), [a, b, c, 1 - t], []

    @classmethod
    def problem_quadratic_l(cls):
        return (4*x*y*z + 9*((x-y)**2+(y-z)**2+(z-x)**2-(x-1)**2-(y-1)**2-(z-1)**2)), [x - 1, y - 1, z - 1], []

    @classmethod
    def problem_quadratic_lr(cls):
        return 8 - x**2*(y + 2) - y**2*(x + 2), [x, y, 3 - (x + 1)*(y + 1)], []

    # @classmethod
    # def problem_quadratic_horn(cls):
    #     return (a+b+c+d+e)**2 - 4*(a*b+b*c+c*d+d*e+e*a), [a,b,c,d,e], []


@pytest.mark.slow
@pytest.mark.parametrize('problem', PivotingProblems.collect().values(),
    ids=PivotingProblems.collect().keys())
def test_pivoting_problems(problem):
    expr, ineq_constraints, eq_constraints = problem()
    pro = InequalityProblem(expr, ineq_constraints, eq_constraints).polylize()
    pivot = Pivoting(pro)
    tree = ProofTree(pivot,
        {ProofTree: {'time_limit': 20.0},
         SolvePolynomial: {'solvers': [
             Pivoting, StructuralSOSSolver, LinearSOSSolver]},
         LinearSOSSolver: {'basis_limit': 1000}
        #  SDPSOSSolver: {'lift_degree_limit': 0}
        }
    )
    tree.solve()
    assert pro.solution is not None,\
        f"Failed to solve the problem using Pivoting {problem.__name__}:\n{str(pro)}"

    from ....testing.doctest_parser import solution_checker
    assert solution_checker(pro, expr, ineq_constraints, eq_constraints),\
        f"Solution checker assert False for Pivoting {problem.__name__}:\n{str(pro)}"
