from sympy.abc import a, b, c, d, e, t, x, y, z
from sympy import sqrt

from ..symsos import SymmetricSubstitution
from ...problem import InequalityProblem
from ...node import ProofTree
from ...structsos import StructuralSOSSolver
from ...sdpsos import SDPSOSSolver
from ...preprocess import SolvePolynomial
from ....utils import SymmetricSum, CyclicSum

import pytest

class SymmetricSubstitutionProblems:
    """
    Each of the problem must return a tuple of (expr, ineq_constraints, eq_constraints)
    """
    @classmethod
    def collect(cls):
        return {k: getattr(cls, k) for k in dir(cls) if k.startswith('problem')}

    @classmethod
    def problem_u3real(cls):
        return CyclicSum((b-c)**2*(b+c-5*a)**2, (a,b,c))/7, [], []

    @classmethod
    def problem_u3positive_cubic(cls):
        return CyclicSum(a*(a-b)*(a-c)+b*(a-c)**2, (a,b,c)), [a,b,c], []

    @classmethod
    def problem_u3positive_quintic(cls):
        return CyclicSum(a*(a-b)*(a-c)*(b+c-a)**2, (a,b,c)), [a,b,c], []

    @classmethod
    def problem_u4real_quartic(cls):
        return SymmetricSum((a-b)**2*(a+b-2*c-2*d)**2, (a,b,c,d)), [], []

    @classmethod
    def problem_u4real_sextic(cls):
        return SymmetricSum((a-b)**2*(c-d)**2*(a+b-3*c-3*d)**2, (a,b,c,d)), [], []


@pytest.mark.slow
@pytest.mark.parametrize('problem', SymmetricSubstitutionProblems.collect().values(),
    ids=SymmetricSubstitutionProblems.collect().keys())
def test_pivoting_problems(problem):
    expr, ineq_constraints, eq_constraints = problem()
    pro = InequalityProblem(expr, ineq_constraints, eq_constraints).polylize()
    pivot = SymmetricSubstitution(pro)
    tree = ProofTree(pivot,
        {ProofTree: {'time_limit': 30.0},
         SolvePolynomial: {'solvers': [
             SymmetricSubstitution, StructuralSOSSolver, SDPSOSSolver]},
         SDPSOSSolver: {'lift_degree_limit': 0}
        }
    )
    tree.solve()
    assert pro.solution is not None,\
        f"Failed to solve the problem using SymmetricSubstitution {problem.__name__}:\n{str(pro)}"

    from ....testing.doctest_parser import solution_checker
    assert solution_checker(pro, expr, ineq_constraints, eq_constraints),\
        f"Solution checker assert False for SymmetricSubstitution {problem.__name__}:\n{str(pro)}"
