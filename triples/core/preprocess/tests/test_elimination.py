from sympy.abc import a, b, c, x, y, z

from ..elimination import eliminate_power_constraints
from ..polynomial import SolvePolynomial
from ...problem import InequalityProblem
from ...node import ProofTree
from ...structsos import StructuralSOSSolver
from ...linsos import LinearSOSSolver

import pytest

class PowerEliminationProblems:
    """
    Each of the problem must return a tuple of (ineq_constraints, eq_constraints, signs)
    """
    @classmethod
    def collect(cls):
        return {k: getattr(cls, k) for k in dir(cls) if k.startswith('problem')}

    @classmethod
    def problem_amgm1(cls):
        return a + b + c - 3*x, [a,b,c], [a*b*c-x**3]

    @classmethod
    def problem_amgm2(cls):
        return a**2 + b**2 + c**2 - 3, [a,b,c], [a*b*c-1]

    @classmethod
    def problem_vasile(cls):
        return (a+b+c)**2 - 3*(a*x+b*y+c*z), [a,b,c,x,y,z], [x**2-a*b, y**2-b*c, z**2-c*a]


@pytest.mark.slow
@pytest.mark.parametrize("problem", PowerEliminationProblems.collect().values(),
    ids=PowerEliminationProblems.collect().keys())
def test_power_elimination(problem):
    expr, ineqs, eqs = problem()
    pro = InequalityProblem(expr, ineqs, eqs).polylize()

    new_pro, trans = eliminate_power_constraints(pro)

    tree = ProofTree(SolvePolynomial(new_pro),
        {ProofTree: {'time_limit': 20.0},
         SolvePolynomial: {'solvers': [
             StructuralSOSSolver, LinearSOSSolver]},
         LinearSOSSolver: {'basis_limit': 1000}
        #  SDPSOSSolver: {'lift_degree_limit': 0}
        }
    )
    tree.solve()

    assert new_pro.solution is not None,\
        f"Failed to solve the problem using Power Elimination {problem.__name__}:\n{str(pro)}"

    from ....testing.doctest_parser import solution_checker
    assert solution_checker(trans(new_pro.solution), expr, ineqs, eqs),\
        f"Solution checker assert False for Power Elimination {problem.__name__}:\n{str(pro)}"
