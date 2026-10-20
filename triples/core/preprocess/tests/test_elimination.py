from sympy.abc import a, b, c, x, y, z
from sympy import Function, Poly

from ..elimination import (
    eliminate_power_constraints, eliminate_var_by_constraint, resultant_elimination
)
from ..polynomial import SolvePolynomial
from ...problem import InequalityProblem
from ...node import ProofTree
from ...structsos import StructuralSOSSolver
from ...linsos import LinearSOSSolver
from ...sdpsos import SDPSOSSolver

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
        return 4*(a**2 + b**2 + c**2) - 3, [a,b,c], [8*a*b*c-1]

    @classmethod
    def problem_vasile(cls):
        return (a+b+c)**2 - 3*(a*x+b*y+c*z), [a,b,c,x,y,z], [x**2-a*b, y**2-b*c, z**2-c*a]

    @classmethod
    def problem_const1(cls):
        return (a + b + c)**2 - 3*x*(a*b + b*c + c*a), [a, b, c], [x - 1]


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
             StructuralSOSSolver, LinearSOSSolver],
             'homogenize': False,
             'eliminate_binomial_constraints': False,
             },
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


def test_resultant_elimination_examples():
    # test solution recovery
    expr = (x**2 + y**2 - 1)**3 - x**2 - y**3
    constraint = x**2 + y**2 - 4
    equality = Function('H')(x, y)
    problem = InequalityProblem(expr, [], {constraint: equality}).polylize()
    reduced, restore = resultant_elimination(problem)

    assert reduced.expr == Poly(-y**3 + y**2 + 23, y)
    assert set(reduced.ineq_constraints) == {Poly(4 - y**2, y)}
    assert not reduced.eq_constraints
    bound_proof = reduced.ineq_constraints[Poly(4 - y**2, y)]
    assert (bound_proof - (x**2 - equality)).expand() == 0

    # Use the projected bound in a certificate, then restore the original
    # equality representative as well as the original target polynomial.
    certificate = 19 + bound_proof + y**2*(bound_proof + (y - 2)**2)/4
    solution = restore(certificate)
    assert (solution.xreplace({equality: constraint}) - expr).expand() == 0


    # test constraints
    for leading in [-1, 1]:
        constraint = leading*x**2 + y**2 - 4
        expr = constraint + y**3 + 1
        problem = InequalityProblem(expr, [constraint]).polylize()
        reduced, restore = eliminate_var_by_constraint(
            problem, next(iter(problem.ineq_constraints)), 0)

        if leading < 0:
            assert set(reduced.ineq_constraints) == {Poly(y**2 - 4, y)}
            proof = next(iter(reduced.ineq_constraints.values()))
            assert (proof - (y**2 - 4)).expand() == 0
        else:
            # x**2 + y**2 >= 4 imposes no upper bound on y**2.
            assert not reduced.ineq_constraints
        assert (restore(reduced.expr.as_expr()) - expr).expand() == 0


class ResultantEliminationProblems:
    """
    Each of the problem must return a tuple of (ineq_constraints, eq_constraints, signs)
    """
    @classmethod
    def collect(cls):
        return {k: getattr(cls, k) for k in dir(cls) if k.startswith('problem')}

    @classmethod
    def problem_p12074(cls):
        return 4*(a**4+b**4+c**4)+11*a*b*c*(a+b+c)-45, [], [a*b+b*c+c*a-3]

    @classmethod
    def problem_heart_and_circle(cls):
        return (x**2+y**2-1)**3 - x**2*y**3, [], [x**2+y**2-4]


@pytest.mark.slow
@pytest.mark.parametrize("problem", ResultantEliminationProblems.collect().values(),
    ids=ResultantEliminationProblems.collect().keys())
def test_resultant_elimination(problem):
    expr, ineqs, eqs = problem()
    pro = InequalityProblem(expr, ineqs, eqs).polylize()

    new_pro, trans = resultant_elimination(pro)

    tree = ProofTree(SolvePolynomial(new_pro),
        {ProofTree: {'time_limit': 20.0},
         SolvePolynomial: {'solvers': [
             StructuralSOSSolver, LinearSOSSolver, SDPSOSSolver],
             'homogenize': True,
             'eliminate_binomial_constraints': True,
             'eliminate_power_constraints': False,
             },
         LinearSOSSolver: {'basis_limit': 1000},
         SDPSOSSolver: {'lift_degree_limit': 1}
        }
    )
    tree.solve()

    assert new_pro.solution is not None,\
        f"Failed to solve the problem using Resultant Elimination {problem.__name__}:\n{str(pro)}"

    from ....testing.doctest_parser import solution_checker
    assert solution_checker(trans(new_pro.solution), expr, ineqs, eqs),\
        f"Solution checker assert False for Resultant Elimination {problem.__name__}:\n{str(pro)}"
