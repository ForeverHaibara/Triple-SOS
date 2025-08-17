from datetime import datetime

from sympy import Expr, Poly, Rational, Integer, fraction

from .problem import InequalityProblem
from ..utils import Solution

class ProofNode:
    """
    The goal of a proof node is to prove a single inequality problem.
    There are two types of workflows during the search for a proof:
    + Sequential tasks
    + Parallel tasks

    Sequential tasks are carried out in order, and are done in a single
    ProofNode. When a ProofNode calls `.explore()`, it moves forward to
    the next task in the sequence.

    Parallel tasks are branches that can be explored concurrently.
    A ProofNode can have multiple children, each representing a different
    branch of exploration.

    A ProofNode instance is mutable and its status can be updated during
    exploration.


    """
    status = 0
    finished = False
    def __init__(self,
        problem: InequalityProblem
    ):
        self.problem = problem
        self.children = []

    def __repr__(self):
        return f"ProofNode.{self.__class__.__name__}({self.problem})"

    def __str__(self):
        return self.__repr__()

    def explore(self, configs):
        ...

    def update(self, node, *args, **kwargs):
        if self.problem.solution is not None:
            self.finished = True


class SolveProblem(ProofNode):
    def explore(self, configs):
        if self.status == 0:
            self.children = [ReformulateAlgebraic(self.problem)]
            self.status = 1

class ReformulateAlgebraic(ProofNode):
    inverse = None
    def explore(self, configs):
        if self.status == 0:
            from .preprocess.modeling import ModelingHelper
            problem = self.problem
            expr, ineq_constraints, eq_constraints = problem.expr, problem.ineq_constraints, problem.eq_constraints
            helper = ModelingHelper(expr, ineq_constraints, eq_constraints)
            new_expr, new_ineqs, new_eqs, inverse = helper.formulate()

            self.children = [
                CancelDenominator(
                    InequalityProblem(
                        new_expr, new_ineqs, new_eqs
                    )
                )
            ]
            self.inverse = inverse
            self.status = 1

    def update(self, *args, **kwargs):
        if self.children and self.children[0].problem.solution is not None:
            self.problem.solution = self.children[0].problem.solution.xreplace(self.inverse)
            self.finished = True


class CancelDenominator(ProofNode):
    _numer = None
    _denom = None
    _numer_sol = None
    _denom_sol = None
    def explore(self, configs):
        problem = self.problem
        poly, ineq_constraints, eq_constraints = problem.expr, problem.ineq_constraints, problem.eq_constraints
        if self.status == 0:
            if isinstance(poly, Expr):
                numer, denom = fraction(poly.doit().together())
            elif isinstance(poly, Poly):
                numer, denom = poly, Integer(1)

            # handle constraints
            new_ineqs = {}
            new_eqs = {}
            for ineq, expr in ineq_constraints.items():
                if isinstance(ineq, Expr):
                    ineq = fraction(ineq.together())
                    new_ineqs[ineq[0]*ineq[1]] = expr * ineq[1]**2
                elif isinstance(ineq, Poly):
                    new_ineqs[ineq] = expr

            for eq, expr in eq_constraints.items():
                if isinstance(eq, Expr):
                    eq = fraction(eq.together())
                    new_eqs[eq[0]] = expr * eq[1]
                elif isinstance(eq, Poly):
                    new_eqs[eq] = expr

            self._numer = InequalityProblem(
                numer, new_ineqs, new_eqs
            )
            self._denom = InequalityProblem(
                denom, new_ineqs, new_eqs
            )
            self.children = [
                SolvePolynomial(self._denom)
            ]

            self.status = 1
            return

    def update(self, *args, **kwargs):
        if not self.children:
            return
        child = self.children[0]
        if child.finished:
            if child.problem is self._denom:
                if child.problem.solution is None:
                    self.finished = True
                    return
                self.status = 3
                self.children = [
                    SolvePolynomial(self._numer)
                ]
            elif child.problem is self._numer:
                if child.problem.solution is None:
                    return

                self.problem.solution = self._numer.solution / self._denom.solution
                self.finished = True


class SolvePolynomial(ProofNode):
    _dense_problem = None
    def explore(self, configs):
        if self.status == 0:
            self._dense_problem = self.problem.polylize()

            solvers = configs.get('solvers', None)
            if solvers is None:
                from .structsos.structsos import StructuralSOSSolver
                from .linsos.linsos import LinearSOSSolver
                from .sdpsos.sdpsos import SDPSOSSolver
                solvers = [
                    StructuralSOSSolver,
                    LinearSOSSolver,
                    SDPSOSSolver,
                ]
            self.children = [solver(self._dense_problem) for solver in solvers]

            self.status = 1

    def update(self, *args, **kwargs):
        if self._dense_problem is not None and self._dense_problem.solution is not None:
            self.problem.solution = self._dense_problem.solution
            self.finished = True


def _sum_of_squares(problem: InequalityProblem, configs = {}):
    start_time = datetime.now()
    root = SolveProblem(problem)
    max_explore = 20
    for _ in range(max_explore):
        # get the deepest child
        cur = root
        path = [cur]
        while cur.children:
            for c in cur.children.copy():
                if c.finished:
                    cur.children.remove(c)
                else:
                    cur = c
                    path.append(c)
                    break
            else:
                break

        # explore the deepest child
        # print(f'Exploring {cur}')
        cur.explore(configs.get(cur.__class__, {}))
        if cur.finished:
            for p in path[::-1]:
                p.update(cur)
        if root.finished:
            break

    if problem.solution is None:
        return None

    end_time = datetime.now()
    solution = Solution(
        problem.expr,
        problem.solution,
    )
    solution._start_time = start_time
    solution._end_time = end_time
    return solution