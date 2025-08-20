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

    @classmethod
    def new_problem(cls, *args, **kwargs) -> InequalityProblem:
        """
        Convenient method to create a new InequalityProblem instance.
        """
        return InequalityProblem(*args, **kwargs)


class SolveProblem(ProofNode):
    def explore(self, configs):
        if self.status == 0:
            from .preprocess.modeling import ReformulateAlgebraic
            self.children = [ReformulateAlgebraic(self.problem)]
            self.status = 1




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