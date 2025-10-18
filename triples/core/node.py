from datetime import datetime
from time import perf_counter
from typing import Optional

from sympy import Expr, Poly, Rational, Integer, fraction

from .problem import InequalityProblem
from ..utils import Solution
from ..sdp import ArithmeticTimeout

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
    default_configs = {}
    def __init__(self,
        problem: InequalityProblem
    ):
        self.problem = problem
        self.children = []

    def __repr__(self):
        return f"ProofNode.{self.__class__.__name__}({self.problem})"

    def __str__(self):
        return self.__repr__()

    def select(self) -> 'ProofNode':
        """
        Select the most promising child node based on heuristics.
        It can also return itself if self is to be explored.
        """
        if self.children:
            return self.children[0]
        return self

    def explore(self, configs):
        pass

    def update(self, *args, **kwargs):
        pass

    def register_solution(self, solution: Optional[Expr]) -> Optional[Expr]:
        if solution is not None:
            if self.problem.solution is None:
                self.problem.solution = solution
            else:
                len_old = len(str(self.problem.solution))
                len_new = len(str(solution))
                if len_new < len_old:
                    self.problem.solution = solution
            # print('Register solution to', self.__class__, self.problem.solution)
        return solution

    @classmethod
    def new_problem(cls, *args, **kwargs) -> InequalityProblem:
        """
        Convenient method to create a new InequalityProblem instance.
        """
        return InequalityProblem(*args, **kwargs)


class TransformNode(ProofNode):
    """
    A special class of nodes that expects solutions from child problems.
    Each child node is associated with a restoration function.

    When any child is solved, `update` is called to restore the solution
    to the original problem.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.restorations = {}

    def update(self, *args, **kwargs):
        if self.finished:
            return
        for child in self.children:
            if child.problem.solution is not None:
                restoration = self.restorations[child]
                self.register_solution(restoration(child.problem.solution))


class SolveProblem(ProofNode):
    def explore(self, configs):
        if self.status == 0:
            from .preprocess.modeling import ReformulateAlgebraic
            self.children = [ReformulateAlgebraic(self.problem)]
            self.status = -1



def _sum_of_squares(
        problem: InequalityProblem,
        configs: dict = {},
        time_limit: float = 3600,
        mode: str = 'fast',
    ):
    start_time = datetime.now()

    configs = configs.copy()
    configs['start_time'] = start_time
    expected_end_time = perf_counter() + time_limit
    # configs['time_limit'] = time_limit

    root = SolveProblem(problem)
    max_explore = 100
    for _ in range(max_explore):
        # get the deepest child
        cur = root
        path = [cur]
        while cur.children:
            for c in cur.children.copy():
                if c.finished:
                    cur.children.remove(c)
            
            new_cur = cur.select()
            if new_cur is cur:
                # explore itself
                break
            else:
                cur = new_cur
                path.append(cur)

        # explore the deepest child
        cfg = cur.default_configs.copy()
        cfg['time_limit'] = (expected_end_time - perf_counter())
        cfg.update(configs.get(cur, {}))
        for cls in cur.__class__.mro()[::-1]:
            if cls in configs:
                cfg.update(configs[cls])

        if cfg.get('verbose', 0):
            print(f'Exploring {" -> ".join([_.__class__.__name__ for _ in path])}')
        try:
            cur.explore(cfg)
        except ArithmeticTimeout:
            break

        # if cur.finished:
        for p in path[::-1]:
            p.update(cur)

            if mode == 'fast' and p.problem.solution is not None:
                p.finished = True
            if p.status < 0 and all(_.finished for _ in p.children):
                p.finished = True
        if root.finished:
            break

        if perf_counter() > expected_end_time:
            break

    if problem.solution is None:
        return None

    end_time = datetime.now()
    solution = Solution(
        problem.expr,
        problem.solution,
        ineq_constraints = problem.ineq_constraints,
        eq_constraints = problem.eq_constraints,
    ).rewrite_symmetry()
    solution._start_time = start_time
    solution._end_time = end_time
    return solution