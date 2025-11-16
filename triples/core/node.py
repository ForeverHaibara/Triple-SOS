from datetime import datetime
from time import perf_counter
from typing import Dict, List, Union, Optional
import os

from sympy import Expr, Poly, Rational, Integer, fraction
import numpy as np

from .problem import InequalityProblem
from .complexity import ProblemComplexity
from .solution import Solution
from ..utils.tree_predictor import TreePredictor
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

    children: List['ProofNode']
    _complexity: Optional[ProblemComplexity] = None
    _complexity_models: Optional[Union[Dict, bool]] = None
    def __init__(self,
        problem: InequalityProblem
    ):
        self.problem = problem
        self.children = []

    def __repr__(self):
        return f"ProofNode.{self.__class__.__name__}({repr(self.problem)}, status={self.status}, children={len(self.children)})"

    def __str__(self):
        return self.__repr__()

    def select(self) -> 'ProofNode':
        """
        Select the most promising child node based on heuristics.
        It can also return itself if self is to be explored.
        """
        if not self.children:
            return self
        return self.children[0]
        # return min(self.children, key=lambda x: x.evaluate_complexity().time)

    def evaluate_complexity(self) -> ProblemComplexity:
        if self._complexity is None or self._complexity.status != self.status:
            self._complexity = self._evaluate_complexity()
        self._complexity.status = self.status
        return self._complexity

    def _evaluate_complexity(self) -> ProblemComplexity:
        if self._complexity_models is None:
            return self.problem.evaluate_complexity()
        if self._complexity_models is True:
            models = self._load_complexity_models()
            self.__class__._complexity_models = models
            self._complexity_models = models
        models = self._complexity_models
        features = self.problem.get_features()
        complexity = ProblemComplexity(
            models["time_model"].predict(features),
            models["prob_model"].predict(features),
            models["length_model"].predict(features),
        )
        return complexity

    def _load_complexity_models(self) -> Dict:
        from importlib import import_module
        models = {}
        clsname = self.__class__.__name__.lower()
        join = os.path.join
        filename = import_module(self.__class__.__module__).__file__
        path = os.path.dirname(os.path.abspath(filename))
        models["time_model"] = TreePredictor.load_model(join(path, "models", clsname+"_time_model.npz"))
        models["prob_model"] = TreePredictor.load_model(join(path, "models", clsname+"_prob_model.npz"))
        models["length_model"] = TreePredictor.load_model(join(path, "models", clsname+"_length_model.npz"))
        _time_func = models["time_model"].get_default_func()
        _length_func = models["length_model"].get_default_func()
        models["time_model"].func = lambda x: max(float(np.exp(_time_func(x)) - 1.), 1e-14)
        models["length_model"].func = lambda x: max(float(np.exp(_length_func(x)) - 1.), 1e-14)
        return models

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
    _default_complexity = (1., 1.)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.restorations = {}

    def _evaluate_complexity(self) -> ProblemComplexity:
        if self.status == 0:
            # when not explored, encourage exploration
            return ProblemComplexity(*self._default_complexity)
        if not self.children:
            return self.problem.evaluate_complexity()
        complexities = [child.evaluate_complexity() for child in self.children]
        c = complexities[0]
        for c2 in complexities[1:]:
            c = c | c2
        return c

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


class ProofTree:
    configs: dict
    parents: dict
    mode: str = 'fast'
    time_limit: float = 3600
    expected_end_time: float = 0

    def __init__(self, root: ProofNode):
        self.root = root
        self.configs = {}
        self.parents = {}

    def get_configs(self, node: ProofNode):
        cfg = node.default_configs.copy()
        cfg['time_limit'] = (self.expected_end_time - perf_counter())
        cfg.update(self.configs.get(node, {}))
        for cls in node.__class__.mro()[::-1]:
            if cls in self.configs:
                cfg.update(self.configs[cls])
        return cfg

    def select(self) -> ProofNode:
        """
        Select the most promising node to explore.

        TODO: perhaps we need a heap to select the most promising node.
        """
        leaves = self.get_leaves()
        complexities = [(leaf, leaf.evaluate_complexity()) for leaf in leaves]
        # print("Leaves:\n    " + "\n    ".join([f'{leaf}: {c}' for leaf, c in complexities]))
        best = min(complexities, key=lambda x: x[1].time/(x[1].prob + x[1].EPS))[0]
        return best

    def get_leaves(self) -> List[ProofNode]:
        """
        Get all leaf nodes in the proof tree.
        """
        cur = self.root
        leaves = []
        queue = [cur]
        while queue:
            cur = queue.pop(0)
            cur.children = [_ for _ in cur.children if not _.finished]
            if (not cur.children) or cur in cur.children:
                leaves.append(cur)
            queue.extend(cur.children)
            for c in cur.children:
                self.parents[c] = cur
        return leaves

    def explore(self):
        """
        Explore the most promising node.
        """
        node = self.select()
        cfg = self.get_configs(node)
        if cfg.get('verbose', False):
            path = [node]
            MAX_PATH_LEN = 5
            while path[-1] in self.parents and len(path) < MAX_PATH_LEN:
                path.append(self.parents[path[-1]])
            print(f'Exploring ... {" -> ".join([_.__class__.__name__ for _ in path[::-1]])}')
        node.explore(cfg)
        self.propagate(node)

    def propagate(self, node: ProofNode):
        """
        Propagate the status of a node to its parents.
        """
        p = node
        while p is not None:
            p.update(None)
            if self.mode == 'fast' and p.problem.solution is not None:
                p.finished = True
            if p.status < 0 and all(_.finished for _ in p.children):
                p.finished = True
            p = self.parents.get(p, None)

    def solve(self) -> Optional[Expr]:
        self.expected_end_time = perf_counter() + self.time_limit
        max_explore = 100
        for _ in range(max_explore):
            try:
                self.explore()
            except ArithmeticTimeout:
                break
            if perf_counter() > self.expected_end_time:
                break
            if self.root.finished:
                break

        return self.root.problem.solution


def _sum_of_squares(
        problem: InequalityProblem,
        configs: dict = {},
        time_limit: float = 3600,
        mode: str = 'fast',
    ):
    start_time = datetime.now()

    root = SolveProblem(problem)
    tree = ProofTree(root)
    tree.configs = configs
    tree.time_limit = time_limit
    tree.mode = mode

    solution = tree.solve()
    if solution is None:
        return None

    end_time = datetime.now()
    solution = Solution(problem, solution).rewrite_symmetry()
    solution._start_time = start_time
    solution._end_time = end_time
    return solution
