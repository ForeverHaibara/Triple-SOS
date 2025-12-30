from datetime import datetime
import traceback
from time import perf_counter
from typing import Dict, List, Union, Optional, Callable, Any
import os

from sympy import Expr
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

    A ProofNode instance is mutable and its state can be updated during
    exploration.


    """
    state = 0
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
        return f"ProofNode.{self.__class__.__name__}({repr(self.problem)}, state={self.state}, children={len(self.children)})"

    def __str__(self):
        return self.__repr__()

    @property
    def status(self) -> int:
        from warnings import warn
        warn("status is renamed, use state instead", DeprecationWarning, stacklevel=2)
        return self.state

    @status.setter
    def status(self, value: int):
        from warnings import warn
        warn("status is renamed, use state instead", DeprecationWarning, stacklevel=2)
        self.state = value

    @property
    def solution(self):
        return self.problem.solution

    @solution.setter
    def solution(self, value: Optional[Solution]):
        # zen of python: there should be one obvious way to do it
        self._register_solution(value)

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
        if self._complexity is None or self._complexity.state != self.state:
            self._complexity = self._evaluate_complexity()
        self._complexity.state = self.state
        return self._complexity

    def _evaluate_complexity(self) -> ProblemComplexity:
        if self._complexity_models is None:
            return self.problem.evaluate_complexity()
        if self._complexity_models is True:
            models = self._load_complexity_models()
            self.__class__._complexity_models = models
            self._complexity_models = models
        models = self._complexity_models
        features = self._evaluate_complexity_get_features()
        complexity = ProblemComplexity(
            models["time_model"].predict(features),
            models["prob_model"].predict(features),
            models["length_model"].predict(features),
        )
        return complexity

    def _evaluate_complexity_get_features(self):
        """Return the features to be used for complexity evaluation."""
        return self.problem.get_features()

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

    def _register_solution(self, solution: Optional[Expr]) -> Optional[Expr]:
        """
        Default behaviour: reserve the better solution
        """
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
        if self.state == 0:
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
                solution = restoration(child.problem.solution)
                if solution is not None:
                    self.solution = solution


class SolveProblem(ProofNode):
    def explore(self, configs):
        if self.state == 0:
            from .preprocess.modeling import ReformulateAlgebraic
            self.children = [ReformulateAlgebraic(self.problem)]
            self.state = -1


class ProofTree:
    configs: dict
    parents: dict
    _expected_end_time: float

    default_configs = {
        "mode": "fast",
        "max_explore": 100,
        "time_limit": 3600.,
        "select_quick_accept_threshold": 0.01,
    }

    def __init__(self, root: ProofNode, configs: Dict = {}):
        self.root = root
        self.configs = configs
        self.parents = {}
        self._expected_end_time = 0

    def get_configs(self, node: Union[ProofNode, 'ProofTree']) -> Dict[str, Any]:
        cfg = node.default_configs.copy()

        if not (node is self):
            # compute the time limit dynamically (the remaining time)
            cfg["time_limit"] = min(self._expected_end_time - perf_counter(),
                                    cfg.get("time_limit", float("inf")))

        for cls in node.__class__.mro()[::-1]:
            if cls in self.configs:
                cfg.update(self.configs[cls])
        cfg.update(self.configs.get(node, {}))
        return cfg

    @property
    def mode(self) -> str:
        return self.get_configs(self)["mode"]

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

        if cfg.get("callback_before_explore"):
            cfg["callback_before_explore"](self, node, cfg)

        try:
            node.explore(cfg)
        except Exception as e:
            # Internal errors are not expected, but could occur as the code is very complex
            # To prevent the program from crashing, we catch them here
            if isinstance(e, ArithmeticTimeout):
                # internal node reaches the time limit, ignore this
                pass
            else:
                # kill this node
                node.finished = True
                if cfg.get("raise_exception", False):
                    raise e
                if cfg.get("verbose", False):
                    traceback.print_exc()

        if cfg.get("callback_after_explore"):
            cfg["callback_after_explore"](self, node, cfg)

        self.propagate(node)

    def propagate(self, node: ProofNode):
        """
        Propagate the state of a node to its parents.
        """
        mode = self.mode
        p = node
        while p is not None:
            p.update(None)
            if mode == 'fast' and p.problem.solution is not None:
                p.finished = True
            if p.state < 0 and all(_.finished for _ in p.children):
                p.finished = True
            p = self.parents.get(p, None)

    def solve_until(self, condition: Callable[['ProofTree'], bool]):
        """
        Repeatly call `explore` until the condition is met
        or it reaches the configured limit.
        This function is helpful when called from nodes to solve
        a subproblem.
        """
        _tree_configs = self.get_configs(self)
        # time_limit = _tree_configs["time_limit"]
        max_explore = _tree_configs["max_explore"]

        for _ in range(max_explore):
            try:
                self.explore()
            except ArithmeticTimeout:
                break
            if perf_counter() > self._expected_end_time:
                break
            if condition(self):
                break

    def solve(self) -> Optional[Expr]:
        # recompute expected end time
        time_limit = self.get_configs(self)["time_limit"]
        self._expected_end_time = time_limit + perf_counter()

        self.solve_until(lambda self: self.root.finished)
        return self.root.problem.solution


def _sum_of_squares(
    problem: InequalityProblem,
    configs: dict = {},
) -> Optional[Solution]:
    start_time = datetime.now()

    root = SolveProblem(problem)
    tree = ProofTree(root, configs)

    solution = tree.solve()
    if solution is None:
        return None

    end_time = datetime.now()
    solution = Solution(problem, solution).rewrite_symmetry()
    solution._start_time = start_time
    solution._end_time = end_time
    return solution
