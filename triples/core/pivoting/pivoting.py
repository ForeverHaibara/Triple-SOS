from typing import List, Dict
from ..problem import InequalityProblem
from ..node import ProofNode

from .quadratic import pivoting_quadratic

class Pivoting(ProofNode):
    problem: InequalityProblem
    _constraints_wrapper = None
    _pivots: List[dict]

    def __init__(self, *args ,**kwargs):
        super().__init__(*args, **kwargs)
        self._pivots = []

    def explore(self, configs):
        if self.status != 0:
            return

        self.status = 2
        problem = self.problem
        poly = problem.expr

        self._constraints_wrapper = problem.wrap_constraints()

        if poly.total_degree() <= 2 or len(poly.gens) <= 1:
            # this should be handled by QCQP solvers or univariate solvers
            self.status = -1
            self.finished = True
            return

        wrapped_problem = self._constraints_wrapper[0]

        funcs = [pivoting_quadratic]
        for func in funcs:
            new_pivots = func(wrapped_problem, configs)
            self._pivots.extend(new_pivots)

        children = set().union(*[set(pivot['children']) for pivot in self._pivots])
        self.children.extend(list(children))

        self.status = -1

    def update(self, *args, **kwargs):
        """
        For each pivot:
        * If all children are finished and have a solution, register the solution
            by calling restoration.
        * If any child claims finished without a solution, remove the whole pivot.

        If any child in `self.children` does not exist in any pivot, remove the child.
        """
        del_inds = []
        for ind, pivot in enumerate(self._pivots.copy()):
            if all(_.problem.solution is not None for _ in pivot['children']):
                solution = pivot['restoration']()
                if solution is not None:
                    solution = self._constraints_wrapper[1](solution)
                    self.register_solution(solution)

            if any(_.finished and _.problem.solution is None for _ in pivot['children']):
                del_inds.append(ind)

        if del_inds:
            for ind in del_inds[::-1]:
                del self._pivots[ind]

            for child in self.children.copy():
                if not any(child in pivot['children'] for pivot in self._pivots):
                    self.children.remove(child)
