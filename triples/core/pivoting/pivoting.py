from ..node import ProofNode

from .quadratic import pivoting_quadratic

class Pivoting(ProofNode):
    _constraints_wrapper = None

    def __init__(self, *args ,**kwargs):
        super().__init__(*args, **kwargs)

    def explore(self, configs):
        if self.state != 0:
            return

        self.state = 2
        problem = self.problem
        poly = problem.expr

        symmetry = problem.identify_symmetry()
        configs["symmetry"] = symmetry
        configs["signs"] = problem.get_symbol_signs()
        self._constraints_wrapper = problem.wrap_constraints(symmetry)

        if poly.total_degree() <= 2 or len(poly.gens) <= 1:
            # this should be handled by QCQP solvers or univariate solvers
            self.state = -1
            self.finished = True
            return

        wrapped_problem = self._constraints_wrapper[0]

        funcs = [pivoting_quadratic]
        for func in funcs:
            self.children.extend(func(wrapped_problem, configs))

        self.state = -1

    def update(self, *args, **kwargs):
        if self._constraints_wrapper[0].solution is not None:
            self.solution = self._constraints_wrapper[1](
                self._constraints_wrapper[0].solution)
