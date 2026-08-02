from .qcqp import QCQPSolver
# from .cauchy import CauchySolver
from ..node import ProofNode
from ..problem import ProblemComplexity

class ProgSOSSolver(ProofNode):
    """
    Solve special forms of inequality problems by calling
    specific solvers.

    This solver differs from `StructuralSOSSolver` in that it
    uses optimization-based (numerical) methods to solve the problem.
    """
    def explore(self, configs):
        if self.state == 0:
            self.children = [
                QCQPSolver(self.problem),
                # CauchySolver(self.problem)
            ]
            self.state = -1

    def _evaluate_complexity(self) -> ProblemComplexity:
        # Fast in most cases
        return ProblemComplexity(0.01, 1.)
