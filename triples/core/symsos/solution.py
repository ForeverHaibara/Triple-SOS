import sympy as sp
from sympy.core.singleton import S

from ...utils.expression.solution import Solution, SolutionSimple


class SolutionSymmetric(Solution):
    method = 'SymmetricSOS'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def as_simple_solution(self):
        sol = SolutionSymmetricSimple(problem = self.problem, solution = self.solution,
            ineq_constraints = self.ineq_constraints, eq_constraints = self.eq_constraints, is_equal = self.is_equal)
        return sol

    @classmethod
    def _extract_nonnegative_exprs(cls, expr: sp.Expr, func_name: str = "_G"):
        """
        Raw output of SymmetricSOS might assume nonnegativity of some symbols,
        we extract these symbols and replace them with _F(x) for further processing.
        This is not intended to be used by end users.

        TODO: Move this to SolutionSimple???
        """
        from ..structsos.solution import SolutionStructuralSimple
        return SolutionStructuralSimple._extract_nonnegative_exprs(expr, func_name)


class SolutionSymmetricSimple(SolutionSimple):#, SolutionSymmetric):
    method = 'SymmetricSOS'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def is_equal(self):
        return True