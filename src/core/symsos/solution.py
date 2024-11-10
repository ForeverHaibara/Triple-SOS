import sympy as sp
from sympy.core.singleton import S

from ...utils.expression.solution import Solution, SolutionSimple


class SolutionSymmetric(Solution):
    method = 'SymmetricSOS'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def as_simple_solution(self):
        """
        When the expression is a nested fraction, we can simplify it.
        """
        numerator, multiplier = sp.fraction(sp.together(self.solution))

        if len(multiplier.free_symbols) == 0:
            const, multiplier = S.One, multiplier
        else:
            const, multiplier = multiplier.as_coeff_Mul()

        if isinstance(numerator, sp.Add):
            numerator = sp.Add(*[arg / const for arg in numerator.args])
        else:
            numerator = numerator / const

        return SolutionSymmetricSimple(
            problem = self.problem, 
            numerator = numerator,
            multiplier = multiplier,
            is_equal = self.is_equal_
        )

    @classmethod
    def _extract_nonnegative_symbols(cls, expr: sp.Expr, func_name: str = "_G"):
        """
        Raw output of SymmetricSOS might assume nonnegativity of some symbols,
        we extract these symbols and replace them with _F(x) for further processing.
        This is not intended to be used by end users.

        TODO: Move this to SolutionSimple???
        """
        from ..structsos.solution import SolutionStructuralSimple
        return SolutionStructuralSimple._extract_nonnegative_symbols(expr, func_name)


class SolutionSymmetricSimple(SolutionSimple):#, SolutionSymmetric):
    method = 'SymmetricSOS'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def is_equal(self):
        return True