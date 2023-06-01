import sympy as sp
from sympy.core.singleton import S

from ...utils.expression.solution import Solution, SolutionSimple


class SolutionSymmetric(Solution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def as_simple_solution(self):
        """
        When the expression is a nested fraction, we can simplify it.
        """
        numerator, multiplier = sp.fraction(sp.together(self.solution))

        if multiplier.is_constant():
            const, multiplier = const, S.One
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

class SolutionSymmetricSimple(SolutionSimple):#, SolutionSymmetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
