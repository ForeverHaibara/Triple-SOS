import sympy as sp

from ...utils.expression.solution import Solution, SolutionSimple

class SolutionStructural(Solution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def as_simple_solution(self):
        """
        When the expression is a nested fraction, we can simplify it.
        """
        numerator, multiplier = sp.fraction(sp.together(self.solution))
        return SolutionStructuralSimple(
            problem = self.problem, 
            numerator = numerator,
            multiplier = multiplier,
            is_equal = self.is_equal_
        )


class SolutionStructuralSimple(SolutionSimple, SolutionStructural):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
