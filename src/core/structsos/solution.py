import sympy as sp
from sympy.core.singleton import S

from ...utils.expression.solution import Solution, SolutionSimple

class SolutionStructural(Solution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def as_simple_solution(self):
        """
        When the expression is a nested fraction, we can simplify it.
        """
        numerator, multiplier = sp.fraction(sp.together(self.solution))

        if multiplier.is_constant():
            const, multiplier = multiplier, S.One
        else:
            # const, multiplier = multiplier.as_coeff_Mul()
            const, multiplier = S.One, multiplier

        if const is not S.One:
            if isinstance(numerator, sp.Add):
                numerator = sp.Add(*[arg / const for arg in numerator.args])
            else:
                numerator = numerator / const

        return SolutionStructuralSimple(
            problem = self.problem, 
            numerator = numerator,
            multiplier = multiplier,
            is_equal = self.is_equal_
        )


class SolutionStructuralSimple(SolutionSimple, SolutionStructural):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # for debug purpose
        mul = self.multiplier.doit().as_poly(*sp.symbols('a b c'))
        num = self.numerator.doit().as_poly(*sp.symbols('a b c'))
        self.is_equal_ = (mul * self.problem - num).is_zero