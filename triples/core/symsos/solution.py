import sympy as sp
from sympy.core.singleton import S

from ...utils import Solution, SolutionSimple


class SolutionSymmetric(SolutionSimple):
    method = 'SymmetricSOS'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def is_equal(self):
        return True

    @classmethod
    def _extract_nonnegative_exprs(cls, expr: sp.Expr, func_name: str = "_G"):
        """
        Raw output of SymmetricSOS might assume nonnegativity of some symbols,
        we extract these symbols and replace them with _F(x) for further processing.
        This is not intended to be used by end users.

        TODO: Move this to SolutionSimple???
        """
        from ..structsos.solution import SolutionStructural
        return SolutionStructural._extract_nonnegative_exprs(expr, func_name)