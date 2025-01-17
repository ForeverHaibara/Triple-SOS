import sympy as sp
from ..utils import (
    Coeff, 
    radsimp, sum_y_exprs, rationalize_func, quadratic_weighting, zip_longest, intervals,
    StructuralSOSError, PolynomialNonpositiveError, PolynomialUnsolvableError
)

from ...symsos import prove_univariate
from ....utils.roots.rationalize import (
    nroots, rationalize, rationalize_bound, univariate_intervals
)
from ....sdp import congruence
from ....utils import CyclicExpr, CyclicSum, CyclicProduct, SymmetricSum, SymmetricProduct

a, b, c, d = sp.symbols("a b c d")

def SymSum(expr):
    return SymmetricSum(expr, (a,b,c,d))

def CycSum(expr):
    return CyclicSum(expr, (a,b,c,d))