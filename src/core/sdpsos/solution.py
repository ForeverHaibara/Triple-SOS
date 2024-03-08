from typing import List, Tuple, Dict

import sympy as sp

from .utils import is_numer_matrix
from ...utils import (
    MonomialReduction, MonomialCyclic,
    CyclicSum, CyclicProduct, is_cyclic_expr,
    SolutionSimple
)

class SolutionSDP(SolutionSimple):
    method = 'SDPSOS'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # @property
    # def is_equal(self):
    #     return True

    @classmethod
    def from_decompositions(self,
        poly: sp.Poly,
        decompositions: Dict[str, Tuple[sp.Matrix, sp.Matrix]],
        **options
    ) -> 'SolutionSDP':
        """
        Create SDP solution from decompositions.
        """
        sos_expr = _decomp_as_sos(decompositions, poly.gens, **options)
        return SolutionSDP(
            problem = poly,
            numerator = sos_expr,
            is_equal = not _is_numer_solution(decompositions)
        )


def monomial_to_expr(monom: Tuple[int, ...], gens: List[sp.Symbol]) -> sp.Expr:
    """
    Convert a monomial to an expression.
    See also in sp.polys.monomials.Monomial.as_expr.
    """
    return sp.Mul(*[gen**exp for gen, exp in zip(gens, monom)])


def _decomp_as_sos(
        decompositions: Dict[str, Tuple[sp.Matrix, sp.Matrix]],
        gens: List[sp.Symbol],
        factor: bool = True,
        **options
    ) -> sp.Expr:
    """
    Convert a {key: (U, S)} dictionary to sum of squares.
    """
    exprs = []
    option = MonomialReduction.from_options(**options)
    option_half = option.base()
    for key, (U, S) in decompositions.items():
        monomial = eval(key)
        monomial_expr = monomial_to_expr(monomial, gens)
        vecs = [option_half.invarraylize(U[i,:], gens).as_expr() for i in range(U.shape[0])]
        if factor:
            vecs = [_.factor() for _ in vecs]
        vecs = [option.cyclic_sum(S[i] * monomial_expr * vecs[i]**2, gens) for i in range(U.shape[0])]

        exprs.extend(vecs)
    return sp.Add(*exprs)


def _as_cyc(multiplier, x):
    """
    Beautifully convert CyclicSum(multiplier * x).
    """
    a, b, c = sp.symbols('a b c')
    if x.is_Pow:
        # Example: (p(a-b))^2 == p((a-b)^2)
        if isinstance(x.base, CyclicProduct) and x.exp > 1:
            x = CyclicProduct(x.base.args[0] ** x.exp)
    elif x.is_Mul:
        args = list(x.args)
        flg = False
        for i in range(len(args)):
            if args[i].is_Pow and isinstance(args[i].base, CyclicProduct):
                args[i] = CyclicProduct(args[i].base.args[0] ** args[i].exp)
                flg = True
        if flg: # has been updated
            x = sp.Mul(*args)

    if is_cyclic_expr(x, (a,b,c)):
        if multiplier == 1 or multiplier == CyclicProduct(a):
            return 3 * multiplier * x
        return CyclicSum(multiplier) * x
    elif multiplier == CyclicProduct(a):
        return multiplier * CyclicSum(x)
    return CyclicSum(multiplier * x)


def _is_numer_solution(decompositions: Dict[str, Tuple[sp.Matrix, sp.Matrix]]) -> bool:
    """
    Check whether the solution is a numerical solution.

    Parameters
    ----------
    decompositions : Dict[(str, Tuple[sp.Matrix, sp.Matrix])]
        The decompositions of the symmetric matrices.

    Returns
    -------
    bool
        Whether the solution is a numerical solution.
    """
    for _, (U, S) in decompositions.items():
        if is_numer_matrix(U) or is_numer_matrix(S):
            return True
    return False