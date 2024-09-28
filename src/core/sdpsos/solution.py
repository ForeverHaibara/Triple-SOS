from typing import List, Tuple, Dict

import sympy as sp
from sympy.simplify import signsimp

from ...utils import MonomialReduction, SolutionSimple

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
        symmetry: MonomialReduction
    ) -> 'SolutionSDP':
        """
        Create SDP solution from decompositions.
        """
        sos_expr = _decomp_as_sos(decompositions, poly.gens, symmetry=symmetry)
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
        symmetry: MonomialReduction,
        factor: bool = True,
    ) -> sp.Expr:
    """
    Convert a {key: (U, S)} dictionary to sum of squares.
    """
    exprs = []
    symmetry_half = symmetry.base()
    for key, (U, S) in decompositions.items():
        monomial = eval(key)
        monomial_expr = monomial_to_expr(monomial, gens)
        vecs = [symmetry_half.invarraylize(U[i,:], gens).as_expr() for i in range(U.shape[0])]
        if factor:
            vecs = [_.factor() for _ in vecs]
        vecs = [symmetry.cyclic_sum(signsimp(S[i] * monomial_expr * vecs[i]**2), gens) for i in range(U.shape[0])]

        exprs.extend(vecs)
    return sp.Add(*exprs)


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

    def is_numer_matrix(M: sp.Matrix) -> bool:
        """
        Check whether a matrix contains sp.Float.
        """
        return any(not isinstance(v, sp.Rational) for v in M)

    for _, (U, S) in decompositions.items():
        if is_numer_matrix(U) or is_numer_matrix(S):
            return True
    return False