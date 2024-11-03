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
        eqvec: Dict[str, sp.Matrix],
        symmetry: MonomialReduction
    ) -> 'SolutionSDP':
        """
        Create SDP solution from decompositions.
        """
        qmodule_expr = _decomp_as_sos(decompositions, poly.gens, symmetry=symmetry)

        ideal_exprs = []
        for key, vec in eqvec.items():
            key = _poly_to_unevaluated(key)
            vec = symmetry.base().invarraylize(vec, poly.gens).as_expr()
            ideal_exprs.append(symmetry.cyclic_sum(vec * key, poly.gens).together())
        ideal_expr = sp.Add(*ideal_exprs)

        return SolutionSDP(
            problem = poly,
            numerator = qmodule_expr + ideal_expr,
            is_equal = not _is_numer_solution(decompositions)
        )

def _poly_to_unevaluated(poly: sp.Poly) -> sp.Expr:
    return poly.as_expr() # if poly.is_monomial else sp.UnevaluatedExpr(poly.as_expr())
    const, fact = poly.factor_list()
    sgn = 1 if const >= 0 else -1
    monomials = [const if sgn > 0 else -const]
    non_monomials = [sgn]
    for p, d in fact:
        if p.is_monomial:
            monomials.append(p.as_expr()**d)
        else:
            non_monomials.append(p.as_expr()**d)
    return sp.Mul(*monomials) * sp.UnevaluatedExpr(sp.Mul(*non_monomials))

def _decomp_as_sos(
        decompositions: Dict[str, Tuple[sp.Matrix, sp.Matrix]],
        gens: List[sp.Symbol],
        symmetry: MonomialReduction,
        factor: bool = True,
    ) -> sp.Expr:
    """
    Convert a {key: (U, S)} dictionary to sum of squares.
    """
    def compute_cyc_sum(expr: sp.Expr) -> sp.Expr:
        # if isinstance(expr, sp.Add):
        #     expr = sp.UnevaluatedExpr(expr)
        return symmetry.cyclic_sum(expr, gens).together()

    exprs = []
    symmetry_half = symmetry.base()
    for key, (U, S) in decompositions.items():
        key = _poly_to_unevaluated(key)
        vecs = [symmetry_half.invarraylize(U[i,:], gens).as_expr() for i in range(U.shape[0])]
        if factor:
            vecs = [_.factor() for _ in vecs]
        vecs = [compute_cyc_sum(S[i] * key * vecs[i]**2) for i in range(U.shape[0])]

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