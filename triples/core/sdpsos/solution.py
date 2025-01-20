from typing import List, Tuple, Dict, Optional

import sympy as sp
from sympy.simplify import signsimp

from ...utils import MonomialManager, SolutionSimple



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
        symmetry: MonomialManager,
        ineq_constraints: Optional[Dict[sp.Poly, sp.Expr]] = None,
        eq_constraints: Optional[Dict[sp.Poly, sp.Expr]] = None
    ) -> 'SolutionSDP':
        """
        Create SDP solution from decompositions.
        """
        qmodule_expr = _decomp_as_sos(decompositions, poly.total_degree(), poly.gens,
                            symmetry=symmetry, ineq_constraints=ineq_constraints)

        ideal_exprs = []
        for key, vec in eqvec.items():
            codeg = poly.total_degree() - key.total_degree()
            key = _get_expr_from_dict(eq_constraints, key)
            vec = symmetry.base().invarraylize(vec, poly.gens, codeg).as_expr()
            ideal_exprs.append(symmetry.cyclic_sum(vec * key, poly.gens).together())
        ideal_expr = sp.Add(*ideal_exprs)

        return SolutionSDP(
            problem = poly,
            solution = qmodule_expr + ideal_expr,
            ineq_constraints = ineq_constraints,
            eq_constraints = eq_constraints,
            is_equal = not _is_numer_solution(decompositions)
        )

def _decomp_as_sos(
        decompositions: Dict[str, Tuple[sp.Matrix, sp.Matrix]],
        degree: int,
        gens: List[sp.Symbol],
        symmetry: MonomialManager,
        ineq_constraints: Optional[Dict[sp.Poly, sp.Expr]] = None,
        factor: bool = True,
    ) -> sp.Expr:
    """
    Convert a {key: (U, S)} dictionary to sum of squares.
    """
    def compute_cyc_sum(expr: sp.Expr) -> sp.Expr:
        # if isinstance(expr, sp.Add):
        #     expr = sp.UnevaluatedExpr(expr)
        return symmetry.cyclic_sum(signsimp(expr), gens).together()

    exprs = []
    symmetry_half = symmetry.base()
    for key, (U, S) in decompositions.items():
        codeg = (degree - key.total_degree()) // 2
        key = _get_expr_from_dict(ineq_constraints, key)
        vecs = [symmetry_half.invarraylize(U[i,:], gens, codeg).as_expr() for i in range(U.shape[0])]
        if factor:
            vecs = [_.factor() for _ in vecs]
        vecs = [compute_cyc_sum(S[i] * key * vecs[i]**2) for i in range(U.shape[0])]

        exprs.extend(vecs)
    return sp.Add(*exprs)

def _get_expr_from_dict(d: Dict, k: sp.Poly):
    if d is None:
        return k.as_expr()
    v = d.get(k)
    return v if v is not None else k.as_expr()

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