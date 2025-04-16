from typing import List, Tuple, Dict, Optional, Callable

import sympy as sp
import numpy as np

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
        is_zz_qq_mat = lambda x: x._rep.domain.is_ZZ or x._rep.domain.is_QQ
        factor = all(is_zz_qq_mat(U) and is_zz_qq_mat(S) for U, S in decompositions.values())
        factor = factor and (poly.domain.is_ZZ or poly.domain.is_QQ)

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
        simplify_poly: Optional[Callable] = None,
    ) -> sp.Expr:
    """
    Convert a {key: (U, S)} dictionary to sum of squares.
    """
    if simplify_poly is None:
        simplify_poly = _default_simplify_poly
    def compute_cyc_sum_of_squares(coeff, key_primitive, poly: sp.Poly) -> sp.Expr:
        """Computes symmetry.cyclic_sum(coeff * key_primitive[0] * key_primitive[1] * poly.as_expr()**2,
        and returns a pretty expression."""
        # if isinstance(expr, sp.Add):
        #     expr = sp.UnevaluatedExpr(expr)
        coeff, key = coeff * key_primitive[0], key_primitive[1]
        c, expr = simplify_poly(poly)
        coeff = coeff * c**2
        expr = key * expr**2
        return coeff * symmetry.cyclic_sum(expr, gens)

    exprs = []
    symmetry_half = symmetry.base()
    for key, (U, S) in decompositions.items():
        codeg = (degree - key.total_degree()) // 2
        key = _get_expr_from_dict(ineq_constraints, key)
        key_primitive = key.primitive()
        vecs = [symmetry_half.invarraylize(U[i,:], gens, codeg) for i in range(U.shape[0])]
        vecs = [compute_cyc_sum_of_squares(S[i], key_primitive, vecs[i]) for i in range(U.shape[0])]

        exprs.extend(vecs)
    return sp.Add(*exprs)

def _get_expr_from_dict(d: Dict, k: sp.Poly):
    if d is None:
        return k.as_expr()
    v = d.get(k)
    return v if v is not None else k.as_expr()

def _default_simplify_poly(poly: sp.Poly, bound: int=10000) -> Tuple[sp.Expr, sp.Expr]:
    """Simplify the polynomial. Return c, expr such that poly = c * expr."""

    def _extract_monomials(p: sp.Poly) -> Tuple[sp.Expr, sp.Poly]:
        monoms = p.monoms()
        if len(monoms) == 0:
            return sp.Integer(1), p
        monoms = np.array(monoms, dtype=int)
        d = np.min(monoms, axis=0)

        if not np.any(d):
            return sp.Integer(1), p

        monoms = monoms - d.reshape(1, -1)
        new_monoms = [tuple(_) for _ in monoms.tolist()]
        rep = dict(zip(new_monoms, p.rep.coeffs()))
        rep = p.rep.from_dict(rep, p.rep.lev, p.rep.dom)
        m = sp.Mul(*(g**i for g, i in zip(p.gens, d.flatten().tolist())))
        return m, poly.new(rep, *p.gens)

    def _standard_form(poly: sp.Poly) -> sp.Poly:
        if poly.domain.is_ZZ or poly.domain.is_QQ:
            if all(abs(_.denominator) <= bound for _ in poly.coeffs()):
                c, parts = poly.factor_list()
                exprs = [p.as_expr()**d for p, d in parts]
                return c, sp.Mul(*exprs)
    
        c, poly = poly.primitive()
        if poly.LC() < 0:
            c, poly = -c, -poly
        return c, poly.as_expr()

        # return sp.Integer(1), poly.as_expr()

    m, p = _extract_monomials(poly)
    c, p = _standard_form(p)
    return c, m * p

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