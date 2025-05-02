from typing import List, Tuple, Dict, Optional, Callable, Any

import sympy as sp
import numpy as np
from sympy import Poly, Expr, Symbol
from sympy.matrices import MutableDenseMatrix as Matrix

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
        poly: Poly,
        decompositions: Dict[Any, Tuple[Matrix, Matrix]],
        eqspace: Dict[Any, Matrix],
        ineq_constraints: Dict[Any, Expr],
        ineq_bases: Dict[Any, MonomialManager],
        ineq_codegrees: Dict[Any, int],
        eq_constraints: Dict[Any, Expr],
        eq_bases: Dict[Any, MonomialManager],
        eq_codegrees: Dict[Any, int],
        cyclic_sum: Optional[Callable[[Expr], Expr]] = None,
        adjoint_operator: Optional[Callable[[Expr], Expr]] = None,
        trace_operator: Optional[Callable[[Expr], Expr]] = None,
    ) -> 'SolutionSDP':
        """
        Create SDP solution from decompositions.
        """
        if cyclic_sum is None:
            cyclic_sum = lambda x: x

        qmodule_expr = _get_qmodule_expr(
            decompositions, poly.gens,
            ineq_constraints=ineq_constraints,
            ineq_bases=ineq_bases,
            ineq_codegrees=ineq_codegrees,
            cyclic_sum=cyclic_sum,
            adjoint_operator=adjoint_operator,
            trace_operator=trace_operator,
        )

        ideal_expr = _get_ideal_expr(
            eqspace, poly.gens,
            eq_constraints=eq_constraints,
            eq_bases=eq_bases,
            eq_codegrees=eq_codegrees,
            cyclic_sum=cyclic_sum,
            adjoint_operator=adjoint_operator,
            trace_operator=trace_operator,
        )

        return SolutionSDP(
            problem = poly,
            solution = qmodule_expr + ideal_expr,
        )

def _get_ideal_expr(
        eqspace: Dict[Any, Matrix],
        gens: List[Symbol],
        eq_constraints: Dict[Any, Expr],
        eq_bases: Dict[Any, MonomialManager],
        eq_codegrees: Dict[Any, int],
        cyclic_sum: Callable[[Expr], Expr],
        adjoint_operator: Optional[Callable[[Expr], Expr]] = None,
        trace_operator: Optional[Callable[[Expr], Expr]] = None,
    ):
        ideal_exprs = []
        for key, vec in eqspace.items():
            codeg = eq_codegrees[key]
            expr = eq_constraints[key].as_expr()
            vec = eq_bases[key].invarraylize(vec, gens, codeg).as_expr()
            ideal_exprs.append(cyclic_sum(vec * expr).together())
        if trace_operator is not None:
            ideal_exprs = map(trace_operator, ideal_exprs)
        return sp.Add(*ideal_exprs)

def _get_qmodule_expr(
        decompositions: Dict[Any, Tuple[Matrix, Matrix]],
        gens: List[Symbol],
        ineq_constraints: Optional[Dict[Any, Expr]],
        ineq_bases: Optional[Dict[Any, MonomialManager]],
        ineq_codegrees: Optional[Dict[Any, int]],
        cyclic_sum: Callable[[Expr], Expr],
        adjoint_operator: Optional[Callable[[Expr], Expr]] = None,
        trace_operator: Optional[Callable[[Expr], Expr]] = None,
        simplify_poly: Optional[Callable] = None,
    ) -> Expr:
    """
    Convert a {key: (U, S)} dictionary to sum of squares.
    """
    if simplify_poly is None:
        simplify_poly = _default_simplify_poly

    def compute_cyc_sum_of_squares(coeff, q_module: Expr, poly: Poly) -> Expr:
        """Computes cyclic_sum(coeff * q_module * poly.as_expr()**2),
        and returns a pretty expression."""
        # if isinstance(expr, sp.Add):
        #     expr = sp.UnevaluatedExpr(expr)
        q_primitive = q_module.primitive()
        coeff, q = coeff * q_primitive[0], q_primitive[1].as_expr()
        c, expr = simplify_poly(poly)
        coeff = coeff * c**2
    
        if adjoint_operator is not None:
            expr = adjoint_operator(expr) * (q + adjoint_operator(q)) * expr
            coeff = coeff/2 # since we have doubled the "q"
        else:
            expr = sp.Mul(expr, q, expr)
        return coeff * cyclic_sum(expr)

    exprs = []
    for key, (U, S) in decompositions.items():
        codeg = ineq_codegrees[key]
        expr = ineq_constraints[key]
        vecs = [ineq_bases[key].invarraylize(U[i,:], gens, codeg) for i in range(U.shape[0])]
        vecs = [compute_cyc_sum_of_squares(S[i], expr, vecs[i]) for i in range(U.shape[0])]
        exprs.extend(vecs)

    if trace_operator is not None:
        exprs = map(trace_operator, exprs)
    return sp.Add(*exprs)

def _default_simplify_poly(poly: Poly, bound: int=10000) -> Tuple[Expr, Expr]:
    """
    Simplify the polynomial. Return c, expr such that poly = c * expr.
    Using factorization would be extremely slow sometimes, e.g. in the case
    of very large integer coefficients, and this function uses a heuristic
    strategy to balance the speed and complexity.
    """

    def _extract_monomials(p: Poly) -> Tuple[Expr, Poly]:
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

    def _standard_form(poly: Poly) -> Poly:
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
