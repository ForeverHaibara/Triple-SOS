from functools import partial
from typing import List, Tuple, Dict, Optional, Callable, Any, Union

import sympy as sp
import numpy as np
from sympy import Poly, Expr, Symbol
from sympy.matrices import MutableDenseMatrix as Matrix

from .algebra import SOSBasis, PolyRing, PseudoSMP, PseudoPoly
from ..solution import Solution
from ...utils import MonomialManager
from ...sdp.arithmetic import is_numerical_mat

def _invarraylize(basis: SOSBasis, vec: Matrix, gens: Tuple[Symbol, ...]) -> Poly:
    b = basis._basis
    vec = vec._rep.rep
    rep = {b[i]: v[0] for i, v in vec.items() if v.get(0)}
    if isinstance(basis.algebra, PolyRing):
        return Poly.from_dict(rep, *gens, domain=vec.domain)
    else:
        rep = PseudoSMP.from_dict(rep, len(gens)-1, vec.domain, algebra=basis.algebra)
        return PseudoPoly.new(rep, *gens)

def _as_expr(poly: Union[Poly, PseudoPoly], state_operator: Optional[Callable[[Expr], Expr]] = None) -> Expr:
    if state_operator is None:
        return poly.as_expr()
    if isinstance(poly, Poly):
        return poly.as_expr()
    elif isinstance(poly, Expr):
        return poly
    return poly.as_expr(state_operator=state_operator)


class SolutionSDP(Solution):
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
        qmodule: Dict[Any, Expr],
        qmodule_bases: Dict[Any, MonomialManager],
        ideal: Dict[Any, Expr],
        ideal_bases: Dict[Any, MonomialManager],
        adjoint_operator: Optional[Callable[[Expr], Expr]] = None,
        state_operator: Optional[Callable[[Expr], Expr]] = None,
    ) -> 'SolutionSDP':
        """
        Create SDP solution from decompositions.
        """

        qmodule_expr = _get_qmodule_expr(
            decompositions, poly.gens,
            qmodule=qmodule,
            qmodule_bases=qmodule_bases,
            adjoint_operator=adjoint_operator,
            state_operator=state_operator,
        )

        ideal_expr = _get_ideal_expr(
            eqspace, poly.gens,
            ideal=ideal,
            ideal_bases=ideal_bases,
            adjoint_operator=adjoint_operator,
            state_operator=state_operator,
        )

        is_equal = not _is_numerical(decompositions, eqspace)

        return SolutionSDP(
            problem = _as_expr(poly, state_operator=state_operator) if state_operator is not None else poly,
            solution = sp.Add(qmodule_expr, ideal_expr),
            is_equal = is_equal,
        )

def _get_ideal_expr(
        eqspace: Dict[Any, Matrix],
        gens: List[Symbol],
        ideal: Dict[Any, Expr],
        ideal_bases: Dict[Any, MonomialManager],
        adjoint_operator: Optional[Callable[[Expr], Expr]] = None,
        state_operator: Optional[Callable[[Expr], Expr]] = None,
    ):
        ideal_exprs = []
        for key, vec in eqspace.items():
            expr = _as_expr(ideal[key], state_operator=state_operator)
            vec = _as_expr(_invarraylize(ideal_bases[key], vec, gens), state_operator=state_operator)
            ideal_exprs.append(vec * expr.together())
        if state_operator is not None:
            ideal_exprs = map(lambda x: state_operator(x), ideal_exprs)
        return sp.Add(*ideal_exprs)

def _get_qmodule_expr(
        decompositions: Dict[Any, Tuple[Matrix, Matrix]],
        gens: List[Symbol],
        qmodule: Optional[Dict[Any, Expr]],
        qmodule_bases: Optional[Dict[Any, MonomialManager]],
        adjoint_operator: Optional[Callable[[Expr], Expr]] = None,
        state_operator: Optional[Callable[[Expr], Expr]] = None,
        simplify_poly: Optional[Callable] = None,
    ) -> Expr:
    """
    Convert a {key: (U, S)} dictionary to sum of squares.
    """
    if simplify_poly is None:
        if state_operator is None:
            simplify_poly = _default_simplify_poly
        else:
            simplify_poly = partial(_default_simplify_poly, state_operator=state_operator)

    def compute_cyc_sum_of_squares(coeff, q_module: Expr, poly: Poly) -> Expr:
        """Computes cyclic_sum(coeff * q_module * poly.as_expr()**2),
        and returns a pretty expression."""
        # if isinstance(expr, sp.Add):
        #     expr = sp.UnevaluatedExpr(expr)
        q_primitive = q_module.primitive()
        coeff, q = coeff * q_primitive[0], _as_expr(q_primitive[1], state_operator=state_operator)
        c, expr = simplify_poly(poly)
        coeff = coeff * c**2

        if adjoint_operator is not None:
            expr = adjoint_operator(expr) * (q + adjoint_operator(q)) * expr
            coeff = coeff/2 # since we have doubled the "q"
        else:
            expr = sp.Mul(expr, q, expr)
        return coeff * state_operator(expr) if state_operator is not None else coeff * expr

    exprs = []
    for key, (U, S) in decompositions.items():
        expr = qmodule[key]
        # vecs = [qmodule_bases[key].invarraylize(U[i,:], gens, codeg) for i in range(U.shape[0])]
        vecs = [_invarraylize(qmodule_bases[key], U[i,:].T, gens) for i in range(U.shape[0])]
        vecs = [compute_cyc_sum_of_squares(S[i], expr, vecs[i]) for i in range(U.shape[0])]
        exprs.extend(vecs)

    return sp.Add(*exprs)

def _default_simplify_poly(poly: Poly, bound: int=10000,
        state_operator: Optional[Callable[[Expr], Expr]] = None) -> Tuple[Expr, Expr]:

    """
    Simplify the polynomial. Return c, expr such that poly = c * expr.
    Using factorization would be extremely slow sometimes, e.g. in the case
    of very large integer coefficients, and this function uses a heuristic
    strategy to balance the speed and complexity.
    """
    if isinstance(poly, PseudoPoly):
        c, p = poly.primitive()
        return c, _as_expr(p, state_operator=state_operator)

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


def _is_numerical(decompositions, eqspace):
    for U, S in decompositions.values():
        if is_numerical_mat(U) or is_numerical_mat(S):
            return True
    for U in eqspace.values():
        if is_numerical_mat(U):
            return True
    return False
