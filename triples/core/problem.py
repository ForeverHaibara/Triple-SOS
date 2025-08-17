from typing import Dict, Tuple
from sympy import Expr, Symbol, Poly, Integer, sympify
from sympy import __version__ as SYMPY_VERSION
from sympy.external.importtools import version_tuple

from ..utils import identify_symmetry_from_lists

# fix the bug in sqf_list before 1.13.0
# https://github.com/sympy/sympy/pull/26182
if tuple(version_tuple(SYMPY_VERSION)) >= (1, 13):
    _sqf_list = lambda p: p.sqf_list()
else:
    _sqf_list = lambda p: p.factor_list() # it would be slower, but correct


def _std_ineq_constraints(p: Poly, e: Expr) -> Tuple[Poly, Expr]:
    if p.is_zero: return p, e
    c, lst = _sqf_list(p)
    ret = Integer(1 if c > 0 else -1).as_poly(*p.gens, domain=p.domain)
    e = e / (c if c > 0 else -c)
    for q, d in lst:
        if d % 2 == 1:
            ret *= q
        e = e / q.as_expr()**(d - d%2)
    return ret, e

def _std_eq_constraints(p: Poly, e: Expr) -> Tuple[Poly, Expr]:
    if p.is_zero: return p, e
    c, lst = _sqf_list(p)
    ret = Integer(1 if c > 0 else -1).as_poly(*p.gens, domain=p.domain)
    e = e / c
    max_d = Integer(max(1, *(d for q, d in lst)))
    for q, d in lst:
        ret *= q
        e = e * q.as_expr()**(max_d - d)
    if max_d != 1:
        e = Pow(e, 1/max_d, evaluate=False)
    if c < 0:
        e = e.__neg__()
    return ret, e


class InequalityProblem:
    _is_commutative = True
    _is_polynomial = None

    counter_example = None
    solution = None

    def __init__(self,
        expr: Expr,
        ineq_constraints: Dict[Expr, Expr] = {},
        eq_constraints: Dict[Expr, Expr] = {}
    ):
        expr = sympify(expr)
        if not isinstance(ineq_constraints, dict):
            ineq_constraints = {e: e for e in ineq_constraints}
        if not isinstance(eq_constraints, dict):
            eq_constraints = {e: e for e in eq_constraints}
        ineq_constraints = dict((sympify(e), sympify(e2).as_expr()) for e, e2 in ineq_constraints.items())
        eq_constraints = dict((sympify(e), sympify(e2).as_expr()) for e, e2 in eq_constraints.items())

        self.expr = expr
        self.ineq_constraints = ineq_constraints
        self.eq_constraints = eq_constraints

    def copy(self):
        return InequalityProblem(
            self.expr,
            self.ineq_constraints.copy(),
            self.eq_constraints.copy()
        )

    @property
    def free_symbols(self):
        return set.union(
            set(self.expr.free_symbols), 
            *[set(e.free_symbols) for e in self.ineq_constraints.keys()],
            *[set(e.free_symbols) for e in self.eq_constraints.keys()]
        )

    def evaluate_complexity(self):
        ...

    def sum_of_squares(self, configs):
        from .node import _sum_of_squares
        return _sum_of_squares(self, configs)

    def polylize(self,
        ineq_constraint_sqf: bool = True,
        eq_constraint_sqf: bool = True,
    ):
        problem = self.copy()
        expr, ineq_constraints, eq_constraints = \
            problem.expr, problem.ineq_constraints, problem.eq_constraints
        symbols = self.free_symbols
        
        if len(symbols) == 0: # and len(original_symbols) == 0:
            symbols = {Symbol('x')}
        symbols = tuple(sorted(list(symbols), key=lambda x: x.name))
        expr = Poly(expr.doit(), *symbols)
        ineq_constraints = dict((Poly(e.doit(), *symbols), e2) for e, e2 in ineq_constraints.items())
        eq_constraints = dict((Poly(e.doit(), *symbols), e2) for e, e2 in eq_constraints.items())

        if ineq_constraint_sqf:
            ineq_constraints = dict(_std_ineq_constraints(*item) for item in ineq_constraints.items())
        ineq_constraints = dict((e, e2) for e, e2 in ineq_constraints.items() if e.total_degree() > 0)

        if eq_constraint_sqf:
            eq_constraints = dict(_std_eq_constraints(*item) for item in eq_constraints.items())
        eq_constraints = dict((e, e2) for e, e2 in eq_constraints.items() if e.total_degree() > 0)
        
        problem.expr, problem.ineq_constraints, problem.eq_constraints = \
            expr, ineq_constraints, eq_constraints
        return problem

    def identify_symmetry(self):
        return identify_symmetry_from_lists(
            [[self.expr], list(self.ineq_constraints), list(self.eq_constraints)]
        )