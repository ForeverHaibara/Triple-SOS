from typing import Dict, Tuple, Optional, Union, Callable
from sympy import Expr, Symbol, Poly, Integer, Rational, Function, Mul, sympify
from sympy import __version__ as SYMPY_VERSION
from sympy.combinatorics.perm_groups import Permutation, PermutationGroup
from sympy.core.symbol import uniquely_named_symbol
from sympy.core.function import AppliedUndef
from sympy.external.importtools import version_tuple

from ..utils import optimize_poly, Root, identify_symmetry_from_lists, Solution

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

    counter_examples = None
    solution = None

    roots = None

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

    def copy(self) -> 'InequalityProblem':
        problem = InequalityProblem(
            self.expr,
            self.ineq_constraints.copy(),
            self.eq_constraints.copy()
        )
        problem.solution = self.solution
        problem.roots = self.roots
        return problem

    @property
    def free_symbols(self):
        return set.union(
            set(self.expr.free_symbols), 
            *[set(e.free_symbols) for e in self.ineq_constraints.keys()],
            *[set(e.free_symbols) for e in self.eq_constraints.keys()]
        )

    def extract_constraints(self, symbols: Union[Symbol, Tuple[Symbol]]) \
            -> Tuple[Dict[Expr, Expr], Dict[Expr, Expr], Dict[Expr, Expr], Dict[Expr, Expr]]:
        if isinstance(symbols, Symbol):
            symbols = {symbols}
        symbols = set(symbols)

        ineqs = [{}, {}]
        eqs = [{}, {}]
        for src, dst in [(self.ineq_constraints, ineqs), (self.eq_constraints, eqs)]:
            for p, e in src.items():
                dst[int(bool(p.free_symbols & symbols))][p] = e

        return ineqs[1], eqs[1], ineqs[0], eqs[0]


    def get_symbol_signs(self):
        from .preprocess import get_symbol_signs
        return get_symbol_signs(self)

    def evaluate_complexity(self):
        ...

    def sum_of_squares(self, configs: dict = {}, time: float = 3600, mode: str = 'fast') -> Solution:
        from .node import _sum_of_squares
        return _sum_of_squares(self, configs, time, mode)

    @property
    def is_homogeneous(self) -> bool:
        return self.expr.is_homogeneous and \
            all(e.is_homogeneous for e in self.ineq_constraints.keys()) and \
            all(e.is_homogeneous for e in self.eq_constraints.keys())

    def polylize(self,
        ineq_constraint_sqf: bool = True,
        eq_constraint_sqf: bool = True,
    ) -> 'InequalityProblem':
        problem = self
        expr, ineq_constraints, eq_constraints = \
            problem.expr, problem.ineq_constraints.copy(), problem.eq_constraints.copy()
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

        problem = InequalityProblem(expr, ineq_constraints, eq_constraints)
        return problem

    def homogenize(self) -> Tuple['InequalityProblem', Optional[Symbol]]:
        if not self.is_homogeneous:
            hom = uniquely_named_symbol('1', tuple(self.free_symbols))
            expr = self.expr.homogenize(hom)
            ineqs = {e.homogenize(hom): v for e, v in self.ineq_constraints.items()}
            ineqs[Poly(hom, expr.gens)] = hom # homogenizer = 1 >= 0
            eqs = {e.homogenize(hom): v for e, v in self.eq_constraints.items()}

            new_problem = InequalityProblem(expr, ineqs, eqs)
            if self.roots is not None:
                new_problem.roots = [Root(r.root + (Integer(1),), r.domain, r.rep + (r.domain.one,)) for r in self.roots]
            return new_problem, hom
        return self, None

    def identify_symmetry(self) -> PermutationGroup:
        return identify_symmetry_from_lists(
            [[self.expr], list(self.ineq_constraints), list(self.eq_constraints)]
        )

    def wrap_constraints(self, symmetry: Optional[PermutationGroup]=None) -> Tuple['InequalityProblem', Callable]:
        gens = self.expr.gens
        i2g, e2h, g2i, h2e = _get_constraints_wrapper(
            gens, self.ineq_constraints, self.eq_constraints, symmetry
        )
        problem = self.copy()
        problem.ineq_constraints = i2g
        problem.eq_constraints = e2h
        def restoration(x):
            if x is None: return None
            return x.xreplace(g2i).xreplace(h2e)
        return problem, restoration

    def find_roots(self):
        """Find the equality cases of the problem heuristically."""
        roots = optimize_poly(self.expr, list(self.ineq_constraints), [self.expr] + list(self.eq_constraints),
                    self.expr.gens, return_type='root')
        self.roots = roots
        return self.roots


def _get_constraints_wrapper(symbols: Tuple[int, ...],
    ineq_constraints: Dict[Poly, Expr], eq_constraints: Dict[Poly, Expr],
    perm_group: Optional[PermutationGroup]=None):
    if perm_group is None:
        # trivial group
        perm_group = PermutationGroup(Permutation(list(range(len(symbols)))))

    def _get_mask(symbols, dlist):
        # only reserve symbols with degree > 0, this reduces time complexity greatly
        return tuple(s for d, s in zip(dlist, symbols) if d != 0)

    def _get_counter(name='_G'):
        # avoid duplicate function counters
        k = len(name)
        exprs = [e for e in ineq_constraints.values()] + [e for e in eq_constraints.values()]
        names = [[f.name for f in e.find(AppliedUndef)] for e in exprs]
        names = [item for sublist in names for item in sublist]
        names = [n[k:] for n in names if n.startswith(name)]
        digits = [int(n) for n in names if n.isdigit()]
        return max(digits, default=-1) + 1

    def _get_dicts(constraints, name='_G', counter=None):
        dt = dict()
        inv = dict()
        rep_dict = dict((p.rep, v) for p, v in constraints.items())
        if counter is None:
            counter = _get_counter(name)

        for base in constraints.keys():
            if base.rep in dt:
                continue
            dlist = base.degree_list()
            for p in perm_group.elements:
                invorder = p.__invert__()(symbols)
                permed_base = base.reorder(*invorder).rep
                permed_expr = rep_dict.get(permed_base)
                if permed_expr is None:
                    raise ValueError("Given constraints are not symmetric with respect to the permutation group.")
                compressed = _get_mask(p(symbols), dlist)
                value = Function(name + str(counter))(*compressed)
                dt[permed_base] = value
                inv[value] = permed_expr
            counter += 1
        dt = dict((Poly.new(k, *symbols), v) for k, v in dt.items())
        return dt, inv
    i2g, g2i = _get_dicts(ineq_constraints, name='_G')
    e2h, h2e = _get_dicts(eq_constraints, name='_H')
    return i2g, e2h, g2i, h2e