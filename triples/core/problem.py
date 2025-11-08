from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from sympy import Expr, Symbol, Poly, Integer, Rational, Function, Mul, sympify, fraction
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
    """
    Represents an inequality problem:

        Prove expr >= 0
            given {g >= 0 for g in ineq_constraints.keys()}
            and   {h == 0 for h in eq_constraints.keys()}.
    """
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

    def __str__(self) -> str:
        ss = [f"Prove {self.expr} >= 0"]
        if len(self.ineq_constraints):
            ss.append(f"    given inequality constraints:")
            for p, e in self.ineq_constraints.items():
                ss.append(f"        {p} >= 0" + (f"    ({e})" if p.as_expr() != e else ""))
        else:
            ss.append("    given no inequality constraints,")

        if len(self.eq_constraints):
            ss.append(f"    and equality constraints:")
            for p, e in self.eq_constraints.items():
                ss.append(f"        {p} == 0" + (f"    ({e})" if e != 0 and p.as_expr() != e else ""))
        else:
            ss.append("    and no equality constraints.")
        return '\n'.join(ss)

    def __repr__(self) -> str:
        nvars = len(self.free_symbols)
        ineqs, eqs = len(self.ineq_constraints), len(self.eq_constraints)
        poly_info = f' and degree {self.expr.total_degree()}' if isinstance(self.expr, Poly) else ''
        return f'<InequalityProblem of {nvars} variables{poly_info}, with {ineqs} inequality and {eqs} equality constraints>'

    def copy(self) -> 'InequalityProblem':
        problem = InequalityProblem(
            self.expr,
            self.ineq_constraints.copy(),
            self.eq_constraints.copy()
        )
        problem.solution = self.solution
        problem.roots = self.roots
        return problem

    def reduce(self, f, reduction = all) -> Any:
        """Apply a function over self.expr, self.ineq_constraints.keys()
        and self.eq_constraints.keys(), and reduce them by a given rule.
        """
        return reduction(map(f, [
            self.expr, *self.ineq_constraints.keys(), *self.eq_constraints.keys()]))

    @property
    def free_symbols(self):
        return self.reduce(lambda e: set(e.free_symbols), lambda x: set.union(*x))

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


    def get_symbol_signs(self) -> Dict[Symbol, Tuple[Optional[int], Expr]]:
        from .preprocess import get_symbol_signs
        return get_symbol_signs(self)

    def evaluate_complexity(self):
        # this is experimental and heuristic, and should be replaced with better estimation in the future
        nvars = len(self.free_symbols)
        return ProblemComplexity(
            time=nvars**4/81 * (len(self.ineq_constraints)+1)*(len(self.eq_constraints)+1),
            prob=min(.95, 3.1**2/((nvars+.1)**2)),
            length=nvars**4)

    def sum_of_squares(self, configs: dict = {}, time_limit: float = 3600, mode: str = 'fast') -> Solution:
        from .node import _sum_of_squares
        return _sum_of_squares(self, configs, time_limit, mode)

    @property
    def is_homogeneous(self) -> bool:
        return self.reduce(lambda e: e.is_homogeneous, all)

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

        problem = InequalityProblem(expr, ineq_constraints, eq_constraints)
        problem, _ = problem.sqr_free(problem_sqf=False,
            ineq_constraint_sqf=ineq_constraint_sqf, eq_constraint_sqf=eq_constraint_sqf, inplace=True)
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

    def sqr_free(self,
            problem_sqf: bool = False,
            ineq_constraint_sqf: bool = True,
            eq_constraint_sqf: bool = True,
            inplace: bool = False
        ) -> Tuple['InequalityProblem', Expr]:
        if not inplace:
            self = self.copy()

        sqr = Integer(1)
        if problem_sqf:
            c, lst = _sqf_list(self.expr)
            sqr = []
            sqf = c.as_poly(*self.expr.gens)
            for p, d in lst:
                sqr.append(p.as_expr()**(d//2))
                if d % 2 == 1:
                    sqf = sqf*p
            sqr = Mul(*sqr)
            self.expr = sqf
            
        if ineq_constraint_sqf:
            ineq_constraints = dict(_std_ineq_constraints(*item) for item in self.ineq_constraints.items())
        self.ineq_constraints = dict((e, e2) for e, e2 in ineq_constraints.items() if e.total_degree() > 0)

        if eq_constraint_sqf:
            eq_constraints = dict(_std_eq_constraints(*item) for item in self.eq_constraints.items())
        self.eq_constraints = dict((e, e2) for e, e2 in eq_constraints.items() if e.total_degree() > 0)
        return self, sqr

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
        if self.roots is not None:
            return self.roots
        roots = optimize_poly(self.expr, list(self.ineq_constraints), [self.expr] + list(self.eq_constraints),
                    self.expr.gens, return_type='root')
        self.roots = roots
        return self.roots

    def transform(self, transform: Dict[Symbol, Expr], inv_transform: Dict[Symbol, Expr]) -> Tuple['InequalityProblem', Callable]:
        """


        Examples
        --------
        A manual approach to solve the IMO-1983 problem by Ravi substitution:

        >>> from sympy.abc import a, b, c, x, y, z
        >>> from sympy import Function
        >>> F = Function('F')
        >>> problem = InequalityProblem(a**2*b*(a-b)+b**2*c*(b-c)+c**2*a*(c-a),{b+c-a:F(a),c+a-b:F(b),a+b-c:F(c)})
        >>> new_pro, restore = problem.transform({a:y+z,b:z+x,c:x+y}, {x:(b+c-a)/2,y:(c+a-b)/2, z:(a+b-c)/2})
        >>> new_pro.expr.expand(), new_pro.ineq_constraints # doctest: +NORMALIZE_WHITESPACE
        (2*x**3*z - 2*x**2*y*z + 2*x*y**3 - 2*x*y**2*z - 2*x*y*z**2 + 2*y*z**3,
         {2*x: F(a), 2*y: F(b), 2*z: F(c)})

        As we find a solution (sympy Expr) to the transformed problem, we use `restore` to 
        transform it back to the original problem.
        >>> sol = (F(a)*F(c)*(x-y)**2 + F(b)*F(a)*(y-z)**2 + F(c)*F(b)*(z-x)**2)/2
        >>> (sol.xreplace({F(a): 2*x, F(b): 2*y, F(c): 2*z}) - new_pro.expr).expand()
        0
        >>> restore(sol) # doctest: +SKIP
        (-a + b)**2*F(a)*F(c)/2 + (a - c)**2*F(b)*F(c)/2 + (-b + c)**2*F(a)*F(b)/2
        >>> (restore(sol).xreplace({F(a):b+c-a, F(b):c+a-b, F(c):a+b-c}) - problem.expr).expand()
        0
        """
        src_dicts = [{self.expr:1}, self.ineq_constraints, self.eq_constraints]
        dst_dicts = [{}, {}, {}]
        if isinstance(self.expr, Poly):
            new_symbols = tuple(sorted(list(inv_transform.keys()), key=lambda x:x.name))
            symbols = tuple([_ for _ in self.expr.gens if (not _ in transform)]) + new_symbols
        for src, dst in zip(src_dicts, dst_dicts):
            for p, e in src.items():
                if isinstance(p, Expr):
                    p = p.xreplace(transform)
                else:
                    p, denom_list = _polysubs_frac(p, transform, symbols)
                    for d, mul in denom_list:
                        e *= d.as_expr()**(((mul+1)//2)*2)
                        if mul % 2 == 1:
                            p = p*d
                dst[p] = e

        problem = InequalityProblem(next(iter(dst_dicts[0].keys())), dst_dicts[1], dst_dicts[2])

        def restore(x: Optional[Expr]) -> Optional[Expr]:
            if x is None:
                return None
            return x.xreplace(inv_transform) / next(iter(dst_dicts[0].values()))
        return problem, restore


class ProblemComplexity:
    """
    time  : `E(time)` expected time to solve the problem.
    prob  : `E(success prob)` assuming the problem is correct.
    length: `E(length of solution | success)` expected length of the solution if the problem is solved.
    status: Status code or timestamp when evaluated.

    #### Comparison of ProblemComplexity
    Consider solving a problem by two methods, A and B, with time t1, t2 and success probability p1, p2.
    There are two choices: 1. try A first, and if it fails, try B; 2. try B first, and if it fails, try A.
    The expected time cost using A->B is `t1 + t2(1 - p1)`, and the expected time cost using B->A is `t2 + t1(1 - p2)`.
    Then:

        `t1 + t2(1 - p1) < t2 + t1(1 - p2)     <=>     t1/p1 < t2/p2`
    """
    EPS = 1e-14
    time: float = 0
    prob: float = 0
    length: float = 0
    status: int = 0
    def __init__(self, time, prob, length=0, status=0):
        self.time = time
        self.prob = prob
        self.length = length
        self.status = status

    def __str__(self) -> str:
        return f"{{time: {self.time:.2f}, prob: {self.prob:.2f}, length: {self.length:.2f}}}"

    def __repr__(self) -> str:
        return self.__str__()

    def copy(self) -> 'ProblemComplexity':
        return ProblemComplexity(self.time, self.prob, self.length, self.status)

    def __and__(a, b) -> 'ProblemComplexity':
        return ProblemComplexity(
            a.time + b.time, a.prob * b.prob, a.length + b.length
        )

    def __or__(a, b) -> 'ProblemComplexity':
        p = 1 - (1 - a.prob) * (1 - b.prob)
        return ProblemComplexity(
            a.time + b.time * (1 - a.prob), p,
            (a.length * a.prob + b.length * (1 - a.prob) * b.prob)/max(p, ProblemComplexity.EPS)
        )

    def __gt__(a, b) -> bool:
        return a.time / max(a.prob, ProblemComplexity.EPS) >  b.time / max(b.prob, ProblemComplexity.EPS)
    def __lt__(a, b) -> bool:
        return a.time / max(a.prob, ProblemComplexity.EPS) <  b.time / max(b.prob, ProblemComplexity.EPS)
    def __ge__(a, b) -> bool:
        return a.time / max(a.prob, ProblemComplexity.EPS) >= b.time / max(b.prob, ProblemComplexity.EPS)
    def __le__(a, b) -> bool:
        return a.time / max(a.prob, ProblemComplexity.EPS) <= b.time / max(b.prob, ProblemComplexity.EPS)

    def __eq__(a, b) -> bool:
        # todo: is it well-defined?
        return a.time == b.time and a.prob == b.prob and a.length == b.length


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


def _polysubs_frac(poly: Poly, transform: Dict[Symbol, Expr], new_gens: List[Symbol]) -> Tuple[Poly, List[Tuple[Poly, int]]]:
    """
    Substitute the variables in the polynomial with a given transform.
    The result can be written in the form of `new_poly/(Mul(*denom_list) * expr)`
    where `denom_list` is a list of (sqr-free) polynomials, and `expr` is a square expression.
    """
    frac = fraction(poly.as_expr().xreplace(transform).together())
    numer = frac[0]

    denom = Mul.make_args(frac[1])
    denom_list = [0] * len(denom)
    for i, arg in enumerate(denom):
        if arg.is_Pow:
            denom_list[i] = (arg.base, arg.exp)
        else:
            denom_list[i] = (arg, 1)
    numer = Poly(numer, new_gens)
    denom_list = [(Poly(d, new_gens), mul) for d, mul in denom_list]
    return numer, denom_list