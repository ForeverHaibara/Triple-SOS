from functools import wraps
from typing import (
    Dict, List, Tuple, Set, Optional, Union, Iterable, Callable,
    Any, TypeVar, Generic
)
from sympy import (
    Basic, Expr, Symbol, Dummy, Poly, Integer, Rational, Function, Mul, Pow,
    signsimp, fraction
)
from sympy.combinatorics.perm_groups import Permutation, PermutationGroup
from sympy.core.symbol import uniquely_named_symbol
from sympy.core.sympify import sympify, CantSympify
from sympy.core.function import AppliedUndef
from sympy.polys.polyerrors import BasePolynomialError

from .complexity import ProblemComplexity
from .dispatch import (
    _dtype_free_symbols, _dtype_gens, _dtype_is_zero, _dtype_convert,
    _dtype_is_homogeneous, _dtype_homogenize, _dtype_sqf_list, _dtype_make_reorder_func
)
from ..utils import optimize_poly, Root, RootList
from ..utils.monomials import (
    verify_closure, _identify_symmetry_from_blackbox, identify_symmetry_from_lists
)


class NonPolynomialError(BasePolynomialError):
    pass

T = TypeVar('T')


class InequalityProblem(Generic[T]):
    """
    Represents an inequality problem:

        Prove expr >= 0
            given {g >= 0 for g in ineq_constraints.keys()}
            and   {h == 0 for h in eq_constraints.keys()}.
    """
    expr: T
    ineq_constraints: Dict[T, Expr]
    eq_constraints: Dict[T, Expr]

    _is_commutative = True
    _is_polynomial = False

    counter_examples: Optional[RootList] = None
    solution: Optional[Expr] = None

    roots: Optional[RootList] = None

    # REPR_LATEX_DELIM_L = "$\\displaystyle "
    # REPR_LATEX_DELIM_R = "$"
    REPR_LATEX_DELIM_L = "$$"
    REPR_LATEX_DELIM_R = "$$"
    REPR_LATEX_ALIGN_AT = 1

    def __new__(cls,
        expr: T,
        ineq_constraints: Union[Dict[T, Expr], Iterable[T]] = {},
        eq_constraints: Union[Dict[T, Expr], Iterable[T]] = {}
    ):
        def _try_sympify(expr):
            if isinstance(expr, CantSympify):
                return expr
            return sympify(expr)
        expr = _try_sympify(expr)
        if not isinstance(ineq_constraints, dict):
            ineq_constraints = {e: e for e in ineq_constraints}
        if not isinstance(eq_constraints, dict):
            eq_constraints = {e: e for e in eq_constraints}
        ineq_constraints = dict((_try_sympify(e), _try_sympify(e2).as_expr()) for e, e2 in ineq_constraints.items())
        eq_constraints = dict((_try_sympify(e), _try_sympify(e2).as_expr()) for e, e2 in eq_constraints.items())

        return cls.new(expr, ineq_constraints, eq_constraints)

    @classmethod
    def new(cls,
        expr: T,
        ineq_constraints: Dict[T, Expr] = {},
        eq_constraints: Dict[T, Expr] = {}
    ) -> 'InequalityProblem':
        """Initialization of objects without sanity checks."""
        obj = object.__new__(cls)
        obj.expr = expr
        obj.ineq_constraints = ineq_constraints
        obj.eq_constraints = eq_constraints
        return obj

    def __str__(self) -> str:
        ss = [f"[Problem] Prove that:\n    {self.expr} >= 0"]
        if len(self.ineq_constraints):
            ss.append(f"given inequality constraints:")
            for p, e in self.ineq_constraints.items():
                ss.append(f"    {p} >= 0" + (f"    ({e})" if p.as_expr() != e else ""))
        else:
            ss.append("given no inequality constraints,")

        if len(self.eq_constraints):
            ss.append(f"and equality constraints:")
            for p, e in self.eq_constraints.items():
                ss.append(f"    {p} == 0" + (f"    ({e})" if e != 0 and p.as_expr() != e else ""))
        else:
            ss.append("and no equality constraints.")

        if self.solution is not None:
            ss.append(f"[Solution]\n    LHS = {self.solution}")
        return '\n'.join(ss)

    def __repr__(self) -> str:
        nvars = len(self.free_symbols)
        ineqs, eqs = len(self.ineq_constraints), len(self.eq_constraints)
        poly_info = f' and degree {self.expr.total_degree()}' if isinstance(self.expr, Poly) else ''
        proved = ""
        if self.solution is not None:
            proved = " (Solved)"
        elif self.counter_examples is not None:
            proved = " (Disproved)"
        return f'<InequalityProblem of {nvars} variables{poly_info}, with {ineqs} inequality and {eqs} equality constraints{proved}>'

    def _repr_latex_(self):
        from sympy import latex
        delim_l, delim_r = self.REPR_LATEX_DELIM_L, self.REPR_LATEX_DELIM_R
        ands = ["&" if self.REPR_LATEX_ALIGN_AT == i else "" for i in range(4)]
        ss = [f"**Problem** Prove that:\n\n{delim_l}{latex(self.expr)} \\geq 0{delim_r}"]
        if len(self.ineq_constraints):
            ss.append(f"given inequality constraints:")
            ss.append(delim_l + "\\begin{aligned}" + "\\\\\n ".join([
                f"{ands[0]} {latex(p)} {ands[1]}\\geq 0" + \
                        (f"{ands[2]} \\qquad {ands[3]} ({latex(e)})" if p.as_expr() != e else "")
                    for p, e in self.ineq_constraints.items()
            ]) + "\\end{aligned}" + delim_r)
        else:
            ss.append("given no inequality constraints,")

        if len(self.eq_constraints):
            ss.append(f"and equality constraints:")
            ss.append(delim_l + "\\begin{aligned}" + "\\\\\n ".join([
                f"{ands[0]} {latex(p)} {ands[1]}= 0" + \
                        (f"{ands[2]} \\qquad {ands[3]} ({latex(e)})" if p.as_expr() != e else "")
                    for p, e in self.eq_constraints.items()
            ]) + "\\end{aligned}" + delim_r)
        else:
            ss.append("and no equality constraints.")
        if self.solution is not None:
            ss.append(f"**Solution**\n\n{delim_l}\\text{{LHS}}={latex(self.solution)}{delim_r}")
        return '\n\n'.join(ss)


    def copy_new(self,
        expr: T,
        ineq_constraints: Dict[T, Expr] = {},
        eq_constraints: Dict[T, Expr] = {}
    ) -> 'InequalityProblem':
        """
        Return a new InequalityProblem
        with the given `expr`, `ineq_constraints` and `eq_constraints`
        while other attributes are copied from self.
        """
        problem = self.new(0, {}, {})
        problem.__dict__.update({k: v for k, v in self.__dict__.items() if k != "__weakref__"})
        problem.expr = expr
        problem.ineq_constraints = ineq_constraints
        problem.eq_constraints = eq_constraints
        problem.roots = self.roots.copy() if self.roots is not None else None
        return problem

    def copy(self) -> 'InequalityProblem':
        return self.copy_new(self.expr,
            self.ineq_constraints.copy(), self.eq_constraints.copy())

    def __iter__(self):
        """This is convenient for `sum_of_squares(*self)`."""
        return iter([self.expr, self.ineq_constraints.keys(), self.eq_constraints.keys()])

    def reduce(self, f: Callable[[T], Any], reduction: Callable[[Iterable[Any]], Any] = all) -> Any:
        """
        Apply a function over self.expr, self.ineq_constraints.keys()
        and self.eq_constraints.keys(), and reduce them by a given rule. Defaults to "all".

        Parameters
        ----------
        f : Callable[[T], Any]
            A function to apply over self.expr, self.ineq_constraints.keys()
            and self.eq_constraints.keys().
        reduction : Callable[[Iterable[Any]], Any], optional
            A reduction function to apply over the results of f, by default "all".

        Examples
        ----------
        >>> from sympy.abc import a, b
        >>> from sympy import Rational
        >>> pro = InequalityProblem(5 - 3*a - 4*b, [a, b], [a**2 + b**2 - 1])
        >>> pro.reduce(lambda x: x.is_Symbol)
        False
        >>> pro.reduce(lambda x: x.subs({a: Rational(3,5), b: Rational(4,5)}), list)
        [0, 3/5, 4/5, 0]
        """
        return reduction(map(f, [
            self.expr, *self.ineq_constraints.keys(), *self.eq_constraints.keys()]))

    def _dtype_is_zero(self, x: T) -> Optional[bool]:
        return _dtype_is_zero(x)

    def _dtype_convert(self, x: T, y: Any) -> T:
        return _dtype_convert(x, y)

    def _dtype_free_symbols(self, x: T) -> Set[Symbol]:
        return _dtype_free_symbols(x)

    def _dtype_gens(self, x: T) -> Tuple[Symbol, ...]:
        return _dtype_gens(x)

    def _dtype_is_homogeneous(self, x: T) -> Optional[bool]:
        return _dtype_is_homogeneous(x)

    def _dtype_homogenize(self, x: T, s: Symbol) -> T:
        return _dtype_homogenize(x, s)

    def _dtype_sqf_list(self, x: T) -> Tuple[Expr, List[Tuple[T, int]]]:
        return _dtype_sqf_list(x)

    def _dtype_std_ineq_constraints(self, p: T, e: Expr) -> Tuple[T, Expr]:
        if self._dtype_is_zero(p): return p, e
        c, lst = self._dtype_sqf_list(p)
        ret = self._dtype_convert(p, 1)
        sgn = 1 if c > 0 else -1
        e = e / (c if sgn > 0 else -c)
        for q, d in lst:
            if d % 2 == 1:
                ret = ret * q
            e = e / q.as_expr()**(d - d%2)
        if sgn == -1:
            ret = ret.__neg__()
        return ret, e

    def _dtype_std_eq_constraints(self, p: T, e: Expr) -> Tuple[T, Expr]:
        if self._dtype_is_zero(p): return p, e
        c, lst = self._dtype_sqf_list(p)
        ret = self._dtype_convert(p, 1)
        sgn = 1 if c > 0 else -1
        e = e / c
        max_d = Integer(max(1, *(d for q, d in lst)))
        for q, d in lst:
            ret = ret * q
            e = e * q.as_expr()**(max_d - d)
        if max_d != 1:
            e = Pow(e, 1/max_d, evaluate=False)
        if sgn == -1:
            e = e.__neg__()
            ret = ret.__neg__()
        return ret, e

    def _dtype_make_reorder_func(self, x: T, gens: Tuple[Symbol, ...]) -> Callable[[Permutation], T]:
        return _dtype_make_reorder_func(x, gens)


    @property
    def free_symbols(self) -> Set[Symbol]:
        __dtype_free_symbols = self._dtype_free_symbols
        return self.reduce(lambda e: __dtype_free_symbols(e), lambda x: set.union(*x))

    @property
    def gens(self) -> Tuple[Symbol, ...]:
        """
        Returns an ordered tuple of symbols in `self.expr`,
        `self.ineq_constraints.keys()` and `self.eq_constraints.keys()`.

        Ordering rules:
        * If `self.expr` is polynomial, its generators come first in the original order.
        * Other free symbols are sorted in names.

        Examples
        ----------
        >>> from sympy.abc import a, b, c, x, y
        >>> from sympy import Poly
        >>> InequalityProblem(b*2 + c, [b-a, a, c-a], [b+c-1]).gens
        (a, b, c)
        >>> InequalityProblem(Poly(y**2*b + 2*c, (c, b)), [b, x], [a-c, a-1]).gens
        (c, b, a, x, y)
        """
        poly_gens = self._dtype_gens(self.expr)
        other_syms = self.free_symbols - set(poly_gens)
        other_syms = sorted(list(other_syms), key=lambda x: x.name)
        return poly_gens + tuple(other_syms)

    def extract_constraints(self, symbols: Union[Symbol, List[Symbol]]) \
            -> Tuple[Dict[T, Expr], Dict[T, Expr], Dict[T, Expr], Dict[T, Expr]]:
        """
        Split constraints into those that contain given symbols and those that do not.
        Returns (contained_ineqs, contained_eqs, uncontained_ineqs, uncontained_eqs).

        Parameters
        ----------
        symbols: Union[Symbol, List[Symbol]]
            The symbols to split constraints by. It can be a symbol or an iterable of symbols.

        Returns
        ----------
        Tuple[Dict[T, Expr], Dict[T, Expr], Dict[T, Expr], Dict[T, Expr]]
            (contained_ineqs, contained_eqs, uncontained_ineqs, uncontained_eqs)

        Examples
        ----------
        >>> from sympy.abc import a, b, c
        >>> problem = InequalityProblem(a*b, [a, b, a*b, b+c], [a-1, b+c-1])
        >>> problem.extract_constraints(a) # doctest: +NORMALIZE_WHITESPACE
        ({a: a, a*b: a*b},
         {a - 1: a - 1},
         {b: b, b + c: b + c},
         {b + c - 1: b + c - 1})
        >>> problem.extract_constraints([b, c]) # doctest: +NORMALIZE_WHITESPACE
        ({b: b, a*b: a*b, b + c: b + c},
         {b + c - 1: b + c - 1},
         {a: a},
         {a - 1: a - 1})
        """
        # TODO: supports symbol-like expressions, e.g. Function, MatrixSymbol
        if isinstance(symbols, Symbol):
            symbols = {symbols}
        symbols = set(symbols)

        ineqs = [{}, {}]
        eqs = [{}, {}]
        has_any = lambda f: bool(f.free_symbols & symbols)
        for src, dst in [(self.ineq_constraints, ineqs), (self.eq_constraints, eqs)]:
            for p, e in src.items():
                dst[int(has_any(p))][p] = e

        return ineqs[1], eqs[1], ineqs[0], eqs[0]

    @property
    def is_polynomial(self) -> bool:
        return bool(self._is_polynomial)

    @property
    def is_of_type(self, dtype) -> bool:
        return self.reduce(lambda e: isinstance(e, dtype), all)

    def clear_roots(self) -> 'InequalityProblem[T]':
        self.roots = None
        return self

    def clear_counter_examples(self) -> 'InequalityProblem[T]':
        self.counter_examples = None
        return self

    def clear_solution(self) -> 'InequalityProblem[T]':
        self.solution = None
        return self

    def get_symbol_signs(self) -> Dict[Symbol, Tuple[Optional[int], Expr]]:
        from .preprocess import get_symbol_signs
        return get_symbol_signs(self)

    def get_features(self) -> Dict[str, Any]:
        from .preprocess.features import get_features
        features = {}
        try:
            features = get_features(self.polylize())
        except BasePolynomialError:
            # XXX: might change in the future
            raise NonPolynomialError("Cannot extract features from non-polynomial problem.")
        return features

    def evaluate_complexity(self) -> ProblemComplexity:
        # The estimation here is only a placeholder.
        # In ProofNodes it will overloaded by model predictions.
        nvars = len(self.free_symbols)
        return ProblemComplexity(
            time=nvars**4/81 * (len(self.ineq_constraints)+1)*(len(self.eq_constraints)+1),
            prob=min(.95, 3.1**2/((nvars+.1)**2)),
            length=nvars**4
        )

    def sum_of_squares(self, configs: dict = {}):
        from .node import _sum_of_squares
        return _sum_of_squares(self, configs)

    @property
    def is_homogeneous(self) -> bool:
        return self.reduce(lambda e: self._dtype_is_homogeneous(e), all)

    def polylize(self,
        ineqs_sqf: bool = True,
        eqs_sqf: bool = True,
    ) -> 'InequalityProblem':
        problem = self
        expr, ineq_constraints, eq_constraints = \
            problem.expr, problem.ineq_constraints.copy(), problem.eq_constraints.copy()
        gens = self.gens

        if len(gens) == 0:
            gens = {Symbol('x')}
        expr = Poly(expr.doit(), *gens)
        ineq_constraints = dict((Poly(e.doit(), *gens), e2) for e, e2 in ineq_constraints.items())
        eq_constraints = dict((Poly(e.doit(), *gens), e2) for e, e2 in eq_constraints.items())

        problem = InequalityProblem(expr, ineq_constraints, eq_constraints)
        problem, _ = problem.sqr_free(problem_sqf=False,
            ineqs_sqf=ineqs_sqf, eqs_sqf=eqs_sqf, inplace=True)
        if self.roots is not None:
            # TODO: sqf ineqs might exclude some roots here
            problem.roots = self.roots.reorder(problem.gens)
        return problem

    def sqr_free(self,
        problem_sqf: bool = False,
        ineqs_sqf: bool = True,
        eqs_sqf: bool = True,
        inplace: bool = False
    ) -> Tuple['InequalityProblem', Expr]:
        """
        Try to make the problem square-free.

        Parameters
        ----------
        problem_sqf: bool, optional
            Whether to make the problem expression square-free. Default is False.
        ineqs_sqf: bool, optional
            Whether to make the inequalities square-free. Default is True.
        eqs_sqf: bool, optional
            Whether to make the equalities square-free. Default is True.
        inplace: bool, optional
            Whether to modify the problem in-place. Default is False.

        Returns
        ----------
        problem: InequalityProblem
            The square-free problem.
        sqr: Expr
            The expression such that `new_problem.expr * sqr**2 == problem.expr`.

        Examples
        ----------
        >>> from sympy.abc import a, b, c, d, x, y, z
        >>> pro = InequalityProblem(a*(b+2) + c + b*d, {a/(b + 2)**3: x, c*b**2: y}, {d**3: z})
        >>> pro.sqr_free()[0].ineq_constraints
        {a*(b + 2): x*(b + 2)**4, c: y/b**2}
        >>> pro.sqr_free()[0].eq_constraints
        {d: z**(1/3)}

        The second argument is the square-free expression from `self.expr`.

        >>> pro = InequalityProblem((x - 2)**2*(x**2 - x + 1))
        >>> pro.sqr_free(problem_sqf = True) # doctest: +NORMALIZE_WHITESPACE
        (<InequalityProblem of 1 variables, with 0 inequality and 0 equality constraints>, x - 2)
        >>> pro.sqr_free(problem_sqf = True)[0].expr
        x**2 - x + 1
        """
        if not inplace:
            self = self.copy()

        sqr = Integer(1)
        if problem_sqf and not self._dtype_is_zero(self.expr):
            c, lst = self._dtype_sqf_list(self.expr)
            sqr = []
            sqf = self._dtype_convert(self.expr, c)
            for p, d in lst:
                sqr.append(p.as_expr()**(d//2))
                if d % 2 == 1:
                    sqf = sqf*p
            sqr = Mul(*sqr)
            self.expr = sqf

        ineq_constraints = self.ineq_constraints
        if ineqs_sqf:
            ineq_constraints = dict(self._dtype_std_ineq_constraints(*item)
                for item in ineq_constraints.items())
        self.ineq_constraints = dict((e, e2) for e, e2 in ineq_constraints.items())

        eq_constraints = self.eq_constraints
        if eqs_sqf:
            eq_constraints = dict(self._dtype_std_eq_constraints(*item)
                for item in eq_constraints.items())
        self.eq_constraints = dict((e, e2) for e, e2 in eq_constraints.items())
        return self, sqr

    def homogenize(self, hom: Optional[Symbol]=None) -> Tuple['InequalityProblem', Optional[Symbol]]:
        """
        Try to homogenize the problem.

        Parameters
        ----------
        hom: Symbol, optional
            The homogenizer symbol.
            * If None, a new symbol named "1" will
            be created if the problem is not homogeneous.
            * If given, it tries to homogenize the problem
            even if it is already homogeneous.

        Returns
        ----------
        problem: InequalityProblem
            The homogenized problem.
        hom: Symbol, optional
            The homogenizer symbol. None if no homogenizer is used.

        Examples
        ----------
        >>> from sympy.abc import a, b, c
        >>> pro = InequalityProblem(1/(a**2 + 1) + 1/(b**2 + 1) - 1, [a, b], [a + b - 2])
        >>> pro.homogenize()
        (<InequalityProblem of 3 variables, with 3 inequality and 1 equality constraints>, 1)
        >>> type(pro.homogenize()[1])
        <class 'sympy.core.symbol.Symbol'>
        >>> pro.homogenize()[0].expr
        1**2/(1**2 + b**2) + 1**2/(1**2 + a**2) - 1

        The homogenizer defaults to a Symbol named "1". It is also possible
        to use a customized Symbol object:

        >>> pro.homogenize(c)[0].expr
        c**2/(b**2 + c**2) + c**2/(a**2 + c**2) - 1
        """
        if hom is None and self.is_homogeneous:
            return self, None
        if hom is None:
            hom = uniquely_named_symbol("1", self.gens, real=True, positive=True, modify=lambda x: "_"+x)
        _homogenize = self._dtype_homogenize
        expr = _homogenize(self.expr, hom)
        ineqs = {_homogenize(e, hom): v for e, v in self.ineq_constraints.items()}
        ineqs[self._dtype_convert(expr, hom)] = hom # homogenizer = 1 >= 0
        eqs = {_homogenize(e, hom): v for e, v in self.eq_constraints.items()}

        new_problem = self.copy_new(expr, ineqs, eqs)
        if self.roots is not None:
            new_problem.roots = RootList.new(self.roots.symbols + (hom,),
                [Root(r.root + (Integer(1),), r.domain, r.rep + (r.domain.one,)) for r in self.roots])
        return new_problem, hom

    def identify_symmetry(self) -> PermutationGroup:
        """
        Try to identify the symmetry of the problem.

        Returns
        ----------
        perm_group: PermutationGroup
            The problem is invariant up to the permutation group.

        Examples
        ----------
        >>> from sympy.abc import a, b, c
        >>> pro = InequalityProblem(a**2*b+b**2*c+c**2*a-3, [a-1, b-1, c-1])
        >>> pro.identify_symmetry()
        PermutationGroup([
            (0 1 2)])
        >>> pro.gens
        (a, b, c)
        """
        gens = self.gens
        reorder_funcs = self.reduce(lambda x: (x, self._dtype_make_reorder_func(x, gens)), dict)
        ls = [[self.expr], list(self.ineq_constraints), list(self.eq_constraints)]
        def verify_func(perm: Permutation) -> bool:
            # TODO: we can use int-index for reorder_funcs and ls
            reorder = lambda x: reorder_funcs[x](perm)
            return all(verify_closure(l, reorder) for l in ls)
        return _identify_symmetry_from_blackbox(verify_func, len(gens))

    def wrap_constraints(self, symmetry: Optional[PermutationGroup]=None) -> Tuple['InequalityProblem', Callable]:
        """
        Wrap the constraints of the problem by dummy functions.

        Parameters
        ----------
        symmetry : PermutationGroup, optional
            The symmetry group of the problem.

        Returns
        ----------
        problem : InequalityProblem
            The problem with wrapped constraints.
        restoration : Callable
            A function to restore the expression from the wrapped expression.

        Examples
        ----------
        Consider proving x >= 0 given x + y >= 1 and x**2 + y**2 == 1:

        >>> from sympy.abc import a, b, c, x, y
        >>> pro = InequalityProblem(x, {x+y-1: x+y-1}, {x**2+y**2-1: x**2+y**2-1})
        >>> newpro, restore = pro.wrap_constraints()
        >>> newpro.ineq_constraints, newpro.eq_constraints
        ({x + y - 1: _G0(x, y)}, {x**2 + y**2 - 1: _H0(x, y)})

        We can define the solution with G0 and H0 and restore it using the restoration function.
        However, restoration expands the brackets and might break the sum-of-squares structure.

        >>> G0, H0 = list(newpro.ineq_constraints.values())[0], list(newpro.eq_constraints.values())[0]
        >>> sol = G0 - H0/2 + x**2/2 + (y-1)**2/2; sol
        x**2/2 + (y - 1)**2/2 + _G0(x, y) - _H0(x, y)/2
        >>> restore(sol)
        x - y**2/2 + y + (y - 1)**2/2 - 1/2
        >>> restore(sol).expand()
        x

        When symmetry is specified, the wrapper tries to exploit the symmetry.

        >>> pro = InequalityProblem(a+b+c, [2*a+b, 2*b+c, 2*c+a])
        >>> pro.wrap_constraints()[0].ineq_constraints # doctest: +SKIP
        {2*a + b: _G0(a, b), 2*b + c: _G1(b, c), a + 2*c: _G2(a, c)}

        >>> from sympy.combinatorics import CyclicGroup
        >>> pro.wrap_constraints(CyclicGroup(3))[0].ineq_constraints # doctest: +SKIP
        {2*a + b: _G0(a, b), 2*b + c: _G0(b, c), a + 2*c: _G0(c, a)}
        """
        gens = self.gens
        reorder_funcs = self.reduce(lambda x: (x, self._dtype_make_reorder_func(x, gens)), dict)
        def reorder_func(x, p):
            return reorder_funcs[x](p)
        i2g, e2h, g2i, h2e = _get_constraints_wrapper(
            gens, self.ineq_constraints, self.eq_constraints, symmetry,
            reorder_func = reorder_func,
            free_symbols_func = self._dtype_free_symbols
        )
        problem = self.copy()
        problem.ineq_constraints = i2g
        problem.eq_constraints = e2h
        def restoration(x):
            if x is None: return None
            return x.xreplace(g2i).xreplace(h2e)
        return problem, restoration

    def find_roots(self) -> RootList:
        """Find the equality cases of the problem heuristically."""
        if self.roots is not None:
            return self.roots
        from sympy.polys.polyerrors import DomainError
        try:
            roots = optimize_poly(self.expr, list(self.ineq_constraints), [self.expr] + list(self.eq_constraints),
                        self.gens, return_type='root')
        except DomainError:
            roots = RootList(self.gens, [])
        self.roots = roots
        return self.roots

    def set_roots(self, roots) -> RootList:
        """
        Safely set the roots of the problem. Accepts
        multiple input types (None or list of tuples or list of dicts).
        """
        if roots is None:
            return
        if not isinstance(roots, RootList):
            if isinstance(roots, (list, tuple)):
                _roots = []
                for r in roots:
                    if isinstance(r, dict):
                        _roots.append(tuple([r[g] for g in self.gens]))
                    elif isinstance(r, (tuple, Root, list)):
                        _roots.append(r)
                    else:
                        raise TypeError(f"Cannot convert {r} to Root.")
                roots = RootList(self.gens, _roots)
            else:
                raise TypeError(f"Cannot convert {roots} to RootList.")
        elif self.gens != roots.symbols:
            raise ValueError(f"RootList symbols {roots.symbols} do not match the problem generators {self.gens}.")
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
        if self.roots is not None:
            problem.roots = self.roots.transform(inv_transform, problem.gens)

        def restore(x: Optional[Expr]) -> Optional[Expr]:
            if x is None:
                return None
            return x.xreplace(inv_transform) / next(iter(dst_dicts[0].values()))
        return problem, restore



def _get_constraints_wrapper(
        symbols: Tuple[int, ...],
        ineq_constraints: Dict[T, Expr],
        eq_constraints: Dict[T, Expr],
        perm_group: Optional[PermutationGroup]=None,
        reorder_func: Callable[[T, Permutation], T]=None,
        free_symbols_func: Callable[[T], Set[Symbol]]=_dtype_free_symbols,
    ):
    if perm_group is None:
        # trivial group
        perm_group = PermutationGroup(Permutation(list(range(len(symbols)))))
    if reorder_func is None:
        def reorder_func(x, p):
            return _dtype_make_reorder_func(x, symbols)(p)

    def _get_mask(symbols, active):
        # only reserve symbols in `fs`, this reduces time complexity greatly
        return tuple(s for s, a in zip(symbols, active) if a)

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
        # rep_dict = dict((p, v) for p, v in constraints.items())
        rep_dict = constraints
        if counter is None:
            counter = _get_counter(name)

        for base in constraints.keys():
            if base in dt:
                continue
            fs = free_symbols_func(base)
            active = [bool(s in fs) for s in symbols]
            for p in perm_group.elements:
                # invorder = p.__invert__()(symbols)
                # permed_base = base.reorder(*invorder)
                permed_base = reorder_func(base, p)
                permed_expr = rep_dict.get(permed_base)
                if permed_expr is None:
                    raise ValueError("Given constraints are not symmetric with respect to the permutation group.")
                compressed = _get_mask(p(symbols), active)
                value = Function(name + str(counter))(*compressed)
                dt[permed_base] = value
                inv[value] = permed_expr
            counter += 1
        # dt = dict((k, v) for k, v in dt.items())
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
