from collections import defaultdict

from typing import List, Tuple, Dict, Union, Optional, TypeVar, Generic
import re
from warnings import warn

import sympy as sp
from sympy.core.singleton import S
from sympy import Poly, Expr, Symbol, Float, Equality, Add, Mul
from sympy.printing.precedence import precedence_traditional
from sympy.printing.str import StrPrinter
from sympy.combinatorics import PermutationGroup, Permutation, CyclicGroup
from sympy.core.relational import Equality

from .problem import InequalityProblem
from ..utils.expressions.cyclic import CyclicProduct, CyclicExpr, rewrite_symmetry
from ..utils.expressions.psatz import SOSlist, PSatz


T = TypeVar('T')

class SolutionBase(Generic[T]):
    pass

class Solution(SolutionBase[T]):
    """
    The `Solution` class is the standard return type of the `sum_of_squares` function.
    It holds information about an inequality problem and its solution.
    In a Jupyter notebook, it is displayed as a SymPy equation.

    >>> from sympy.abc import a
    >>> from triples.core import sum_of_squares
    >>> sol = sum_of_squares(a**2 - 2*a + 1)
    >>> sol # doctest: +SKIP
    Solution(problem = <InequalityProblem of 1 variables, with 0 inequality and 0 equality constraints>,
     solution = (a - 1)**2)

    The problem expression and the solution can be accessed via `.expr` and `.solution` properties,
    which are sympy objects.

    >>> sol.expr
    a**2 - 2*a + 1
    >>> sol.solution # doctest: +SKIP
    (a - 1)**2

    >>> sol.time # doctest: +SKIP
    0.014049
    """
    _start_time = None
    _end_time = None
    _is_equal = None

    method = ''
    def __init__(self,
        problem: Optional[Union[InequalityProblem[T], T]]=None,
        solution: Optional[Expr]=None,
        ineq_constraints: Optional[Dict[T, Expr]]=None,
        eq_constraints: Optional[Dict[T, Expr]]=None,
        is_equal: Optional[bool]=None
    ):

        if not isinstance(problem, InequalityProblem):
            ineq_constraints = ineq_constraints if ineq_constraints is not None else dict()
            eq_constraints = eq_constraints if eq_constraints is not None else dict()
            problem = InequalityProblem(problem, ineq_constraints, eq_constraints)
            problem.solution = solution
        else:
            # isinstance(problem, InequalityProblem)
            if ineq_constraints is not None or eq_constraints is not None:
                raise ValueError("Inequality and equality constraints must be None if the problem is an InequalityProblem.")

        if solution is None:
            solution = problem.solution

        self.problem = problem
        self.solution = solution
        self._is_equal = is_equal

    @property
    def expr(self) -> T:
        return self.problem.expr

    @property
    def ineq_constraints(self) -> Dict[T, Expr]:
        return self.problem.ineq_constraints

    @property
    def eq_constraints(self) -> Dict[T, Expr]:
        return self.problem.eq_constraints

    @expr.setter
    def expr(self, value: T):
        """Bypass immutability: directly overwrite the problem expression; use with care."""
        self.problem.expr = value

    @ineq_constraints.setter
    def ineq_constraints(self, value: Dict[T, Expr]):
        """Bypass immutability: directly overwrite the problem inequality constraints; use with care."""
        self.problem.ineq_constraints = value

    @eq_constraints.setter
    def eq_constraints(self, value: Dict[T, Expr]):
        """Bypass immutability: directly overwrite the problem equality constraints; use with care."""
        self.problem.eq_constraints = value

    @property
    def time(self) -> float:
        """Get the elapsed time for computing the solution. Return -1. if not registered."""
        return (self._end_time - self._start_time).total_seconds() if self._end_time is not None else -1.

    def __str__(self) -> str:
        return f"Solution(problem = {self.problem!r}, solution = {self.solution})"

    def __repr__(self) -> str:
        return self.__str__()

    def copy(self) -> 'Solution':
        from copy import copy
        return copy(self)

    @property
    def gens(self) -> Tuple[Symbol, ...]:
        """
        Get the free symbols of problem.

        Examples
        ---------
        >>> from sympy.abc import a, b, c
        >>> from sympy import Function
        >>> sol = Solution((a**2+b**2+c**2-a*b-b*c-c*a)*2, (a-b)**2+(b-c)**2+(c-a)**2)
        >>> sol.gens
        (a, b, c)
        >>> Function('F')(*sol.gens)
        F(a, b, c)
        """
        return self.problem.gens

    @property
    def is_equal(self) -> bool:
        """
        Verify whether the solution is correct. This is heuristic and might fail
        for hard cases or take a very long time. It is more suggested to verify
        the solution manually by sampling a few points.

        Examples
        ----------
        >>> from sympy.abc import a, b
        >>> from sympy import sqrt
        >>> sol1 = Solution(a**2 + a*b + b**2, (((2-sqrt(3))*a + b)**2 + (a + (2-sqrt(3))*b)**2)/(8-4*sqrt(3)))
        >>> sol1.is_equal
        True
        >>> sol2 = Solution(a**2 + a*b + b**2, (((2-3**.5)*a + b)**2 + (a + (2-3**.5)*b)**2)/(8-4*3**.5))
        >>> sol2.is_equal
        False
        """
        if self._is_equal is None:
            if self.as_eq().simplify() in (sp.true, True):
                self._is_equal = True
            else:
                self._is_equal = False
        return self._is_equal

    @property
    def is_ill(self) -> bool:
        """
        Whether the solution is ill-defined, e.g. +oo, -oo, NaN, etc.
        This avoids bugs when encountering 0/0, etc.

        Examples
        ----------
        >>> from sympy.abc import a
        >>> from sympy import nan
        >>> sol = Solution(a**2 - 2*a + 1, nan)
        >>> sol.is_ill
        True
        """
        if self.solution in (None, S.NaN, S.Infinity, S.NegativeInfinity, S.ComplexInfinity):
            return True
        if self.solution is S.Zero and isinstance(self.expr, Poly) and not self.expr.is_zero:
            return True
        return False

    @property
    def is_Exact(self) -> bool:
        """
        Whether the solution does not contain floating point numbers.

        Examples
        ----------
        >>> from sympy.abc import a
        >>> from sympy import sqrt
        >>> sol = Solution(a**2 - 2*sqrt(3)*a + 3, (a - 1.73205080757)**2)
        >>> sol.is_Exact
        False
        """
        return not (self.is_ill or self.solution.has(Float))

    def _str_f(self, name='f') -> str:
        return "%s(%s)"%(name, ','.join(str(_) for _ in self.gens))

    def as_eq(self, lhs_expr=None, together=False, cancel=False) -> Equality:
        """
        Convert the solution to a sympy equality object.

        Examples
        ---------
        >>> from sympy.abc import a
        >>> sol = Solution(a**2 - 2 + 1/a**2, (a**2 - 1)**2/a**2)
        >>> sol.as_eq()
        Eq(a**2 - 2 + a**(-2), (a**2 - 1)**2/a**2)
        >>> sol.as_eq().lhs, sol.as_eq().rhs
        (a**2 - 2 + a**(-2), (a**2 - 1)**2/a**2)
        >>> sol.as_eq().simplify()
        True
        >>> sol.as_eq(cancel=True)
        Eq(a**2*(a**2 - 2 + a**(-2)), (a**2 - 1)**2)

        >>> from sympy import Function
        >>> f = Function('f')
        >>> sol.as_eq(lhs_expr=f(a), cancel=True)
        Eq(a**2*f(a), (a**2 - 1)**2)
        """
        lhs = self.expr.as_expr() if lhs_expr is None else lhs_expr
        if cancel:
            rhs, denom = self.as_fraction(together=together)
            lhs = lhs * denom
        else:
            rhs = self.solution.together() if together else self.solution
        return Equality(lhs, rhs, evaluate=False)

    def as_sos_list(self) -> Optional[SOSlist]:
        return SOSlist.from_sympy(self.solution)

    def as_psatz(self) -> Optional[PSatz]:
        return PSatz.from_sympy(self.solution,
            {v: k for k, v in self.ineq_constraints.items()},
            {v: k for k, v in self.eq_constraints.items()})

    def to_string(self, mode: str = 'latex', lhs_expr=None, together=False, cancel=False, settings=None) -> str:
        """
        Convert the solution to a string. The mode can be 'latex', 'txt', or 'formatted'.

        Parameters
        ----------
        mode : str, optional
            The mode of the string, by default 'latex'.
            'latex': Convert to latex string.
            'txt': Convert to plain text string.
            'formatted': Convert to formatted string where "s" and "p" stands for
            cyclic sum and cyclic product, respectively. This is not safe when
            the symbols contain s or p.
        lhs_expr : Expr
            Sympy expressions to replace the left-hand side problem.
        together : bool, optional
            Whether to apply `sympy.together` on the right-hand side solution.
        cancel : bool, optional
            Whether to apply `sympy.cancel` on the right-hand side solution.
        settings : dict, optional
            Settings for printing. See `sympy.printing.str.StrPrinter._print` for details.

        Examples
        ---------
        >>> from sympy.abc import a, b, c
        >>> sol = Solution(a**2 + a*b + b**2, (a + b/2)**2 + 3*b**2/4)
        >>> sol.to_string()  # doctest: +SKIP
        'a^{2} + a b + b^{2} = \\frac{3 b^{2}}{4} + \\frac{\\left(2 a + b\\right)^{2}}{4}'
        >>> sol.to_string(mode = 'txt')  # doctest: +SKIP
        'a² + ab + b² = 3b²/4 + (2a + b)²/4'

        >>> from sympy import Function
        >>> F = Function('F')
        >>> sol.to_string(mode = 'txt', lhs_expr=F(a,b))  # doctest: +SKIP
        'F(a, b) = 3b²/4 + (2a + b)²/4'
        """
        eq = self.as_eq(lhs_expr=lhs_expr, together=together, cancel=cancel)
        lhs, rhs = eq.lhs, eq.rhs
        if mode == 'latex':
            to_str = lambda x: _print_latex(x, settings=settings)
        elif mode == 'txt' or mode == 'formatted':
            if mode == 'txt':
                cyclic_sum_name = 'Σ'
                cyclic_product_name = '∏'
                with_cyclic_parens = False
            else:
                cyclic_sum_name = 's'
                cyclic_product_name = 'p'
                with_cyclic_parens = True
            _to_str = lambda x: _print_str(x, cyclic_sum_name=cyclic_sum_name,
                cyclic_product_name=cyclic_product_name, with_cyclic_parens=with_cyclic_parens,
                settings=settings).replace('**','^')

            if mode == 'txt':
                def _convert_superscript(s):
                    pow_trans = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')
                    return re.sub(r'\^(\d+)', lambda m: m.group(1).translate(pow_trans), s)
                to_str = lambda x: _convert_superscript(_to_str(x).replace('*',''))
            else:
                to_str = lambda x: _to_str(x)
        else:
            raise ValueError(f"Unknown mode {mode}.")

        lhs_str = to_str(lhs)
        rhs_str = to_str(rhs)
        return f"{lhs_str} = {rhs_str}"

    def _repr_latex_(self):
        eq = self.as_eq()
        return eq._repr_latex_()
        # s = sp.latex(eq, mode='plain', long_frac_ratio=2)
        # return "$\\displaystyle %s$" % s

    def together(self, *args, **kwargs) -> 'Solution':
        """
        Make a copy of the solution and apply `together` on it.
        See also: sympy.together.

        Examples
        ---------
        >>> from sympy.abc import a, b
        >>> sol = Solution(a**2 - a*b + b**2, (a + b/2)**2 + 3*b**2/4)
        >>> sol.solution
        3*b**2/4 + (a + b/2)**2
        >>> sol.together().solution
        (3*b**2 + (2*a + b)**2)/4
        """
        self = self.copy()
        self.solution = sp.together(self.solution, *args, **kwargs)
        return self

    def signsimp(self, *args, **kwargs) -> 'Solution':
        """
        Make a copy of the solution and apply `signsimp` on it.
        See also: sympy.signsimp.

        Examples
        ---------
        >>> from sympy.abc import a, b
        >>> sol = Solution(a**2 - 2*a*b + b**2, (-a - b)**2)
        >>> sol.solution
        (-a - b)**2
        >>> sol.signsimp().solution
        (a + b)**2
        """
        self = self.copy()
        self.solution = sp.signsimp(self.solution, *args, **kwargs)
        return self

    def xreplace(self, *args, **kwargs) -> 'Solution':
        """
        Make a copy of the solution and apply `xreplace` on it.
        See also: sympy.xreplace.

        Examples
        ---------
        >>> from sympy.abc import a, b, x
        >>> sol = Solution(a**2 - 2*a*b + b**2, (a - b)**2)
        >>> sol.xreplace({a: x}).solution
        (-b + x)**2
        """
        self = self.copy()
        self.solution = self.solution.xreplace(*args, **kwargs)
        return self

    def doit(self, *args, **kwargs) -> 'Solution':
        """
        Make a copy of the solution and apply doit on it.
        This is useful to expand cyclic expressions.
        See also: sympy.doit.

        Examples
        ---------
        >>> from sympy.abc import a, b, c
        >>> from triples.utils import CyclicSum
        >>> sol = Solution((a+b+c)*(a*b+b*c+c*a)-9*a*b*c, CyclicSum(a*(b-c)**2, (a,b,c)))
        >>> sol.solution
        Σ(a*(b - c)**2)
        >>> sol.doit().solution
        a*(b - c)**2 + b*(-a + c)**2 + c*(a - b)**2
        """
        self = self.copy()
        self.solution = self.solution.doit(*args, **kwargs)
        return self

    def collect(self, *args, **kwargs) -> 'Solution':
        """
        Make a copy of the solution and apply `collect` on it.
        See also: sympy.collect.

        Examples
        ---------
        >>> from sympy.abc import a, b, c
        >>> sol = Solution((a+b**2+c)*(b-c)**2, a*(b-c)**2 + b**2*(b-c)**2 + c*(b-c)**2)
        >>> sol.solution
        a*(b - c)**2 + b**2*(b - c)**2 + c*(b - c)**2
        >>> sol.collect((b-c)**2).solution
        (b - c)**2*(a + b**2 + c)
        """
        self = self.copy()
        self.solution = self.solution.collect(*args, **kwargs)
        return self

    def n(self, *args, **kwargs) -> 'Solution':
        """
        Make a copy of the solution and apply `n` on it.
        See also: sympy.n.

        Examples
        ---------
        >>> from sympy.abc import a, b
        >>> from sympy import sqrt
        >>> sol = Solution(a**2+a*b+b**2, (((2-sqrt(3))*a+b)**2+((2-sqrt(3))*b+a)**2)/(8-4*sqrt(3)))
        >>> sol.n(4).solution
        0.933*(0.2679*a + b)**2 + 0.933*(a + 0.2679*b)**2
        """
        self = self.copy()
        self.solution = self.solution.n(*args, **kwargs)
        return self

    def evalf(self, *args, **kwargs) -> 'Solution':
        """
        Make a copy of the solution and apply `evalf` on it.
        See also: sympy.evalf.

        Examples
        ---------
        >>> from sympy.abc import a, b
        >>> from sympy import sqrt
        >>> sol = Solution(a**2+a*b+b**2, (((2-sqrt(3))*a+b)**2+((2-sqrt(3))*b+a)**2)/(8-4*sqrt(3)))
        >>> sol.evalf(4).solution
        0.933*(0.2679*a + b)**2 + 0.933*(a + 0.2679*b)**2
        """
        self = self.copy()
        self.solution = self.solution.evalf(*args, **kwargs)
        return self

    def as_expr(self, *args, **kwargs) -> Expr:
        """
        Return the solution as an expression. It is equivalent to .solution.
        """
        return self.solution#.doit(*args, **kwargs)

    def dehomogenize(self, homogenizer: Optional[Symbol] = None):
        """
        Dehomogenize the solution. Used internally.
        """
        if homogenizer is None:
            return self

        expr = self
        if isinstance(self, Solution):
            self = self.copy()
            self.problem = self.problem.subs(homogenizer, 1)
            expr = self.solution
        def _deflat_perm_group(expr):
            # e.g.
            # CyclicSum(a**2, (a,b,c,d), PermutationGroup(Permutation([1,2,0,3])))
            # => CyclicSum(a**2, (a,b,c), PermutationGroup(Permutation([1,2,0])))
            f, symbols, perm_group = expr.args
            n = len(symbols)
            if symbols[n-1] == homogenizer and len(perm_group.orbit(n-1)) == 1:
                new_perm = PermutationGroup(*[Permutation(_.array_form[:-1]) for _ in perm_group.args])
                return expr.func(f, symbols[:n-1], new_perm)
            return expr
        def dehom(f):
            f = f.xreplace({homogenizer: 1})
            f = f.replace(lambda x: isinstance(x, CyclicExpr), _deflat_perm_group)
            return f

        if isinstance(self, Solution):
            self.solution = dehom(expr)
        else:
            self = dehom(expr)
        return self

    def as_fraction(self, together=True, inplace=False):
        """
        Denest the fractions and express the solution as the division of two fraction-free expressions.

        Examples
        ---------
        >>> from sympy.abc import a, b
        >>> sol = Solution(a**2 - 2 + 1/a**2, (a**2 - 1)**2/a**2)
        >>> sol.as_fraction()
        ((a**2 - 1)**2, a**2)
        """
        solution = self.solution.together() if together else self.solution
        numerator, multiplier = sp.fraction(solution)
        v, m = multiplier.as_content_primitive()
        if v < 0:
            v, m = -v, -m
        multiplier = m

        inv_v = S.One/v
        numerator = inv_v * numerator
        v, m = numerator.as_content_primitive()
        if v < 0:
            v, m = -v, -m
        if isinstance(m, Add):
            numerator = Add(*[v * arg for arg in m.args])

        if inplace:
            self.solution = numerator / multiplier
        return numerator, multiplier


    @property
    def numerator(self):
        warn("Calling Solution.numerator will be deprecated. Use Solution.as_fraction()[0] instead.", DeprecationWarning, stacklevel=2)
        return self.as_fraction()[0]

    @property
    def multiplier(self):
        warn("Calling Solution.multiplier will be deprecated. Use Solution.as_fraction()[1] instead.", DeprecationWarning, stacklevel=2)
        return self.as_fraction()[1]

    def rewrite_symmetry(self, symbols: Optional[Tuple[Symbol, ...]]=None, perm_group: Optional[PermutationGroup]=None) -> 'Solution':
        """
        Rewrite the expression heuristically with respect to the given permutation group.
        After rewriting, it is expected all cyclic expressions are expanded or in the given permutation group.
        This avoids the ambiguity of the cyclic expressions.

        It makes a copy of the solution and applies the rewriting on it. Note that
        the rewriting is not reversible if cyclic expressions are expanded. If this
        method is to be called multiple times, it is recommended to call on the original solution.

        Parameters
        ----------
        symbols : Tuple[Symbol, ...]
            The symbols that the permutation group acts on.
        perm_group : PermutationGroup
            Sympy permutation group object. Defaults to the CyclicGroup if not given.

        Returns
        ----------
        Solution
            A new solution object with the rewritten expression.

        See also
        ----------
        rewrite_symmetry
        """
        if symbols is None and (perm_group is not None):
            raise ValueError("Symbols must be given if perm_group is given.")
        if (symbols is not None) and perm_group is None:
            perm_group = CyclicGroup(len(symbols))

        self = self.copy()

        if symbols is None:
            # align perm groups to a single kind
            cyc_exprs = self.solution.find(CyclicExpr)
            if len(cyc_exprs) <= 1:
                return self

            perms = set([_.args[1:] for _ in cyc_exprs])
            if len(perms) == 1:
                return self

            def _count_weight(arg):
                if arg.is_Atom:
                    return 1
                return sum([_count_weight(_) for _ in arg.args])

            # get the perm group with the maximum weight
            counter = defaultdict(int)
            for cyc_expr in cyc_exprs:
                counter[(cyc_expr.args[1], cyc_expr.args[2])] += _count_weight(cyc_expr.args[0])
            (symbols, perm_group), weight = max(counter.items(), key=lambda x: x[1])

        self.solution = rewrite_symmetry(self.solution, symbols, perm_group)
        return self



# class SolutionNull(Solution):
#     def __init__(self, problem = None, solution = None):
#         super().__init__(problem = problem, solution = None)


def _arg_sqr_core(arg):
    if arg.is_constant():
        return S.One
    if isinstance(arg, Symbol):
        return arg
    if arg.is_Pow:
        return S.One if arg.args[1] % 2 == 0 else arg.args[0]
    if arg.is_Mul:
        return Mul(*[_arg_sqr_core(x) for x in arg.args])
    if isinstance(arg, CyclicProduct):
        return CyclicProduct(_arg_sqr_core(arg.args[0]), arg.symbols).doit()
    return None



def _print_str(expr: Expr, cyclic_sum_name = 'Σ', cyclic_product_name = '∏',
               with_cyclic_parens = False, settings=None):
    """Advanced printing to handle cyclic expressions."""
    settings = {} if settings is None else settings
    printer = StrPrinter(**settings)
    def _print_CyclicExpr(prefix):
        def _str_str(expr):
            s = printer._print(expr.args[0])
            if with_cyclic_parens or precedence_traditional(expr.args[0]) < expr.__class__.precedence:
                s = '(%s)'%s
            return prefix + s
        return _str_str
    _print_CyclicSum = _print_CyclicExpr(cyclic_sum_name)
    _print_CyclicProduct = _print_CyclicExpr(cyclic_product_name)
    setattr(printer, '_print_CyclicSum', lambda expr: _print_CyclicSum(expr))
    setattr(printer, '_print_CyclicProduct', lambda expr: _print_CyclicProduct(expr))
    return printer.doprint(expr)

def _print_latex(expr: Expr, settings=None):
    # if 'long_frac_ratio' not in settings:
    #     settings['long_frac_ratio'] = 2
    settings = {} if settings is None else settings
    return sp.latex(expr, **settings)
