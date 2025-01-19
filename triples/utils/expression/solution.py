from typing import Optional, List, Union, Tuple
import re
from warnings import warn

import sympy as sp
from sympy.core.singleton import S
from sympy.polys import Poly
from sympy.printing.precedence import precedence_traditional, PRECEDENCE
from sympy.combinatorics import PermutationGroup, Permutation

from ..expression.cyclic import is_cyclic_expr, CyclicSum, CyclicProduct, CyclicExpr, rewrite_symmetry

def _solution_latex_to_txt(s: str) -> str:
    """
    Turn a latex string into a text string.
    WARNING: It has bug when handling nested fractions.

    Deprecated.
    """
    s = s.strip('$')
    
    parener = lambda x: '(%s)'%x if '+' in x or '-' in x else x
    s = re.sub('frac\{(.*?)\}\{(.*?)\}',
                    lambda x: '%s/%s'%(parener(x.group(1)), parener(x.group(2))),
                    s)

    replacements = dict([
        (' ',''),
        ('\\', ''),
        ('left', ''),
        ('right', ''),
        ('{', ''),
        ('}', ''),
        ('cdot', ''),
        ('sum', 'Σ'),
        ('prod', '∏'),
        ('sqrt', '√'),
        *map(lambda x: ('^%d'%(x[0]+2), x[1]), enumerate('²³⁴⁵⁶⁷⁸⁹'))
    ])

    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def _multiline_equation_latex(
        s: List = [], 
        head: str = '',
        max_line_length: Optional[int] = None, 
        environment: str = 'aligned',
        equal_sign: str = '=',
        allow_sort: bool = True,
    ) -> str:
    """
    Automatically split a sum into multilines.
    """

    if len(s) == 0:
        return head
    if max_line_length is None:
        max_line_length = max(3 * len(head), 120)

    alignment = ' &'
    if environment is None or len(environment) == 0:
        alignment = ''

    if allow_sort:
        # rearrange the longer ones in the front
        s = sorted(s, key = lambda x: len(x), reverse = True)

    lines = ['%s %s %s'%(alignment, equal_sign, s[0])]
    for i in range(1, len(s)):
        expected_length = len(lines[-1]) + len(s[i])
        if abs(expected_length - max_line_length) < abs(len(lines[-1]) - max_line_length):
            lines[-1] += ' + ' + s[i]
        else:
            max_line_length = len(lines[0])
            lines.append('%s + %s'%(alignment, s[i]))

    s = ' \\\\ '.join(lines)
    if environment is not None and len(environment) > 0:
        s = '\\begin{%s} %s %s \\end{%s}'%(environment, head, s, environment)
    else:
        s = '%s %s'%(head, s)
    return s



class Solution():
    PRINT_CONFIGS = {
        'WITH_CYC': False,
        'MULTILINE_ENVIRON': 'aligned',
        'MULTILINE_ALLOW_SORT': True
    }
    method = ''
    def __init__(self, problem=None, solution=None, ineq_constraints=None, eq_constraints=None, is_equal=None):
        self.problem = problem
        self.solution = solution
        self.ineq_constraints = ineq_constraints if ineq_constraints is not None else dict()
        self.eq_constraints = eq_constraints if eq_constraints is not None else dict()
        self._start_time = None
        self._end_time = None
        self._is_equal = None

    @property
    def time(self) -> float:
        """Get the elapsed time for computing the solution. Return -1. if not registered."""
        return (self._end_time - self._start_time).total_seconds() if self._end_time is not None else -1.

    def __str__(self) -> str:
        return f"Solution({self.problem})"

    def __repr__(self) -> str:
        return self.__str__()

    def copy(self) -> 'Solution':
        obj = self.__class__.__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

    @property
    def gens(self) -> List[sp.Symbol]:
        return self.problem.gens if hasattr(self.problem, 'gens') else tuple(self.problem.free_symbols)

    @property
    def is_equal(self) -> bool:
        """
        Verify whether the solution is strictly equal to the problem.
        When the problem is numerical (rather algebraic), it returns False.

        See also: is_ill.
        """
        return self._is_equal

    @property
    def is_ill(self) -> bool:
        """
        Whether the solution is ill-defined, e.g. +oo, -oo, NaN, etc.
        This avoids bugs when encountering 0/0, etc.
        """
        if self.solution in (None, S.NaN, S.Infinity, S.NegativeInfinity, S.ComplexInfinity):
            return True
        if self.solution is S.Zero and isinstance(self.problem, Poly) and not self.problem.is_zero:
            return True
        return False

    def _str_f(self, name='f') -> str:
        return "%s(%s)"%(name, ','.join(str(_) for _ in self.gens))

    @property
    def str_latex(self) -> str:
        s = sp.latex(self.solution)
        equal_sign = '=' if self.is_equal else '\\approx'
        if not self.PRINT_CONFIGS['WITH_CYC']:
            s = s.replace('_{\\mathrm{cyc}}', '')
        return "$$%s %s %s$$"%(self._str_f(), equal_sign, s)

    @property
    def str_txt(self) -> str:
        CyclicExpr.PRINT_WITH_PARENS = False
        # s = _solution_latex_to_txt(self.str_latex)
        s = str(self.solution)
        equal_sign = '=' if self.is_equal else '≈'
        replacements = dict([
            ('**', '^'),
            ('*', ' '),
            *map(lambda x: ('^%d'%(x[0]+2), x[1]), enumerate('²³⁴⁵⁶⁷⁸⁹'))
        ])
        for old, new in replacements.items():
            s = s.replace(old, new)
        s = '%s %s %s'%(self._str_f(), equal_sign, s)
        return s

    @property
    def str_formatted(self):
        CyclicExpr.PRINT_WITH_PARENS = True
        s = str(self.solution)
        equal_sign = '=' if self.is_equal else '≈'
        replacements = dict([
            (' ',''),
            ('**','^'),
            ('*',''),
            ('Σ', 's'),
            ('∏', 'p'),
        ])
        for old, new in replacements.items():
            s = s.replace(old, new)
        s = '%s %s %s'%(self._str_f(), equal_sign, s)
        return s

    def as_simple_solution(self):
        """
        When the solution is a sympy expression class, it is converted to SolutionSimple.
        """
        sol = SolutionSimple(problem = self.problem, solution = self.solution,
            ineq_constraints = self.ineq_constraints, eq_constraints = self.eq_constraints, is_equal = self.is_equal)
        return sol

    def signsimp(self) -> 'Solution':
        """
        Make a copy of the solution and apply signsimp on it.
        """
        self = self.copy()
        self.solution = sp.signsimp(self.solution)
        return self

    def xreplace(self, *args, **kwargs) -> 'Solution':
        """
        Make a copy of the solution and apply xreplace on it.
        """
        self = self.copy()
        self.solution = self.solution.xreplace(*args, **kwargs)
        return self

    def doit(self, *args, **kwargs) -> sp.Expr:
        """
        Return the evaluated solution expression (by expanding CyclicSum, etc.).
        """
        return self.solution.doit(*args, **kwargs)

    def as_expr(self, *args, **kwargs) -> sp.Expr:
        """
        Return the solution as an expression. It is equivalent to .solution.
        """
        return self.solution#.doit(*args, **kwargs)

    def rewrite_symmetry(self, symbols: Tuple[sp.Symbol], perm_group: PermutationGroup) -> 'Solution':
        """
        Rewrite the expression heuristically with respect to the given permutation group.
        After rewriting, it is expected all cyclic expressions are expanded or in the given permutation group.
        This avoids the ambiguity of the cyclic expressions.

        It makes a copy of the solution and applies the rewriting on it. Note that
        the rewriting is not reversible if cyclic expressions are expanded. If this
        method is to be called multiple times, it is recommended to call on the original solution.

        Parameters
        ----------
        symbols : Tuple[sp.Symbol]
            The symbols that the permutation group acts on.
        perm_group : PermutationGroup
            Sympy permutation group object.
        
        Returns
        ----------
        Solution
            A new solution object with the rewritten expression.

        See also
        ----------
        rewrite_symmetry
        """
        self = self.copy()
        self.solution = rewrite_symmetry(self.solution, symbols, perm_group)
        return self

# class SolutionNull(Solution):
#     def __init__(self, problem = None, solution = None):
#         super().__init__(problem = problem, solution = None)


def _arg_sqr_core(arg):
    if arg.is_constant():
        return S.One
    if isinstance(arg, sp.Symbol):
        return arg
    if arg.is_Pow:
        return S.One if arg.args[1] % 2 == 0 else arg.args[0]
    if arg.is_Mul:
        return sp.Mul(*[_arg_sqr_core(x) for x in arg.args])
    if isinstance(arg, CyclicProduct):
        return CyclicProduct(_arg_sqr_core(arg.args[0]), arg.symbols).doit()
    return None


class SolutionSimple(Solution):
    """
    Most of SOS solutions can be represented in f(a,b,c) = (some sympy expression that is trivially nonnegative),
    where the original problem is on the one side and the solution is on the other side.
    This class is designed to handle such cases.
    """
    def __init__(self, problem=None, solution=None, ineq_constraints=None, eq_constraints=None, is_equal=None):
        self.problem = problem
        self.solution = solution
        self.ineq_constraints = ineq_constraints if ineq_constraints is not None else dict()
        self.eq_constraints = eq_constraints if eq_constraints is not None else dict()
        self._start_time = 0
        self._end_time = 0
        self._is_equal = is_equal

    @property
    def is_equal(self):
        if self._is_equal is None:
            symbols = self.gens # | set(self.numerator.free_symbols) | set(self.multiplier.free_symbols)
            difference = (self.problem  * self.multiplier - self.numerator)
            difference = difference.doit().as_poly(*symbols)
            self._is_equal = difference.is_zero

        return self._is_equal

    def _str_multiplier(self, add_paren = None, get_str = None):
        """
        Util function for printing.
        """
        if isinstance(self.multiplier, (int, sp.Integer)):
            if self.multiplier == 1:
                s_multiplier = self._str_f()
            else:
                s_multiplier = "%s %s"%(self.multiplier, self._str_f())
        else:
            s_multiplier = get_str(self.multiplier)
            if isinstance(self.multiplier, sp.Add) or isinstance(self.multiplier, CyclicSum):
            # if s_multiplier[-1] != ')':
                s_multiplier = "%s %s"%(add_paren(s_multiplier), self._str_f())
            else:
                s_multiplier = "%s %s"%(s_multiplier, self._str_f())
        return s_multiplier

    def _str_extract_constant_afront(self, add_paren = None, get_str = None):
        """
        Util function for printing. Extract the constant of CyclicSum, e.g.
        4 CyclicSum(a) / 7 shall be written in the form of 4/7 CyclicSum(a).
        """
        s_args = []
        if isinstance(self.numerator, sp.Add):
            numerator_args = self.numerator.args
        else:
            numerator_args = [self.numerator]
        for x in numerator_args:
            # if isinstance(x, sp.Mul) and len(x.args) >= 2 and x.args[0].is_constant():
            #     if len(x.args) == 2:
            #         # extract the rational coefficient and move it to the front
            #         if precedence_traditional(x.args[1]) < PRECEDENCE["Mul"]:
            #             s_args.append(get_str(x.args[0]) + '%s'%add_paren(get_str(x.args[1])))
            #         else:
            #             s_args.append(get_str(x.args[0]) + get_str(x.args[1]))
            #     else:
            #         s_args.append(get_str(x.args[0]) + get_str(sp.Mul(*x.args[1:])))
            # else:
            #     s_args.append(get_str(x))

            x_as_coeff_Mul = x.as_coeff_Mul()
            if x_as_coeff_Mul[0] is S.One or x_as_coeff_Mul[1] is S.One:
                s_args.append(get_str(x))
            elif precedence_traditional(x_as_coeff_Mul[1]) < PRECEDENCE["Mul"]:
                s_args.append(get_str(x_as_coeff_Mul[0]) + '%s'%add_paren(get_str(x_as_coeff_Mul[1])))
            else:
                s_args.append(get_str(x_as_coeff_Mul[0]) + get_str(x_as_coeff_Mul[1]))

        return s_args

    @property
    def str_latex(self):
        """
        1. Move the multiplier to the left hand side to cancel the denominator.
        2. Move rational coefficients in front of each term.
        """
        parener = lambda x: "\\left(%s\\right)"%x

        s_multiplier = self._str_multiplier(add_paren = parener, get_str = lambda x: sp.latex(x))

        s_args = self._str_extract_constant_afront(add_paren = parener, get_str = lambda x: sp.latex(x))

        s = _multiline_equation_latex(
            s_args, 
            head = s_multiplier,
            equal_sign = '=' if self.is_equal else '\\approx',
            environment = self.PRINT_CONFIGS['MULTILINE_ENVIRON'],
            allow_sort = self.PRINT_CONFIGS['MULTILINE_ALLOW_SORT']
        )
    
        # s = s.replace('\\left(- a + c\\right)^{2}', '\\left(a - c\\right)^{2}')

        s = '$$%s$$'%s


        if not self.PRINT_CONFIGS['WITH_CYC']:
            s = s.replace('_{\\mathrm{cyc}}', '')
        return s

    @property
    def str_txt(self):
        CyclicExpr.PRINT_WITH_PARENS = False
        equal_sign = '=' if self.is_equal else '≈'

        parener = lambda x: "(%s)"%x
        s_multiplier = self._str_multiplier(add_paren = parener, get_str = lambda x: str(x))
        s_args = self._str_extract_constant_afront(add_paren = parener, get_str = lambda x: str(x))
        s = '+'.join(s_args)
        s = '%s%s%s'%(s_multiplier, equal_sign, s)

        replacements = dict([
            (' ',''),
            ('**', '^'),
            ('*', ''),
            *map(lambda x: ('^%d'%(x[0]+2), x[1]), enumerate('²³⁴⁵⁶⁷⁸⁹'))
        ])
        for old, new in replacements.items():
            s = s.replace(old, new)
        return s

    @property
    def str_formatted(self):
        CyclicExpr.PRINT_WITH_PARENS = True
        equal_sign = '=' if self.is_equal else '≈'

        parener = lambda x: "(%s)"%x
        s_multiplier = self._str_multiplier(add_paren = parener, get_str = lambda x: str(x))
        s_args = self._str_extract_constant_afront(add_paren = parener, get_str = lambda x: str(x))
        s = '+'.join(s_args)
        s = '%s%s%s'%(s_multiplier, equal_sign, s)

        replacements = dict([
            (' ',''),
            ('**','^'),
            ('*',''),
            ('Σ', 's'),
            ('∏', 'p'),
        ])
        for old, new in replacements.items():
            s = s.replace(old, new)
        return s

    def dehomogenize(self, homogenizer: Optional[sp.Symbol] = None):
        """
        Dehomogenize the solution.
        """
        if homogenizer is None:
            return self

        self = self.copy()
        self.problem = self.problem.subs(homogenizer, 1)
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
        self.solution = dehom(self.solution)
        return self

    def as_fraction(self, together = True, inplace = False):
        """
        Denest the fractions and express the solution as the division of two fraction-free expressions.
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
        if isinstance(m, sp.Add):
            numerator = sp.Add(*[v * arg for arg in m.args])

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
