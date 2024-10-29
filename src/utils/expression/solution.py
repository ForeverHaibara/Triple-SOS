from typing import Optional, List, Union
import re

import sympy as sp
from sympy.core.singleton import S
from sympy.printing.precedence import precedence_traditional, PRECEDENCE

from ..expression.cyclic import is_cyclic_expr, CyclicSum, CyclicProduct, CyclicExpr

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
    def __init__(self, problem: Optional[sp.Poly] = None, solution: Optional[sp.Expr] = None, is_equal: bool = None):
        self.problem: sp.Poly = problem
        self.solution: sp.Expr = solution
        self.is_equal_ = is_equal

    def __str__(self) -> str:
        return f"{self.problem} = {self.solution}"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def gens(self) -> List[sp.Symbol]:
        return self.problem.gens

    @property
    def is_equal(self) -> bool:
        """
        Verify whether the solution is strictly equal to the problem.
        When the problem is numerical (rather algebraic), it returns False.

        See also: is_ill.
        """
        return self.is_equal_

    @property
    def is_ill(self) -> bool:
        """
        Whether the solution is ill-defined, e.g. +oo, -oo, NaN, etc.
        This avoids bugs when encountering 0/0, etc.
        """
        if self.solution in (None, S.NaN, S.Infinity, S.NegativeInfinity, S.ComplexInfinity):
            return True
        if self.solution is S.Zero and not self.problem.is_zero:
            return True
        return False

    def _str_f(self) -> str:
        return "f(%s)"%(','.join(str(_) for _ in self.gens))

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
        When the expression is a nested fraction, we can simplify it.
        """
        numerator, multiplier = sp.fraction(sp.together(self.solution))
    
        if multiplier.is_constant():
            const, multiplier = S.One, multiplier
        else:
            const, multiplier = multiplier.as_coeff_Mul()

        if isinstance(numerator, sp.Add):
            numerator = sp.Add(*[arg / const for arg in numerator.args])
        else:
            numerator = numerator / const

        return SolutionSimple(
            problem = self.problem, 
            numerator = numerator,
            multiplier = multiplier,
            is_equal = self.is_equal_
        )

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
    All (rational) SOS solutions can be presented in the form of f(a,b,c) = g(a,b,c) / h(a,b,c)
    where g and h are polynomials.
    """
    def __init__(self, problem = None, numerator = None, multiplier = None, is_equal = None):
        if multiplier is None:
            multiplier = S.One
        self.problem = problem
        self.solution = numerator / multiplier
        self.numerator = numerator
        self.multiplier = multiplier
        self.is_equal_ = is_equal

    @property
    def is_equal(self):
        if self.is_equal_ is None:
            symbols = self.gens # | set(self.numerator.free_symbols) | set(self.multiplier.free_symbols)
            difference = (self.problem  * self.multiplier - self.numerator)
            difference = difference.doit().as_poly(*symbols)
            self.is_equal_ = difference.is_zero

        return self.is_equal_

    def as_congruence(self):
        """
        Note that (part of) g(a,b,c) can be represented sum of squares. For example, polynomial of degree 4 
        has form [a^2,b^2,c^2,ab,bc,ca] * M * [a^2,b^2,c^2,ab,bc,ca]' where M is positive semidefinite matrix.

        We can first reconstruct and M and then find its congruence decomposition, 
        this reduces the number of terms.

        WARNING: WE ONLY SUPPORT 3-VAR CASE.
        """
        if not isinstance(self.numerator, sp.Add):
            # in this case, the numerator is already simplified
            return self

        return self

        # not implemented yet

        # now we only handle cyclic expressions
        sqr_args = {}
        unsqr_args = []
        
        def _is_symbol(s):
            return isinstance(s, sp.Symbol) and s in self.problem.symbols

        def _is_core_monomial(core):
            """Whether core == a^i * b^j * c^k."""
            if core.is_constant() or _is_symbol(core):
                return True
            if isinstance(core, sp.Pow) and _is_symbol(core.args[0]):
                return True
            if isinstance(core, sp.Mul):
                return all(_is_core_monomial(x) for x in core.args)
            return False

        for arg in self.numerator.args:
            core = None
            if is_cyclic_expr(arg, self.problem.symbols):
                if isinstance(arg, CyclicSum):
                    core = _arg_sqr_core(arg.args[0])
                else:
                    core = _arg_sqr_core(arg)
                if not _is_core_monomial(core):
                    core = None

            if core is not None:
                # reduce monomial core once more, e.g. a^4 b^5 c^3 -> bc
                core = _arg_sqr_core(core)
                if len(core.free_symbols) not in sqr_args:
                    sqr_args[len(core.free_symbols)] = []
                sqr_args[len(core.free_symbols)].append(arg)
            else:
                unsqr_args.append(arg)

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

        self.problem = self.problem.subs(homogenizer, 1)
        self.numerator = self.numerator.xreplace({homogenizer: 1})
        self.multiplier = self.multiplier.xreplace({homogenizer: 1})
        self.solution = self.numerator / self.multiplier
        return self

    def as_content_primitive(self):
        """
        Move the constant of the multiplier to the numerator.
        """
        # return self.multiplier.as_content_primitive()
        v, m = self.multiplier.as_content_primitive()
        if v < 0:
            v, m = -v, -m
        self.multiplier = m

        inv_v = S.One/v
        self.numerator = inv_v * self.numerator
        v, m = self.numerator.as_content_primitive()
        if v < 0:
            v, m = -v, -m
        if isinstance(m, sp.Add):
            self.numerator = sp.Add(*[v * arg for arg in m.args])

        self.solution = self.numerator / self.multiplier
        return self

    def signsimp(self):
        """
        Simplify the signs.
        """
        self.numerator = sp.signsimp(self.numerator)
        self.multiplier = sp.signsimp(self.multiplier)
        self.solution = self.numerator / self.multiplier
        return self