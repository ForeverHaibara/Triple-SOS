from typing import Optional, List, Union
import re

import numpy as np
import sympy as sp
from sympy.core.singleton import S
from sympy.printing.precedence import precedence_traditional, PRECEDENCE

from ..expression.cyclic import is_cyclic_expr, CyclicSum, CyclicProduct, CyclicExpr
from ..basis_generator import generate_expr
from ..roots.rationalize import cancel_denominator

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
    def __init__(self, problem = None, solution = None, is_equal = None):
        self.problem = problem
        self.solution = solution
        self.is_equal_ = is_equal

    def __str__(self) -> str:
        return f"{self.problem} = {self.solution}"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def is_equal(self):
        return self.is_equal_

    @property
    def str_latex(self):
        s = sp.latex(self.solution)
        equal_sign = '=' if self.is_equal else '\\approx'
        if not self.PRINT_CONFIGS['WITH_CYC']:
            s = s.replace('_{\\mathrm{cyc}}', '')
        return "$$f(a,b,c) %s %s$$"%(equal_sign, s)

    @property
    def str_txt(self):
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
        s = 'f(a,b,c) %s %s'%(equal_sign, s)
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
        s = 'f(a,b,c) %s %s'%(equal_sign, s)
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
            symbols = set(self.problem.gens) # | set(self.numerator.free_symbols) | set(self.multiplier.free_symbols)
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
                s_multiplier = "f(a,b,c)"
            else:
                s_multiplier = "%s f(a,b,c)"%self.multiplier
        else:
            s_multiplier = get_str(self.multiplier)
            if s_multiplier[-1] != ')':
                s_multiplier = "%s f(a,b,c)"%add_paren(s_multiplier)
            else:
                s_multiplier = "%s f(a,b,c)"%s_multiplier
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
            if x_as_coeff_Mul[0] is S.One:
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
    
        s = s.replace('\\left(- a + c\\right)^{2}', '\\left(a - c\\right)^{2}')

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



def congruence(M: Union[sp.Matrix, np.ndarray]) -> Union[None, tuple]:
    """
    Write a symmetric matrix as a sum of squares.
    M = U.T @ S @ U where U is upper triangular and S is diagonal.

    Returns
    -------
    U : sp.Matrix | np.ndarray
        Upper triangular matrix.
    S : sp.Matrix | np.ndarray
        Diagonal vector (1D array).

    Return None if M is not positive semidefinite.
    """
    M = M.copy()
    n = M.shape[0]
    if isinstance(M[0,0], sp.Expr):
        U, S = sp.Matrix.zeros(n), sp.Matrix.zeros(n, 1)
        One = sp.S.One
    else:
        U, S = np.zeros((n,n)), np.zeros(n)
        One = 1
    for i in range(n-1):
        if M[i,i] > 0:
            S[i] = M[i,i]
            U[i,i+1:] = M[i,i+1:] / (S[i])
            U[i,i] = One
            M[i+1:,i+1:] -= U[i:i+1,i+1:].T @ (U[i:i+1,i+1:] * S[i])
        elif M[i,i] < 0:
            return None
        elif M[i,i] == 0 and any(_ for _ in M[i+1:,i]):
            return None
    U[-1,-1] = One
    S[-1] = M[-1,-1]
    if S[-1] < 0:
        return None
    return U, S



def congruence_as_sos(M, multiplier = S.One, symbols = 'a b c', cancel = True, cyc = True):
    # (n+1)(n+2)/2 = M.shape[0]
    n = round((M.shape[0] * 2 + .25)**.5 - 1.5)
    U, S = congruence(M)

    if isinstance(symbols, str):
        symbols = sp.symbols(symbols)
    a, b, c = symbols

    exprs = []
    coeffs = []

    monoms = generate_expr(n, cyc = 0)[1]
    for i, s in enumerate(S):
        if s == 0:
            continue
        val = sp.S(0)
        if cancel:
            r = cancel_denominator(U[i,i:])
        for j in range(i, len(monoms)):
            monom = monoms[j]
            val += U[i,j] / r * a**monom[0] * b**monom[1] * c**monom[2]
        exprs.append(val**2)
        coeffs.append(s * r**2)

    exprs = [multiplier * expr for expr in exprs]
    if cyc:
        exprs = [CyclicSum(expr, symbols) for expr in exprs]

    exprs = [coeff * expr for coeff, expr in zip(coeffs, exprs)]
    expr = sp.Add(*exprs)

    return expr

