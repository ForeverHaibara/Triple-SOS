from typing import Union, List, Tuple, Optional, Dict
from functools import partial
from collections import defaultdict
# import re

from sympy import Expr, Poly, QQ, RR, Rational, Integer, Float, Symbol
from sympy import parse_expr, sympify, fraction, cancel, latex
from sympy import symbols as sp_symbols
from sympy.polys import ring
from sympy.polys.polyclasses import DMP
from sympy.combinatorics import Permutation, PermutationGroup, CyclicGroup
from sympy.printing.precedence import precedence_traditional, PRECEDENCE

from .expression import Coeff, CyclicSum, CyclicProduct
from .monomials import poly_reduce_by_symmetry

def cycle_expansion(
        f: str,
        symbol: str = 's',
        gens: Tuple[Symbol] = sp_symbols("a b c"),
        perm: Optional[PermutationGroup] = None
    ) -> str:
    """
    Parameters
    -------
    f: str
        The string of the polynomial with given gens.
    symbol: char, 's' or 'p'
        When symbol == 's':
            a^3 * b^2 * c   ->   a^3 * b^2 * c + b^3 * c^2 * a + c^3 * a^2 * b
        When symbol == 'p':
            a^3 * b^2 * c   ->   a^3 * b^2 * c * b^3 * c^2 * a * c^3 * a^2 * b
        Warning : Please add parenthesis yourself before expansion if necessary.
    gens: Tuple[Symbol]
        The generators of the polynomial.
    perm: Optional[PermutationGroup]
        The permutation group of the expression. If None, it will be cyclic group.

    Returns
    -------
    str
        A string, the result of the cycle expansion.
    """
    if perm is None:
        perm = CyclicGroup(len(gens))

    original_names = [ord(_.name) for _ in gens]
    translations = [
        dict(zip(original_names, p(original_names))) for p in perm.elements
    ]
    symbol = ' * ' if symbol == 'p' else ' + '
    return symbol.join(f.translate(t) for t in translations)

##########################################################################
#
#                             Preprocess text
#
##########################################################################

def _preprocess_text_delatex(poly: str, funcs: Dict[str, Tuple[str, int]]) -> str:
    """
    Convert a latex formula to standard representation. This is experimental and unsafe.
    """

    # frac{..}{...} -> (..)/(...)
    poly = poly.replace(' ','')
    poly = poly.replace('left','')
    poly = poly.replace('right','')
    poly = poly.replace('}{' , ')/(')
    poly = poly.replace('frac' , '')
    poly = poly.translate({123: 40, 125: 41, 92: 32, 36: 32, 91: 40, 93: 41, 65288: 40, 65289: 41}) # { -> ( , } -> ) , \ -> space, $ -> space, [ -> (, ] -> )
    poly = poly.replace(' ','')

    # sum ... -> s(...)
    # prod ... -> p(...)
    parenthesis = 0
    paren_depth = [-1]
    precedences = []
    i = 0

    if '' in funcs:
        funcs.pop('')
    funckeys = sorted(list(funcs), key=lambda x: -len(x))
    def _match_pattern(poly, i):
        for pattern in funckeys:
            if len(poly) >= i + len(pattern) and poly[i:i+len(pattern)] == pattern:
                return pattern
        return None

    while i < len(poly):
        matched = _match_pattern(poly, i)
        if matched is not None:
            parenthesis += 1
            paren_depth.append(parenthesis)
            replacement, precedence = funcs[matched]
            precedences.append(precedence)
            poly = poly[:i] + replacement + '(' + poly[i+len(matched):]
            i += len(replacement) + 1
            continue
        elif poly[i] == '(':
            parenthesis += 1
        elif parenthesis == paren_depth[-1]:
            # auto parenthesize
            need = False
            if precedences[-1] <= PRECEDENCE["Add"]:
                need = poly[i] in '+-'
            elif precedences[-1] <= PRECEDENCE["Mul"]:
                need = poly[i] in '+-(*/' # or poly[i].isalpha()
            if need:
                # auto parenthesize
                poly = poly[:i] + ')' + poly[i:]
                parenthesis -= 1
                paren_depth.pop()
                precedences.pop()
        elif poly[i] == ')':
            parenthesis -= 1
        i += 1
    poly += ')' * (len(paren_depth) - 1)

    return poly


def _preprocess_text_expansion(poly: str, gens: Tuple[Symbol], perm: PermutationGroup) -> str:
    """
    Expand the polynomial with cycle expansion.

    s(ab)    ->   (ab + bc + ca)

    p(a+b)   ->   (a+b)(b+c)(c+a)
    """
    parenthesis = 0
    paren_depth = [-1]
    cycle_begin = []
    _cyc_expand = partial(cycle_expansion, gens=gens, perm=perm)

    i = 0
    while i < len(poly):
        if (poly[i] == 's' and i+4 <= len(poly) and poly[i:i+4]!='sqrt')  or poly[i] == 'p':
            paren_depth.append(parenthesis)
            cycle_begin.append(i)
        elif poly[i] == '(':
            parenthesis += 1
        elif poly[i] == ')':
            parenthesis -= 1
            if paren_depth[-1] == parenthesis:
                tmp = '(' + _cyc_expand(poly[cycle_begin[-1]+1:i+1], symbol=poly[cycle_begin[-1]]) + ')'
                poly = poly[:cycle_begin[-1]] + tmp + poly[i+1:]
                i = cycle_begin[-1] + len(tmp) - 1
                paren_depth.pop()
                cycle_begin.pop()
        i += 1
    return poly


def _preprocess_text_completion(
        poly: str,
        scientific_notation: bool = False,
        preserve_patterns: List[str] = ('sqrt',)
    ) -> str:
    """
    Complete the polynomial with * and ^. E.g.
    1/5a3b2c   ->   1/5*a^3*b^2*c

    Parameters
    ----------
    poly: str
        The polynomial to complete.
    scientific_notation: bool
        Whether to parse the scientific notation. If True, 1e2 will be parsed as 100.
        If False, 1e2 will be parsed as e^2 where e is a free variable.
    """
    SCI = 'e' if scientific_notation else ''
    preserve_patterns = set(preserve_patterns)
    if scientific_notation:
        preserve_patterns.add('e')
    if '' in preserve_patterns:
        preserve_patterns.remove('') # will cause infinite loop
    preserve_patterns = sorted(list(preserve_patterns), key=lambda x: -len(x))
    def _pattern_match(poly, i):
        lenp = len(poly)
        for pattern in preserve_patterns:
            if lenp >= i + len(pattern) and poly[i:i+len(pattern)] == pattern:
                return pattern
        return None
    poly = poly.replace(' ','')
    i = 0
    while i < len(poly) - 1:
        if poly[i].isdigit(): # '0'~'9'
            if poly[i+1] == '(' or (poly[i+1].isalpha() and poly[i+1] != SCI): # alphabets
                # when using scientific notation, e.g. '1e' should not be converted to '1*e'
                poly = poly[:i+1] + '*' + poly[i+1:]
                i += 1
        elif poly[i] == ')' or poly[i].isalpha(): # alphabets
            matched = _pattern_match(poly, i)
            if matched is not None:
                i += len(matched) - 1
                if i + 1 < len(poly):
                    if poly[i+1].isalpha(): # alphabets
                        poly = poly[:i+1] + '*' + poly[i+1:]
                        i += 1
            elif poly[i+1] == '(' or poly[i+1].isalpha():
                poly = poly[:i+1] + '*' + poly[i+1:]
                i += 1
            elif poly[i+1].isdigit(): # '0'~'9'
                poly = poly[:i+1] + '^' + poly[i+1:]
                i += 1
        i += 1

    return poly


def expand_poly(expr: Expr, gens=None) -> Union[Expr, Poly]:
    """
    Faster implementation of `sympy.expand`. This is experimental.
    """
    class UnhandledExpr(Exception):
        pass
    expr = sympify(expr)
    symbols = tuple(sorted(list(expr.free_symbols), key=lambda x:x.name))
    other_symbols = []
    if gens is None:
        gens = symbols
    else:
        other_symbols = [_ for _ in symbols if _ not in gens]

    if len(gens) == 0:
        return expr
    if not all(_.is_commutative for _ in symbols):
        return expr.expand()
    dom = QQ if not expr.has(Float) else RR
    dom_ext = dom[other_symbols] if len(other_symbols) else dom
    expr_ring = ring(gens, dom_ext)[0]
    def _expandpoly(_):
        if _.is_Symbol or _.is_Rational or _.is_Float:
            return expr_ring(_)

        elif _.is_Add:
            s = expr_ring.zero
            for i in _.args:
                s = s + _expandpoly(i)
            return s

        elif _.is_Mul:
            s = expr_ring.one
            for i in _.args:
                s = s * _expandpoly(i)
            return s

        elif _.is_Pow:
            if isinstance(_.exp, Integer) and int(_.exp) >= 0:
                return _expandpoly(_.base) ** int(_.exp)

        elif isinstance(_, (CyclicSum, CyclicProduct)):
            arg0dict = _expandpoly(_.args[0]).to_dict()
            symbol_inds = [gens.index(i) for i in _.args[1]]
            n = len(symbol_inds)
            new_args = []
            for p in _.args[2].elements: # permutations
                new_inds = p(symbol_inds)
                new_perm = list(range(len(gens)))
                for i in range(n):
                    new_perm[symbol_inds[i]] = new_inds[i]
                new_dict = {}
                for k, v in arg0dict.items():
                    new_dict[tuple(k[i] for i in new_perm)] = v
                new_args.append(expr_ring.from_dict(new_dict))
            if isinstance(_, CyclicSum):
                s = expr_ring.zero
                for i in new_args:
                    s = s + i
                return s
            elif isinstance(_, CyclicProduct):
                s = expr_ring.one
                for i in new_args:
                    s = s * i
                return s
        raise UnhandledExpr
    arg0 = None
    try:
        arg0 = _expandpoly(expr)
        dmp = DMP.from_dict(arg0.to_dict(), len(gens)-1, dom_ext)
        arg0 = Poly.new(dmp, *gens)
    except UnhandledExpr:
        arg0 = expr.expand()
    return arg0


def preprocess_text(
        poly: str,
        gens: Tuple[Symbol] = sp_symbols("a b c"),
        perm: Optional[PermutationGroup] = None,
        return_type: str = "poly",
        cyclic_sum_func: str = 's',
        cyclic_prod_func: str = 'p',
        scientific_notation: bool = False,
        lowercase: bool = True,
        latex: bool = False,
        preserve_patterns: List[str] = ('sqrt',),
        parse_expr_kwargs: Optional[Dict] = None,
    ) -> Union[str, Expr, Poly, Tuple[Poly, Poly]]:
    """
    Parse a text to a sympy polynomial with respect to the given generators conveniently.
    The function assumes each variable to be a single character.
    For more general cases, please do not rely on this function.

    Parameters
    ----------
    return_type: str
        One of ['text', 'expr', 'poly', 'frac'].
        If 'text', return the text of the expression.
        If 'expr', return the sympy expression of the expression.
        If 'poly', return the sympy polynomial of the expression.
        If 'frac', return a tuple of sympy polynomials (numerator, denominator). If
            it fails to cancel the polynomial, return (None, None).
    gens: Tuple[Symbol]
        The generators of the cyclic sum or products.
    perm: Optional[PermutationGroup]
        The permutation group of the expression. If None, it will be cyclic group.
    cyclic_sum_func: str
        Stands for the cyclic sum. Defaults to 's'.
    cyclic_prod_func: str
        Stands for the cyclic product. Defaults to 'p'.
    scientific_notation: bool
        Whether to parse the scientific notation. If True, 1e2 will be parsed as 100.
        If False, 1e2 will be parsed as e^2 where e is a free variable.
    lowercase: bool
        Whether to convert the text to lowercase. Defaults to True.
    latex: bool
        Whether to parse the latex expression. THIS IS EXPERIMENTAL AND UNSAFE.
        Defaults to False.
    preserve_patterns: List[str]
        The patterns to be preserved when completing the text. Defaults to ['sqrt'].
    parse_expr_kwargs: Dict
        The arguments for sympy sympyify.

    Returns
    --------
    See return_type.


    Examples
    --------
    By default, the function will return a sympy polynomial with respect to a, b, c.
    Omitted multiplication signs and powers will be completed.
    >>> from sympy.abc import x, y, z, a, b, c
    >>> preprocess_text('xa+1/4yb2+z2c3')
    Poly(x*a + y/4*b**2 + z**2*c**3, a, b, c, domain='QQ[x,y,z]')

    >>> preprocess_text('xa+1/4yb2+z2c3', (x,y,z,a,b,c))
    Poly(x*a + 1/4*y*b**2 + z**2*c**3, x, y, z, a, b, c, domain='QQ')

    >>> preprocess_text('sqrt(2)abc/x-2')
    Poly(sqrt(2)/x*a*b*c - 2, a, b, c, domain='EX')

    >>> preprocess_text('1+a2/b') is None # since it is not a polynomial in a, b, c
    True


    Configure the return type.

    >>> preprocess_text('1+a2/b', return_type='expr')
    a**2/b + 1
    >>> preprocess_text('3/4a3b2c + (x2)2', return_type='text')
    '3/4*a^3*b^2*c+(x^2)^2'
    >>> preprocess_text('a/(b+c)+b/(c+a)', return_type='frac')
    (Poly(a**2 + a*c + b**2 + b*c, a, b, c, domain='QQ'), Poly(a*b + a*c + b*c + c**2, a, b, c, domain='QQ'))


    Strings 's' and 'p' are used to represent the cyclic sums and products respectively.
    They are computed with respect to the given generators and the permutation group.

    >>> preprocess_text('s(a2b-c)+3/4p(a3)')
    Poly(3/4*a**3*b**3*c**3 + a**2*b + a*c**2 - a + b**2*c - b - c, a, b, c, domain='QQ')
    >>> preprocess_text('s(x(x-y)(x-z))')
    Poly(3*x**3 - 3*x**2*y - 3*x**2*z + 3*x*y*z, a, b, c, domain='QQ[x,y,z]')
    >>> preprocess_text('s(x2-xy)', (x,y,z))
    Poly(x**2 - x*y - x*z + y**2 - y*z + z**2, x, y, z, domain='QQ')

    >>> from sympy.combinatorics import SymmetricGroup
    >>> preprocess_text('s(x2-xy)', (x,y,z), SymmetricGroup(3))
    Poly(2*x**2 - 2*x*y - 2*x*z + 2*y**2 - 2*y*z + 2*z**2, x, y, z, domain='QQ')

    >>> preprocess_text('s(1/x2)', (x,a,b), return_type='expr').doit()
    x**(-2) + b**(-2) + a**(-2)

    Configure cyclic_sum_func and cyclic_prod_func to use other strings for cyclic sums and products.
    Sometimes the expression involves uppercase letters, and it requires to set lowercase to False.

    >>> preprocess_text('Σ(x(y-z)2) - s', (x,y,z), cyclic_sum_func='Σ', lowercase=False)
    Poly(x**2*y + x**2*z + x*y**2 - 6*x*y*z + x*z**2 + y**2*z + y*z**2 - s, x, y, z, domain='QQ[s]')


    To avoid certain patterns from being completed, set preserve_patterns.

    >>> preprocess_text('cbrt(x2)+y2',return_type='expr')
    b*c*r*t*x**2 + y**2
    >>> preprocess_text('cbrt(x2)+y2',return_type='expr',preserve_patterns=('cbrt','x'))
    x2**(1/3) + y**2
    >>> preprocess_text('x1x2x3-y1y2y3',return_type='expr',preserve_patterns=('y',))
    x**6 - y1*y2*y3


    See also
    ---------
    pl, degree_of_zero, sympify, parse_expr
    """
    if lowercase:
        poly = poly.lower()

    if perm is None:
        perm = CyclicGroup(len(gens))

    if latex:
        poly = _preprocess_text_delatex(poly,
            funcs = {'sum': (cyclic_sum_func, PRECEDENCE["Add"]),
                     'prod': (cyclic_prod_func, PRECEDENCE["Mul"])}
                    #  'sqrt': ('sqrt', PRECEDENCE["Mul"])}
        )
    else:
        poly = poly.replace(' ','')
        poly = poly.translate({123: 40, 125: 41, 92: 32, 36: 32, 91: 40, 93: 41, 65288: 40, 65289: 41})
    # poly = _preprocess_text_expansion(poly, gens, perm)

    preserve_patterns = set(preserve_patterns)
    if cyclic_sum_func: preserve_patterns.add(cyclic_sum_func)
    if cyclic_prod_func: preserve_patterns.add(cyclic_prod_func)
    poly = _preprocess_text_completion(poly,
        scientific_notation=scientific_notation,
        preserve_patterns=preserve_patterns
    )

    if return_type == 'text':
        return poly

    if parse_expr_kwargs is None:
        parse_expr_kwargs = {}
    if 'local_dict' not in parse_expr_kwargs:
        parse_expr_kwargs['local_dict'] = {}
    else:
        parse_expr_kwargs['local_dict'] = parse_expr_kwargs['local_dict'].copy()

    for s in gens:
        parse_expr_kwargs['local_dict'][s.name] = s

    _cyclic_sum = lambda x: CyclicSum(x, gens, perm)
    _cyclic_prod = lambda x: CyclicProduct(x, gens, perm)
    parse_expr_kwargs['local_dict'].update({cyclic_sum_func: _cyclic_sum, cyclic_prod_func: _cyclic_prod})

    poly = poly.replace('^','**')
    poly = parse_expr(poly, **parse_expr_kwargs)

    def _parse_poly(expr, gens):
        if isinstance(expr, Expr):
            expr = expand_poly(expr, gens)
        if isinstance(expr, Poly):
            if expr.gens != gens:
                expr = expr.as_poly(gens, extension = True)
            return expr
        return expr.doit().as_poly(gens, extension = True)

    if return_type == 'expr':
        return poly

    try:
        # try if it has no fractions, which avoids expanding CyclicExprs manually
        poly0 = _parse_poly(poly, gens)
        if poly0 is not None:
            if return_type == 'poly':
                return poly0
            elif return_type == 'frac':
                return poly0, Poly(1, gens)
    except Exception as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise e
        pass

    if return_type == 'frac':
        try:
            # doit is currently needed as `together` does not apply to CyclicExprs
            frac = fraction(poly.doit().together())
            # frac = fraction(cancel(poly.doit()))

            poly0 = _parse_poly(frac[0], gens)
            poly1 = _parse_poly(frac[1], gens)
            if poly0.domain.is_Numerical and poly1.domain.is_Numerical:
                poly0, poly1 = poly0.cancel(poly1, include=True)

            if len(poly1.free_symbols) == 0:
                div0, div1 = poly0.div(poly1)
                if div1.is_zero:
                    one = Poly(1, gens, extension = True)
                    return div0, one
            return poly0, poly1
        except Exception as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise e
            return None, None

    return None
    # return poly


def pl(*args, **kwargs):
    """
    Parse a text to a sympy polynomial with respect to the given generators conveniently.
    This is a shortcut for preprocess_text.
    """
    return preprocess_text(*args, **kwargs)
pl = preprocess_text


def degree_of_zero(poly: str, gens: Tuple[Symbol] = sp_symbols("a b c"), *args, **kwargs) -> int:
    """
    Infer the degree of a homogeneous zero polynomial.
    Idea: delete the additions and subtractions, which do not affect the degree.

    Parameters
    ----------
    poly: str
        The polynomial of which to infer the degree.
    gens: Tuple[Symbol]
        The generators of the polynomial.
    args, kwargs:
        Other arguments for preprocess_text.

    Returns
    -------
    int
        The degree of the (homogeneous) polynomial.

    Examples
    --------
    >>> from sympy.abc import a, b, c, x, y, z
    >>> degree_of_zero('p(a)(s(a2)2-s(a4+2a2b2))')
    7
    >>> degree_of_zero('(r2+r+1)s((x-y)2(x+y-tz)2)-s(((y-z)(y+z-tx)-r(x-y)(x+y-tz))2)', (x,y,z))
    4
    """
    if 'return_type' in kwargs:
        raise ValueError("return_type should not be specified.")
    if 'parse_expr_kwargs' in kwargs:
        kwargs['parse_expr_kwargs'] = kwargs['parse_expr_kwargs'].copy()
    else:
        kwargs['parse_expr_kwargs'] = {}
    kwargs['parse_expr_kwargs']['evaluate'] = False
    expr = preprocess_text(poly, gens, *args, **kwargs, return_type='expr')

    def _get_degree(expr):
        if expr.is_Atom:
            return 1 if expr.is_Symbol and expr in gens else 0
        elif expr.is_Add:
            return max(_get_degree(_) for _ in expr.args) if len(expr.args) else 0
        elif expr.is_Mul:
            return sum(_get_degree(_) for _ in expr.args) if len(expr.args) else 0
        elif expr.is_Pow:
            if expr.exp.is_Integer:
                return int(expr.exp) * _get_degree(expr.base)
            return 0
        elif isinstance(expr, CyclicSum):
            return _get_degree(expr.args[0])
        elif isinstance(expr, CyclicProduct):
            return _get_degree(expr.args[0]) * int(expr.args[2].order())
        return 0
    return _get_degree(expr)


def _is_single_paren(s: str) -> bool:
    """
    Infer whether the string is wrapped by a single pair of parentheses.
    """
    if not s.startswith('(') or not s.endswith(')'):
        return False
    stack = 0
    m = len(s) - 1
    for i, c in enumerate(s):
        if c == '(':
            stack += 1
        elif c == ')':
            stack -= 1
            if stack == 0 and i != m:
                return False
    return True

def _get_coeff_str(coeff, MUL = '*') -> str:
    """
    Get the coefficient string.
    """
    if isinstance(coeff, Rational):
        if coeff == 1: coeff_str = '+'
        elif coeff == -1: coeff_str = '-'
        elif coeff > 0: coeff_str = '+%s%s'%(coeff, MUL)
        else: coeff_str = '%s%s'%(coeff, MUL)
    else:
        coeff_str = str(coeff).replace('**', '^').replace(' ','')

        if _is_single_paren(coeff_str): pass
        elif coeff_str.startswith('-(') and _is_single_paren(coeff_str[1:]): pass
        else:
            coeff_str = '+(%s)'%coeff_str
        coeff_str += MUL
    return coeff_str


def poly_get_standard_form(
        poly: Poly,
        perm: Optional[PermutationGroup] = None,
        omit_mul: bool = True,
        omit_pow: bool = True,
        _is_cyc: Optional[bool] = None
    ) -> str:
    """
    Express a polynomial in the standard form.

    Parameters
    ----------
    poly : Poly
        The polynomial to be expressed.
    perm : PermutationGroup
        The permutation group to be considered. If None,
        it uses the cyclic group generated by the variables.
    omit_mul : bool
        Whether to omit the multiplication sign. Defaults to True.
    omit_pow : bool
        Whether to omit the power sign. Defaults to True.
    _is_cyc : Optional[bool]
        If it is None, it will be inferred from the permutation group.
        If given, it will be used directly.

    Examples
    --------
    >>> from sympy.abc import x, y, z, a, b, c
    >>> poly_get_standard_form(((x*a+y*b+z*c)**2).as_poly(x,y,z))
    '((a^2)x2+(2*a*b)xy+(2*a*c)xz+(b^2)y2+(2*b*c)yz+(c^2)z2)'
    >>> poly_get_standard_form(((x*a+y*b+z*c)**2).as_poly(x,y,z,a,b,c), PermutationGroup(Permutation([1,2,0,4,5,3])))
    's(x2a2+2xyab)'
    >>> poly_get_standard_form((a*(a-b)*(a-c)+b*(b-c)*(b-a)+c*(c-a)*(c-b)).as_poly(a,b,c))
    's(a3-a2b-a2c+abc)'
    """
    if poly.total_degree() == 0:
        # is constant polynomial
        s = str(poly.coeff_monomial((0,) * len(poly.gens)))
        s = s.replace('**', '^').replace(' ','')
        if not _is_single_paren(s):
            s = '(%s)'%s
        return s

    if perm is None:
        perm = CyclicGroup(len(poly.gens))
    if _is_cyc is None:
        _is_cyc = Coeff(poly).is_cyclic(perm)

    extracted = []
    if _is_cyc:
        extracted = poly_reduce_by_symmetry(poly, perm).terms()
    else:
        extracted = poly.terms()

    MUL = '*' if not omit_mul else ''
    POW = '^' if not omit_pow else ''
    gens = poly.gens
    nvars = len(gens)
    def _gen_pow(g, p):
        if p == 0: return ''
        if p == 1: return str(g)
        return str(g) + POW + str(p)
    def _filter_list(l):
        return [_ for _ in l if len(_)]
    def _concat(coeff_str, monom_str):
        if len(monom_str)==0 and coeff_str.endswith('*'):
            return coeff_str[:-1] # + monom_str
        return coeff_str + monom_str
    def _get_monom_str(monom):
        return MUL.join(_filter_list([_gen_pow(gens[i], monom[i]) for i in range(nvars)]))
    def get_string(monom, coeff):
        if not any(monom): # constant
            if coeff == 1: return '+1'
            if coeff == -1: return '-1'
            return _get_coeff_str(coeff, '')
        coeff_str = _get_coeff_str(coeff, MUL)
        return _concat(coeff_str, _get_monom_str(monom))

    strings = []
    for monom, coeff in extracted:
        strings.append(get_string(monom, coeff))
    s = ''.join(strings)
    if s.startswith('+'):
        s = s[1:]
    if _is_cyc and not perm.is_trivial:
        s = 's(%s)'%s
    else:
        s = '(%s)'%s
    return s


def _reduce_factor_list(poly: Poly, perm_group: PermutationGroup) -> Tuple[Expr, List[Tuple[Poly, int]], List[Tuple[Poly, int]]]:
    """
    Reduce the factor list of a polynomial with respect to a permutation group.

    Parameters
    ----------
    poly : Poly
        The polynomial to be factorized.
    perm_group : PermutationGroup
        The permutation group to be considered.

    Returns
    -------
    Tuple[Expr, List[Tuple[Poly, int]], List[Tuple[Poly, int]]]
        The reduced coefficient, the reduced parts, and the cyclic parts.

    Examples
    --------
    >>> from sympy.abc import a, b, c
    >>> _reduce_factor_list((b**8*a**6*c**3*(a**2+b*c)*(b**2+c*a)*(a-b)**7*(b-c)**6*(c-a)**8).as_poly(a,b,c), CyclicGroup(3)) # doctest:+SKIP
    (1,
     [(Poly(b, a, b, c, domain='ZZ'), 5),
      (Poly(a*c + b**2, a, b, c, domain='ZZ'), 1),
      (Poly(a, a, b, c, domain='ZZ'), 3),
      (Poly(a - b, a, b, c, domain='ZZ'), 1),
      (Poly(a - c, a, b, c, domain='ZZ'), 2),
      (Poly(a**2 + b*c, a, b, c, domain='ZZ'), 1)],
     [(Poly(c, a, b, c, domain='ZZ'), 3), (Poly(b - c, a, b, c, domain='ZZ'), 6)])
    """
    coeff, factors = poly.factor_list()
    pow_dict = dict((p, power) for p, power in factors)
    rep_dict = dict((p.rep, p) for p, _ in factors)
    cyc_factors = []
    for base, power in factors:
        subdict = defaultdict(int)
        representative = base
        sign = 1
        for p in perm_group.elements:
            permed_base = base.reorder(*p(poly.gens)).rep
            permed_poly = rep_dict.get(permed_base)
            if permed_poly is None:
                permed_poly = rep_dict.get(-permed_base)
                sign *= -1
            if permed_poly is None:
                break
            subdict[permed_poly] += 1
            if pow_dict[permed_poly] < subdict[permed_poly]:
                break
            if permed_poly.compare(representative) < 0:
                representative = permed_poly
        else:
            # get max common power
            d = min(pow_dict[p]//v for p, v in subdict.items())
            if sign == -1 and d % 2 == 1: coeff = -coeff
            for p, v in subdict.items():
                pow_dict[p] -= d * v
                if pow_dict[p] == 0:
                    del pow_dict[p]
                    del rep_dict[p.rep]
            cyc_factors.append((representative, d))
    return coeff, list(pow_dict.items()), cyc_factors


def poly_get_factor_form(
        poly: Poly,
        perm: Optional[PermutationGroup] = None,
        omit_mul: bool = True,
        omit_pow: bool = True,
        # return_type: str = 'text',
    ) -> str:
    """
    Get the factorized form of a polynomial.

    Parameters
    ----------
    poly : Poly
        The polynomial to be factorized.
    perm : PermutationGroup
        The permutation group to be considered. If None,
        it uses the cyclic group generated by the variables.
    omit_mul : bool
        Whether to omit the multiplication sign. Defaults to True.
    omit_pow : bool
        Whether to omit the power sign. Defaults to True.
    """
    if poly.total_degree() == 0:
        return poly_get_standard_form(poly, perm, omit_mul, omit_pow)

    if perm is None:
        perm = CyclicGroup(len(poly.gens))

    coeff, factors, cyc_factors = _reduce_factor_list(poly, perm)

    MUL = '*' if not omit_mul else ''
    POW = '^' if not omit_pow else ''

    strings = []

    def get_pow_string(base: Poly, power: Integer):
        if base.is_monomial:
            if power == 1: return str(base.as_expr())
            return '%s%s%s'%(base.as_expr(), POW, power)

        base = poly_get_standard_form(base, perm, omit_mul, omit_pow)
        if _is_single_paren(base): pass
        elif base.startswith('s(') and _is_single_paren(base[1:]): pass
        else: base = '(%s)'%base
        if power == 1: return base
        return '%s%s%s'%(base, POW, power)

    def get_cyc_pow_string(base: Poly, power: Integer):
        s = get_pow_string(base, power)
        if s.startswith('('): s = 'p%s'%s
        else: s = 'p(%s)'%s
        return s

    for base, power in factors:
        strings.append(get_pow_string(base, power))
    for base, power in cyc_factors:
        strings.append(get_cyc_pow_string(base, power))

    strings = sorted(strings, key = lambda x: (len(x), x))
    s = _get_coeff_str(coeff, MUL) + MUL.join(strings)
    if s.startswith('+'): s = s[1:]
    return s



def short_constant_parser(x):
    """
    Parse a sympy constant using limited characters.
    """
    if x.is_Number:
        if isinstance(x, Rational):
            txt = str(x)
        elif isinstance(x, Float):
            txt = '%.4f'%(x)
        else:
            v = x.as_numer_denom()
            txt = f'{v[0]}' + (f'/{v[1]}' if v[1] != 1 else '')
        if len(txt) > 10 and not isinstance(x, Float):
            txt = '%.4f'%(x)
    else:
        txt = str(x).replace('**','^').replace('*','').replace(' ','')
    return txt


def coefficient_triangle(poly: Poly, degree: int = None) -> str:
    """
    Convert the coefficients of a polynomial to a list.

    The degree should be specified when the polynomial is zero to
    indicate the degree of the zero polynomial.

    Parameters
    ----------
    poly : Poly
        The polynomial to convert.
    degree : int
        The degree of the polynomial. If None, it will be computed.
    """
    if degree is None:
        degree = poly.total_degree()
    if poly.is_homogeneous and len(poly.gens) == 4:
        if not poly.is_zero:
            from .monomials import arraylize_sp
            vec = arraylize_sp(poly)
            return [short_constant_parser(_) for _ in vec]
        else:
            return ['0'] * ((degree+1)*(degree+2)*(degree+3)//6)
    else:
        coeffs = poly.coeffs()
        monoms = poly.monoms()
        monoms.append((-1,-1,0))  # tail flag

        t = 0
        triangle = []
        for i in range(degree+1):
            for j in range(i+1):
                if monoms[t][0] == degree - i and monoms[t][1] == i - j:
                    txt = short_constant_parser(coeffs[t])
                    t += 1
                else:
                    txt = '0'
                triangle.append(txt)
    return triangle


def coefficient_triangle_latex(poly, tabular: bool = True, document: bool = True, zeros: str ='\\textcolor{lightgray}') -> str:
    """
    Return the LaTeX format of the coefficient triangle.

    Parameters
    ----------
    poly : Poly
        The polynomial to be factorized.
    tabular : bool
        Whether to use the tabular environment.
    document : bool
        If True, it will be wrapped by an additional arraystretch command.
    zeros : str
        The color of the zeros.
    """
    if poly is None:
        return ''
    if isinstance(poly, str):
        poly_str = poly
        poly = pl(poly)
    else:
        poly_str = 'f(%s)'%','.join([str(_) for _ in poly.gens])

    zero_wrapper = lambda x: x
    if zeros is not None and len(zeros):
        zero_wrapper = lambda x: '%s{%s}'%(zeros, x)

    n = poly.total_degree()
    emptyline = '\\\\ ' + '\\ &' * (n * 2) + '\\  \\\\ '
    strings = ['' for i in range(n+1)]

    coeffs = poly.coeffs()
    monoms = poly.monoms()
    monoms.append((-1,-1,0))  # tail flag
    t = 0
    for i in range(n+1):
        for j in range(i+1):
            if monoms[t][0] == n - i and monoms[t][1] == i - j:
                txt = latex(coeffs[t])
                t += 1
            else:
                txt = zero_wrapper('0')
            strings[j] = strings[j] + '&\\ &' + txt if len(strings[j]) != 0 else txt
    monoms.pop()

    for i in range(n+1):
        strings[i] = '\\ &'*i + strings[i] + '\\ &'*i + '\\ '
    s = emptyline.join(strings)
    if tabular:
        s = '\\left[\\begin{matrix}\\begin{tabular}{' + 'c' * (n * 2 + 1) + '} ' + s
        s += ' \\end{tabular}\\end{matrix}\\right]'
    else:
        s = '\\left[\\begin{matrix} ' + s
        s += ' \\end{matrix}\\right]'

    s = (' \\\\ '.join(s.split('\\\\')[::2]))
    s = s.replace('&\\','& \\')
    if document:
        s = '\\renewcommand*{\\arraystretch}{1.732}$' + s + '$'
        # s = '\\textnormal{'+ poly_str +'  =}\n' + s
    return s




def swa(x, verbose = True):
    """
    Helper function for experiment. The output can be sent as
    input of wolframalpha.
    """
    a = str(x).replace(' ','').replace('**','^')
    if verbose: print(a)
    return a

def sdesmos(x, verbose = True):
    """
    Helper function for experiment. The output can be sent as
    input of desmos.
    """
    a = (latex(x).replace(' ',''))
    if verbose: print(a)
    return a

def wrap_desmos(x: List[Expr]) -> str:
    """
    Wrap everything in desmos calculator javascript.
    """
    html_str = r"""
    <script src="https://www.desmos.com/api/v1.8/calculator.js?apiKey=dcb31709b452b1cf9dc26972add0fda6"></script>
    <div id="calculator" style="width: 1000px; height: 600px;"></div>
    <script>
    var elt = document.getElementById('calculator');
    var calculator = Desmos.GraphingCalculator(elt);
    """
    for i, content in enumerate(x):
        html_str += "\ncalculator.setExpression({id: '%d', latex: '%s'});"%(i, latex(content).replace('\\','\\\\'))
    html_str += "\n</script>"
    return html_str


class PolyReader:
    """
    A class for reading polynomials from a file or a list of strings.
    This is an iterator class.
    """
    def __init__(self,
        polys: Union[List[Union[Poly, str]], str],
        gens: Tuple[Symbol] = sp_symbols("a b c"),
        perm: Optional[PermutationGroup] = None,
        ignore_errors: bool = False,
        **kwargs
    ):
        """
        Read polynomials from a file or a list of strings.
        This is a generator function.

        Parameters
        ----------
        polys : Union[List[Union[Poly, str]], str]
            The polynomials to read. If it is a string, it will be treated as a file name.
            If it is a list of strings, each string will be treated as a polynomial.
            Empty lines will be ignored.
        gens : Tuple[Symbol]
            The generators of the polynomial.
        perm : Optional[PermutationGroup]
            The permutation group of the expression. If None, it will be cyclic group.
        ignore_errors : bool
            Whether to ignore errors. If True, invalid polynomials will be skipped by
            yielding None. If False, invalid polynomials will raise a ValueError.
        kwargs:
            Other arguments for preprocess_text.

        Yields
        ----------
        Poly
            The read polynomials.
        """
        self.source = None
        if isinstance(polys, str):
            self.source = polys
            with open(polys, 'r') as f:
                polys = f.readlines()

        polys = map(
            lambda x: x.strip() if isinstance(x, str) else x,
            polys
        )

        polys = list(filter(
            lambda x: (isinstance(x, str) and len(x)) or isinstance(x, Poly),
            polys
        ))

        self.polys = polys

        self.kwargs = kwargs
        if 'gens' not in self.kwargs:
            self.kwargs['gens'] = gens
        if 'perm' not in self.kwargs:
            self.kwargs['perm'] = perm

        self.index = 0
        self.ignore_errors = ignore_errors

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.polys):
            raise StopIteration
        poly = self.polys[self.index]
        if isinstance(poly, str):
            try:
                poly = preprocess_text(poly, **self.kwargs)
            except:
                if not self.ignore_errors:
                    raise ValueError(f'Invalid polynomial at index {self.index}: {poly}')
                poly = None
        self.index += 1
        return poly

    def __len__(self):
        return len(self.polys)
