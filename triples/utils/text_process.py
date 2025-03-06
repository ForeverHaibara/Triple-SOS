from typing import Union, List, Tuple, Optional
from functools import partial
from collections import defaultdict
# import re

from sympy import Expr, Poly, Rational, Integer, Float, Symbol, sympify, fraction, cancel, latex
from sympy import symbols as sp_symbols
from sympy.core.singleton import S
from sympy.combinatorics import Permutation, PermutationGroup, CyclicGroup

from .expression import Coeff

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

    Return
    -------
    a string, the result of cycle expansion
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

def _preprocess_text_delatex(poly: str) -> str:
    """
    Convert a latex formula to standard representation.
    """

    # \frac{..}{...} -> (..)/(...)
    poly = poly.replace(' ','')
    poly = poly.replace('left','')
    poly = poly.replace('right','')
    poly = poly.replace('}{' , ')/(')
    poly = poly.replace('frac' , '')
    poly = poly.translate({123: 40, 125: 41, 92: 32, 36: 32}) # { -> ( , } -> ) , \ -> space, $ -> space
    poly = poly.replace(' ','')

    # \sum ... -> s(...)
    # \prod ... -> p(...)
    parenthesis = 0
    paren_depth = [-1]
    i = 0
    while i < len(poly) - 4:
        if poly[i:i+3] == 'sum':
            parenthesis += 1
            paren_depth.append(parenthesis)
            poly = poly[:i] + 's(' + poly[i+3:]
            i += 1
        elif poly[i:i+4] == 'prod':
            parenthesis += 1
            paren_depth.append(parenthesis)
            poly = poly[:i] + 'p(' + poly[i+4:]
            i += 1
        elif poly[i] == '(':
            parenthesis += 1
        elif poly[i] == ')':
            parenthesis -= 1
        elif parenthesis == paren_depth[-1] and (poly[i] == '+' or poly[i] == '-'):
            poly = poly[:i] + ')' + poly[i:]
            parenthesis -= 1
            paren_depth.pop()
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
        preserve_sqrt: bool = True,
        preserve_cbrt: bool = False
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
    CHECK_SQRT = (lambda poly, i: len(poly) >= i + 5 and poly[i:i+5] == 'sqrt(') if preserve_sqrt else (lambda poly, i: False)
    CHECK_CBRT = (lambda poly, i: len(poly) >= i + 5 and poly[i:i+5] == 'cbrt(') if preserve_cbrt else (lambda poly, i: False)
    poly = poly.replace(' ','')
    i = 0 
    while i < len(poly) - 1:
        if 48 <= ord(poly[i]) <= 57: # '0'~'9'
            if poly[i+1] == '(' or (97 <= ord(poly[i+1]) <= 122 and poly[i+1] != SCI): # alphabets
                poly = poly[:i+1] + '*' + poly[i+1:]
                i += 1
        elif poly[i] == ')' or (97 <= ord(poly[i]) <= 122 and poly[i] != SCI): # alphabets
            if CHECK_SQRT(poly, i) or CHECK_CBRT(poly, i):
                i += 4
            elif poly[i+1] == '(' or 97 <= ord(poly[i+1]) <= 122:
                poly = poly[:i+1] + '*' + poly[i+1:]
                i += 1
            elif 48 <= ord(poly[i+1]) <= 57: # '0'~'9'
                poly = poly[:i+1] + '^' + poly[i+1:]  
                i += 1
        i += 1

    return poly


def preprocess_text(
        poly: str,
        gens: Tuple[Symbol] = sp_symbols("a b c"),
        perm: Optional[PermutationGroup] = None,
        return_type: str = "poly",
        scientific_notation: bool = False,
        preserve_sqrt: bool = True,
        preserve_cbrt: bool = False
    ) -> Union[str, Expr, Poly, Tuple[Poly, Poly]]:
    """
    Parse a text to sympy polynomial with respect to the given generators conveniently.
    The function assumes each variable to be a single character.
    For more general cases, please do not rely on this function.

    Parameters
    ----------
    return_type: str
        One of ['text', 'expr', 'poly', 'frac'].
        If 'text', return the text of the polynomial.
        If 'expr', return the sympy expression of the polynomial.
        If 'poly', return the sympy polynomial of the polynomial.
        If 'frac', return a tuple of sympy polynomials (numerator, denominator). If
            it fails to cancel the polynomial, return (None, None).
    gens: Tuple[Symbol]
        The generators of the polynomial.
    perm: Optional[PermutationGroup]
        The permutation group of the expression. If None, it will be cyclic group.
    scientific_notation: bool
        Whether to parse the scientific notation. If True, 1e2 will be parsed as 100.
        If False, 1e2 will be parsed as e^2 where e is a free variable.
    preserve_sqrt: bool
        Whether to preserve the sqrt function. If True, sqrt will be inferred as square root
        rather than s*q*r*t.
    preserve_cbrt: bool
        Whether to preserve the cbrt function. If True, cbrt will be inferred as cubic root
        rather than c*b*r*t.

    Returns
    -------
    See return_type.    
    """
    poly = poly.lower()

    if perm is None:
        perm = CyclicGroup(len(gens))

    poly = _preprocess_text_delatex(poly)
    poly = _preprocess_text_expansion(poly, gens, perm)
    poly = _preprocess_text_completion(poly,
        scientific_notation=scientific_notation,
        preserve_sqrt=preserve_sqrt,
        preserve_cbrt=preserve_cbrt
    )
    
    if return_type == 'text':
        return poly

    poly = sympify(poly)
    if return_type == 'expr':
        return poly
    elif return_type == 'frac':
        try:
            frac = fraction(cancel(poly))

            poly0 = Poly(frac[0], gens, extension = True)
            poly1 = Poly(frac[1], gens, extension = True)

            if len(frac[1].free_symbols) == 0:
                div0, div1 = poly0.div(poly1)
                if div1.is_zero:
                    one = Poly(1, gens, extension = True)
                    return div0, one
            return poly0, poly1
        except:
            return None, None
    else:
        try:
            poly = Poly(poly, gens, extension = True)
        except:
            poly = None

    return poly


def pl(*args, **kwargs):
    """
    Parse a text to sympy polynomial with respect to the given generators conveniently.
    This is a shortcut for preprocess_text.
    """
    return preprocess_text(*args, **kwargs)
pl = preprocess_text

def degree_of_zero(
        poly: str,
        gens: Tuple[Symbol] = sp_symbols("a b c"),
        perm: Optional[PermutationGroup] = None,
        scientific_notation: bool = False,
        preserve_sqrt: bool = True,
        preserve_cbrt: bool = False
    ) -> int:
    """
    Infer the degree of a homogeneous zero polynomial.
    Idea: delete the additions and subtractions, which do not affect the degree.

    Parameters
    ----------
    poly: str
        The polynomial of which to infer the degree.
    gens: Tuple[Symbol]
        The generators of the polynomial.
    perm: Optional[PermutationGroup]
        The permutation group of the expression. If None, it will be cyclic group.
    scientific_notation: bool
        Whether to parse the scientific notation. If True, 1e2 will be parsed as 100.
        If False, 1e2 will be parsed as e^2 where e is a free variable.
    preserve_sqrt: bool
        Whether to preserve the sqrt function. If True, sqrt will be inferred as square root
        rather than s*q*r*t.
    preserve_cbrt: bool
        Whether to preserve the cbrt function. If True, cbrt will be inferred as cubic root
        rather than c*b*r*t.

    Returns
    -------
    int
        The degree of the (homogeneous) polynomial.

    Examples
    --------
    >>> degree_of_zero('p(a)(s(a2)2-s(a4+2a2b2))')
    7
    """
    poly = poly.lower()

    if perm is None:
        perm = CyclicGroup(len(gens))

    poly = _preprocess_text_delatex(poly)
    poly = _preprocess_text_expansion(poly, gens, perm)
    poly = _preprocess_text_completion(poly,
        scientific_notation=scientific_notation,
        preserve_sqrt=preserve_sqrt,
        preserve_cbrt=preserve_cbrt
    )
    gen_names = set([_.name for _ in gens])

    i = 0
    length = len(poly)
    bracket = 0
    while i < length:
        if poly[i] == '+' or poly[i] == '-': 
            # run to the end of this bracket (sum does not affect the degree)
            # e.g. a*(a^2+a*b)*c -> a*(a^2)*c
            bracket_cur = bracket
            j = i + 1
            is_constant = True 
            while j < length:
                if poly[j] == '(':
                    bracket += 1
                elif poly[j] == ')':
                    bracket -= 1
                    if bracket < bracket_cur:
                        break 
                elif poly[j] in gen_names:
                    is_constant = False 
                j += 1
            if is_constant == False:
                poly = poly[:i] + poly[j:]
                length = len(poly)
        elif poly[i] == ')':
            bracket -= 1
        elif poly[i] == '(':
            bracket += 1
            # e.g. a*(-b*c) ,    a*(--+-b*c)
            i += 1
            while i < length and (poly[i] == '-' or poly[i] == '+'):
                i += 1
            if i == length:
                break 
            
        i += 1
        
    try:
        deg = lambda x: x.total_degree()
    #     degree = deg(Poly(poly))
        poly = fraction(sympify(poly))
        if poly[1].is_constant():
            degree = deg(Poly(poly[0]))
        else:
            degree = deg(Poly(poly[0])) - deg(Poly(poly[1]))
    except:
        degree = 0
        
    return degree


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
    >>> poly_get_standard_form(((x*a+y*b+z*c)**2).as_poly(x,y,z))
    '(a^2)x2+(2*a*b)xy+(2*a*c)xz+(b^2)y2+(2*b*c)yz+(c^2)z2'
    >>> poly_get_standard_form(((x*a+y*b+z*c)**2).as_poly(x,y,z,a,b,c), PermutationGroup(Permutation([1,2,0,4,5,3])))
    's(x2a2+2xyab)'
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
        perm_group_gens = perm.generators
        perm_order = perm.order()
        ufs = {}
        # monomials invariant under the permutation group is recorded in ufs
        def ufs_find(monom):
            v = ufs.get(monom, monom)
            if v == monom:
                return monom
            w = ufs_find(v)
            ufs[monom] = w
            return w
        for m1, coeff in poly.terms():
            for p in perm_group_gens:
                m2 = tuple(p(m1))
                f1, f2 = ufs_find(m1), ufs_find(m2)
                # merge to the maximum
                if f1 > f2:
                    ufs[f2] = f1
                else:
                    ufs[f1] = f2

        ufs_size = defaultdict(int)
        for m in ufs.keys():
            ufs_size[ufs_find(m)] += 1

        def get_order(monom):
            # get the multiplicity of the monomials given the permutation group
            # i.e. how many permutations make it invariant
            return perm_order // ufs_size[ufs_find(monom)]
        
        # only reserve the keys for ufs[monom] == monom
        for monom, coeff in poly.terms():
            if ufs_find(monom) == monom:
                extracted.append((sum(monom), monom, coeff))
    else:
        One = S.One
        def get_order(monom):
            return One
        for monom, coeff in poly.terms():
            extracted.append((sum(monom), monom, coeff))
 
    extracted = sorted(extracted, reverse=True)

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
    for _, monom, coeff in extracted:
        order = get_order(monom)
        coeff = coeff / order
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
    >>> _reduce_factor_list((b**8*a**6*c**3*(a**2+b*c)*(b**2+c*a)*(a-b)**7*(b-c)**6*(c-a)**8).as_poly(a,b,c), CyclicGroup(3))
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
