from typing import Union, List, Tuple, Optional
from functools import partial
# import re

from sympy import Expr, Poly, Rational, Float, Symbol, sympify, fraction, cancel, latex
from sympy import symbols as sp_symbols
from sympy.combinatorics import Permutation, PermutationGroup, CyclicGroup

from .polytools import deg

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

    if return_type == 'expr':
        return sympify(poly)
    elif return_type == 'frac': 
        poly = sympify(poly)

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
    #     degree = deg(Poly(poly))
        poly = fraction(sympify(poly))
        if poly[1].is_constant():
            degree = deg(Poly(poly[0]))
        else:
            degree = deg(Poly(poly[0])) - deg(Poly(poly[1]))
    except:
        degree = 0
        
    return degree


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
