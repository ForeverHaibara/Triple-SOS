from typing import Union, List, Tuple
import re

import sympy as sp
from sympy.polys import Poly

from .polytools import deg

def next_permute(f: str) -> str:
    """a^3 * b^2 * c   ->   b^3 * c^2 * a"""
    return f.translate({97: 98, 98: 99, 99: 97})

def reflect_permute(f: str) -> str:
    """a^3 * b^2 * c   ->   c^3 * b^2 * a"""
    return f.translate({97: 99, 98: 97, 99: 98})

def cycle_expansion(f, symbol='s'):
    """
    Parameters
    -------
    f: str
        the string of the polynomial with variables a , b , c
    symbol: char, 's' or 'p'
        when symbol == 's':
            a^3 * b^2 * c   ->   a^3 * b^2 * c + b^3 * c^2 * a + c^3 * a^2 * b
        when symbol == 'p':
            a^3 * b^2 * c   ->   a^3 * b^2 * c * b^3 * c^2 * a * c^3 * a^2 * b
        Warning : Please add parenthesis yourself before expansion if necessary

    Return
    -------
    a string, the result of cycle expansion
    """
    fb = next_permute(f)
    fc = next_permute(fb)
    if symbol != 'p':
        return ' + '.join([f, fb, fc])
    return ' * '.join([f, fb, fc])


##########################################################################
#
#                             Preprocess text
#
##########################################################################

def _preprocess_text_delatex(poly: str):
    """
    Convert a latex formula to normal representation.
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


def _preprocess_text_expansion(poly: str):
    """
    Expand the polynomial with cycle expansion.

    s(ab)    ->   (ab + bc + ca)     

    p(a+b)   ->   (a+b)(b+c)(c+a)
    """
    parenthesis = 0
    paren_depth = [-1]
    cycle_begin = []

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
                tmp = '(' + cycle_expansion(poly[cycle_begin[-1]+1:i+1], symbol=poly[cycle_begin[-1]]) + ')'
                poly = poly[:cycle_begin[-1]] + tmp + poly[i+1:]
                i = cycle_begin[-1] + len(tmp) - 1
                paren_depth.pop()
                cycle_begin.pop()
        i += 1
    return poly


def _preprocess_text_completion(poly: str):
    """
    Complete the polynomial with * and ^. E.g. 
    1/5a3b2c   ->   1/5*a^3*b^2*c
    """
    poly = poly.replace(' ','')
    i = 0 
    while i < len(poly) - 1:
        if 48 <= ord(poly[i]) <= 57: # '0'~'9'
            if poly[i+1] == '(' or (97 <= ord(poly[i+1]) <= 122 and poly[i+1] != 'e'): # alphabets
                poly = poly[:i+1] + '*' + poly[i+1:]
                i += 1
        elif poly[i] == ')' or (97 <= ord(poly[i]) <= 122 and poly[i] != 'e'): # alphabets
            if poly[i+1] == '(' or 97 <= ord(poly[i+1]) <= 122:
                poly = poly[:i+1] + '*' + poly[i+1:]
                i += 1
            elif 48 <= ord(poly[i+1]) <= 57: # '0'~'9'
                poly = poly[:i+1] + '^' + poly[i+1:]  
                i += 1     
        i += 1

    poly = poly.replace('s*q*r*t*','sqrt')
    return poly


def _preprocess_text_cyclize(poly: str):
    """
    Automatically perform cycle expansion on poly if it is not cyclic.
    """
    cyc_poly = Poly(cycle_expansion(poly), sp.symbols('a b c'), extension = True)
    poly = Poly(poly)
    for coeff in (cyc_poly - 3*poly).coeffs():
        # some coefficient is larger than tolerance, not cyclic
        if coeff != 0:
            return cyc_poly
    return poly


def _preprocess_text_get_domain(poly: str):
    """
    Get the domain of a polynomial, e.g.
    (5^0.5 + 1)/2 a -> QQ(sqrt(5))

    Deprecated. DO NOT USE.
    """
    extensions = re.findall('sqrt\((.*?)\)', poly)
    domain_ext = set()
    if len(extensions) > 0:
        for ext in extensions:
            t = sp.sympify(ext)
            if isinstance(t, sp.Rational):
                t_ = abs(sp.ntheory.factor_.core(t.p) * sp.ntheory.factor_.core(t.q))
                if t_ != 1 and t_ != 0:
                    domain_ext.add(int(t))

    if len(domain_ext) == 0:
        return sp.QQ 
    
    return sp.QQ.algebraic_field(*tuple(sp.sqrt(i) for i in domain_ext))


def preprocess_text(
        poly,
        cyc = False, 
        retText = False, 
        cancel = False, 
        variables = None
    ) -> Union[Poly, Tuple[Poly, Poly]]:
    """
    Parse a text to sympy polynomial with respect to a, b, c.

    Parameters
    ----------
    cyc: bool  
        check whether the polynomial is cyclic, if not then cyclize it
    retText: bool
        whether return the preprocessed text rather than a sympy polynomial
    cancel: bool
        whether cancel the denominator when encountering fractions 

    Returns
    -------
    If cancel == False:
        return a sympy polynomial.
        If it fails to parse the polynomial, return None.
    If cancel == True:
        return a tuple of sympy polynomials (numerator, denominator).
        If it fails to parse the polynomial, return (None, None).
    """
    poly = poly.lower()
    poly = _preprocess_text_delatex(poly)
    # dom = preprocess_text_GetDomain(poly)
    poly = _preprocess_text_expansion(poly)
    poly = _preprocess_text_completion(poly)
    
    if retText:
        if cyc: return cycle_expansion(poly)
        else:   return poly

    # if symbols is None:
    symbols = sp.symbols('a b c')

    if cancel or (variables is not None):
        assert (not cyc), 'Cyclic is not enabled when cancel == True or variabels is not None'
        
        poly = sp.sympify(poly)
        if variables is not None: 
            poly = poly.subs(variables)
            # for name, value in variables.items():
            #     try:
            #         poly = poly.subs(name, value)
            #     except:
            #         pass 
                
        if cancel:
            try:
                frac = sp.fraction(sp.cancel(poly))
                poly0 = Poly(frac[0], symbols, extension = True)
                poly1 = Poly(frac[1], symbols, extension = True)
                return poly0, poly1
            except:
                return None, None
        
        try:
            poly = Poly(poly, symbols, extension = True)
        except:
            return None
    else:
        if cyc:
            poly = _preprocess_text_cyclize(poly)
        else:
            try:
                poly = Poly(poly, symbols, extension = True)
            except:
                poly = None

    return poly


def pl(*args, **kwargs):
    """
    Abbreviation for preprocess_text.
    """
    return preprocess_text(*args, **kwargs)


def degree_of_zero(poly):
    """
    Compute the degree of a homogeneous zero polynomial
    Idea: delete the additions and substractions, which do not affect the degree.
    """
    poly = poly.lower()
    poly = _preprocess_text_delatex(poly)
    poly = _preprocess_text_expansion(poly)
    poly = _preprocess_text_completion(poly)
    
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
                elif poly[j] in 'abc':
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
        poly = sp.fraction(sp.sympify(poly))
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
        if isinstance(x, sp.Rational):
            txt = str(x)
        elif isinstance(x, sp.Float):
            txt = '%.4f'%(x)
        else:
            v = x.as_numer_denom()
            txt = f'{v[0]}' + (f'/{v[1]}' if v[1] != 1 else '')
        if len(txt) > 10 and not isinstance(x, sp.Float):
            txt = '%.4f'%(x)
    else:
        txt = str(x).replace('**','^').replace('*','').replace(' ','')
    return txt


def coefficient_triangle(poly: sp.Poly, degree: int = None) -> str:
    """
    Convert the coefficients of a polynomial to a list.

    The degree should be specified when the polynomial is zero to
    indicate the degree of the zero polynomial.

    Parameters
    ----------
    poly : sp.Poly
        The polynomial to convert.
    degree : int
        The degree of the polynomial. If None, it will be computed.
    """
    if degree is None:
        degree = deg(poly)
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
    a = (sp.latex(x).replace(' ',''))
    if verbose: print(a)
    return a

def wrap_desmos(x: List[sp.Expr]) -> str:
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
        html_str += "\ncalculator.setExpression({id: '%d', latex: '%s'});"%(i, sp.latex(content).replace('\\','\\\\'))
    html_str += "\n</script>"
    return html_str


class PolyReader:
    """
    A class for reading polynomials from a file or a list of strings.
    This is an iterator class.
    """
    def __init__(self,
        polys: Union[List[Union[sp.Poly, str]], str],
        ignore_errors: bool = False
    ):
        """
        Read polynomials from a file or a list of strings.
        This is a generator function.

        Parameters
        ----------
        polys : Union[List[Union[sp.Poly, str]], str]
            The polynomials to read. If it is a string, it will be treated as a file name.
            If it is a list of strings, each string will be treated as a polynomial.
            Empty lines will be ignored.
        ignore_errors : bool    
            Whether to ignore errors. If True, invalid polynomials will be skipped by
            yielding None. If False, invalid polynomials will raise a ValueError.

        Yields
        ----------
        sp.Poly
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
            lambda x: (isinstance(x, str) and len(x)) or isinstance(x, sp.Poly),
            polys
        ))

        self.polys = polys
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
                poly = preprocess_text(poly)
            except:
                if not self.ignore_errors:
                    raise ValueError(f'Invalid polynomial at index {self.index}: {poly}')
                poly = None
        self.index += 1
        return poly

    def __len__(self):
        return len(self.polys)
