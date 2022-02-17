import sympy as sp
import numpy as np
from math import gcd

def NextPermute(f):
    '''a^3 * b^2 * c   ->   b^3 * c^2 * a'''
    fb = f.replace('a','#')
    fb = fb.replace('c','a')
    fb = fb.replace('b','c')
    fb = fb.replace('#','b')
    return fb

def ReflectPermute(f):
    '''a^3 * b^2 * c   ->   c^3 * b^2 * a'''
    fb = f.replace('a','#')
    fb = fb.replace('c','a')
    fb = fb.replace('#','c')
    return fb

def CycleExpansion(f, symbol='s'):
    '''
    Params
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
    '''
    fb = NextPermute(f)
    fc = NextPermute(fb)
    if symbol != 'p':
        return ' + '.join([f, fb, fc])
    return ' * '.join([f, fb, fc])

def deg(f):
    '''return the degree of a polynomial'''
    return sum(f.monoms()[0])


def PreprocessText_DeLatex(poly):
    '''convert a latex formula to normal representation'''

    poly = poly.replace(' ','')
    poly = poly.replace('$','')

    # \frac{..}{...} -> (..)/(...)
    poly = poly.replace('left','')
    poly = poly.replace('right','')
    poly = poly.replace('}{' , ')/(')
    poly = poly.replace('}' , ')').replace('{' , '(')
    poly = poly.replace('frac' , '')
    poly = poly.replace('\\' , '')

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


def PreprocessText_Expansion(poly):
    '''
    s(ab)    ->   (ab + bc + ca)     

    p(a+b)   ->   (a+b)(b+c)(c+a)
    '''
    parenthesis = 0
    paren_depth = [-1]
    cycle_begin = []

    i = 0
    while i < len(poly):
        if poly[i] == 's' or poly[i] == 'p':
            paren_depth.append(parenthesis)
            cycle_begin.append(i)
        elif poly[i] == '(':
            parenthesis += 1
        elif poly[i] == ')':
            parenthesis -= 1
            if paren_depth[-1] == parenthesis:
                tmp = '(' + CycleExpansion(poly[cycle_begin[-1]+1:i+1], symbol=poly[cycle_begin[-1]]) + ')'
                poly = poly[:cycle_begin[-1]] + tmp + poly[i+1:]
                i = cycle_begin[-1] + len(tmp) - 1
                paren_depth.pop()
                cycle_begin.pop()
        i += 1
    return poly


def PreprocessText_Completion(poly):
    '''1/5a3b2c   ->   1/5*a^3*b^2*c'''
    poly = poly.replace(' ','')
    i = 0 
    while i < len(poly) - 1:
        if 48 <= ord(poly[i]) <= 57: # '0'~'9'
            if poly[i+1] == '(' or 97 <= ord(poly[i+1]) <= 122: # alphabets
                poly = poly[:i+1] + '*' + poly[i+1:]
                i += 1
        elif poly[i] == ')' or 97 <= ord(poly[i]) <= 122: # alphabets
            if poly[i+1] == '(' or 97 <= ord(poly[i+1]) <= 122:
                poly = poly[:i+1] + '*' + poly[i+1:]
                i += 1
            elif 48 <= ord(poly[i+1]) <= 57: # '0'~'9'
                poly = poly[:i+1] + '^' + poly[i+1:]  
                i += 1     
        i += 1
    return poly


def PreprocessText_Cyclize(poly, tol=1e-7):
    '''automatically perform cycle expansion on poly if it is not cyclic'''
    cyc_poly = sp.polys.polytools.Poly(CycleExpansion(poly))
    poly = sp.polys.polytools.Poly(poly)
    for coeff in (cyc_poly - 3*poly).coeffs():
        # some coefficient is larger than tolerance, not cyclic
        if abs(coeff) > tol:
            return cyc_poly
    return poly


def PreprocessText(poly, cyc=False, retText=False, retn=False):
    '''
    Params
    -------
    cyc: bool  
        check whether the polynomial is cyclic, if not then cyclize it
    retText: bool
        whether return the preprocessed text rather than a sympy polynomial
    retn: bool
        whether return the degree of the polynomial, only available when retText = False
    
    Return
    -------
    '''
    poly = poly.lower()
    poly = PreprocessText_DeLatex(poly)
    poly = PreprocessText_Expansion(poly)
    poly = PreprocessText_Completion(poly)
    
    #print(poly)
    if retText:
        if cyc:
            return CycleExpansion(poly)
        else:
            return poly
        
    if cyc:
        poly = PreprocessText_Cyclize(poly)
    else:
        try:
            poly = sp.polys.polytools.Poly(poly)
        except:
            poly = None
    
    if retn:
        if poly is not None:
            n = deg(poly)
        else:
            n = 0
        return poly, n
        
    return poly


def prettyprint(y, names, precision=6, linefeed=2, formatt=0, dectofrac=False):
    '''
    prettily format a cyclic polynomial sum into a certain formatt

    Params
    -------
    y: list / array / tuple / iterator ... 
        the coeffcients of each partial polynomial

    names: list / tuple / iterator of str
        the display format of each polynomial

    precision: unsigned int
        the precision for displaying floating point numbers
    
    linefeed: int
        feed a new line every certain terms, no linefeed if set to zero

    formatt: 
        0: LaTeX   2: formatted

    dectofrac: bool
        whether or not convert all decimals to fractions

    Return
    -------
    a string, the formatted result
    '''
    result = ''
    linefeed_terms = 0
    for coeff, name in zip(y, names):
        if coeff[0] != 0:
            # write a new line every {linefeed} terms
            linefeed_terms += 1
            if linefeed > 0 and linefeed_terms > 1 and linefeed_terms % linefeed == 1:
                result += r'\\ '

            # coefficient
            if coeff[1] != -1: # fraction format
                if coeff[1] != 1:
                    result += '+ \\frac{%d}{%d}'%(coeff[0],coeff[1])
                elif coeff[0] != 1:
                    result += f'+ {coeff[0]} '
                else:
                    result += f'+ '
            else: # decimal format
                result += f'+ {round(coeff[0],precision)}'
            tmp = sp.latex(sp.sympify(name))
            flg = 0
            if formatt == 0:
                parenthesis = 0
                for char in tmp:
                    if char == '(':
                        parenthesis += 1
                    elif char == ')':
                        parenthesis -= 1
                    if parenthesis == 0 and (char == '+' or char == '-'):
                        flg = 1
                        break
                if flg == 0:
                    result += '\\sum ' + sp.latex(sp.sympify(name))
                else:
                    result += '\\sum (' + sp.latex(sp.sympify(name)) +')'
            else:
                result += 's(' + sp.latex(sp.sympify(name)) +')'

    if dectofrac:
        i = 0
        while i < len(result):
            if result[i] == '.':
                j1 = i - 1
                while j1 >= 0 and 48 <= ord(result[j1]) <= 57: # '0'~'9'
                    j1 -= 1
                j2 = i + 1
                while j2 < len(result) and 48 <= ord(result[j2]) <= 57:
                    j2 += 1
                a , b = int(result[j1+1:i]) , int(result[i+1:j2])
                m = 10 ** (j2 - i - 1)
                a , b = a*m + b , m
                _gcd = gcd(a,b)
                a //= _gcd
                b //= _gcd
                result = result[:j1+1] + '\\frac{%d}{%d}'%(a,b) + result[j2:]
                i = j1 + 10
            i += 1

    if result.startswith('+ '):
        result = result[2:]
    if formatt == 0:
        result = '$$' + result + '$$'
    else:
        result = result.replace(' ','').replace('\\','').replace('frac','')
        result = result.replace('left','').replace('right','')
        result = result.replace('}{','/').replace('{','').replace('}','')

    return result

if __name__ == '__main__':
    print(prettyprint([(1,2)], ['a(a-0.5*b)^2'],dectofrac=True))