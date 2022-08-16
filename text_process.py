import sympy as sp
import numpy as np
from math import gcd
import re 

def NextPermute(f: str) -> str:
    '''a^3 * b^2 * c   ->   b^3 * c^2 * a'''
    fb = f.replace('a','#')
    fb = fb.replace('c','a')
    fb = fb.replace('b','c')
    fb = fb.replace('#','b')
    return fb

def ReflectPermute(f: str) -> str:
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


def PreprocessText_Cyclize(poly, tol=1e-10):
    '''automatically perform cycle expansion on poly if it is not cyclic'''
    cyc_poly = sp.polys.polytools.Poly(CycleExpansion(poly))
    poly = sp.polys.polytools.Poly(poly)
    for coeff in (cyc_poly - 3*poly).coeffs():
        # some coefficient is larger than tolerance, not cyclic
        if abs(coeff) > tol:
            return cyc_poly
    return poly


def CheckHomCyclic(poly, n):
    '''check whether a polynomial is homogenous and cyclic'''
    if len(poly.args) != 4:
        for monom in poly.monoms():
            if sum(monom) != n:
                return False, False
        return True, False

    coeffs = {}
    for coeff, monom in zip(poly.coeffs(), poly.monoms()):
        if sum(monom) != n:
            return False, False
        coeffs[(monom[0], monom[1])] = coeff 
        
    for i in range((n-1)//3+1, n+1):
        # 0 <= k = n-i-j <= i
        for j in range(max(0,n-2*i), min(i+1,n-i+1)):
            # a^i * b^j * c^{n-i-j}
            u = coeffs.get((i,j))
            v = coeffs.get((j,n-i-j))
            if u == v: # Nones are treated the same
                w = coeffs.get((n-i-j,i))
                if u == w:
                    continue 
            return True, False 
    return True, True 


def PreprocessText(poly, cyc=False, retText=False, cancel=False, variables = None):
    '''
    Params
    -------
    cyc: bool  
        check whether the polynomial is cyclic, if not then cyclize it
    retText: bool
        whether return the preprocessed text rather than a sympy polynomial
    cancel: bool
        whether cancel the denominator when encountering fractions 

    Return
    -------
    '''
    poly = poly.lower()
    poly = PreprocessText_DeLatex(poly)
    poly = PreprocessText_Expansion(poly)
    poly = PreprocessText_Completion(poly)
    
    if retText:
        if cyc:
            return CycleExpansion(poly)
        else:
            return poly
        
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
                if not frac[1].is_constant():
                    poly = sp.polys.polytools.Poly(frac[0])
                    return poly , True 
                else:
                    poly = sp.polys.polytools.Poly(poly)
                    return poly , False 
            except:
                return None, True 
        
        try:
            poly = sp.polys.polytools.Poly(poly)
        except:
            return None
    else:
        if cyc:
            poly = PreprocessText_Cyclize(poly)
        else:
            try:
                poly = sp.polys.polytools.Poly(poly)
            except:
                poly = None
                
    if cancel:
        return poly , False 
    return poly


def DegreeofZero(poly):
    '''Compute the degree of a homogeneous zero polynomial
    idea: delete the additions and substractions, which do not affect the degree'''
    poly = poly.lower()
    poly = PreprocessText_DeLatex(poly)
    poly = PreprocessText_Expansion(poly)
    poly = PreprocessText_Completion(poly)
    
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
    #     degree = deg(sp.polys.polytools.Poly(poly))
        poly = sp.fraction(sp.sympify(poly))
        if poly[1].is_constant():
            degree = deg(sp.polys.polytools.Poly(poly[0]))
        else:
            degree = deg(sp.polys.polytools.Poly(poly[0])) - deg(sp.polys.polytools.Poly(poly[1]))
    except:
        degree = 0
        
    return degree


def getsuffix(name):
    try:
        bracket = 0
        right = 0
        left = 0 
        j = len(name) - 1
        is_square = False 
        while j >= 0:
            if name[j] == '^':
                is_square = True
            elif name[j] == ')':
                if bracket == 0: # first right bracket 
                    if is_square:
                        right = j
                    else: # not square, e.g. a*(a-b)*(a-c)
                        return None 
                bracket += 1
            elif name[j] == '(':
                bracket -= 1
                if bracket == 0:
                    left = j
                    break 
            j -= 1
        
        # permute the first alphabet to 'b' for alignment,
        # e.g. (a2-b2+2bc-2ca) and (b2-c2+2ca-2ab) are the same
        for i in range(left, right):
            if name[i] in 'abc':
                alpha = name[i]
                break 
        else:
            return None 

        for i in range(left, -1, -1):
            if name[i] == '*':
                cut = i 
                break 
        
        if alpha == 'a':
            return (NextPermute(name), cut, left, right)
        elif alpha == 'c':
            return (NextPermute(NextPermute(name)), cut, left, right)
        return (name, cut, left, right)
    except:
        return None 


def TextCompresser(y, names):   
    print(y)
    new_y = []
    new_names = []

    suffixes = {}
    for coeff, name in zip(y, names):
        result = getsuffix(name)
        if result is not None:
            name, cut, left, right = result 
            analogue = suffixes.get(name[left:])
            if analogue is not None: 
                analogue.append((coeff, name[:cut]))
            else:
                suffixes[name[left:]] = [(coeff, name[:cut])]
        else:
            new_y.append(coeff)
            new_names.append(name)

    for suffix, val in suffixes.items():
        if len(val) > 1: # more than 1
            merge = 0 
            p , q = 0, 1
            is_fraction = True 
            for coeff, _ in val:
                is_fraction = is_fraction and (isinstance(coeff[0], int) or coeff[0].is_integer)
                if is_fraction:
                    p = gcd(p, coeff[0]) 
                print(q)
                q = q * coeff[1] // gcd(q, coeff[1])

            if not is_fraction:
                p = 1
                
            for coeff, prefix in val:
                is_fraction = (isinstance(coeff[0], int) or coeff[0].is_integer)
                if is_fraction:
                    merge += (coeff[0]//p) * (q//coeff[1]) * sp.sympify(prefix)
                else: # p = 1
                    merge += coeff[0] * (q//coeff[1]) * sp.sympify(prefix)
        else:
            # only one term
            coeff, prefix = val[0]
            p , q = coeff 
            merge = sp.sympify(prefix)
                  
        new_y.append((p,q))
        new_names.append(merge * sp.sympify(suffix))
        
    return new_y, new_names 


def TextSorter(y, names):
    '''
    Sort the texts of 'sum of square' in order to balance the length of each line.
    '''

    # first evaluate the length of latex of each name 
    lengths = []
    for coeff, name in zip(y, names):
        if isinstance(name, str):
            latex = sp.latex(sp.sympify(name))
        else:
            latex = sp.latex(name) 
        length = max(len(str(coeff[0])), len(str(coeff[1]))) + 4 # assume '+sum' is of length 4
        for i in latex:
            if 48 <= ord(i) <= 57 or 97 <= ord(i) <= 99 or i == '+' or i == '-': # 0123456789 abc +-
                length += 1
        lengths.append(length)

    # after that, sort the lengths
    index = sorted(list(range(len(lengths))), key = lambda i: lengths[i])

    # length of each line should not exceeed the linelength
    linelength = max(40, sum(lengths) * 2 // len(lengths))

    linefeed = [False] * len(lengths)
    accumulate_length = 10 # the first line is longer because it starts with f(a,b,c)=
    new_y = []
    new_names = []
    for j in range(len(lengths)):
        i = index[j]
        accumulate_length += lengths[i]
        if accumulate_length > linelength: # start a new line
            linefeed[j] = True 
            accumulate_length = lengths[i]
        
        new_y.append(y[i])
        new_names.append(names[i])
    
    return new_y, new_names, linefeed 


def prettyprint(y, names, precision=6, linefeed=2, formatt=0, dectofrac=False):
    '''
    Prettily format a cyclic polynomial sum into a certain formatt

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
            if isinstance(linefeed, list):
                if linefeed[linefeed_terms-1]:
                    result += r'\\ '
            elif linefeed > 0 and linefeed_terms > 1 and linefeed_terms % linefeed == 1:
                result += r'\\ '

            # coefficient
            if coeff[1] != -1: # fraction format
                if coeff[1] != 1:
                    coeff_str = sp.latex(coeff[0]) if isinstance(coeff[0], sp.Expr) else str(coeff[0]) 
                    result += '+ \\frac{%s}{%d}'%(coeff_str,coeff[1])
                elif coeff[0] != 1:
                    coeff_str = sp.latex(coeff[0]) if isinstance(coeff[0], sp.Expr) else str(coeff[0]) 
                        
                    if isinstance(coeff[0], sp.Add):
                        coeff_str = '\\left(%s\\right)'%coeff_str 
                    result += f'+ {coeff_str} '
                else:
                    result += '+ '
            else: # decimal format
                result += f'+ {round(coeff[0],precision)}'
            
            if isinstance(name, str):
                latex = sp.latex(sp.sympify(name))
            else:
                latex = sp.latex(name)
            flg = 0
            if formatt == 0:
                parenthesis = 0
                for char in latex:
                    if char == '(':
                        parenthesis += 1
                    elif char == ')':
                        parenthesis -= 1
                    if parenthesis == 0 and (char == '+' or char == '-'):
                        flg = 1
                        break
                if flg == 0:
                    result += '\\sum ' + latex
                else:
                    result += '\\sum (' + latex +')'
            else:
                result += 's(' + latex +')'

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
        result = result.replace(' ','').replace('\\','')
        result = result.replace('left','').replace('right','')

        # handle fractions
        parener = lambda x: '(%s)'%x if '+' in x or '-' in x else x
        result = re.sub('frac\{(.*?)\}\{(.*?)\}', 
                        lambda x: '%s/%s'%(parener(x.group(1)), parener(x.group(2))),
                        result)


        # handle sqrt, e.g. 8sqrt{13} -> 8*13^0.5
        result = re.sub('(\d+?)sqrt\{(\d+?)\}', lambda x: x.group(1)+'*'+x.group(2)+'^0.5', result)
        result = re.sub('sqrt\{(\d+?)\}', lambda x: x.group(1)+'^0.5', result)
    
        result = result.replace('{','').replace('}','')
    
    return result



def TextMultiplier(multipliers):
    """
    Return the LaTeX text and formatted text for the multipliers.
    """
    
    if multipliers is None or len(multipliers) == 0:
        return '', '' 
    
    merged_multipliers = {}
    for multiplier in multipliers:
        if multiplier is None:
            continue
        multiplier = multiplier.replace(' ','')
        t = merged_multipliers.get(multiplier) 
        if t is not None:
            merged_multipliers[multiplier] = t + 1
        else:
            merged_multipliers[multiplier] = 1

    result_latex , result_txt = '' , ''
    for multiplier, power in merged_multipliers.items():
        multiplier = sp.latex(sp.sympify(multiplier))
        power_suffix = '^{%d}'%power if power > 1 else ''
        result_latex += '\\left(\\sum %s\\right)%s'%(multiplier, power_suffix)
        result_txt   += 's(%s)%s'%(multiplier, power_suffix)
    
    # result_txt = result_txt.replace('^','')
    return result_latex, result_txt


if __name__ == '__main__':
    print(prettyprint([(1,2)], ['a(a-0.5*b)^2'],dectofrac=True))