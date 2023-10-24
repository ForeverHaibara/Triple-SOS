from collections import defaultdict

import sympy as sp

from .cyclic import CyclicSum, CyclicProduct
from ..polytools import verify_hom_cyclic, deg
from ..text_process import pl

def poly_get_standard_form(poly, formatt = 'short', is_cyc = None):
    if formatt == 'short':
        def _title_parser(char, deg):
            return '' if deg == 0 else (char if deg == 1 else (char + str(deg)))
        def _formatter(x):
            if x == 1:
                return '+'
            elif x >= 0:
                return f'+{x}'
            elif x == -1:
                return f'-'
            else:
                return f'{x}'
        if is_cyc is not None and is_cyc == True:
            txt = ''
            for coeff, monom in zip(poly.coeffs(), poly.monoms()):
                a , b , c = monom 
                if a >= b and a >= c:
                    if a == b and a == c:
                        txt += _formatter(coeff/3) + _title_parser('a',a) + _title_parser('b',b) + _title_parser('c',c)
                    elif (a != b and a != c) or a == b:
                        txt += _formatter(coeff) + _title_parser('a',a) + _title_parser('b',b) + _title_parser('c',c)
            if txt.startswith('+'):
                txt = txt[1:]
            return 's(' + txt + ')'

        else: # not cyclic 
            txt = ''
            for coeff, monom in zip(poly.coeffs(), poly.monoms()):
                a , b , c = monom
                txt += _formatter(coeff) + _title_parser('a',a) + _title_parser('b',b) + _title_parser('c', c)
            if txt.startswith('+'):
                txt = txt[1:]
            return txt 


def poly_get_factor_form(poly, return_type='text'):
    """
    Get the factorized form of a polynomial.

    Parameters
    ----------
    poly : Poly
        The polynomial to be factorized.
    return_type : str
        The type of the return value. Can be 'text' or 'expr'.
    """
    a, b, c = sp.symbols('a b c')
    coeff, parts = poly.factor_list()
    factors = defaultdict(int)
    is_cyc = {}
    origin_polys = {}
    for part, mul in parts:
        name = str(part)[5:].split(',')[0].replace(' ','')
        factors[name] = mul
        is_cyc[name] = verify_hom_cyclic(part)[1]
        origin_polys[name] = part
    result = []

    if factors['a'] > 0 and factors['a'] == factors['b'] and factors['a'] == factors['c']:
        if return_type == 'expr':
            result.append(CyclicProduct(a ** factors['a']))
        elif return_type == 'text':
            result.append('p(a)' if factors['a'] == 1 else 'p(a%d)'%factors['a'])
        del factors['a'], factors['b'], factors['c']

    if factors['a-b'] > 0 and factors['a-b'] == factors['b-c'] and factors['a-b'] == factors['a-c']:
        if return_type == 'expr':
            result.append(CyclicProduct((a-b) ** factors['a-b']))
        elif return_type == 'text':
            result.append('p(a-b)' if factors['a-b'] == 1 else 'p(a-b)%d'%factors['a-b'])

        if factors['a-b'] % 2 == 1: coeff *= -1
        del factors['a-b'], factors['b-c'], factors['a-c']

    if return_type == 'text':
        _formatter1 = lambda x: '' if x == 1 else str(x)
        _formatter2 = lambda x: x if x.startswith('s(')  or len(x) == 1 else '(%s)'%x
        result += [
            (_formatter2(poly_get_standard_form(origin_polys[part], is_cyc = is_cyc[part]))
                + _formatter1(mul)) if mul > 0 else ''
                for part, mul in factors.items()
        ]

        if coeff == 1: coeff = ''
        elif coeff == -1: coeff = '-'
        return str(coeff) + ''.join(sorted(result, key = lambda x: len(x)))

    elif return_type == 'expr':
        def _formatted2(part):
            x = origin_polys[part]
            x2 = poly_get_standard_form(x, is_cyc = is_cyc[part])
            if x2.startswith('s('):
                return CyclicSum(pl(x2[2:-1]).as_expr())
            else:
                return x.as_expr()
        result += [
            (_formatted2(part) ** mul if mul > 0 else sp.S(1))
                for part, mul in factors.items()
        ]
        return coeff * sp.Mul(*result)


def latex_coeffs(poly, tabular=True, document=True):
    """
    Return the LaTeX format of the coefficient triangle.
    """
    if poly is None:
        return ''
    if isinstance(poly, str):
        poly_str = poly
        poly = pl(poly)
    else:
        poly_str = 'f(a,b,c)'

    n = deg(poly)
    emptyline = '\\\\ ' + '\\ &' * (n * 2) + '\\  \\\\ '
    strings = ['' for i in range(n+1)]

    coeffs = poly.coeffs()
    monoms = poly.monoms()
    monoms.append((-1,-1,0))  # tail flag
    t = 0
    for i in range(n+1):
        for j in range(i+1):
            if monoms[t][0] == n - i and monoms[t][1] == i - j:
                txt = sp.latex(coeffs[t])
                t += 1
            else:
                txt = '0'
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
