from collections import defaultdict

from ..polytools import verify_hom_cyclic

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


def poly_get_factor_form(poly):
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
        result.append('p(a)' if factors['a'] == 1 else 'p(a%d)'%factors['a'])
        del factors['a'], factors['b'], factors['c']
    if factors['a-b'] > 0 and factors['a-b'] == factors['b-c'] and factors['a-b'] == factors['a-c']:
        result.append('p(a-b)' if factors['a-b'] == 1 else 'p(a-b)%d'%factors['a-b'])
        if factors['a-b'] % 2 == 1: coeff *= -1
        del factors['a-b'], factors['b-c'], factors['a-c']
    
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