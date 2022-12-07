# author: https://github.com/ForeverHaibara
import warnings
import re

import sympy as sp
import numpy as np
from scipy.optimize import linprog
from scipy.optimize import OptimizeWarning

from ..utils.basis_generator import arraylize, arraylize_sp, invarraylize, reduce_basis, generate_expr, generate_basis
from ..utils.text_process import deg, next_permute, reflect_permute, cycle_expansion
from ..utils.text_process import PreprocessText, PreprocessText_GetDomain, prettyprint
from ..utils.root_guess import rationalize, rationalize_array, root_findroot, root_tangents
from ..utils.root_guess import verify
from .structsos.structsos import SOS_Special


def up_degree(poly, n, updeg):
    """
    Generator that returns 
    """
    
    for m in range(n, updeg+1):
        codeg = m - n 
        if codeg > 0:
            poly_mul = poly * sp.polys.polytools.Poly(f'a^{codeg}+b^{codeg}+c^{codeg}')
            multiplier = f'a^{codeg}' if codeg > 1 else 'a'
        else:
            poly_mul = poly 
            multiplier = None 
        
        yield multiplier, poly_mul, m 


def exact_coefficient(poly, multipliers, y, names, polys, sos_manager):
    '''
    Get the exact coefficients of y that sum up to polynomials 

    Returns 
    -------
    y: arraylike
    
    names: list

    equal: bool
        whether it is an exact equation
    '''
    
    for multiplier in multipliers:
        if multiplier is None:
            continue 
        poly = poly * sp.polys.polytools.Poly(cycle_expansion(multiplier))

    def _filter_zero(y, names, polys):
        index = list(filter(lambda i: y[i][0] != 0 , list(range(len(y)))))
        y     = [y[i]     for i in index] 
        names = [names[i] for i in index] if names is not None else names 
        polys = [polys[i] for i in index] if polys is not None else polys 
        return y , names , polys 

    # Approximates the coefficients to fractions if possible
    y = rationalize_array(y, reliable = True)
        
    # Filter out zero coefficients
    y , names , polys = _filter_zero(y, names, polys)
    if polys is None:
        polys = [sp.polys.polytools.Poly(cycle_expansion(name),
                                    domain = PreprocessText_GetDomain(name)) for name in names]

    # verify whether the equation is strict
    equal = verify(y, polys, poly, 0)
    
    if not equal: 
        # backsubstitude to re-solve the coefficients
        try:
            dict_monom, inv_monom = sos_manager.InquireMonoms(deg(poly))
            b2 = arraylize_sp(poly, dict_monom, inv_monom)
            basis: sp.Matrix = sp.Matrix([arraylize_sp(poly, dict_monom, inv_monom) for poly in polys])
            basis = basis.reshape(len(polys), b2.shape[0]).T
            new_y = basis.LUsolve(b2)
            new_y = rationalize_array(new_y, reliable=True)
            for coeff in new_y:
                if coeff[0] < 0:
                    break 
            else:
                if verify(new_y, polys, poly, 0):
                    equal = True 
                    y , names , polys = _filter_zero(new_y, names, polys)
        except:
            equal = False 
        
        if not equal:
            try:
                # print(y)
                # handle sqrt if any 
                sqrt_ext = re.findall('\d+\^0.5', sos_manager.polytxt)
                if sqrt_ext is not None and len(sqrt_ext) > 0:
                    sqrt_ext = sp.sqrt(int(sqrt_ext[0][:-4]))
                    y = [sp.nsimplify(ip/iq, [sqrt_ext], tolerance = 1e-8).as_numer_denom() 
                                for ip,iq in y]
            except:
                pass 
            
        
    return y , names, equal 


def SOS(poly, tangents = [], maxiter = 5000, roots = [], tangent_points = [], updeg = 10,
        silent = False, show_tangents = True, show_roots = True,
        mod = None, verifytol = 1e-8,
        precision = 6, linefeed = 2):
    '''
    Represent a cyclic, homogenous, 3-variable (a,b,c) polynomial into Sum of Squares form.

    Params
    -------
    tangents: list of str, e.g. ['a+b-c']
        Additional tangent inputs.

    maxiter: unsigned int
        Maximum iteration in searching roots. Set to zero to disable root searching. 

    roots: list of tuple, e.g. [(1/3,1/3)]
        A list of initial root guess (a,b)  (where WLOG c = 1 by homogenousity). 

    tangent_points: list of tuple, e.g. [(1/3,1/3)]
        An additional list of tangent points based on which the tangents are automatically generated.

    updeg: int
        If one try fail, it will automatically multiply the polynomial by \sum (a^t) and retry. 
        Repeat until it has reached the degree of {updeg} and still fails.
    
    silent: bool
        If silent == true, then no information will be printed. Dominates other printing settings.

    show_tangents: bool
        Whether to print out the adopted tangents.

    show_roots: bool
        Whether to print out the potential roots or local minima.
    
    mod: unsigned int / tuples ...
        Denominator guesses for approximating the coefficients into fractions.

    verifytol: float
        Each coefficient of the SOS result must be close to the accurate coefficient with bound 
        {verifytol}.

    precision: unsigned int
        Decimal precision of displaying result.

    linefeed: unsigned int
        Feed a new line every certain terms, no linefeed if set to zero.

    Return
    -------
    result: str
        The result of SOS. If it is an empty string, it means that it has failed.
    '''
    warns = []

    original_poly = poly
    retry = True

    # get the polynomial from text and obtain the degree
    poly = PreprocessText(poly,cyc=True)
    n = deg(poly)
    original_n = n 

    if type(tangents) == str:
        tangents = [tangents]
    tangents += ['a2-bc','a3-bc2','a3-b2c']

    if type(tangent_points) == tuple:
        tangent_points = [tangent_points]
    
    # search the roots
    strict_roots = []
    if maxiter:
        roots, strict_roots = root_findroot(poly, maxiter=maxiter, roots=roots)
        if show_roots and not silent:
            print('Roots =',roots)
        
        roots += tangent_points

        # generate the tangents
        tangents += root_tangents(roots)
        if show_tangents and not silent:
            print('Tangents =',tangents)

    while retry:
        if type(poly) == str:
            poly = PreprocessText(poly,cyc=True)
            n = deg(poly)
        
        dict_monom , inv_monom = generate_expr(n)

        # generate basis with degree n
        names, polys, basis = generate_basis(n,dict_monom,inv_monom,tangents,strict_roots)
        b = arraylize(poly,dict_monom,inv_monom)
        
        # reduce the basis according to the strict roots
        names, polys, basis = reduce_basis(n, dict_monom, inv_monom, names, polys, basis, strict_roots)
        x = None
        
        if len(names) > 0:
            with warnings.catch_warnings(record=True) as __warns:
                warnings.simplefilter('once')
                try:
                    x = linprog(np.ones(basis.shape[0]), A_eq=basis.T, b_eq=b, method='simplex')
                #, options={'tol':1e-9})
                except:
                    pass
                warns += __warns
    
        if len(names) == 0 or x is None or not x.success:
            if not silent:
                print(f'Failed with degree {n}, basis size = {len(names)} x {len(inv_monom)}')
            if n < updeg:
                # move up a degree and retry!
                n += 1
                poly = f's(a{n - original_n})(' + original_poly + ')'
            else:
                if not silent:
                    for warn in warns:
                        if issubclass(warn.category, OptimizeWarning):
                            #warnings.warn('Unstable Computation')
                            print('Warning: Unstable computation due to too large basis or coefficients.')
                            break
                return ''
        else: # success
            retry = 0

    # Approximates the coefficients to fractions if possible
    rounding = 0.1
    y = rationalize_array(x.x, reliable=True)
        
    # Filter out zero coefficients
    index = list(filter(lambda i: y[i][0] != 0 , list(range(len(y)))))
    y = [y[i] for i in index]
    names = [names[i] for i in index]
    polys = [polys[i] for i in index]

    # verify whether the equation is strict
    if not verify(y, polys, poly, 0): 
        # backsubstitude to re-solve the coefficients
        equal = False 
        try:
            b2 = arraylize_sp(poly,dict_monom,inv_monom)
            basis = sp.Matrix([arraylize_sp(poly,dict_monom,inv_monom) for poly in polys])
            basis = basis.reshape(len(polys), b2.shape[0]).T
        
            new_y = basis.LUsolve(b2)
            new_y = [(r.p, r.q) for r in new_y]
            for coeff in new_y:
                if coeff[0] < 0:
                    break 
            else:
                if verify(new_y, polys, poly, 0):
                    equal = True 
                    y = new_y
                
                    # Filter out zero coefficients
                    index = list(filter(lambda i: y[i][0] !=0 , list(range(len(y)))))
                    y = [y[i] for i in index]
                    names = [names[i] for i in index]
                    polys = [polys[i] for i in index]
        except:
            pass             
    else:
        equal = True 
    
    # obtain the LaTeX format
    result = prettyprint(y, names, precision=precision, linefeed=linefeed)
    if not silent:
        print(result)

    return result



if __name__ == '__main__':
    s = 's(a2)2-3s(a3b)'
    # s = r'(21675abcs(a)3+250s(a2b)s(a)3-185193abcs(a2b))/250'
    # x = SOS(s,[],maxiter=1,precision=10,updeg=10)

    # Hexagon
    s = 's(a2b-ab2)2+s(a3b3+a4bc-0a2b3c-5a2c3b+3a2b2c2)'
    s = 'p(a-b)2'
    s = 's((a2c-b2c-2*(a2b-abc)+3*(ab2-abc))2)'

    # Hexagram 
    # s = PreprocessText('s(a3b3+a4bc-0a2b3c-5a2c3b+3a2b2c2)')
    # s = PreprocessText('s(a3b3+7a4bc-29a3b2c+12a3bc2+9a2b2c2)s(a)')
    s = 's(16a5bc+4a4b3-80a4b2c+3a4bc2+7a4c3+64a3b3c-14a3b2c2)'
    s = 's(72a5bc+24a4b3+156a4b2c-453a4bc2+44a4c3+176a3b3c-19a3b2c2)'

    # Eighth Star
    s = ' s(18a4b4+9a5b2c+11a4b3c-66a3b4c+2a2b5c+10a4b2c2+16a3b3c2)-abcs(a(b2-3c2+2bc-5(ab-bc)+3(ac-bc))2)'
    s = 's(18a4b4+9a5b2c+11a4b3c-66a3b4c+2a2b5c+10a4b2c2+16a3b3c2)'

    # Hexagon
    # s = 's(a2b-3/2ab2+1/2abc)2s(a)+0s(a2b-ab2)2s(a)+0s(b(ab2-abc)2)'
    # s = '(6s(a)s(c3b)-abc(37s(a2)-19s(ab)))s(ab)'
    # s = '(s(a)2s(a2b)-9abcs(a2))s(ab)'
    # s = 's(a(a2c-b2c+-3/1*(a2b-abc)+5/1*(ab2-abc))2)+11/21s(c(2ab2-ca2-a2b-bc2+c2a)2)'
    # s = 's(a2c(a-b)(a+c-4b))s(ab)+1/3s(b(a2b-3ab2+2abc)2)'
    # s = 's(c2a(c-b)(c+a-4b))s(cb)+1/3s(b(c2b-3cb2+2cba)2)'
    # s = '(s(a2(a-b)(a2+ab-5bc))-s(a(a2-ab+0(b2-ab)-7/10(ac-ab)+2/3(bc-ab))2))s(ab)-0*(29-21*117*117/10000)/(1-117*117/10000)s(a)s(a2b-ab2)2'
    
    
    s = '(s(a2+ab))2-4s(a)s(a2b)'
    s = PreprocessText(s) if isinstance(s, str) else s 
    print(SOS_Special(s, deg(s)))