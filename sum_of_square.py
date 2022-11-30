# author: https://github.com/ForeverHaibara 
from math import gcd
from itertools import product
import warnings
import re

import sympy as sp
import numpy as np
from scipy.optimize import linprog
from scipy.optimize import OptimizeWarning
from sympy.solvers.diophantine.diophantine import diop_DN

from basis_generator import arraylize, arraylize_sp, invarraylize, reduce_basis, generate_expr, generate_basis
from text_process import deg, next_permute, reflect_permute, cycle_expansion
from text_process import PreprocessText, PreprocessText_GetDomain, prettyprint
from root_guess import rationalize, rationalize_array, root_findroot, root_tangents
from root_guess import verify, optimize_determinant, square_perturbation
from peeling import FastPositiveChecker


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


def _merge_sos_results(multipliers, y, names, result, abc = False):
    """
    Merge SOS results in DFS.
    """
    if result is None:
        return None, None, None  
    if y is None:
        y = []
        names = [] 
    if multipliers is None:
        multipliers = [] 

    # (multipliers * f - y * names )  *  m' == y' * names'
    m2 , y2 , names2 = result 

    multipliers += m2 
    y = y + y2 

    # names: e.g.  m2 = ['a','a^2'] -> names = ['(a+b+c)*(a^2+b^2+c^2)'*...]
    if len(m2) > 0:
        for i in range(len(names)):
            if names[i][0] != '(' or names[i][-1] != ')':
                names[i] = '(' + names[i] + ')'
            for m in m2:
                names[i] = '('+m+'+'+next_permute(m)+'+'+next_permute(next_permute(m))+')*' + names[i]
    if abc:
        for i in range(len(names2)):
            if names2[i][0] != '(' or names2[i][-1] != ')':
                names2[i] = '(' + names2[i] + ')'
            names2[i] = 'a*b*c*' + names2[i]

    names = names + names2 

    return multipliers , y , names 


def _try_perturbations(poly, degree, multipliers, a, b, base, name: str, times = 4):
    substractor = sp.polys.polytools.Poly(cycle_expansion(name))
    
    for t in square_perturbation(a, b, times = times):
        y = [t * base]
        names = [name]
        poly2 = poly - y[0] * substractor 
        multipliers , y , names = _merge_sos_results(multipliers, y, names, SOS_Special(poly2, degree))
        if y is not None:
            break 

    return multipliers, y, names 


def SOS_Special(poly, degree, ext = False):
    """
    SOS for special structures.

    Params
    -------
    poly: sp.polys.polytools.Poly
        The target polynomial.

    degree: int
        Degree of the polynomial.

    Returns
    -------
    multipliers: list of str
        The multipliers.
        
    y: list of tuples
        Rational coefficients of each component.

    names:

    """
    coeffs = {}
    for coeff, monom in zip(poly.coeffs(), poly.monoms()):
        if degree > 4 and not isinstance(coeff, sp.Rational): #isinstance(coeff, sp.Float):
            coeff = sp.Rational(*rationalize(coeff, reliable = True))
            # coeff = coeff.as_numer_denom()
        coeffs[monom] = coeff 
    
    if len(coeffs) == 1 and poly.monoms()[0] == (0,0,0): # zero polynomial
        return [], [(0,1)], [f'a^{degree}+b^{degree}+c^{degree}']

    def coeff(x):
        t = coeffs.get(x)
        if t is None:
            return 0
        return t 
    
    if len(coeffs) <= 6: # commonly Muirhead or AM-GM or trivial ones
        monoms = poly.monoms()
        if len(coeffs) == 1: # e.g.  abc
            if coeff(monoms[0]) >= 0:
                monom = monoms[0]
                return [], [coeff(monom) / 3], [f'a^{monom[0]}*b^{monom[1]}*c^{monom[2]}']
        elif len(coeffs) == 3: # e.g. (a2b + b2c + c2a)
            if coeff(monoms[0]) >= 0:
                monom = monoms[0]
                return [], [coeff(monom)], [f'a^{monom[0]}*b^{monom[1]}*c^{monom[2]}']
        elif len(coeffs) == 4: # e.g. (a2b + b2c + c2a - 8/3abc)
            if coeff(monoms[0]) >= 0 and coeff(monoms[0])*3 + coeff((degree//3, degree//3, degree//3)) >= 0:
                monom = monoms[0]
                return [], \
                    [coeff(monom), coeff(monoms[0]) + coeff((degree//3, degree//3, degree//3))/3], \
                    [f'a^{monom[0]}*b^{monom[1]}*c^{monom[2]}-a^{degree//3}*b^{degree//3}*c^{degree//3}',
                     f'a^{degree//3}*b^{degree//3}*c^{degree//3}']
        elif len(coeffs) == 6: # e.g. s(a5b4 - a4b4c)
            monoms = [i for i in monoms if (i[0]>i[1] and i[0]>i[2]) or (i[0]==i[1] and i[0]>i[2])]
            monoms = sorted(monoms)
            small , large = monoms[0], monoms[1]
            if coeff(small) >= 0 and coeff(large) >= 0:
                return [], \
                    [coeff(small), coeff(large)], \
                    [f'a^{small[0]}*b^{small[1]}*c^{small[2]}', f'a^{large[0]}*b^{large[1]}*c^{large[2]}']
            elif coeff(large) >= 0 and coeff(large) + coeff(small) >= 0:
                det = 3*large[0]*large[1]*large[2] - (large[0]**3+large[1]**3+large[2]**3)
                deta = small[0]*(large[1]*large[2]-large[0]**2)+small[1]*(large[2]*large[0]-large[1]**2)+small[2]*(large[0]*large[1]-large[2]**2)
                detb = small[0]*(large[2]*large[0]-large[1]**2)+small[1]*(large[0]*large[1]-large[2]**2)+small[2]*(large[1]*large[2]-large[0]**2)
                detc = small[0]*(large[0]*large[1]-large[2]**2)+small[1]*(large[1]*large[2]-large[0]**2)+small[2]*(large[2]*large[0]-large[1]**2)
                det, deta, detb, detc = -det, -deta, -detb, -detc
                # print(det, deta, detb, detc)
                if det > 0 and deta >= 0 and detb >= 0 and detc >= 0:
                    d = gcd(det, gcd(deta, gcd(detb, detc)))
                    det, deta, detb, detc = det//d, deta//d, detb//d, detc//d
                    return [], \
                        [coeff(large)/det, coeff(large) + coeff(small)], \
                        [f'{deta}*a^{large[0]}*b^{large[1]}*c^{large[2]}+{detb}*a^{large[1]}*b^{large[2]}*c^{large[0]}+{detc}*a^{large[2]}*b^{large[0]}*c^{large[1]}-{det}*a^{small[0]}*b^{small[1]}*c^{small[2]}', 
                        f'a^{small[0]}*b^{small[1]}*c^{small[2]}']

    y = None 
    names = None 
    multipliers = [] 

    if degree == 4:
        # Theorem: if 3m(m+n) - (p*p+q*q+p*q) >= 0, then we have formula: ...
        m = coeff((4,0,0))
        n = coeff((2,2,0))
        p = coeff((3,1,0))
        q = coeff((1,3,0))
        if m > 0:
            det = 3*m*(m+n)-(p*p+p*q+q*q)
            if det >= 0 and (p != 0 or q != 0):
                y = [m/2, det/6/m, coeff((2,1,1))+m+n+p+q]
                all_rational = isinstance(m,sp.Rational) and isinstance(n,sp.Rational) and isinstance(p,sp.Rational) and isinstance(q,sp.Rational)
                formatter = (lambda x: x) if all_rational else (lambda x: '(%s)'%sp.simplify(x))
                names = [f'(a*a-b*b+{formatter((p+2*q)/m/3)}*c*a+{formatter((p-q)/m/3)}*a*b-{formatter((2*p+q)/m/3)}*b*c)^2',
                        f'a^2*(b-c)^2',
                        f'a^2*b*c']
                        
                if not all_rational:
                    y = [sp.simplify(i) for i in y]
                
            else:
                # try substracting t*s(ab(a-c-u(b-c))2) and use theorem
                # solve all extrema
                n , p , q = n / m , p / m , q / m
                extrema = sp.polys.polyroots.roots(
                            sp.polys.polytools.Poly(f'2*x^4+{p}*x^3-{q}*x-2'))
                    
                symmetric = lambda _x: ((2*q+p)*_x + 6)*_x + 2*p+q
                for root in extrema:
                    if root.is_real and root > -1e-6 and sp.minimal_polynomial(root, polys=True).degree() < 3:
                        symmetric_axis = symmetric(root)
                        if symmetric_axis >= 0:
                            det = sp.simplify(p*p+p*q+q*q-3*n-3-symmetric_axis*symmetric_axis/(4*(root*root*(root*root+1)+1)))
                            if det == 0 or (det < 0 and isinstance(det, sp.Rational)):
                                # we need simplify det here for quadratic roots
                                # e.g. (s(a2+ab))2-4s(a)s(a2b)
                                if det < 0:
                                    # we consider rational approximations
                                    numer_r = float(root)
                                    for rounding in (.5, .2, .1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8):
                                        numer_r2 = sp.Rational(*rationalize(numer_r, rounding=rounding, reliable=False))
                                        symmetric_axis = symmetric(numer_r2)
                                        if symmetric_axis >= 0 and \
                                            p*p+p*q+q*q-3*n-3-symmetric_axis*symmetric_axis/(4*(numer_r2**2*(numer_r2**2+1)+1)) <= 0:
                                            root = numer_r2
                                            break 
                            elif det > 0:
                                continue 
                        else:
                            continue
                    else:
                        numer_r = complex(root)
                        if numer_r.imag < -1e-6 or abs(numer_r.imag) > 1e-12 or symmetric(numer_r.real) < 1e-6:
                            continue
                        numer_r = numer_r.real 
                        for rounding in (.5, .2, .1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8):
                            numer_r2 = sp.Rational(*rationalize(numer_r, rounding=rounding, reliable=False))
                            symmetric_axis = symmetric(numer_r2)
                            if symmetric_axis >= 0 and \
                                p*p+p*q+q*q-3*n-3-symmetric_axis*symmetric_axis/(4*(numer_r2**2*(numer_r2**2+1)+1)) <= 0:
                                root = numer_r2 
                                break 
                        else:
                            continue 
                    
                    y = [sp.simplify(symmetric_axis / (2*(root*root*(root*root+1)+1)) * m)]
                    names = [f'a*b*(a-c-({root})*(b-c))^2']
                    poly2 = poly - y[0] * sp.sympify(cycle_expansion(names[0]))
                    multipliers , y , names = _merge_sos_results(multipliers, y, names, SOS_Special(poly2, 4))



    elif degree == 5:
        if coeff((5,0,0))==0:
            # try hexagon
            multipliers = ['a*b']
            poly2 = poly * sp.polys.polytools.Poly('a*b+b*c+c*a')
            multipliers , y , names = _merge_sos_results(multipliers, y, names, SOS_Special(poly2, 7))
        else:
            a = coeff((5,0,0))
            if a > 0:
                # try Schur to hexagon
                b = coeff((4,1,0))
                if b >= -2 * a:
                    fpc = FastPositiveChecker()
                    name = '(a^2+b^2+c^2-a*b-b*c-c*a)*a*(a-b)*(a-c)'
                    poly2 = poly - a * sp.sympify(cycle_expansion(name))
                    fpc.setPoly(poly2)
                    if fpc.check() == 0:
                        y = [a]
                        names = [name]
                        multipliers , y , names = _merge_sos_results(multipliers, y, names, SOS_Special(poly2, 5))
                    if y is None and b >= -a:
                        name = 'a^3*(a-b)*(a-c)'
                        poly2 = poly - a * sp.sympify(cycle_expansion(name))
                        fpc.setPoly(poly2)
                        if fpc.check() == 0:
                            y = [a]
                            names = [name]
                            multipliers , y , names = _merge_sos_results(multipliers, y, names, SOS_Special(poly2, 5))
                        

            

    elif degree == 6:
        if coeff((6,0,0))==0 and coeff((5,1,0))==0 and coeff((5,0,1))==0:
            # hexagon
            if coeff((4,2,0)) != coeff((4,0,2)):
                # first try whether we can cancel this two corners in one step 
                # CASE 1. 
                if coeff((4,2,0)) == 0 or coeff((4,0,2)) == 0:
                    if coeff((4,0,2)) == 0:
                        y = [coeff((4,2,0))]       
                        names = ['(a*a*b+b*b*c+c*c*a-3*a*b*c)^2']
                    else:
                        y = [coeff((4,0,2))]
                        names = ['(a*b*b+b*c*c+c*a*a-3*a*b*c)^2']                 

                    poly2 = poly - y[0] * sp.sympify(names[0])
                    v = 0 # we do not need to check the positivity, just try
                    if v != 0:
                        y, names = None, None 
                    else: # positive
                        multipliers , y , names = _merge_sos_results(multipliers, y, names, SOS_Special(poly2, 6))

                # CASE 2. 
                else:
                    a , b = (coeff((4,2,0)) / coeff((4,0,2))).as_numer_denom()
                    if sp.ntheory.primetest.is_square(a) and sp.ntheory.primetest.is_square(b):
                        y = [coeff((4,2,0)) / a] 
                        names = [f'({sp.sqrt(a)}*(a*a*b+b*b*c+c*c*a-3*a*b*c)-{sp.sqrt(b)}*(a*b*b+b*c*c+c*a*a-3*a*b*c))^2']
                    
                        poly2 = poly - y[0] * sp.sympify(names[0])
                        v = 0 # we do not need to check the positivity, just try
                        if v != 0:
                            y, names = None, None 
                        else: # positive
                            multipliers , y , names = _merge_sos_results(multipliers, y, names, SOS_Special(poly2, 6))

                    # CASE 3.
                    # a = t(p^2+1) , b = t(q^2+1)
                    # => b^2p^2 + b^2 = ab(q^2 + 1) => (bp)^2 - ab q^2 = b(a-b)
                    #                            or => (aq)^2 - ab p^2 = a(b-a)
                    # here we should require b is squarefree and not too large
                    elif a < 30 and b < 30: # not too large for solving Pell's equation                        
                        pairs = []
                        for pair in diop_DN(a*b, b*(a-b)):
                            if pair[0] % b == 0:
                                pairs.append((abs(pair[0] // b), abs(pair[1])))
                        for pair in diop_DN(a*b, a*(b-a)):
                            if pair[0] % a == 0:
                                pairs.append((abs(pair[1]), abs(pair[0] // a)))
                        pairs = set(pairs)
                        
                        for p , q in pairs:
                            p , q = abs(p) , abs(q) 
                            t = coeff((4,2,0)) / (p*p + 1)
                            if coeff((3,3,0)) + t * 2 * p * q < 0 or coeff((4,1,1)) + t * 2 * (p + q) < 0:
                                # negative vertex, skip it
                                continue 
                            y = [t]
                            names = [f'(a*a*c-b*b*c-{p}*(a*a*b-a*b*c)+{q}*(a*b*b-a*b*c))^2']
                            poly2 = poly - t * sp.sympify(cycle_expansion(names[0]))
                            multipliers , y , names = _merge_sos_results(multipliers, y, names, SOS_Special(poly2, 6))
                            if y is not None:
                                break                 


            else:# coeff((4,2,0)) == coeff((4,0,2)):
                if coeff((4,2,0)) == 0:
                    # hexagram (star)
                    poly = poly * sp.polys.polytools.Poly('a+b+c')
                    multipliers , y , names = _merge_sos_results(['a'], y, names, SOS_Special(poly, 7))
                else:
                    y = [coeff((4,2,0)) / 3] 
                    names = [f'(a-b)^2*(b-c)^2*(c-a)^2']

                    poly2 = poly - y[0] * 3 * sp.sympify(names[0])
                    v = 0 # we do not need to check the positivity, just try
                    if v != 0:
                        y, names = None, None 
                    else: # positive
                        multipliers , y , names = _merge_sos_results(multipliers, y, names, SOS_Special(poly2, 6))
        elif coeff((5,1,0))==0 and coeff((5,0,1))==0 and coeff((4,2,0))==0 and coeff((4,0,2))==0\
            and coeff((3,2,1))==0 and coeff((3,1,2))==0:
            t = coeff((6,0,0))
            # t != 0 by assumption
            u = coeff((3,3,0))/t
            if u >= -2:
                v = coeff((4,1,1))/t
                roots = sp.polys.polyroots.roots(sp.polys.polytools.Poly('x^3-3*x')-u)
                for r in roots:
                    numer_r = complex(r)
                    if abs(numer_r.imag) < 1e-12 and numer_r.real >= 1:
                        numer_r = numer_r.real 
                        if not isinstance(r, sp.Rational):
                            for gap, rounding in ((.5, .1), (.3, .1), (.2, .1), (.1, .1), (.05, .01), (.01, .002),
                                                (1e-3, 2e-4), (1e-4, 2e-5), (1e-5, 2e-6), (1e-6, 1e-7)):
                                r = sp.Rational(*rationalize(numer_r-gap, rounding=rounding, reliable = False))
                                if r*r*r-3*r <= u and 3*r*(r-1)+v >= 0:
                                    break 
                                rounding *= .1
                            else:
                                break 
                        elif 3*r*(r-1)+v < 0:
                            break 
                        
                        # now r is rational 
                        y = [t/2, t*(u-(r*r*r-3*r)), t*(v+3*r*(r-1)), 
                            coeff((2,2,2))/3+coeff((6,0,0))+coeff((4,1,1))+coeff((3,3,0))]
                        names = [f'(a^2+b^2+c^2+{r}*(a*b+b*c+c*a))*(a-b)^2*(a+b-{r}*c)^2',
                                'a^3*b^3-a^2*b^2*c^2',
                                'a^4*b*c-a^2*b^2*c^2',
                                'a^2*b^2*c^2']


    elif degree == 7:
        if coeff((7,0,0))==0 and coeff((6,1,0))==0 and coeff((6,0,1))==0:
            if coeff((5,2,0))==0 and coeff((5,0,2))==0:
                # star
                if coeff((4,3,0)) == 0:
                    a , b = 1 , 0 
                else:
                    a , b = (coeff((3,4,0)) / coeff((4,3,0))).as_numer_denom()
                    
                if sp.ntheory.primetest.is_square(a) and sp.ntheory.primetest.is_square(b):
                    t = coeff((4,3,0))
                    
                    y = []
                    names = []  

                    if b == 0 and coeff((3,4,0)) == 0: # abc | poly
                        pass 
                    elif t < 0:
                        return 
                    else:
                        if b != 0:
                            z = sp.sqrt(a / b)
                            determinant = '-(3*m*(m+n-(-2*x^2+2*y^2+4*y*z-4*y-2*z^2+4*z-2))-(p-(x^2-2*x*y+2*x+y^2+2*y))^2-(q-(x^2+2*x*y+2*x*z+y^2-2*y*z))^2-(p-(x^2-2*x*y+2*x+y^2+2*y))*(q-(x^2+2*x*y+2*x*z+y^2-2*y*z)))'
                            # determinant = '-3*m^2-3*m*n+6*m*y^2+p^2+p*q-3*p*y^2-2*p*y+q^2-3*q*y^2+2*q*y+3*x^4+12*x^3-x^2*(6*m+3*p+3*q-10*y^2-12)-x*(-2*p*y+6*p+2*q*y+6*q-4*y^2)+3*y^4+4*y^2'
                            determinant = sp.polys.polytools.Poly(determinant).subs('z',z) 
                        else:
                            t = coeff((3,4,0))
                            determinant = '-(3*m*(m+n-(-2*x^2+2*y^2-4*y-2))-(p-(x^2-2*x*y+y^2))^2-(q-(x^2+2*x*y-2*x+y^2+2*y))^2-(p-(x^2-2*x*y+y^2))*(q-(x^2+2*x*y-2*x+y^2+2*y)))'
                            determinant = sp.polys.polytools.Poly(determinant)
                        determinant = determinant.subs((('m',coeff((5,1,1))/t), ('n',coeff((3,3,1))/t), ('p',coeff((4,2,1))/t), ('q',coeff((2,4,1))/t)))#, simultaneous=True)
                        
                        result = optimize_determinant(determinant)
                        if result is None:
                            return 
                        a , b = result 
                        
                        # now we have guaranteed v <= 0
                        if coeff((4,3,0)) != 0:
                            y = [t]
                            names = [f'b*(a^2*b-a*b*c-{z}*(b*c^2-a*b*c)+{a+b}*(a^2*c-a*b*c)+{b-a}*(a*c^2-a*b*c))^2']
                        else: # t = 0
                            y = [t] 
                            names = [f'b*((b*c^2-a*b*c)+{a+b}*(a^2*c-a*b*c)+{b-a}*(a*c^2-a*b*c))^2']


                        poly = poly - t * sp.polys.polytools.Poly(cycle_expansion(names[0]))
                    
                    try:
                        poly = sp.cancel(poly / sp.polys.polytools.Poly('a*b*c'))
                        poly = sp.polys.polytools.Poly(poly)
                        
                        multipliers , y , names = _merge_sos_results(multipliers, y, names, 
                                                                SOS_Special(poly, 4), abc = True)
                    except:
                        # zero polynomial
                        pass 
        
                elif a > 0 and b > 0:
                    # we shall NOTE that we actually do not require a/b is square
                    # take coefficients like sqrt(a/b) also works -- though it is not pretty
                    # but we can make a sufficiently small perturbation such that a/b is square
                    # e.g. a/b = 7/4 = z^2 + epsilon
                    # solve it by Newton's algorithm for squareroot, starting with z = floor(7/4) = 1
                    # 1 -> (1 + 7/4)/2 = 11/8 -> (11/8+14/11)/2 = 233/176
                    
                    name = 'c*(a^2*b-a^2*c-a*b^2+b^2*c)^2'
                    
                    multipliers, y, names = _try_perturbations(poly, degree, multipliers, a, b,
                                                            coeff((4,3,0))/b, name)

            else:
                # hexagon
                
                if coeff((5,2,0)) == 0:
                    a , b = 1 , 0 
                else:
                    a , b = (coeff((2,5,0)) / coeff((5,2,0))).as_numer_denom()   
                
                if sp.ntheory.primetest.is_square(a) and sp.ntheory.primetest.is_square(b):
                    t = coeff((5,2,0)) / b if b != 0 else coeff((2,5,0))
                    
                    y = []
                    names = []  

                    if t < 0:
                        return 

                    # 's((a(u(a2b-abc)-v(a2c-abc)+x(bc2-abc)+y(b2c-abc)+z(ac2-abc)+w(ab2-abc))2))' (u>0,v>0)
                    # z^2 + 2uw = coeff((4,3,0)) = m,   w^2 - 2vz = coeff((3,4,0)) = n
                    # => (w^2-n)^2 = 4v^2(m-2uw)
                    # More generally, we can find w,z such that z^2+2uw < m and w^2-2vz < n
                    # such that the remaining coeffs are slightly positive to form a hexagram (star)
                    m = coeff((4,3,0)) / t
                    n = coeff((3,4,0)) / t

                    u = sp.sqrt(b)
                    v = sp.sqrt(a)
                    candidates = []
                    
                    if v == 0: # first reflect the problem, then reflect it back
                        u , v = v , u
                        a , b = b , a
                        m , n = n , m

                    for w in sp.polys.polyroots.roots(sp.polys.polytools.Poly(f'(x*x-{n})^2-4*{a}*({m}-2*{u}*x)')):
                        if isinstance(w, sp.Rational):
                            z = (w*w - n) / 2 / v 
                            candidates.append((w, z))
                        elif w.is_real is None or w.is_real == False:
                                continue 
                        else:
                            w2 = complex(w).real 
                            w2 -= abs(w2) / 1000 # slight perturbation
                            if m < 2*u*w2:
                                continue
                            z2 = (m - 2*u*w2)**0.5 
                            z2 -= abs(z2) / 1000 # slight perturbation
                            rounding = 1e-2
                            for i in range(4):
                                w = sp.Rational(*rationalize(w2, rounding = rounding, reliable = False))
                                z = sp.Rational(*rationalize(z2, rounding = rounding, reliable = False))
                                rounding *= 0.1
                                if z*z + 2*u*w <= m and w*w - 2*v*z <= n:
                                    candidates.append((w, z))


                    for perturbation in (100, 3, 2):
                        m2 = m - abs(m / perturbation)
                        n2 = n - abs(n / perturbation)
                        for w in sp.polys.polyroots.roots(sp.polys.polytools.Poly(f'(x*x-{n2})^2-4*{a}*({m2}-2*{u}*x)')):
                            if isinstance(w, sp.Rational):      
                                z = (w*w - n) / 2 / v 
                                candidates.append((w, z))
                            elif w.is_real is None or w.is_real == False:
                                continue 
                            else:
                                rounding = 1e-2
                                w2 = complex(w).real 
                                if m + m2 < 4*u*w2:
                                    continue
                                z2 = ((m + m2)/2 - 2*u*w2)**0.5 
                                for i in range(4):
                                    w = sp.Rational(*rationalize(w2, rounding = rounding, reliable = False))
                                    z = sp.Rational(*rationalize(z2, rounding = rounding, reliable = False))
                                    rounding *= 0.1
                                    if z*z + 2*u*w <= m and w*w - 2*v*z <= n:
                                        candidates.append((w, z))
                    
                    


                    candidates = list(set(candidates))
                    if coeff((2,5,0)) == 0: # reflect back
                        u , v = v , u
                        m , n = n , m 
                        a , b = b , a
                        candidates = [(-z, -w) for w,z in candidates]

                    # sort according to their weights
                    weights = [abs(i[0].p) + abs(i[0].q) + abs(i[1].p) + abs(i[1].q) for i in candidates]
                    indices = sorted(range(len(candidates)), key = lambda x: weights[x])
                    candidates = [candidates[i] for i in indices]
                    # print(candidates, u, v, m, n)

                    for w, z in candidates:
                        determinant = '3*(m-(-2*u*v))*(m-(-2*u*v)+n-(-2*u*w+2*u*y-2*u*z+2*v*w-2*v*x+2*v*z-2*w^2-2*w*x-2*w*y-4*w*z+2*x*y-2*x*z-2*y*z-2*z^2))'
                        determinant += '-(p-(-2*u^2+2*u*v-2*u*w-2*u*x-2*u*y-2*u*z-2*v*w+2*x*z+y^2))^2-(q-(2*u*v+2*u*z-2*v^2+2*v*w+2*v*x+2*v*y+2*v*z+2*w*y+x^2))^2'
                        determinant += '-(p-(-2*u^2+2*u*v-2*u*w-2*u*x-2*u*y-2*u*z-2*v*w+2*x*z+y^2))*(q-(2*u*v+2*u*z-2*v^2+2*v*w+2*v*x+2*v*y+2*v*z+2*w*y+x^2))'
                        determinant = -sp.polys.polytools.Poly(determinant)
                        determinant = determinant.subs((('u', u), ('v', v), ('z', z), ('w', w), 
                                            ('m', coeff((5,1,1))/t), ('n', coeff((3,3,1))/t), ('p', coeff((4,2,1))/t), ('q', coeff((2,4,1))/t)))
                        
                        result = optimize_determinant(determinant, soft = True)
                        if result is None:
                            continue
                        a , b = result
                        
                        y = [t] 
                        names = [f'a*({u}*(a^2*b-a*b*c)-{v}*(a^2*c-a*b*c)+{a}*(b*c^2-a*b*c)+{b}*(b^2*c-a*b*c)+{z}*(a*c^2-a*b*c)+{w}*(a*b^2-a*b*c))^2']
                        poly2 = poly - t * sp.polys.polytools.Poly(cycle_expansion(names[0]))
                        multipliers , y , names = _merge_sos_results(multipliers, y, names, SOS_Special(poly2, 7))
                        if y is not None:
                            break 
                elif a > 0 and b > 0:
                    name = 'a*(a-b)^2*(b-c)^2*(c-a)^2'
                    multipliers, y, names = _try_perturbations(poly, degree, multipliers, a, b,
                                                            coeff((5,2,0))/b, name)

    elif degree == 8:
        if coeff((8,0,0))==0 and coeff((7,1,0))==0 and coeff((7,0,1))==0 and coeff((6,2,0))==0 and\
            coeff((6,1,1))==0 and coeff((6,0,2))==0:

            # equivalent to degree-7 hexagon when applying (a,b,c) -> (1/a,1/b,1/c)
            poly2 = f'{coeff((0,3,5))}*a^5*b^2+{coeff((1,2,5))}*a^4*b^3+{coeff((2,1,5))}*a^3*b^4+{coeff((3,0,5))}*a^2*b^5+{coeff((3,3,2))}*a^2*b^2*c^3'
            poly2 += f'+{coeff((0,4,4))}*a^5*b*c+{coeff((1,3,4))}*a^4*b^2*c+{coeff((2,2,4))}*a^3*b^3*c+{coeff((3,1,4))}*a^2*b^4*c'
            poly2 = sp.polys.polytools.Poly(cycle_expansion(poly2))
            result = SOS_Special(poly2, 7)
            if False:
                if coeff((5,1,2))==0 and coeff((5,2,1))==0:
                    # equivalent to degree-4 polynomial with respect to ab, bc, ca
                    m = coeff((4,4,0))
                    p = coeff((3,4,1))
                    n = coeff((2,4,2))
                    q = coeff((1,4,3))
                    if m > 0:
                        r = 3*m*(m+n)-(p*p+p*q+q*q)
                        if r >= 0 and (p != 0 or q != 0):
                            y = [m/2, r/(18*m*(p*p+p*q+q*q)), coeff((2,1,1))+m+n+p+q]
                            names = [f'(a*a*b*b-b*b*c*c+{(p+2*q)/m/3}*c*a*a*b+{(p-q)/m/3}*a*b*b*c-{(2*p+q)/m/3}*b*c*c*a)^2',
                                    f'({p+2*q}*c*a*a*b+{p-q}*a*b*b*c-{2*p+q}*b*c*c*a)^2',
                                    f'a^3*b^3*c^2']
                            if p + 2*q != 0:
                                t = p + 2*q
                                y[1] = y[1] * t * t
                                names[1] = f'(c*a*a*b+{(p-q)/t}*a*b*b*c-{(2*p+q)/t}*b*c*c*a)^2'
                            else: # p+2q = 0  but  p != q
                                t = p - q
                                y[1] = y[1] * t * t
                                names[1] = f'({(p+2*q)/t}*c*a*a*b+a*b*b*c-{(2*p+q)/t}*b*c*c*a)^2'
                
                else:
                        
                    # star
                    if coeff((5,2,1)) == 0:
                        a , b = 1 , 0 
                    else:
                        a , b = (coeff((5,1,2)) / coeff((5,2,1))).as_numer_denom()
                        
                    if sp.ntheory.primetest.is_square(a) and sp.ntheory.primetest.is_square(b):
                        t = coeff((5,2,1))
                        
                        y = []
                        names = []  

                        if t < 0:
                            return 
                        if b != 0:
                            z = sp.sqrt(a / b)
                            determinant = '-(3*m*(m+n-(-2*x^2+2*y^2+4*y*z-4*y-2*z^2+4*z-2))-(p-(x^2-2*x*y-2*x*z+y^2-2*y*z))^2-(q-(x^2+2*x*y-2*x+y^2+2*y))^2-(p-(x^2-2*x*y-2*x*z+y^2-2*y*z))*(q-(x^2+2*x*y-2*x+y^2+2*y)))'
                            # determinant = '-3*m^2-3*m*n+6*m*y^2+p^2+p*q-3*p*y^2-2*p*y+q^2-3*q*y^2+2*q*y+3*x^4+12*x^3-x^2*(6*m+3*p+3*q-10*y^2-12)-x*(-2*p*y+6*p+2*q*y+6*q-4*y^2)+3*y^4+4*y^2'
                            determinant = sp.polys.polytools.Poly(determinant).subs('z',z) 
                        else:
                            t = coeff((5,1,2))
                            determinant = '-(3*m*(m+n-(-2*x^2+2*y^2-4*y-2))-(p-(x^2-2*x*y+2*x+y^2+2*y))^2-(q-(x^2+2*x*y+y^2))^2-(p-(x^2-2*x*y+2*x+y^2+2*y))*(q-(x^2+2*x*y+y^2)))'
                            determinant = sp.polys.polytools.Poly(determinant)
                        determinant = determinant.subs((('m',coeff((4,4,0))/t), ('n',coeff((2,4,2))/t), ('p',coeff((3,4,1))/t), ('q',coeff((1,4,3))/t)))#, simultaneous=True)
                        
                        result = optimize_determinant(determinant)
                        if result is None:
                            return 
                        a , b = result
                        
                        # now we have guaranteed v <= 0
                        if coeff((5,2,1)) != 0:
                            y = [t]
                            names = [f'a*b*c*c*((b^2-a*b)-{z}*(a^2-a*b)+{a+b}*(a*c-a*b)-{a-b}*(b*c-a*b))^2']
                        else: # t = 0
                            y = [t] 
                            names = [f'a*b*c*c*((a^2-a*b)+{a+b}*(a*c-a*b)-{a-b}*(b*c-a*b))^2']


                        poly2 = poly - t * sp.polys.polytools.Poly(cycle_expansion(names[0]))
                        multipliers , y , names = _merge_sos_results(multipliers, y, names, SOS_Special(poly2, 8))
                
                    elif a > 0 and b > 0:
                        # Similarly to degree 7, we actually do not require a/b is square
                        # take coefficients like sqrt(a/b) also works -- though it is not pretty
                        # but we can make a sufficiently small perturbation such that a/b is square 
                            
                        name = 'a*b*c*(a*b+b*c+c*a)*a*(a-b)*(a-c)'
                        multipliers, y, names = _try_perturbations(poly, degree, multipliers, a, b,
                                                                coeff((5,2,1))/b, name)
            
            if result is not None:
                multipliers, y, names = result 
                if len(multipliers) == 0:
                    for i in range(len(names)):
                        names[i] = names[i].replace('a', '(1/a)').replace('b', '(1/b)').replace('c', '(1/c)')
                        names[i] = '(' + names[i] + ')*a^5*b^5*c^5'
                        name2 = sp.factor(sp.sympify(names[i]))
                        name2, denominator = sp.fraction(name2)
                        names[i] = str(name2).replace('**','^')
                        if isinstance(y[i], sp.Expr):
                            y[i] = y[i] / denominator
                        elif y[i][0] != 0: # y[i] is tuple
                            d = gcd(denominator, y[i][0])
                            y[i] = (y[i][0] // d, y[i][1] * (denominator // d))
            
                        
                        

    if (y is None) or (names is None) or len(y) == 0:
        return None 
    
    y = [x.as_numer_denom() if not isinstance(x, tuple) else x for x in y]
    return multipliers, y , names 


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