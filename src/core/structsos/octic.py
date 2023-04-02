from math import gcd

import sympy as sp

from ...utils.text_process import cycle_expansion
from ...utils.root_guess import rationalize, optimize_determinant
from .peeling import _merge_sos_results, _try_perturbations, FastPositiveChecker

def _sos_struct_octic(poly, degree, coeff, recurrsion):
    multipliers, y, names = [], None, None

    if coeff((8,0,0))==0 and coeff((7,1,0))==0 and coeff((7,0,1))==0 and coeff((6,2,0))==0 and\
        coeff((6,1,1))==0 and coeff((6,0,2))==0:

        # equivalent to degree-7 hexagon when applying (a,b,c) -> (1/a,1/b,1/c)
        poly2 = f'{coeff((0,3,5))}*a^5*b^2+{coeff((1,2,5))}*a^4*b^3+{coeff((2,1,5))}*a^3*b^4+{coeff((3,0,5))}*a^2*b^5+{coeff((3,3,2))}*a^2*b^2*c^3'
        poly2 += f'+{coeff((0,4,4))}*a^5*b*c+{coeff((1,3,4))}*a^4*b^2*c+{coeff((2,2,4))}*a^3*b^3*c+{coeff((3,1,4))}*a^2*b^4*c'
        poly2 = sp.polys.polytools.Poly(cycle_expansion(poly2))
        result = recurrsion(poly2, 7)
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
                    multipliers , y , names = _merge_sos_results(multipliers, y, names, recurrsion(poly2, 8))
            
                elif a > 0 and b > 0:
                    # Similarly to degree 7, we actually do not require a/b is square
                    # take coefficients like sqrt(a/b) also works -- though it is not pretty
                    # but we can make a sufficiently small perturbation such that a/b is square
                        
                    name = 'a*b*c*(a*b+b*c+c*a)*a*(a-b)*(a-c)'
                    multipliers, y, names = _try_perturbations(poly, degree, multipliers, a, b,
                                                            coeff((5,2,1))/b, name, recurrsion = recurrsion)
        
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
        
    return multipliers, y, names