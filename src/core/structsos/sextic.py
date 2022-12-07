import sympy as sp
from sympy.solvers.diophantine.diophantine import diop_DN

from ...utils.text_process import cycle_expansion
from ...utils.root_guess import rationalize
from .peeling import _merge_sos_results, FastPositiveChecker

def _sos_struct_sextic(poly, degree, coeff, recurrsion):
    multipliers, y, names = [], None, None

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
                    multipliers , y , names = _merge_sos_results(multipliers, y, names, recurrsion(poly2, 6))

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
                        multipliers , y , names = _merge_sos_results(multipliers, y, names, recurrsion(poly2, 6))

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
                        multipliers , y , names = _merge_sos_results(multipliers, y, names, recurrsion(poly2, 6))
                        if y is not None:
                            break


        else:# coeff((4,2,0)) == coeff((4,0,2)):
            if coeff((4,2,0)) == 0:
                # hexagram (star)
                poly = poly * sp.polys.polytools.Poly('a+b+c')
                multipliers , y , names = _merge_sos_results(['a'], y, names, recurrsion(poly, 7))
            else:
                y = [coeff((4,2,0)) / 3] 
                names = [f'(a-b)^2*(b-c)^2*(c-a)^2']

                poly2 = poly - y[0] * 3 * sp.sympify(names[0])
                v = 0 # we do not need to check the positivity, just try
                if v != 0:
                    y, names = None, None 
                else: # positive
                    multipliers , y , names = _merge_sos_results(multipliers, y, names, recurrsion(poly2, 6))
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

    return multipliers, y, names