import sympy as sp

from ...utils.text_process import cycle_expansion
from ...utils.root_guess import rationalize, optimize_determinant
from .peeling import _merge_sos_results, _try_perturbations, FastPositiveChecker

def _sos_struct_septic(poly, degree, coeff, recurrsion):
    multipliers, y, names = [], None, None

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
                                                            recurrsion(poly, 4), abc = True)
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
                                                        coeff((4,3,0))/b, name, recurrsion = recurrsion)

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
                    multipliers , y , names = _merge_sos_results(multipliers, y, names, recurrsion(poly2, 7))
                    if y is not None:
                        break 
            elif a > 0 and b > 0:
                name = 'a*(a-b)^2*(b-c)^2*(c-a)^2'
                multipliers, y, names = _try_perturbations(poly, degree, multipliers, a, b,
                                                        coeff((5,2,0))/b, name, recurrsion = recurrsion)

    return multipliers, y, names