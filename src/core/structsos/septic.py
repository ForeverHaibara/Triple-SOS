import sympy as sp

from ...utils.text_process import cycle_expansion
from ...utils.root_guess import rationalize, optimize_determinant
from .peeling import _merge_sos_results, _try_perturbations, FastPositiveChecker

def _sos_struct_septic(poly, degree, coeff, recurrsion):
    multipliers, y, names = [], None, None

    if coeff((7,0,0))==0 and coeff((6,1,0))==0 and coeff((6,0,1))==0:
        if coeff((5,2,0))==0 and coeff((5,0,2))==0:
            # star
            multipliers, y, names = _sos_struct_septic_star(coeff, poly, recurrsion)
        else:
            # hexagon
            multipliers, y, names = _sos_struct_septic_hexagon(coeff, poly, recurrsion)
            
    return multipliers, y, names


def _sos_struct_septic_star(coeff, poly, recurrsion):
    """
    Solve s(ua4b3 + va3b4) + abcf(a,b,c) >= 0 where f is degree 4.

    Idea: Subtract something and apply the quadratic theorem, which is a 
    discriminant minimization theorem.

    Examples
    -------
    s(ab(a-c)2(b-c)2)s(a)

    s((a-b)2(a-c)2(b+c))s(ab)-s(a)p(a-b)2 

    s(a3b3+7a4bc-29a3b2c+12a3bc2+9a2b2c2)s(a)

    s(16a5bc+3a4b3-77a4b2c+3a4bc2+3a4c3+72a3b3c-20a3b2c2)

    s(20a5bc+4a4b3-31a4b2c-4a4bc2+4a4c3-46a3b3c+53a3b2c2)

    (3s(6a4c-31a3bc+6a3c2+19a2b2c)s(a2)-18s(c(a3-abc+(-8/3)(a2b-abc)+(1/3)(ab2-abc)- (bc2-abc))2))

    s(16a5bc+4a4b3-80a4b2c+3a4bc2+7a4c3+64a3b3c-14a3b2c2)
    
    s(2a4b3+9a3b4+abc(18a4-66a3b+10a2b2+11ab3+16a2bc))

    s(72a5bc+24a4b3+156a4b2c-453a4bc2+44a4c3+176a3b3c-19a3b2c2)
    """
    multipliers, y, names = [], None, None
    if any(coeff(i) for i in ((7,0,0), (6,1,0), (6,0,1), (5,2,0), (5,0,2))):
        return multipliers, y, names

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
            return [], None, None
        else:
            if b != 0:
                z = sp.sqrt(a / b)
                discriminant = '-(3*m*(m+n-(-2*x^2+2*y^2+4*y*z-4*y-2*z^2+4*z-2))-(p-(x^2-2*x*y+2*x+y^2+2*y))^2-(q-(x^2+2*x*y+2*x*z+y^2-2*y*z))^2-(p-(x^2-2*x*y+2*x+y^2+2*y))*(q-(x^2+2*x*y+2*x*z+y^2-2*y*z)))'
                # discriminant = '-3*m^2-3*m*n+6*m*y^2+p^2+p*q-3*p*y^2-2*p*y+q^2-3*q*y^2+2*q*y+3*x^4+12*x^3-x^2*(6*m+3*p+3*q-10*y^2-12)-x*(-2*p*y+6*p+2*q*y+6*q-4*y^2)+3*y^4+4*y^2'
                discriminant = sp.polys.polytools.Poly(discriminant).subs('z',z) 
            else:
                t = coeff((3,4,0))
                discriminant = '-(3*m*(m+n-(-2*x^2+2*y^2-4*y-2))-(p-(x^2-2*x*y+y^2))^2-(q-(x^2+2*x*y-2*x+y^2+2*y))^2-(p-(x^2-2*x*y+y^2))*(q-(x^2+2*x*y-2*x+y^2+2*y)))'
                discriminant = sp.polys.polytools.Poly(discriminant)
            discriminant = discriminant.subs((('m',coeff((5,1,1))/t), ('n',coeff((3,3,1))/t), ('p',coeff((4,2,1))/t), ('q',coeff((2,4,1))/t)))#, simultaneous=True)
            
            result = optimize_determinant(discriminant)
            if result is None:
                return [], None, None
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
        
        multipliers, y, names = _try_perturbations(poly, 7, multipliers, a, b,
                                                coeff((4,3,0))/b, name, recurrsion = recurrsion)

    return multipliers, y, names


def _sos_struct_septic_hexagon(coeff, poly, recurrsion):
    """
    Solve septic without s(a7), s(a6b), s(a6c).

    Idea: subtract something to form a star.

    Examples
    -------
    s(2a5b2-5a5bc+8a5c2-5a4b3+21a4b2c-21a4bc2+a4c3-7a3b3c+6a3b2c2)

    s(20a5bc+4a5c2+4a4b3-23a4b2c-15a4bc2+8a4c3-78a3b3c+80a3b2c2)

    (s(a2(a-b)(a2+b2-3ac+3c2))s(a2+ab)-s(a(a3-a2c+0(a2b-abc)-(ac2-abc)+5/4(bc2-abc))2)) 

    (s(a2(a-b)(a2+ab-5bc))s(a)-s(a(a-b)(a-c))2-3s(ac(a2-bc-(c2-bc)-3/2(ab-bc))2))s(a)

    s(4a5b2-2a5bc+4a5c2+8a4b3-8a4b2c+a4bc2-10a4c3+2a3b3c+a3b2c2)
    """
    
    if any(coeff(i) for i in ((7,0,0), (6,1,0), (6,0,1))):
        return [], None, None

    multipliers, y, names = [], None, None

    if coeff((5,2,0)) == 0:
        a , b = 1 , 0 
    else:
        a , b = (coeff((2,5,0)) / coeff((5,2,0))).as_numer_denom()   
    
    if sp.ntheory.primetest.is_square(a) and sp.ntheory.primetest.is_square(b):
        t = coeff((5,2,0)) / b if b != 0 else coeff((2,5,0))
        
        y = []
        names = []

        if t < 0:
            return [], None, None

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
                # print(w, v, z, n)
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
            discriminant = '3*(m-(-2*u*v))*(m-(-2*u*v)+n-(-2*u*w+2*u*y-2*u*z+2*v*w-2*v*x+2*v*z-2*w^2-2*w*x-2*w*y-4*w*z+2*x*y-2*x*z-2*y*z-2*z^2))'
            discriminant += '-(p-(-2*u^2+2*u*v-2*u*w-2*u*x-2*u*y-2*u*z-2*v*w+2*x*z+y^2))^2-(q-(2*u*v+2*u*z-2*v^2+2*v*w+2*v*x+2*v*y+2*v*z+2*w*y+x^2))^2'
            discriminant += '-(p-(-2*u^2+2*u*v-2*u*w-2*u*x-2*u*y-2*u*z-2*v*w+2*x*z+y^2))*(q-(2*u*v+2*u*z-2*v^2+2*v*w+2*v*x+2*v*y+2*v*z+2*w*y+x^2))'
            discriminant = -sp.polys.polytools.Poly(discriminant)
            discriminant = discriminant.subs((('u', u), ('v', v), ('z', z), ('w', w), 
                                ('m', coeff((5,1,1))/t), ('n', coeff((3,3,1))/t), ('p', coeff((4,2,1))/t), ('q', coeff((2,4,1))/t)))
            
            result = optimize_determinant(discriminant, soft = True)
            if result is None:
                continue
            a , b = result
            # print('w z =', w, z, 'a b =', a, b, discriminant)
            
            y = [t] 
            names = [f'a*({u}*(a^2*b-a*b*c)-{v}*(a^2*c-a*b*c)+{a}*(b*c^2-a*b*c)+{b}*(b^2*c-a*b*c)+{z}*(a*c^2-a*b*c)+{w}*(a*b^2-a*b*c))^2']
            poly2 = poly - t * sp.polys.polytools.Poly(cycle_expansion(names[0]))
            multipliers , y , names = _merge_sos_results(multipliers, y, names, recurrsion(poly2, 7))
            if y is not None:
                break 

    elif a > 0 and b > 0:
        name = 'a*(a-b)^2*(b-c)^2*(c-a)^2'
        multipliers, y, names = _try_perturbations(poly, 7, multipliers, a, b,
                                                coeff((5,2,0))/b, name, recurrsion = recurrsion)

    return multipliers, y, names