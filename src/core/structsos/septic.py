import sympy as sp

from .utils import CyclicSum, CyclicProduct, _sum_y_exprs, _try_perturbations
from ...utils.roots.rationalize import rationalize
from ...utils.roots.findroot import optimize_discriminant


a, b, c = sp.symbols('a b c')

def sos_struct_septic(poly, coeff, recurrsion):
    if coeff((7,0,0))==0 and coeff((6,1,0))==0 and coeff((6,0,1))==0:
        if coeff((5,2,0))==0 and coeff((5,0,2))==0:
            # star
            return _sos_struct_septic_star(coeff, poly, recurrsion)
        else:
            # hexagon
            return _sos_struct_septic_hexagon(coeff, poly, recurrsion)
    return None



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
    if any(coeff(i) for i in ((7,0,0), (6,1,0), (6,0,1), (5,2,0), (5,0,2))):
        return None

    if coeff((4,3,0)) == 0:
        p, q = 1 , 0
    else:
        p, q = (coeff((3,4,0)) / coeff((4,3,0))).as_numer_denom()
        
    if sp.ntheory.primetest.is_square(p) and sp.ntheory.primetest.is_square(q):
        t = coeff((4,3,0))

        if q == 0 and coeff((3,4,0)) == 0: # abc | poly
            pass
        elif t < 0:
            return None
        else:
            if q != 0:
                z = sp.sqrt(p / q)
                discriminant = '-(3*m*(m+n-(-2*x^2+2*y^2+4*y*z-4*y-2*z^2+4*z-2))-(p-(x^2-2*x*y+2*x+y^2+2*y))^2-(q-(x^2+2*x*y+2*x*z+y^2-2*y*z))^2-(p-(x^2-2*x*y+2*x+y^2+2*y))*(q-(x^2+2*x*y+2*x*z+y^2-2*y*z)))'
                # discriminant = '-3*m^2-3*m*n+6*m*y^2+p^2+p*q-3*p*y^2-2*p*y+q^2-3*q*y^2+2*q*y+3*x^4+12*x^3-x^2*(6*m+3*p+3*q-10*y^2-12)-x*(-2*p*y+6*p+2*q*y+6*q-4*y^2)+3*y^4+4*y^2'
                discriminant = sp.polys.Poly(discriminant).subs('z',z)
            else:
                t = coeff((3,4,0))
                discriminant = '-(3*m*(m+n-(-2*x^2+2*y^2-4*y-2))-(p-(x^2-2*x*y+y^2))^2-(q-(x^2+2*x*y-2*x+y^2+2*y))^2-(p-(x^2-2*x*y+y^2))*(q-(x^2+2*x*y-2*x+y^2+2*y)))'
                discriminant = sp.polys.Poly(discriminant)
            discriminant = discriminant.subs((('m',coeff((5,1,1))/t), ('n',coeff((3,3,1))/t), ('p',coeff((4,2,1))/t), ('q',coeff((2,4,1))/t)))#, simultaneous=True)
            
            result = optimize_discriminant(discriminant)
            # print(result, print(sp.latex(discriminant)), 'here')
            if result is None:
                return None

            x, y = sp.symbols('x y')
            u, v = result[x], result[y]
            
            # now we have guaranteed discriminant <= 0
            if coeff((4,3,0)) != 0:
                solution = t * CyclicSum(b*(a**2*b+(z-1-2*v)*a*b*c-z*b*c**2+(u+v)*a**2*c+(v-u)*(a*c**2))**2)
            else: # 
                solution = t * CyclicSum(b*(b*c**2+(-2*v-1)*a*b*c+(u+v)*a**2*c+(v-u)*a*c**2)**2)


            poly = poly - solution.doit().as_poly(a,b,c)
            # print(result, poly, discriminant.subs(result),'\n',discriminant)
            poly = sp.cancel(poly / (a*b*c).as_poly(a,b,c)).as_poly(a,b,c)

            new_solution = recurrsion(poly)
            if new_solution is not None:
                return solution + new_solution * CyclicProduct(a)
            
            return None


    elif p > 0 and q > 0:
        # we shall NOTE that we actually do not require a/b is square
        # take coefficients like sqrt(a/b) also works -- though it is not pretty
        # but we can make a sufficiently small perturbation such that a/b is square
        # e.g. a/b = 7/4 = z^2 + epsilon
        # solve it by Newton's algorithm for squareroot, starting with z = floor(7/4) = 1
        # 1 -> (1 + 7/4)/2 = 11/8 -> (11/8+14/11)/2 = 233/176

        perturbation = CyclicSum(c*(a-b)**2*(a*b-a*c-b*c)**2)
        
        solution = _try_perturbations(poly, coeff((3,4,0)), coeff((4,3,0)), perturbation, recurrsion = recurrsion)
        if solution is not None:
            return solution

    return None


def _sos_struct_septic_biased(coeff):
    """
    Solve septic hexagons without s(a5b2) and s(a4b3)


    Examples
    -------
    (s(a(a2c-ab2)(a2c-ab2-3abc))+5s(a3b3c-a3b2c2)-2s(a2b4c-a3b2c2))

    s(a(a2c-ab2)(a2c-ab2-3abc))+s((a2-b2+2(ab-ac)+5(bc-ab))2)abc

    s(a5c2-3a4bc2+a4c3+3a3b3c-2a3b2c2)+15s(a3b3c-a3b2c2)-7s(a2b4c-a3b2c2)

    s(a5c2-16a4bc2+a4c3+54a3b3c-40a3b2c2)

    s(4a5c2-6a4b2c-12a4bc2+8a4c3-11a3b3c+17a3b2c2)

    s(a5c2+a4b2c+a4bc2-7a3b3c+4a3b2c2)
    """

    if coeff((5,2,0)) or coeff((4,3,0)):
        # reflect the polynomial so that coeff((5,2,0)) == 0
        def new_coeff(c):
            return coeff((c[0], c[2], c[1]))
        solution = _sos_struct_septic_biased(new_coeff)
        if solution is not None:
            solution = solution.xreplace({b: c, c: b})
        return solution

    if coeff((5,2,0)) or coeff((4,3,0)):
        return None

    if coeff((3,4,0)) == coeff((2,5,0)) and coeff((3,4,0)) != 0:
        coeff34 = coeff((3,4,0))
        m, p, n, q = coeff((5,1,1)) / coeff34, coeff((4,2,1)) / coeff34 + 2, coeff((3,3,1)) / coeff34, coeff((2,4,1)) / coeff34
        if m > 0:
            det = m*(-3*m**2 - 3*m*n - 7*m - 4*n + p**2 + p*q - p + q**2 - 2*q - 3) + p*p
            if det <= 0:
                x = -(6*m + p + 2*q + 6)/(3*m + 4)
                r = x.as_numer_denom()[1] # cancel the denominator is good

                n2 = n - (x**2 + 4*x + 3)
                q2 = q + (2*x + 3)
                u_, v_ = -(p + 2*q2) / 3 / m, -(2*p + q2) / 3 / m

                multiplier = CyclicSum(a)
                y = [
                    1 / r**2,
                    1 / r**2,
                    m / 2,
                    (3*(m + n2) - (p**2 + q2**2 + p*q2) / m) / 6,
                    m + p + n + q + 2 + coeff((3,2,2)) / coeff34
                ]
                y = [_ * coeff34 for _ in y]
                exprs = [
                    CyclicSum(a*c*(r*a**2*c-r*b*c**2+r*a*b*c-(x+2)*r*a*b**2+(x+1)*r*b**2*c)**2),
                    CyclicSum((r*a**3*c-r*a*b**3+r*b**2*c**2-r*a**2*b**2-r*a*b*c**2-(x+1)*r*a**2*b*c+(x+2)*r*a*b**2*c)**2),
                    CyclicSum((a**2-b**2+(u_-v_)*a*b-u_*a*c+v_*b*c)**2) * CyclicSum(a) * CyclicProduct(a),
                    CyclicSum(a**2*(b-c)**2) * CyclicSum(a) * CyclicProduct(a),
                    CyclicSum(a)**2 * CyclicProduct(a**2)
                ]
                return _sum_y_exprs(y, exprs) / multiplier

        elif m == 0 and p >= -2:
            if p == -2:
                x = (-q - 3) / 2
            0

    return None


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

    (1/5(18s(a3(13b2+5c2)(13c2+5a2))-p(13a2+5b2)s(a))-585/64s(a(a2c-b2c-8/3(a2b-abc)+7/4(ab2-abc))2))

    s(a5b2-a5bc+a5c2-a4b3-2a4b2c-2a4bc2-a4c3+10a3b3c-5a3b2c2)
    """
    
    if any(coeff(i) for i in ((7,0,0), (6,1,0), (6,0,1))):
        return None

    if (coeff((5,2,0)) == 0 and coeff((4,3,0)) == 0) or (coeff((3,4,0)) == 0 and coeff((2,5,0)) == 0):
        solution = _sos_struct_septic_biased(coeff)
        if solution is not None:
            return solution

    if coeff((5,2,0)) == 0:
        p, q = 1 , 0
    else:
        p, q = (coeff((2,5,0)) / coeff((5,2,0))).as_numer_denom()
    
    if sp.ntheory.primetest.is_square(p) and sp.ntheory.primetest.is_square(q):
        t = coeff((5,2,0)) / q if q != 0 else coeff((2,5,0))

        if t < 0:
            return None

        # 's((a(u(a2b-abc)-v(a2c-abc)+x(bc2-abc)+y(b2c-abc)+z(ac2-abc)+w(ab2-abc))2))' (u>0,v>0)
        # z^2 + 2uw = coeff((4,3,0)) = m,   w^2 - 2vz = coeff((3,4,0)) = n
        # => (w^2-n)^2 = 4v^2(m-2uw)
        # More generally, we can find w,z such that z^2+2uw < m and w^2-2vz < n
        # such that the remaining coeffs are slightly positive to form a hexagram (star)
        m = coeff((4,3,0)) / t
        n = coeff((3,4,0)) / t

        u = sp.sqrt(q)
        v = sp.sqrt(p)
        candidates = []
        
        if v == 0: # first reflect the problem, then reflect it back
            u , v = v , u
            p , q = q , p
            m , n = n , m

        x = sp.symbols('x')
        eq = ((x**2 - n)**2 - 4*p*(m - 2*u*x)).as_poly(x)
        for w in sp.polys.polyroots.roots(eq):
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
                    w = rationalize(w2, rounding = rounding, reliable = False)
                    z = rationalize(z2, rounding = rounding, reliable = False)
                    rounding *= 0.1
                    if z*z + 2*u*w <= m and w*w - 2*v*z <= n:
                        candidates.append((w, z))

        for perturbation in (100, 3, 2):
            m2 = m - abs(m / perturbation)
            n2 = n - abs(n / perturbation)
            eq = ((x**2 - n2)**2 - 4*p*(m2 - 2*u*x)).as_poly(x)
            for w in sp.polys.roots(eq):
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
                        w = rationalize(w2, rounding = rounding, reliable = False)
                        z = rationalize(z2, rounding = rounding, reliable = False)
                        rounding *= 0.1
                        if z*z + 2*u*w <= m and w*w - 2*v*z <= n:
                            candidates.append((w, z))

        candidates = list(set(candidates))
        if coeff((2,5,0)) == 0: # reflect back
            u , v = v , u
            m , n = n , m
            p , q = q , p
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
            discriminant = -sp.polys.Poly(discriminant)
            discriminant = discriminant.subs((('u', u), ('v', v), ('z', z), ('w', w),
                                ('m', coeff((5,1,1))/t), ('n', coeff((3,3,1))/t), ('p', coeff((4,2,1))/t), ('q', coeff((2,4,1))/t)))
            
            result = optimize_discriminant(discriminant, soft = True)
            if result is None:
                continue

            x, y = sp.symbols('x y')
            r, s = result[x], result[y]
            # print('w z =', w, z, 'a b =', a, b, discriminant)

            expr = (u*(a**2*b-a*b*c)-v*(a**2*c-a*b*c)+r*(b*c**2-a*b*c)+s*(b**2*c-a*b*c)+z*(a*c**2-a*b*c)+w*(a*b**2-a*b*c)).expand()
            solution = t * CyclicSum(a * expr**2)
            poly2 = poly - solution.doit().as_poly(a,b,c)

            new_solution = recurrsion(poly2)
            if new_solution is not None:
                return solution + new_solution
            return None

    elif p > 0 and q > 0:
        perturbation = CyclicSum(a) * CyclicProduct((a-b)**2)
        solution = _try_perturbations(poly, coeff((2,5,0)), coeff((5,2,0)), perturbation, recurrsion=recurrsion)

        if solution is not None:
            return solution

    return None