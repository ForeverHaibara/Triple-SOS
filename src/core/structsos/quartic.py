import sympy as sp

from ...utils.text_process import cycle_expansion
from ...utils.root_guess import rationalize
from .peeling import _merge_sos_results

def _sos_struct_quartic(poly, degree, coeff, recurrsion):
    multipliers, y, names = [], None, None

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
                multipliers , y , names = _merge_sos_results(multipliers, y, names, recurrsion(poly2, 4))

    return multipliers, y, names