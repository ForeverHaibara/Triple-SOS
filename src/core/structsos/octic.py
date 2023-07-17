import sympy as sp

from .utils import CyclicSum, CyclicProduct, _sum_y_exprs, _try_perturbations
from ...utils.text_process import cycle_expansion
from ...utils.roots.findroot import optimize_discriminant



def sos_struct_octic(poly, coeff, recurrsion):
    a, b, c = sp.symbols('a b c')
    if any((coeff((8,0,0)), coeff((7,1,0)), coeff((7,0,1)))):
        return None

    if coeff((6,2,0)) == coeff((2,6,0)) and coeff((5,3,0)) == coeff((3,5,0))\
         and coeff((5,2,1)) == coeff((2,5,1)) and coeff((4,3,1)) == coeff((3,4,1)):
        return _sos_struct_octic_symmetric_hexagon(coeff, poly, recurrsion)

    if coeff((6,2,0))==0 and coeff((6,1,1))==0 and coeff((6,0,2))==0:

        # equivalent to degree-7 hexagon when applying (a,b,c) -> (1/a,1/b,1/c)
        poly2 = coeff((0,3,5))*a**5*b**2 + coeff((1,2,5))*a**4*b**3 + coeff((2,1,5))*a**3*b**4 + coeff((3,0,5))*a**2*b**5\
                + coeff((3,3,2))*a**2*b**2*c**3+coeff((0,4,4))*a**5*b*c+coeff((1,3,4))*a**4*b**2*c+coeff((2,2,4))*a**3*b**3*c+coeff((3,1,4))*a**2*b**4*c
        poly2 = CyclicSum(poly2).doit().as_poly(a,b,c)
        solution = recurrsion(poly2)
        if False:
            None
            # if coeff((5,1,2))==0 and coeff((5,2,1))==0:
            #     # equivalent to degree-4 polynomial with respect to ab, bc, ca
            #     m = coeff((4,4,0))
            #     p = coeff((3,4,1))
            #     n = coeff((2,4,2))
            #     q = coeff((1,4,3))
            #     if m > 0:
            #         r = 3*m*(m+n)-(p*p+p*q+q*q)
            #         if r >= 0 and (p != 0 or q != 0):
            #             y = [m/2, r/(18*m*(p*p+p*q+q*q)), coeff((2,1,1))+m+n+p+q]
            #             names = [f'(a*a*b*b-b*b*c*c+{(p+2*q)/m/3}*c*a*a*b+{(p-q)/m/3}*a*b*b*c-{(2*p+q)/m/3}*b*c*c*a)^2',
            #                     f'({p+2*q}*c*a*a*b+{p-q}*a*b*b*c-{2*p+q}*b*c*c*a)^2',
            #                     f'a^3*b^3*c^2']
            #             if p + 2*q != 0:
            #                 t = p + 2*q
            #                 y[1] = y[1] * t * t
            #                 names[1] = f'(c*a*a*b+{(p-q)/t}*a*b*b*c-{(2*p+q)/t}*b*c*c*a)^2'
            #             else: # p+2q = 0  but  p != q
            #                 t = p - q
            #                 y[1] = y[1] * t * t
            #                 names[1] = f'({(p+2*q)/t}*c*a*a*b+a*b*b*c-{(2*p+q)/t}*b*c*c*a)^2'
            
            # else:
                    
            #     # star
            #     if coeff((5,2,1)) == 0:
            #         a , b = 1 , 0
            #     else:
            #         a , b = (coeff((5,1,2)) / coeff((5,2,1))).as_numer_denom()
                    
            #     if sp.ntheory.primetest.is_square(a) and sp.ntheory.primetest.is_square(b):
            #         t = coeff((5,2,1))
                    
            #         y = []
            #         names = []

            #         if t < 0:
            #             return
            #         if b != 0:
            #             z = sp.sqrt(a / b)
            #             determinant = '-(3*m*(m+n-(-2*x^2+2*y^2+4*y*z-4*y-2*z^2+4*z-2))-(p-(x^2-2*x*y-2*x*z+y^2-2*y*z))^2-(q-(x^2+2*x*y-2*x+y^2+2*y))^2-(p-(x^2-2*x*y-2*x*z+y^2-2*y*z))*(q-(x^2+2*x*y-2*x+y^2+2*y)))'
            #             # determinant = '-3*m^2-3*m*n+6*m*y^2+p^2+p*q-3*p*y^2-2*p*y+q^2-3*q*y^2+2*q*y+3*x^4+12*x^3-x^2*(6*m+3*p+3*q-10*y^2-12)-x*(-2*p*y+6*p+2*q*y+6*q-4*y^2)+3*y^4+4*y^2'
            #             determinant = sp.polys.polytools.Poly(determinant).subs('z',z)
            #         else:
            #             t = coeff((5,1,2))
            #             determinant = '-(3*m*(m+n-(-2*x^2+2*y^2-4*y-2))-(p-(x^2-2*x*y+2*x+y^2+2*y))^2-(q-(x^2+2*x*y+y^2))^2-(p-(x^2-2*x*y+2*x+y^2+2*y))*(q-(x^2+2*x*y+y^2)))'
            #             determinant = sp.polys.polytools.Poly(determinant)
            #         determinant = determinant.subs((('m',coeff((4,4,0))/t), ('n',coeff((2,4,2))/t), ('p',coeff((3,4,1))/t), ('q',coeff((1,4,3))/t)))#, simultaneous=True)
                    
            #         result = optimize_determinant(determinant)
            #         if result is None:
            #             return
            #         a , b = result
                    
            #         # now we have guaranteed v <= 0
            #         if coeff((5,2,1)) != 0:
            #             y = [t]
            #             names = [f'a*b*c*c*((b^2-a*b)-{z}*(a^2-a*b)+{a+b}*(a*c-a*b)-{a-b}*(b*c-a*b))^2']
            #         else: # t = 0
            #             y = [t]
            #             names = [f'a*b*c*c*((a^2-a*b)+{a+b}*(a*c-a*b)-{a-b}*(b*c-a*b))^2']


            #         poly2 = poly - t * sp.polys.polytools.Poly(cycle_expansion(names[0]))
            #         multipliers , y , names = _merge_sos_results(multipliers, y, names, recurrsion(poly2, 8))
            
            #     elif a > 0 and b > 0:
            #         # Similarly to degree 7, we actually do not require a/b is square
            #         # take coefficients like sqrt(a/b) also works -- though it is not pretty
            #         # but we can make a sufficiently small perturbation such that a/b is square
                        
            #         name = 'a*b*c*(a*b+b*c+c*a)*a*(a-b)*(a-c)'
            #         multipliers, y, names = _try_perturbations(poly, degree, multipliers, a, b,
            #                                                 coeff((5,2,1))/b, name, recurrsion = recurrsion)

        if solution is not None:
            # unrobust method handling fraction
            solution = sp.together(solution.xreplace({a:b*c,b:c*a,c:a*b}))

            def _try_factor(expr):
                if isinstance(expr, (sp.Add, sp.Mul, sp.Pow)):
                    return expr.func(*[_try_factor(arg) for arg in expr.args])
                elif isinstance(expr, CyclicSum):
                    # Sum(a**3*b**2*c**2*(...)**2)
                    if isinstance(expr.args[0], sp.Mul):
                        args2 = expr.args[0].args
                        symbol_degrees = {}
                        other_args = []
                        for s in args2:
                            if s in (a,b,c):
                                symbol_degrees[s] = 1
                            elif isinstance(s, sp.Pow) and s.base in (a,b,c):
                                symbol_degrees[s.base] = s.exp
                            else:
                                other_args.append(s)
                        if len(symbol_degrees) == 3:
                            degree = min(symbol_degrees.values())
                            da, db, dc = symbol_degrees[a], symbol_degrees[b], symbol_degrees[c]
                            da, db, dc = da - degree, db - degree, dc - degree
                            other_args.extend([a**da, b**db, c**dc])
                            return CyclicSum(sp.Mul(*other_args)) * CyclicProduct(a) ** degree
                elif isinstance(expr, CyclicProduct):
                    # Product(a**2) = Product(a) ** 2
                    if isinstance(expr.args[0], sp.Pow) and expr.args[0].base in (a,b,c):
                        return CyclicProduct(expr.args[0].base) ** expr.args[0].exp
                return expr
            
            solution = sp.together(_try_factor(solution)) / CyclicProduct(a) ** 2
            return solution


    return None


def _sos_struct_octic_symmetric_hexagon(coeff, poly, recurrsion):
    """
    Try to solve symmetric octic hexagon, without terms a^8, a^7b and a^7c.

    For octics and structural method, the core is not to handle very complicated cases.
    Instead, we explore the art of sum of squares by using simple tricks.
    """
    a, b, c = sp.symbols('a b c')
    if True:
        # Case 1. use 
        # s(a(b-c)2)2s(xa2+yab)+p(a-b)2s(za2+wab)
        c1, c2, c3, c4 = [coeff(_) for _ in ((6,2,0),(5,3,0),(6,1,1),(5,2,1))]
        x_ = c1/2 + c3/4
        y_ = c1 + c2/4 + c3/2 + c4/4
        z_ = c1/2 - c3/4
        w_ = -c1 + 3*c2/4 - 3*c3/2 - c4/4
        if x_ >= 0 and z_ >= 0 and x_ + y_ >= 0 and z_ + w_ >= 0:
            m_ = coeff((4,4,0)) - (-2*w_ + 2*x_ + 2*y_ + 2*z_)
            p_ = coeff((4,3,1)) - (w_ - 8*x_ - 7*y_)
            n_ = coeff((4,2,2)) - (2*w_ + 44*x_ - 18*y_ - 4*z_)
            r_ = coeff((3,3,2)) - (-2*w_ - 18*x_ + 22*y_ + 2*z_)
            if m_ >= 0:
                y = [
                    sp.S(1),
                    sp.S(1),
                    m_ / 2,
                    (n_ - ((p_ / m_)**2 - 1) * m_) / 2,
                    m_ + 2*p_ + n_ + r_
                ]
                if all(_ >= 0 for _ in y):
                    exprs = [
                        CyclicSum(a*(b-c)**2)**2 * CyclicSum(x_*a**2 + y_*b*c),
                        CyclicProduct((a-b)**2) * CyclicSum(z_*a**2 + w_*b*c),
                        CyclicSum(a**2*(b-c)**2*(a*b+a*c + (p_ / m_) *b*c)**2),
                        CyclicProduct(a**2) * CyclicSum((a-b)**2),
                        CyclicSum(a**3*b**3*c**2),
                    ]
                    return _sum_y_exprs(y, exprs)
                if p_ >= 0:
                    y = [
                        sp.S(1),
                        sp.S(1),
                        m_ / 2,
                        p_ + m_,
                        (n_ + 2*(p_ + m_)) / 2,
                        m_ + 2*p_ + n_ + r_
                    ]
                    if all(_ >= 0 for _ in y):
                        exprs = [
                            CyclicSum(a*(b-c)**2)**2 * CyclicSum(x_*a**2 + y_*b*c),
                            CyclicProduct((a-b)**2) * CyclicSum(z_*a**2 + w_*b*c),
                            CyclicSum(a**2*(b-c)**2*(a*b+a*c-b*c)**2),
                            CyclicProduct(a) * CyclicSum(a**3*(b-c)**2),
                            CyclicProduct(a**2) * CyclicSum((a-b)**2),
                            CyclicSum(a**3*b**3*c**2),
                        ]
                        return _sum_y_exprs(y, exprs)