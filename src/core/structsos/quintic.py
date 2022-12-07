import sympy as sp

from ...utils.text_process import cycle_expansion
from .peeling import _merge_sos_results, FastPositiveChecker

def _sos_struct_quintic(poly, degree, coeff, recurrsion):
    multipliers, y, names = [], None, None

    if coeff((5,0,0)) == 0:
        if coeff((4,1,0)) == 0 and coeff((1,4,0)) >= 0 and coeff((3,2,0)) > 0:
            # https://tieba.baidu.com/p/6472739202
            t = coeff((1,4,0)) / coeff((3,2,0))
            if sp.ntheory.primetest.is_square(t.p) and sp.ntheory.primetest.is_square(t.q):
                t, p_, q_ = coeff((3,2,0)), sp.sqrt(t.p), sp.sqrt(t.q)
                x_ = p_ / q_
                if coeff((2,3,0)) == -t * p_ / q_ * 2:
                    v = sp.symbols('v')
                    for root in sp.polys.roots(x_ * (v**3 - 3*v*v + 7*v - 13) + 4*(v + 1)).keys():
                        if isinstance(root, sp.Rational) and root >= -1:
                            v = root
                            break
                    else:
                        v = None
                    
                    if v is not None:
                        y_ = 4*(v**2-4*v+7)*(2*v**3-3*v**2+6*v-1)/(v**3-3*v**2+7*v-13)**2
                        diff = coeff((3,1,1)) / t - (-2*x_*x_ + 2*x_ - y_)
                        if diff >= 0:
                            diff2 = coeff((2,2,1)) / t + diff - (x_*x_ + y_ - 1)
                            if diff2 >= 0:
                                if diff >= y_:
                                    # use trivial method
                                    diff -= y_
                                    y = [sp.S(1), diff / 2, diff2]
                                    y = [_ * t for _ in y]
                                    names = [f'a*b*b*(a-{x_}*b+{x_-1}*c)^2', 'a*b*c*(a-b)^2', 'a^2*b^2*c']
                                else:
                                    u = (v*v - 2*v + 5) / 4
                                    multipliers = [f'(a*a+{(v*v + 2*v + 9)/4}*b*c)']
                                    y = [4/(v**3 - 3*v*v + 7*v - 13)**2, 8*(v*v - v + 1)*(v*v + 2*v + 13)/(v**3 - 3*v*v + 7*v - 13)**2,
                                        diff, diff2]
                                    y = [_ * t for _ in y]
                                    # names = [f'a*(({v**2-2*v+9}*b-{v**2-1}*c)*(a*a-b*b+{u}*(a*b-a*c)+{v}*(b*c-a*b))'
                                    #                     + f'+({2*(v+1)}*a+{v**2-4*v+7}*b)*(b*b-c*c+{u}*(b*c-a*b)+{v}*(c*a-b*c)))^2']
                                    names = [f'a*({-2*u*v-2*u+v**2-2*v+9}*a^2*b+{2*u*v+2*u-v**3+2*v**2-7*v+2}*a*b^2+{-2*v-2}*b^3'
                                                    + f'+{v**2+2*v+1}*a^2*c+{-2*u*v**2+4*u*v-6*u+2*v**3-6*v**2+4*v}*a*b*c'
                                                    + f'+{u*v**2-4*u*v+7*u+3*v**2+2*v-1}*b^2*c+{u*v**2-u-2*v-2}*a*c^2+{-v**3-v**2+5*v-7}*b*c^2)^2']
                                    names += [f'a*b*c*(a*a-b*b+{u}*(a*b-a*c)+{v}*(b*c-a*b))^2',
                                                f'a*b*c*(a*a+{(v*v + 2*v + 9)/4}*b*c)*(a*a+b*b+c*c-a*b-b*c-c*a)',
                                                f'a*b*c*(a*a+{(v*v + 2*v + 9)/4}*b*c)*(a*b+b*c+c*a)']

        elif coeff((1,4,0)) == 0 and coeff((4,1,0)) >= 0 and coeff((2,3,0)) > 0:
            # https://tieba.baidu.com/p/6472739202
            t = coeff((4,1,0)) / coeff((2,3,0))
            if sp.ntheory.primetest.is_square(t.p) and sp.ntheory.primetest.is_square(t.q):
                t, p_, q_ = coeff((2,3,0)), sp.sqrt(t.p), sp.sqrt(t.q)
                x_ = p_ / q_
                if coeff((3,2,0)) == -t * p_ / q_ * 2:
                    v = sp.symbols('v')
                    for root in sp.polys.roots(x_ * (v**3 - 3*v*v + 7*v - 13) + 4*(v + 1)).keys():
                        if isinstance(root, sp.Rational) and root >= -1:
                            v = root
                            break
                    else:
                        v = None
                    
                    if v is not None:
                        y_ = 4*(v**2-4*v+7)*(2*v**3-3*v**2+6*v-1)/(v**3-3*v**2+7*v-13)**2
                        diff = coeff((3,1,1)) / t - (-2*x_*x_ + 2*x_ - y_)
                        if diff >= 0:
                            diff2 = coeff((2,2,1)) / t + diff - (x_*x_ + y_ - 1)
                            if diff2 >= 0:
                                if diff >= y_:
                                    # use trivial method
                                    diff -= y_
                                    y = [sp.S(1), diff / 2, diff2]
                                    y = [_ * t for _ in y]
                                    names = [f'a*c*c*(a-{x_}*c+{x_-1}*b)^2', 'a*b*c*(a-b)^2', 'a^2*b^2*c']
                                else:
                                    u = (v*v - 2*v + 5) / 4
                                    multipliers = [f'(a*a+{(v*v + 2*v + 9)/4}*b*c)']
                                    y = [4/(v**3 - 3*v*v + 7*v - 13)**2, 8*(v*v - v + 1)*(v*v + 2*v + 13)/(v**3 - 3*v*v + 7*v - 13)**2,
                                        diff, diff2]
                                    y = [_ * t for _ in y]
                                    names = [f'a*({-2*u*v-2*u+v**2-2*v+9}*a^2*c+{2*u*v+2*u-v**3+2*v**2-7*v+2}*a*c^2+{-2*v-2}*c^3'
                                                    + f'+{v**2+2*v+1}*a^2*b+{-2*u*v**2+4*u*v-6*u+2*v**3-6*v**2+4*v}*a*b*c'
                                                    + f'+{u*v**2-4*u*v+7*u+3*v**2+2*v-1}*c^2*b+{u*v**2-u-2*v-2}*a*b^2+{-v**3-v**2+5*v-7}*c*b^2)^2']
                                    names += [f'a*b*c*(a*a-c*c+{u}*(a*c-a*b)+{v}*(b*c-a*c))^2',
                                                f'a*b*c*(a*a+{(v*v + 2*v + 9)/4}*b*c)*(a*a+b*b+c*c-a*b-b*c-c*a)',
                                                f'a*b*c*(a*a+{(v*v + 2*v + 9)/4}*b*c)*(a*b+b*c+c*a)']

        if y is None:
            # try hexagon
            multipliers = ['a*b']
            poly2 = poly * sp.polys.polytools.Poly('a*b+b*c+c*a')
            multipliers , y , names = _merge_sos_results(multipliers, y, names, recurrsion(poly2, 7))
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
                    multipliers , y , names = _merge_sos_results(multipliers, y, names, recurrsion(poly2, 5))
                if y is None and b >= -a:
                    name = 'a^3*(a-b)*(a-c)'
                    poly2 = poly - a * sp.sympify(cycle_expansion(name))
                    fpc.setPoly(poly2)
                    if fpc.check() == 0:
                        y = [a]
                        names = [name]
                        multipliers , y , names = _merge_sos_results(multipliers, y, names, recurrsion(poly2, 5))
                    
    return multipliers, y, names