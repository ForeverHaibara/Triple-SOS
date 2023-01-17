from math import gcd
from itertools import product

import sympy as sp
import numpy as np

from .text_process import PreprocessText_GetDomain, deg


def rationalize(v, rounding = 1e-2, mod = None, reliable = False) -> tuple:
    '''
    Approximates a floating number to a reasonable fraction.

    Params
    -------
    v: float
        input
    
    rounding: float
        Maximize rounding error allowed.

    mod: unsigned int / list / ...
        Possible denominators that put into tries. Set to None to trigger default settings.

    Return
    -------
    (a,b): tuple
        if b > 0, v approximates a/b (rationalization succeeds)

        else return (a,b) = (v,-1)
    '''
    if v == 0:
        return 0 , 1
    else:
        if True: #reliable:
            # https://tieba.baidu.com/p/7846250213 
            x = sp.Rational(v)
            t = sp.floor(x)
            x = x - t
            fracs = [t]
            i = 0
            j = -1
            while i <= 31:
                x = 1 / x 
                t = sp.floor(x)
                if (t == 0 or t == sp.nan or t == sp.zoo):
                    # truncate at the largest element
                    if reliable:
                        if len(fracs) > 1:
                            j = max(range(1, len(fracs)), key = lambda u: fracs[u]) 
                        else: 
                            j = 1
                    break
                fracs.append(t)
                x = x - t
                i += 1
            # print(fracs)
            if j < 0:
                j = len(fracs)

            if reliable:
                x = 0
                # truncate the fraction list at j
                for t in fracs[:j][::-1]:
                    x += t
                    x = 1 / x 

                x = 1 / x
                if abs(v - x) < 1e-6: # close approximation
                    return x.p , x.q 
                
                # by experiment, |v-x| >> eps only happens when x.q = 2^k 
                # where we should use the full fraction list
                x = 0
                for t in fracs[::-1]:
                    x += t
                    x = 1 / x 

                x = 1 / x
                # if abs(v - x) < 1e-6: # close approximation
                return x.p , x.q
            else: # not reliable
                # if not reliable, we accept the result only when p,q is not too large
                # theorem: |x - p/q| < 1/(2q²) only if p/q is continued fraction of x
                for length in range(1, len(fracs)):
                    x = 0
                    for t in fracs[:length][::-1]:
                        x += t
                        x = 1 / x 

                    x = 1 / x
                    if abs(v - x) < rounding: # close approximation
                        if length <= 1 or abs(v - x) < rounding ** 2: # very nice
                            return x.p , x.q 
                        # cancel this move and use shorter truncation
                        x = 0
                        for t in fracs[:length-1][::-1]:
                            x += t
                            x = 1 / x 

                        x = 1 / x
                        return x.p , x.q 



    ####################################################
    # backup plan, has been deprecated, do not use it     
    ####################################################
    if round(v) != 0 and abs(v - round(v)) < rounding: # close to an integer
        return round(v) , 1
    
    if mod is None:
        mod = (1081080,1212971760,1327623480,904047048,253627416,373513896,
                438747624,383320080,1920996000)
    if type(mod) == int:
        mod = (mod,)
        
    for m in mod:
        val = v * m 
        # if the resulting val is close to an integer, then treat it as integer
        if abs(val - round(val)) < rounding:
            val = round(val)
            _gcd = gcd(val, m)
            return val//_gcd , m//_gcd
    
    # fails to approximate in the given rounding error tolerance
    return v , -1
    

def rationalize_array(x, tol = 1e-7, reliable = True):
    '''
    Approximates each NONNEGATIVE floating number to a reasonable fraction and
    leave the floating number unchanged if failed.

    Params
    ------
    x: arraylike

    tol: values smaller than tolerance get set to zero
    '''
    y = []
    for v in x:
        if isinstance(v, float) or isinstance(v, sp.Float):
            if abs(v) < tol: 
                y.append((0, 1))
            else:            
                y.append(rationalize(v, reliable = reliable))
        elif isinstance(v, tuple):
            y.append(v)
        elif isinstance(v, sp.Expr):
            y.append(v.as_numer_denom())
    return y


def rationalize_bound(v, direction = 1, roundings = None, compulsory = True):
    """
    Yield rational approximation of v.

    Parameters
    -------
    direction: 1 or -1
        If direction = 1, find something > v
        If direction = -1, find something < v
    """
    if isinstance(v, sp.Rational):
        yield v
        return
    if roundings is None:
        roundings = (.5, .2, .1, .05, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8)

    previous_v = None
    for rounding in roundings:
        v_ = sp.Rational(*rationalize(v + direction * rounding * 3, rounding=rounding, reliable=False))
        if v_ != previous_v:
            previous_v = v_
            yield v_
    
    if not compulsory:
        return
    
    for rounding in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
        v_ = sp.nsimplify(v + direction * rounding * 10, rational = True, tolerance = rounding)
        if v_ != previous_v:
            previous_v = v_
            yield v_
    

def square_perturbation(a, b, times = 4):
    """
    Find t such that (a-t)/(b-t) is square, please be sure a/b is not a square
    """
    if a > b:
        z = max(1, int((a / b)**0.5))
    else:
        z = max(1, int((b / a)**0.5))
    z = sp.Rational(z)  # convert to rational

    for i in range(times): # Newton has quadratic convergence, we only try a few times
        # (a-t)/(b-t) = z^2  =>  t = (a - z^2 b) / (1 - z^2) 
        if i > 0 or z == 1:
            # easy to see z > sqrt(a/b) (or z > sqrt(b/a))
            z = (z + a/b/z)/2 if a > b else (z + b/a/z)/2
        if a > b:
            t = (a - z*z*b) / (1 - z*z)
            if t < 0 or b < t:
                continue 
        else:
            t = (b - z*z*a) / (1 - z*z)
            if t < 0 or a < t: 
                continue 
        yield t 


def verify(y, polys, poly, tol: float = 1e-10) -> bool:
    '''
    Verify whether the fraction approximation is valid
    by substracting the partial sums and checking whether the remainder is zero.
    '''
    try:
        for coeff, f in zip(y, polys):
            if coeff[0] != 0:
                if coeff[1] != -1:
                    if not isinstance(coeff[0], sp.Expr):
                        poly = poly - sp.Rational(coeff[0] , coeff[1]) * f
                    else: 
                        v = coeff[0] / coeff[1]
                        coeff_dom = PreprocessText_GetDomain(str(v))
                        if coeff_dom != sp.QQ:
                            v = sp.polys.polytools.Poly(str(v)+'+a', domain=coeff_dom)\
                                 - sp.polys.polytools.Poly('a',domain=coeff_dom)
                        poly = poly - v * f 
                else:
                    poly = poly - coeff[0] * f

        for coeff in poly.coeffs():
            # some coefficient is larger than tolerance, approximation failed
            if abs(coeff) > tol:
                return False    
        return True
    except:
        return False 


def verify_isstrict(func, root, tol=1e-9):
    '''
    Verify whether a root is strict.

    Warning: Better get the function regularized beforehand.
    '''
    return abs(func(root)) < tol


def findbest(choices, func, init_choice=None, init_val=2147483647):
    '''
    Find the best choice of all choices to minimize the function.
    
    Return the best choice and the corresponding value.
    '''

    best_choice = init_choice
    val = init_val
    for choice in choices:
        val2 = func(choice)
        if val2 < val:
            val = val2 
            best_choice = choice
    return best_choice , val


def optimize_determinant(determinant, soft = False):
    best_choice = (2147483647, 0, 0)
    for a, b in product(range(-5, 7, 2), repeat = 2): # integer
        v = determinant(a, b)
        if v <= 0:
            best_choice = (v, a, b)
            break  
        elif v < best_choice[0]:
            best_choice = (v, a, b)

    v , a , b = best_choice
    if v > 0:
        for a, b in product(range(a-1, a+2), range(b-1, b+2)): # search a neighborhood
            v = determinant(a, b)
            if v <= 0:
                best_choice = (v, a, b)
                break  
            elif v < best_choice[0]:
                best_choice = (v, a, b)

    if v > 0:
        a = a * 1.0
        b = b * 1.0
        da = determinant.diff('x')
        db = determinant.diff('y')
        da2 = da.diff('x')
        dab = da.diff('y')
        db2 = db.diff('y')
        # x =[a',b'] <- x - inv(nabla)^-1 @ grad 
        for i in range(20):
            lasta , lastb = a , b 
            da_  = da(a,b)
            db_  = db(a,b)
            da2_ = da2(a,b)
            dab_ = dab(a,b)
            db2_ = db2(a,b)
            det_ = da2_ * db2_ - dab_ * dab_ 
            if det_ == 0:
                break 
            else:
                a , b = a - (db2_ * da_ - dab_ * db_) / det_ , b - (-dab_ * da_ + da2_ * db_) / det_
                if abs(lasta - a) < 1e-9 and abs(lastb - b) < 1e-9:
                    break 
        v = determinant(a, b)
        
    if v > 1e-6 and not soft:
        return None

    # iterative deepening
    a_ , b_ = (a, 1), (b, 1)
    rounding = 0.5
    for i in range(5):
        a_ = sp.Rational(*rationalize(a, rounding, reliable = False))
        b_ = sp.Rational(*rationalize(b, rounding, reliable = False))
        v = determinant(a_, b_)
        if v <= 0:
            break 
        rounding *= .1
    else:
        return (a_, b_) if soft else None 

    a , b = a_ , b_

    return a , b


def root_findroot(
        poly, 
        most = 5,
        grid_coor = None,
        grid_value = None,
        method = 'newton'
    ):
    '''
    Find the possible roots of a cyclic polynomial by gradient descent and guessing. 
    The polynomial is automatically standardlized so no need worry the stability. 

    Both the interior points and the borders are searched.

    Params
    -------
    gridval:

    method: 'nsolve' or 'newton'

    Returns
    -------
    
    roots: list of tuples
        Containing (a,b) where (a,b,1) is (near) a local minima.
    
    strict_roots: list of tuples
        Containing (a,b) where the function is possibly zero at (a,b,1).
    '''
    extrema = _root_findroot_initial_guess(grid_coor, grid_value)
    
    if method == 'nsolve':
        result_roots = _root_findroot_nsolve(poly, initial_guess = extrema)
    elif method == 'newton':
        result_roots = _root_findroot_newton(poly, initial_guess = extrema)

    for i, (roota, rootb) in enumerate(result_roots):
        if roota > 1:
            if rootb > roota:
                roota, rootb = 1 / rootb, roota / rootb
            else:
                roota, rootb = rootb / roota, 1 / roota
        elif rootb > 1: # and roota < 1 < rootb
            roota, rootb = 1 / rootb, roota / rootb
        else:
            continue
        result_roots[i] = (roota, rootb)

    # compute roots on the border
    poly_univariate_diff = poly.subs([('b', 0), ('c', 1)]).diff('a')
    poly_univariate_diff2 = poly_univariate_diff.diff('a')
    poly_univariate_diff = poly_univariate_diff.factor_list()
    try:
        for poly_part in poly_univariate_diff[1]:
            poly_part = poly_part[0]
            for r in sp.polys.nroots(poly_part):
                if r.is_real and r >= 0 and poly_univariate_diff2(r) >= 0:
                    result_roots.append((r, sp.S(0)))
    except:
        pass

    # remove repetitive roots
    result_roots = dict(((r[0].n(4), r[1].n(4)), r) for r in result_roots)
    result_roots = result_roots.values()

    vals = [(poly(a, b, 1), (a, b)) for a, b in result_roots]
    vals = sorted(vals)
    if len(vals) > most:
        vals = vals[:most]

    reg = max(abs(i) for i in poly.coeffs()) * deg(poly) * 5e-9
    
    result_roots = [r for v, r in vals]
    strict_roots = [r for v, r in vals if v < reg]

    print('Tolerance =', reg, '\nStrict Roots =', strict_roots,'\nNormal Roots =', 
            list(set(result_roots) ^ set(strict_roots)))
    return result_roots, strict_roots


def _root_findroot_initial_guess(grid_coor, grid_value):
    # grid_coor[k] = (i,j) stands for the value  f(n-i-j, i, j)
    # (grid_size + 1) * (grid_size + 2) // 2 = len(grid_coor)
    n = round((2 * len(grid_coor) + .25) ** .5 - 1.5)
    grid_dict = dict(zip(grid_coor, grid_value))

    trunc = (2*n + 3 - n // 3) * (n // 3) // 2
    
    extrema = []
    for (i, j), v in zip(grid_coor[trunc:], grid_value[trunc:]):
        # without loss of generality we may assume j = max(i,j,n-i-j)
        # need to be locally convex
        if i > j or n - i - j > j or i == 0 or v >= grid_dict[(i,j-1)] or v >= grid_dict[(i+1,j-1)]:
            continue
        if v >= grid_dict[(i-1,j)] or v >= grid_dict[(i-1,j-1)] or v >= grid_dict[(i-1,j+1)]:
            continue
        if i+j < n and (v >= grid_dict[(i+1,j)] or v >= grid_dict[(i,j+1)]):
            continue
        if i+j+1 < n and v >= grid_dict[(i+1,j+1)]:
            continue
        extrema.append(((n-i-j)/j, i/j))
    
    order = (sorted(list(range(len(grid_value))), key = lambda x: grid_value[x]))
    # print(sorted(grid_value))
    # print([(j/(i+1e-14) ,(n-i-j)/(i+1e-14)) for i,j in [grid_coor[o] for o in order]])
    # print([(i, j) for i,j in [grid_coor[o] for o in order]])
    # print(extrema)
    return extrema

def _root_findroot_nsolve(
        poly,
        initial_guess = []
    ):
    """
    Numerically find roots with sympy nsolve.
    """
    result_roots = []
    
    poly = poly.subs('c',1).as_expr()
    poly_diffa = poly.diff('a')
    poly_diffb = poly.diff('b')

    for e in initial_guess:
        try:
            roota, rootb = sp.nsolve(
                (poly_diffa, poly_diffb),
                sp.symbols('a b'),
                e
            )
            result_roots.append((roota, rootb))
        except:
            pass

    return result_roots



def _root_findroot_newton(
        poly,
        initial_guess = None
    ):
    """
    Numerically find roots with newton's algorithm.
    """

    # replace c = 1
    poly = poly.eval('c',1)

    result_roots = []
    
    # regularize the function to avoid numerical instability
    # reg = 2. / sum([abs(coeff) for coeff in poly.coeffs()]) / deg(poly)

    # Newton's method
    # we pick up a starting point which is locally convex and follows the Newton's method
    da = poly.diff('a')
    db = poly.diff('b')
    da2 = da.diff('a')
    dab = da.diff('b')
    db2 = db.diff('b')

    # initial_guess = None
    if initial_guess is None:
        initial_guess = product(np.linspace(0.1,0.9,num=10), repeat = 2)

    for a , b in initial_guess:
        for iter in range(20): # by experiment, 20 is oftentimes more than enough
            # x =[a',b'] <- x - inv(nabla)^-1 @ grad 
            lasta = a
            lastb = b
            da_  = da(a,b)
            db_  = db(a,b)
            da2_ = da2(a,b)
            dab_ = dab(a,b)
            db2_ = db2(a,b)
            det_ = da2_ * db2_ - dab_ * dab_ 
            if det_ <= 0: # not locally convex / not invertible
                break 
            else:
                a , b = a - (db2_ * da_ - dab_ * db_) / det_ , b - (-dab_ * da_ + da2_ * db_) / det_
                if abs(a - lasta) < 5e-15 and abs(b - lastb) < 5e-15:
                    # stop updating
                    break 

        if det_ <= -1e-6 or abs(a) < 1e-6 or abs(b) < 1e-6:
            # trivial roots
            pass
        # if (abs(a-1) < 1e-6 and abs(b-1) < 1e-6):
        #     pass
        else:
            result_roots.append((sp.Float(a), sp.Float(b)))
            # if poly(a,b) * reg < 1e-6:
            #     # having searched one nontrivial root is enough as we cannot handle more
            #     break 

    return result_roots



if __name__ == '__main__':
    from tqdm import tqdm 
    for i in range(1, 3):
        for j in tqdm(range(1, 65537)):
            if gcd(i,j) == 1:
                p, q = rationalize(i/j, reliable=True)
                if q*i != p*j:
                    print('%d/%d != %d/%d'%(i,j,p,q))
