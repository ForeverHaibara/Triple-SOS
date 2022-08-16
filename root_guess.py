from text_process import *
import sympy as sp
from math import gcd
from itertools import product

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


def verify(y, polys, poly, tol: float = 1e-10) -> bool:
    '''
    Verify whether the fraction approximation is valid
    by substracting the partial sums and checking whether the remainder is zero.
    '''
    for coeff, f in zip(y, polys):
        if coeff[0] != 0:
            if coeff[1] != -1:
                if not isinstance(coeff[0], sp.Expr):
                    poly -= sp.Rational(coeff[0] , coeff[1]) * f
                else: 
                    poly -= coeff[0] / coeff[1] * f 
            else:
                poly -= coeff[0] * f

    for coeff in poly.coeffs():
        # some coefficient is larger than tolerance, approximation failed
        if abs(coeff) > tol:
            return False
    return True


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


def findroot(poly, alpha=2e-1, drawback=1e-3, tol=1e-7, maxiter=5000, roots=None, most=5):
    '''
    Find the possible roots of a cyclic polynomial by gradient descent and guessing. 
    The polynomial is automatically standardlized so no need worry the stability. 

    Both the interior points and the borders are searched.

    Params
    -------
    alpha: float
        Gradient descent step size.
    
    drawback: float
        If in the gradient descent falls back {drawback}, then cut down the alpha to half.

    tol: float
        When the gradient is less than tolerance, the gradient process stops.

    maxiter: unsigned int
        Maximum iteration of the gradient descent.

    roots: list of tuple, e.g. [(1/3,1/3)]
        A list of initial root guess (a,b)  (where WLOG c = 1 by homogenousity). Good guesses
        might lead to faster convergence.

    Returns
    -------
    
    roots: list of tuples
        Containing (a,b) where (a,b,1) is (near) a local minima.
    
    strict_roots: list of tuples
        Containing (a,b) where the function is possibly zero at (a,b,1).
    '''

    # replace c = 1
    _alpha = alpha
    poly = poly.eval('c',1)
    poly_univariate = poly.eval('b',0)

    result_roots = []
    
    # regularize the function to avoid numerical instability
    reg = 2. / sum([abs(coeff) for coeff in poly.coeffs()]) / deg(poly)
    poly_reg = poly * reg

    if False:
        # gradient descent (first order method)
        
        grada = poly.diff('a')
        gradb = poly.diff('b')
        # find the best start (minimize the function) for the gradient descent
        # warning: do not start too near to (1,1) or it will fall to the trivial root (1,1)

        # some classical starts are considered, for example the Vasile
        best_start, val2 = findbest(((0.643104,0.198062), (0.198062,0.643104), (2./3,1./3), (1./3,2./3)),
                                    lambda x: float(poly(*x)))
                                    
        if type(roots) != list and type(roots) != tuple:
            roots = (roots,)
        best_start, val2 = findbest(roots, lambda x: float(poly(*x)), best_start, val2)

        best_start, val2 = findbest(product(np.linspace(0.2,0.5,num=10),repeat=2),
                                    lambda x: float(poly(*x)), best_start, val2)
        
        
        #print(best_start)
        a , b = best_start

        for _ in range(maxiter):
            val1, val2 = val2, float(poly(a,b))
            if val2 < -1e-1 or val2 > 100: # ill-conditioned
                break
            if val1 - val2 < -drawback:
                alpha *= 0.5
            u , v = grada(a,b), gradb(a,b)
            a -= alpha * u
            b -= alpha * v 
            if max(abs(u),abs(v)) < tol: # stop when the criterion is met
                result_roots.append((a,b))
                break
        else:
            # not annihilate
            if min(abs(a),abs(1-a)) > 0.06 and min(abs(b),abs(1-b)) > 0.06:
                result_roots.append((a,b))

        
        # copy a regularized polynomial
        poly_reg = poly
    else:
        # Newton's method
        # we pick up a starting point which is locally convex and follows the Newton's method
        da = poly.diff('a')
        db = poly.diff('b')
        da2 = da.diff('a')
        dab = da.diff('b')
        db2 = db.diff('b')
        for a , b in product(np.linspace(0.1,0.9,num=10),repeat=2):
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
                if det_ <= -1e-6: # not locally convex
                    break 
                elif det_ == 0: # not invertible
                    break 
                else:
                    a , b = a - (db2_ * da_ - dab_ * db_) / det_ , b - (-dab_ * da_ + da2_ * db_) / det_
                    if abs(a - lasta) < 1e-9 and abs(b - lastb) < 1e-9:
                        # stop updating
                        break 

            if det_ <= -1e-6 or (abs(a-1) < 1e-6 and abs(b-1) < 1e-6) or abs(a) < 1e-6 or abs(b) < 1e-6:
                # trivial roots
                pass 
            else:
                flg = True 
                for a2, b2 in result_roots:
                    if abs(a2-a) < 1e-5 and abs(b2-b) < 1e-5:
                        # do not append two (nearly) identical roots
                        flg = False 
                        break 
                if not flg:
                    continue 
                result_roots.append((a,b))
                if poly_reg(a,b) < 1e-6:
                    # having searched one nontrivial root is enough as we cannot handle more
                    break 


    # search the roots on the border
    # replace b = 0, c = 1
    if False:
        # use gradient descent
        poly = poly.eval('b',0)
        grada = poly.diff('a')
        alpha = _alpha
        
        val1, val2 = 100, 100
        a, val2 = findbest((1./4, 3./4, 5./4, 7./4), lambda x: float(poly(x)))
        a, val2 = findbest(np.linspace(0.2,1.8,num=61), lambda x: float(poly(x)), a, val2)

        # do not use Newton method to avoid nonzero local minima cases
        # still use gradient descent
        for _ in range(maxiter):
            val1, val2 = val2, float(poly(a))
            if val2 < -1e-1 or val2 > 100: # ill-conditioned
                break
            if val1 - val2 < -drawback:
                alpha *= 0.1
            u = grada(a)
            a -= alpha * u
            if abs(u) < tol:
                result_roots.append((a,0))
                break
        else:
            if abs(a) > 1e-1:
                result_roots.append((a,0))
        # check whether each root is strict
        strict_roots = [root for root in result_roots if verify_isstrict(lambda x: float(poly_reg(*x)), root)]
    else:
        # check whether each root is strict
        vals = [float(poly_reg(*root)) for root in result_roots]
        if len(result_roots) > most:
            index = sorted(range(len(vals)), key = lambda i: vals[i])[:most]
            result_roots = [result_roots[i] for i in index] 
            vals = [vals[i] for i in index]

        strict_roots = [root for i, root in enumerate(result_roots) if abs(vals[i]) < 1e-6]
        # strict_roots = [root for root in result_roots if verify_isstrict(lambda x: float(poly_reg(*x)), root)]

        # use sympy root finding strategy
        for root in sp.polys.polyroots.roots(poly_univariate.diff('a')):
            root_numerical = complex(root)
            # real nonnegative root
            if abs(root_numerical.imag) < 1e-7 and root_numerical.real > 0:
                result_roots.append((root, 0))
                root_numerical = root_numerical.real 
                if abs(root_numerical * reg) < 1e-10:
                    strict_roots.append((root_numerical, 0))
                break 
            

    return result_roots, strict_roots


def root_tangents(roots, tol=1e-6, rounding=0.001, mod=(180,252,336)):
    '''
    Generate possible tangents according to the roots given. 

    Linear, quadratic and cubic tangents have been implemented. 

    Linear: ka + b - c

    Quadratic: a^2 - b^2 - p (ac - ab) + q (bc - ab)

    Cubic: a^2c - b^2c + u (a^2b - abc) + v (ab^2 - abc)

    Roots and coefficients of the tangents are guessed into possible fraction forms. 

    Params
    -------
    roots: list of tuples
        Containing (a,b) where (a,b,1) is possibly some local minima.

    tol: float
        
    rounding: float
        Rounding used for rationalization. Do not set to too large because 
        backward errors are inevitable in gradient descent.
    
    mod: unsigned int / tuple ...
        Denominator guess for rationalization. Do not set to too large because 
        backward errors are inevitable in gradient descent.

    Return
    -------
    tangents: list of str
        a list of str that are tangents generated by the roots.
    '''
    if not roots:
        return []

    #print(roots)
    tangents = []
    for root in roots:
        a , b = root
        if b == 0:
            # on the edge (a,0,1) where a might be symbolic
            if a != 0 and a != 1:
                if isinstance(a, sp.core.Rational):
                    # (1/a)*a + (1-1/a)*b - c
                    tangents.append(f'{a.q}/{a.p}*a+{a.p-a.q}/{a.p}*b-c')
                    # a = (a.p, a.q)
                elif a.is_real is not None:
                    # is_real is a fast way to check whether it is a simple expression
                    # e.g. cubic roots with imaginary unit returns None when querying is_real
                    
                    # e.g. x*x-5*x+1   -> a*a-5*a*c+c*c + b*b-5*b*c + 7*b*a
                    try:
                        mini_poly = sp.polys.polytools.Poly(sp.minimal_polynomial(a))
                        if mini_poly.degree() == 2:
                            if (1,) not in mini_poly.monoms(): # no degree-1 term:
                                u , w = mini_poly.coeffs()
                                v = 0
                            else:
                                u , v , w = mini_poly.coeffs()
                            tangents.append(f'{u}*a*a+{v}*a*c+{w}*c*c')
                            t = - u - v - w - w*w/u - w*v/u
                            tangents.append(f'{u}*a*a+{v}*a*c+{w}*c*c+{w*w}/{u}*b*b+{w*v}/{u}*b*c+{t.p}/{t.q}*b*a')
                            t = - u - v - w - u*u/w - u*v/w
                            tangents.append(f'{w}*c*c+{v}*a*c+{u}*a*a+{u*u}/{w}*b*b+{u*v}/{w}*b*a+{t.p}/{t.q}*b*c')
                            # a = (complex(a).real, -1)

                            # Symmetric Forms 
                            if abs(a) > 1e-3:
                                v = complex(a + 1 / a).real
                                v = rationalize(v, rounding=rounding, mod=mod)
                                if v[1] != -1:
                                    tangents += [f'a2+b2+c2-{v[0]}/{v[1]}*(ab+bc+ca)']
                            continue
                    except: 
                        # sympy.polys.polyerrors.NotAlgebraic: 
                        # 0.932752395204472 doesn't seem to be an algebraic element
                        pass 
            elif a == 1:
                tangents.append('a+b-c')
                continue 
            # b = (0, 1)

            # to numerical
            root = (complex(a).real, 0)
        else:
            # Great available knowledge on Vasc / Vasile
            if abs(a - 0.643104) < tol and abs(b - 0.198062) < tol:
                tangents += ['b2+a2-2ab-bc','2c2+ab-3ac-bc','b2-a2+ab+ac-2bc','s(a3-a2b-2ab2)+6abc',
                    '2ab2-ca2-a2b-bc2+c2a','3a3+3b3-6c3+3b2c-c2a-2a2b-2bc2+16ca2-14ab2','s(2a2b-3ab2)+3abc',
                    '16a3-38a2b+a2c+17ab2+15abc-6ac2-6b2c+bc2','a3c+bc3+a2bc+4ab2c-5abc2-2ab3',
                    'a2b+b2c+c2a-6abc']
                continue
            elif abs(a - 0.198062) < tol and abs(b - 0.643104) < tol:
                tangents += ['b2+c2-2cb-ba', '2a2+cb-3ca-ba', 'b2-c2+cb+ca-2ba', 's(c3-c2b-2cb2)+6cba', 
                    '2cb2-ac2-c2b-ba2+a2c', '3c3+3b3-6a3+3b2a-a2c-2c2b-2ba2+16ac2-14cb2', 's(2c2b-3cb2)+3cba',
                    '16c3-38c2b+c2a+17cb2+15cba-6ca2-6b2a+ba2','c3a+ba3+c2ba+4cb2a-5cba2-2cb3',
                    'ab2+bc2+ca2-6abc']
                continue

            a = rationalize(a, rounding=1e-3, mod=mod)
            b = rationalize(b, rounding=1e-3, mod=mod)
            if b[0] == 0 and a[0] != 0 and a[1] != -1:
                tangents += [f'{a[1]}/{a[0]}*(a-b)+b-c']
        
        

        # Quadratic
        '''
        Sometimes roots are irrational, but coefficients are not
        e.g. roots (a,b,1) = (0.307974173856697, 0.198058832471963, 1)
            By some numerical calculation, 
            p = 3.0000275840807333 ,  q = 5.000061226507757
            where (p,q) = (3,5) is a confident guess

        Backup choices have now been deprecated.
        
        '''



        # initialize
        p , q , p2 , q2 = -1 , -1 , -1 , -1
        p3 , q3 , r3 = 0 , 0 , 0
        if False and a[1] != -1 and b[1] != -1: # both rational numbers
            a , b = sp.Rational(a[0],a[1]) , sp.Rational(b[0],b[1])

            # Solve the equations:
            # p(ab-a)+q(b-ab)+a2-b2 = 0
            # p(b-ba)+q(a-b)+b2-1 = 0
            
            t = ((a*b-a)*(a-b)-(b*(a-1))**2)
            if abs(t) > 1e-3: # backup
                p2 = ((b*b-a*a)*(a-b) - (b-a*b)*(1-b*b))/t
                q2 = ((a*b-a)*(1-b*b)-(b*b-a*a)*(b-b*a))/t

        a , b = root
        t = ((a*b-a)*(a-b)-(b*(a-1))**2)
        if abs(t) > 1e-3:
            p = ((b*b-a*a)*(a-b) - (b-a*b)*(1-b*b))/t
            q = ((a*b-a)*(1-b*b)-(b*b-a*a)*(b-b*a))/t
            
            p = rationalize(p, rounding=rounding, mod=mod)
            q = rationalize(q, rounding=rounding, mod=mod)
            
            
            
            if p[1] != -1 and q[1] != -1:
                p = sp.Rational(p[0],p[1])
                q = sp.Rational(q[0],q[1])
            else:
                if isinstance(p2,sp.core.Rational): # use the backup choice
                    p , q = p2 , q2                    
        
        if isinstance(p,sp.core.Rational): # nice choice
            tangents += [f'a2-b2+{p.p}/{p.q}*(ab-ac)+{q.p}/{q.q}*(bc-ab)']
            # p2 , q2 = p , q   # backup
        
        
        # Cubic        
        '''
        Same steps --- first numerical, then backup choices.
        '''
        a , b = root
        t = ((a*b-a)*(a-b)-(b*(a-1))**2)
        r = 0
        if abs(t) > 1e-3:
            if isinstance(p2,sp.core.Rational): # has backup choice
                a , b = p2 , q2
                p3 = a*a + b
                q3 = - b*b - a  
                r3 = 1 - a*b
                if abs(r3) > 1e-3: 
                    p3 /= r3 
                    q3 /= r3
                else: # abandon the backup
                    p3 , q3 , r3 = 0 , 0 , 0

            # non-backup solution (numerically)  
            a , b = root
            p = ((b*b-a*a)*(a-b) - (b-a*b)*(1-b*b))/t
            q = ((a*b-a)*(1-b*b)-(b*b-a*a)*(b-b*a))/t
            a , b = p , q
            p = a*a + b
            q = - b*b - a  
            r = 1 - a*b

            p = rationalize(p, rounding=rounding, mod=mod)
            q = rationalize(q, rounding=rounding, mod=mod)
            r = rationalize(r, rounding=rounding, mod=mod)
            if p[1] != -1 and q[1] != -1 and r[1] != -1:
                p = sp.Rational(p[0],p[1])
                q = sp.Rational(q[0],q[1])
                r = sp.Rational(r[0],r[1])
                if abs(r) > 1e-3: # r == 0 is degenerated
                    p /= r
                    q /= r 
                else: # use the backup choice
                    p , q , r = p3 , q3 , r3
            else:
                p , q , r = p[0] / abs(p[1]) , q[0] / abs(q[1]) , r[0] / abs(r[1])
                if abs(r) > 1e-3:
                    p /= r
                    q /= r
                    p = rationalize(p, rounding=rounding, mod=mod)
                    q = rationalize(q, rounding=rounding, mod=mod)
                    if p[1] != -1 and q[1] != -1 :
                        p = sp.Rational(p[0],p[1])
                        q = sp.Rational(q[0],q[1])
                    else: # use the backup choice
                        p , q , r = p3 , q3 , r3
            
        if abs(r) > 1e-3 and isinstance(p,sp.core.Rational):
            tangents += [f'a2c-b2c+{p.p}/{p.q}*(a2b-abc)+{q.p}/{q.q}*(ab2-abc)']


        # Symmetric Forms
        a , b = root 
        u = a * b + a + b 
        if abs(u) > 1e-3:
            v = ( a*a + b*b + 1 ) / u
            v = rationalize(v, rounding=rounding, mod=mod)
            if v[1] != -1 and v[0] > 0 and v != (1, 1):
                tangents += [f'a2+b2+c2-{v[0]}/{v[1]}*(ab+bc+ca)']
        
        a , b = root 
        if abs(a) > 1e-3 and abs(b) > 1e-3:
            u = a * b
            v = ((a*a + b)*b + a) / u
            v = rationalize(v, rounding=rounding, mod=mod)
            if v[1] != -1 and v[0] > 0:
                tangents += [f'a2b+b2c+c2a-{v[0]}/{v[1]}abc']
                
            v = ((b*b + a)*a + b) / u
            v = rationalize(v, rounding=rounding, mod=mod)
            if v[1] != -1 and v[0] > 0:
                tangents += [f'ab2+bc2+ca2-{v[0]}/{v[1]}abc']


    return tangents



if __name__ == '__main__':
    from tqdm import tqdm 
    for i in range(1, 3):
        for j in tqdm(range(1, 65537)):
            if gcd(i,j) == 1:
                p, q = rationalize(i/j, reliable=True)
                if q*i != p*j:
                    print('%d/%d != %d/%d'%(i,j,p,q))