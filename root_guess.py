from ctypes.wintypes import PWIN32_FIND_DATAW
from text_process import *
import sympy as sp
from math import gcd
from itertools import product

def rationalize(v, rounding = 1e-2, mod=None):
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
    return 0 , 1

def rationalize_array(x, rounding = 1e-2, tol=1e-7, mod=None):
    '''
    Approximates each nonnegative floating number to a reasonable fraction and
    leave the floating number unchanged if failed.

    Params
    ------
    x: ndarray

    rounding: float
        Maximize rounding error allowed.

    tol: float
        When some number is below the tolerance, then set it to 0.

    mod: unsigned int / list / ...
        Possible denominators that put into tries. Set to None to trigger default settings.

    '''
    x = np.where(x > tol, x, 0)
    y = []
    for v in x:
        y.append(rationalize(v, rounding=rounding, mod=mod))
    return y

def verify(y, polys, poly, tol = 1e-7):
    '''
    Verify whether the fraction approximation is valid
    by substracting the partial sums and checking whether the remainder is zero.
    '''
    for coeff, f in zip(y, polys):
        if coeff[0] != 0:
            if coeff[1] != -1:
                poly -= (coeff[0] / coeff[1]) * f
            else:
                poly -= coeff[0] * f

    for coeff in poly.coeffs():
        # some coefficient is larger than tolerance, approximation failed
        if abs(coeff) > tol:
            return False
    return True


def verify_isstrict(func, root, tol=2e-6):
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


def findroot(poly, alpha=2e-1, drawback=1e-3, tol=1e-7, maxiter=5000, roots=None):
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

    # regularize the function to avoid numerical instability
    reg = 2. / sum([abs(coeff) for coeff in poly.coeffs()]) / deg(poly)
    poly = poly * reg
    grada = poly.diff('a')
    gradb = poly.diff('b')
    result_roots  = []
    

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
    polycopy = poly

    # search the roots on the border
    # replace b = 0, c = 1
    poly = poly.eval('b',0)
    grada = poly.diff('a')
    alpha = _alpha
    
    val1, val2 = 100, 100
    a, val2 = findbest(np.linspace(0.2,1.8,num=61), lambda x: float(poly(x)))

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
    strict_roots = [root for root in result_roots if verify_isstrict(lambda x: float(polycopy(*x)), root)]
    
    return result_roots, strict_roots


def root_tengents(roots, tol=1e-6, rounding=0.19, mod=(180,252,336)):
    '''
    Generate possible tangents according to the roots given. 

    Linear, quadratic and cubic tangents have been implemented. 

    Linear: ka + b - c

    Quadratic: a^2 - b^2 - p (ac - ab) + q (bc - ab)

    Cubic: a^2c-b^2c + u (a^2b - abc) + v (ab^2 - abc)

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
        # Great available knowledge on Vasc / Vasile
        if abs(a - 0.643104) < tol and abs(b - 0.198062) < tol:
            tangents += ['b2+a2-2ab-bc','2c2+ab-3ac-bc','b2-a2+ab+ac-2bc','s(a3-a2b-2ab2)+6abc',
                '2ab2-ca2-a2b-bc2+c2a','3a3+3b3-6c3+3b2c-c2a-2a2b-2bc2+16ca2-14ab2','s(2a2b-3ab2)+3abc',
                '16a3-38a2b+a2c+17ab2+15abc-6ac2-6b2c+bc2','a3c+bc3+a2bc+4ab2c-5abc2-2ab3']
            continue
        elif abs(a - 0.198062) < tol and abs(b - 0.643104) < tol:
            tangents += ['b2+c2-2cb-ba', '2a2+cb-3ca-ba', 'b2-c2+cb+ca-2ba', 's(c3-c2b-2cb2)+6cba', 
                '2cb2-ac2-c2b-ba2+a2c', '3c3+3b3-6a3+3b2a-a2c-2c2b-2ba2+16ac2-14cb2', 's(2c2b-3cb2)+3cba',
                '16c3-38c2b+c2a+17cb2+15cba-6ca2-6b2a+ba2','c3a+ba3+c2ba+4cb2a-5cba2-2cb3']
            continue

        a = rationalize(a, rounding=1e-1, mod=mod)
        b = rationalize(b, rounding=1e-1, mod=mod)
        if b[0] == 0 and a[0] != 0 and a[1] != -1:
            tangents += [f'{a[1]}/{a[0]}*a+b-c',f'{a[1]}/{a[0]}*a-b-c']



        # Quadratic
        '''
        Sometimes roots are irrational, but coefficients are not
        e.g. roots (a,b,1) = (0.307974173856697, 0.198058832471963, 1)
            By some numerical calculation, 
            p = 3.0000275840807333 ,  q = 5.000061226507757
            where (p,q) = (3,5) is a confident guess
        
        Backup choice:
        If p , q are far from any fraction, but a , b are near to some fractions 
        e.g. If a guess (a,b) = (3/10, 1/5)
            then p , q can be obtained by applying computation on the guesses of a , b:
            (p,q) = (697/218, 1117/218)
            
            As this often results in ugly fractional coefficients, we call it a 'backup' choice.

        '''



        # initialize
        p , q , p2 , q2 = -1 , -1 , -1 , -1
        if a[1] != -1 and b[1] != -1: # both rational numbers
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
            if p[1] !=-1 and q[1] != -1:
                p = sp.Rational(p[0],p[1])
                q = sp.Rational(q[0],q[1])
            else:
                if isinstance(p2,sp.core.Rational): # use the backup choice
                    p , q = p2 , q2                    
        
        if isinstance(p,sp.core.Rational): # nice choice
            tangents += [f'a2-b2+{p.p}/{p.q}*(ab-ac)+{q.p}/{q.q}*(bc-ab)']
            p2 , q2 = p , q   # backup
        
        
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


    return tangents