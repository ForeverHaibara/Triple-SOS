from typing import Optional

import sympy as sp


def rationalize(
        v: float,
        rounding: float = 1e-2,
        reliable: bool = False, 
        truncate: Optional[int] = 32
    ) -> sp.Rational:
    """
    Approximates a floating number to a reasonable fraction.

    Parameters
    -----------
    v : float
        The floating number to be approximated.
    rounding : float
        The tolerance for approximation, it is used only if reliable == False.
    reliable : bool
        If True, use complete continued fraction to approximate.
        If False, approximate a continued fraction when abs(p/q - v) < rounding.
    truncate : int
        Truncate the continued fraction at the given length. It is used only if reliable == True.

    Return
    ----------
    p/q : sp.Rational
        The approximation of v.
    """
    if isinstance(v, (sp.Rational, int)):
        return sp.S(v)
    else:
        if True: # reliable:
            # https://tieba.baidu.com/p/7846250213
            x = sp.Rational(v)
            t = sp.floor(x)
            x = x - t
            fracs = [t]
            i = 0
            j = -1
            while i < truncate:
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
                    return x
                
                # by experiment, |v-x| >> eps only happens when x.q = 2^k
                # where we should use the full fraction list
                x = 0
                for t in fracs[::-1]:
                    x += t
                    x = 1 / x

                x = 1 / x
                # if abs(v - x) < 1e-6: # close approximation
                return x
            else: # not reliable
                return sp.nsimplify(v, rational = True, tolerance = rounding)

                # deprecated
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
                            return x
                        # cancel this move and use shorter truncation
                        x = 0
                        for t in fracs[:length-1][::-1]:
                            x += t
                            x = 1 / x

                        x = 1 / x
                        if abs(v - x) < rounding:
                            return x

                # if not found, use the full fraction list
                # else:
                #     return x

    return sp.nsimplify(v, rational = True, tolerance = rounding)
    

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
        if isinstance(v, (float, sp.Float)):
            if abs(v) < tol:
                y.append(0)
            else:
                y.append(rationalize(v, reliable = reliable))
        elif isinstance(v, tuple):
            y.append(v[0] / v[1])
        elif isinstance(v, sp.Expr) and not isinstance(v, sp.Rational):
            v_ = v.as_numer_denom()
            y.append(v_[0] / v_[1])
        else:
            y.append(v)
    return y



def rationalize_bound(v, direction = 1, roundings = None, compulsory = True):
    """
    Yield rational approximation of v.

    Parameters
    -------
    direction: 1 or 0 or -1
        If direction = 1, find something > v
        If direction = -1, find something < v
        If direction = 0, find anything close to v
    """
    if isinstance(v, sp.Rational):
        yield v
        return
    if roundings is None:
        roundings = (.5, .2, .1, .05, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8)

    if direction == 0:
        compare = lambda a, b: True
    else:
        compare = lambda a, b: True if ((a > b) ^ (direction == -1)) else False

    previous_v = None
    for rounding in roundings:
        v_ = rationalize(v + direction * rounding * 3, rounding = rounding, reliable = False)
        if v_ != previous_v and compare(v_, v):
            previous_v = v_
            yield v_
    
    if not compulsory:
        return
    
    for rounding in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
        v_ = sp.nsimplify(v + direction * rounding * 10, rational = True, tolerance = rounding)
        if v_ != previous_v and compare(v_, v):
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


def cancel_denominator(nums):
    """
    Extract the gcd of numerators and lcm of denominators.
    """
    from functools import reduce
    nums = list(filter(lambda x: isinstance(x, sp.Rational), nums))
    if len(nums) <= 1:
        return sp.S(1)

    p = reduce(sp.gcd, [_.as_numer_denom()[0] for _ in nums])
    q = reduce(sp.lcm, [_.as_numer_denom()[1] for _ in nums])

    return p / q



def rationalize_quadratic_curve(curve: sp.Expr, one_point = False):
    """
    (EXPERIMENTAL) Rationalize a quadratic curve.

    Parameters
    ----------
    curve: sympy expression
        A quadratic curve expression.

    one_point: bool
        If True, only one rational point is returned.
    """
    from sympy.solvers.diophantine.diophantine import diop_DN
    from sympy.ntheory.factor_ import core
    x, y = list(curve.free_symbols)
    curve = curve.as_poly(x)
    if curve.degree() == 1:
        x = (-curve.coeffs()[1] / curve.coeffs()[0]).factor()
        return (x, y)
    elif curve.degree() > 2:
        return
    if curve.all_coeffs()[0] < 0:
        curve = -curve

    # curve = a*(x-...)^2 + curve_y
    curve_y = -curve.all_coeffs()[1]**2/4/curve.all_coeffs()[0] + curve.all_coeffs()[2]
    curve_y = curve_y.as_poly(y)

    # curve = a*(x-...)^2 + b*(y-...)^2 + const = 0
    const = -curve_y.all_coeffs()[1]**2/4/curve_y.all_coeffs()[0] + curve_y.all_coeffs()[2]
    x0 = -curve.all_coeffs()[1]/2/curve.all_coeffs()[0]
    y0 = -curve_y.all_coeffs()[1]/2/curve_y.all_coeffs()[0]
    a, b, c = curve.all_coeffs()[0], curve_y.all_coeffs()[0], const
    
    # cancel the denominator
    r = cancel_denominator((a,b,c))
    a, b, c = a / r, b / r, c / r
    if a < 0:
        a, b, c = -a, -b, -c
    
    # convert to (sqrt(a)*(x-..))^2 + b*(y-...)^2 + c = 0
    t = core(a)
    a, b, c = a * t, b * t, c * t
    b_, c_ = core(abs(b.p*b.q)) * sp.sign(b), core(abs(c.p*c.q)) * sp.sign(c)

    sol = diop_DN(-b_, -c_)
    if len(sol) == 0:
        return
    
    # u^2 + b_v^2 + c_ = 0 => (u * sqrt(c/c_))^2 + b * (v * sqrt(c/c_) / sqrt(b/b_))^2 + c = 0
    u, v = sol[0]
    tmp = sp.sqrt(c / c_)
    x_, y_ = u * tmp / sp.sqrt(a), v * tmp / sp.sqrt(b / b_)
    y_ = y_ + y0
    x_ = x_ + x0.subs(y, y_)
    if one_point:
        return {x: x_, y: y_}
    
    # now that (x_, y_) is a rational point on the curve, we can find a rational line
    t = sp.symbols('t')
    curve_secant = curve.subs(y, t*(x-x_)+y_).as_poly(x) # x = x_ must be a root
    x__ = (curve_secant.all_coeffs()[2] / curve_secant.all_coeffs()[0] / x_).factor()
    y__ = (t*(x__-x_)+y_).factor()
    return {x: x__, y: y__}