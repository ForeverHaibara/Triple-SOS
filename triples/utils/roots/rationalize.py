from typing import Optional, Tuple, List, Dict, Union, Generator

import sympy as sp
from sympy import Poly, Expr, Symbol, Rational, Integer, Float, sympify

from .polysolve import nroots

def univariate_intervals(polys: Union[Poly, List[Poly]]) -> Generator[Rational, None, None]:
    """
    Compute rational points where polys have sign changes.

    Parameters
    ----------
    polys: sympy.Poly or list of sympy.Poly
        Univariate polynomials to compute intervals for.

    Yields
    ----------
    v: sympy.Rational
        Rational points where the signs of polynomials get changed.
    """
    pre = sp.nan
    for ij in sp.intervals(polys):
        for v in ij[0]:
            if pre != v:
                pre = v
                yield v


def rationalize(
        v: float,
        rounding: float = 1e-2,
        reliable: bool = False,
        truncate: Optional[int] = 32
    ) -> Rational:
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
    p/q : Rational
        The approximation of v.
    """
    if isinstance(v, (Rational, int)):
        return sympify(v)
    else:
        if (not isinstance(v, (float, Float))) and isinstance(v, Expr):
            v = v.n(15)
        if True: # reliable:
            x = Rational(v)
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
        if isinstance(v, (float, Float)):
            if abs(v) < tol:
                y.append(0)
            else:
                y.append(rationalize(v, reliable = reliable))
        elif isinstance(v, tuple):
            y.append(v[0] / v[1])
        elif isinstance(v, Expr) and not isinstance(v, Rational):
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
    if isinstance(v, Rational):
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
    z = Rational(z)  # convert to rational

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
    nums = list(filter(lambda x: isinstance(x, Rational), nums))
    if len(nums) <= 1:
        return Integer(1)

    p = reduce(sp.gcd, [_.as_numer_denom()[0] for _ in nums])
    q = reduce(sp.lcm, [_.as_numer_denom()[1] for _ in nums])

    return p / q



def rationalize_quadratic_curve(
        curve: Expr,
        gen: Symbol = Symbol('t'),
        point: Union[Dict[Symbol, Rational], Tuple[Rational, Rational]] = None,
        one_point: bool = False
    ) -> Dict[Symbol, Rational]:
    """
    (EXPERIMENTAL) Rationalize a quadratic curve using the secant method.

    Parameters
    ----------
    curve : sympy expression
        A quadratic curve expression.

    gen : sympy symbol
        The parametrized parameter.

    point : Dict[Symbol, Rational] or Tuple[Rational, Rational]
        One of the rational point on the curve for secant method.
        If not given, it is searched automatically.

    one_point : bool
        If True, only one rational point is returned.

    Returns
    ----------
    dict :
        The substituion parametrization. If one_point is False,
        both variables are parametrized by symbol `gen`. If one_point
        is True, it returns a rational point on the curve.
    """
    from sympy.solvers.diophantine.diophantine import diop_DN
    from sympy.ntheory.factor_ import core
    x, y = curve.gens if isinstance(curve, Poly) else list(curve.free_symbols)
    if point is not None:
        if not isinstance(point, dict):
            point = {x: point[0], y: point[1]}
        x_, y_ = point[x], point[y]
        if curve.subs({x: x_, y: y_}) != 0:
            raise ValueError(f"Given point does not {point} lie on the curve.")
    else:
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
    t = gen
    curve_secant = curve.subs(y, t*(x-x_)+y_).as_poly(x) # x = x_ must be a root
    x__ = (-curve_secant.all_coeffs()[1] / curve_secant.all_coeffs()[0] - x_).factor()
    y__ = (t*(x__-x_)+y_).factor()
    return {x: x__, y: y__}


def common_region_of_conics(polys: List[Poly], _tol = 1e-10) -> Optional[Tuple[Rational, Rational]]:
    """
    Find (x, y) such that polys[i](x, y) >= 0 for all i.
    where polys are rational conics of two variables.
    """
    # assert len(polys) > 0, "At least one conic is required."

    # remove constant conics
    non_const_conics = []
    for p in polys:
        if p.total_degree() > 0:
            non_const_conics.append(p)
        elif p.coeff_monomial(Integer(1)) < 0:
            return None
    polys = non_const_conics

    if len(polys) == 0:
        return (Integer(0), Integer(0))
    assert len(polys[0].gens) == 2, "The conics must be 2D."
    assert all(p.gens == polys[0].gens for p in polys), "The conics must have the same variables."
    x, y = polys[0].gens


    if len(polys) == 1:
        polys.append(Poly.from_dict({}, (x, y)))

    def _common_region_of_two_conics(f1, f2):
        # try centers, which are easy to compute
        def _center(f):
            if f.total_degree() == 0:
                return (Integer(0), Integer(0))
            sol = sp.solve([f.diff(x), f.diff(y)], [x,y], dict = True)
            def _get_default(k, v):
                r = k.get(v, Integer(0))
                r = r.subs(dict(zip(r.free_symbols, [0]*len(r.free_symbols))))
                return r
            if len(sol) == 1:
                return _get_default(sol[0], x), _get_default(sol[0], y)
            return (sp.nan, sp.nan)
        def _is_finite(sol):
            return all(i.is_finite for i in sol)
        w1, w2 = _center(f1), _center(f2)
        if _is_finite(w1) and f1(*w1) >= 0 and f2(*w1) >= 0:
            yield w1
        if _is_finite(w2) and f1(*w2) >= 0 and f2(*w2) >= 0:
            yield w2

        # if the conics are degenerated and have common factors
        gcd = sp.gcd(f1, f2).as_poly(x, y)
        if gcd.total_degree() == 2:
            # f1 is a multiple of f2
            yield rationalize_quadratic_curve(f1, one_point=True)
        elif gcd.total_degree() == 1:
            a, b, c = [gcd.coeff_monomial(_) for _ in [(1,0),(0,1),(0,0)]]
            # ax + by + c = 0
            if b != 0:
                yield (Integer(0), -c/b)
            if a != 0:
                yield (-c/a, Integer(0))

        # sometimes f1, f2 intersect at the infinity line, e.g. xy = 1 and xy = 4.
        # we can try out x = 0 and y = 0 and y = x three lines to cut the conics
        if True:
            for y_ in univariate_intervals([f1.subs(x, 0), f2.subs(x, 0)]):
                if f1(0, y_) >= 0 and f2(0, y_) >= 0:
                    yield Integer(0), y_
            for x_ in univariate_intervals([f1.subs(y, 0), f2.subs(y, 0)]):
                if f1(x_, 0) >= 0 and f2(x_, 0) >= 0:
                    yield x_, Integer(0)
            for x_ in univariate_intervals([f1.subs(y, x).as_poly(x), f2.subs(y, x).as_poly(x)]):
                if f1(x_, x_) >= 0 and f2(x_, x_) >= 0:
                    yield x_, x_

        def _grad(f, x_, y_):
            return f.diff(x)(x_, y_), f.diff(y)(x_, y_)
        def _norm(v):
            if v[0] == 0 and v[1] == 0: return 0, 0
            m = (v[0]**2 + v[1]**2)**.5
            return v[0]/m, v[1]/m

        # find the intersection of the two conics
        res = sp.polys.resultant(f1, f2, y).as_poly(x)
        for x_ in nroots(res, method='factor', real=True):
            for y_ in nroots(f1.subs(x, x_).as_poly(y), method='factor', real=True):
                if abs(f1(x_, y_)) < _tol and abs(f2(x_, y_)) < _tol:
                    if (not isinstance(x_, Rational)) or (not isinstance(y_, Rational)):
                        grad1 = _norm(_grad(f1, x_, y_))
                        grad2 = _norm(_grad(f2, x_, y_))
                        grad_merged = (grad1[0] + grad2[0], grad1[1] + grad2[1])
                        for h in [1, .5, .1, .05, .01, 1e-5, 1e-8, 1e-12, 1e-15]:
                            x2 = rationalize(x_ + h*grad_merged[0], rounding = h*.01)
                            y2 = rationalize(y_ + h*grad_merged[1], rounding = h*.01)
                            # print('h =', h, 'f =', f1(x2, y2), f2(x2, y2))
                            if f1(x2, y2) >= 0 and f2(x2, y2) >= 0:
                                yield x2, y2
                    elif f1(x_, y_) >= 0 and f2(x_, y_) >= 0:
                        yield x_, y_

    def _reg(f):
        return (f / sum(abs(_) for _ in f.coeffs())).as_poly(x, y)
    for i in range(len(polys)):
        if not polys[i].is_zero:
            polys[i] = _reg(polys[i])

    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            for point in _common_region_of_two_conics(polys[i], polys[j]):
                if point is None:
                    continue
                if all(p(*point) >= 0 for p in polys):
                    return point
