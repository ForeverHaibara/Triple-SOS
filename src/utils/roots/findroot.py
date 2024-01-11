from itertools import product
from typing import Union, Callable, List, Optional, Tuple

import numpy as np
import sympy as sp
from sympy.plotting.experimental_lambdify import vectorized_lambdify

from .grid import GridRender, GridPoly
from .rationalize import rationalize
from .roots import Root, RootAlgebraic, RootRational
from .rootsinfo import RootsInfo
from ..polytools import deg


def find_best(
        choices: List,
        func: Callable,
        init_choice: Optional[object] = None,
        init_val: float = 2147483647,
        objective: str = 'min'
    ):
    """
    Find the best choice of all choices to minimize / maximize the function.

    Return the best choice and the corresponding value.

    Parameters
    ----------
    choices : list
        A list of choices.
    func : callable
        The function to be minimized.
    init_choice : object
        The initial choice, used as default if no better choice is found.
    init_val : float
        The initial value of f(init_choice)
    objective : str
        One of 'min' or 'max'. Whether to minimize or maximize the function.

    Returns
    ----------
    best_choice : object
        The best choice.
    """
    best_choice = init_choice
    val = init_val
    is_better = lambda x, y: x < y if objective == 'min' else x > y
    for choice in choices:
        val2 = func(choice)
        if is_better(val2, val):
            val = val2
            best_choice = choice
    return best_choice, val


def nroots(poly, method = 'numpy', real = False, nonnegative = False):
    """
    Wrapper function to find the numerical roots of a sympy polynomial.
    Note that sympy nroots is not stable when the polynomial has multiplicative roots,
    so we need to factorize the polynomial sometimes.

    Parameters
    ----------
    poly : sympy.Poly
        The polynomial to be solved.
    method : str, optional
        The method to be used. 'numpy' uses numpy.roots, 'sympy' uses sympy.nroots.
    real : bool, optional
        Whether to only return real roots.
    nonnegative : bool, optional
        Whether to only return nonnegative roots.
    """
    if method == 'numpy':
        roots = [sp.S(_) for _ in np.roots(poly.all_coeffs())]
    elif method == 'sympy':
        roots = sp.polys.nroots(poly)
    elif method == 'factor':
        roots_rational = []
        roots = []
        for part, mul in poly.factor_list()[1]:
            if part.degree() == 1:
                roots_rational.append(-part.all_coeffs()[1] / part.all_coeffs()[0])
            else:
                roots.extend(sp.polys.nroots(part))
        roots = roots_rational + roots

    if real:
        roots = [_ for _ in roots if _.is_real]
    if nonnegative:
        roots = [_ for _ in roots if _.is_nonnegative]
    
    return roots


def find_nearest_root(poly, v, method = 'rootof'):
    """
    Find the nearest root of a univariate polynomial to a given value.
    This helps select the closed-form root corresponding to a numerical value.

    Parameters
    ----------
    poly : sympy.Poly
        The polynomial to be solved.
    v : float
        The approximated root value.
    method : str
        One of 'roots' or 'rootof'.
    """
    if poly.degree() == 1:
        c1, c0 = poly.all_coeffs()
        return sp.Rational(-c0, c1)
    v = v.n(20)
    if method == 'roots':
        roots = sp.polys.roots(poly)
    elif method == 'rootof':
        roots = poly.all_roots()
    best, best_dist = None, None
    for r in roots:
        dist = abs(r.n(20) - v)
        if best is None or dist < best_dist:
            best, best_dist = r, dist
    return best


def _compute_hessian(f: sp.Expr, vars: Tuple[sp.Symbol, sp.Symbol] = None, vectorized = True):
    """
    Compute df/dx, df/dy, d2f/dx2, d2f/dxdy, d2f/dy2
    with respect to a 2d function f(x,y).
    """
    vars = vars or f.gens
    x, y = vars
    if vectorized:
        wrapper = lambda g: vectorized_lambdify((x, y), g)
    else:
        wrapper = lambda g: g

    dx = f.diff(x)
    dy = f.diff(y)
    dx2 = dx.diff(x)
    dxy = dx.diff(y)
    dy2 = dy.diff(y)
    return list(map(wrapper, [dx, dy, dx2, dxy, dy2]))


def optimize_discriminant(discriminant, soft = False, verbose = False):
    x, y = discriminant.gens
    best_choice = (2147483647, 0, 0)
    for a, b in product(range(-5, 7, 2), repeat = 2): # integer
        v = discriminant(a, b)
        if v <= 0:
            best_choice = (v, a, b)
        elif v < best_choice[0]:
            best_choice = (v, a, b)

    v , a , b = best_choice
    if v > 0:
        for a, b in product(range(a-1, a+2), range(b-1, b+2)): # search a neighborhood
            v = discriminant(a, b)
            if v <= 0:
                best_choice = (v, a, b)
                break
            elif v < best_choice[0]:
                best_choice = (v, a, b)
    if verbose:
        print('Starting Search From', best_choice[1:], ' f =', best_choice[0])
    if v <= 0:
        return {x: a, y: b}

    if v > 0:
        a = a * 1.0
        b = b * 1.0
        dervs = _compute_hessian(discriminant, (x, y))
        # x =[a',b'] <- x - inv(nabla)^-1 @ grad
        for i in range(20):
            lasta , lastb = a , b
            da_, db_, da2_, dab_, db2_ = [f(a,b) for f in dervs]
            det_ = da2_ * db2_ - dab_ * dab_
            if verbose:
                print('Step Position %s, f = %s, H = %s'%((a,b), discriminant(a,b).n(20), det_))
            if det_ == 0:
                break
            else:
                a , b = a - (db2_ * da_ - dab_ * db_) / det_ , b - (-dab_ * da_ + da2_ * db_) / det_
                if abs(lasta - a) < 1e-9 and abs(lastb - b) < 1e-9:
                    break
        v = discriminant(a, b)

    if v > 1e-6 and not soft:
        return None

    # iterative deepening
    rounding = 0.5
    for i in range(5):
        a_ = rationalize(a, rounding, reliable = False)
        b_ = rationalize(b, rounding, reliable = False)
        v = discriminant(a_, b_)
        if v <= 0:
            break
        rounding *= .1
    else:
        return {x: a_, y: b_} if soft else None

    return {x: a_, y: b_}


def findroot(
        poly: sp.Poly,
        most: int = 5,
        grid: GridPoly = None,
        method: str = 'newton',
        verbose: bool = False,
        with_tangents: Union[bool, Callable] = True,
    ):
    """
    Find the possible local minima of a cyclic polynomial by gradient descent and guessing. 
    The polynomial is automatically standardlized so no need worry the stability. 

    Both the interior points and the borders are searched.

    Parameters
    ----------
    poly : sympy.Poly
        The polynomial to be solved.
    most : int
        The maximum number of roots to be returned.
    grid : GridPoly
        The grid to be used. If None, a new grid will be generated.
    method : str
        The method to be used. 'nsolve' uses sympy.nsolve, 'newton' uses newton's method.
    verbose : bool
        Whether to print the process.
    with_tangents : bool | callable
        Whether to compute the tangents at the roots. If callable, it will be used as the function to compute the tangents.

    Returns
    ----------
    rootsinfo : RootsInfo
        The roots information.
    """
    if grid is None:
        grid = GridRender.render(poly, with_color=False)

    if not (poly.domain in (sp.polys.ZZ, sp.polys.QQ, sp.polys.RR)):
        roots = []
    else:
        roots = _findroot_helper.findroot(poly, grid, method)

        # compute roots on the border
        roots += _findroot_helper._findroot_border(poly)

        # remove repetitive roots
        roots = _findroot_helper._remove_repetitive_roots(roots)

    vals = [(poly(a, b, 1), (a, b)) for a, b in roots]
    vals = sorted(vals)
    if len(vals) > most:
        vals = vals[:most]

    reg = max(abs(i) for i in poly.coeffs()) * deg(poly) * 5e-9
    
    roots = [r for v, r in vals]
    strict_roots = [r for v, r in vals if v < reg]

    if verbose:
        print('Tolerance =', reg, '\nStrict Roots =', strict_roots,'\nNormal Roots =',
                list(set(roots) ^ set(strict_roots)))

    rootsinfo = RootsInfo(
        poly = poly,
        roots = roots,
        strict_roots = strict_roots,
        reg = reg,
        with_tangents = with_tangents,
    )

    return rootsinfo


class _findroot_helper():
    """
    Helper class for findroot. It stores multiple root-finding functions.
    """
    @classmethod
    def findroot(cls, 
            poly: sp.Poly,
            grid: GridPoly, 
            method: str = 'newton'
        ) -> List[Tuple[float, float]]:
        """
        Find roots of a polynomial.

        Parameters
        ----------
        poly : sympy.Poly
            The polynomial to be solved.
        grid : GridPoly
            The grid to be used. If None, a new grid will be generated.
        method : str
            The method to be used. 'nsolve' uses sympy.nsolve, 'newton' uses newton's method.

        Returns
        ----------
        roots : list of tuple
            The roots of the polynomial, each in the form (a, b), indicating (a, b, 1).
        """
        grid_coor, grid_value = grid.grid_coor, grid.grid_value

        extrema = cls._initial_guess(grid_coor, grid_value)
        roots = cls.get_method(method)(poly, initial_guess = extrema)
        roots = cls._standardize_roots(roots)

        return roots

    @classmethod
    def available_methods(cls):
        return ['nsolve', 'newton']

    @classmethod
    def get_method(cls, name: str) -> Callable:
        if name not in cls.available_methods():
            raise ValueError(f'Unknown method {name}. Please use f{__class__}.available_methods() to check.')
        return getattr(cls, '_findroot_' + name)

    @classmethod
    def _standardize_roots(cls, roots: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        for i, (a, b) in enumerate(roots):
            if a > 1:
                if b > a:
                    a, b = 1 / b, a / b
                else:
                    a, b = b / a, 1 / a
            elif b > 1: # and a < 1 < b
                a, b = 1 / b, a / b
            else:
                continue
            roots[i] = (a, b)
        return roots

    @classmethod
    def _remove_repetitive_roots(cls, roots: List[Tuple[float, float]], prec: int = 4) -> List[Tuple[float, float]]:
        """
        Remove repetitive roots if they are close enough < 10^(-prec).
        """
        roots = dict(((r[0].n(prec), r[1].n(prec)), r) for r in roots).values()
        return roots

    @classmethod
    def _initial_guess(cls, grid_coor, grid_value):
        """
        Given grid coordinate and grid value, search for local minima.    
        """
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
        
        # order = (sorted(list(range(len(grid_value))), key = lambda x: grid_value[x]))
        # print(sorted(grid_value))
        # print([(j/(i+1e-14) ,(n-i-j)/(i+1e-14)) for i,j in [grid_coor[o] for o in order]])
        # print([(i, j) for i,j in [grid_coor[o] for o in order]])
        # print(extrema)
        return extrema

    @classmethod
    def _findroot_border(cls, poly: sp.Poly) -> List[Tuple[float, float]]:
        """
        Return roots on the border of a polynomial.
        """
        roots = []
        a, b, c = sp.symbols('a b c')
        poly_diff = poly.subs({b: 0, c: 1}).as_poly(a).diff(a)
        poly_diff2 = poly_diff.diff(a)
        try:
            for r in nroots(poly_diff, method = 'factor', real = True, nonnegative = True):
                if poly_diff2(r) >= 0:
                    roots.append((r, sp.S(0)))
        except:
            pass
        return roots

    @classmethod
    def _findroot_nsolve(cls, poly, initial_guess = []):
        """
        Numerically find roots with sympy nsolve.
        """
        roots = []

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
                roots.append((roota, rootb))
            except:
                pass

        return roots

    @classmethod
    def _findroot_newton(cls, poly, initial_guess = []):
        """
        Numerically find roots with newton's algorithm.
        """
        roots = []

        # replace c = 1
        poly = poly.subs('c', 1).as_expr()

        # regularize the function to avoid numerical instability
        # reg = 2. / sum([abs(coeff) for coeff in poly.coeffs()]) / deg(poly)

        # Newton's method
        # we pick up a starting point which is locally convex and follows the Newton's method
        dervs = _compute_hessian(poly, sp.symbols('a b'))

        if initial_guess is None:
            initial_guess = product(np.linspace(0.1,0.9,num=10), repeat = 2)

        for a , b in initial_guess:
            for times in range(20): # by experiment, 20 is oftentimes more than enough
                # x =[a',b'] <- x - inv(nabla)^-1 @ grad
                lasta = a
                lastb = b
                da_, db_, da2_, dab_, db2_ = [f(a,b) for f in dervs]
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
                roots.append((sp.Float(a), sp.Float(b)))
                # if poly(a,b) * reg < 1e-6:
                #     # having searched one nontrivial root is enough as we cannot handle more
                #     break

        return roots


def findroot_resultant(poly: sp.Poly):
    """
    Find the roots of a polynomial using the resultant method. This is
    essential in SDP SOS, because we need to construct exact subspace
    constrainted by roots of the polynomial to perform SDP on a full-rank manifold.

    The method does not guarantee to find all roots, but find at least one in 
    each Galois group. For instance, if (1,1,sqrt(2)) is a root, then (1,1,-sqrt(2))
    must be a root as well, due to the principle of minimal polynomial.
    Then it is possible for us to return (1,1,-sqrt(2)), although it is not
    positive. This seems to miss some roots, but it is not a problem in SDP SOS,
    because the spanned subspace algorithm includes the permutation of roots.

    TODO:
    1. Optimize the speed by removing redundant permuations.

    Parameters
    ----------
    poly : sympy.Poly
        The polynomial to find roots of.

    Returns
    ----------
    roots : List[RootAlgebraic]
        A list of roots of the polynomial.
    """
    return _findroot_helper_resultant._findroot_resultant(poly)



class _findroot_helper_resultant():
    """
    Helper class for findroot using resultant method.
    """
    @classmethod    
    def _findroot_resultant(cls, poly):
        """See findroot_resultant."""
        a, b, c = sp.symbols('a b c')
        poly = poly.subs(c,1) # -a-b).as_poly(a,b)
        parts = poly.factor_list()[1]
        roots = []

        findroot_func = cls._findroot_resultant_irreducible

        # when the multiplicity at origin > 1, we require hessian == 0
        mult_at_origin = 0

        for part in parts:
            poly, multiplicity = part
            if poly.degree() == 1:
                continue
            elif poly.degree() == 2:
                if poly(1,1) == 0:
                    mult_at_origin += 1
                    roots.append(RootRational((1,1,1)))
                continue
            else:
                if len(parts) > 1 and poly(1,1) == 0:
                    mult_at_origin += 1
                roots.extend(findroot_func(poly))


        # put the positive roots in front
        # roots_positive, roots_negative = [], []
        # for root in roots:
        #     if root.root[0] >= 0 and root.root[1] >= 0 and root.root[2] >= 0:
        #         roots_positive.append(root)
        #     else:
        #         roots_negative.append(root)
        # roots = roots_positive + roots_negative

        # remove duplicate roots
        roots_clear = []
        for root in roots:
            if any(r.root == root.root for r in roots_clear):
                continue
            if mult_at_origin > 1 and root.root[0] == 1 and root.root[1] == 1 and root.root[2] == 1:
                root.multiplicity = mult_at_origin
            roots_clear.append(root)
        return roots_clear

    @classmethod
    def _findroot_resultant_root_pairs(cls, poly, factors, prec = 20, tolerance = 1e-12):
        """
        Find the numerical root pairs of a polynomial using the resultant method.
        The polynomial should be of two variables. Factors are the minimal polynomial
        of the second variable.

        Suppose (a, b, 1) is a root of the original 3-var polynomial, then b is 
        a root in one of the `factors`. Take the cyclic permutation, (b/a, 1/a, 1) is also
        valid. Thus, variable a must be the inverse of one of the factors. There are exceptions
        where ab = 0. When both are zeros, it can be discarded and handled automatically through
        polynomial convex hull in SDP SOS. When b = 0, it does not matter actually. When a = 0,
        we have already know the value of a.

        As there are multiple factors, we need to pair them into (a, b) pairs.

        Parameters
        ----------
        poly : sympy.Poly
            The polynomial of two variables.
        factors: List[Tuple[sympy.Poly, int]]
            The minimal polynomials of the second variable. It is resultant.factor_list()[1].
        prec: int
            The precision of solving roots of each factor numerically.
        tolerance: float
            The tolerance of a pair (a,b) is recognized as a root pair.

        Returns
        ----------
        pairs : List[Tuple[Tuple[float, float], Tuple[int, int]]]
            The root pairs of the polynomial. Each pair is in the form ((a, b), (i, j)),
            where (a, b) is the root pair and (i, j) is the index of the corresponding factors.
        """
        all_roots = []
        all_roots_inv = []
        for i, factor_ in enumerate(factors):
            for r in sp.polys.nroots(factor_[0], n = prec):
                if r.is_real:
                    all_roots.append((r, i))
                    if r != 0:
                        all_roots_inv.append((1/r, i))
                    else:
                        all_roots_inv.append((0, i))

        pairs = []
        for b_, j in all_roots:
            for a_, i in all_roots_inv:
                if abs(a_) > 1 or abs(b_) > 1:
                    # we could assume |c| = max{|a|, |b|, |c|} = 1
                    # as the polynomial is cyclic
                    continue
                v = poly(a_, b_)
                if abs(v) < tolerance:
                    pairs.append(((a_, b_), (i, j)))
        return pairs

    @classmethod
    def _findroot_resultant_irreducible(cls, poly):
        """
        Find root of a 2-var polynomial with respect to a, b using the method of 
        resultant. The polynomial should be irreducible.

        Parameters
        ----------
        poly : sympy.Poly
            The polynomial of two variables. It should be irreducible.

        Returns
        -------
        roots: List[RootAlgebraic]
            The roots of the polynomial.    
        """
        a, b, c = sp.symbols('a b c')
        grad = poly.diff(a)
        res = sp.resultant(poly, grad)
        factors = set(sp.polys.gcd(res, res.diff(b)).factor_list()[1])

        # roots on the border might not be local minima
        poly_border = poly.subs(a, 0)
        grad_border = poly_border.diff(b)
        factors_border = set(sp.polys.gcd(poly_border, grad_border).factor_list()[1])
        if len(factors_border):
            # conjugate factors, b = 0
            factors_border.add((sp.Poly([1,0], b), len(factors_border)))
        factors |= factors_border
        factors = list(factors)

        pairs = cls._findroot_resultant_root_pairs(poly, factors)
        roots = []

        _is_rational = lambda x: isinstance(x, sp.Rational) or (isinstance(x, sp.polys.polyclasses.ANP) and len(x.rep) <= 1)

        for (a_, b_), (i, j) in pairs:
            fx, fy = factors[i][0], factors[j][0]
            # reverse fx
            fx = sp.Poly(fx.all_coeffs()[::-1], fy.gens[0])
            if fx.degree() == 0:
                a_ = sp.S(0)
            else:
                a_ = find_nearest_root(fx, a_, method = 'rootof')
            b_ = find_nearest_root(fy, b_, method = 'rootof')
            root = RootAlgebraic((a_, b_))
            roots.append(root)

            if not ((not _is_rational(a_)) and (not _is_rational(b_)) and _is_rational(root.uv_[0]) and _is_rational(root.uv_[1])):
                roots.append(RootAlgebraic((sp.S(1), a_, b_)))
                roots.append(RootAlgebraic((b_, sp.S(1), a_)))

        return roots