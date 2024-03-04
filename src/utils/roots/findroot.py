from itertools import product
from typing import Union, Callable, List, Optional, Tuple

import numpy as np
import sympy as sp
from sympy.plotting.experimental_lambdify import vectorized_lambdify
from scipy.optimize import minimize

from .grid import GridRender, GridPoly
from .rationalize import rationalize
from .roots import Root, RootAlgebraic, RootRational
from .rootsinfo import RootsInfo
from ..polytools import deg, verify_hom_cyclic


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
        dervs = _compute_hessian(discriminant, (x, y), vectorized=False)
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
        method: str = 'nsolve',
        standardize_method: str = 'partial',
        verbose: bool = False,
        with_tangents: Union[bool, Callable] = False,
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
        The method to be used. See _findroot_helper.available_methods() for available methods.
    standardize_method : str
        The method to be used to de-homogenize the polynomial. 'simplex' sets a+b+c=3, 'partial' sets c=1.
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
        roots = _findroot_helper.findroot(poly, grid, method, standardize_method = standardize_method)
        roots = [r.approximate() for r in roots]
        roots = [r for r in roots if r.is_nontrivial]
        roots = _findroot_helper._remove_duplicative_roots(roots)

        # compute roots on the border
        roots += _findroot_helper._findroot_border(poly)

        # compute roots on the symmetric axis
        roots += _findroot_helper._findroot_symmetric(poly, skip_border = True)


    vals = [(root.eval(poly), i) for i, root in enumerate(roots)]
    vals = sorted(vals)
    if len(vals) > most:
        vals = vals[:most]

    reg = max(abs(i) for i in poly.coeffs()) * deg(poly) * 5e-9

    strict_roots = [roots[i] for v, i in vals if v < reg]
    roots = [roots[i] for v, i in vals]

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
            method: str = 'nsolve',
            standardize_method: str = 'simplex',
        ) -> List[Root]:
        """
        Find roots of a polynomial.

        Parameters
        ----------
        poly : sympy.Poly
            The polynomial to be solved.
        grid : GridPoly
            The grid to be used. If None, a new grid will be generated.
        method : str
            The method to be used. 'nsolve' uses sympy.nsolve.
        standardize_method : str
            The method to be used to de-homogenize the polynomial. 'simplex' sets a+b+c=3, 'partial' sets c=1.

        Returns
        ----------
        roots : list of tuple
            The roots of the polynomial, each in the form (a, b), indicating (a, b, 1).
        """
        extrema = cls._initial_guess(grid)
        roots = cls.get_method(method)(poly, initial_guess = extrema)

        return roots

    @classmethod
    def available_methods(cls):
        return ['nsolve', 'bfgs', 'l-bfgs-b', 'cg', 'trust-constr']

    @classmethod
    def get_method(cls, name: str) -> Callable:
        if name not in cls.available_methods():
            raise ValueError(f'Unknown method {name}. Please use f{__class__}.available_methods() to check.')
        return getattr(cls, '_findroot_' + name.replace('-', '_'))

    @classmethod
    def _standardize_poly(cls, poly: sp.Poly, standardize_method: str = 'simplex') -> sp.Expr:
        """
        De-homogenize the polynomial. 
        """
        a, b, c = sp.symbols('a b c')
        if standardize_method == 'simplex':
            poly = poly.as_expr().subs(c, 3 - a - b).expand()
        elif standardize_method == 'partial':
            poly = poly.subs(c, 1).as_expr()
        else:
            raise ValueError(f'Unknown standardize method {standardize_method}.')
        return poly

    @classmethod
    def _as_xy(cls, root: Union[Root, List[Root]], standardize_method: str = 'simplex') -> Tuple[float, float]:
        """
        Given a Root object, convert it to (x, y) according to the standardization method
        to de-homogenize.
        """
        only_one = isinstance(root, Root)
        if only_one:
            roots = [root]
        else:
            roots = root

        xys = []
        for root in roots:
            if standardize_method == 'simplex':
                a, b, c = root
                s = (a + b + c) / 3
                xy = (a/s, b/s)
            elif standardize_method == 'partial':
                xy = root[0], root[1]
            xy = (float(xy[0]), float(xy[1]))
            xys.append(xy)
        if only_one:
            xys = xys[0]
        return xys

    @classmethod
    def _as_root(cls, xy: Tuple[float, float], standardize_method: str = 'simplex') -> Root:
        """
        Given (x, y), restore it to a Root object according to the standardization method.
        """
        only_one = len(xy) == 2 and not hasattr(xy[0], '__iter__')
        if only_one:
            xys = [xy]
        else:
            xys = xy

        roots = []
        for x, y in xys:
            if standardize_method == 'simplex':
                roots.append(Root((x, y, 3 - x - y)))
            elif standardize_method == 'partial':
                roots.append(Root((x, y, 1)))
        return roots

    @classmethod
    def _remove_duplicative_roots(cls, roots: List[Root], prec: int = 4) -> List[Root]:
        """
        Remove duplicative roots if they are close enough < 10^(-prec).
        """
        roots = [r.standardize(cyc = True, inplace = True) for r in roots]
        inds = dict(((r[0].n(prec), r[1].n(prec)), i) for i, r in enumerate(roots)).values()
        return [roots[i] for i in inds]        
        return roots

    @classmethod
    def _initial_guess(cls, grid: GridPoly) -> List[Root]:
        """
        Given grid coordinate and grid value, search for local minima.    
        """
        extrema = grid.local_minima(filter_nontrivial=True)

        return extrema

    @classmethod
    def _findroot_border(cls, poly: sp.Poly) -> List[Root]:
        """
        Return roots on the border of a polynomial.
        """
        roots = []
        a = sp.Symbol('a')
        rep = [_[0] if len(_) else 0 for _ in poly.rep.rep[-1]]
        poly = sp.Poly.from_list(rep, a)
        poly_diff = poly.diff(a)
        poly_diff2 = poly_diff.diff(a)
        try:
            for r in nroots(poly_diff, method = 'factor', real = True, nonnegative = True):
                if poly_diff2(r) >= 0:
                    roots.append(Root((r, 1, 0)))
        except:
            pass
        return roots

    @classmethod
    def _findroot_symmetric(cls, poly: sp.Poly, skip_border: bool = False) -> List[Root]:
        """
        Return roots on the symmetric axis of a polynomial.
        If skip_border is True, the root (0,1,1) is not included.
        """
        roots = []
        a = sp.Symbol('a')
        rep = [sum(sum(__) for __ in _) for _ in poly.rep.rep]
        poly = sp.Poly.from_list(rep, a)
        poly_diff = poly.diff(a)
        poly_diff2 = poly_diff.diff(a)
        try:
            for r in nroots(poly_diff, method = 'factor', real = True, nonnegative = True):
                if r == 0 and skip_border:
                    continue
                if poly_diff2(r) >= 0:
                    roots.append(Root((r, 1, 1)))
        except:
            pass
        return roots

    @classmethod
    def _findroot_nsolve(cls, poly, initial_guess = [], standardize_method = 'simplex'):
        """
        Numerically find roots with sympy nsolve.
        """
        roots = []

        poly = cls._standardize_poly(poly, standardize_method = standardize_method)
        poly_diffa = poly.diff('a')
        poly_diffb = poly.diff('b')

        for e in cls._as_xy(initial_guess, standardize_method = standardize_method):
            try:
                roota, rootb = sp.nsolve(
                    (poly_diffa, poly_diffb),
                    sp.symbols('a b'),
                    e
                )
                roots.append((roota, rootb))
            except:
                pass

        return cls._as_root(roots, standardize_method = standardize_method)

    @classmethod
    def _findroot_bfgs(cls, poly, initial_guess = [], standardize_method = 'simplex'):
        return cls._findroot_scipy(poly, method = 'bfgs', initial_guess = initial_guess, standardize_method = standardize_method)

    @classmethod
    def _findroot_l_bfgs_b(cls, poly, initial_guess = [], standardize_method = 'simplex'):
        return cls._findroot_scipy(poly, method = 'l-bfgs-b', initial_guess = initial_guess, standardize_method = standardize_method)

    @classmethod
    def _findroot_cg(cls, poly, initial_guess = [], standardize_method = 'simplex'):
        return cls._findroot_scipy(poly, method = 'cg', initial_guess = initial_guess, standardize_method = standardize_method)

    @classmethod
    def _findroot_trust_constr(cls, poly, initial_guess = [], standardize_method = 'simplex'):
        return cls._findroot_scipy(poly, method = 'trust-constr', initial_guess = initial_guess, standardize_method = standardize_method)

    @classmethod
    def _findroot_scipy(cls, poly, method = 'bfgs', initial_guess = [], standardize_method = 'simplex'):
        if standardize_method == 'simplex':
            f = lambda x: poly(x[0]/3, x[1]/3, (3 - x[0] - x[1])/3)
            def early_stop(x):
                if x[0] < -3 or x[1] < -3 or x[0] + x[1] > 6:
                    raise ValueError

        elif standardize_method == 'partial':
            f = lambda x: poly(x[0], x[1], 1)
            def early_stop(x):
                if x[0] < 0 or x[1] < 0:
                    raise ValueError

        roots = []
        for a, b in cls._as_xy(initial_guess, standardize_method = standardize_method):
            try:
                res = minimize(f, (a, b), method = method, tol=1e-6, callback = early_stop)
                if res.success:
                    roots.append((res.x[0], res.x[1]))
            except:
                pass

        return cls._as_root(roots, standardize_method = standardize_method)


def findroot_resultant(poly: sp.Poly) -> List[Root]:
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
    if len(poly.gens) != 3 or not poly.is_homogeneous:
        return []
    return _findroot_helper_resultant._findroot_resultant_ternary(poly)



class _findroot_helper_resultant():
    """
    Helper class for findroot using resultant method.
    """
    @classmethod    
    def _findroot_resultant_ternary(cls, poly):
        """See findroot_resultant."""
        a, b, c = poly.gens
        is_cyc = verify_hom_cyclic(poly)[1]
        poly = poly.subs(c,1) # -a-b).as_poly(a,b)
        parts = poly.factor_list()[1]
        roots = []

        findroot_func = cls._findroot_resultant_ternary_irreducible

        for part in parts:
            poly, multiplicity = part
            if poly.total_degree() == 1:
                continue
            # elif multiplicity % 2 == 0:
            #     continue
            elif poly.total_degree() == 2 and is_cyc:
                if poly(1,1) == 0:
                    roots.append(RootRational((1,1,1)))
                continue
            else:
                roots.extend(findroot_func(poly, is_cyc = is_cyc))


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
            # if mult_at_origin > 1 and root.root[0] == 1 and root.root[1] == 1 and root.root[2] == 1:
            #     root.multiplicity = mult_at_origin
            roots_clear.append(root)
        return roots_clear

    @classmethod
    def _ternary_cyclic_root_pairs(cls, poly, factors, prec = 20, tolerance = 1e-12):
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
        factors : List[Tuple[sympy.Poly, int]]
            The minimal polynomials of the second variable. It is resultant.factor_list()[1].
        prec : int
            The precision of solving roots of each factor numerically.
        tolerance : float
            The tolerance of a pair (a,b) is recognized as a root pair.
        is_cyc : bool
            Whether the polynomial is cyclic.

        Returns
        ----------
        pairs : List[Tuple[Tuple[float, float], Tuple[int, int]]]
            The root pairs of the polynomial. Each pair is in the form ((a, b), (i, j)),
            where (a, b) is the root pair and (i, j) is the index of the corresponding factors.
        """
        pairs = []
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
    def _ternary_acyclic_root_pairs(cls, poly, factors1, factors2, prec = 20, tolerance = 1e-12):
        """
        Find the numerical root pairs of a polynomial using the resultant method.
        The polynomial should be of two variables.

        Parameters
        ----------
        poly : sympy.Poly
            The polynomial of two variables.
        factors1 : List[Tuple[sympy.Poly, int]]
            The minimal polynomials of the first variable. It is resultant.factor_list()[1].
        factors2 : List[Tuple[sympy.Poly, int]]
            The minimal polynomials of the second variable. It is resultant.factor_list()[1].
        prec : int
            The precision of solving roots of each factor numerically.
        tolerance : float
            The tolerance of a pair (a,b) is recognized as a root pair.

        Returns
        ----------
        pairs : List[Tuple[Tuple[float, float], Tuple[int, int]]]
            The root pairs of the polynomial. Each pair is in the form ((a, b), (i, j)),
            where (a, b) is the root pair and (i, j) is the index of the corresponding factors.
        """
        pairs = []
        all_roots1 = []
        all_roots2 = []
        for i, factor_ in enumerate(factors1):
            for r in sp.polys.nroots(factor_[0], n = prec):
                if r.is_real:
                    all_roots1.append((r, i))
        for i, factor_ in enumerate(factors2):
            for r in sp.polys.nroots(factor_[0], n = prec):
                if r.is_real:
                    all_roots2.append((r, i))

        for a_, i in all_roots1:
            for b_, j in all_roots2:
                v = poly(a_, b_)
                if abs(v) < tolerance:
                    pairs.append(((a_, b_), (i, j)))
        return pairs

    @classmethod
    def _from_pairs_to_roots(cls, pairs, factors1, factors2, is_cyc = True):
        roots = []
        for (a_, b_), (i, j) in pairs:
            fx, fy = factors1[i][0], factors2[j][0]
            # reverse fx
            fx = sp.Poly(fx.all_coeffs()[::-1], fy.gens[0]) if is_cyc else fx
            if fx.degree() == 0:
                a_ = sp.S(0)
            else:
                a_ = find_nearest_root(fx, a_, method = 'rootof')
            b_ = find_nearest_root(fy, b_, method = 'rootof')
            root = RootAlgebraic((a_, b_, 1))
            roots.append(root)
        return roots

    @classmethod
    def _findroot_resultant_ternary_irreducible(cls, poly, is_cyc = True):
        """
        Find root of a 2-var polynomial with respect to a, b using the method of 
        resultant. The polynomial should be irreducible.

        Parameters
        ----------
        poly : sympy.Poly
            The polynomial of two variables. It should be irreducible.
        is_cyc : bool
            Whether the polynomial is cyclic.

        Returns
        -------
        roots: List[RootAlgebraic]
            The roots of the polynomial.    
        """
        a, b = poly.gens
        grad = poly.diff(a)

        if is_cyc:
            res = sp.resultant(poly, grad, a).as_poly(b)
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

            pairs = cls._ternary_cyclic_root_pairs(poly, factors)
            roots = cls._from_pairs_to_roots(pairs, factors, factors, is_cyc = is_cyc)

        else:
            resb, resa = sp.resultant(poly, grad, a).as_poly(b), sp.resultant(poly, grad, b).as_poly(a)
            factors1 = set(sp.polys.gcd(resa, resa.diff(a)).factor_list()[1])
            factors2 = set(sp.polys.gcd(resb, resb.diff(b)).factor_list()[1])
            factors1 = list(factors1)
            factors2 = list(factors2)

            pairs = cls._ternary_acyclic_root_pairs(poly, factors1, factors2)
            roots = cls._from_pairs_to_roots(pairs, factors1, factors2, is_cyc = is_cyc)

        return roots