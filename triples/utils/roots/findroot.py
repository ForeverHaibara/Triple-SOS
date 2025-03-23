from collections import deque
from itertools import product
from typing import Union, Callable, List, Optional, Tuple, Any

import numpy as np
import sympy as sp
from sympy.core import Symbol, Expr
from sympy.polys.polytools import Poly
from sympy.plotting.experimental_lambdify import vectorized_lambdify
from scipy.optimize import minimize

from .polysolve import nroots
from .rationalize import rationalize
from .roots import Root
from ..expression import Coeff

def _is_cyclic(poly) -> bool:
    return Coeff(poly).is_cyclic()

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


def _compute_hessian(f: Expr, vars: Tuple[Symbol, Symbol] = None, vectorized = True):
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
    # TODO: DEPRECATE IT?
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
        poly: Poly,
        most: int = 5,
        grid: Optional[Any] = None,
        method: str = 'nsolve',
        standardize_method: str = 'partial',
        verbose: bool = False,
        with_tangents: Union[bool, Callable] = False,
    ) -> List[Root]:
    """
    Find the possible local minima of a 3-var cyclic polynomial by gradient descent and guessing. 
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
    roots : list of Root
        The searched roots of the polynomial.
    """
    if not isinstance(poly, Poly) or len(poly.gens) != 3:
        # not implemented
        return []

    if grid is None:
        from ...gui.grid import GridRender
        grid = GridRender.render(poly, with_color=False)

    roots = []
    if len(poly.gens) == 3 and (poly.domain in (sp.polys.ZZ, sp.polys.QQ, sp.polys.RR)):
        is_cyc = _is_cyclic(poly)
        roots = _findroot_helper.findroot(poly, grid, method, standardize_method = standardize_method)
        roots = [r.approximate() if hasattr(r, 'approximate') else r for r in roots]
        roots = [r for r in roots if r.is_nontrivial]
        roots = _findroot_helper._remove_duplicative_roots(roots, cyc = is_cyc)

        # compute roots on the border
        roots += _findroot_helper._findroot_border(poly, cyc = is_cyc)

        # compute roots on the symmetric axis
        roots += _findroot_helper._findroot_symmetric(poly, skip_border = True)


    vals = [(root.eval(poly), i) for i, root in enumerate(roots)]
    vals = sorted(vals)
    if len(vals) > most:
        vals = vals[:most]

    reg = max(abs(i) for i in poly.coeffs()) * poly.total_degree() * 5e-9

    strict_roots = [roots[i] for v, i in vals if v < reg]
    roots = [roots[i] for v, i in vals]

    if verbose:
        print('Tolerance =', reg, '\nStrict Roots =', strict_roots,'\nNormal Roots =',
                list(set(roots) ^ set(strict_roots)))

    # rootsinfo = RootsInfo(
    #     poly = poly,
    #     roots = roots,
    #     strict_roots = strict_roots,
    #     reg = reg,
    #     with_tangents = with_tangents,
    # )

    return roots # info


class _findroot_helper():
    """
    Helper class for findroot. It stores multiple root-finding functions.
    """
    @classmethod
    def findroot(cls, 
            poly: Poly,
            grid: Any, #GridPoly, 
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
    def _destd_poly(cls, poly: Poly, standardize_method: str = 'simplex') -> Tuple[Expr, List[Symbol]]:
        """
        De-homogenize the polynomial and convert it to a sympy expression.
        """
        gens = poly.gens
        nvars = len(gens)
        if standardize_method == 'simplex':
            gensub = nvars - sp.Add(*gens[:-1])
            poly = poly.as_expr().subs(gens[-1], gensub).expand()
        elif standardize_method == 'partial':
            poly = poly.subs(gens[-1], 1).as_expr()
        else:
            raise ValueError(f'Unknown standardize method {standardize_method}.')
        return poly, gens[:-1]

    @classmethod
    def _destd_root(cls, root: Union[Root, List[Root]], standardize_method: str = 'simplex') -> Tuple[float, float]:
        """
        Dehomogenize a root or multiple roots.
        """
        only_one = isinstance(root, Root)
        if only_one:
            roots = [root]
        else:
            roots = root

        nonstd_roots = []
        for root in roots:
            if standardize_method == 'simplex':
                s = sum(root) / len(root)
                nonstd_root = tuple(i/s for i in root[:-1])
            elif standardize_method == 'partial':
                nonstd_root = tuple(root[:-1])
            nonstd_root = tuple(float(i) for i in nonstd_root)
            nonstd_roots.append(nonstd_root)
        if only_one:
            nonstd_roots = nonstd_roots[0]
        return nonstd_roots

    @classmethod
    def _std_root(cls, nonstd_root: Union[Tuple[float, ...], List[Tuple[float, ...]]], standardize_method: str = 'simplex') -> Root:
        """
        Homogenize a dehomogenized root or multiple roots.
        """
        only_one = len(nonstd_root) and not hasattr(nonstd_root[0], '__iter__')
        if only_one:
            nonstd_roots = [nonstd_root]
        else:
            nonstd_roots = nonstd_root

        roots = []
        for r in nonstd_roots:
            if standardize_method == 'simplex':
                nvars = len(r) + 1
                roots.append(Root(r + (nvars - sum(r),)))
            elif standardize_method == 'partial':
                roots.append(Root(r + (1,)))
        if only_one:
            roots = roots[0]
        return roots

    @classmethod
    def _remove_duplicative_roots(cls, roots: List[Root], prec: int = 4, cyc: bool = True) -> List[Root]:
        """
        Remove duplicative roots if they are close enough < 10^(-prec).
        """
        roots = [r.standardize(cyc = cyc, inplace = True) for r in roots]
        inds = dict((tuple(j.n(prec) for j in r), i) for i, r in enumerate(roots)).values()
        return [roots[i] for i in inds]
        return roots

    @classmethod
    def _initial_guess(cls, grid) -> List[Root]:
        """
        Given grid coordinate and grid value, search for local minima.    
        """
        extrema = grid.local_minima(filter_nontrivial=True, cyc=_is_cyclic(grid.poly))

        return extrema

    @classmethod
    def _findroot_border(cls, poly: Poly, cyc: bool = True) -> List[Root]:
        """
        Return roots on the border of a 3-var polynomial.
        """
        roots = []
        if cyc:
            a = poly.gens[0]
            rep = [_[0] if len(_) else 0 for _ in poly.rep.to_list()[-1]]
            poly = Poly.from_list(rep, a)
            poly_diff = poly.diff(a)
            poly_diff2 = poly_diff.diff(a)
            try:
                for r in nroots(poly_diff, method = 'factor', real = True, nonnegative = True):
                    if poly_diff2(r) >= 0:
                        roots.append(Root((r, 1, 0)))
            except:
                pass
        else:
            gens = poly.gens
            nvars = len(gens)
            subs = [1] + [0] * (nvars - 2)
            gens2 = deque(gens)
            for i in range(nvars):
                a = gens2.popleft()
                poly0 = poly.subs(dict(zip(gens2, subs))).as_poly(a)
                poly_diff = poly0.diff(a)
                poly_diff2 = poly_diff.diff(a)
                try:
                    for r in nroots(poly_diff, method = 'factor', real = True, nonnegative = True):
                        if poly_diff2(r) >= 0:
                            root = [0] * nvars
                            root[i] = r
                            root[(i+1)%nvars] = 1
                            roots.append(Root(root))
                except Exception as e:
                    pass
                gens2.append(a)
        return roots

    @classmethod
    def _findroot_symmetric(cls, poly: Poly, skip_border: bool = False) -> List[Root]:
        """
        Return roots on the symmetric axis of a 3-var polynomial.
        If skip_border is True, the root (0,1,1) is not included.
        """
        roots = []
        a = poly.gens[0]
        rep = [sum(sum(__) for __ in _) for _ in poly.rep.to_list()]
        poly = Poly.from_list(rep, a)
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

        poly, gens = cls._destd_poly(poly, standardize_method = standardize_method)
        poly_diffs = [poly.diff(gen) for gen in gens]

        for e in cls._destd_root(initial_guess, standardize_method = standardize_method):
            try:
                extrema = sp.nsolve(poly_diffs, gens, e)
                roots.append(tuple(list(extrema)))
            except:
                pass

        return cls._std_root(roots, standardize_method = standardize_method)

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
        nvars = len(poly.gens)
        def _get_f_and_early_stop(standardize_method: str):
            if standardize_method == 'simplex':
                def f(x):
                    x = [i/nvars for i in x]
                    return poly(*x, 1 - sum(x))
                def early_stop(x):
                    if any(i < -nvars for i in x) or 1 - sum(x) < -nvars:
                        raise ValueError
            elif standardize_method == 'partial':
                def f(x):
                    return poly(*x, 1)
                def early_stop(x):
                    if any(i < 0 for i in x):
                        raise ValueError
            return f, early_stop
        f, early_stop = _get_f_and_early_stop(standardize_method)

        roots = []
        for root in cls._destd_root(initial_guess, standardize_method = standardize_method):
            try:
                res = minimize(f, root, method = method, tol=1e-6, callback = early_stop)
                if res.success:
                    roots.append(tuple(res.x))
            except:
                pass

        return cls._std_root(roots, standardize_method = standardize_method)


def findroot_resultant(poly: Poly) -> List[Root]:
    """
    Find the roots of a 3-var homogeneous polynomial using the resultant method. 
    This is essential in SDP SOS, because we need to construct exact subspace
    constrainted by roots of the polynomial to perform SDP on a full-rank manifold.

    The method does not guarantee to find all roots, but find at least one in 
    each Galois group. For instance, if (1,1,sqrt(2)) is a root, then (1,1,-sqrt(2))
    must be a root as well, due to the principle of minimal polynomial.
    Then it is possible for us to return (1,1,-sqrt(2)), although it is not
    positive. This seems to miss some roots, but it is not a problem in SDP SOS,
    because the spanned subspace algorithm includes the permutation of roots.

    Parameters
    ----------
    poly : sympy.Poly
        The polynomial to find roots of.

    Returns
    ----------
    roots : List[Root]
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
        is_cyc = _is_cyclic(poly)
        poly0 = poly
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
                    roots.append(Root((1,1,1)))
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

        if not is_cyc:
            roots += cls._ternary_acyclic_border_roots(poly0)

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
    def _ternary_acyclic_border_roots(cls, poly):
        a, b, c = poly.gens
        subs = [{b: 1, c: 0}, {c: 1, a: 0}, {a: 1, b: 0}]
        all_roots = []
        for i, sub in enumerate(subs):
            poly0 = poly.subs(sub)
            poly0diff = poly0.diff(poly.gens[i])
            roots = sp.polys.gcd(poly0, poly0diff).all_roots()
            for root in roots:
                if root.is_real:
                    r = [0, 0, 0]
                    r[i] = root
                    r[(i+1)%3] = 1
                    r[(i+2)%3] = 0
                    all_roots.append(Root(tuple(r)))
        return all_roots

    @classmethod
    def _from_pairs_to_roots(cls, pairs, factors1, factors2, is_cyc = True):
        roots = []
        for (a_, b_), (i, j) in pairs:
            fx, fy = factors1[i][0], factors2[j][0]
            # reverse fx
            fx = Poly(fx.all_coeffs()[::-1], fy.gens[0]) if is_cyc else fx
            if fx.degree() == 0:
                a_ = sp.S(0)
            else:
                a_ = find_nearest_root(fx, a_, method = 'rootof')
            b_ = find_nearest_root(fy, b_, method = 'rootof')
            root = Root((a_, b_, 1))
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
        roots: List[Root]
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
                factors_border.add((Poly([1,0], b), len(factors_border)))
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