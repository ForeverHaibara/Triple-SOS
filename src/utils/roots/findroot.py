from itertools import product

import numpy as np
import sympy as sp

from .rationalize import rationalize
from .rootsinfo import RootsInfo
from .tangents import root_tangents
from .grid import GridRender
from ..polytools import deg


def findbest(choices, func, init_choice = None, init_val = 2147483647):
    """
    Find the best choice of all choices to minimize the function.
    
    Return the best choice and the corresponding value.
    """

    best_choice = init_choice
    val = init_val
    for choice in choices:
        val2 = func(choice)
        if val2 < val:
            val = val2
            best_choice = choice
    return best_choice , val


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
        print('Starting Search From', best_choice[1:], ' f = ', best_choice[0])
    if v <= 0:
        return {x: a, y: b}

    if v > 0:
        a = a * 1.0
        b = b * 1.0
        da = discriminant.diff(x)
        db = discriminant.diff(y)
        da2 = da.diff(x)
        dab = da.diff(y)
        db2 = db.diff(y)
        # x =[a',b'] <- x - inv(nabla)^-1 @ grad
        for i in range(20):
            lasta , lastb = a , b
            da_  = da(a,b)
            db_  = db(a,b)
            da2_ = da2(a,b)
            dab_ = dab(a,b)
            db2_ = db2(a,b)
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
        poly,
        most = 5,
        grid = None,
        method = 'newton',
        verbose = False,
        with_tangents = True,
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
    if grid is None:
        grid = GridRender.render(poly, method='integer', with_color=False)

    grid_coor, grid_value = grid.grid_coor, grid.grid_value
    extrema = _findroot_initial_guess(grid_coor, grid_value)
    
    if method == 'nsolve':
        result_roots = _findroot_nsolve(poly, initial_guess = extrema)
    elif method == 'newton':
        result_roots = _findroot_newton(poly, initial_guess = extrema)

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

    if verbose:
        print('Tolerance =', reg, '\nStrict Roots =', strict_roots,'\nNormal Roots =',
                list(set(result_roots) ^ set(strict_roots)))

    rootsinfo = RootsInfo(
        poly = poly,
        roots = result_roots,
        strict_roots = strict_roots,
        reg = reg,
        with_tangents = with_tangents,
    )

    return rootsinfo


def _findroot_initial_guess(grid_coor, grid_value):
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

def _findroot_nsolve(
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



def _findroot_newton(
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