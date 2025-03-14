"""
This module provides heuristic functions to optimize a polynomial (algebraically)
with inequality and equality constraints. Directly solving the KKT system might
be EXTREMELY SLOW and fail for nonzero dimensional variety. The functions here
uses a heuristic approach to compute the extrema of the polynomial.
"""
from functools import wraps, partial
from itertools import product
from typing import List, Dict, Tuple, Union

import sympy as sp
from sympy import Symbol, Float, Expr, Poly, Rational
from sympy.core.sorting import default_sort_key
from sympy.polys.polyerrors import BasePolynomialError, PolificationFailed, GeneratorsNeeded
from sympy.polys.rootoftools import ComplexRootOf as CRootOf

from .polysolve import univar_realroots, solve_poly_system_crt, PolyEvalf, _filter_trivial_system

def polysubs(poly: Poly, subs: Dict[Symbol, Expr], symbols: List[Symbol]) -> Poly:
    """Substitute the symbols in a polynomial with the given values."""
    if len(subs) == 0:
        return poly
    poly = poly.as_expr().xreplace(subs)
    if len(symbols) == 0:
        return poly.expand()
    return poly.as_poly(*symbols)

def _get_minimal(poly: Poly, points: List[Tuple[CRootOf]]) -> List[Tuple[CRootOf]]:
    """
    Get the minimal points in the list of points up to `eps` precision.
    Points are evaluated at precision `n`.
    """
    if len(points) == 0:
        return points
    original_points = points

    pevalf = PolyEvalf()
    eps = Float(f'1e-{pevalf.prec//3}')
    f0 = pevalf.polyeval(poly, points[0])
    points = [points[0]]
    for point in original_points[1:]:
        f = pevalf.polyeval(poly, point)
        if f < f0 - eps:
            f0 = f
            points = [point]
        elif f < f0 + eps:
            f0 = min(f0, f)
            points.append(point)
    return points

def _checkineq_decorator(func):
    """TODO: is it necessary to be a decorator?"""
    @wraps(func)
    def wrapper(poly, ineq_constraints, eq_constraints, symbols, *args, **kwargs):
        points = func(poly, ineq_constraints, eq_constraints, symbols, *args, **kwargs)
        if len(points) == 0:
            return points
        if len(symbols) == 0:
            return points if all(_ >= 0 in (sp.true, True) for _ in ineq_constraints) else []
        ineq_constraints = [_.as_poly(symbols) for _ in ineq_constraints]
        pevalf = PolyEvalf()
        points = [p for p in points if all(pevalf.polysign(_, p) >= 0 for _ in ineq_constraints)]
        return points
    return wrapper

def kkt(
        f: Union[Expr, Poly],
        ineq_constraints: List[Union[Expr, Poly]] = [],
        eq_constraints: List[Union[Expr, Poly]] = [],
        as_poly: bool = True,
        fj: bool = False,
        with_self: bool = False
    ) -> Tuple[List[Poly], Tuple[Tuple[Symbol], Tuple[Symbol], Tuple[Symbol], Tuple[Symbol]]]:
    """
    Compute the Karush-Kuhn-Tucker system of a function with inequality and equality constraints.

    Parameters
    ----------
    f : sympy.Expr
        The function to be optimized.
    ineq_constraints : List[sympy.Expr]
        The inequality constraints, G1, G2, ... (>= 0).
    eq_constraints : List[sympy.Expr]
        The equality constraints. H1, H2, ... (== 0).
    as_poly : bool
        Whether to return the system as polynomials. Defaults to True.
    fj : bool
        Whether to use the Fritz John conditions. Defaults to False.
    with_self : bool
        Whether to include f in the system (i.e. f = 0). Defaults to False.

    Returns
    ----------
    system : List[sympy.Poly]
        The KKT system.
    (symbs0, lambs, mus, symbs) : Tuple[Tuple[sympy.Symbol], Tuple[sympy.Symbol], Tuple[sympy.Symbol], Tuple[sympy.Symbol]]
        The symbols used in the system.
        symbs0 are the original variables,
        lambs are the Lagrange multipliers for ineq_constraints,
        mus are the Lagrange multipliers for eq_constraints,
        symbs = symbs0 + lambs + mus.
        Also, KKT / FJ condition requires lambs >= 0.

    Examples
    ----------
    >>> x, y, z = sp.symbols('x y z')
    >>> kkt(2*x + 3*y, [x], [x**2 + y**2 - 1], as_poly = False) # doctest: +NORMALIZE_WHITESPACE
    ([-_\lambda_0 + 2*_\mu_0*x + 2, 2*_\mu_0*y + 3, _\lambda_0*x, x**2 + y**2 - 1],
        ((x, y), (_\lambda_0,), (_\mu_0,), (x, y, _\lambda_0, _\mu_0)))

    For some cases, the optimal solution does not satisfy the KKT conditions.
    The following is an example where the optimal solution is x = 0.

    >>> kkt(x, [], [x**2]) # doctest: +NORMALIZE_WHITESPACE
    ([Poly(2*x*_\mu_0 + 1, x, _\mu_0, domain='ZZ'),
     Poly(x**2, x, _\mu_0, domain='ZZ')],
     ((x,), (), (_\mu_0,), (x, _\mu_0)))

    However, it satisfies the Fritz John conditions.

    >>> system = kkt(x, [], [x**2], fj = True)
    >>> system[0] # doctest: +NORMALIZE_WHITESPACE
    [Poly(2*x*_\mu_0 + _\lambda_0, x, _\lambda_0, _\mu_0, domain='ZZ'),
     Poly(x**2, x, _\lambda_0, _\mu_0, domain='ZZ')]

    >>> from sympy import solve
    >>> solve(system[0], system[1][-1], dict=True) # doctest: +SKIP
    [{_\lambda_0: 0, x: 0}]
    """
    f = sp.sympify(f)
    symbs = f.gens if hasattr(f, 'gens') else tuple(sorted(f.free_symbols, key=lambda x: x.name))
    symbs = symbs + tuple(set.union(set(), 
                *[_.free_symbols for _ in ineq_constraints],
                *[_.free_symbols for _ in eq_constraints]) - set(symbs))
    symbs0 = symbs

    fj = 1 if fj else 0
    lambs = tuple(sp.Dummy('\\lambda_%d' % i) for i in range(len(ineq_constraints) + fj))
    mus = tuple(sp.Dummy('\\mu_%d' % i) for i in range(len(eq_constraints)))
    f0 = f * lambs[0] if fj else f
    lag = f0 - sum(l*ineq for l, ineq in zip(lambs[fj:], ineq_constraints)) \
            + sum(m*eq for m, eq in zip(mus, eq_constraints))

    stationarity = [lag.diff(_) for _ in symbs]
    complementarity = [l*ineq for l, ineq in zip(lambs[fj:], ineq_constraints)]

    system = stationarity + complementarity + eq_constraints
    symbs = symbs + lambs + mus

    if as_poly:
        system = [_.as_poly(symbs) for _ in system]
    if with_self:
        system.append(f.as_poly(symbs) if as_poly else f)
    return system, (symbs0, lambs, mus, symbs)



def _solve_2vars_zero_extrema(poly: Poly, symbols: Symbol) -> List[Tuple[CRootOf]]:
    """
    Solve the system poly = poly.diff(x) = poly.diff(y) = 0 via resultants.
    It exploits the fact that the roots have multiplicity 2 and finds them
    by gcd(res, res.diff()), which avoids factorization to achieve better performance.
    """
    x, y = symbols
    dx, dy = poly.diff(x), poly.diff(y)
    res0 = sp.resultant(poly, dy, y).as_poly(x)
    res0 = sp.gcd(res0, res0.diff(x))
    roots1 = univar_realroots(res0, x)

    if len(roots1) == 0:
        return []

    if all(isinstance(_, Rational) for _ in roots1):
        # solve the other root by direct substitution
        roots = []
        for root1 in roots1:
            poly2 = poly.eval(y, root1)
            roots2 = univar_realroots(poly2, x)
            for root2 in roots2:
                roots.append((root1, root2))
        return roots

    # compute the resultant of the other variable
    res1 = sp.resultant(poly, dx, x).as_poly(y)
    res1 = sp.gcd(res1, res1.diff(y))
    roots2 = univar_realroots(res1, y)

    pevalf = PolyEvalf()
    roots = []
    for root1 in roots1:
        for root2 in roots2:
            # test which is the root
            if pevalf.polysign(poly, (root1, root2)) == 0:
                roots.append((root1, root2))
    return roots


@_checkineq_decorator
def _optimize_by_eq_kkt(poly, ineq_constraints, eq_constraints, symbols,
        max_different = 2):
    """
    Optimize a polynomial with given equality constraints using the KKT system.
    All inequality constraints are assumed to be inactive and not used in the KKT system.
    Inequality constraints are checked to filter out invalid solutions.

    To include active inequality constraints, put them into the equality constraints
    or use the upstream `_optimize_by_ineq_comb` instead.
    """
    if len(symbols) == 0:
        return [tuple()]
    elif len(symbols) > max_different:
        return []

    # filter out trivial constraints
    eq_constraints = _filter_trivial_system(eq_constraints)
    if eq_constraints is None: # inconsistent
        return []

    if len(symbols) == 1:
        # Only one var. if there exist equality constraints,
        # then it should be solved directly.
        if len(eq_constraints):
            eqgcd = eq_constraints[0].as_poly(symbols[0], extension=True)
            for eq in eq_constraints[1:]:
                eqgcd = sp.gcd(eqgcd, eq.as_poly(symbols[0], extension=True))
            sol = [(root,) for root in univar_realroots(eqgcd, symbols[0])]
        else:
            # Solve by derivative.
            sol = [(root,) for root in univar_realroots(poly.diff(symbols[0]), symbols[0])]
        return sol

    elif len(symbols) == 2:
        x, y = symbols
        poly = poly.as_poly(x, y, extension=True)
        if len(eq_constraints) >= 2:
            sol = solve_poly_system_crt(eq_constraints, [x, y])
            return sol # either inconsistent or has finite solutions
            # TODO: what about nonzero dimensional variety?

        sol = []
        if poly in eq_constraints:
            # This often happens when solving for the equality case of a nonnegative poly.
            # Using the special solver saves time.
            _sol = _solve_2vars_zero_extrema(poly, symbols)
            pevalf = PolyEvalf()
            for point in _sol:
                if all(poly == eq or pevalf.polysign(poly, point) == 0 for eq in eq_constraints):
                    sol.append(point)
            return sol

        # 
        sol = solve_poly_system_crt(eq_constraints + [poly.diff(x), poly.diff(y)], [x, y])
        for eq in eq_constraints:
            sol.extend(solve_poly_system_crt(
                eq_constraints + [poly.diff(x)*eq.diff(y)-poly.diff(y)*eq.diff(x)], [x, y]))
        sol = set(sol)
        return sol

    # else:
    # MIGHT BE VERY SLOW
    kkt_sys, (_, __, ___, kkt_symbols) = kkt(poly, [], eq_constraints)
    sol = solve_poly_system_crt(kkt_sys, kkt_symbols)
    symbinds = [kkt_symbols.index(s) for s in symbols]
    sol = [_ for _ in sol if all(v.is_real for v in _)]
    sol = [tuple(_[i] for i in symbinds) for _ in sol]
    return sol

def _eliminate_linear(polys, symbols):
    """Eliminate linear symbols in the eq_constraints."""
    has_eliminated = True
    eliminated = {}

    def is_linear(poly, gen, allow_zero = True):
        """Check whether a polynomial is linear in a symbol and the coeff is constant."""
        d = poly.degree(gen) if gen in poly.gens else 0
        if d <= 0:
            return allow_zero
        if d > 1:
            return False
        genind = poly.gens.index(gen)
        return all(m[genind] != 1 or sum(m) == 1 for m in poly.monoms())

    while has_eliminated:
        has_eliminated = False
        new_eliminated_symbols = []
        new_eliminated_polys = set()
        for symbol in symbols:
            if not all(is_linear(polys[i], symbol, allow_zero=True) for i in new_eliminated_polys):
                continue
            for i, poly in enumerate(polys):
                if (i not in new_eliminated_polys) and is_linear(poly, symbol, allow_zero=False)\
                    and all(is_linear(poly, s, allow_zero=True) for s in new_eliminated_symbols):
                    new_eliminated_symbols.append(symbol)
                    new_eliminated_polys.add(i)
                    break

        # this should be a linear system
        if len(new_eliminated_symbols) == 0:
            break
        else:
            has_eliminated = True
        rest_inds = set(range(len(polys))) - new_eliminated_polys
        new_eliminated_polys = [polys[i] for i in new_eliminated_polys]

        linsys = sp.linear_eq_to_matrix(
            [_.as_expr() for _ in new_eliminated_polys], new_eliminated_symbols)
        sol = sp.linsolve(linsys, *new_eliminated_symbols)
        if sol is sp.S.EmptySet or len(sol) == 0:
            return None, polys
        sol = sol[0] if not isinstance(sol, sp.FiniteSet) else sol.args[0]
        sol = dict(zip(new_eliminated_symbols, sol))

        symbols = [s for s in symbols if s not in new_eliminated_symbols]
        polys = [polys[i] for i in rest_inds]
        polys = [polysubs(_, sol, symbols) for _ in polys]
        if len(symbols) == 0:
            if any(_ != sp.S.Zero for _ in polys):
                return None, polys

        # update eliminated
        for k, v in eliminated.items():
            eliminated[k] = v.xreplace(sol)
        eliminated.update(sol)

    return eliminated, polys

def _restore_solution(points: List[Tuple[CRootOf]], elim: Dict[Symbol, Expr],
        symbols: List[Symbol], symbols2: List[Symbol]) -> List[Tuple[CRootOf]]:
    """
    After solving the reduced system, restore the eliminated variables.
    """
    if len(elim) and len(points):
        newpoints = []
        for point in points:
            substitution = dict(zip(symbols2, point))
            newpoint = [0] * len(symbols)
            for i, p in enumerate(symbols):
                if p in elim:
                    newpoint[i] = elim[p].xreplace(substitution)
                else:
                    newpoint[i] = substitution[p]
            newpoints.append(tuple(newpoint))
        return newpoints
    return points


@_checkineq_decorator
def _optimize_by_ineq_comb(poly: Poly, ineq_constraints: List[Poly], eq_constraints: List[Poly],
        symbols: Symbol, eliminate_func=None, solver=None) -> List[Tuple[CRootOf]]:
    """
    Optimize a polynomial with inequality constraints by considering all possible
    combinations of active inequality constraints. After each dicision of active
    inequality constraints, the new system is first eliminated by `eliminate_func`
    and then passed into the downstream solver `solver`.
    """
    all_points = []
    elim = {}
    for active in product([False, True], repeat=len(ineq_constraints)):
        active_ineq0 = [ineq for ineq, act in zip(ineq_constraints, active) if act]
        if eliminate_func is not None:
            elim, active_ineq = eliminate_func(active_ineq0, symbols)
        if elim is None: # inconsistent
            continue

        if len(elim):
            symbols2 = [s for s in symbols if s not in elim]
            # poly2 = poly.as_expr().xreplace(elim).as_poly(*symbols2)
            poly2 = polysubs(poly, elim, symbols2)
            eq_constraints2 = [polysubs(_, elim, symbols2) for _ in eq_constraints]
            eq_constraints2 = _filter_trivial_system(eq_constraints2)
            if eq_constraints2 is None:
                continue
        else:
            poly2, eq_constraints2, symbols2 = poly, eq_constraints, symbols
        
        points = solver(poly2, {}, active_ineq + eq_constraints2, symbols2)

        # restore the eliminated variables
        points = _restore_solution(points, elim, symbols, symbols2)
        all_points.extend(points)
    return all_points

def _optimize_by_symbol_reduction(poly: Poly, ineq_constraints: List[Poly], eq_constraints: List[Poly],
        symbols: Symbol, max_different=2, solver=None, include_zero=True) -> List[Tuple[CRootOf]]:
    """
    Optimize a polynomial by reducing the number of different variables.
    If the current number of different variables exceeds `max_different`,
    then some of them are set to zero or equal to each other. After
    the reduction, the system is passed into the downstream solver `solver`.

    Larger `max_different` might not lead to more extrema when there
    exists nonzero dimensional variety.
    """
    if len(symbols) <= max_different:
        return solver(poly, ineq_constraints, eq_constraints, symbols)

    all_points = []
    new_symbols = [sp.Dummy('x') for _ in range(max_different)]
    if include_zero:
        new_symbols.append(sp.S.Zero)

    inds = list(range(len(new_symbols)))
    for comb in product(inds, repeat=len(symbols)):
        active_symbols = tuple(new_symbols[i] for i in set(comb))
        if len(active_symbols) < max_different:
            continue

        replacement = dict(zip(symbols, [new_symbols[i] for i in comb]))

        poly2 = polysubs(poly, replacement, active_symbols)
        ineq_constraints2 = [polysubs(_, replacement, active_symbols) for _ in ineq_constraints]
        eq_constraints2 = [polysubs(_, replacement, active_symbols) for _ in eq_constraints]

        points = solver(poly2, ineq_constraints2, eq_constraints2, active_symbols)
        points = _restore_solution(points, replacement, symbols, active_symbols)
        all_points.extend(points)
    return all_points



def optimize_poly(poly: Union[Poly, Expr], ineq_constraints: List[Union[Poly, Expr]] = [], eq_constraints: List[Union[Poly, Expr]] = [],
        symbols: List[Symbol] = None, objective: str = 'min', return_dict: bool = False, max_different: int = 2
    ) -> Union[List[Tuple[CRootOf]], List[Dict[Symbol, CRootOf]]]:
    """
    Algebraically optimize a polynomial with given inequality and equality constraints
    using heuristic methods. It uses incomplete algorithm to balance the efficiency
    and effectiveness.
    If there exists zero dimensional variety, particular solutions will be sampled.

    Parameters
    ----------
    poly : Poly or Expr
        The polynomial to be optimized.
    ineq_constraints : List[Poly or Expr]
        The inequality constraints, G1, G2, ... (>= 0).
    eq_constraints : List[Poly or Expr]
        The equality constraints, H1, H2, ... (== 0).
    symbols : List[Symbol]
        The symbols to optimize on. If None, it is inferred from the polynomial and constraints.
    objective : str
        The objective to optimize. Either 'min' or 'max' or 'all'.
        When 'min', the function returns the global minimizers
        When 'max', the function returns the global maximizers.
        When 'all', the function returns all recognized extrema.
    return_dict : bool
        Whether to return the result as a list of dictionaries.
    max_different : int
        The maximum number of different variables to consider.
        This is a heuristic to accelerate the computation.
        It does not force all extrema to be bounded by this number
        if they are easy to compute.

    Returns
    ----------
    solutions : List[Tuple[CRootOf]] or List[Dict[Symbol, CRootOf]]
        The extrema of the polynomial. If return_dict is True, then
        the extrema are returned as dictionaries.

    Examples
    ----------
    >>> from sympy.abc import a, b, c, x, y
    >>> optimize_poly(a + 2*b, [a, b], [a**2 + b**2 - 1], (a, b), objective='max')
    [(CRootOf(5*a**2 - 1, 1), CRootOf(5*b**2 - 4, 1))]

    >>> optimize_poly((a**2+b**2+c**2)**2-3*(a**3*b+b**3*c+c**3*a)) # doctest: +NORMALIZE_WHITESPACE
    [(1, 1, 1),
     (CRootOf(a**3 - 6*a**2 + 5*a - 1, 0), CRootOf(b**3 - 5*b**2 + 6*b - 1, 1), 1),
     (CRootOf(a**3 - 6*a**2 + 5*a - 1, 1), CRootOf(b**3 - 5*b**2 + 6*b - 1, 0), 1),
     (CRootOf(a**3 - 6*a**2 + 5*a - 1, 2), CRootOf(b**3 - 5*b**2 + 6*b - 1, 2), 1)]
    """
    if symbols is None:
        if isinstance(poly, Poly):
            symbols = poly.gens
        else:
            symbols = set.union(poly.free_symbols,
                *[_.free_symbols for _ in ineq_constraints],
                *[_.free_symbols for _ in eq_constraints])
            symbols = sorted(symbols, key=lambda x: x.name)
    if not (objective in ('min', 'max', 'all')):
        raise ValueError('Objective must be either "min" or "max" or "all".')
    if len(symbols) == 0:
        return []

    def polylize(f, symbols):
        if not isinstance(f, Poly):
            f = Poly(f, *symbols) #, extension=True)
        if f is None:
            raise PolificationFailed({}, f, f)
        return f
    poly = polylize(poly, symbols)
    ineq_constraints = [polylize(ineq, symbols) for ineq in ineq_constraints]
    eq_constraints = [polylize(eq, symbols) for eq in eq_constraints]

    poly0 = poly
    symbols0 = symbols

    # TODO: sometimes there are partial homogeneity
    # For instance, (a^2+b^2+c^2-a*b-b*c-c*a)+(x^6+y^6+z^6-x^2*y^2*z^2)
    # is homogeneous in (a,b,c) and also in (x,y,z), but not in (a,b,c,x,y,z),
    # and the variety of the KKT system is not zero-dimensional.
    # HOW TO DEAL WITH THIS?

    # TODO 2: dehomogenize shall we consider x = 0?
    dehomogenize = {}
    if poly.is_homogeneous and all(_.is_homogeneous for _ in ineq_constraints)\
        and all(_.is_homogeneous for _ in eq_constraints):
        gen = symbols[-1]
        one = sp.S.One
        dehomogenize[gen] = one
        poly = poly.eval(gen, one)
        ineq_constraints = [_.eval(gen, one) for _ in ineq_constraints]
        eq_constraints = [_.eval(gen, one) for _ in eq_constraints]
        symbols = symbols[:-1]

    # always remove linear variables
    elim_linear, eq_constraints = _eliminate_linear(eq_constraints, symbols)
    if elim_linear is None: # inconsistent
        return []
    elim_linear.update(dehomogenize)

    if len(elim_linear):
        symbols = [s for s in symbols if s not in elim_linear]
        poly = polysubs(poly, elim_linear, symbols)
        ineq_constraints = [polysubs(_, elim_linear, symbols) for _ in ineq_constraints]

    # if len(symbols) > max_different:
    #     return []

    solver = partial(_optimize_by_symbol_reduction, max_different=max_different,
        solver=partial(_optimize_by_eq_kkt, max_different=max_different)
    )
    points = _optimize_by_ineq_comb(poly, ineq_constraints, eq_constraints, symbols,
        eliminate_func = _eliminate_linear, solver=solver)

    points = _restore_solution(points, elim_linear, symbols0, symbols)

    if len(points) > 1:
        points = list(set(points))
        if len(points) > 1 and objective != 'all':
            if objective == 'max':
                poly0 = -poly0
                # objective = 'min'
            points = sorted(_get_minimal(poly0, points), key=default_sort_key)

    if return_dict:
        points = [dict(zip(symbols0, _)) for _ in points]
    return points