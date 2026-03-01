"""
This module provides heuristic functions to optimize a polynomial (algebraically)
with inequality and equality constraints. Directly solving the KKT system might
be EXTREMELY SLOW and fail for nonzero dimensional variety. The functions here
uses a heuristic approach to compute the extrema of the polynomial.
"""
from functools import wraps, partial
from itertools import product
from typing import List, Dict, Tuple, Union, Optional, Callable, Generator

from sympy import (Expr, Poly, Integer, Rational, Float,
    Dummy, Symbol, EmptySet, FiniteSet,
    linear_eq_to_matrix, linsolve, sympify, true
)
from sympy.combinatorics import PermutationGroup
from sympy.polys.polyerrors import PolificationFailed, DomainError
from sympy.polys.polyclasses import DMP

from .polysolve import univar_realroots, solve_poly_system_crt, PolyEvalf, _filter_trivial_system
from .roots import Root
from .root_list import RootList
from ..expressions import identify_symmetry_from_lists


# Comparison of tuples of sympy Expressions, compatible with sympy <= 1.9
default_sort_key = lambda x: tuple(_.sort_key() for _ in x) if not isinstance(x, Expr) else x.sort_key()

def polysubs(poly: Poly, subs: Dict[Symbol, Expr], symbols: List[Symbol]) -> Poly:
    """Substitutes the symbols in a polynomial with the given values and
    makes it a polynomial in the given symbols."""
    if len(subs) == 0:
        return poly
    if len(symbols) == 0:
        # TODO: make it faster
        return poly.as_expr().xreplace(subs).expand()

    poly0 = poly
    # poly = poly.as_expr().xreplace(subs)
    # return poly.as_poly(*symbols)

    inds = [poly.gens.index(s) if s in poly.gens else -1 for s in symbols]
    rest_inds = [i for i, g in enumerate(poly.gens) if (g not in symbols)]
    marginalize1 = lambda x: tuple(x[i] if i != -1 else 0 for i in inds)
    marginalize2 = lambda x: tuple(x[i] for i in rest_inds)
    coeffs = {}
    # Rearrange the poly to be a polynomial in the to-be-substituted-variables.
    for monom, coeff in poly.rep.terms():
        m1 = marginalize1(monom)
        m2 = marginalize2(monom)
        if not (m2 in coeffs):
            coeffs[m2] = {}
        coeffs[m2][m1] = coeff
    lev = len(inds) - 1
    coeffs = {k: DMP.from_dict(rep, lev, poly.domain) for k, rep in coeffs.items()}

    subs = [Poly(subs[poly.gens[i]], symbols, extension=True).rep for i in rest_inds]
    s = 0
    for monom, coeff in coeffs.items():
        for i, d in enumerate(monom):
            if d:
                coeff = coeff * subs[i]**d
        s = s + coeff
    return Poly.new(s, *symbols)


def _infer_symbols(symbols: Optional[List[Symbol]], poly: Poly, *constraint_lists: List[Poly]) -> List[Symbol]:
    """Infer the symbols from the polynomial and constraints."""
    if symbols is None:
        if isinstance(poly, Poly):
            symbols = poly.gens
        else:
            symbols = poly.free_symbols
            for c in constraint_lists:
                symbols = symbols.union(*[_.free_symbols for _ in c])
            symbols = sorted(symbols, key=lambda x: x.name)
    return symbols

def polylize_input(poly: Expr, ineq_constraints: List[Expr], eq_constraints: List[Expr], symbols: List[Symbol],
        check_poly: Callable=None) -> Tuple[Poly, List[Poly], List[Poly]]:
    """Convert input expressions to sympy polynomial instances."""
    if check_poly is None:
        check_poly = lambda *args, **kwargs: True
    def polylize(f, symbols):
        f = sympify(f)
        if not isinstance(f, Poly) or not f.domain.is_Numerical:
            f = Poly(f.doit(), *symbols, extension=True)
        if f is None:
            raise PolificationFailed({}, f, f)
        if not check_poly(f):
            raise DomainError('Polynomial domains must be exact.')
        return f
    poly = polylize(poly, symbols)
    ineq_constraints = [polylize(ineq, symbols) for ineq in ineq_constraints]
    eq_constraints = [polylize(eq, symbols) for eq in eq_constraints]
    return poly, ineq_constraints, eq_constraints

def _get_minimal(poly: Poly, points: List[Tuple[Expr]]) -> List[Tuple[Expr]]:
    """
    Get the minimal points in the list of points up to `eps` precision.
    Points are evaluated at precision `n`.
    """
    if len(points) == 0:
        return points
    original_points = points

    pevalf = PolyEvalf()
    eps = Float(f'1e-{pevalf.dps//3}')
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
            return points if all(_ >= 0 in (true, True) for _ in ineq_constraints) else []
        ineq_constraints = [_.as_poly(symbols) for _ in ineq_constraints]
        pevalf = PolyEvalf()
        points = [p for p in points if all(pevalf.polysign(_, p) >= 0 for _ in ineq_constraints)]
        return points
    return wrapper

class PolyCombSymmetry:
    """
    Class to manipulate the combination of a polynomial list given a symmetry group.
    """
    def __init__(self, polys: List[Poly], symmetry: Optional[PermutationGroup]=None):
        self.polys = polys
        self.symmetry = symmetry
        self.rep = [_.rep for _ in polys]
        self.rep_dict = {r: i for i, r in enumerate(self.rep)}
    def __len__(self) -> int:
        return len(self.polys)
    def generate(self) -> Generator[List[int], None, None]:
        """Generate the combinations of the polynomials."""
        if self.symmetry is None:
            for comb in product([False, True], repeat=len(self.polys)):
                yield tuple(i for i, j in enumerate(comb) if j)
            return

        def reorder(perm, poly):
            gens = poly.gens
            return poly.reorder(*perm(gens)).rep
        checked = {}
        cnt = 0
        polys, rep_dict = self.polys, self.rep_dict
        generators = [_ for _ in self.symmetry.generators if not _.is_identity]
        for comb in product([False, True], repeat=len(self.polys)):
            inds = tuple(i for i, j in enumerate(comb) if j)
            if inds in checked:
                continue
            new_comb = True
            new_checked = {inds: True}
            for p in generators: # TODO: using elements might be more proper
                new_inds = tuple(sorted([rep_dict.get(reorder(p, polys[i]), -1) for i in inds]))
                if new_inds in checked:
                    # print('Polys', [polys[i] for i in inds], 'equiv to', [polys[i] for i in new_inds])
                    new_comb = False
                    # break # is breaking correct?
                new_checked[new_inds] = True
            if new_comb:
                checked.update(new_checked)
                cnt += 1
                yield inds

def kkt(
    f: Union[Expr, Poly],
    ineq_constraints: List[Union[Expr, Poly]] = [],
    eq_constraints: List[Union[Expr, Poly]] = [],
    as_poly: bool = True,
    fj: bool = False,
    with_self: bool = False
) -> Tuple[List[Poly], Tuple[Tuple[Symbol, ...], Tuple[Symbol, ...], Tuple[Symbol, ...], Tuple[Symbol, ...]]]:
    r"""
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
    >>> from sympy.abc import x, y, z
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
    f = sympify(f)
    symbs = f.gens if hasattr(f, 'gens') else tuple(sorted(f.free_symbols, key=lambda x: x.name))
    symbs = symbs + tuple(set.union(set(),
                *[_.free_symbols for _ in ineq_constraints],
                *[_.free_symbols for _ in eq_constraints]) - set(symbs))
    symbs0 = symbs

    fj = 1 if fj else 0
    lambs = tuple(Dummy('\\lambda_%d' % i) for i in range(len(ineq_constraints) + fj))
    mus = tuple(Dummy('\\mu_%d' % i) for i in range(len(eq_constraints)))
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



def _solve_2vars_zero_extrema(poly: Poly, symbols: Symbol) -> List[Tuple[Expr]]:
    """
    Solve the system poly = poly.diff(x) = poly.diff(y) = 0 via resultants.
    It exploits the fact that the roots have multiplicity 2 and finds them
    by gcd(res, res.diff()), which avoids factorization to achieve better performance.
    """
    x, y = symbols
    dx, dy = poly.diff(x), poly.diff(y)
    res0 = poly.reorder(y, x).resultant(dy.reorder(y, x))
    res0 = res0.gcd(res0.diff(x))
    roots1 = univar_realroots(res0, x)

    if len(roots1) == 0:
        return []

    if all(isinstance(_, Rational) for _ in roots1):
        # solve the other root by direct substitution
        roots = []
        for root1 in roots1:
            poly2 = poly.eval(x, root1)
            poly2 = poly2.gcd(poly2.diff(y))
            roots2 = univar_realroots(poly2, y)
            for root2 in roots2:
                roots.append((root1, root2))
        return roots

    # compute the resultant of the other variable
    res1 = poly.resultant(dx)
    res1 = res1.gcd(res1.diff(y))
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
        if all(_ == 0 or (isinstance(_, Poly) and _.is_zero) for _ in eq_constraints):
            return [tuple()]
        return []
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
                eqgcd = eqgcd.gcd(eq.as_poly(symbols[0], extension=True))
            sol = [(root,) for root in univar_realroots(eqgcd, symbols[0])]
        else:
            # Solve by derivative.
            sol = [(root,) for root in univar_realroots(poly.diff(symbols[0]), symbols[0])]
        return sol

    elif len(symbols) == 2:
        x, y = symbols
        poly = poly.as_poly(x, y, extension=True)
        # print(eq_constraints)
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

def _eliminate_linear(polys, symbols) -> Tuple[Dict[Symbol, Expr], List[Poly]]:
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
        # rest_inds = sorted(list(rest_inds))
        # new_eliminated_polys = sorted(list(new_eliminated_polys))
        new_eliminated_polys = [polys[i] for i in new_eliminated_polys]

        # TODO: clean this
        linsys = linear_eq_to_matrix(
            [_.as_expr() for _ in new_eliminated_polys], new_eliminated_symbols)
        sol = linsolve(linsys, *new_eliminated_symbols)
        if sol is EmptySet or len(sol) == 0:
            return None, polys
        sol = sol[0] if not isinstance(sol, FiniteSet) else sol.args[0]
        eliminated_set = set(new_eliminated_symbols)
        if any(_.free_symbols.intersection(eliminated_set) for _ in sol):
            # Underdetermined system -> nonzero dimensional variety
            # TODO: handle this properly
            return None, polys
        sol = dict(zip(new_eliminated_symbols, sol))

        symbols = [s for s in symbols if s not in new_eliminated_symbols]
        polys = [polys[i] for i in rest_inds]
        polys = [polysubs(_, sol, symbols) for _ in polys]
        if len(symbols) == 0:
            if any(_ != 0 for _ in polys):
                return None, polys

        # update eliminated
        for k, v in eliminated.items():
            eliminated[k] = v.xreplace(sol)
        eliminated.update(sol)

    return eliminated, polys

def _restore_solution(points: List[Tuple[Expr]], elim: Dict[Symbol, Expr],
        symbols: List[Symbol], symbols2: List[Symbol]) -> List[Tuple[Expr]]:
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


def _assign_groups(nvars: int, m: int, include_zero: bool = True):
    l = [0] * nvars
    total = []
    start = 0 if include_zero else 1
    def _recur(total, l, i, group_ind):
        if i == nvars:
            if group_ind == m:
                total.append(tuple(l))
            return
        for j in range(start, group_ind + 1):
            l[i] = j
            _recur(total, l, i + 1, group_ind)
        if group_ind < m:
            l[i] = group_ind + 1
            _recur(total, l, i + 1, group_ind + 1)
    _recur(total, l, 0, 0)
    return total


def _optimize_by_symbol_reduction(
    poly: Poly,
    ineq_constraints: List[Poly],
    eq_constraints: List[Poly],
    symbols: List[Symbol],
    max_different=2,
    solver=None,
    include_zero=True
) -> List[Tuple[Expr]]:
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
    new_symbols = [Integer(0)] + [Dummy(f'x{i}') for i in range(max_different)]

    # assign symbols to some groups
    # group indices are standardized: e.g. (a,a,b,a,b) is equivalent as (b,b,a,b,a)
    for comb in _assign_groups(len(symbols), max_different, include_zero):
        active_symbols = new_symbols[1: 1 + max(comb)]

        replacement = dict(zip(symbols, [new_symbols[i] for i in comb]))

        poly2 = polysubs(poly, replacement, active_symbols)
        eq_constraints2 = [polysubs(_, replacement, active_symbols) for _ in eq_constraints]
        eq_constraints2 = _filter_trivial_system(eq_constraints2)
        if eq_constraints2 is None:
            continue
        ineq_constraints2 = [polysubs(_, replacement, active_symbols) for _ in ineq_constraints]

        points = solver(poly2, ineq_constraints2, eq_constraints2, active_symbols)
        points = _restore_solution(points, replacement, symbols, active_symbols)
        all_points.extend(points)
    return all_points


def _optimize_by_ineq_comb(
    poly: Poly,
    ineq_constraints: List[Poly],
    eq_constraints: List[Poly],
    symbols: List[Symbol],
    eliminate_func=None,
    solver=None,
    symmetry: Optional[PermutationGroup]=None
) -> List[Tuple[Expr]]:
    """
    Optimize a polynomial with inequality constraints by considering all possible
    combinations of active inequality constraints. After each dicision of active
    inequality constraints, the new system is first eliminated by `eliminate_func`
    and then passed into the downstream solver `solver`.
    """

    # decide all non-equivalent combinations of active ineq_constraints
    # given a symmetry group
    if symmetry is None:
        symmetry = identify_symmetry_from_lists([[poly], ineq_constraints, eq_constraints])
    ineq_comb = PolyCombSymmetry(ineq_constraints, symmetry)

    #############################################
    # always remove linear variables
    symbols0 = symbols
    outer_elim, eq_constraints = _eliminate_linear(eq_constraints, symbols)
    if outer_elim is None: # inconsistent
        return []

    if len(outer_elim):
        symbols = [s for s in symbols if s not in outer_elim]
        poly = polysubs(poly, outer_elim, symbols)
        ineq_constraints = [polysubs(_, outer_elim, symbols) for _ in ineq_constraints]
    #############################################

    all_points = []
    elim = {}
    pevalf = PolyEvalf()

    for active in ineq_comb.generate():
        active_ineq0 = [ineq_constraints[i] for i in active]
        if eliminate_func is not None:
            elim, active_ineq = eliminate_func(active_ineq0, symbols)
        if elim is None: # inconsistent
            continue

        if len(elim):
            symbols2 = [s for s in symbols if s not in elim]
            # poly2 = poly.as_expr().xreplace(elim).as_poly(*symbols2)
            poly2 = polysubs(poly, elim, symbols2)
            eq_constraints2 = [polysubs(_, elim, symbols2) for _ in eq_constraints]
            eq_constraints2 = _filter_trivial_system(active_ineq + eq_constraints2)
            if eq_constraints2 is None:
                continue
        else:
            poly2, eq_constraints2, symbols2 = poly, active_ineq + eq_constraints, symbols

        points = solver(poly2, {}, eq_constraints2, symbols2)

        #############################################
        # restore the eliminated variables
        points = _restore_solution(points, elim, symbols, symbols2)

        # check inactive ineqs >= 0
        inactive_ineq = [ineq_constraints[i] for i in set(range(len(ineq_constraints))) - set(active)]
        points = [_ for _ in points if all(pevalf.polysign(ineq, _) >= 0 for ineq in inactive_ineq)]

        points = _restore_solution(points, outer_elim, symbols0, symbols)
        if len(active_ineq0):
            # restore the permutations of points given the symmetry group
            points = set(points)
            for p in ineq_comb.symmetry.elements:
                points.update([tuple(p(point)) for point in points])
            points = list(points)
        #############################################

        all_points.extend(points)
        # print('Time =', time()-time0, 'Active =', active_ineq0, 'Points =', points)

    return all_points


def _optimize_poly(
    poly: Poly,
    ineq_constraints: List[Poly],
    eq_constraints: List[Poly],
    symbols: List[Symbol],
    max_different: int = 2,
    symmetry: Optional[PermutationGroup]=None
) -> List[Tuple[Expr]]:
    """
    Internal function to optimize a polynomial with inequality and equality constraints.
    """
    # TODO: sometimes there are partial homogeneity
    # For instance, (a^2+b^2+c^2-a*b-b*c-c*a)+(x^6+y^6+z^6-x^2*y^2*z^2)
    # is homogeneous in (a,b,c) and also in (x,y,z), but not in (a,b,c,x,y,z),
    # and the variety of the KKT system is not zero-dimensional.
    # HOW TO DEAL WITH THIS?

    if symmetry is None:
        symmetry = identify_symmetry_from_lists([[poly], ineq_constraints, eq_constraints])
    points = []
    nvars = len(symbols)

    if poly.is_homogeneous and all(_.is_homogeneous for _ in ineq_constraints)\
        and all(_.is_homogeneous for _ in eq_constraints):

        if nvars == 1:
            points = [(x,) for x in [Integer(-1),Integer(0),Integer(1)] if poly(x) == 0 \
                     and all(ineq(x) >= 0 for ineq in ineq_constraints)\
                     and all(eq(x) == 0 for eq in eq_constraints)]
            return points

        nonnegative_symbols = set()
        if len(ineq_constraints) >= nvars:
            for ineq in ineq_constraints:
                if (not ineq.is_zero) and ineq.is_monomial:
                    monom = tuple(ineq.LM())
                    if sum(monom) != 1:
                        continue
                    nonnegative_symbols.add(monom.index(1))
        if len(nonnegative_symbols) < nvars:
            # consider the degenerated case sum(symbols) == 0
            gen, rest_symbols = symbols[-1], symbols[:-1]
            eliminated = {gen: -sum(rest_symbols).as_poly(rest_symbols)}

            new_points = _optimize_poly(
                polysubs(poly, eliminated, rest_symbols),
                [polysubs(_, eliminated, rest_symbols) for _ in ineq_constraints],
                [polysubs(_, eliminated, rest_symbols) for _ in eq_constraints],
                symbols=rest_symbols, max_different=max_different
            )
            new_points = [x + (-sum(x),) for x in new_points]
            points.extend(new_points)
        elif all(_.coeff_monomial((0,)*nvars) >= 0 for _ in ineq_constraints)\
            and all(_.coeff_monomial((0,)*nvars) == 0 for _ in eq_constraints):
            points.append((Integer(0),)*nvars)

        # By homogeneity we assume sum(symbols) == 1
        # This should be eliminated by `_eliminate_linear`
        # TODO: whether necessary to assume sum(symbols) == 1
        eq_constraints.append(Poly(sum(symbols) - 1, *symbols))

    solver = partial(_optimize_by_symbol_reduction, max_different=max_different,
        solver=partial(_optimize_by_eq_kkt, max_different=max_different)
    )
    points.extend(_optimize_by_ineq_comb(poly, ineq_constraints, eq_constraints, symbols,
        eliminate_func=_eliminate_linear, solver=solver, symmetry=symmetry))
    return points


def optimize_poly(
    poly: Union[Poly, Expr],
    ineq_constraints: List[Union[Poly, Expr]] = [],
    eq_constraints: List[Union[Poly, Expr]] = [],
    symbols: List[Symbol] = None,
    objective: str = 'min',
    return_type: str = 'tuple',
    max_different: int = 2
) -> List[Root]:
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
    return_type : str
        The returned type, should be one of "tuple", "root" or "dict".
        Warning: "root" might spend a lot of time constructing
        the algebraic Root instances.
    max_different : int
        The maximum number of different variables to consider.
        This is a heuristic to accelerate the computation.
        It does not force all extrema to be bounded by this number
        if they are easy to compute.

    Returns
    ----------
    solutions : List[Tuple[Expr]] or List[Dict[Symbol, Expr]]
        The extrema of the polynomial. If return_dict is True, then
        the extrema are returned as dictionaries.

    Examples
    ----------
    >>> from sympy.abc import a, b, c, x, y
    >>> optimize_poly(a + 2*b, [a, b], [a**2 + b**2 - 1], (a, b), objective='max') # doctest: +SKIP
    [(CRootOf(5*a**2 - 1, 1), CRootOf(5*b**2 - 4, 1))]

    >>> optimize_poly((a**2+b**2+c**2)**2-3*(a**3*b+b**3*c+c**3*a)) # doctest: +SKIP
    [(1, 1, 1),
     (CRootOf(a**3 - 6*a**2 + 5*a - 1, 0), CRootOf(b**3 - 5*b**2 + 6*b - 1, 1), 1),
     (CRootOf(a**3 - 6*a**2 + 5*a - 1, 1), CRootOf(b**3 - 5*b**2 + 6*b - 1, 0), 1),
     (CRootOf(a**3 - 6*a**2 + 5*a - 1, 2), CRootOf(b**3 - 5*b**2 + 6*b - 1, 2), 1)]
    """
    symbols = _infer_symbols(symbols, poly, ineq_constraints, eq_constraints)
    if not (objective in ('min', 'max', 'all')):
        raise ValueError('Objective must be either "min" or "max" or "all".')
    if not (return_type in ('root', 'tuple', 'dict')):
        raise ValueError('Return type must be either "root" or "tuple" or "dict".')
    if len(symbols) == 0:
        return [] if return_type != 'root' else RootList((), [])

    poly, ineq_constraints, eq_constraints = polylize_input(
        poly, ineq_constraints, eq_constraints, symbols=symbols,
        check_poly=lambda p: p.domain.is_Numerical and p.domain.is_Exact
    )


    solver = partial(_optimize_poly, max_different=max_different)
    points = []
    if poly in eq_constraints:
        # This often happens when solving for the equality case of a nonnegative poly.
        parts = [_[0] for _ in poly.factor_list()[1]]
        eq_constraints.remove(poly)
        for part_poly in parts:
            new_eq_constraints = eq_constraints.copy()
            new_eq_constraints.append(part_poly)
            points.extend(solver(part_poly, ineq_constraints, new_eq_constraints, symbols))
    else:
        points = solver(poly, ineq_constraints, eq_constraints, symbols)

    if len(points) > 1:
        points = list(set(points))
        if len(points) > 1 and objective != 'all':
            if objective == 'max':
                poly = -poly
                # objective = 'min'
            points = _get_minimal(poly, points)
        points = sorted(points, key=default_sort_key)

    if return_type == 'root':
        points = RootList(symbols, [Root(_) for _ in points])
    elif return_type == 'dict':
        points = [dict(zip(symbols, _)) for _ in points]
    return points
