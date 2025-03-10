from datetime import datetime
from functools import wraps
from inspect import signature
from typing import Tuple, Dict, List, Union, Optional

import sympy as sp
from sympy import sympify, Function
from sympy.core.symbol import uniquely_named_symbol
from sympy.combinatorics import Permutation, PermutationGroup

from ..utils.expression import Solution, SolutionSimple, CyclicExpr
from ..utils.monomials import MonomialManager

class PropertyDict(dict):
    def __getattr__(self, key):
        return self.get(key)

class _sos_solver_class():
    """
    A shared class for registering and calling solvers across different modules.
    """
    _dict = PropertyDict()
    def _register_solver(self, *args):
        if len(args) < 2:
            raise ValueError('At least two arguments are required.')

        pointer = self._dict
        for path in args[:-1]:
            if pointer.get(path) is None:
                pointer[path] = PropertyDict()
            pointer = pointer.get(path)

        if callable(args[-1]):
            pointer[args[-1].__name__] = args[-1]
        elif isinstance(args[-1], (list, tuple)):
            for solver in args[-1]:
                pointer[solver.__name__] = solver

    def __getattr__(self, key):
        return self._dict.get(key)

SS = _sos_solver_class()


def homogenize(poly: sp.Poly, t: Optional[sp.Symbol] = None) -> Tuple[sp.Poly, sp.Symbol]:
    """
    Automatically homogenize a polynomial if it is not homogeneous.

    Parameters
    ----------
    poly : sp.Poly
        The polynomial to homogenize.
    t : Optional[sp.Symbol]
        The symbol to use for homogenization. If None, a new symbol will be created.

    Returns
    ----------
    Tuple[sp.Poly, sp.Symbol]
        The homogenized polynomial and the homogenizer. If the polynomial is already homogeneous,
        the homogenizer will be None.
    """
    is_hom = poly.is_homogeneous
    if is_hom:
        return poly, None

    original_poly = poly
    # create a symbol for homogenization
    homogenizer = t
    if homogenizer is None:
        homogenizer = uniquely_named_symbol('t', sp.Tuple(*original_poly.free_symbols))
    poly = original_poly.homogenize(homogenizer)
    return poly, homogenizer


def homogenize_expr_list(expr_list: List[Union[sp.Expr, sp.Poly]], homogenizer: sp.Symbol) -> List[sp.Expr]:
    """
    Homogenize a list of sympy expressions or polynomials.    
    """
    symbols = set.union(set(), *[set(e.free_symbols) for e in expr_list])
    if homogenizer in symbols:
        symbols.remove(homogenizer)
    translation = {s: s/homogenizer for s in symbols}
    def hom(x):
        if isinstance(x, sp.Expr):
            x = x.subs(translation).together()
            d = sp.fraction(x)[1].as_poly(homogenizer).degree()
            return x * homogenizer**d
        elif isinstance(x, sp.Poly):
            return x.homogenize(homogenizer)
    return [hom(x) for x in expr_list]



def identify_symmetry_from_lists(lst_of_lsts: List[List[sp.Poly]], has_homogenized: bool = False) -> PermutationGroup:
    """
    Infer a symmetric group so that each list of (list of polynomials) is symmetric with respect to the rule.
    It only identifies very simple groups like complete symmetric and cyclic groups.

    Parameters
    ----------
    lst_of_lsts : List[List[sp.Poly]]
        A list of lists of polynomials.
    has_homogenized : bool
        If has_homogenized is True, it hints the function to check the symmetry before homogenization.

    Returns
    ----------
    PermutationGroup
        The inferred permutation group.

    Examples
    ----------
    >>> identify_symmetry_from_lists([[(a+b+c-3).as_poly(a,b,c)], [a.as_poly(a,b,c), b.as_poly(a,b,c), c.as_poly(a,b,c)]]).is_symmetric
    True

    >>> identify_symmetry_from_lists([[(a+b+c-3).as_poly(a,b,c)], [(2*a+b).as_poly(a,b,c), (2*b+c).as_poly(a,b,c), (2*c+a).as_poly(a,b,c)]], has_homogenized=True)
    PermutationGroup([
        (0 1 2)])

    See Also
    ----------
    identify_symmetry
    """
    gens = None
    for l in lst_of_lsts:
        for p in l:
            gens = p.gens
            break
        if gens is not None:
            break

    def check_symmetry(polys, perm):
        rep_set = set()
        reorder_set = set()
        for poly in polys:
            rep = poly.rep
            reorder = poly.reorder(*perm(gens)).rep
            if rep == reorder:
                continue
            rep_set.add(rep)
            reorder_set.add(reorder)
        for r in reorder_set:
            if r not in rep_set:
                return False
        return True

    # List a few candidates: symmetric, alternating, cyclic groups...
    nvars = len(gens)
    def _rotated(n, start=0):
        return list(range(start+1, n+start)) + [start]
    def _reflected(n, start=0):
        return [start+1, start] + list(range(start+2, n+start))

    verified = [] # storing permutations that fit the input
    candidates = [] # a list of permutations
    if nvars > 1:
        candidates.append(_rotated(nvars))
        if nvars > 2:
            candidates.append(_reflected(nvars))

        # bi-symmetric group etc.
        if nvars > 3 and (nvars - int(has_homogenized)) % 2 == 0:
            half = nvars // 2
            p1 = _rotated(half) + _rotated(half, half)
            p2 = _reflected(half) + _reflected(half, half)
            p3 = list(range(half,half*2)) + list(range(half))
            if has_homogenized:
                p1.append(nvars - 1)
                p2.append(nvars - 1)
                p3.append(nvars - 1)
            candidates.append(p1)
            candidates.append(p2)
            candidates.append(p3)
            
        if has_homogenized and nvars > 2:
            candidates.append(_rotated(nvars - 1) + [nvars - 1])
            if nvars > 3:
                candidates.append(_reflected(nvars - 1) + [nvars - 1])

    for perm in map(Permutation, candidates):
        if all(check_symmetry(l, perm) for l in lst_of_lsts):
            verified.append(perm)

    if len(verified) == 0:
        verified.append(Permutation(list(range(nvars))))

    return PermutationGroup(*verified)


def clear_polys_by_symmetry(polys: List[Union[sp.Expr, Tuple[sp.Expr, ...]]],
        symbols: Tuple[sp.Symbol, ...], symmetry: MonomialManager) -> List[Union[sp.Expr, Tuple[sp.Expr, ...]]]:
    """
    Remove duplicate polys by symmetry.
    """
    if symmetry.is_trivial:
        return polys if isinstance(polys, list) else list(polys)

    def _get_representation(t: sp.Expr):
        """Get the standard representation of the poly given symmetry."""
        t = sp.Poly(t, symbols) if not isinstance(t, tuple) else sp.Poly(t[0], symbols)
        # if t.is_monomial and len(t.free_symbols) == 1:
        #     return None
        vec = symmetry.base().arraylize_sp(t)
        mat = symmetry.permute_vec(vec, t.total_degree())
        cols = [tuple(mat[:, i]) for i in range(mat.shape[1])]
        return max(cols)
 
    representation = dict(((_get_representation(t), t) for i, t in enumerate(polys)))
    if None in representation:
        del representation[None]
    return list(representation.values())


# fix the bug in sqf_list before 1.13.0
# https://github.com/sympy/sympy/pull/26182
if sp.__version__ >= '1.13.0':
    _sqf_list = lambda p: p.sqf_list()
else:
    _sqf_list = lambda p: p.factor_list() # it would be slower, but correct

def _std_ineq_constraints(p: sp.Poly, e: sp.Expr) -> Tuple[sp.Poly, sp.Expr]:
    if p.is_zero: return p, e
    c, lst = _sqf_list(p)
    ret = sp.S(1 if c > 0 else -1).as_poly(*p.gens, domain=p.domain)
    e = e / (c if c > 0 else -c)
    for q, d in lst:
        if d % 2 == 1:
            ret *= q
        e = e / q.as_expr()**(d - d%2)
    return ret, e

def _std_eq_constraints(p: sp.Poly, e: sp.Expr) -> Tuple[sp.Poly, sp.Expr]:
    if p.is_zero: return p, e
    c, lst = _sqf_list(p)
    ret = sp.S(1 if c > 0 else -1).as_poly(*p.gens, domain=p.domain)
    e = e / c
    max_d = sp.Integer(max(1, *(d for q, d in lst)))
    for q, d in lst:
        ret *= q
        e = e * q.as_expr()**(max_d - d)
    if max_d != 1:
        e = sp.Pow(e, 1/max_d, evaluate=False)
    if c < 0:
        e = e.__neg__()
    return ret, e


def sanitize_output(*_args, **_kwargs):
    """
    Decorator for sum of square functions. It writes extra information to the solution object.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(poly: sp.Expr,
                ineq_constraints: Union[List[sp.Expr], Dict[sp.Expr, sp.Expr]] = {},
                eq_constraints: Union[List[sp.Expr], Dict[sp.Expr, sp.Expr]] = {}, *args, **kwargs):
            start_time = datetime.now()
            sol = func(poly, ineq_constraints, eq_constraints, *args, **kwargs)
            end_time = datetime.now()
            if isinstance(sol, Solution):
                sol._start_time = start_time
                sol._end_time = end_time
                sol.problem = poly
                sol.ineq_constraints = ineq_constraints
                sol.eq_constraints = eq_constraints
            return sol
        return wrapper
    return decorator


def sanitize_input(
        homogenize: bool = False,
        ineq_constraint_sqf: bool = True,
        eq_constraint_sqf: bool = True,
        infer_symmetry: bool = False,
        wrap_constraints: bool = False
    ):
    """
    Decorator for sum of square functions. It sanitizes the input type before calling the solver function.

    For inequality and equality constraints, squared parts are extracted and the rest is
    standardized. For example, -3(x+y)²z >= 0 will be converted to -z >= 0, while
    5(x+y)²z == 0 will be converted to (x+y)z == 0.

    Symmetric groups will be inferred if infer_symmetry is True. This will be called by
    LinearSOS and SDPSOS. It checks the symmetry of the input polynomials and constraints
    and parse constraints dictionary to a form that is compatible with the symmetry.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(poly: sp.Expr,
                ineq_constraints: Union[List[sp.Expr], Dict[sp.Expr, sp.Expr]] = {},
                eq_constraints: Union[List[sp.Expr], Dict[sp.Expr, sp.Expr]] = {}, *args, **kwargs):
            if not isinstance(ineq_constraints, dict):
                ineq_constraints = {e: e for e in ineq_constraints}
            if not isinstance(eq_constraints, dict):
                eq_constraints = {e: e for e in eq_constraints}
            ineq_constraints = dict((sympify(e), sympify(e2).as_expr()) for e, e2 in ineq_constraints.items())
            eq_constraints = dict((sympify(e), sympify(e2).as_expr()) for e, e2 in eq_constraints.items())

            original_symbols = [] if not isinstance(poly, sp.Poly) else poly.gens
            symbols = set.union(
                set(poly.free_symbols), 
                *[set(e.free_symbols) for e in ineq_constraints.keys()],
                *[set(e.free_symbols) for e in eq_constraints.keys()]
            )
            if len(symbols) == 0 and len(original_symbols) == 0:
                symbols = {sp.Symbol('x')}
                # raise ValueError('No symbols found in the input.')
            symbols = symbols - set(original_symbols)
            symbols = tuple(sorted(list(symbols), key=lambda x: x.name))
            symbols = tuple(original_symbols) + symbols

            poly = sp.Poly(poly.doit(), *symbols)
            ineq_constraints = dict((sp.Poly(e.doit(), *symbols), e2) for e, e2 in ineq_constraints.items())
            eq_constraints = dict((sp.Poly(e.doit(), *symbols), e2) for e, e2 in eq_constraints.items())

            homogenizer = None
            is_hom = poly.is_homogeneous
            is_hom = is_hom and all(e.is_homogeneous for e in ineq_constraints) and all(e.is_homogeneous for e in eq_constraints)
            if (not is_hom) and homogenize:
                homogenizer = uniquely_named_symbol('t', 
                    tuple(set.union(set(symbols), *(e.free_symbols for e in ineq_constraints.values()), *(e.free_symbols for e in eq_constraints.values()))))
                poly = poly.homogenize(homogenizer)
                ineq_constraints = dict((e.homogenize(homogenizer), e2) for e, e2 in ineq_constraints.items())
                ineq_constraints[homogenizer.as_poly(*poly.gens)] = homogenizer
                eq_constraints = dict((e.homogenize(homogenizer), e2) for e, e2 in eq_constraints.items())
                if '_homogenizer' in signature(func).parameters.keys():
                    kwargs['_homogenizer'] = homogenizer

            if ineq_constraint_sqf:
                ineq_constraints = dict(_std_ineq_constraints(*item) for item in ineq_constraints.items())
            ineq_constraints = dict((e, e2) for e, e2 in ineq_constraints.items() if e.total_degree() > 0)

            if eq_constraint_sqf:
                eq_constraints = dict(_std_eq_constraints(*item) for item in eq_constraints.items())
            eq_constraints = dict((e, e2) for e, e2 in eq_constraints.items() if e.total_degree() > 0)
            # print('Ineq =', ineq_constraints, '\nEq =', eq_constraints)


            symmetry = kwargs.get('symmetry')
            if homogenizer is not None and symmetry is not None:
                # the generators might increase after homogenization
                symmetry = symmetry.perm_group if isinstance(symmetry, MonomialManager) else symmetry
                nvars = len(poly.gens)
                if symmetry.degree != nvars:
                    symmetry = PermutationGroup(*[Permutation(_.array_form + [nvars-1]) for _ in symmetry.args])

            if infer_symmetry and symmetry is None:
                symmetry = identify_symmetry_from_lists(
                    [[poly], list(ineq_constraints.keys()), list(eq_constraints.keys())], has_homogenized=homogenizer is not None)
            
            symmetry = MonomialManager(len(poly.gens), symmetry)

            _has_symmetry_kwarg = signature(func).parameters.get('symmetry') is not None
            if _has_symmetry_kwarg:
                kwargs['symmetry'] = symmetry

            constraints_wrapper = None
            if wrap_constraints:
                # wrap ineq/eq constraints to be sympy function class wrt. generators rather irrelevent symbols
                constraints_wrapper = _get_constraints_wrapper(poly.gens, ineq_constraints, eq_constraints, symmetry.perm_group)
                ineq_constraints, eq_constraints = constraints_wrapper[0], constraints_wrapper[1]

            ########################################################
            #               Call the solver function
            ########################################################
            sol: SolutionSimple = func(poly, ineq_constraints, eq_constraints, *args, **kwargs)
            if sol is None:
                return None

            if constraints_wrapper is not None:
                # note: first restore the constraints (with homogenizer), and then dehomogenize
                sol = sol.xreplace(constraints_wrapper[2])
                sol = sol.xreplace(constraints_wrapper[3])
            if homogenizer is not None:
                sol = sol.dehomogenize(homogenizer)
            return sol
        return wrapper
    return decorator


def _get_constraints_wrapper(symbols: Tuple[int, ...], ineq_constraints: Dict[sp.Poly, sp.Expr], eq_constraints: Dict[sp.Poly, sp.Expr], perm_group: PermutationGroup):
    def _get_mask(symbols, dlist):
        # only reserve symbols with degree > 0, this reduces time complexity greatly
        return tuple(s for d, s in zip(dlist, symbols) if d != 0)

    def _get_dicts(constraints, name='_G'):
        dt = dict()
        inv = dict()
        rep_dict = dict((p.rep, v) for p, v in constraints.items())
        counter = 0  
        for base in constraints.keys():
            if base.rep in dt:
                continue
            dlist = base.degree_list()
            for p in perm_group.elements:
                invorder = p.__invert__()(symbols)
                permed_base = base.reorder(*invorder).rep
                permed_expr = rep_dict.get(permed_base)
                if permed_expr is None:
                    raise ValueError("Given constraints are not symmetric with respect to the permutation group.")
                compressed = _get_mask(p(symbols), dlist)
                value = sp.Function(name + str(counter))(*compressed)
                dt[permed_base] = value
                inv[value] = permed_expr
            counter += 1
        dt = dict((sp.Poly.new(k, *symbols), v) for k, v in dt.items())
        return dt, inv
    i2g, g2i = _get_dicts(ineq_constraints, name='_G')
    e2h, h2e = _get_dicts(eq_constraints, name='_H')
    return i2g, e2h, g2i, h2e