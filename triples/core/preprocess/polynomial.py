from functools import wraps
from inspect import signature
from typing import List, Dict, Union, Tuple

from sympy import __version__ as SYMPY_VERSION
from sympy import Expr, Poly, Symbol, Integer, Pow, Function
from sympy.core.symbol import uniquely_named_symbol
from sympy.combinatorics import PermutationGroup, Permutation

from ...utils import (
    Solution, MonomialManager, CyclicSum,
    identify_symmetry_from_lists, verify_symmetry, poly_reduce_by_symmetry
)
from sympy.external.importtools import version_tuple

# fix the bug in sqf_list before 1.13.0
# https://github.com/sympy/sympy/pull/26182
if tuple(version_tuple(SYMPY_VERSION)) >= (1, 13):
    _sqf_list = lambda p: p.sqf_list()
else:
    _sqf_list = lambda p: p.factor_list() # it would be slower, but correct


def _std_ineq_constraints(p: Poly, e: Expr) -> Tuple[Poly, Expr]:
    if p.is_zero: return p, e
    c, lst = _sqf_list(p)
    ret = Integer(1 if c > 0 else -1).as_poly(*p.gens, domain=p.domain)
    e = e / (c if c > 0 else -c)
    for q, d in lst:
        if d % 2 == 1:
            ret *= q
        e = e / q.as_expr()**(d - d%2)
    return ret, e

def _std_eq_constraints(p: Poly, e: Expr) -> Tuple[Poly, Expr]:
    if p.is_zero: return p, e
    c, lst = _sqf_list(p)
    ret = Integer(1 if c > 0 else -1).as_poly(*p.gens, domain=p.domain)
    e = e / c
    max_d = Integer(max(1, *(d for q, d in lst)))
    for q, d in lst:
        ret *= q
        e = e * q.as_expr()**(max_d - d)
    if max_d != 1:
        e = Pow(e, 1/max_d, evaluate=False)
    if c < 0:
        e = e.__neg__()
    return ret, e


def _polylize(poly: Expr,
        ineq_constraints: Dict[Expr, Expr] = {},
        eq_constraints: Dict[Expr, Expr] = {},
        ineq_constraint_sqf: bool = True,
        eq_constraint_sqf: bool = True
    ) -> Tuple[Poly, Dict[Poly, Expr], Dict[Poly, Expr], Tuple[Symbol, ...]]:
    """
    Convert every expression in the input to be a sympy polynomial.
    """
    original_symbols = [] if not isinstance(poly, Poly) else poly.gens
    symbols = set.union(
        set(poly.free_symbols), 
        *[set(e.free_symbols) for e in ineq_constraints.keys()],
        *[set(e.free_symbols) for e in eq_constraints.keys()]
    )
    if len(symbols) == 0 and len(original_symbols) == 0:
        symbols = {Symbol('x')}
        # raise ValueError('No symbols found in the input.')
    symbols = symbols - set(original_symbols)
    symbols = tuple(sorted(list(symbols), key=lambda x: x.name))
    symbols = tuple(original_symbols) + symbols

    poly = Poly(poly.doit(), *symbols)
    ineq_constraints = dict((Poly(e.doit(), *symbols), e2) for e, e2 in ineq_constraints.items())
    eq_constraints = dict((Poly(e.doit(), *symbols), e2) for e, e2 in eq_constraints.items())


    if ineq_constraint_sqf:
        ineq_constraints = dict(_std_ineq_constraints(*item) for item in ineq_constraints.items())
    ineq_constraints = dict((e, e2) for e, e2 in ineq_constraints.items() if e.total_degree() > 0)

    if eq_constraint_sqf:
        eq_constraints = dict(_std_eq_constraints(*item) for item in eq_constraints.items())
    eq_constraints = dict((e, e2) for e, e2 in eq_constraints.items() if e.total_degree() > 0)

    return poly, ineq_constraints, eq_constraints, symbols


def handle_polynomial(
        ineq_constraint_sqf: bool = True,
        eq_constraint_sqf: bool = True,
    ):
    """
    Wrap a solver function so that it converts sympy expr inputs to sympy polynomials.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(poly: Expr,
                ineq_constraints: Union[List[Expr], Dict[Expr, Expr]] = {},
                eq_constraints: Union[List[Expr], Dict[Expr, Expr]] = {}, *args, **kwargs):

            ################################################################
            #               Convert inputs to sympy polynomials
            ################################################################
            poly, ineq_constraints, eq_constraints, symbols = _polylize(
                poly, ineq_constraints, eq_constraints,
                ineq_constraint_sqf=ineq_constraint_sqf, eq_constraint_sqf=eq_constraint_sqf)

            sol = func(poly, ineq_constraints, eq_constraints, *args, **kwargs)
            return sol
        return wrapper
    return decorator


def sanitize_input(
        homogenize: bool = False,
        infer_symmetry: bool = False,
        wrap_constraints: bool = False
    ):
    """
    Most inner decorator for solver functions. Sanitize input types for internal solvers.

    Parameters
    -----------
    homogenize: bool
        Whether to homogenize the polynomial and inequality and equality constraints by
        introducing a new variable. This is useful for solvers that only accepts homogeneous problems.
    infer_symmetry: bool
        Whether to automatically inferred the symmetry group and convert it to a MonomialManager
        object. This is useful for solvers that needs access to the symmetry of the problem.
    wrap_constraints: bool
        Whether to convert the constraint dictionary {key: value} to {key: value'} to avoid
        collision in the symbols in the new values of the dictionary.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(poly: Poly,
            ineq_constraints: Dict[Poly, Expr],
            eq_constraints: Dict[Poly, Expr], *args, **kwargs
        ):
            symbols = poly.gens
            ################################################################
            #           Homogenize the polynomial and constraints
            ################################################################
            # homogenizer = None
            # is_hom = poly.is_homogeneous and all(e.is_homogeneous for e in ineq_constraints)\
            #     and all(e.is_homogeneous for e in eq_constraints)
            # if (not is_hom) and homogenize:
            #     homogenizer = uniquely_named_symbol('1', 
            #         tuple(set.union(
            #             set(symbols), *(e.free_symbols for e in ineq_constraints.values()), 
            #             *(e.free_symbols for e in eq_constraints.values()))))
            #     poly = poly.homogenize(homogenizer)
            #     ineq_constraints = dict((e.homogenize(homogenizer), e2) for e, e2 in ineq_constraints.items())
            #     ineq_constraints[homogenizer.as_poly(*poly.gens)] = homogenizer
            #     eq_constraints = dict((e.homogenize(homogenizer), e2) for e, e2 in eq_constraints.items())
            #     if '_homogenizer' in signature(func).parameters.keys():
            #         kwargs['_homogenizer'] = homogenizer

            quotient_ring_reduction = reduce_over_quotient_ring(
                poly, ineq_constraints, eq_constraints, homogenize=homogenize
            )
            poly, ineq_constraints, eq_constraints = quotient_ring_reduction['problem']
            homogenizer = quotient_ring_reduction['homogenizer']
            restoration = quotient_ring_reduction['restoration']
            _has_homogenizer_kwarg = signature(func).parameters.get('_homogenizer') is not None
            if _has_homogenizer_kwarg:
                kwargs['_homogenizer'] = homogenizer


            ################################################################
            #                       Infer symmetry
            ################################################################
            symmetry = kwargs.get('symmetry')
            if homogenizer is not None and symmetry is not None:
                # the generators might increase after homogenization
                symmetry = symmetry.perm_group if isinstance(symmetry, MonomialManager) else symmetry
                nvars = len(poly.gens)
                if symmetry.degree != nvars:
                    symmetry = PermutationGroup(*[Permutation(_.array_form + [nvars-1]) for _ in symmetry.args])

            if infer_symmetry and symmetry is None:
                symmetry = identify_symmetry_from_lists(
                    [[poly], list(ineq_constraints.keys()), list(eq_constraints.keys())])

            symmetry = MonomialManager(len(poly.gens), symmetry)

            _has_symmetry_kwarg = signature(func).parameters.get('symmetry') is not None
            if _has_symmetry_kwarg:
                kwargs['symmetry'] = symmetry


            ################################################################
            #                    Wrap constraints
            ################################################################
            constraints_wrapper = None
            if wrap_constraints:
                # wrap ineq/eq constraints to be sympy function class wrt. generators rather irrelevent symbols
                constraints_wrapper = _get_constraints_wrapper(
                    poly.gens, ineq_constraints, eq_constraints, symmetry.perm_group)
                ineq_constraints, eq_constraints = constraints_wrapper[0], constraints_wrapper[1]


            ########################################################
            #               Call the solver function
            ########################################################
            sol: Solution = func(poly, ineq_constraints, eq_constraints, *args, **kwargs)
            if sol is None:
                return None

            if constraints_wrapper is not None:
                # note: first restore the constraints (with homogenizer), and then dehomogenize
                sol = sol.xreplace(constraints_wrapper[2])
                sol = sol.xreplace(constraints_wrapper[3])

            # if homogenizer is not None:
            #     sol = sol.dehomogenize(homogenizer)
            sol = restoration(sol)
            return sol
        return wrapper
    return decorator



def _get_constraints_wrapper(symbols: Tuple[int, ...],
    ineq_constraints: Dict[Poly, Expr], eq_constraints: Dict[Poly, Expr], perm_group: PermutationGroup):
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
                value = Function(name + str(counter))(*compressed)
                dt[permed_base] = value
                inv[value] = permed_expr
            counter += 1
        dt = dict((Poly.new(k, *symbols), v) for k, v in dt.items())
        return dt, inv
    i2g, g2i = _get_dicts(ineq_constraints, name='_G')
    e2h, h2e = _get_dicts(eq_constraints, name='_H')
    return i2g, e2h, g2i, h2e

#########################################################
#
#           Transformations on the Problem
#
#########################################################

def _is_bidegree(p):
    """Check whether a multivariate polynomial has two different degrees.
    Returns l1, l2, sgn such that p = sgn * (l2 - l1). Also,
    l1, l2 are homogeneous, deg(l1) < deg(l2) and l1.LC() > 0. Returns
    None if p does not have the property.
    
    Examples
    ---------
    >>> from sympy.abc import a, b, c
    >>> _is_bidegree((a**2 + b**2 + c**2).as_poly(a,b,c)) is None
    True
    >>> _is_bidegree((2 - 3*(a+b+c) - a*b*c).as_poly(a,b,c)) is None
    True
    >>> _is_bidegree((3*(a+b+c) - a*b*c).as_poly(a,b,c))
    (Poly(3*a + 3*b + 3*c, a, b, c, domain='ZZ'), Poly(a*b*c, a, b, c, domain='ZZ'), -1)
    >>> _is_bidegree((-3*(a+b+c) + a*b*c).as_poly(a,b,c))
    (Poly(3*a + 3*b + 3*c, a, b, c, domain='ZZ'), Poly(a*b*c, a, b, c, domain='ZZ'), 1)
    """
    terms = {}
    for m, c in p.terms():
        d = sum(m)
        if d not in terms:
            if len(terms) == 2:
                return None
            terms[d] = [(m, c)]
        else:
            terms[d].append((m, c))
    if len(terms) != 2:
        return None
    d1, l1 = terms.popitem()
    d2, l2 = terms.popitem()
    if d1 > d2:
        d1, d2 = d2, d1
        l1, l2 = l2, l1
    # d1 < d2
    l1 = Poly(dict(l1), p.gens, domain=p.domain)
    l2 = Poly(dict(l2), p.gens, domain=p.domain)
    if l1.LC() < 0:
        l1 = -l1
        sgn = 1
    else:
        l2 = -l2
        sgn = -1
    return l1, l2, sgn

def _align_degree(p, p1, p2, accept_odd_degree=False):
    """Homogenize p given p1 == p2 and deg(p1) < deg(p2).
    Returns q, d, x such that q = p1**d * p + x(p2 - p1) where q, x are polynomials
    and q is homogeneous. Returns None if it fails"""
    ddiff = p2.total_degree() - p1.total_degree()
    terms = {}
    for m, c in p.terms():
        d = sum(m)
        if d not in terms:
            terms[d] = [(m, c)]
        else:
            terms[d].append((m, c))
    # p = sum(d * polys[d] for d in terms)
    keys = sorted(list(terms.keys()))
    for i in range(len(keys)-1):
        if (keys[i+1]-keys[i])%ddiff != 0:
            return None
    muldeg = (keys[-1] - keys[0])//ddiff
    if muldeg == 0:
        # nothing to do
        return p, 0, Poly({}, p.gens, domain=p.domain)
    if (not accept_odd_degree) and muldeg % 2 != 0:
        # d is not even, might multiply a negative term
        return None
    polys = [Poly(dict(terms[d]), p.gens, domain=p.domain) for d in keys]
    q = Poly({}, p.gens, domain=p.domain)
    for d, poly in zip(keys, polys):
        codeg = (keys[-1] - d)//ddiff
        # poly -> poly * p2**codeg * p1**muldeg
        q += poly * p2**codeg * p1**(muldeg - codeg)
        # x += poly * (p2**codeg - p1**codeg)/(p2 - p1) * p1**(muldeg - codeg)
    divrem = (q - p1**muldeg*p).div(p2 - p1)
    if not divrem[1].is_zero:
        return None
    # print(p, '- (hom) ->', q, muldeg, divrem[0])
    return q, muldeg, divrem[0]


def reduce_over_quotient_ring(poly: Poly,
        ineq_constraints: Dict[Poly, Expr],
        eq_constraints: Dict[Poly, Expr],
        homogenize: bool = False,
    ):
    """
    Perform quotient ring reduction of the problem, including operations like
    homogenization.

    Given equality constraint f(x) = g(x) where f,g are nonzero homogeneous polynomials
    and deg(f) = deg(g), we obtain 1 = (f(x)/g(x)) and can be used for homogenization.

    Parameters
    ----------
    homogenize: bool
        Whether to homogenize the problem by introducing a new variable.

    TODO: move it elsewhere

    Returns a dict containing
    {
        'problem': (new_poly, new_ineq_constraints, new_eq_constraints),
        'restoration': restoration,
        **kwargs
    }
    """
    symbols = poly.gens
    restorations = []
    ################################################################
    #           Homogenize the polynomial and constraints
    ################################################################
    homogenizer = None
    is_hom = poly.is_homogeneous and all(e.is_homogeneous for e in ineq_constraints)\
        and all(e.is_homogeneous for e in eq_constraints)
    if is_hom:
        # nothing to do
        return {
            'problem': (poly, ineq_constraints, eq_constraints),
            'homogenizer': homogenizer,
            'restoration': lambda x: x
        }

    ################################################################
    #         Homogenize using bidegree constraints
    ################################################################

    # tested_bidgree = 0
    for eq, expr in eq_constraints.items():
        bideg = _is_bidegree(eq)
        if bideg is None:
            continue
        p1, p2, sgn = bideg
        # diffdeg = p2.total_degree() - p1.total_degree()
        accept_odd = True if (p1.total_degree() == 0 and p1.LC() > 0) else False
        # print(f'Bidegree: {eq} == 0  <=>  {p1} == {p2}')

        new_poly = _align_degree(poly, p1, p2, accept_odd_degree=accept_odd)
        if new_poly is None:
            continue
        sgn_expr = sgn * expr
        new_ineqs = {}
        new_eqs = {}
        success = True
        for ineq, ineq_expr in ineq_constraints.items():
            new_ineq = _align_degree(ineq, p1, p2, accept_odd_degree=accept_odd)
            if new_ineq is None:
                success = False
                break
            new_ineq, muldeg, quo = new_ineq
            new_ineqs[new_ineq] = (muldeg, ineq_expr, quo)
        if not success:
            continue
        for eq2, eq_expr in eq_constraints.items():
            if eq2 == eq:
                continue
            new_eq = _align_degree(eq2, p1, p2, accept_odd_degree=accept_odd)
            if new_eq is None:
                success = False
                break
            new_eq, muldeg, quo = new_eq
            new_eqs[new_eq] = (muldeg, eq_expr, quo)
        if not success:
            continue

        # update the expression associated with each constraint after homogenization
        symmetry = identify_symmetry_from_lists(
                    [[new_poly[0]], list(new_ineqs.keys()), list(new_eqs.keys())])
        if symmetry.is_trivial:
            symmetry = None
        def p2expr(p: Poly) -> Expr:
            # convert a polynomial to expr wisely by exploting the symmetry
            if (symmetry is not None) and verify_symmetry(p, symmetry):
                p = poly_reduce_by_symmetry(p, symmetry)
                return CyclicSum(p.as_expr(), p.gens, symmetry)
            return p.as_expr()
        p1_expr = p2expr(p1)
        for new_ineq, (muldeg, ineq_expr, quo) in new_ineqs.items():
            new_ineqs[new_ineq] = p1_expr**muldeg * ineq_expr + p2expr(quo)*sgn_expr 
        for new_eq, (muldeg, eq_expr, quo) in new_eqs.items():
            new_eqs[new_eq] = p1_expr**muldeg * eq_expr + p2expr(quo)*sgn_expr

        # homogenize successfully
        poly = new_poly[0]
        is_hom = True
        ineq_constraints, eq_constraints = new_ineqs, new_eqs
        def _align_degree_restore(x):
            if not isinstance(x, Solution): return None
            x.solution = (x.solution - p2expr(new_poly[2])*sgn_expr) / p1_expr**new_poly[1]
            return x
        restorations.append(_align_degree_restore)
        break


    if (not is_hom) and homogenize:
        homogenizer = uniquely_named_symbol('1', 
            tuple(set.union(
                set(symbols), *(e.free_symbols for e in ineq_constraints.values()), 
                *(e.free_symbols for e in eq_constraints.values()))))
        poly = poly.homogenize(homogenizer)
        ineq_constraints = dict((e.homogenize(homogenizer), e2) for e, e2 in ineq_constraints.items())
        ineq_constraints[homogenizer.as_poly(*poly.gens)] = homogenizer
        eq_constraints = dict((e.homogenize(homogenizer), e2) for e, e2 in eq_constraints.items())
        restorations.append(lambda sol: sol.dehomogenize(homogenizer))

    def restoration(sol):
        if sol is None:
            return None
        for rs in restorations[::-1]:
            sol = rs(sol)
        return sol

    return {
        'problem': (poly, ineq_constraints, eq_constraints),
        'homogenizer': homogenizer,
        'restoration': restoration
    }
