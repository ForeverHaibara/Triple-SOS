from functools import wraps
from inspect import signature
from typing import List, Dict, Union, Tuple

from sympy import __version__ as SYMPY_VERSION
from sympy import Expr, Poly, Symbol, Integer, Pow, Function
from sympy.core.symbol import uniquely_named_symbol
from sympy.combinatorics import PermutationGroup, Permutation

from ...utils import Solution, MonomialManager, identify_symmetry_from_lists
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
    homogenize: bool = False,
    ineq_constraint_sqf: bool = True,
    eq_constraint_sqf: bool = True,
    infer_symmetry: bool = False,
    wrap_constraints: bool = False
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


            ################################################################
            #           Homogenize the polynomial and constraints
            ################################################################
            homogenizer = None
            is_hom = poly.is_homogeneous and all(e.is_homogeneous for e in ineq_constraints)\
                and all(e.is_homogeneous for e in eq_constraints)
            if (not is_hom) and homogenize:
                homogenizer = uniquely_named_symbol('1', 
                    tuple(set.union(
                        set(symbols), *(e.free_symbols for e in ineq_constraints.values()), 
                        *(e.free_symbols for e in eq_constraints.values()))))
                poly = poly.homogenize(homogenizer)
                ineq_constraints = dict((e.homogenize(homogenizer), e2) for e, e2 in ineq_constraints.items())
                ineq_constraints[homogenizer.as_poly(*poly.gens)] = homogenizer
                eq_constraints = dict((e.homogenize(homogenizer), e2) for e, e2 in eq_constraints.items())
                if '_homogenizer' in signature(func).parameters.keys():
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
            if homogenizer is not None:
                sol = sol.dehomogenize(homogenizer)
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