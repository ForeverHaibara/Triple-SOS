from inspect import signature
from typing import Tuple, Dict, List, Union, Optional

import sympy as sp
from sympy import sympify, Function
from sympy.core.symbol import uniquely_named_symbol
from sympy.combinatorics import Permutation, PermutationGroup, SymmetricGroup, AlternatingGroup, CyclicGroup

from ..utils.expression.form import _reduce_factor_list
from ..utils.expression import SolutionSimple, CyclicExpr
from ..utils.basis_generator import MonomialReduction, MonomialFull, MonomialPerm, MonomialHomogeneousFull

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

    class PseudoPoly():
        def __init__(self, polys, gens, d):
            self.polys = polys
            self.gens = gens
            self.order = d
        def factor_list(self,*args,**kwargs):
            return (sp.S(1), [(p, self.order) for p in self.polys])
    def check_symmetry(polys, perm_group):
        r = _reduce_factor_list(PseudoPoly(polys, gens, perm_group.order()), perm_group)
        return len(r[1]) == 0

    # List a few candidates: symmetric, alternating, cyclic groups...
    nvars = len(gens)
    if nvars > 1:
        candidates = [SymmetricGroup(nvars)]
        if nvars > 3:
            candidates.append(AlternatingGroup(nvars))
        if nvars > 2:
            candidates.append(CyclicGroup(nvars))
        if has_homogenized and nvars > 2:
            # exclude the homogenizer
            candidates.append(SymmetricGroup(nvars).stabilizer(nvars - 1))
            candidates.append(PermutationGroup(Permutation(list(range(1, nvars - 1)) + [0, nvars - 1])))
        for group in candidates:
            if all(check_symmetry(l, group) for l in lst_of_lsts):
                return group

    return PermutationGroup(Permutation(list(range(nvars))))


def clear_polys_by_symmetry(polys: List[Union[sp.Expr, Tuple[sp.Expr, ...]]],
        symbols: Tuple[sp.Symbol, ...], symmetry: MonomialReduction) -> List[Union[sp.Expr, Tuple[sp.Expr, ...]]]:
    """
    Remove duplicate polys by symmetry.
    """
    if isinstance(symmetry, MonomialFull):
        return polys if isinstance(polys, list) else list(polys)

    def _get_representation(t: sp.Expr):
        """Get the standard representation of the poly given symmetry."""
        t = sp.Poly(t, symbols) if not isinstance(t, tuple) else sp.Poly(t[0], symbols)
        # if t.is_monomial and len(t.free_symbols) == 1:
        #     return None
        vec = symmetry.base().arraylize_sp(t)
        mat = symmetry.permute_vec(len(symbols), vec)
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
    e = sp.Pow(e, 1/max_d, evaluate=False)
    if c < 0:
        e = e.__neg__()
    return ret, e

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

            poly = sp.Poly(poly, *symbols)
            ineq_constraints = dict((sp.Poly(e, *symbols), e2) for e, e2 in ineq_constraints.items())
            eq_constraints = dict((sp.Poly(e, *symbols), e2) for e, e2 in eq_constraints.items())

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
                if isinstance(symmetry, MonomialPerm):
                    symmetry = symmetry.perm_group
                nvars = len(poly.gens)
                if symmetry.degree != nvars:
                    symmetry = PermutationGroup(*[Permutation(_.array_form + [nvars-1]) for _ in symmetry.args])

            if infer_symmetry and symmetry is None:
                symmetry = identify_symmetry_from_lists(
                    [[poly], list(ineq_constraints.keys()), list(eq_constraints.keys())], has_homogenized=homogenizer is not None)
            if symmetry is not None and not isinstance(symmetry, MonomialReduction):
                symmetry = MonomialPerm(symmetry)
            elif symmetry is None:
                symmetry = MonomialHomogeneousFull()

            _has_symmetry_kwarg = signature(func).parameters.get('symmetry') is not None
            if _has_symmetry_kwarg:
                kwargs['symmetry'] = symmetry

            constraints_wrapper = None
            if wrap_constraints:
                # wrap ineq/eq constraints to be sympy function class wrt. generators rather irrelevent symbols
                constraints_wrapper = _get_constraints_wrapper(poly.gens, ineq_constraints, eq_constraints, symmetry.to_perm_group(len(poly.gens)))
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
    i2g = {ineq: Function("_G%d"%i)(*symbols) for i, ineq in enumerate(ineq_constraints.keys())}
    e2h = {eq: Function("_H%d"%i)(*symbols) for i, eq in enumerate(eq_constraints.keys())}

    def _get_inverse(constraints, name='_G'):
        inv = dict()
        rep_dict = dict((p.rep, v) for p, v in constraints.items())
        for p in perm_group.elements:
            reorder = p(symbols)
            invorder = p.__invert__()(symbols)
            for i, base in enumerate(constraints.keys()):
                permed_base = base.reorder(*invorder).rep
                permed_poly = rep_dict.get(permed_base)
                if permed_poly is None:
                    raise ValueError("Given constraints are not symmetric with respect to the permutation group.")
                inv[Function(name + str(i))(*reorder)] = permed_poly
        return inv
    g2i = _get_inverse(ineq_constraints, name='_G')
    h2e = _get_inverse(eq_constraints, name='_H')
    return i2g, e2h, g2i, h2e