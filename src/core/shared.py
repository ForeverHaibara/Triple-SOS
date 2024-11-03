from inspect import signature
from typing import Tuple, Optional, List, Union

import sympy as sp
from sympy.core.symbol import uniquely_named_symbol
from sympy.combinatorics import Permutation, PermutationGroup, SymmetricGroup, AlternatingGroup, CyclicGroup

from ..utils.expression.form import _reduce_factor_list
from ..utils.expression.solution import SolutionSimple
from ..utils.basis_generator import MonomialReduction, MonomialFull, MonomialPerm

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


def clear_polys_by_symmetry(polys: List[sp.Expr], symbols: Tuple[sp.Symbol, ...], symmetry: MonomialReduction) -> List[sp.Expr]:
    """
    Remove duplicate polys by symmetry.
    """
    if isinstance(symmetry, MonomialFull):
        return polys

    def _get_representation(t: sp.Expr):
        """Get the standard representation of the poly given symmetry."""
        t = t.as_poly(symbols)
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



def _sqf_part(p: sp.Poly, discard_square: bool = True) -> sp.Poly:
    """Get the square-free part of a polynomial. Keep the sign."""
    if p.is_zero: return p
    c, lst = p.sqf_list()
    ret = sp.S(1 if c > 0 else -1).as_poly(*p.gens, domain=p.domain)
    for q, d in lst:
        if (not discard_square) or d % 2 == 1:
            ret *= q
    return ret



def sanitize_input(
        homogenize: bool = False,
        ineq_constraint_sqf: bool = True,
        eq_constraint_sqf: bool = True,
        infer_symmetry: bool = False
    ):
    """
    Decorator for sum of square functions. It sanitizes the input
    so that each input is a polynomial.
    """
    def decorator(func):
        def wrapper(poly: sp.Expr, ineq_constraints: List[sp.Expr] = [], eq_constraints: List[sp.Expr] = [], *args, **kwargs):

            original_symbols = [] if not isinstance(poly, sp.Poly) else poly.gens
            symbols = set.union(set(poly.free_symbols), *[set(e.free_symbols) for e in ineq_constraints + eq_constraints])
            if len(symbols) == 0:
                symbols = {sp.Symbol('x')}
                # raise ValueError('No symbols found in the input.')
            symbols = symbols - set(original_symbols)
            symbols = tuple(sorted(list(symbols), key=lambda x: x.name))
            symbols = tuple(original_symbols) + symbols

            poly = sp.Poly(poly, *symbols)
            ineq_constraints = [sp.Poly(e, *symbols) for e in ineq_constraints]
            eq_constraints = [sp.Poly(e, *symbols) for e in eq_constraints]

            homogenizer = None
            is_hom = poly.is_homogeneous
            is_hom = is_hom and all(e.is_homogeneous for e in ineq_constraints) and all(e.is_homogeneous for e in eq_constraints)
            if (not is_hom) and homogenize:
                homogenizer = uniquely_named_symbol('t', symbols)
                poly = poly.homogenize(homogenizer)
                ineq_constraints = [e.homogenize(homogenizer) for e in ineq_constraints]
                ineq_constraints.append(homogenizer.as_poly(*poly.gens))
                eq_constraints = [e.homogenize(homogenizer) for e in eq_constraints]
                if '_homogenizer' in signature(func).parameters.keys():
                    kwargs['_homogenizer'] = homogenizer

            if ineq_constraint_sqf:
                ineq_constraints = [_sqf_part(e) for e in ineq_constraints]
            ineq_constraints = [e for e in ineq_constraints if e.total_degree() > 0]

            if eq_constraint_sqf:
                eq_constraints = [_sqf_part(e, discard_square = False) for e in eq_constraints]
            eq_constraints = [e for e in eq_constraints if e.total_degree() > 0]

            if infer_symmetry and signature(func).parameters.get('symmetry') is not None:
                symmetry = kwargs.get('symmetry')
                if symmetry is None:
                    symmetry = identify_symmetry_from_lists([[poly], ineq_constraints, eq_constraints], has_homogenized=homogenizer is not None)
                if not isinstance(symmetry, MonomialReduction):
                    symmetry = MonomialPerm(symmetry)
                kwargs['symmetry'] = symmetry

            # Call the solver function
            sol: SolutionSimple = func(poly, ineq_constraints, eq_constraints, *args, **kwargs)
            if sol is None:
                return None

            if (not is_hom) and homogenize:
                sol = sol.dehomogenize(homogenizer)
            return sol
        return wrapper
    return decorator