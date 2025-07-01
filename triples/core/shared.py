from typing import Tuple, Dict, List, Union, Optional

import sympy as sp
from sympy.core.symbol import uniquely_named_symbol

from ..utils import MonomialManager

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


def clear_polys_by_symmetry(polys: List[Union[sp.Expr, Tuple[sp.Expr, ...]]],
        symbols: Tuple[sp.Symbol, ...], symmetry: MonomialManager) -> List[Union[sp.Expr, Tuple[sp.Expr, ...]]]:
    """
    Remove duplicate polys by symmetry.
    """
    if symmetry.is_trivial:
        return polys if isinstance(polys, list) else list(polys)

    base = symmetry.base()
    def _get_representation(t: sp.Expr):
        """Get the standard representation of the poly given symmetry."""
        t = sp.Poly(t, symbols) if not isinstance(t, tuple) else sp.Poly(t[0], symbols)
        # if t.is_monomial and len(t.free_symbols) == 1:
        #     return None
        vec = base.arraylize_sp(t)
        shape = vec.shape[0]

        rep = vec._rep.rep.to_list_flat() # avoid conversion from rep to sympy
        # getvalue = lambda i: vec[i,0] # get a single value
        getvalue = lambda i: rep[i]
        if shape <= 1:
            return tuple(getvalue(i) for i in range(shape))

        # # The naive implementation below could take minutes for calling 50 times on a 6-order group
        # mat = symmetry.permute_vec(vec, t.total_degree())
        # cols = [tuple(mat[:, i]) for i in range(mat.shape[1])]
        # return max(cols)

        # We should highly optimize the algorithm.
        dict_monoms = base.dict_monoms(t.total_degree())
        inv_monoms = base.inv_monoms(t.total_degree())

        # get the value of index i in the vector after permutation
        v = lambda perm, i: getvalue(dict_monoms[tuple(perm(inv_monoms[i]))])

        perms = list(symmetry.perm_group.elements)
        queue, queue_len, best_perm = [0]*shape, 0, perms[0]
        for perm in perms[1:]:
            for j in range(shape):
                s = v(perm, j)
                if j >= queue_len:
                    # compare the next element
                    queue[j] = v(best_perm, j)
                    queue_len += 1
                if s > queue[j]:
                    queue[j], queue_len, best_perm = s, j + 1, perm
                    break
                elif s < queue[j]:
                    break
        for j in range(queue_len, shape): # fill the rest
            queue[j] = v(best_perm, j)
        return tuple(queue)

 
    representation = dict(((_get_representation(t), t) for i, t in enumerate(polys)))
    if None in representation:
        del representation[None]
    return list(representation.values())
