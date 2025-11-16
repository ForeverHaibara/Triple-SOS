from typing import Tuple, Dict, List, Union, Optional

import sympy as sp
from sympy.core.symbol import uniquely_named_symbol

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
