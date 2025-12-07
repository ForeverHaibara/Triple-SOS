from typing import Tuple, Dict, List, Union, Optional

import sympy as sp
from sympy import Poly, Expr, Symbol
from sympy.core.symbol import uniquely_named_symbol

def homogenize(poly: Poly, t: Optional[Symbol] = None) -> Tuple[Poly, Symbol]:
    """
    Automatically homogenize a polynomial if it is not homogeneous.

    Parameters
    ----------
    poly : Poly
        The polynomial to homogenize.
    t : Optional[Symbol]
        The symbol to use for homogenization. If None, a new symbol will be created.

    Returns
    ----------
    Tuple[Poly, Symbol]
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


def homogenize_expr_list(expr_list: List[Union[Expr, Poly]], homogenizer: Symbol) -> List[Expr]:
    """
    Homogenize a list of sympy expressions or polynomials.
    """
    symbols = set.union(set(), *[set(e.free_symbols) for e in expr_list])
    if homogenizer in symbols:
        symbols.remove(homogenizer)
    translation = {s: s/homogenizer for s in symbols}
    def hom(x):
        if isinstance(x, Expr):
            x = x.subs(translation).together()
            d = sp.fraction(x)[1].as_poly(homogenizer).degree()
            return x * homogenizer**d
        elif isinstance(x, Poly):
            return x.homogenize(homogenizer)
    return [hom(x) for x in expr_list]
