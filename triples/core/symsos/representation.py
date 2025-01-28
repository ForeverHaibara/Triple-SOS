from typing import Tuple, Dict, List

import sympy as sp

from .basic import SymmetricTransform
from .symmetric import SymmetricPositive, SymmetricReal
from ...utils import Coeff

_METHOD_TO_TRANSFORM = {
    'real': SymmetricReal,
    'positive': SymmetricPositive
}

def _get_transform_from_method(method: str, nvars: int) -> SymmetricTransform:
    if method not in _METHOD_TO_TRANSFORM:
        raise ValueError(f"Unknown method {method}.")
    return _METHOD_TO_TRANSFORM[method]


def sym_representation(poly, symbols, return_poly = False, method = 'real'):
    """
    Represent a polynoimal to the symmetric form.

    Please refer to functions `_sym_representation_positive` and `_sym_representation_real`
    for the details of the representation.
    
    Parameters
    ----------
    poly : sympy.Poly
        The polynomial to be represented. Could be either a polynomial of (a,b,c)
        or its pqr representation.
    symbols : Tuple[sympy.Symbol]
    return_poly : bool
        If True, ...
    method : str
        'real' or 'positive'.

    Returns
    -------
    sympy.Expr
        The polynomial in the new representation.
    """
    if not poly.is_homogeneous:
        raise ValueError("The polynomial must be homogeneous.")

    trans = _get_transform_from_method(method, len(poly.gens))

    return trans.transform(poly, symbols, return_poly=return_poly)


def sym_representation_inv(expr, original_symbols, new_symbols, method = 'real'):
    trans = _get_transform_from_method(method, len(original_symbols))
    return trans.inv_transform(expr, original_symbols=original_symbols, new_symbols=new_symbols)


def sym_transform(poly: sp.Poly, ineq_constraints: Dict[sp.Poly, sp.Expr], eq_constraints: Dict[sp.Poly, sp.Expr],
                symbols: List[sp.Symbol], deparametrize: bool = False, method: str = 'real') -> Tuple[sp.Poly, Dict[sp.Poly, sp.Expr], Dict[sp.Poly, sp.Expr], sp.Expr]:
    """
    Transform a symmetric inequality problem along with its constraints.
    """
    trans = _get_transform_from_method(method, len(poly.gens))
    numerator, denominator = trans.transform(poly, symbols, return_poly=True)

    ineq_constraints2 = dict()
    eq_constraints2 = dict()

    for collection, new_collection in ((ineq_constraints, ineq_constraints2), 
                                       (eq_constraints, eq_constraints2)):
        for p, value in collection.items():
            pgens = p.gens
            p = p.as_poly(*poly.gens)
            if not Coeff(p).is_symmetric():
                continue
            n, d = trans.transform(p, symbols, return_poly=True)
            new_collection[n] = value * d

    extra_ineq, extra_eq = trans.get_default_constraints(symbols)
    ineq_constraints2.update(extra_ineq)
    eq_constraints2.update(extra_eq)

    return numerator, ineq_constraints2, eq_constraints2, denominator