from typing import Tuple, Dict, List, Union, Optional

from sympy import Poly, Expr, Symbol

from .basic import SymmetricTransform
from .symmetric import UE3Real, UE3Positive
from ...utils import verify_symmetry

_METHOD_TO_TRANSFORM = {
    'real': UE3Real,
    'positive': UE3Positive
}

def _get_transform_from_method(method: str, nvars: int) -> SymmetricTransform:
    if method not in _METHOD_TO_TRANSFORM:
        raise ValueError(f"Unknown method {method}.")
    return _METHOD_TO_TRANSFORM[method]


def sym_representation(
    poly: Poly,
    symbols: Optional[Tuple[Symbol, ...]] = None,
    return_poly: bool = False,
    method: str = 'real'
) -> Union[Expr, Tuple[Poly, Expr]]:
    """
    Represent a polynoimal to the symmetric form.

    Please refer to functions `_sym_representation_positive` and `_sym_representation_real`
    for the details of the representation.

    Parameters
    ----------
    symbols : Tuple[Symbol, ...]
        List of symbols used in the polynomial representation.
    return_poly : bool, optional
        If False, returns a symbolic expression.
        If True, returns a tuple (numerator, denominator) where numerator is a polynomial
        object in the new symbols, while the denominator is a sympy expression that is
        ensured to be positive semidefinite.
        Default is False.
    method : str, optional
        Method to use for the transformation. Can be either 'real' or 'positive'.
        'real' uses standard symmetric representation.
        'positive' uses representation suitable for positive variables.
        Default is 'real'.

    Returns
    ----------
    (Poly, Expr) or Expr

    Examples
    ----------
    >>> from sympy.abc import a, b, c, x, y, z
    >>> sym_representation((a**2*(a-b)*(a-c)+b**2*(b-c)*(b-a)+c**2*(c-a)*(c-b)).as_poly(a,b,c), (x,y,z), method='real')
    2*(4*z + (x + 2*y)**2)/(9*(Σ((a - b)**2)))
    >>> sym_representation((a**4*(a-b)*(a-c)+b**4*(b-c)*(b-a)+c**4*(c-a)*(c-b)).as_poly(a,b,c), (x,y,z), method='real', return_poly=True)
    (Poly(1/81*x**4 + 8/81*x**3*y + 8/27*x**2*y**2 + 8/27*x**2*z + 32/81*x*y**3 + 32/81*x*y*z + 16/81*y**4 + 28/81*y**2*z + 4/27*z**2, x, y, z, domain='QQ'), (Σ((a - b)**2))**3/8)
    """
    if not poly.is_homogeneous:
        raise ValueError("The polynomial must be homogeneous.")
    if symbols is None:
        symbols = poly.gens
    trans = _get_transform_from_method(method, len(poly.gens))
    return trans.transform(poly, symbols, return_poly=return_poly)


def sym_representation_inv(
    expr: Expr,
    original_symbols: Tuple[Symbol, ...],
    new_symbols: Tuple[Symbol, ...],
    method: str = 'real'
) -> Expr:
    """
    Compute the inverse transformation of a symbolic expression based on a specified method.
    This function takes a symbolic expression and reverses a previously applied
    transformation, restoring it to its original form using the provided original
    and new symbols.

    Parameters
    ----------
    expr : Expr
        The symbolic expression to be inversely transformed.
    original_symbols : Tuple[Symbol, ...]
        The list of original symbols in the expression before transformation.
    new_symbols : Tuple[Symbol, ...]
        The list of new symbols currently used in the transformed expression.
    method : str, optional
        The transformation method to use. Defaults to 'real'. Must be a valid
        method recognized by the transform object.

    Returns
    --------
    Expr
        The symbolic expression after applying the inverse transformation to
        restore it to its original representation.

    Examples
    --------
    >>> from sympy.abc import a, b, c, x, y, z
    >>> sym_representation_inv(2*(4*z + (x + 2*y)**2), (a,b,c), (x,y,z))
    2*(∏(-a - b + 2*c) + (Σ(a))*(Σ((a - b)**2))/2)**2 + 54*(∏((a - b)**2))
    """
    trans = _get_transform_from_method(method, len(original_symbols))
    return trans.inv_transform(expr, original_symbols=original_symbols, new_symbols=new_symbols)


def sym_transform(
    poly: Poly,
    ineq_constraints: Dict[Poly, Expr],
    eq_constraints: Dict[Poly, Expr],
    symbols: Tuple[Symbol, ...],
    deparametrize: bool = False,
    method: str = 'real'
) -> Tuple[Poly, Dict[Poly, Expr], Dict[Poly, Expr], Expr]:
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
            if not verify_symmetry(p, "sym"):
                continue
            n, d = trans.transform(p, symbols, return_poly=True)
            new_collection[n] = value * d

    extra_ineq, extra_eq = trans.get_default_constraints(symbols)
    ineq_constraints2.update(extra_ineq)
    eq_constraints2.update(extra_eq)

    return numerator, ineq_constraints2, eq_constraints2, denominator
