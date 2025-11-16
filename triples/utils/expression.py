from .expressions import (
    Coeff, CyclicExpr, CyclicSum, CyclicProduct, SymmetricSum, SymmetricProduct, is_cyclic_expr,
    rewrite_symmetry,
    verify_symmetry, identify_symmetry, identify_symmetry_from_lists,
    arraylize_up_to_symmetry, clear_polys_by_symmetry,
    SOSlist, PSatz
)

__all__ = [
    'Coeff',
    'CyclicExpr', 'CyclicSum', 'CyclicProduct', 'SymmetricSum', 'SymmetricProduct',
    'is_cyclic_expr', 'rewrite_symmetry',
    'verify_symmetry', 'identify_symmetry', 'identify_symmetry_from_lists',
    'arraylize_up_to_symmetry', 'clear_polys_by_symmetry',
    'SOSlist', 'PSatz'
]

from warnings import warn

warn("expression.py is deprecated. Use expressions.py instead.", DeprecationWarning,
    stacklevel=2)
