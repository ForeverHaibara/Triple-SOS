"""
This module provides customized SymPy expressions and classes.
"""

from .coeff import Coeff

from .cyclic import (
    CyclicExpr, CyclicSum, CyclicProduct, SymmetricSum, SymmetricProduct,
    is_cyclic_expr, rewrite_symmetry,
    verify_symmetry, identify_symmetry, identify_symmetry_from_lists,
    arraylize_up_to_symmetry, clear_polys_by_symmetry
)

from .exraw import EXRAW
from .soscone import SOSCone, SOSElement, SOSlist
from .psatz import PSatzDomain, PSatzElement, PSatz


__all__ = [
    'Coeff',
    'CyclicExpr', 'CyclicSum', 'CyclicProduct', 'SymmetricSum', 'SymmetricProduct',
    'is_cyclic_expr', 'rewrite_symmetry',
    'verify_symmetry', 'identify_symmetry', 'identify_symmetry_from_lists',
    'arraylize_up_to_symmetry', 'clear_polys_by_symmetry',
    'EXRAW', 'SOSCone', 'SOSElement', 'SOSlist', 'PSatzDomain', 'PSatzElement', 'PSatz'
]
