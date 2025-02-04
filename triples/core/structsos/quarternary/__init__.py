from .cubic import quarternary_cubic_symmetric
from .quartic import quarternary_quartic

from .solver import structural_sos_4vars
from ...shared import SS

__all__ = ['structural_sos_4vars']

_registry = [
    quarternary_cubic_symmetric,
    quarternary_quartic,
    structural_sos_4vars,
]
SS._register_solver('structsos', 'quarternary', _registry)