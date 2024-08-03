from .cubic import quarternary_cubic_symmetric

from .solver import structural_sos_4vars
from ...solver import SS

__all__ = ['structural_sos_4vars']

_registry = [
    quarternary_cubic_symmetric,
    structural_sos_4vars,
]
SS._register_solver('structsos', 'quarternary', _registry)