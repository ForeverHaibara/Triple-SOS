from .solver import structural_sos_constrained

from ...shared import SS

__all__ = ['structural_sos_constrained']

_registry = [
    structural_sos_constrained
]

SS._register_solver('structsos', 'constrained', _registry)