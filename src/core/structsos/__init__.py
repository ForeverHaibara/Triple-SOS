from .structsos import StructuralSOS, _structural_sos
from .utils import radsimp, quadratic_weighting, zip_longest
from .solution import SolutionStructural, SolutionStructuralSimple
from ..shared import SS

__all__ = [
    'StructuralSOS', 'radsimp', 'quadratic_weighting', 'zip_longest',
    'SolutionStructural', 'SolutionStructuralSimple'
]

_registry = [
    StructuralSOS, _structural_sos
]
SS._register_solver('structsos', _registry)