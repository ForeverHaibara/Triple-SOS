from .structsos import StructuralSOS, StructuralSOSSolver, _structural_sos
from .utils import radsimp, quadratic_weighting, zip_longest
from .univariate import prove_univariate
from .solution import SolutionStructural
from ..shared import SS

__all__ = [
    'StructuralSOS', 'StructuralSOSSolver', 'radsimp', 'quadratic_weighting', 'zip_longest',
    'prove_univariate', 'SolutionStructural'
]

_registry = [
    StructuralSOS, _structural_sos
]
SS._register_solver('structsos', _registry)
