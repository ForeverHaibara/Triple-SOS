from .structsos import StructuralSOS, _structural_sos
from .utils import radsimp, quadratic_weighting, zip_longest
from .pivoting import prove_univariate, prove_univariate_interval
from .solution import SolutionStructural, SolutionStructuralSimple
from ..shared import SS

__all__ = [
    'StructuralSOS', 'radsimp', 'quadratic_weighting', 'zip_longest',
    'prove_univariate', 'prove_univariate_interval',
    'SolutionStructural', 'SolutionStructuralSimple'
]

_registry = [
    StructuralSOS, _structural_sos
]
SS._register_solver('structsos', _registry)