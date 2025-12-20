from .lift import lift_degree, _get_multipliers, LinearBasisMultiplier

from warnings import warn
warn("module updegree has been renamed to lift.", DeprecationWarning,
    stacklevel=2)

__all__ = ['lift_degree', '_get_multipliers', 'LinearBasisMultiplier']