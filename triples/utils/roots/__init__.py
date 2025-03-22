from .roots import Root

from .polysolve import univar_realroots, nroots

from .extrema import kkt, optimize_poly

from .findroot import (
    findroot, find_nearest_root, findroot_resultant
)

from .rationalize import (
    univariate_intervals, rationalize, rationalize_array, rationalize_bound,
    rationalize_quadratic_curve, common_region_of_conics, square_perturbation,
    cancel_denominator
)

from .monotonic_opt import rpa_monotonic, rpa_gmop, poly_as_dm, rpa_polyopt

__all__ = [
    'Root',
    'univar_realroots', 'nroots',
    'optimize_poly',
    'findroot', 'find_nearest_root', 'findroot_resultant', 'kkt',
    'nroots', 'univariate_intervals', 'rationalize', 'rationalize_array', 'rationalize_bound',
    'rationalize_quadratic_curve', 'common_region_of_conics', 'square_perturbation',
    'cancel_denominator',
    'rpa_monotonic', 'rpa_gmop', 'poly_as_dm', 'rpa_polyopt'
]
