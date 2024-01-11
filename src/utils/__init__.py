from .basis_generator import generate_expr, arraylize, arraylize_sp

from .text_process import preprocess_text, pl

from .polytools import deg, verify_hom_cyclic, verify_is_symmetric, convex_hull_poly

from .expression import (
    CyclicSum, CyclicProduct, is_cyclic_expr,
    poly_get_factor_form, poly_get_standard_form,
    latex_coeffs,
    Solution, SolutionSimple,
    congruence,
)

from .roots import (
    RootsInfo, Root, RootAlgebraic, RootRational, RootUV,
    GridRender,
    findroot, find_nearest_root, findroot_resultant, nroots,
    root_tangents,
    rationalize, rationalize_array, rationalize_bound,
    rationalize_quadratic_curve,
    square_perturbation,
    cancel_denominator,
)