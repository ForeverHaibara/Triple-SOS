"""
Extra utility functions for graphic user interfaces and deployment.
"""
from ast import literal_eval
from functools import partial, wraps
from typing import Tuple, List, Dict, Optional, Any, Union, Callable

from sympy import Expr, Poly, Symbol, sympify
from sympy.combinatorics import Permutation, PermutationGroup

from .grid import GridRender
from ..utils.text_process import (
    preprocess_text, poly_get_factor_form, poly_get_standard_form,
    degree_of_expr, coefficient_triangle
)
from ..core import Solution, sum_of_squares
from ..core.sum_of_squares import StructuralSOSSolver, LinearSOSSolver, SDPSOSSolver


class SOSManager:
    """
    Helper class for graphic user interfaces and deployment,
    with sanity checks and error handling.

    The APIs might change and is not expected to
    be used by the end users.
    """
    verbose = True
    time_limit = 3600.0
    configs = {}

    # Configuration parameters
    # HEATMAP_3VARS_DPI = 60
    # HEATMAP_4VARS_DPI = 18

    @classmethod
    def _restrict_input(cls,
        text: str
    ) -> bool:
        if not isinstance(text, str):
            # we think that this is already converted
            # and just ignore
            return True
        is_safe_char = lambda x: x <= 1023 or (x in (65288,65289))
        unsafe = [c for c in text if not is_safe_char(ord(c))]
        if any(unsafe):
            raise ValueError(f"Input contains unsafe characters {set(unsafe)}.")
        return True

    @classmethod
    def make_parser(cls,
        gens: Union[str, Tuple[Symbol, ...]],
        symmetry: Union[str, PermutationGroup],
        *,
        lowercase: bool = True,
        cyclic_sum_func: str = "s",
        cyclic_prod_func: str = "p",
        preserve_patterns: Union[str, List[str]] = ("sqrt",),
        scientific_notation: bool = False,
    ) -> Callable:
        if isinstance(gens, str):
            gens = tuple(Symbol(_) for _ in gens)
            if len(set(gens)) != len(gens):
                raise ValueError("Duplicate generators are not allowed.")
        if isinstance(symmetry, str):
            symmetry = cls.parse_perm_group(symmetry)
            if symmetry.degree != len(gens):
                raise ValueError("The degree of the permutation group"\
                                 + "must be equal to the number of generators.")
        if isinstance(preserve_patterns, str):
            preserve_patterns = [_.strip() for _ in preserve_patterns.split(',')]
            preserve_patterns = [_ for _ in preserve_patterns if _]
       
        func = partial(preprocess_text,
            gens=gens,
            symmetry=symmetry,
            lowercase=lowercase,
            cyclic_sum_func=cyclic_sum_func,
            cyclic_prod_func=cyclic_prod_func,
            preserve_patterns=preserve_patterns,
            scientific_notation=scientific_notation,
        )
        @wraps(func)
        def wrapped_parser(text, *args, **kwargs):
            if not cls._restrict_input(text):
                raise ValueError("Input is unsafe.")
            return func(text, *args, **kwargs)
        return wrapped_parser

    @classmethod
    def apply_transformations(cls,
        expr: Union[Poly, Tuple[Poly, Poly]],
        gens: Tuple[Symbol, ...],
        perm_group: PermutationGroup,
        *,
        cancel: bool = True,
        homogenize: bool = False,
        dehomogenize: bool = None,
        standardize_text: Optional[Union[str, bool]] = None,
        cyclic_sum_func: str = "s",
        cyclic_prod_func: str = "p",
        omit_mul: bool = True,
        omit_pow: bool = True,
    ) -> Tuple[Poly, str]:
        if standardize_text is False:
            standardize_text = None

        if cancel:
            if isinstance(expr, tuple):
                # discard the denominator if required
                if expr[1] != Poly(1, expr[1].gens, domain=expr[1].domain) \
                        and (standardize_text is None):
                    # the denominator is not one, and the expression is modified
                    # -> recompute the expression
                    standardize_text = "sort"
                expr = expr[0]

        if isinstance(expr, Poly):
            poly = expr
            if homogenize and (not poly.is_homogeneous) and poly.degree(len(gens)-1) == 0:
                gen = poly.gens[-1]
                poly = poly.eval(gen, 0).homogenize(gen)
                if standardize_text is None:
                    standardize_text = "sort"

            if dehomogenize is not None and dehomogenize is not False:
                dehomogenize_val = sympify(dehomogenize)
                if len(dehomogenize_val.free_symbols) == 0:  # dehomogenize is a constant
                    gens = poly.gens
                    for i in range(len(gens) - 1, -1, -1):
                        if poly.degree(i) != 0:
                            poly = poly.eval(gens[i], dehomogenize_val)
                            poly = poly.as_poly(*gens, domain=poly.domain)
                            if standardize_text is None:
                                standardize_text = "sort"
                            break
            expr = poly

        elif isinstance(expr, tuple):
            pass

        text = None
        if standardize_text:
            func = None
            if standardize_text == "factor":
                func = poly_get_factor_form
            elif standardize_text == "sort":
                func = poly_get_standard_form
            elif standardize_text == "expand":
                func = poly_get_standard_form
                perm_group = "trivial" # warning: unsafe

            if func is not None:
                text = func(poly, perm_group,
                    cyclic_sum_func=cyclic_sum_func,
                    cyclic_prod_func=cyclic_prod_func,
                    omit_mul=omit_mul, omit_pow=omit_pow
                )

        return expr, text

    @classmethod
    def render_coeff_triangle_and_heatmap(cls,
        raw_expr: Expr,
        expr: Union[Poly, Tuple[Poly, Poly]],
        return_grid: bool = False
    ) -> Tuple[Optional[int], Optional[list], Optional[list]]:
        if isinstance(expr, tuple):
            # discard the denominator if required
            expr = expr[0]
        if not isinstance(expr, Poly):
            return None, None, None

        poly = expr
        degree = poly.total_degree()
        if poly.is_zero:
            degree = degree_of_expr(raw_expr, poly.gens)

        heatmap = None
        triangle = None
        if len(poly.gens) == 3 or len(poly.gens) == 4:
            triangle = coefficient_triangle(poly, degree)
            if poly.domain.is_Numerical and poly.is_homogeneous:
                size = 60 if len(poly.gens) == 3 else 18
                grid = GridRender.render(poly, size=size, with_color=True)
                if return_grid:
                    heatmap = grid
                else:
                    heatmap = grid.grid_color if grid is not None else None

        return degree, triangle, heatmap

    @classmethod
    def parse_constraints_dict(cls,
        source: Dict[str, str],
        parser: Callable[[str], Expr]
    ) -> Dict[Expr, Expr]:
        constraints = {}
        for key, value in source.items():
            key, value = key.strip(), value.strip()
            if len(key) == 0:
                continue
            key = parser(key, return_type = "expr")
            if len(value) != 0:
                # we do not use parser here
                value = sympify(value)
            else:
                value = key
            constraints[key] = value
        return constraints

    @classmethod
    def sum_of_squares(cls,
        expr: Expr,
        ineq_constraints: Dict[Expr, Expr] = {},
        eq_constraints: Dict[Expr, Expr] = {},
        time_limit: Optional[float] = time_limit,
        methods: Optional[List[str]] = None,
        configs: Dict[str, Any] = configs
    ) -> Optional[Solution]:
        if expr is None:
            return None

        if time_limit > cls.time_limit:
            time_limit = cls.time_limit

        # perhaps this conversion should be moved to `sum_of_squares`
        configs = configs.copy()
        mappings = {
            'StructuralSOS': StructuralSOSSolver,
            'LinearSOS': LinearSOSSolver,
            'SDPSOS': SDPSOSSolver,
        }
        for key in mappings:
            if key in configs:
                configs[mappings[key]] = configs.pop(key)

        solution = sum_of_squares(
            expr,
            ineq_constraints,
            eq_constraints,
            methods=methods,
            verbose=cls.verbose,
            time_limit=time_limit,
            configs=configs
        )

        return solution

    @classmethod
    def parse_perm_group(cls, 
        text: Union[str, List[List[int]]]
    ) -> Optional[PermutationGroup]:
        if isinstance(text, str):
            text = literal_eval(text)

        if isinstance(text, list):
            return PermutationGroup(*(Permutation(perm) for perm in text))

        if isinstance(text, PermutationGroup):
            return text
        raise TypeError(f"Invalid permutation group format: {text}")


def render_latex(
    latex_str: str,
    path: str,
    usetex: bool = True,
    show: bool = False,
    dpi: int = 500,
    fontsize: int = 20
) -> str:
    """
    Render a LaTeX string to an image file.

    Args:
        latex_str: LaTeX string to render
        path: Output file path
        usetex: Whether to use LaTeX for rendering
        show: Whether to display the rendered image
        dpi: Resolution of the output image
        fontsize: Font size for the LaTeX text
    """
    import matplotlib.pyplot as plt

    original_str = latex_str

    # Create a small figure that will be expanded when saving
    plt.figure(figsize=(0.3, 0.3))

    if usetex:
        try:
            # Format the LaTeX string for display
            latex_str = f'$\\displaystyle {latex_str.strip("$")} $'
            plt.text(-0.3, 0.9, latex_str, fontsize=fontsize, usetex=usetex)
        except Exception:
            # Fall back to non-LaTeX rendering if there's an error
            usetex = False

    if not usetex:
        # Simple text rendering without LaTeX
        latex_str = original_str.strip("$").split("\\\\")
        latex_str = "\\n".join([f" $ {line} $ " for line in latex_str])
        plt.text(-0.3, 0.9, latex_str, fontsize=fontsize, usetex=False)

    # Set limits and hide axes
    plt.ylim(0, 1)
    plt.xlim(0, 6)
    plt.axis('off')

    # Save the figure with tight bounding box
    plt.savefig(path, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return path
