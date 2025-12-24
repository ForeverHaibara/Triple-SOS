"""
Extra utility functions for graphic user interfaces and deployment.
"""
from ast import literal_eval
from typing import Tuple, List, Dict, Optional, Any, Union

from sympy import Expr, Poly, Symbol, sympify
from sympy.combinatorics import Permutation, PermutationGroup, CyclicGroup

from .grid import GridRender
from ..utils.text_process import (
    preprocess_text, poly_get_factor_form, poly_get_standard_form,
    degree_of_zero, coefficient_triangle, coefficient_triangle_latex
)
from ..core import Solution, sum_of_squares


class SOSManager:
    """
    A convenient class to manage the sum of squares decomposition of a polynomial,
    providing commonly used functions and properties.

    It adds more sanity checks and error handling to the core functions.
    """
    verbose = True
    time_limit = 300.0
    configs = {}

    # Configuration parameters
    DEFAULT_GENS = tuple(Symbol(_) for _ in "abc")
    DEFAULT_PERM_GROUP = CyclicGroup(3)
    RESTRICT_INPUT = None
    ALLOW_NONSTANDARD_GENS = True
    STANDARDIZE_CYCLICEXPR = True

    HEATMAP_3VARS_DPI = 60
    HEATMAP_4VARS_DPI = 18

    @classmethod
    def _default_restrict_input_chars(cls, txt: str) -> bool:
        """
        Check if the input text contains only safe characters.
        Forbid certain characters to avoid potential security risks.

        Args:
            txt: Input text to check

        Returns:
            True if the text is safe, False otherwise
        """
        is_safe_char = lambda x: x < 128 or (945 <= x <= 969)  # ASCII or lower Greek
        return all(is_safe_char(ord(c)) for c in txt)

    @classmethod
    def set_poly(
        cls,
        txt: str,
        gens: Tuple[Symbol, ...] = DEFAULT_GENS,
        symmetry: PermutationGroup = DEFAULT_PERM_GROUP,
        return_type: str = "frac",
        *,
        render_triangle: bool = True,
        render_grid: bool = True,
        homogenize: bool = False,
        dehomogenize: Optional[Expr] = None,
        standardize_text: Optional[str] = None,
        omit_mul: bool = True,
        omit_pow: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Convert a string to a polynomial and perform various operations.

        TODO: converting a string to a dense polynomial might be unsafe
        in memory.

        Args:
            txt: The string representation of the polynomial
            gens: Tuple of generators (symbols)
            symmetry: Permutation group for symmetry
            return_type: Type of return value ('expr', 'frac', 'poly')
            render_triangle: Whether to render the coefficient triangle
            render_grid: Whether to render the grid heatmap
            homogenize: If True, homogenize the polynomial if possible
            dehomogenize: Value to set for dehomogenization
            standardize_text: Format to standardize the polynomial text ('factor', 'sort', 'expand')
            omit_mul: Whether to omit multiplication signs in output
            omit_pow: Whether to omit power signs in output

        Returns:
            Dictionary containing the processed polynomial and related information,
            or None if processing fails
        """
        # Input safety check
        restrict_func = cls.RESTRICT_INPUT or cls._default_restrict_input_chars
        if not restrict_func(txt):
            return None

        try:
            poly, denom = preprocess_text(txt, gens, symmetry, return_type=return_type)

            # Determine the degree
            if poly.is_zero:
                degree = degree_of_zero(txt, gens, symmetry)
            else:
                degree = poly.total_degree()
        except Exception as e:
            if cls.verbose:
                print(f"Error parsing polynomial: {e}")
            return None

        if poly is None:
            return None

        # Apply transformations to the polynomial
        if homogenize and not poly.is_homogeneous and len(poly.gens) > 0 and poly.degree(-1) == 0:
            gen = poly.gens[-1]
            poly = poly.eval(gen, 0).homogenize(gen)
            if standardize_text is None:
                standardize_text = "sort"

        if dehomogenize is not None and dehomogenize is not False:
            try:
                dehomogenize_val = sympify(dehomogenize)
                if len(dehomogenize_val.free_symbols) == 0:  # dehomogenize is a constant
                    for i in range(len(poly.gens) - 1, -1, -1):
                        if poly.degree(i) != 0:
                            poly = poly.eval(poly.gens[i], dehomogenize_val)
                            poly = poly.as_poly(*gens, domain=poly.domain)
                            degree = poly.total_degree()
                            if standardize_text is None:
                                standardize_text = "sort"
                            break
            except Exception:
                # Ignore dehomogenization errors
                pass

        try:
            if standardize_text:
                if standardize_text == "factor":
                    txt2 = poly_get_factor_form(poly, symmetry, omit_mul=omit_mul, omit_pow=omit_pow)
                elif standardize_text == "sort":
                    txt2 = poly_get_standard_form(poly, symmetry, omit_mul=omit_mul, omit_pow=omit_pow)
                elif standardize_text == "expand":
                    txt2 = poly_get_standard_form(poly, "trivial", omit_mul=omit_mul, omit_pow=omit_pow)
                else:
                    txt2 = txt
            elif denom.total_degree() != 0:
                txt2 = poly_get_standard_form(poly, symmetry)
            else:
                txt2 = txt

            if isinstance(txt2, str):
                txt = txt2
        except Exception as e:
            if cls.verbose:
                print(f"Error standardizing polynomial: {e}")

        result = {"poly": poly, "degree": degree, "txt": txt}

        if render_triangle:
            try:
                result["triangle"] = coefficient_triangle(poly, degree)
            except Exception as e:
                if cls.verbose:
                    print(f"Error rendering coefficient triangle: {e}")

        if render_grid:
            try:
                if (poly is not None and not poly.is_zero and
                        3 <= len(poly.gens) <= 4 and
                        poly.domain.is_Numerical and
                        poly.is_homogeneous):
                    size = cls.HEATMAP_3VARS_DPI \
                        if len(poly.gens) == 3 else cls.HEATMAP_4VARS_DPI
                    grid = GridRender.render(poly, size=size, with_color=True)
                    result["grid"] = grid
            except Exception as e:
                if cls.verbose:
                    print(f"Error rendering grid: {e}")

        return result

    @classmethod
    def sum_of_squares(
        cls,
        expr: Expr,
        ineq_constraints: List[Expr] = [],
        eq_constraints: List[Expr] = [],
        gens: Tuple[Symbol, ...] = DEFAULT_GENS,
        symmetry: PermutationGroup = DEFAULT_PERM_GROUP,
        time_limit: Optional[float] = time_limit,
        methods: Optional[List[str]] = None,
        configs: Dict[str, Any] = configs
    ) -> Optional[Solution]:
        """
        Perform sum of squares decomposition on an expression

        Args:
            expr: The expression to perform sum-of-squares on
            ineq_constraints: List of inequality constraints
            eq_constraints: List of equality constraints
            gens: Tuple of generators (symbols)
            symmetry: Permutation group for symmetry
            time_limit: Time limit to compute
            methods: List of methods to be used in `sum_of_squares`
            configs: Additional configs passed to `sum_of_squares`

        Returns:
            Solution object containing the decomposition, or None if decomposition fails
        """
        if expr is None:
            return None

        if time_limit > cls.time_limit:
            time_limit = cls.time_limit

        try:
            solution = sum_of_squares(
                expr,
                ineq_constraints=ineq_constraints,
                eq_constraints=eq_constraints,
                methods=methods,
                verbose=cls.verbose,
                time_limit=time_limit,
                configs=configs
            )

            # Standardize cyclic expressions if needed
            if cls.STANDARDIZE_CYCLICEXPR and solution is not None:
                solution = solution.rewrite_symmetry(gens, symmetry)

            return solution
        except Exception as e:
            if cls.verbose:
                print(f"Error in sum of squares decomposition: {e}")
            return None

    @classmethod
    def latex_coeffs(
        cls,
        txt: str,
        gens: Tuple[Symbol, ...] = DEFAULT_GENS,
        symmetry: PermutationGroup = DEFAULT_PERM_GROUP,
        *args,
        **kwargs
    ) -> str:
        """
        Generate LaTeX representation of the coefficient triangle for a polynomial.

        Args:
            txt: String representation of the polynomial
            gens: Tuple of generators (symbols)
            symmetry: Permutation group for symmetry
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            LaTeX string for the coefficient triangle, or empty string if processing fails
        """
        try:
            poly, denom = preprocess_text(txt, gens, symmetry, return_type='frac')
            return coefficient_triangle_latex(poly, *args, **kwargs)
        except Exception as e:
            if cls.verbose:
                print(f"Error generating LaTeX coefficients: {e}")
            return ''

    @classmethod
    def parse_perm_group(cls, txt: Union[str, List[List[int]]]) -> Optional[PermutationGroup]:
        """
        Parse a string or list to a permutation group.

        Args:
            txt: String or list representation of permutations

        Returns:
            PermutationGroup object, or None if parsing fails
        """
        try:
            if isinstance(txt, str):
                txt = literal_eval(txt)

            if isinstance(txt, list):
                return PermutationGroup(*(Permutation(perm) for perm in txt))

            if isinstance(txt, PermutationGroup):
                return txt
        except (ValueError, SyntaxError) as e:
            if cls.verbose:
                print(f"Error parsing permutation group: {e}")

        return None


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
