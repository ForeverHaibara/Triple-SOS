# author: https://github.com/ForeverHaibara
from ast import literal_eval
from typing import Tuple, List, Dict, Optional, Any, Union, Callable

import sympy as sp
from sympy import Expr, Poly, Symbol
from sympy.combinatorics import Permutation, PermutationGroup, CyclicGroup

from .grid import GridRender
from ..utils import Solution, SolutionSimple, CyclicExpr, CyclicSum, CyclicProduct
from ..utils.text_process import (
    preprocess_text, poly_get_factor_form, poly_get_standard_form,
    degree_of_zero, coefficient_triangle, coefficient_triangle_latex
)
from ..core.sum_of_squares import sum_of_squares, DEFAULT_CONFIGS
# from ..core.linsos import root_tangents


def _default_polynomial_check(poly: Poly, method_order: List[str]) -> List[str]:
    """
    Check the degree and nvars of a polynomial to decide
    whether a method is applicable. For too high degree polynomials,
    methods like SDPSOS are removed to avoid long computation time.
    """
    is_hom = int(poly.is_homogeneous)
    nvars = len(poly.gens) + (1 - is_hom)
    degree = poly.total_degree()
    upper_bounds = [30, 30, 30, 14, 12, 8, 6, 4, 4, 4, 4]
    if degree > upper_bounds[nvars]:
        # remove LinearSOS and SDPSOS
        method_order = [method for method in method_order if method not in ('LinearSOS', 'SDPSOS')]
    return method_order


class SOS_Manager():
    """
    A convenient class to manage the sum of square decomposition of a polynomial,
    providing commonly used functions and properties.

    It adds more sanity checks and error handling to the core functions.
    """
    verbose = True

    CONFIG_DEFAULT_GENS = sp.symbols("a b c")
    CONFIG_DEFAULT_PERM = CyclicGroup(3)
    CONFIG_RESTRICT_INPUT_CHARS = True
    CONFIG_METHOD_CHECK = _default_polynomial_check
    CONFIG_ALLOW_NONSTANDARD_GENS = True
    CONFIG_STANDARDIZE_CYCLICEXPR = True

    @classmethod
    def set_poly(cls, 
            txt: str,
            gens: Tuple[Symbol] = CONFIG_DEFAULT_GENS,
            perm: PermutationGroup = CONFIG_DEFAULT_PERM,
            render_triangle: bool = True,
            render_grid: bool = True,
            homogenize: bool = False,
            dehomogenize: Optional[Expr] = None,
            standardize_text: Optional[str] = None,
            omit_mul: bool = True,
            omit_pow: bool = True,
        ) -> Dict[str, Any]:
        """
        Convert a text to a polynomial, and render the coefficient triangle and grid heatmap.

        Parameters
        ----------
        txt : str
            The text of the polynomial.
        render_triangle : bool
            Whether to render the coefficient triangle, by default True.
        render_grid : bool
            Whether to render the grid heatmap, by default True.
        homogenize : bool
            If True, homogenize the polynomial if it any variable is missing, by default False.
        dehomogenize : bool
            If True, set the last variable to the given value, by default None.
        standardize_text : str, optional
            If not None, it should be one of ['factor', 'sort', 'expand']. This will rewrite the polynomial
            in the corresponding form. By default None.
        omit_mul : bool
            Whether to omit the multiplication sign when rewriting the polynomial, by default True.
        omit_pow : bool
            Whether to omit the power sign when rewriting the polynomial, by default True.

        Returns
        -------
        dict
            A dictionary containing the polynomial, degree, text, coefficient triangle, and grid heatmap.
        """
        if cls.CONFIG_RESTRICT_INPUT_CHARS:
            # forbid non-ascii characters to avoid potential security risks
            if not all(ord(_) < 128 for _ in txt):
                return None

        try:
            poly, denom = preprocess_text(txt, gens=gens, perm=perm, return_type='frac')

            if poly.is_zero:
                n = degree_of_zero(txt, gens=gens, perm=perm)
            else:
                n = poly.total_degree()
        except Exception as e:
            # raise e
            return None

        if poly is None:
            return None

        if homogenize and not poly.is_homogeneous and len(poly.gens) and poly.degree(-1) == 0:
            gen = poly.gens[-1]
            poly = poly.eval(gen, 0).homogenize(gen)
            if not standardize_text:
                standardize_text = 'sort'

        if dehomogenize is not None and dehomogenize is not False:
            try:
                dehomogenize = sp.S(dehomogenize)
                if len(dehomogenize.free_symbols) == 0: # dehomogenize is a number
                    for i in range(len(poly.gens)-1, -1, -1):
                        if poly.degree(i) != 0:
                            poly = poly.eval(poly.gens[i], dehomogenize)
                            poly = poly.as_poly(*gens, domain=poly.domain)
                            n = poly.total_degree()
                            if not standardize_text:
                                standardize_text = 'sort'
                            break
            except:
                pass

        try:
            if standardize_text:
                if standardize_text == 'factor':
                    txt2 = poly_get_factor_form(poly, perm, omit_mul=omit_mul, omit_pow=omit_pow)
                elif standardize_text == 'sort':
                    txt2 = poly_get_standard_form(poly, perm, omit_mul=omit_mul, omit_pow=omit_pow)
                elif standardize_text == 'expand':
                    triv_group = PermutationGroup([Permutation(list(range(len(poly.gens))))])
                    txt2 = poly_get_standard_form(poly, triv_group, omit_mul=omit_mul, omit_pow=omit_pow)
            elif not denom.total_degree() == 0:
                txt2 = poly_get_standard_form(poly, perm)
            if isinstance(txt2, str):
                txt = txt2
        except:
            pass


        return_dict = {'poly': poly, 'degree': n, 'txt': txt}
        if render_triangle:
            return_dict['triangle'] = coefficient_triangle(poly, n)


        if render_grid:
            if poly is not None and (not poly.is_zero) and 3 <= len(poly.gens) <= 4\
                    and poly.domain.is_Numerical and poly.is_homogeneous:
                size = 60 if len(poly.gens) == 3 else 18
                grid = GridRender.render(poly, size=size, with_color=True)
                return_dict['grid'] = grid
        return return_dict

    @classmethod
    def check_poly(cls, poly: Poly) -> bool:
        """
        Check whether a polynomial is a valid polynomial:
        3-var, non-zero, homogeneous, and numerical domain.
        """
        if poly is None or (not isinstance(poly, Poly)):
            return False
        if len(poly.gens) != 3 or (poly.is_zero) or (not poly.is_homogeneous) or poly.total_degree() < 1:
            return False
        if not poly.domain.is_Numerical:
            return False
        return True

    @classmethod
    def get_standard_form(cls, poly: Poly, perm: PermutationGroup = CONFIG_DEFAULT_PERM, **kwargs) -> str:
        """
        Rewrite a polynomial in the standard form.
        """
        return poly_get_standard_form(poly, perm, **kwargs)

    @classmethod
    def get_factor_form(cls, poly: Poly, perm: PermutationGroup = CONFIG_DEFAULT_PERM, **kwargs) -> str:
        """
        Rewrite a polynomial in the factor form.
        """
        return poly_get_factor_form(poly, perm, **kwargs)

    # @classmethod
    # def findroot(cls, poly, grid = None, verbose = True):
    #     """
    #     Find the roots / local minima of a polynomial.
    #     """
    #     if not cls.check_poly(poly):
    #         return []

    #     roots = findroot(
    #         poly, 
    #         most = 5, 
    #         grid = grid, 
    #         with_tangents = root_tangents
    #     )
    #     if verbose:
    #         print(roots)
    #     return roots

    @classmethod
    def sum_of_squares(cls,
            poly,
            ineq_constraints: Dict[Poly, Expr] = [],
            eq_constraints: Dict[Poly, Expr] = [],
            gens = CONFIG_DEFAULT_GENS,
            perm = CONFIG_DEFAULT_PERM,
            method_order = None,
            configs = DEFAULT_CONFIGS
        ):
        """
        Perform the sum of square decomposition of a polynomial.
        The keyword arguments are passed to the function sum_of_squares.
        """
        if poly is None or (not isinstance(poly, Poly)):
            return None

        if cls.CONFIG_ALLOW_NONSTANDARD_GENS:
            if len(poly.free_symbols_in_domain) > 0:
                poly = poly.as_poly(*sorted(list(poly.gens) + list(poly.free_symbols_in_domain), key=lambda x:x.name))
            degree_of_each_gen = [poly.degree(_) for _ in poly.gens]
            if any(_ == 0 for _ in degree_of_each_gen):
                # remove the gen
                nonzero_gens = [gen for gen, d in zip(poly.gens, degree_of_each_gen) if d > 0]
                poly = poly.as_poly(*nonzero_gens)

        if cls.verbose is False:
            for method in ('LinearSOS', 'SDPSOS'):
                if configs.get(method) is None:
                    configs[method] = {}
                configs[method]['verbose'] = False
        method_order = cls.CONFIG_METHOD_CHECK(poly, method_order)

        solution = sum_of_squares(
            poly,
            ineq_constraints = ineq_constraints,
            eq_constraints = eq_constraints,
            method_order = method_order,
            configs = configs
        )
        if cls.CONFIG_STANDARDIZE_CYCLICEXPR:
            solution = solution.rewrite_symmetry(gens, perm)
        return solution

    # def save_heatmap(self, poly, *args, **kwargs):
    #     return self._poly_info['grid'].save_heatmap(*args, **kwargs)

    # def save_coeffs(self, poly, *args, **kwargs):
    #     return self._poly_info['grid'].save_coeffs(*args, **kwargs)

    @classmethod
    def latex_coeffs(cls, txt, gens=CONFIG_DEFAULT_GENS, perm=CONFIG_DEFAULT_PERM, *args, **kwargs):
        try:
            poly, denom = preprocess_text(txt, gens=gens, perm=perm, return_type='frac')
        except:
            return ''
        return coefficient_triangle_latex(poly, *args, **kwargs)

    @classmethod
    def _parse_perm_group(cls, txt: Union[str, List[List[int]]]) -> PermutationGroup:
        """
        Parse a text or a list to a permutation group.
        """
        if isinstance(txt, str):
            txt = literal_eval(txt)
        if isinstance(txt, list):
            txt = PermutationGroup(*(Permutation(_) for _ in txt))
        if isinstance(txt, PermutationGroup):
            return txt
        return


def _render_LaTeX(a, path, usetex=True, show=False, dpi=500, fontsize=20):
    '''render a text in LaTeX and save it to path'''
    
    import matplotlib.pyplot as plt

    acopy = a
    # linenumber = a.count('\\\\') + 1
    # plt.figure(figsize=(12,10 ))
    
    # set the figure small enough
    # even though the text cannot be display as a whole in the window
    # it will be saved correctly by setting bbox_inches = 'tight'
    plt.figure(figsize=(0.3,0.3))
    if usetex:
        try:
            a = '$\\displaystyle ' + a.strip('$') + ' $'
            #plt.figure(figsize=(12, linenumber*0.5 + linenumber**0.5 * 0.3 ))
            #plt.text(-0.3,0.75+min(0.35,linenumber/25), a, fontsize=15, usetex=usetex)
            #fontfamily='Times New Roman')
            plt.text(-0.3,0.9, a, fontsize=fontsize, usetex=usetex)#
        except:
            usetex = False
    
    if not usetex:
        a = acopy
        a = a.strip('$')
        a = '\n'.join([' $ '+_+' $ ' for _ in a.split('\\\\')])
        plt.text(-0.3,0.9, a, fontsize=fontsize, usetex=usetex)#, fontfamily='Times New Roman')
        
    plt.ylim(0,1)
    plt.xlim(0,6)
    plt.axis('off')
    plt.savefig(path, dpi=dpi, bbox_inches ='tight')
    if show:
        plt.show()
    else:
        plt.close()