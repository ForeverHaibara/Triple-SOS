from typing import Union, Dict, Optional, Any

import sympy as sp
from sympy.core.symbol import uniquely_named_symbol

from .utils import Coeff
from .solution import SolutionStructural, SolutionStructuralSimple
from .nvars import sos_struct_nvars_quartic_symmetric
from .sparse import sos_struct_linear, sos_struct_quadratic
from .ternary import structural_sos_3vars, structural_sos_3vars_nonhom
from .quarternary import structural_sos_4vars
from ..symsos import prove_univariate


def StructuralSOS(
        poly: sp.Poly,
        rootsinfo: Optional[Any] = None,
        real: bool = True,
    ) -> SolutionStructuralSimple:
    """
    Main function of structural SOS. For a given polynomial, if it is in a well-known form,
    then we can solve it directly. For example, quartic 3-var cyclic polynomials have a complete
    algorithm.

    Params
    -------
    poly: sp.polys.polytools.Poly
        The target polynomial.
    real: bool
        Whether solve inequality on real numbers rather on nonnegative reals. This may
        requires lifting the degree.

    Returns
    -------
    solution: SolutionStructuralSimple

    """
    if not poly.domain.is_Numerical:
        return None
    solution = _structural_sos(poly, real=real)
    if solution is None:
        return None
    solution = SolutionStructural(problem = poly, solution = solution, is_equal = True)
    solution = solution.as_simple_solution()
    return solution


def _structural_sos(
        poly: sp.Poly,
        real: bool = True,
    ) -> sp.Expr:
    """
    Internal function of structural SOS, returns a sympy expression only.
    """
    d = poly.total_degree()
    if d == 1:
        return sos_struct_linear(poly)
    elif d == 2:
        pass

    nvars = len(poly.gens)
    if nvars == 1:
        if d == 0:
            expr = poly.as_expr()
            if expr >= 0:
                return expr
            else:
                return None
        if poly.domain in (sp.ZZ, sp.QQ):
            return prove_univariate(poly)
        return None
    
    is_hom = poly.is_homogeneous
    if is_hom:
        if nvars == 2:
            # two cases: homogeneous bivariate or univariate before homogenization
            a, b = poly.gens
            return prove_univariate(poly.subs(b, 1)).xreplace({a: a/b}).together() * b**d

    coeff = Coeff(poly)
    if is_hom:
        solution = _structural_sos_hom(poly, real=real)
    else:
        is_sym = coeff.is_symmetric()
        if is_sym:
            solution = _structural_sos_nonhom_symmetric(poly, real=real)
        else:
            solution = _structural_sos_nonhom_asymmetric(poly, real=real)

    if solution is None:
        if d == 2:        
           solution = sos_struct_quadratic(poly)
    return solution
    # original_poly = poly
    # d = poly.total_degree()
    # nvars = len(poly.gens)
    # solution = sos_struct_linear(poly)
    # is_hom = True

    # if solution is None:
    #     is_hom = poly.is_homogeneous
    #     homogenizer = None
    #     if not is_hom:
    #         # create a symbol for homogenization
    #         homogenizer = uniquely_named_symbol('t', sp.Tuple(*original_poly.free_symbols))
    #         poly = original_poly.homogenize(homogenizer)
    #         d = poly.total_degree()
    #         nvars += 1


    # ########################################
    # #              main solver
    # ########################################
    # if solution is None:
    #     if nvars == 2:
    #         # two cases: homogeneous bivariate or univariate before homogenization
    #         a, b = poly.gens
    #         solution = prove_univariate(original_poly.subs(b, 1)).xreplace({a: a/b}).together() * b**d
    #     elif nvars == 3:
    #         solution = structural_sos_3vars(poly, real = real)
    #     elif nvars == 4:
    #         solution = structural_sos_4vars(poly, real = real)

    # if solution is None and d == 2:
    #     solution = sos_struct_quadratic(poly)

    # if solution is None:
    #     return None
    # if not is_hom:
    #     # substitute the original variables
    #     solution = solution.subs(homogenizer, 1)

    # solution = SolutionStructural(problem = original_poly, solution = solution, is_equal = True)
    # solution = solution.as_simple_solution()
    # return solution


def _structural_sos_hom(poly, **kwargs):
    nvars = len(poly.gens)
    solution = None
    if nvars == 3:
        solution = structural_sos_3vars(poly, **kwargs)
    elif nvars == 4:
        solution = structural_sos_4vars(poly, **kwargs)

    if solution is None:
        solution = sos_struct_nvars_quartic_symmetric(poly)

    return solution



def _structural_sos_nonhom_symmetric(poly, **kwargs):
    """
    Solve nonhomogeneous but symmetric inequalities. It first
    exploits the symmetry of the polynomial.
    If it fails, it falls back to the asymmetric cases.
    """
    solution = None
    nvars = len(poly.gens)
    if nvars == 3:
        solution = structural_sos_3vars_nonhom(poly, **kwargs)
    if solution is not None:
        return solution

    return _structural_sos_nonhom_asymmetric(poly, **kwargs)


def _structural_sos_nonhom_asymmetric(poly, **kwargs):
    """
    Solve nonhomogeneous and asymmetric inequalities by homogenizing
    to homogeneous, acyclic inequalities.
    """    
    original_poly = poly
    # create a symbol for homogenization
    homogenizer = uniquely_named_symbol('t', sp.Tuple(*original_poly.free_symbols))
    poly = original_poly.homogenize(homogenizer)
    solution = _structural_sos_hom(poly, **kwargs)
    if solution is not None:
        solution = solution.subs(homogenizer, 1)
    return solution