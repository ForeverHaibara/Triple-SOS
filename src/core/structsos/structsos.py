from typing import Union, List, Dict, Optional, Any

import sympy as sp
from sympy.core.symbol import uniquely_named_symbol

from .utils import Coeff
from .solution import SolutionStructural, SolutionStructuralSimple
from .nvars import sos_struct_nvars_quartic_symmetric
from .sparse import sos_struct_linear, sos_struct_quadratic
from .ternary import structural_sos_3vars
from .quarternary import structural_sos_4vars
from .univariate import structural_sos_2vars
from ..shared import sanitize_input

@sanitize_input(homogenize=True, infer_symmetry=True, wrap_constraints=False)
def StructuralSOS(
        poly: sp.Poly,
        ineq_constraints: Union[List[sp.Poly], Dict[sp.Poly, sp.Expr]] = {},
        eq_constraints: Union[List[sp.Poly], Dict[sp.Poly, sp.Expr]] = {},
    ) -> SolutionStructuralSimple:
    """
    Main function of structural SOS. For a given polynomial, if it is in a well-known form,
    then we can solve it directly. For example, quartic 3-var cyclic polynomials have a complete
    algorithm.

    Params
    -------
    poly: sp.Poly
        The polynomial to perform SOS on.
    ineq_constraints: List[sp.Poly]
        Inequality constraints to the problem. This assume g_1(x) >= 0, g_2(x) >= 0, ...
    eq_constraints: List[sp.Poly]
        Equality constraints to the problem. This assume h_1(x) = 0, h_2(x) = 0, ...
    real: bool
        Whether solve inequality on real numbers rather on nonnegative reals. This may
        requires lifting the degree.
        **TO BE DEPRECATED**

    Returns
    -------
    solution: SolutionStructuralSimple

    """
    solution = _structural_sos(poly, ineq_constraints, eq_constraints)
    if solution is None:
        return None
    solution = SolutionStructural(problem = poly, solution = solution, is_equal = True)
    solution = solution.as_simple_solution()
    if solution.is_ill:
        return None
    return solution


def _structural_sos(poly: sp.Poly, ineq_constraints: Dict[sp.Poly, sp.Expr] = {}, eq_constraints: Dict[sp.Poly, sp.Expr] = {}) -> sp.Expr:
    """
    Internal function of structural SOS, returns a sympy expression only.
    The polynomial must be homogeneous.
    """
    d = poly.total_degree()
    nvars = len(poly.gens)
    if nvars == 1 or d == 0:
        # since the poly is homogeneous, it must be a monomial
        if poly.is_monomial and poly.LC() >= 0:
            return poly.as_expr()
        return None

    if d == 1:
        # return sos_struct_linear(poly)
        ...
    elif d == 2:
        pass

    solution = None
    if nvars == 2:
        # homogeneous bivariate
        solution = structural_sos_2vars(poly, ineq_constraints, eq_constraints)
    elif nvars == 3:
        solution = structural_sos_3vars(poly, ineq_constraints, eq_constraints)
    elif nvars == 4:
        solution = structural_sos_4vars(poly, ineq_constraints, eq_constraints)

    if solution is None and nvars > 3:
        solution = sos_struct_nvars_quartic_symmetric(poly)
    if solution is None and nvars > 3 and d == 2:
        solution = sos_struct_quadratic(poly)
    return solution


# def _structural_sos_hom(poly, **kwargs):
#     nvars = len(poly.gens)
#     solution = None
#     if nvars == 3:
#         solution = structural_sos_3vars(poly, **kwargs)
#     elif nvars == 4:
#         solution = structural_sos_4vars(poly, **kwargs)

#     if solution is None:
#         solution = sos_struct_nvars_quartic_symmetric(poly)

#     return solution

# def _structural_sos_nonhom_symmetric(poly, **kwargs):
#     """
#     Solve nonhomogeneous but symmetric inequalities. It first
#     exploits the symmetry of the polynomial.
#     If it fails, it falls back to the asymmetric cases.
#     """
#     solution = None
#     nvars = len(poly.gens)
#     if nvars == 3:
#         solution = structural_sos_3vars_nonhom(poly, **kwargs)
#     if solution is not None:
#         return solution

#     return _structural_sos_nonhom_asymmetric(poly, **kwargs)


# def _structural_sos_nonhom_asymmetric(poly, **kwargs):
#     """
#     Solve nonhomogeneous and asymmetric inequalities by homogenizing
#     to homogeneous, acyclic inequalities.
#     """    
#     original_poly = poly
#     # create a symbol for homogenization
#     homogenizer = uniquely_named_symbol('t', sp.Tuple(*original_poly.free_symbols))
#     poly = original_poly.homogenize(homogenizer)
#     solution = _structural_sos_hom(poly, **kwargs)
#     if solution is not None:
#         solution = solution.subs(homogenizer, 1)
#     return solution