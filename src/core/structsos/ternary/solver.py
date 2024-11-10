from typing import Union, Dict, Optional, Any

import sympy as sp
from sympy.core.symbol import uniquely_named_symbol

from .sparse  import sos_struct_sparse, sos_struct_heuristic
from .quadratic import sos_struct_quadratic, sos_struct_acyclic_quadratic
from .cubic   import sos_struct_cubic, sos_struct_acyclic_cubic
from .quartic import sos_struct_quartic, sos_struct_acyclic_quartic
from .quintic import sos_struct_quintic
from .sextic  import sos_struct_sextic
from .septic  import sos_struct_septic
from .octic   import sos_struct_octic
from .nonic   import sos_struct_nonic
from .acyclic import sos_struct_acyclic_sparse

from ..utils import Coeff, PolynomialNonpositiveError, PolynomialUnsolvableError
from ..sparse import sos_struct_common, sos_struct_degree_specified_solver
from ..solution import SolutionStructural
from ....utils import verify_hom_cyclic

SOLVERS = {
    2: sos_struct_quadratic,
    3: sos_struct_cubic,
    4: sos_struct_quartic,
    5: sos_struct_quintic,
    6: sos_struct_sextic,
    7: sos_struct_septic,
    8: sos_struct_octic,
    9: sos_struct_nonic,
}

SOLVERS_ACYCLIC = {
    2: sos_struct_acyclic_quadratic,
    3: sos_struct_acyclic_cubic,
    4: sos_struct_acyclic_quartic
}


def _structural_sos_3vars_cyclic(
        coeff: Union[sp.Poly, Coeff, Dict],
        real: bool = True
    ):
    """
    Internal function to solve a 3-var homogeneous cyclic polynomial using structural SOS.
    The function assumes the polynomial is wrt. (a, b, c).
    It does not check the homogeneous / cyclic property of the polynomial to save time.
    """
    if not isinstance(coeff, Coeff):
        coeff = Coeff(coeff)
    
    return sos_struct_common(coeff,
        sos_struct_sparse,
        sos_struct_degree_specified_solver(SOLVERS, homogeneous=True),
        sos_struct_heuristic,
        real=real
    )

def _structural_sos_3vars_acyclic(
        coeff: Union[sp.Poly, Coeff, Dict],
        real: bool = True
    ):
    """
    Internal function to solve a 3-var homogeneous acyclic polynomial using structural SOS.
    The function assumes the polynomial is wrt. (a, b, c).
    It does not check the homogeneous / cyclic property of the polynomial to save time.
    """
    if not isinstance(coeff, Coeff):
        coeff = Coeff(coeff)

    return sos_struct_common(coeff,
        sos_struct_acyclic_sparse,
        sos_struct_degree_specified_solver(SOLVERS_ACYCLIC, homogeneous=True),
        real=real
    )

def structural_sos_3vars(poly, ineq_constraints: Dict[sp.Poly, sp.Expr] = {}, eq_constraints: Dict[sp.Poly, sp.Expr] = {}) -> Optional[sp.Expr]:
    """
    Main function of structural SOS for 3-var homogeneous polynomials. 
    It first assumes the polynomial has variables (a,b,c) and
    latter substitutes the variables with the original ones.
    """
    if len(poly.gens) != 3: # should not happen
        raise ValueError("structural_sos_3vars only supports 3-var polynomials.")

    is_hom, is_cyc = verify_hom_cyclic(poly)
    if not is_hom: # should not happen
        raise ValueError("structural_sos_3vars only supports homogeneous polynomials.")

    if is_cyc:
        func = _structural_sos_3vars_cyclic
    else:
        func = _structural_sos_3vars_acyclic

    try:
        solution = func(Coeff(poly), real = 1)
    except (PolynomialNonpositiveError, PolynomialUnsolvableError):
        return None

    if solution is None:
        return None

    if poly.gens != (sp.symbols("a b c")):
        solution = solution.xreplace(dict(zip(sp.symbols("a b c"), poly.gens)))


    # replace assumed-nonnegative symbols with inequality constraints
    func_name = uniquely_named_symbol('G', poly.gens + tuple(ineq_constraints.values()))
    func = sp.Function(func_name)
    solution = SolutionStructural._extract_nonnegative_symbols(solution, func_name=func_name)
    if solution is None:
        return None

    replacement = {}
    for k, v in ineq_constraints.items():
        if len(k.free_symbols) == 1 and k.is_monomial and k.LC() >= 0:
            replacement[func(k.free_symbols.pop())] = v/k.LC()
    solution = solution.xreplace(replacement)

    if solution.has(func):
        # unhandled nonnegative symbols -> not a valid solution
        return None

    return solution