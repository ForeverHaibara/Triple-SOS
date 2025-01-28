from typing import Tuple, Set, List, Optional

import sympy as sp
from sympy.core.symbol import uniquely_named_symbol

from .basic import prove_by_pivoting
from .representation import sym_transform, sym_representation_inv
from .solution import SolutionSymmetric, SolutionSymmetricSimple
from ..shared import sanitize_input, sanitize_output
from ...utils import Coeff


def _nonnegative_vars(ineq_constraints: List[sp.Poly]) -> Set[sp.Symbol]:
    """
    Infer the nonnegativity of each variable from the inequality constraints.
    """
    nonnegative = set()
    for ineq in ineq_constraints:
        if ineq.is_monomial and ineq.total_degree() == 1 and ineq.LC() >= 0:
            nonnegative.update(ineq.free_symbols)
    return nonnegative


@sanitize_output()
@sanitize_input(homogenize=True)
def SymmetricSOS(
        poly: sp.Poly,
        ineq_constraints: List[sp.Poly] = [],
        eq_constraints: List[sp.Poly] = [],
    ) -> Optional[SolutionSymmetricSimple]:
    """
    Represent a 3-var symmetric polynomial in SOS using special
    changes of variables. The algorithm is described in [1].

    Parameters
    ----------
    poly: sp.Poly
        The polynomial to perform SOS on.
    ineq_constraints: List[sp.Poly]
        Inequality constraints to the problem. This assumes g_1(x) >= 0, g_2(x) >= 0, ...
    eq_constraints: List[sp.Poly]
        Equality constraints to the problem. This assumes h_1(x) = 0, h_2(x) = 0, ...

    Returns
    -----------
    SolutionSymmetricSimple
        The solution of the problem.
        
    Reference
    -------
    [1] https://zhuanlan.zhihu.com/p/616532245
    """

    # check symmetricity here # and (1,1,1) == 0
    if (len(poly.gens) != 3 or not (poly.domain in (sp.ZZ, sp.QQ))):
        return None
    if not poly.is_homogeneous or not Coeff(poly).is_symmetric():
        return None

    methods = ['real']
    nonneg = _nonnegative_vars(ineq_constraints)
    if all(i in nonneg for i in poly.gens):
        methods.append('positive')

    dummys = [sp.Dummy("s") for _ in range(len(poly.gens))]

    for method in methods:
        numerator, ineq_constraints2, eq_constraints2, denominator = sym_transform(
            poly, ineq_constraints, eq_constraints, dummys, method=method
        )
        numerator = prove_by_pivoting(numerator, nonnegative_symbols=_nonnegative_vars(ineq_constraints2))
        if numerator is None:
            continue
        expr = numerator / denominator

        solution = sym_representation_inv(expr, poly.gens, dummys, method=method)

        ####################################################################
        # replace assumed-nonnegative symbols with inequality constraints
        ####################################################################
        func_name = uniquely_named_symbol('G', poly.gens + tuple(ineq_constraints.values()))
        func = sp.Function(func_name)
        solution = SolutionSymmetric._extract_nonnegative_exprs(solution, func_name=func_name)
        if solution is None:
            continue

        replacement = {}
        for k, v in ineq_constraints.items():
            if len(k.free_symbols) == 1 and k.is_monomial and k.LC() >= 0:
                replacement[func(k.free_symbols.pop())] = v/k.LC()
        solution = solution.xreplace(replacement)

        if solution.has(func):
            # unhandled nonnegative symbols -> not a valid solution
            continue


        solution = SolutionSymmetric(
            problem = poly,
            solution = solution,
            is_equal = True
        ).as_simple_solution()

        return solution

    return None