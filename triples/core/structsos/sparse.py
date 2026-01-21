from typing import Callable, Union, Dict

from sympy import Poly, Expr, Integer, Mul

from .utils import Coeff, PolynomialUnsolvableError, PolynomialNonpositiveError
from ...utils import CyclicSum, CyclicProduct

def _null_solver(*args, **kwargs):
    return None

def sos_struct_extract_factors(poly: Union[Poly, Coeff], solver: Callable, real: bool = True, **kwargs):
    """
    Wrap a solver to handle factorizable polynomials in advance.

    Cases.
    + If the polynomial is a*b*c * f(a,b,c), it first solves f(a,b,c) and then multiplies the result by a*b*c.
    + If the polynomial is f(a^k, b^k, c^k), it first solves f(a,b,c) and then replaces a,b,c with a^k,b^k,c^k.
    """
    coeff = poly if isinstance(poly, Coeff) else Coeff(poly)
    symbols = coeff.gens

    def coeff_to_poly(new_coeff):
        if isinstance(poly, Poly):
            return new_coeff.as_poly(*symbols)
        return new_coeff

    monoms, new_coeff = coeff.cancel_abc()
    if any(i > 0 for i in monoms):
        new_coeff = coeff_to_poly(new_coeff)
        solution = solver(new_coeff, real = real and all(i % 2 == 0 for i in monoms), **kwargs)
        if solution is not None:
            if coeff.nvars == 3 and all(i == monoms[0] for i in monoms):
                multiplier = CyclicProduct(symbols[0]**monoms[0], symbols)
            else:
                multiplier = Mul(*[s**i for s, i in zip(symbols, monoms)])
            return multiplier * solution
        return None

    i, new_coeff = coeff.cancel_k()
    if i > 1:
        new_coeff = coeff_to_poly(new_coeff)
        solution = solver(new_coeff, real = False if (i % 2 == 0) else real, **kwargs)
        if solution is not None:
            solution = solution.xreplace(dict((s, s**i) for s in symbols))
            return solution
        return None

    return solver(poly, **kwargs)


def sos_struct_common(poly: Union[Poly, Coeff], *solvers, **kwargs):
    """
    A method wrapper for multiple solvers.
    """
    def _wrapped_solver(poly, **kwargs):
        solution = None
        for solver in solvers:
            if solver is None:
                continue
            try:
                solution = solver(poly, **kwargs)
                if solution is not None:
                    break

            except PolynomialUnsolvableError as e:
                # When we are sure that the polynomial is nonpositive,
                # we can return None directly.
                if isinstance(e, PolynomialNonpositiveError):
                    return None
        if solution is not None:
            if not isinstance(solution, Expr):
                # this should automatically cast the solution to Expr
                solution = solution + Integer(0)
        return solution

    # cancel abc

    return sos_struct_extract_factors(poly, _wrapped_solver, **kwargs)



def sos_struct_degree_specified_solver(solvers: Dict[int, Callable], homogeneous: bool = False) -> Callable:
    """
    A method wrapper for structural SOS with degree specified solvers.

    When `degree <= 2` and the solver is not provided, it uses the default solvers
    from `triples.core.structsos.nvars` for general linear and quadratic solvers.
    """
    def _sos_struct_degree_specified_solver(poly: Union[Poly, Coeff], *args, **kwargs):
        if homogeneous and isinstance(poly, Coeff):
            degree = poly.total_degree()
        else:
            degree = poly.total_degree()
        solver = solvers.get(degree, None)
        if solver is None:
            from .nvars import sos_struct_nvars_linear, sos_struct_nvars_quadratic
            if degree == 1:
                solver = sos_struct_nvars_linear
            elif degree == 2:
                solver = sos_struct_nvars_quadratic
            else:
                solver = _null_solver
        return solver(poly, *args, **kwargs)
    return _sos_struct_degree_specified_solver
