from typing import Set, Optional

import sympy as sp

from ..structsos.pivoting.bivariate import structural_sos_2vars
from ...utils import pqr_sym

class SymmetricTransform():
    """
    Class to store transformations of variables.    
    """
    @classmethod
    def transform(cls, poly, symbols, return_poly=True):
        pqr_symbols = [sp.Dummy("s") for _ in poly.gens]
        poly_pqr = pqr_sym(poly, pqr_symbols).as_poly(*pqr_symbols)
        return cls._transform_pqr(poly_pqr, poly.gens, symbols, return_poly=return_poly)

    @classmethod
    def _transform_pqr(cls, poly_pqr, original_symbols, new_symbols, return_poly=True):
        """
        Apply the transformation on a pqr-represented polynomial.
        """
        raise NotImplementedError

    @classmethod
    def inv_transform(cls, expr, original_symbols, new_symbols):
        inv_dict = cls.get_inv_dict(original_symbols, new_symbols)
        return expr.xreplace(inv_dict)

    @classmethod
    def get_default_constraints(cls, symbols):
        raise NotImplementedError

    @classmethod
    def get_dict(cls, symbols, new_symbols):
        raise NotImplementedError

    @classmethod
    def get_inv_dict(cls, symbols, new_symbols):
        raise NotImplementedError



def extract_factor(poly, factor, points=[]):
    """
    Given a polynomial and a factor, compute degree and remainder such
    that `poly = (factor ^ degree) * remainder`.

    Parameters
    ----------
    poly : sympy.Poly
        The polynomial to be factorized.
    factor : sympy.Poly
        The factor to be extracted.
    points : List[Tuple]
        A list of points such that the factor equals to zero.
        The polynomial is first checked to vanish at these points
        before factorization, which may save some time.

    Returns
    -------
    degree : int
        The degree of the factor.
    remainder : sympy.Poly
        The remainder of the polynomial after extracting the factor.    
    """
    if poly.is_zero:
        return 0, poly

    degree = 0
    while True:
        if not all(poly(*point) == 0 for point in points):
            break
        quotient, remain = divmod(poly, factor)
        if remain.is_zero:
            poly = quotient
            degree += 1
        else:
            break

    return degree, poly



def prove_by_pivoting(poly: sp.Poly, nonnegative_symbols: Set[sp.Symbol]) -> Optional[sp.Expr]:
    """
    This function is only for internal use and does not have stable API.
    It will be integrated into the StructuralSOS in the future.
    """
    nvars = len(poly.gens)
    if poly.total_degree() <= 0:
        if poly.coeff_monomial((0,)*nvars) >= 0:
            return poly.coeff_monomial((0,)*nvars)
        return None

    for gen in poly.gens:
        if (not gen in nonnegative_symbols) and poly.degree(gen) % 2 == 1:
            # do not use % 2 != 0 because degree(gen) = -oo if the degree is zero
            return None

    if nvars == 1 or (nvars == 2 and poly.is_homogeneous):
        homogenizer = None
        if nvars == 1: #and not poly.is_homogeneous:
            homogenizer = sp.Symbol('_'+poly.gen.name)
            poly = poly.homogenize(homogenizer)

        ineq_constraints = dict()
        for gen in poly.gens + (homogenizer,):
            if gen is not None and gen in nonnegative_symbols:
                ineq_constraints[gen.as_poly(*poly.gens)] = gen
                # Thought: shall we wrap the generator as a function?
        sol = structural_sos_2vars(poly, ineq_constraints=ineq_constraints, eq_constraints=dict())
        if sol is not None and homogenizer is not None:
            sol = sol.xreplace({homogenizer: sp.S(1)})
        return sol

    def get_rest_gens(gen):
        _rest_gens = tuple(g for g in poly.gens if g != gen)
        return _rest_gens

    priority = [(poly.degree(gen), gen in nonnegative_symbols, gen) for gen in poly.gens]
    priority = sorted(priority, key=lambda _:_[:2])
    for degree, is_nonnegative, gen in priority:
        rest_gens = get_rest_gens(gen)
        if degree <= 0:
            return prove_by_pivoting(poly.as_poly(*rest_gens), nonnegative_symbols)
        poly_gen = poly.as_poly(gen)

        if is_nonnegative or all(_[0] % 2 == 0 for _ in poly_gen.monoms()):
            sols = []
            for monom, coeff in poly_gen.terms():
                sol = prove_by_pivoting(coeff.as_poly(rest_gens), nonnegative_symbols)
                if sol is not None:
                    sols.append(sol * gen**monom[0])
                else:
                    break
            else:
                # not breaking from the for-loop: succeeded
                return sp.Add(*sols)

        if degree == 2:
            # try discriminant
            ...

        if degree == 3:
            ...

        if degree == 4:
            # try discriminant
            ...

    return None