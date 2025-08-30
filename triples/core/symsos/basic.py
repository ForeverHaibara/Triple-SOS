from typing import Tuple, Dict, Set, Optional, Callable

from sympy import Poly, Expr, Dummy, Symbol, Integer, Add

from ..structsos.pivoting.bivariate import structural_sos_2vars
from ...utils import pqr_sym, verify_symmetry
from ..problem import InequalityProblem

class SymmetricTransform():
    """
    Class to store transformations of variables.    
    """

    nvars = 0
    symmetry = None

    @classmethod
    def transform(cls, poly, symbols, return_poly=True):
        pqr_symbols = [Dummy(f"s{i}") for i in range(cls.nvars)]
        poly_pqr = pqr_sym(poly, pqr_symbols).as_poly(*pqr_symbols)
        return cls._transform_pqr(poly_pqr, poly.gens, symbols, return_poly=return_poly)

    @classmethod
    def _transform_pqr(cls, poly_pqr, original_symbols, new_symbols, return_poly=True):
        """
        Apply the transformation on a polynomial of elementary symmetric functions.
        For example, when nvars == 3, the polynomial is represented by pqr.
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
    def get_natural_constraints(cls, symbols: Tuple[Symbol], new_symbols: Tuple[Symbol], problem: InequalityProblem) \
            -> Optional[Tuple[Dict[Poly, Expr], Dict[Poly, Expr]]]:
        """
        Consider a transform of variables from `symbols` to `new_symbols`,
        new natural constraints over `new_symbols` are introduced
        """
        raise NotImplementedError

    @classmethod
    def get_constraints(cls, symbols: Tuple[Symbol], new_symbols: Tuple[Symbol], problem: InequalityProblem) \
            -> Optional[Tuple[Dict[Poly, Expr], Dict[Poly, Expr]]]:
        """
        Apply the transform on the constraints of the problem.
        """
        ineqs = problem.ineq_constraints
        eqs = problem.eq_constraints
        new_cons = cls.get_natural_constraints(symbols, new_symbols, problem)
        if new_cons is None:
            return None
        new_ineqs, new_eqs = new_cons

        trans = lambda x: cls.transform(x.as_poly(symbols), new_symbols)
        for old, new in ((ineqs, new_ineqs), (eqs, new_eqs)):
            for p, e in old.items():
                p = p.as_poly(symbols)
                if not verify_symmetry(p, cls.symmetry):
                    continue # TODO
                new_p, mul = trans(p, symbols)
                new[new_p] = e * mul
        return new_ineqs, new_eqs

    @classmethod
    def get_dict(cls, symbols: Tuple[Symbol], new_symbols: Tuple[Symbol]) -> Dict[Symbol, Expr]:
        """
        Get the dictionary {new_symbol: expr(old_symbols)}
        that maps old symbols to new symbols.
        """
        raise NotImplementedError

    @classmethod
    def get_inv_dict(cls, symbols: Tuple[Symbol], new_symbols: Tuple[Symbol]) -> Dict[Symbol, Expr]:
        """
        Get the dictionary {old_symbol: expr(new_symbols)}
        that maps new symbols back to old symbols.
        """
        raise NotImplementedError

    @classmethod
    def apply(cls, problem: InequalityProblem, symbols: Tuple[Symbol], new_symbols: Tuple[Symbol]=None) \
            -> Tuple[InequalityProblem, Callable]:

        expr = problem.expr

        trans = lambda x: cls.transform(x.as_poly(symbols), new_symbols)
        new_expr = trans(expr, symbols)

        new_cons = cls.get_constraints(symbols, new_symbols, problem)
        if new_cons is None:
            return None
        new_ineqs, new_eqs = new_cons

        pro = InequalityProblem(new_expr, new_ineqs, new_eqs)
        restoration = lambda x: cls.inv_transform(x, symbols, new_symbols)
        return pro, restoration

        

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



def prove_by_pivoting(poly: Poly, nonnegative_symbols: Set[Symbol]) -> Optional[Expr]:
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
            homogenizer = Symbol('_'+poly.gen.name)
            poly = poly.homogenize(homogenizer)

        ineq_constraints = dict()
        for gen in poly.gens + (homogenizer,):
            if gen is not None and gen in nonnegative_symbols:
                ineq_constraints[gen.as_poly(*poly.gens)] = gen
                # Thought: shall we wrap the generator as a function?
        sol = structural_sos_2vars(poly, ineq_constraints=ineq_constraints, eq_constraints=dict())
        if sol is not None and homogenizer is not None:
            sol = sol.xreplace({homogenizer: Integer(1)})
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
                return Add(*sols)

        if degree == 2:
            # try discriminant
            ...

        if degree == 3:
            ...

        if degree == 4:
            # try discriminant
            ...

    return None