from functools import wraps
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union, Any

from sympy import (
    QQ, RR, Add, Expr, Integer, MatrixBase, Mul, Poly,
    Rational, fraction, sympify
)
from sympy.combinatorics import Permutation
from sympy.core.symbol import uniquely_named_symbol

from ..problem import InequalityProblem
from ..preprocess.signs import sign_sos
from ...sdp import congruence
from ...utils.expressions import Coeff, CyclicProduct, CyclicSum
from ...utils.polytools import intervals
from ...utils.roots import nroots, rationalize_bound

if TYPE_CHECKING:
    from sympy import MutableDenseMatrix as Matrix
    from sympy import Symbol
    from sympy.polys.domains import Domain


# use imports to keep linter happy
(uniquely_named_symbol, Coeff, CyclicSum, CyclicProduct)

class StructuralSOSError(Exception): ...

class PolynomialUnsolvableError(StructuralSOSError): ...

class PolynomialNonpositiveError(PolynomialUnsolvableError): ...


class DomainExpr:
    """Mix in this class for classes that require gens."""
    def __init__(self, coeff: Coeff):
        self._coeff = coeff

    @property
    def coeff(self) -> Coeff:
        return self._coeff

    @property
    def gens(self) -> Tuple['Symbol', ...]:
        return self._coeff.gens

    def cyclic_sum(self, expr) -> Expr:
        return self._coeff.cyclic_sum(expr)

    def cyclic_product(self, expr) -> Expr:
        return self._coeff.cyclic_product(expr)

def ufsfind(ufs: dict, x):
    if ufs[x] == x:
        return x
    ufs[x] = ufsfind(ufs, ufs[x])
    return ufs[x]


def radsimp(expr: Union[Expr, List[Expr]]) -> Expr:
    """
    Rationalize the denominator by removing square roots. Wrapper of sympy.radsimp.
    Also refer to sympy.simplify.
    """
    if isinstance(expr, (list, tuple, MatrixBase)):
        return [radsimp(e) for e in expr]
    if not isinstance(expr, Expr):
        expr = sympify(expr)
    if isinstance(expr, Rational):
        return expr

    numer, denom = expr.as_numer_denom()
    from sympy import radsimp as _radsimp
    n, d = fraction(_radsimp(1/denom, symbolic=False, max_terms=1))
    # if n is not S.One:
    expr = (numer*n).expand()/d
    return expr


def sum_y_exprs(y: List[Expr], exprs: List[Expr]) -> Expr:
    """
    Return sum(y_i * expr_i).
    """
    def _mul(v, expr):
        if v == 0: return 0
        x, f = (v * expr).radsimp(symbolic=False).together().as_coeff_Mul()
        return radsimp(x) * f
    return sum(_mul(*args) for args in zip(y, exprs))


def common_region_of_curves(polys: List[Poly], domain: "Domain"):
    """
    Find a point in the domain so that poly(x, y) >= 0 holds for all
    given polynomials.
    """
    if len(polys) != 2 and any(len(p.gens) != 2 for p in polys):
        # current implementation only works for 2 polynomials
        raise ValueError("common_region_of_curves() takes 2 bivariate polynomials")
    if not domain.is_Field:
        raise ValueError("domain must be a field")

    def _convert(x):
        return domain.convert(x)

    if not domain.is_RR:
        q_polys = [f.set_domain(domain) for f in polys]
    else:
        # RR does not support resultants -> convert to QQ first
        q_polys = [f.set_domain(RR).set_domain(QQ)
                        if not f.domain.is_QQ or not f.domain.is_ZZ
                    else f.to_field() for f in polys]
    discs = [p.discriminant() for p in q_polys]

    p1, p2 = q_polys[0], q_polys[1]
    disc1, disc2 = discs
    res = p1.resultant(p2)

    def test_y(y):
        _fx = [p.rep.eval(y, 1) for p in q_polys]
        fx = [Poly.new(f, *p.gens[:-1]) for f, p in zip(_fx, q_polys)]
        for x in intervals(fx, domain):
            point = test_x(x, y)
            if point is not None:
                return point

    def test_x(x, y):
        if all(p.domain.to_sympy(p.rep.eval(x).eval(y)) >= 0 for p in q_polys):
            return tuple(_convert(i) for i in (x, y))

    for y in intervals([disc1, disc2, res], domain):
        point = test_y(y)
        if point is not None:
            return point


def rationalize_func(
    poly: Union[Poly, Rational],
    validation: Callable[[Rational], bool],
    validation_initial: Optional[Callable[[Rational], bool]] = None,
    direction: int = 0,
) -> Optional[Rational]:
    """
    Find a rational number near the roots of `poly` that satisfies certain conditions.

    Parameters
    ----------
    poly : Union[Poly, Rational]
        Initial values are near to the roots of the polynomial.
    validation : Callable
        Return True if validation(..) >= 0.
    validation_initial : Optional[Callable]
        The function first uses numerical roots of the poly, and it
        might not satisfy the validation function because of the numerical error.
        Configure this function to roughly test whether a root is proper.
        When None, it uses the validation function as default.
    direction : int
        When direction = 1, requires poly(..) >= 0. When direction = -1, requires
        poly(..) <= 0. When direction = 0 (defaulted), no addition requirement is imposed.

    Returns
    ----------
    t : Rational
        Proper rational number that satisfies the validation conditions.
        Return None if no such t is found.
    """
    from sympy import sign
    validation_initial = validation_initial or validation

    if isinstance(poly, Poly):
        candidates = nroots(poly, method = 'factor', real = True)
        poly_diff = poly.diff()
        if direction != 0:
            def direction_t(t):
                return direction if poly_diff(t) >= 0 else -direction
            def validation_t(t):
                return sign(poly(t)) * direction >= 0 and validation(t)
        else:
            direction_t = lambda t: 0
            validation_t = lambda t: validation(t)

    elif isinstance(poly, (int, float, Rational)):
        candidates = [poly]
        direction_t = lambda t: direction
        validation_t = lambda t: validation(t)


    for t in candidates:
        if isinstance(t, Rational):
            if validation(t):
                return t
        elif validation_initial(t):
            # make a perturbation
            for t_ in rationalize_bound(t, direction = direction_t(t), compulsory = True):
                if validation_t(t_):
                    return t_


def congruence_solve(M: 'Matrix', mapping = Union[List, Callable]) -> Optional[Expr]:
    cong = congruence(M)
    if cong is None:
        return None
    U, S = cong

    _mapping = mapping
    if isinstance(mapping, (list, tuple)):
        _mapping = lambda z: Add(*[z[i]*mapping[i] for i in range(len(mapping))])**2

    args = []
    for i in range(M.shape[0]):
        args.append(S[i] * _mapping(U[i,:]))
    return Add(*args)


def quadratic_weighting(coeff: Coeff, c1, c2, c3,
    mapping: Union[List[Expr], Callable] = None,
) -> Optional[Expr]:
    """
    Give solution to c1*a^2 + c2*a*b + c3*b^2 >= 0 where a,b in R.

    Parameters
    ----------
    c1, c2, c3 : Expr
        Coefficients of the quadratic form.
    """
    c1, c2, c3 = [coeff.convert(c) for c in [c1, c2, c3]]
    return congruence_solve(
        coeff.as_matrix([[c1,c2/2],[c2/2,c3]], (2,2)), mapping=mapping)


def zip_longest(*args):
    """
    Zip longest generators and pad the length with the final element.
    """
    if len(args) == 0: return
    args = [iter(arg) for arg in args]
    lasts = [None] * len(args)
    stops = [False] * len(args)
    while True:
        for i, gen in enumerate(args):
            if stops[i]:
                continue
            try:
                lasts[i] = next(gen)
            except StopIteration:
                stops[i] = True
                if all(stops):
                    return
        yield tuple(lasts)


def block_partition(blocks: List[int], groups: Tuple[int, ...]) -> List[int]:
    """
    Returns a vector `c` such that
    `blocks[k] == sum(groups[i] for i in range(m) if c[i] == k)`

    Examples
    --------
    >>> block_partition((2, 4), (1, 2, 3))
    [1, 0, 1]
    """
    if sum(blocks) != sum(groups):
        raise ValueError("No solution: sum mismatch")

    n, m = len(blocks), len(groups)
    sorted_groups = sorted(((groups[i], i) for i in range(m)), key=lambda x: (-x[0], x[1]))

    remaining = list(blocks)
    result = [0] * m

    def backtrack(index: int) -> bool:
        if index == m:
            return True
        val, original_idx = sorted_groups[index]
        for k in range(n):
            if remaining[k] >= val:
                if k > 0 and remaining[k] == remaining[k-1]:
                    continue
                remaining[k] -= val
                result[original_idx] = k
                if backtrack(index + 1):
                    return True
                remaining[k] += val
        return False
    if not backtrack(0):
        raise ValueError("No valid partition found")
    return result


def structsos_extract_factors(solver: Callable) -> Callable:
    """
    Decorate a solver to extract monomial factors or common powers first.

    For a monomial times f, solve f and multiply the result by the monomial.
    For f(a^k, b^k, ...), solve f and substitute the powers back into the result.
    Preserve Poly inputs; convert other inputs to Coeff. The wrapped solver
    receives real=0 for an odd monomial factor, and at most real=1 for an
    even power substitution. Only one reduction is applied per call.
    """
    @wraps(solver)
    def _wrapped_solver(coeff, real: int = 1, **kwargs):
        poly = coeff
        coeff = poly if isinstance(poly, Coeff) else Coeff(poly)
        symbols = coeff.gens
        multiplier = Integer(1)
        replacements = {}

        monom, new_coeff = coeff.cancel_abc()
        if any(i > 0 for i in monom):
            if not all(i % 2 == 0 for i in monom):
                real = min(int(real), 1)
            if len(symbols) > 2 and all(i == monom[0] for i in monom):
                multiplier = CyclicProduct(symbols[0]**monom[0], symbols)
            else:
                multiplier = Mul(*[s**i for s, i in zip(symbols, monom)])

        i, new_coeff = new_coeff.cancel_k()
        if i > 1:
            real = min(int(real), 1) if i % 2 == 0 else real
            replacements = {s: s**i for s in symbols}

        if isinstance(poly, Poly):
            new_coeff = poly if new_coeff is coeff else new_coeff.as_poly(*symbols)
        solution = solver(new_coeff, real=real, **kwargs)
        if solution is None:
            return None
        if not isinstance(solution, Expr):
            solution = solution + Integer(0)
        return multiplier * solution.xreplace(replacements)

    return _wrapped_solver


def _reorder_helper(
    obj: Union[Poly, Coeff, "InequalityProblem[Poly]"]
) -> Tuple[Callable[[Permutation], bool], Callable[[List[int]], Any]]:
    """
    Given an object, returns two functions that:
    1. checks whether it is symmetric with respect to a permutation;
    2. reorders the generators with respect to a permutation.
    """
    if isinstance(obj, Poly):
        cf = Coeff(obj)
        return cf.is_symmetric, cf.reorder
    elif isinstance(obj, Coeff):
        return obj.is_symmetric, obj.reorder
    elif isinstance(obj, InequalityProblem):
        G = obj.identify_symmetry()
        is_sym = lambda perm: perm in G
        gens = obj.gens
        def reorder(perm: List[int]):
            new_gens = [gens[i] for i in perm]
            expr = obj.expr.reorder(*new_gens)
            ineqs, eqs = {}, {}
            for k, v in obj.ineq_constraints.items():
                ineqs[k.reorder(*new_gens)] = v
            for k, v in obj.eq_constraints.items():
                eqs[k.reorder(*new_gens)] = v
            pro = obj.copy_new(expr, ineqs, eqs)
            if pro.roots is not None:
                pro.roots = pro.roots.reorder(tuple(perm))
            return pro
        return is_sym, reorder

    raise TypeError("Unsupported object type. Expected"
        "Poly, Coeff, or InequalityProblem[Poly], but received %s." % type(obj))


def structsos_reorder_symmetry(groups: Tuple[int, ...]) -> Callable:
    """
    Decorator for the solver function to reorder the generators
    so that they are in the given symmetry.

    Parameters
    ----------
    groups : Tuple[int, ...]
        The degree of each symmetric group. E.g., when there are four variables
        and `groups = (3, 1)`, it makes the resulting polynomial symmetric with
        respect to the first three variables and then calls the solver.
    """
    def wrapper(solver: Callable) -> Callable:
        @wraps(solver)
        def _wrapped_solver(poly: Union[Poly, Coeff], *args, need_reorder=True, **kwargs):
            if not need_reorder:
                return solver(poly, *args, **kwargs)

            is_sym, reorder = _reorder_helper(poly)

            n = len(poly.gens)
            ufs = {i: i for i in range(n)}
            for i in range(n):
                for j in range(i+1, n):
                    if ufsfind(ufs, i) == ufsfind(ufs, j):
                        continue
                    if is_sym(Permutation(size=n)(i, j)):
                        ufs[ufsfind(ufs, j)] = ufsfind(ufs, i)

            blocks = {i: [] for i in range(n) if ufsfind(ufs, i) == i}
            for i in range(n):
                blocks[ufsfind(ufs, i)].append(i)
            blocks = list(blocks.values())
            ufs_size = [len(b) for b in blocks]

            partition = []
            try:
                partition = block_partition(ufs_size, groups)
            except ValueError:
                return None

            inds = []
            for g, p in zip(groups, partition):
                inds.extend(blocks[p][:g])
                blocks[p] = blocks[p][g:]
            new_coeff = reorder(inds)
            return solver(new_coeff, *args, **kwargs)
        return _wrapped_solver
    return wrapper


def structsos_constrained(
    checker: Callable[[Poly], bool],
    positive=True
) -> Callable:
    def wrapper(solver: Callable) -> Callable:
        @wraps(solver)
        def _wrapped_solver(problem: InequalityProblem[Poly], **kwargs):
            sign = -1
            for k, v in problem.eq_constraints.items():
                if checker(k):
                    con, con_v, sign = k, v, 0
                    break
            for k, v in problem.ineq_constraints.items():
                if checker(k):
                    con, con_v, sign = k, v, 1
                    break
            if sign == -1:
                return None

            dom = problem.expr.domain.unify(con.domain)
            poly = problem.expr.set_domain(dom)
            con = con.set_domain(dom)

            F = problem.uniquely_named_symbol('_F')
            sol = solver(Coeff(poly), Coeff(con), F, sign=sign, **kwargs)
            if sol is None:
                return None

            signs = {F: (sign, con_v)}
            if positive:
                signs.update(problem.get_symbol_signs())

            return sign_sos(sol, signs)

        return _wrapped_solver
    return wrapper
