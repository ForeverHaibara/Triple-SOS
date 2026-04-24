from typing import Union, Tuple, List, Dict, Callable, Optional
from functools import wraps

from sympy import (
    Poly, Expr, Symbol, Integer, Rational, MatrixBase, Add,
    QQ, ZZ, sympify, fraction
)
from sympy import MutableDenseMatrix as Matrix
from sympy.combinatorics import Permutation
from sympy.core.symbol import uniquely_named_symbol

from ...sdp import congruence
from ...utils.expressions import Coeff, CyclicSum, CyclicProduct
from ...utils.roots import nroots, rationalize_bound

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
    def gens(self) -> Tuple[Symbol, ...]:
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

def intervals(polys: List[Poly]) -> List[Expr]:
    """
    Return points where the polynomials change their signs.
    When one of the polynomials is not in QQ or ZZ, return [].
    If no signs are changed, return [0].
    """
    if len(polys) == 0:
        return [Integer(0)]
    if any(_.domain not in [QQ, ZZ] for _ in polys):
        return []
    ret = []
    pre = None
    from sympy import intervals as _intervals
    for (l,r), mul in _intervals(polys):
        if l != pre:
            ret.append(l)
            pre = l
        if r != pre:
            ret.append(r)
            pre = r
    if len(ret):
        return ret
    return [Integer(0)]


def sum_y_exprs(y: List[Expr], exprs: List[Expr]) -> Expr:
    """
    Return sum(y_i * expr_i).
    """
    def _mul(v, expr):
        if v == 0: return 0
        x, f = (v * expr).radsimp(symbolic=False).together().as_coeff_Mul()
        return radsimp(x) * f
    return sum(_mul(*args) for args in zip(y, exprs))


def rationalize_func(
    poly: Union[Poly, Rational],
    validation: Callable[[Rational], bool],
    validation_initial: Optional[Callable[[Rational], bool]] = None,
    direction: int = 0,
) -> Optional[Rational]:
    """
    Find a rational number near the roots of poly that satisfies certain conditions.

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


def congruence_solve(M: Matrix, mapping = Union[List, Callable]) -> Optional[Expr]:
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


def has_gen(gen: Symbol, *args):
    """
    Test whether a symbol is involved in a (list of) polynomial(s).
    """
    to_iter = lambda x: (x,) if isinstance(x, Poly) else x
    return any(any(gen in p.free_symbols for p in arg) for arg in map(to_iter, args))


def clear_free_symbols(poly: Poly, ineq_constraints: Dict[Poly, Expr] = {}, eq_constraints: Dict[Poly, Expr] = {}) -> Tuple[Poly, Dict[Poly, Expr], Dict[Poly, Expr]]:
    """
    Clear nuisance free symbols from the polynomial and constraints.
    For example, if we want to solve x>=4 with constraints x>=y, x*y>=4, y>=0, a>=0. Then
    we can remove the symbol "a" from the constraints. But we cannot remove the symbol "y"
    even though it is not in the polynomial, as it is correlated with "x".
    """
    from ..problem import InequalityProblem
    pro = InequalityProblem(poly, ineq_constraints, eq_constraints)
    pro.remove_redundancy()
    return pro.expr, pro.ineq_constraints, pro.eq_constraints

    # # Construct the "correlation" graph of the free symbols: symbols that are path-connected to free
    # # vars in the polynomial are considered as active symbols.
    # gens = poly.gens
    # ufs = dict((gen, gen) for i, gen in enumerate(gens))
    # def ufsunion(ufs: dict, p):
    #     v = p.free_symbols
    #     if len(v):
    #         x0 = v.pop()
    #         y0 = ufsfind(ufs, x0)
    #         for x in v:
    #             ufs[ufsfind(ufs, x)] = y0
    # ufsunion(ufs, poly)
    # for p in ineq_constraints:
    #     ufsunion(ufs, p)
    # for p in eq_constraints:
    #     ufsunion(ufs, p)

    # # Using .free_symbols is not the same as .gens. Because free_symbols exclude 0-degree gens.
    # active_gens = set(ufsfind(ufs, gen) for gen in poly.free_symbols)
    # active_gens = [x for x in gens if ufsfind(ufs, x) in active_gens]
    # if len(active_gens) == 0:
    #     active_gens = (gens[0],) # the polynomial is a constant, but we need a gen to create a poly

    # def is_active(p):
    #     return len(p.free_symbols.intersection(active_gens))

    # poly = poly.as_poly(active_gens)
    # ineq_constraints = {p.as_poly(active_gens): e for p, e in ineq_constraints.items() if is_active(p)}
    # eq_constraints = {p.as_poly(active_gens): e for p, e in eq_constraints.items() if is_active(p)}
    # return poly, ineq_constraints, eq_constraints


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


def sos_struct_reorder_symmetry(groups: Tuple[int, ...]) -> Callable:
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

            coeff = poly
            if isinstance(poly, Coeff):
                pass
            elif isinstance(poly, Poly):
                coeff = Coeff(poly)
            else:
                raise TypeError("Unsupported polynomial type. Expected Coeff or Poly, but received %s." % type(poly))

            n = len(poly.gens)
            ufs = {i: i for i in range(n)}
            for i in range(n):
                for j in range(i+1, n):
                    if ufsfind(ufs, i) == ufsfind(ufs, j):
                        continue
                    if coeff.is_symmetric(Permutation(size=n)(i, j)):
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
            new_coeff = coeff.reorder(inds)
            return solver(new_coeff, *args, **kwargs)
        return _wrapped_solver
    return wrapper
