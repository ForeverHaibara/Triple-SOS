from itertools import combinations
from typing import List, Tuple, Optional, Union, Dict

import sympy as sp
from sympy import count_ops
import numpy as np

from .basis import LinearBasis, LinearBasisTangent
from .updegree import LinearBasisMultiplier
from ...utils.expression.cyclic import CyclicSum
from ...utils import MonomialManager, Coeff, SolutionSimple

def _merge_common_basis(
        y: List[sp.Expr], powers: List[Tuple], symbols: List[sp.Symbol]
    ) -> sp.Expr:
    """
    Get the common base and the extraction of the expression.
    """
    basis = sp.Add(*(yi*sp.Mul(*[s**p for s, p in zip(symbols, power)]) for yi, power in zip(y, powers))).together()
    return basis


class SolutionLinear(SolutionSimple):
    """
    Solution of Linear SOS. It takes the form of 
    f(a,b,c) * multiplier = sum(y_i * basis_i)
    """
    method = 'LinearSOS'
    @classmethod
    def _from_y_basis(cls,
            problem: sp.Poly,
            y: List[sp.Rational],
            basis: List[LinearBasis],
            symmetry: MonomialManager,
            ineq_constraints: Dict[sp.Poly, sp.Expr] = {},
            eq_constraints: Dict[sp.Poly, sp.Expr] = {},
            is_equal: bool = True,
            collect: bool = True
        ) -> 'SolutionLinear':
        """
        Generate the solution from y and basis by LinearSOS. Collect terms
        automatically if collect is True.

        Parameters
        ----------
        problem : sp.Poly
            The polynomial problem.
        y : List[sp.Rational]
            The coefficients of the bases.
        basis : List[LinearBasis]
            The list of bases of the solution.
        symmetry : MonomialManager
            The monomial manager object indicating the symmetry group.
        ineq_constraints : Dict[sp.Poly, sp.Expr]
            The inequality constraints.
        eq_constraints : Dict[sp.Poly, sp.Expr]
            The equality constraints.
        is_equal : bool
            Whether the solution is equal to the problem. Should be false
            for numerical or inexact solutions.
        collect : bool
            Whether to collect the terms in the solution. Default is True.
        """
        solution = _collect_constraints(y, basis, problem.gens, symmetry,
                        ineq_constraints, eq_constraints, collect=collect)
        obj = SolutionLinear(problem, solution,
                            ineq_constraints=ineq_constraints, eq_constraints=eq_constraints,
                             is_equal=is_equal)
        return obj


def _collect_constraints(
        y: List[sp.Rational], basis: List[LinearBasis], symbols: List[sp.Symbol], symmetry: MonomialManager,
        ineq_constraints: Dict[sp.Poly, sp.Expr], eq_constraints: Dict[sp.Poly, sp.Expr], collect: bool = True
    ) -> sp.Expr:
    """
    Collect terms and extract common factors wisely to simplify the expression.
    For instance, `a*b*(a+b-c)**2 + b*c*(a+b-c)**2` may be simplified to `b*(a+c)*(a+b-c)**2`.

    Parameters
    ----------
    y : List[sp.Rational]
        The coefficients of the bases.
    basis : List[LinearBasis]
        The list of bases of the solution.
    symbols : List[sp.Symbol]
        The list of symbols.
    symmetry : MonomialManager
        The monomial manager object indicating the symmetry group.
    ineq_constraints : Dict[sp.Poly, sp.Expr]
        The inequality constraints.
    eq_constraints : Dict[sp.Poly, sp.Expr]
        The equality constraints.
    collect : bool
        Whether to collect the terms in the solution. Default is True.
    """
    multiplier, y, basis = _collect_multipliers(y, basis, symbols, symmetry)

    # now y and basis must not include LinearBasisMultiplier objects

    inv_ineq_constraints = {v: k for k, v in ineq_constraints.items()}
    inv_eq_constraints = {v: k for k, v in eq_constraints.items()}
    if (not collect) or (len(inv_ineq_constraints) < len(ineq_constraints)) or (len(inv_eq_constraints) < len(eq_constraints)):
        # unsafe and is not expected to happen
        return sp.Add(*[symmetry.cyclic_sum(v * base.as_expr(symbols), symbols) for v, base in zip(y, basis)]) / multiplier

    nontangent_part = []
    ineq_part = []
    eq_part = {k: [] for k in inv_eq_constraints}

    def has_eq(tangent):
        """Check whether a sympy expression involves equality constraints.
        If True, return the constraint and the remaining part (= tangent/constraint)."""
        args = tangent.args if tangent.is_Mul else (tangent,)
        for i, arg0 in enumerate(args):
            arg = arg0.base if arg0.is_Pow else arg0
            if arg in inv_eq_constraints:
                mul_other_args = args[:i] + args[i+1:]
                if arg0.is_Pow:
                    mul_other_args += (arg ** (arg0.exp - 1),)
                mul_other_args = sp.Mul(*mul_other_args) # == tangent / arg
                return arg, mul_other_args
        return None

    for v, base in zip(y, basis):
        if isinstance(base, LinearBasisTangent):
            tangent = base.tangent(symbols)
            has_eq_part = has_eq(tangent)
            if has_eq_part is not None:
                if not (has_eq_part[0] in eq_part): # not expected to happen
                    eq_part[has_eq_part[0]] = []
                eq_part[has_eq_part[0]].append(v * has_eq_part[1] * sp.Mul(*[s**p for s, p in zip(symbols, base.powers)]))
            else:
                ineq_part.append((tangent, v * sp.Mul(*[s**p for s, p in zip(symbols, base.powers)])))
        else:
            nontangent_part.append(symmetry.cyclic_sum(v * base.as_expr(symbols), symbols))

    # Nontangent part: sum them up
    nontangent_part = sp.Add(*nontangent_part)

    # Equality constraints: extract them afront
    eq_part = _collect_eq_constraints(eq_part, symbols, symmetry, inv_eq_constraints)

    # Inequality constraints: extract them afront
    ineq_part = _collect_ineq_constraints(ineq_part, symbols, symmetry, inv_ineq_constraints)

    return (nontangent_part + ineq_part + eq_part) / multiplier



def _collect_multipliers(y: List[sp.Rational], basis: List[LinearBasis], symbols: List[sp.Symbol],
        symmetry: MonomialManager) -> Tuple[sp.Expr, List[sp.Rational], List[LinearBasis]]:
    """
    Separate the multipliers from other parts.

    In LinearSOS, the problem takes a general pattern

        `0 = - f*sum(z_j * multiplier_j) + sum(y_i * basis_i)`

    so that `f = sum(y_i * basis_i) / sum(z_j * multiplier_j)`.
    This function separates the multipliers from the other parts.

    Parameters
    ----------
    y : List[sp.Rational]
        The coefficients of the bases.
    basis : List[LinearBasis]
        The list of bases of the solution.
    symbols : List[sp.Symbol]
        The list of symbols.
    symmetry : MonomialManager
        The monomial manager object indicating the symmetry group.

    Returns
    -------
    multiplier : sp.Expr
        The multiplier.
    non_mul_y : List[sp.Rational]
        The coefficients of the bases without multipliers.
    non_mul_basis : List[LinearBasis]
        The list of bases without multipliers.
    """
    multipliers = []
    non_mul_y = []
    non_mul_basis = []
    for v, base in zip(y, basis):
        if isinstance(base, LinearBasisMultiplier):
            multipliers.append(v * base.multiplier)
        else:
            non_mul_y.append(v)
            non_mul_basis.append(base)

    # discard the constant part
    r, multiplier = sp.Add(*multipliers).as_content_primitive()
    # r = S.One

    non_mul_y = [v / r for v in non_mul_y]
    multiplier = symmetry.cyclic_sum(multiplier, symbols)
    return multiplier, non_mul_y, non_mul_basis


def _collect_eq_constraints(eqs_and_exprs: Dict[sp.Poly, List[sp.Expr]], symbols: List[sp.Symbol],
        symmetry: MonomialManager, inv_eq_constraints: Dict[sp.Expr, sp.Poly]) -> sp.Expr:
    """
    Collect terms involving equality constraints (vanishing polynomials).

    Suppose the problem involves linear constraints h1(x) == 0, ..., hk(x) == 0,
    and the final solution can be written as

        `f = (sum of squares) + ... + h1 * poly1 + ... + hk * polyk`.
    
    We gather terms involving h1, ..., hk to form poly1, ..., polyk.

    Parameters
    ----------
    eqs_and_exprs : Dict[sp.Poly, List[sp.Expr]]
        The dictionary of equality constraints and the corresponding expressions.
    symbols : List[sp.Symbol]
        The list of symbols.
    symmetry : MonomialManager
        The monomial manager object indicating the symmetry group.
    inv_eq_constraints : Dict[sp.Expr, sp.Poly]
        The dictionary mapping equality expressions to the their polynomial forms.
    """
    perm_group = symmetry.perm_group
    eqs_and_exprs = {k: sp.Add(*v).together() for k, v in eqs_and_exprs.items()}
    eq_part = []

    for k, v in eqs_and_exprs.items():
        poly = inv_eq_constraints.get(k)
        if poly is not None and Coeff(poly).is_cyclic(perm_group):
            # if it is cyclic with respect to the symmetry group, we can
            # extract the eq_constraint outside the cyclic sum operator
            eq_part.append(k * symmetry.cyclic_sum(v, symbols))
        else:
            eq_part.append(symmetry.cyclic_sum(k * v, symbols))

    return sp.Add(*eq_part)


def _generate_01_vectors(m: int, nz: int):
    """Generate 01 vectors of shape (m,) with `nz` nonzero entries."""
    if nz == 0:
        return np.zeros((1, m), dtype=int)
    if nz == m:
        return np.ones((1, m), dtype=int)
    indices = list(combinations(range(m), nz))
    result = np.zeros((len(indices), m), dtype=int)
    rows = np.repeat(np.arange(len(indices)), nz)
    cols = np.concatenate(indices)
    result[rows, cols] = 1
    return result

def _solve_optimal_factor(A: np.ndarray, c: np.ndarray, nz: int = 5) -> np.ndarray:
    """
    Given A (N, m) and c (m,), where A is a 01 matrix, solve for vector v such that
    v is 01 to maximize (k-1) * (v^Tc) where k = (np.all(A >= v, axis=1)).sum()
    and 1^Tv <= nz.
    """
    N, m = A.shape
    best, best_v = 0, np.zeros(m, dtype=int)
    for nonzero in range(1, min(nz, m) + 1):
        candidates = _generate_01_vectors(m, nonzero)
        vdotc = candidates @ c
        ks = [np.all(A >= v, axis=1).sum() for v in candidates]
        ks = np.array(ks, dtype=int)
        rewards = (ks - 1) * vdotc
        best_idx = np.argmax(rewards)
        if rewards[best_idx] > best:
            best = rewards[best_idx]
            best_v = candidates[best_idx]
    return best_v

def _solve_optimal_factor_recursively(A: np.ndarray, c: np.ndarray,
        nz: int = 5, iter_times: int = 1000) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Call `_solve_optimal_factor` recursively until no factor can be found.
    Return a list of tuples (v, indices) where v represents each factor
    and indices are terms containing the factor.
    """
    ret = []
    v = np.array([1] * A.shape[1], dtype=int)
    indices = np.arange(A.shape[0])

    for _ in range(iter_times): # prevent infinite loop
        v = _solve_optimal_factor(A, c, nz)
        mask = np.all(A >= v, axis=1)
        if (not v.any()) or (not mask.any()):
            break
        ret.append((v, indices[mask]))
        A = A[~mask]
        indices = indices[~mask]

    ret.append((np.zeros(A.shape[1], dtype=int), indices)) # remaining
    return ret

def _collect_ineq_constraints(tangents_and_exprs: List[Tuple[sp.Expr, sp.Expr]], symbols: List[sp.Symbol],
        symmetry: MonomialManager, inv_ineq_constraints: Dict[sp.Poly, sp.Expr]) -> sp.Expr:
    """
    Collect terms involving inequality constraints heuristically.

    Typically, every basis of LinearSOS can be written in the form of

        `coefficient * g1^p1 * g2^p2 * ... * gn^pn * (other exprs)`

    where g1, g2, ..., gn >= 0 are given constraint functions or
    something trivially nonnegative, e.g., the square of a polynomial. Values
    p1, p2, ..., pn are in {0,1}, indicating whether the corresponding
    constraint is present. We can assign the "complexity" of each basis by

        `p1 * complexity(g1) + p2 * complexity(g2) + ... + pn * complexity(gn)`.

    Here we ignore the complexity of the coefficient and "other exprs".
    Complexity of a single `gi` can be defined as the operator counts using
    `count_ops` from SymPy. The total complexity of the solution is the sum
    of the complexities of all bases. Such complexity metric evaluates how
    simple an expression is. We wish to collect terms so that the complexity
    is heuristically minimized.

    Suppose we are collecting `g1^v1 * g2^v2 * ... * gn^vn` where `vi` are in {0,1}.
    Then bases with powers no lower than the vector `v` extract this factor.
    Suppose there are k bases that can extract this factor. Then they reduce to

        `(g1^v1 * g2^v2 * ... * gn^vn) * (... + ... + ...)`.

    Suppose there are k bases, then k common factors reduce to one after collection.
    Thus we reduce the total complexity by `(k-1) * complexity(g1^v1 * g2^v2 * ... * gn^vn)`.
    Hence the goal is to maximize `(k-1) * (v^T * complexity(g))` where `v` is a binary
    vector. See `_solve_optimal_factor` for the detailed algorithm. This
    collecting process is repeated until no factor can be found.
 
    Parameters
    ----------
    tangents_and_exprs : List[Tuple[sp.Expr, sp.Expr]]
        The list of tuples (tangent, expression) so that y[i]*basis[i] = tangent * expression.
        The tangent part is where we can extract common factors.
    symbols : List[sp.Symbol]
        The list of symbols.
    symmetry : MonomialManager
        The monomial manager object indicating the symmetry group.
    inv_ineq_constraints : Dict[sp.Poly, sp.Expr]
        The dictionary mapping inequality expressions to the their polynomial forms.
    """
    # 0. if we skip collection:
    # return sp.Add(*[symmetry.cyclic_sum(t * e, symbols) for t, e in tangents_and_exprs])

    # 1. gather all factors
    one = sp.S.One
    factors = set(inv_ineq_constraints.keys())
    for i, (tangent, expr) in enumerate(tangents_and_exprs):
        r, args = tangent.as_coeff_mul()
        if r < 0:
            tangents_and_exprs[i] = (one, tangent * expr) # keep everything unchanged
            continue
        elif not (r is one):
            tangents_and_exprs[i] = (sp.Mul(*args), r * expr) # move the constant to expr

        factors.update(set(args))

    # 2. compute the "complexity" metric of each factor
    factors = list(factors)
    m = len(factors)
    factor_inds = {f: i for i, f in enumerate(factors)}

    complexity = [min(count_ops(inv_ineq_constraints.get(k, k)), 1000) for k in factors]
    complexity = np.array(complexity, dtype=int)
    complexity = np.where(complexity > 1, complexity, 0) # treat monomials as 0 complexity

    # 3. get the degree of each factor to form the binary matrix
    degrees = np.zeros((len(tangents_and_exprs), m), dtype=int)
    for i, (tangent, expr) in enumerate(tangents_and_exprs):
        args = tangent.as_coeff_mul()[1]
        for arg in args:
            degrees[i, factor_inds[arg]] += 1

    # 4. compute optimal factors recursively
    selections_and_inds = _solve_optimal_factor_recursively(degrees, complexity)

    # 5. group by selections of common factors
    ret = []
    for selection, inds in selections_and_inds:
        if np.any(selection): # not all zero
            common_factor = sp.Mul(*[factors[j] for j in range(m) if selection[j]])
            other_degrees = (degrees[inds, :] - selection.reshape(1, -1)) > 0
            other_factors = [sp.Mul(*[factors[j] for j in range(m) if other_degrees[i, j]])
                                    for i in range(len(inds))]
            other_factors = [_ * tangents_and_exprs[i][1] for i, _ in zip(inds, other_factors)]
            other_factors = sp.Add(*other_factors).together()
            ret.append(symmetry.cyclic_sum(common_factor * other_factors, symbols))

        else:
            ret += [symmetry.cyclic_sum(
                        tangents_and_exprs[i][0] * tangents_and_exprs[i][1], symbols) for i in inds]

    return sp.Add(*ret)