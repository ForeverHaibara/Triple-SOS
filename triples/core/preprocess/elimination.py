from typing import Tuple, List, Set, Optional, Callable, TYPE_CHECKING

from sympy import Poly, Symbol, Integer, Mul, QQ, ZZ
from sympy import MutableDenseMatrix as Matrix
from sympy.polys.matrices.sdm import SDM

from ..solution import Solution
from ...utils import (
    CyclicSum, verify_symmetry, poly_reduce_by_symmetry,
    resultant_bezout
)

if TYPE_CHECKING:
    from sympy import Expr
    from ..problem import InequalityProblem


def _identify_matrix_symmetry(S: Matrix) -> List[List[int]]:
    """
    Given a symmetric matrix `S`, indices `i, j` are
    equivalent if exchanging `i, j` does not change `S`.
    Identify the clusters of equivalent indices.
    """
    n = S.shape[0]
    parent = list(range(n))

    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    ddm = S._rep.rep.to_ddm()
    _S = lambda i, j: ddm[i][j]

    # this can be optimized, e.g. checking the elements
    # in the first row
    for i in range(n):
        for j in range(i + 1, n):
            if _S(i, i) != _S(j, j):
                continue

            is_equivalent = True
            for k in range(n):
                if k == i or k == j:
                    continue
                if _S(i, k) != _S(j, k):
                    is_equivalent = False
                    break

            if is_equivalent:
                union(i, j)

    cluster_dict = {}
    for i in range(n):
        root = find(i)
        if root not in cluster_dict:
            cluster_dict[root] = []
        cluster_dict[root].append(i)

    clusters = sorted(cluster_dict.values(), key=lambda x: min(x))
    return clusters

def _rowwise_primitive(A: Matrix) -> Matrix:
    """
    Apply primitive to each row of `A`.
    """
    from sympy.polys.densetools import dup_primitive
    rows = A._rep.rep.to_sdm()
    dom = A._rep.domain
    for row, terms in rows.items():
        prim_terms = dup_primitive(list(terms.values()), dom)[1]
        rows[row] = dict(zip(terms.keys(), prim_terms))
    return Matrix._fromrep(A._rep.from_rep(SDM(rows, A.shape, dom)).convert_to(ZZ))


def _symmetry_adapted_nullspace(A: Matrix) -> Tuple[Matrix, List[List[int]]]:
    m, n = A.shape
    A = (-Matrix.eye(m, m)).row_join(A)

    P2 = A.T * (A * A.T).inv() * A
    P = Matrix.eye(*P2.shape) - P2

    # rearrange indices by clustered symmetry
    clusters = _identify_matrix_symmetry(P[m:, m:])
    concatenated_clusters = list(range(m)) + [i+m for group in clusters for i in group]
    P = P[concatenated_clusters, concatenated_clusters]

    basis, pivots = P.rref()
    rank = max(pivots) + 1
    basis = basis[:rank, m:].T

    return basis, clusters

def _inv_integer_matrix(X: Matrix) -> Matrix:
    """
    Given an integer, full-column-rank matrix `X`, find an integer matrix
    `A` such that `AX = I` by Smith normal form.

    If such integer matrix exists, returns `A`. If not, returns a rational
    matrix `A` such that `AX = I` but `A._rep.domain = QQ`.
    """
    if X.shape[0] < X.shape[1]:
        raise ValueError("X should have more rows than columns.")
    from ...utils.normalforms import smith_normal_decomp
    smith, left, right = smith_normal_decomp(X._rep)

    # left * X * right == smith
    # =>  right * smith.pinv() * left * X == I

    # on low versions of SymPy, domainmatrices does not have `diagonal()`
    diag = [smith.rep.getitem(i, i) for i in range(min(smith.rep.shape))]
    if any(v == 0 for v in diag):
        raise ValueError("X should be full rank.")
    if not all(v == 1 for v in diag):
        # cast pinv to QQ
        pinv = smith.from_rep(SDM({i: {i: QQ(1, v)} for i, v in enumerate(diag)},
                (smith.shape[1], smith.shape[0]), QQ))
        left = pinv * left
    return Matrix._fromrep(right * left)


def _get_free_symbols(symbols: Set[Symbol], n: int, prefix: str="x") -> List[Symbol]:
    counter = 0
    m = len(prefix)
    for symbol in symbols:
        name = symbol.name
        if name.startswith(prefix) and name[m:].isdecimal():
            counter = max(counter, int(name[m:]))
    counter += 1
    return [Symbol(prefix+str(i)) for i in range(counter, counter+n)]


def _get_power_signs(
    A: Matrix,
    signs: List[Tuple[int, Tuple[Optional[int], Optional["Expr"]]]],
    check_signs: bool = True
) -> List[Tuple[Optional[int], Optional["Expr"]]]:
    """
    Infer the signs of new generators defined by
    ```
    new_gens[i] = Mul(*[g**p for g, p in zip(gens, row)])
    ```
    """
    inferred = [None] * A.shape[0]

    is_qq = A._rep.domain.is_QQ
    is_even = lambda x: x % 2 == 0
    if is_qq:
        is_even = lambda x: x.numerator % 2 == 0 or x.denominator % 2 == 0

    has_zero = [i for i, (s, e) in enumerate(signs) if s == 0]

    for i, row in A._rep.rep.to_sdm().items():
        if check_signs and is_qq:
            # TODO: the positive check should be done on all symbols,
            # e.g., sqrt(a*b) requires only a*b >= 0, not a >= 0 and b >= 0.
            for j, v in row.items():
                if v.denominator % 2 == 0:
                    if (signs[j][0] is None) or signs[j][0] < 0:
                        raise ValueError(f"Require squareroots on non-positive symbol {i}.")

        if check_signs:
            for j in has_zero:
                if row.get(j, 0) < 0:
                    raise ValueError(f"Require negative powers on zero symbol {i}.")

        sgn = 1
        for j, v in row.items():
            s = signs[j][0]
            if s == 0:
                sgn = 0
                break
            if is_even(v):
                continue
            # if odd, it is determined by the sign
            if s is None:
                # the sign is undetermined
                sgn = None
                break
            elif s > 0:
                pass
            elif s < 0:
                sgn = -sgn
        if sgn is None:
            inferred[i] = (None, None)
        else:
            proof = Mul(*[signs[j][1]**v for j, v in row.items()])
            inferred[i] = (sgn, proof)
    return inferred


def eliminate_power_constraints(
    problem: "InequalityProblem",
    irrational_expr: bool = True,
    check_signs: bool = True,
    recompute_constraints: bool = True,
    remove_redundacy: bool = True,
):
    """
    Eliminate power-type equality constraints from the problem.

    Parameters
    ----------
    problem : InequalityProblem
        The problem to eliminate power-type equality constraints from.
    irrational_expr : bool, optional
        Whether to allow irrational expressions in the substitution.
        Default is True.
    check_signs : bool, optional
        Whether to check the signs of the symbols in the problem before
        taking radicals. Default is True.
    recompute_constraints : bool, optional
        Whether to recompute the constraints of the problem to simplify the
        problem. Default is True.
    remove_redundacy : bool, optional
        Whether to remove the redundant constraints of the problem to
        simplify the problem. Default is True.
    """
    ineq_constraints = problem.ineq_constraints
    eq_constraints   = problem.eq_constraints

    mat = []
    exprs = []
    eq_inds = set()
    for i, (eq, e) in enumerate(eq_constraints.items()):
        terms = eq.terms()
        if len(terms) != 2:
            continue
        m0, c0 = terms[0]
        m1, c1 = terms[1]
        c = (-c0/c1)
        m = tuple(i - j for i, j in zip(m0, m1))
        if c != 1:
            continue
        exprs.append(e/Mul(-c1, *[g**p for g, p in zip(eq.gens, m1)]))
        mat.append(list(m))
        eq_inds.add(i)
    if len(mat) == 0:
        return problem, lambda x: x
    mat = Matrix(mat)

    basis, clusters = _symmetry_adapted_nullspace(mat)

    # use an integer basis
    basis = _rowwise_primitive(basis.T).T

    try:
        inv_basis = _inv_integer_matrix(basis)
    except ValueError:
        return problem, lambda x: x
    # print('basis =', repr(basis))
    # print('inv_basis =', repr(inv_basis))
    if (not irrational_expr) and (not inv_basis._rep.domain.is_ZZ):
        return problem, lambda x: x

    gens = problem.gens
    signs = problem.get_symbol_signs()
    ind_signs = [signs[gens[i]] for i in range(len(gens))]
    try:
        new_signs = _get_power_signs(inv_basis, ind_signs, check_signs=check_signs)
    except ValueError:
        return problem, lambda x: x

    new_gens = _get_free_symbols(problem.free_symbols, basis.shape[1])
    transform = {
        g0: Mul(*[g**p for g, p in zip(new_gens, row)])
            for g0, row in zip(problem.gens, basis.tolist())
    }
    inv_transform = {
        g: Mul(*[g0**p for g0, p in zip(problem.gens, row)])
            for g, row in zip(new_gens, inv_basis.tolist())
    }
    # print('transform =', transform, '\ninv_transform =', inv_transform)

    new_eqs = {k: e for i, (k, e) in enumerate(eq_constraints.items()) if i not in eq_inds}
    problem = problem.copy_new(problem.expr, ineq_constraints, new_eqs)

    problem, restore_transform = problem.transform(transform, inv_transform)

    # push symbol signs to constraints
    for g, (s, e) in zip(new_gens, new_signs):
        if s is None:
            continue
        if g in problem.expr.gens:
            g = Poly(g, problem.gens)
        if s == 0:
            problem.eq_constraints[g] = e
        elif s > 0:
            problem.ineq_constraints[g] = e
        elif s < 0:
            problem.ineq_constraints[-g] = e

    problem, restore_marginalize = problem.marginalize(
        {g: Integer(1) for g in new_gens[:mat.shape[0]]},
        {g: (e + 1).together()**p - 1 for g, e, p in zip(
            new_gens[:mat.shape[0]], exprs, inv_basis.diagonal()[:mat.shape[0]])})

    if recompute_constraints:
        problem = problem.recompute_constraints()
    if remove_redundacy:
        problem = problem.remove_redundancy()

    def composed(x):
        y = restore_transform(restore_marginalize(x))
        if y is None:
            return None
        return y.xreplace(
            {g: (e + 1).together()**p for g, e, p in zip(
                new_gens[:mat.shape[0]], exprs, inv_basis.diagonal()[:mat.shape[0]])})

    return problem, composed



#########################################################
#
#                       Resultant
#
#########################################################


def p2expr(p: Poly, symmetry = None) -> "Expr":
    """Convert a polynomial to expr wisely by leveraging the symmetry."""
    if (symmetry is not None) and verify_symmetry(p, symmetry):
        p = poly_reduce_by_symmetry(p, symmetry)
        return CyclicSum(p.as_expr(), p.gens, symmetry)
    return p.as_expr()


def eliminate_var_by_constraint(
    problem: "InequalityProblem",
    constraint: Poly,
    gen_index: int,
    remove_redundancy: bool = True,
    symmetry = None,
) -> Optional[Tuple["InequalityProblem", Callable]]:
    """
    Eliminate a variable using a constraint by the method
    of resultant.

    For each expression `F` in the problem, it computes
        `U*F + V*constraint == Resultant(F, constraint, gen)`
    to try eliminating the variable `gen`.
    """

    ineqs, eqs = problem.ineq_constraints, problem.eq_constraints
    is_eq = constraint in eqs
    if (not is_eq) and (constraint not in ineqs):
        return None

    constraint_expr = eqs[constraint] if is_eq else ineqs[constraint]

    from .signs import sign_sos
    signs = problem.get_symbol_signs()
    solver = lambda x: sign_sos(x, signs)

    gens = problem.gens
    gen_index = (gen_index + len(gens))%len(gens)
    _p2expr = lambda p: p2expr(p.as_poly(gens), symmetry)
    srcs = [(0, {-problem.expr: 0}), (1, ineqs), (2, eqs)]
    dst = [{}, {}, {}]

    has_trivial_cons = False

    new_expr = problem.expr
    restoration = lambda x: x

    #############################################
    # For fast check whether it is valid on the
    # marginalized problem, which avoids computing
    # a large resultant directly
    #############################################
    point = [[2,3,5][i%3] for i in range(len(gens))]
    reordered_gens = tuple(gens[:gen_index]) + tuple(gens[gen_index+1:]) + (gens[gen_index],)
    for i, g in enumerate(reordered_gens):
        sgn = signs[g][0]
        if sgn is None or sgn <= 0:
            point[i] = -point[i]
        elif sgn is not None and sgn == 0:
            point[i] = 0

    def _eval(f: Poly, point):
        f = f.reorder(*reordered_gens)
        f = f.eval(tuple(point)[:len(gens) - 2])
        const, f = f.primitive()
        if const < 0:
            f = -f
        return f
    #############################################

    for tp, src in srcs:
        for F, value in src.items():
            if F.degree(gen_index) <= 0:
                dst[tp][F] = value
                continue

            if tp != 0 and F == constraint:
                continue


            def _certificate_coeffs(U, V, res, p2expr=_p2expr, solver=solver):
                """
                Automatically selects the sign and tries to establish
                `U*F + V*constraint == res`
                """
                u_proof = p2expr(U)
                if tp <= 1:
                    u_proof = solver(u_proof)
                    if u_proof is None:
                        U, V, res = -U, -V, -res
                        u_proof = solver(p2expr(U))
                        if u_proof is None:
                            # the sign of U is not determined
                            return None


                v_proof = p2expr(V)
                if not is_eq: # constraint is ineq
                    v_proof = solver(v_proof)
                    if v_proof is None:
                        if tp <= 1:
                            # U >= 0 but we cannot prove V >= 0
                            return None
                        else:
                            U, V, res = -U, -V, -res
                            u_proof = -u_proof
                            v_proof = solver(p2expr(V))
                            if v_proof is None:
                                # the sign of V is not determined
                                return None
                return U, V, res, u_proof, v_proof


            #############################################
            # Fast check whether it is valid on the
            # marginalized problem, which avoids computing
            # a large resultant directly
            #############################################
            if len(gens) >= 3 and (not (is_eq and tp == 2)):
                F2 = _eval(F, point)
                constraint2 = _eval(constraint, point)
                U2, V2, res2 = resultant_bezout(F2, constraint2, gens[gen_index], reduced=True)
                cert = _certificate_coeffs(U2, V2, res2, p2expr=lambda x: x.as_expr())
                # print(f"Fast check U2 = {U2}; V2 = {V2};\ncert = {cert}")
                if cert is None:
                    # failed to prove U2, V2 >= 0
                    return None
            #############################################


            # U*F + V*constraint == res
            U, V, res = resultant_bezout(F, constraint, gens[gen_index], reduced=True)

            if tp == 0 and U.is_zero:
                # it should be handled differently because
                # U*F vanishes
                lc, rem = F.div(constraint)
                if rem.is_zero:
                    # F is a multiple of constraint
                    lc_expr = lc.as_expr()
                    if is_eq or (len(lc_expr.free_symbols) == 0 and lc_expr >= 0):
                        problem.solution = lc_expr * value
                        return problem, lambda x: x
                return None

            cert = _certificate_coeffs(U, V, res)
            # print(f'U = {U};\nV = {V};\nres = {res};\ncert = {cert}')
            if cert is None:
                # failed to prove U, V >= 0
                return None
            U, V, res, u_proof, v_proof = cert

            # now that we have U*F + V*constraint == res
            # and also U, V >= 0
            res = res.as_poly(gens)
            if tp == 0:
                # expr = -F = (-res + V*constraint)/U
                # hence we shall prove -res >= 0 to imply expr >= 0
                new_expr = -res
                u0_proof, v0_proof = u_proof, v_proof
                def restoration(x):
                    if x is None: return None
                    return (x + v0_proof*constraint_expr)/u0_proof

            else:
                # res == U*F + V*constraint >= 0
                dst_tp = 2 if is_eq and tp == 2 else 1

                if sign_sos(res.as_expr(), signs) is None:
                    # if not None, then res >= 0 is trivial and should not
                    # be pushed into constraints
                    dst[dst_tp][res] = u_proof*value + v_proof*constraint_expr
                else:
                    has_trivial_cons = True

    new_problem = problem.copy_new(new_expr, dst[1], dst[2])
    if remove_redundancy:
        new_problem = new_problem.remove_redundancy()

    if has_trivial_cons:
        new_problem = _inject_problem_signs(new_problem, signs)

    return new_problem, restoration


def _inject_problem_signs(problem: "InequalityProblem", signs):
    """
    Check whether the problem recovers the signs. If not,
    inject the signs into the problem.
    """
    new_signs = problem.get_symbol_signs()
    need_update = False
    new_ineqs, new_eqs = {}, {}
    for s in problem.gens:
        if new_signs[s][0] is None and signs[s][0] is not None:
            need_update = True
            sgn, val = signs[s]
            poly = Poly(s, problem.gens)
            if sgn == 0:
                new_eqs[poly] = val
            else:
                if sgn < 0:
                    poly = -poly
                new_ineqs[poly] = val

    if need_update:
        new_ineqs.update(problem.ineq_constraints)
        new_eqs.update(problem.eq_constraints)
        problem = problem.copy_new(
            problem.expr, new_ineqs, new_eqs)
    return problem


def is_binomial_in(poly: Poly, gen_index: int) -> bool:
    """
    Check if the polynomial is binomial in the variable
    at the given index.

    Examples
    --------
    >>> from sympy.abc import a, b, c, d, e
    >>> p = ((a**2+a*b**2+3*c**2*(d + 1)-3)*e).as_poly(a,b,c,d,e)
    >>> [is_binomial_in(p, i) for i in range(5)]
    [False, True, True, True, False]
    """
    a, b = -1, -1
    for m in poly.monoms():
        d = m[gen_index]
        if a == -1:
            a = d
        elif d == a:
            pass
        elif b == -1:
            b = d
        elif d == b:
            pass
        else:
            return False
    return (a != -1) and (b != -1)


def _try_resultant_elimination_in(
    problem: "InequalityProblem",
    gen_index: int,
    symmetry = None,
    **kwargs,
) -> Optional[Tuple["InequalityProblem", Callable]]:
    srcs = [problem.eq_constraints, problem.ineq_constraints]
    for src in srcs:
        for con in src:
            if is_binomial_in(con, gen_index):
                if symmetry is not None and not verify_symmetry(con, symmetry):
                    # breaks symmetry, skip it
                    continue
                attempt = eliminate_var_by_constraint(
                    problem, con, gen_index, symmetry=symmetry, **kwargs)
                if attempt is not None:
                    return attempt
    return None


def resultant_elimination(
    problem: "InequalityProblem",
    homogenize: bool = True,
    eliminate_binomial_constraints: bool = True,
    verbose: int = 0,
) -> Tuple["InequalityProblem", Callable]:
    """
    Heuristic elimination of variables using the method of resultant.
    """
    ineqs, eqs = problem.ineq_constraints, problem.eq_constraints
    if (not ineqs) and (not eqs):
        return problem, lambda x: x

    restorations = []
    eliminated_vars = []

    if homogenize and not problem.is_homogeneous:
        problem_hom, hom = problem.homogenize()
        symmetry = problem_hom.identify_symmetry()
        attempt = _try_resultant_elimination_in(
            problem_hom, -1, symmetry=symmetry)
        if attempt is not None:
            restorations.append(
                lambda x: Solution.dehomogenize(x, hom))
            problem = attempt[0]
            restorations.append(attempt[1])
            eliminated_vars.append(hom)

    if all(_.is_monomial for _ in problem.eq_constraints)\
        and all(_.is_monomial for _ in problem.ineq_constraints):
        # needs no transformation
        eliminate_binomial_constraints = False

    if eliminate_binomial_constraints:
        found = True
        ngens = len(problem.gens)
        while found and ngens:
            found = False
            symmetry = problem.identify_symmetry()
            orbits = symmetry.orbits()
            for orbit in orbits:
                if len(orbit) == 1:
                    # it is standalone and has no symmetry
                    ind = next(iter(orbit))
                    attempt = _try_resultant_elimination_in(
                        problem, ind, symmetry=symmetry)
                    if attempt is not None:
                        eliminated_vars.append(problem.gens[ind])
                        problem = attempt[0]
                        restorations.append(attempt[1])

                        if len(problem.gens) < ngens:
                            ngens = len(problem.gens)
                            found = True
                            break
                        else:
                            found = False
                            break

    if eliminated_vars and verbose:
        print("Resultant Elimination:", eliminated_vars)

    def restoration(sol):
        if sol is None:
            return None
        for rs in restorations[::-1]:
            sol = rs(sol)
        return sol
    return problem, restoration
