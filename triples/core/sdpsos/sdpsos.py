from itertools import combinations
from time import perf_counter
from typing import Union, Optional, List, Tuple, Dict, Callable, Generator, Any

from sympy import Poly, Expr, Integer, Mul, ZZ, QQ, RR
from sympy.combinatorics import PermutationGroup
from sympy import MutableDenseMatrix as Matrix

from .sos import SOSPoly
from .solution import SolutionSDP
from ..preprocess import ProofNode, ProofTree, SolvePolynomial
from ...utils import MonomialManager, Root, clear_polys_by_symmetry
from ...sdp import ArithmeticTimeout
from ...sdp.rationalize import SDPRationalizeError, DualRationalizer
from ...sdp.transforms.linear import SDPLinearTransform

class _lazy_iter:
    """A wrapper for recyclable iterators but initialized when first called."""
    def __init__(self, initializer: Callable[[], Any]):
        self._initializer = initializer
        self._iter = None
    def __iter__(self):
        if self._iter is None:
            self._iter = self._initializer()
        return iter(self._iter)

def _lazy_find_roots(problem, verbose=False):
    poly = problem.expr
    time1 = perf_counter()
    if not poly.domain.is_Exact:
        return []
    roots = problem.find_roots()
    if verbose:
        print(f"Time for finding roots num = {len(roots):<6d}     : {perf_counter() - time1:.6f} seconds.")
    return roots


def _vector_complement(row: Matrix):
    """
    Vector `row` is of shape (1, n).
    Returns (A, b) such that
    * A is of shape (n, n - 1), full rank, SPARSE, and `row @ A == 0`.
    * b is of shape (n, 1) and `row @ b == 1`.
    """
    from sympy.polys.matrices.domainmatrix import DomainMatrix
    from sympy.polys.matrices.sdm import SDM
    rep = row._rep.to_field()
    dok = rep.to_dok()
    if len(dok) == 0:
        # row is the zero vector
        # XXX: this is not expected to happen in the current.
        # However, if in the future there is block diagonalization,
        # this could be unsafe. Instead, we shall constrain the SDP
        # using the chain of transforms.
        raise ValueError("No poly_qmodule key found in child SDP.")
    items = list(dok.items())
    (_, nz), v = items[-1]

    n = row.shape[1]
    one = rep.domain.one
    A = {i: {i: one} for i in range(nz)}
    A.update({i: {i - 1: one} for i in range(nz + 1, n)})
    Anz = {}
    for (_, k), v2 in items[:-1]:
        Anz[k - int(k > nz)] = -v2 / v
    A[nz] = Anz
    A = Matrix._fromrep(DomainMatrix.from_rep(SDM(A, (n, n-1), rep.domain)))

    b = {nz: {0: one / v}}
    b = Matrix._fromrep(DomainMatrix.from_rep(SDM(b, (n, 1), rep.domain)))
    return A, b


def _get_qmodule_list(
    poly: Poly,
    ineq_constraints: List[Tuple[Poly, Expr]],
    degree: Optional[int] = None,
    ineq_constraints_with_trivial: bool = True,
    preordering: str = "linear-progressive",
    is_homogeneous: bool = True
) -> Generator[List[Tuple[Poly, Expr]], None, None]:
    """
    Generate the (generators of the) qmodule for the given problem.
    """
    _ACCEPTED_PREORDERINGS = ['none', 'linear', 'linear-progressive']
    if not preordering in _ACCEPTED_PREORDERINGS:
        raise ValueError("Invalid preordering method, expected one of %s, received %s." % (str(_ACCEPTED_PREORDERINGS), preordering))

    if degree is None:
        degree = poly.total_degree()
    poly_one = poly.one

    def degree_filter(polys_and_exprs):
        return [pe for pe in polys_and_exprs if pe[0].total_degree() <= degree \
                    and ((degree - pe[0].total_degree()) % 2 == 0 if is_homogeneous else True)]

    if preordering == 'none':
        if ineq_constraints_with_trivial:
            ineq_constraints = [(poly_one, Integer(1))] + ineq_constraints
        ineq_constraints = degree_filter(ineq_constraints)
        yield ineq_constraints
        return

    linear_ineqs = []
    nonlin_ineqs = [(poly_one, Integer(1))]
    for ineq, e in ineq_constraints:
        if ineq.is_linear:
            linear_ineqs.append((ineq, e))
        else:
            nonlin_ineqs.append((ineq, e))

    qmodule = nonlin_ineqs if ineq_constraints_with_trivial else nonlin_ineqs[1:]
    qmodule = degree_filter(qmodule)
    if 'progressive' in preordering:
        yield qmodule.copy()
    for n in range(1, len(linear_ineqs) + 1):
        has_additional = False
        for comb in combinations(linear_ineqs, n):
            mul = poly_one
            for c in comb:
                mul = mul * c[0]
            d = mul.total_degree()
            if d > degree:
                continue
            mul_expr = Mul(*(c[1] for c in comb))
            for ineq, e in nonlin_ineqs:
                new_d = d + ineq.total_degree()
                if new_d <= degree and ((not is_homogeneous) or (degree - new_d) % 2 == 0):
                    qmodule.append((mul * ineq, mul_expr * e))
                    has_additional = True

        if has_additional and 'progressive' in preordering:
            yield qmodule.copy()
    if 'progressive' not in preordering:
        # yield them all at once
        yield qmodule


def _is_infeasible(
    poly: Poly,
    qmodule: List[Poly],
    ideal: Union[list, dict],
    perm_group: PermutationGroup = None,
) -> bool:
    """
    Check if the problem is trivially infeasible. Returns True if infeasible.
    """
    if poly.is_zero:
        return False
    if len(qmodule) == 0 and len(ideal) == 0:
        return True

    nvars = len(poly.gens)
    if len(ideal) == 0:
        if all(_.is_monomial for _ in qmodule):
            qmodule_m = [_.monoms()[0] for _ in qmodule]
            # if the poly has odd degree on some var, but all monomials are even up to permutation,
            # then the poly is not SOS
            odd_degree_vars = [i for i in range(nvars) if poly.degree(i) % 2 == 1]
            for i in odd_degree_vars:
                for i2 in perm_group.orbit(i):
                    if any(m[i2] % 2 == 1 for m in qmodule_m):
                        # odd degree monomial found -> okay
                        break
                else:
                    return True
    return False


class SDPSOSSolver(ProofNode):
    """
    Solve a constrained polynomial inequality problem using semidefinite programming (SDP).

    Although the theory of numerically solving sum-of-squares problems using SDP is well established,
    there are certain limitations in practice. One of the most significant concerns is that we
    require an accurate, rational solution rather than a numerical one. If the SDP is strictly feasible
    and has strictly positive definite solutions, then a rational solution can be obtained by
    rounding an interior solution. See [1]. However, if the solution is semipositive definite,
    it is generally difficult or even impossible to round it to a rational solution.

    ### Facial Reduction

    To handle such cases, we need to compute the low-rank feasible set of the SDP in advance
    and solve the SDP on this subspace. This process is known as facial reduction.
    For example, consider Vasile's inequality `(a^2+b^2+c^2)^2 - 3*(a^3*b+b^3*c+c^3*a) >= 0`.
    Up to scaling, there are four equality cases. If this inequality can be written in the
    semidefinite form `x'Mx`, then `x'Mx = 0` at these four points. This implies that Mx = 0
    for these four vectors. As a result, the semidefinite matrix `M` lies in a subspace orthogonal
    to these four vectors. We can assume `M = QSQ'`, where `Q` is the nullspace of the four
    vectors`x`, and solve the SDP with respect to `S`. Currently, the low-rank subspace is computed
    heuristically by finding the equality cases of the inequality. Note that SOS that exploits
    term-sparsity through the Newton polytope is also a special case of facial reduction.

    This class provides a node to solve inequality problems using SDPSOS. For more flexible or
    low-level usage, such as manipulating the Gram matrices, please use `SOSPoly`.

    Reference
    ----------
    [1] Helfried Peyrl and Pablo A. Parrilo. 2008. Computing sum of squares decompositions with
    rational coefficients. Theor. Comput. Sci. 409, 2 (December, 2008), 269-281.

    Parameters
    ----------
    dof_limit: int
        The maximum degree of freedom of the SDP. When it exceeds `dof_limit`,
        the node will be pruned. This prevents crash in external SDP solvers. Default is 7000.
    solver: str
        The numerical SDP solver to use. When set to None, it is automatically selected. Default is None.
    allow_numer: int
        Whether to allow inexact numerical solution. This is useful when it fails to obtain an
        exact solution by rationalization.
    preordering: str
        The preordering method for extending the generators of the quadratic module. It can be
        'none', 'linear', 'linear-progressive'. Default is 'linear-progressive'.
    unstable_eig_threshold: float
        If it fails to rationalize but the smallest eigenvalue of the SDP is larger than
        `unstable_eig_threshold`, then it considers the problem as numerically unstable
        and stops further search. Default is -0.1.
    verbose: bool
        Whether to print the progress. Default is False.
    """
    default_configs = {
        "lift_degree_limit": 2,
        "dof_limit": 7000,
        "solver": None,
        "allow_numer": 0,
        "solve_kwargs": {},
        "ineq_constraints_with_trivial": True,
        "preordering": "linear-progressive",
        "unstable_eig_threshold": -0.1,
        "verbose": False,
    }

    _complexity_models = True
    _wrapped_problem = None
    _symmetry: MonomialManager

    def _prepare_qmodule(self, lift_degree: int, configs) -> Generator:
        from ..structsos.utils import zip_longest

        problem = self._wrapped_problem[0]
        poly = problem.expr
        ineq_constraints = problem.ineq_constraints
        eq_constraints = problem.eq_constraints
        symmetry = self._symmetry

        qmodule_list = _get_qmodule_list(
            poly, ineq_constraints.items(),
            degree = poly.total_degree() + lift_degree,
            ineq_constraints_with_trivial = configs['ineq_constraints_with_trivial'],
            preordering = configs['preordering'],
            is_homogeneous = self.problem.is_homogeneous
        )

        poly_qmodule_list = (None,)
        if lift_degree > 0:
            poly_qmodule_list = _get_qmodule_list(
                poly, ineq_constraints.items(),
                degree = lift_degree,
                ineq_constraints_with_trivial = configs['ineq_constraints_with_trivial'],
                preordering = configs['preordering'],
                is_homogeneous = self.problem.is_homogeneous
            )

        ideal = eq_constraints
        for qmodule, poly_qmodule in zip_longest(qmodule_list, poly_qmodule_list):
            qmodule_tuples = clear_polys_by_symmetry(qmodule, poly.gens, symmetry)
            qmodule = [e[0] for e in qmodule_tuples]

            if _is_infeasible(poly, qmodule, ideal, symmetry.perm_group):
                continue

            if configs["verbose"]:
                print(f"Lift Degree = {lift_degree}")
                print(f"Qmodule = {qmodule}\nIdeal   = {list(eq_constraints.keys())}")

            poly_qmodule_tuples = None
            if poly_qmodule:
                poly_qmodule_tuples = clear_polys_by_symmetry(poly_qmodule, poly.gens, symmetry)
                poly_qmodule_tuples = [(k * -poly, v) for k, v in poly_qmodule_tuples]
            yield qmodule_tuples, poly_qmodule_tuples


    def _create_lifted_sos_problem(self,
        qmodule: List[Poly],
        ideal: List[Poly],
        poly_qmodule: Optional[List[Poly]] = None,
        degree: int = 0,
        configs: dict = {},
    ) -> SOSPoly:
        poly = self.problem.expr

        roots = _lazy_iter(lambda: _lazy_find_roots(self.problem, configs['verbose']))
        # roots = self.problem.find_roots()
        symmetry = self._symmetry

        if poly_qmodule is None:
            sos = SOSPoly(poly, poly.gens,
                qmodule = qmodule, ideal = ideal,
                symmetry = symmetry.perm_group, roots = roots, degree=degree
            )
        else:
            sos = SOSPoly(poly.zero, poly.gens,
                qmodule = qmodule + poly_qmodule, ideal = ideal,
                symmetry = symmetry.perm_group, roots = roots, degree=degree
            )
        sdp = sos.construct(
            verbose=configs['verbose'],
            time_limit=configs['expected_end_time'] - perf_counter()
        )

        dof = sos.sdpp.dof
        if dof > configs["dof_limit"]:
            # XXX: the error will be caught,
            # but perhaps we shall use other errors
            raise MemoryError(f"Current dof = {dof} > dof_limit = {configs['dof_limit']}")

        if poly_qmodule is not None:
            time0 = perf_counter()
            self._constrain_poly_qmodule(sos, len(qmodule), len(poly_qmodule))
            if configs["verbose"]:
                print(f"Time to constrain poly_qmodule          :"\
                      + f" {perf_counter() - time0:.6f} seconds. Dof = {sos.sdpp.dof}")

        return sos

    def _constrain_poly_qmodule(self, sos: SOSPoly, q1: int, q2: int):
        """
        Constrain the sum of the diagonals of the poly_qmodule to be constant.
        """
        sdp = sos.sdpp
        dof = sdp.dof

        if dof == 0:
            # cannot happen
            raise ValueError("Zero dof")

        keys = [(sos, i) for i in range(q1, q1 + q2)]

        row = Matrix.zeros(1, dof)
        for key in keys:
            if not (key in sdp._x0_and_space):
                continue
            size = sdp.get_size(key)
            for i in range(size):
                # the diagonal element is in the i*(size+1)th row
                row = row + sdp._x0_and_space[key][1][i*(size+1),:]

        A, b = _vector_complement(row)
        sdpp = SDPLinearTransform.apply_from_affine(sdp, A, b)

        return sdpp


    def _explore_lift_degree(self, configs):
        if (self.state < 0) or self.state > configs["lift_degree_limit"]:
            # prevent dead loop
            self.state = -1
            self.finished = True
            return

        # Increase the status here to prevent the node getting killed in
        # the middle (which will not change the status, leading to infinite loop)
        lift_degree = self.state
        self.state += 1

        problem = self._wrapped_problem[0]
        poly = problem.expr
        ineq_constraints = problem.ineq_constraints
        eq_constraints = problem.eq_constraints

        verbose = configs["verbose"]

        degree = poly.total_degree() + lift_degree

        ideal = list(eq_constraints.keys())
        for qmodule_tuples, poly_qmodule_tuples in self._prepare_qmodule(lift_degree, configs):
            qmodule = [e[0] for e in qmodule_tuples]
            poly_qmodule = [e[0] for e in poly_qmodule_tuples] if poly_qmodule_tuples else None

            time0 = perf_counter()
            sos_problem = None
            # now we solve the problem
            try:
                sos_problem = self._create_lifted_sos_problem(
                    qmodule, ideal, poly_qmodule,
                    degree=degree, configs=configs
                )

                sdp_sol = sos_problem.solve(verbose=verbose,
                    solver=configs["solver"],
                    time_limit=configs['expected_end_time'] - perf_counter(),
                    allow_numer=configs['allow_numer'],
                    kwargs=configs['solve_kwargs']
                )

                if sdp_sol is not None:
                    if verbose:
                        print(f"Time for solving SDP{' ':20s}: {perf_counter() - time0:.6f} seconds. \033[32mSuccess\033[0m.")
                    self._as_solution(
                        sos_problem,
                        qmodule=dict(enumerate([e[1] for e in qmodule_tuples])),
                        ideal=dict(enumerate(eq_constraints.values())),
                        poly_qmodule=dict(
                            enumerate([e[1] for e in poly_qmodule_tuples], start=len(qmodule)))\
                                if poly_qmodule_tuples else None,
                        configs=configs
                    )
                    return

            except Exception as e:
                if verbose:
                    print(f"Time for solving SDP{' ':20s}: {perf_counter() - time0:.6f} seconds. \033[31mFailed with exceptions\033[0m.")
                    print(f"{e.__class__.__name__}: {e}")
                if isinstance(e, (ArithmeticTimeout, MemoryError)):
                    # do not try further
                    self.state = -1
                    self.finished = True
                    break
                elif isinstance(e, SDPRationalizeError):
                    # XXX: this is very heuristic
                    y = e.y
                    if y is not None:
                        dr = DualRationalizer(sos_problem.sdpp)
                        # y[:-1] to discard the eigenvalue relaxation,
                        # see source code in SDPProblem.solve
                        mineig = dr.mineig(y[:-1])
                        if mineig > configs["unstable_eig_threshold"]:
                            # we think that it is numerically unstable
                            self.state = -1
                            self.finished = True
                            break

                continue

        # We add a second check here to prevent
        # it triggers a new round of `explore`
        # note that self.state has already increased
        if self.state > configs["lift_degree_limit"]:
            self.state = -1
            self.finished = True
            return

        # loop
        self._explore_lift_degree(configs)

    def _as_solution(self,
        sos: SOSPoly,
        qmodule: Dict[int, Expr],
        ideal: Dict[int, Expr],
        poly_qmodule: Optional[Dict[int, Expr]] = None,
        configs: dict = {}
    ):
        def inject_zeros(d, keys):
            if not keys: return d
            d = d.copy()
            for key in keys:
                d[key] = Integer(0)
            return d

        solution = sos.as_solution(
            qmodule=inject_zeros(qmodule, poly_qmodule),
            ideal=ideal,
        ).solution

        if poly_qmodule:
            denom = sos.as_solution(
                qmodule=inject_zeros(poly_qmodule, qmodule),
                ideal={k: Integer(0) for k in ideal.keys()}
            ).solution

            solution = solution / denom

        cons_restoration = self._wrapped_problem[1]
        solution = cons_restoration(solution)

        self.problem.solution = solution
        return solution


    def explore(self, configs):
        if self.state < 0:
            return

        configs['expected_end_time'] = perf_counter() + configs['time_limit']

        nvars = len(self.problem.expr.gens)
        if self.state == 0:
            poly = self.problem.expr
            if nvars < 1 or (not self.problem.reduce(lambda p: p.domain in (ZZ, QQ, RR), all)):
                self.state = -1
                self.finished = True
                return None

            symmetry = MonomialManager(len(poly.gens), self.problem.identify_symmetry())
            self._symmetry = symmetry
            self._wrapped_problem = self.problem.wrap_constraints(symmetry.perm_group)

        if configs['verbose']:
            degree = self.problem.reduce(lambda x: x.total_degree(), max)
            print(f'SDPSOS nvars = {nvars} degree = {degree}')
            print('Identified Symmetry = %s' % \
                  str(self._symmetry.perm_group).replace('\n', '').replace('  ',''))

        # explore lift degree, from degree = 0
        self._explore_lift_degree(configs)


def SDPSOS(
    expr: Expr,
    ineq_constraints: Union[List[Expr], Dict[Expr, Expr]] = {},
    eq_constraints: Union[List[Expr], Dict[Expr, Expr]] = {},
    *,
    symmetry: Optional[PermutationGroup] = None,
    roots: Optional[List[Root]] = None,
    lift_degree_limit: int = 2,
    dof_limit: int = 7000,
    solver: Optional[str] = None,
    allow_numer: int = 0,
    solve_kwargs: Dict[str, Any] = {},
    ineq_constraints_with_trivial: bool = True,
    preordering: str = 'linear-progressive',
    unstable_eig_threshold: float = -0.1,
    verbose: bool = False,
    time_limit: float = 3600.,
) -> Optional[SolutionSDP]:
    """
    Solve a constrained polynomial inequality problem by semidefinite programming (SDP).

    See the documentation of `SDPSOSSolver` for more details.

    Parameters
    ----------
    expr: Expr
        The expression to perform SOS on.
    ineq_constraints: Union[List[Expr], Dict[Expr, Expr]]
        Inequality constraints to the problem. This assumes g_1(x) >= 0, g_2(x) >= 0, ...
    eq_constraints: Union[List[Expr], Dict[Expr, Expr]]
        Equality constraints to the problem. This assumes h_1(x) = 0, h_2(x) = 0, ...
    symmetry: Optional[PermutationGroup]
        CURRENTLY UNUSED.
    roots: Optional[List[Root]]
        The roots of the polynomial satisfying constraints. When it is None, it will be automatically generated.
    lift_degree_limit: int
        The maximum lift degree to explore.
    dof_limit: int
        The maximum degree of freedom of the SDP. When it exceeds `dof_limit`,
        the node will be pruned. This prevents crash in external SDP solvers. Default is 7000.
    solver: str
        The numerical SDP solver to use. When set to None, it is automatically selected. Default is None.
    allow_numer: int
        Whether to allow inexact numerical solution. This is useful when it fails to obtain an
        exact solution by rationalization.
    ineq_constraints_with_trivial: bool
        Whether to add the trivial inequality constraint 1 >= 0. This is used to generate the
        quadratic module. Default is True.
    preordering: str
        The preordering method for extending the generators of the quadratic module. It can be
        'none', 'linear', 'linear-progressive'. Default is 'linear-progressive'.
    unstable_eig_threshold: float
        If it fails to rationalize but the smallest eigenvalue of the SDP is larger than
        `unstable_eig_threshold`, then it considers the problem as numerically unstable
        and stops further search. Default is -0.1.
    verbose: bool
        Whether to print the progress. Default is False.
    """
    problem = ProofNode.new_problem(expr, ineq_constraints, eq_constraints)
    problem.set_roots(roots)
    configs = {
        ProofTree: {
            "time_limit": time_limit,
        },
        SolvePolynomial: {
            "solvers": [SDPSOSSolver],
        },
        SDPSOSSolver: {
            "lift_degree_limit": lift_degree_limit,
            "dof_limit": dof_limit,
            "solver": solver,
            "allow_numer": allow_numer,
            "solve_kwargs": solve_kwargs,
            "ineq_constraints_with_trivial": ineq_constraints_with_trivial,
            "preordering": preordering,
            "unstable_eig_threshold": unstable_eig_threshold,
            "verbose": verbose,
        }
    }
    return problem.sum_of_squares(configs)
