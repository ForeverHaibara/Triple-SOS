from typing import Tuple, List, Optional, Any, TYPE_CHECKING

from sympy import Add, QQ
from sympy import MutableDenseMatrix as Matrix
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.sdm import SDM

from ..problem import InequalityProblem, ProblemComplexity
from ..node import ProofNode
from ...sdp.arithmetic import congruence, reshape
from ...sdp import SDPProblem

if TYPE_CHECKING:
    from sympy import Poly

class QCQP(InequalityProblem):
    """
    Quadratic Constrained Quadratic Program (QCQP). Representing
    the inequality:

    Prove that `x.T * P0 * x >= 0` where
    `x.T * P_ineqs[i] * x >= 0` and `x.T * P_eqs[i] * x = 0` hold
    for all `i`.

    Matrices in `P0`, `P_ineqs`, and `P_eqs` have been augmented an extra row / column
    to represent the linear (nonhomogeneous) part of the expressions.
    """
    P0: Matrix
    P_ineqs: List[Matrix]
    P_eqs: List[Matrix]

    _is_convex: Optional[bool] = None

    @property
    def is_convex(self) -> bool:
        if self._is_convex is not None:
            return self._is_convex
        # the last row / column is the linear part
        if congruence(self.P0[:-1, :-1]) is None:
            self._is_convex = False
        elif any(eq[:-1, :-1]._rep.nnz() for eq in self.P_eqs):
            # every equality constraint must be linear
            self._is_convex = False
        elif all(congruence(-ineq[:-1, :-1]) is not None for ineq in self.P_ineqs):
            # every inequality constraint must be (-convex) >= 0
            self._is_convex = True
        else:
            self._is_convex = False
        return self._is_convex


def _compress_monom(n: int, m: Tuple[int, ...]) -> Tuple[int, int]:
    """
    Compress a quadratic monomial of a dense polynomial to a pair of indices.
    Defaults to (n, n).

    Examples
    --------
    >>> _compress_monom(5, (0, 1, 1, 0))
    (1, 2)
    >>> _compress_monom(5, (0, 0, 1, 0))
    (2, 5)
    >>> _compress_monom(5, (0, 0, 0, 0))
    (5, 5)
    """
    a, b = n, n
    for i, mi in enumerate(m):
        if mi:
            if a == n:
                a = i
                if mi == 2:
                    b = i
                    break
            else:
                b = i
                break
    return (a, b)


def formulate_qcqp(problem: InequalityProblem) -> Optional[Tuple[QCQP, Any]]:
    if isinstance(problem, QCQP):
        return problem, lambda x: x

    problem = problem.polylize(field=True, unify=True)

    # TODO: formulate the qcqp from expressions
    # without converting to a dense polynomial

    if problem.reduce(lambda x: x.total_degree() > 2, any):
        return None

    n = len(problem.gens)

    dom = problem.expr.domain

    def build_mat(expr: "Poly") -> Matrix:
        mat = {}
        for m, v in expr.rep.terms():
            a, b = _compress_monom(n, m)
            if a != b:
                v = v/2
                mat.setdefault(b, {})[a] = v
            mat.setdefault(a, {})[b] = v
        return Matrix._fromrep(DomainMatrix.from_rep(
            SDM(mat, (n + 1, n + 1), dom)))

    P0 = build_mat(problem.expr)
    P_ineqs = [build_mat(ineq) for ineq in problem.ineq_constraints]
    P_eqs = [build_mat(eq) for eq in problem.eq_constraints]

    pro = QCQP.new(problem.expr, problem.ineq_constraints, problem.eq_constraints)
    pro.P0 = P0
    pro.P_ineqs = P_ineqs
    pro.P_eqs = P_eqs
    return pro, lambda x: x


class QCQPSolver(ProofNode):
    """
    Specialized solver for Quadratic Constrained Quadratic Program (QCQP).

    It identifies QCQP and attempts to solve it before sending it to the
    generic polynomial optimization solvers. By leveraging matrix
    arithmetic, it reduces the overhead spent on polynomial manipulation.
    """
    problem: InequalityProblem
    wrapped_problem: QCQP
    restoration: Any

    default_configs = {
        "allow_numer": False,
    }

    def _evaluate_complexity(self) -> ProblemComplexity:
        # Fast in most cases
        return ProblemComplexity(0.005, 1.)

    def explore(self, configs):
        if self.state < 0:
            self.finished = True
            return
        if self.state == 0:
            result = formulate_qcqp(self.problem)
            if result is None:
                self.state = -1
                self.finished = True
                return
            problem, restoration = result


            self.wrapped_problem = problem
            self.restoration = restoration
            self.state = 1
            return

        if self.state == 1:
            self.state += 1

            solution = self.solve_dual(configs)
            if solution is not None:
                self.wrapped_problem.solution = solution
                self.solution = self.restoration(solution)
            return

        if self.state >= 2:
            self.state = -1
            self.finished = True

        # if not problem.is_convex:
        #     # nonconvex problems not implemented
        #     # TODO: relaxations
        #     self.state = -1
        #     self.finished = True
        #     return


    def make_dual_sdp(self, configs) -> SDPProblem:
        """
        Formulate the dual SDP problem for the QCQP.
        """
        problem = self.wrapped_problem
        n = problem.P0.shape[0] - 1
        m = (n + 1)**2
        x0 = reshape(problem.P0, (m, 1))
        dof = len(problem.P_ineqs) + len(problem.P_eqs)

        if not dof:
            sdp = SDPProblem([(x0, Matrix.zeros(m, 0))])
            return sdp

        space = Matrix.hstack(*[reshape(ineq, (m, 1)) for ineq in problem.P_ineqs],
                            *[reshape(eq, (m, 1)) for eq in problem.P_eqs])

        # lambda <= 0 constraints
        def neg_onehot(i):
            sdm = SDM({0: {i: -QQ.one}}, (1, dof), QQ)
            return Matrix._fromrep(DomainMatrix.from_rep(sdm))
        zero_mat = Matrix.zeros(1, 1)
        sdp = SDPProblem(
            [(x0, space)] + [(zero_mat, neg_onehot(i)) for i in range(len(problem.P_ineqs))]
        )
        return sdp

    def solve_dual(self, configs):
        """
        Try to solve the QCQP by finding multipliers that
        `P0 + sum(lambda * P_ineq) + sum(mu * P_eq)` is PSD
        and `lambda <= 0`.
        """
        problem = self.wrapped_problem
        sdp = self.make_dual_sdp(configs)
        sdp.constrain_block_structures()

        y = None
        try:
            y = sdp.solve(
                time_limit=configs["time_limit"],
                verbose=configs["verbose"],
                allow_numer=configs["allow_numer"]
            )
        except Exception as e:
            if configs["verbose"]:
                print(e)

        if y is None:
            return

        # form the sum-of-squares certificate
        U, D = sdp.decompositions[0]
        x = Matrix(problem.gens + (1,))
        ineqs = problem.ineq_constraints.values()
        eqs = problem.eq_constraints.values()
        sol = Add(*[d * row**2 for d, row in zip(D, U * x)],
                  *[(-lam) * val for lam, val in zip(y[:len(problem.P_ineqs)], ineqs)],
                  *[(-mu) * val for mu, val in zip(y[len(problem.P_ineqs):], eqs)],
                  evaluate=False)
        return sol
