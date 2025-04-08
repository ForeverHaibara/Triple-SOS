from .backend import DualBackend
from .settings import SDPStatus, SolverConfigs

class DualBackendCVXPY(DualBackend):
    """
    CVXPY backend for SDP problems.
    CVXPY is a Python-embedded modeling language for convex optimization problems.

    Warning: It seems that CVXPY cannot recognize dual SDP problems properly,
    which would lead to very slow performance.

    Installation:
    pip install cvxpy

    Reference:
    [1] https://www.cvxpy.org/api_reference/cvxpy.html
    """
    _dependencies = ('cvxpy',)

    _opt_ineq_to_1d = False
    _opt_eq_to_ineq = False

    def _create_problem(self):
        import cvxpy as cp
        y = cp.Variable(self.dof, name='y')
        constraints = []

        for i, (A, b, n) in enumerate(zip(self.As, self.bs, self.mat_sizes)):
            S = cp.Variable((n, n), symmetric=True, name=f'S_{i}')
            constraints.append(cp.vec(S) == A @ y + b.flatten())
            constraints.append(S >> 0)

        if self.ineq_lhs.shape[0] > 0:
            constraints.append(self.ineq_lhs @ y >= self.ineq_rhs)
        if self.eq_lhs.shape[0] > 0:
            constraints.append(self.eq_lhs @ y == self.eq_rhs)

        objective = cp.Minimize(self.c @ y)
        problem = cp.Problem(objective, constraints)
    
        return problem

    def _solve(self, configs: SolverConfigs):
        from cvxpy import settings as s
        problem = self._create_problem()
        obj = problem.solve(verbose=configs.verbose, **configs.solver_options)
        if problem.status in (s.OPTIMAL, s.OPTIMAL_INACCURATE):
            self.set_status(SDPStatus.OPTIMAL)
            for var in problem.variables():
                if var.name() == 'y':
                    value = var.value.flatten()
                    return value
        elif problem.status in (s.INFEASIBLE, s.INFEASIBLE_INACCURATE):
            self.set_status(SDPStatus.INFEASIBLE)
        elif problem.status in (s.UNBOUNDED, s.UNBOUNDED_INACCURATE):
            self.set_status(SDPStatus.UNBOUNDED)
        elif problem.status in (s.INFEASIBLE_OR_UNBOUNDED,):
            self.set_status(SDPStatus.INFEASIBLE_OR_UNBOUNDED)
        self.set_status(SDPStatus.ERROR)
        # raise NotImplementedError