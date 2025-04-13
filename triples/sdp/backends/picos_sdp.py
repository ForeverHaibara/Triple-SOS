import numpy as np

from .backend import DualBackend
from .settings import SDPStatus, SolverConfigs

class DualBackendPICOS(DualBackend):
    """
    PICOS backend for SDP problems.
    Picos is a Python interface to conic optimization solvers,
    and the default solver is CVXOPT.

    Installation:
    pip install picos

    Reference:
    [1] https://picos-api.gitlab.io/picos/index.html
    """
    _dependencies = ('picos',)

    _opt_ineq_to_1d = False
    _opt_eq_to_ineq = False

    def _create_problem(self, configs: SolverConfigs=None):
        from picos import Problem, RealVariable, SymmetricVariable
        problem = Problem()
        y = RealVariable('y', shape = self.dof)
        for i, (A, b, n) in enumerate(zip(self.As, self.bs, self.mat_sizes)):
            S = SymmetricVariable(f'S{i}', shape = (n, n))
            problem.add_constraint(S.vec == A * y + b)
            problem.add_constraint(S >> 0)

        if self.ineq_lhs.shape[0] > 0:
            problem.add_constraint(self.ineq_lhs * y >= self.ineq_rhs)
        if self.eq_lhs.shape[0] > 0:
            problem.add_constraint(self.eq_lhs * y == self.eq_rhs)

        problem.set_objective('min', self.c * y)

        if configs is not None:
            problem.options['verbosity'] = configs.verbose
            problem.options['max_iterations'] = configs.max_iters
            problem.options['abs_dual_fsb_tol'] = configs.tol_fsb_abs
            problem.options['rel_dual_fsb_tol'] = configs.tol_fsb_rel
            problem.options['abs_prim_fsb_tol'] = configs.tol_fsb_abs
            problem.options['rel_prim_fsb_tol'] = configs.tol_fsb_rel
            problem.options['abs_ipm_opt_tol'] = configs.tol_gap_abs
            problem.options['rel_ipm_opt_tol'] = configs.tol_gap_rel
        return problem

    def _solve(self, configs: SolverConfigs):
        from picos.modeling import Strategy
        from picos.modeling.solution import (
            SS_OPTIMAL, SS_INFEASIBLE, SS_UNKNOWN, SS_EMPTY,
            PS_INFEASIBLE, PS_UNBOUNDED, PS_UNKNOWN, PS_INF_OR_UNB
        )
        problem = self._create_problem(configs)
        strategy = Strategy.from_problem(problem, primals=True, **configs.solver_options)
        solution = strategy.execute(primals=True, **configs.solver_options)
        if solution.primalStatus == SS_OPTIMAL:
            solution.apply(snapshotStatus=True)
            value = problem.variables['y'].value
            if isinstance(value, (float, int)):
                value = [value]
            value = np.array(value).flatten()
            return value

        elif solution.primalStatus in (SS_INFEASIBLE, PS_INFEASIBLE):
            self.set_status(SDPStatus.INFEASIBLE)
        elif solution.primalStatus in (PS_UNBOUNDED,):
            self.set_status(SDPStatus.UNBOUNDED)
        elif solution.primalStatus in (SS_UNKNOWN, PS_UNKNOWN, SS_EMPTY, PS_INF_OR_UNB):
            self.set_status(SDPStatus.INFEASIBLE_OR_UNBOUNDED)
        self.set_status(SDPStatus.ERROR)