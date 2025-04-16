import numpy as np

from .backend import DualBackend
from .settings import SolverConfigs

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
            SS_OPTIMAL, SS_INFEASIBLE, SS_PREMATURE, SS_FAILURE,
            PS_INFEASIBLE, PS_UNBOUNDED, PS_INF_OR_UNB
        )
        problem = self._create_problem(configs)
        strategy = Strategy.from_problem(problem, primals=True, **configs.solver_options)
        solution = strategy.execute(primals=True, **configs.solver_options)
        solution.apply(snapshotStatus=True)
        y = problem.variables['y'].value
        if y is not None:
            if isinstance(y, (float, int)):
                y = [y]
            y = np.array(y, dtype=float).flatten()

        result = {'y': y}

        # PICOS have multiple status APIs, and I am not sure which to use.
        # Note that some unbounded problems will be claimed wrongly as primal infeasible
        # by ".status" or ".lastStatus" properties.
        ps = solution.problemStatus
        ss = solution.primalStatus
        if ss == SS_OPTIMAL:
            result['optimal'] = True
        elif ss == SS_INFEASIBLE or ps == PS_INFEASIBLE:
            result['infeasible'] = True
        elif ps == PS_UNBOUNDED:
            result['unbounded'] = True
        elif ss == PS_INF_OR_UNB or ps == PS_INF_OR_UNB:
            result['inf_or_unb'] = True
        elif ss == SS_PREMATURE or ps == SS_PREMATURE:
            result['inaccurate'] = True
        elif ss == SS_FAILURE:
            result['error'] = True
        return result