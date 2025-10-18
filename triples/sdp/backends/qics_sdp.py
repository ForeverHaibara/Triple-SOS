import numpy as np

from .backend import DualBackend
from .settings import SolverConfigs

class DualBackendQICS(DualBackend):
    """
    QICS backend for SDP problems.

    QICS solves CLP (Conic Linear Programming) problems of the form:
    min c^Tx: b - Ax = 0, h - Gx in K where K is a cone.

    Installation:
    pip install qics

    Reference:
    [1] https://qics.readthedocs.io/en/latest/api/qics.html

    Adapted from contributor 数学规划-试验最优化.
    """
    _dependencies = ('qics',)
    
    _opt_ineq_to_1d = False
    _opt_eq_to_ineq = False

    def _create_model(self):
        c = self.c[:, None]
        A = self.eq_lhs
        b = self.eq_rhs[:, None]
        G = -np.concatenate([self.ineq_lhs] + self.As)
        h = np.concatenate([-self.ineq_rhs] + self.bs)[:, None]

        from qics.cones import NonNegOrthant, PosSemidefinite
        from qics import Model
        cones = []
        if len(self.ineq_lhs):
            cones.append(NonNegOrthant(len(self.ineq_lhs)))
        cones += [PosSemidefinite(x) for x in filter(lambda x: x > 0, self.mat_sizes)]
        return Model(c, A = A, b = b, G = G, h = h, cones = cones)

    def _create_problem(self, configs=None):
        if configs is None:
            configs = SolverConfigs()
        from qics import Solver
        return Solver(self._create_model(),
            max_iter = configs.max_iters,
            max_time = configs.time_limit,
            tol_gap = configs.tol_gap_rel,
            tol_feas = configs.tol_fsb_rel,
            verbose = int(configs.verbose))

    def _solve(self, configs: SolverConfigs) -> dict:
        problem = self._create_problem(configs)
        info = problem.solve()
        y = info['x_opt'].flatten()
        
        result = {'y': y}
        status = info['sol_status']
        status2 = info['exit_status']
        
        if status in ('optimal', 'near_optimal'):
            result['optimal'] = True
        elif status in ('pinfeas', 'near_pinfeas'):
            result['infeasible'] = True
        elif status in ('dinfeas', 'near_dinfeas'):
            if y is not None and self.is_feasible(y,
                    tol_fsb_abs=configs.tol_fsb_abs, tol_fsb_rel=configs.tol_fsb_rel):
                result['unbounded'] = True
            result['inf_or_unb'] = True

        if status2 in ('max_iter', 'max_time', 'slow_progress'):
            result['inaccurate'] = True
        if status2 in ('step_failure',):
            result['error'] = True
        return result