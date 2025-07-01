import numpy as np

from .backend import DualBackend
from .settings import SolverConfigs

class DualBackendCLARABEL(DualBackend):
    """
    Clarabel backend for SDP problems.

    Clarabel solves CLP (Conic Linear Programming) problems of the form:
    min x^TPx/2 + q^Tx: Ax + s = b, s in K where K is a cone.

    Installation:
    pip install clarabel

    Reference:
    [1] https://clarabel.org/stable/python/getting_started_py/
    """
    _dependencies = ('clarabel', 'scipy')

    _opt_isometric  = 'col'
    _opt_sparse     = 'csc'
    _opt_ineq_to_1d = False
    _opt_eq_to_ineq = False

    def _get_PqAbcones(self):
        from scipy import sparse
        from clarabel import PSDTriangleConeT, ZeroConeT, NonnegativeConeT
        P = sparse.csc_matrix((self.dof, self.dof)) # zero matrix
        q = self.c
        A = sparse.vstack([-A for A in self.As] + [-self.ineq_lhs] + [self.eq_lhs], format='csc')
        b = np.concatenate(self.bs + [-self.ineq_rhs] + [self.eq_rhs])
        cones = []

        for n in self.mat_sizes:
            cones.append(PSDTriangleConeT(n))
        if self.ineq_lhs.size > 0:
            cones.append(NonnegativeConeT(self.ineq_lhs.shape[0]))
        if self.eq_lhs.size > 0:
            cones.append(ZeroConeT(self.eq_lhs.shape[0]))
        return P, q, A, b, cones

    def _create_problem(self, configs: SolverConfigs=None):
        if configs is None:
            configs = SolverConfigs()
        from clarabel import DefaultSolver, DefaultSettings
        P, q, A, b, cones = self._get_PqAbcones()
        settings = DefaultSettings()
        settings.verbose = False
        if configs is not None:
            settings.verbose = bool(configs.verbose)
            settings.max_iter = configs.max_iters
            settings.tol_gap_abs = configs.tol_gap_abs
            settings.tol_gap_rel = configs.tol_gap_rel
            settings.tol_feas = configs.tol_fsb_abs
            for key, value in configs.solver_options.items():
                setattr(settings, key, value)
        solver = DefaultSolver(P, q, A, b, cones, settings)
        return solver

    def _solve(self, configs: SolverConfigs):
        from clarabel import SolverStatus as s
        solver = self._create_problem(configs)
        sol = solver.solve()

        status = sol.status
        y = np.array(sol.x).flatten() if sol.x is not None else None
        result = {'y': y}

        if status in (s.Solved, s.AlmostSolved):
            result['optimal'] = True
        elif status in (s.PrimalInfeasible, s.AlmostPrimalInfeasible):
            result['infeasible'] = True
        elif status in (s.DualInfeasible, s.AlmostDualInfeasible):
            if y is not None and self.is_feasible(y,
                    tol_fsb_abs=configs.tol_fsb_abs, tol_fsb_rel=configs.tol_fsb_rel):
                result['unbounded'] = True
            result['inf_or_unb'] = True

        if status in (s.MaxIterations, s.MaxTime, s.InsufficientProgress):
            result['inaccurate'] = True

        return result