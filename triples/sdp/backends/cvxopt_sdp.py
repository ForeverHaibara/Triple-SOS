from numpy import array as np_array

from .backend import DualBackend
from .settings import SDPStatus, SolverConfigs, SDPError

class DualBackendCVXOPT(DualBackend):
    """
    CVXOPT backend for SDP problems.

    Installation:
    pip install cvxopt
    
    Reference:
    [1] https://cvxopt.org/userguide/coneprog.html#semidefinite-programming
    """
    _dependencies = ('cvxopt',)

    _opt_ineq_to_1d = False
    _opt_eq_to_ineq = False

    def _solve(self, configs: SolverConfigs):
        from cvxopt import matrix, solvers
        Gl = matrix(-self.ineq_lhs) if self.ineq_lhs.size > 0 else None
        hl = matrix(-self.ineq_rhs) if self.ineq_rhs.size > 0 else None
        Gs = [matrix(-A) for A in self.As]
        hs = [matrix(b.reshape(n, n)) for b, n in zip(self.bs, self.mat_sizes)]
        A = matrix(self.eq_lhs) if self.eq_lhs.size > 0 else None
        b = matrix(self.eq_rhs) if self.eq_rhs.size > 0 else None
        c = matrix(self.c)

        # configure kktsolver to handle ValueError: Rank(A) < p or Rank([P; A; G]) < n
        # https://ask.csdn.net/questions/1102440
        options = {
            'show_progress': bool(configs.verbose),
            'kktreg': 1e-9,
            # 'kktsolver': 'ldl',
            'maxiters': configs.max_iters,
            'abstol': configs.tol_gap_abs,
            'reltol': configs.tol_gap_rel,
            'feastol': configs.tol_fsb_abs,
        }
        solver_options = configs.solver_options.copy()
        options.update(solver_options.pop('options', {}))
        kktsolver = solver_options.pop('kktsolver', 'ldl')
        sol = solvers.sdp(c, Gl=Gl, hl=hl, Gs=Gs, hs=hs, A=A, b=b,
            kktsolver=kktsolver,
            options=options,
            **solver_options
        )
        status = sol['status']
        if status == 'optimal':
            self.set_status(SDPStatus.OPTIMAL)
            return np_array(sol['x']).flatten()
        elif status == 'primal infeasible':
            self.set_status(SDPStatus.INFEASIBLE)
        elif status in ('dual infeasible', 'unknown'):
            self.set_status(SDPStatus.INFEASIBLE_OR_UNBOUNDED)

        self.set_status(SDPStatus.ERROR)