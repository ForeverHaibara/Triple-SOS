from numpy import array as np_array

from .backend import DualBackend
from .settings import SolverConfigs

class cvxopt_problem:
    """Internal class to store the problem data for CVXOPT backend
    that can be created by `._create_problem()` and has a `.solve()` method."""
    Gl = None
    hl = None
    Gs = None
    hs = None
    A = None
    b = None
    c = None
    options = None
    solver_options = None
    kktsolver = None
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    def solve(self):
        from cvxopt import solvers
        c, Gl, hl, Gs, hs, A, b = self.c, self.Gl, self.hl, self.Gs, self.hs, self.A, self.b
        options, solver_options, kktsolver = self.options, self.solver_options, self.kktsolver
        sol = solvers.sdp(c, Gl=Gl, hl=hl, Gs=Gs, hs=hs, A=A, b=b,
            kktsolver=kktsolver,
            options=options,
            **solver_options
        )
        return sol

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

    def _create_problem(self, configs: SolverConfigs=None):
        if configs is None:
            configs = SolverConfigs()
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

        return cvxopt_problem(Gl=Gl, hl=hl, Gs=Gs, hs=hs, A=A, b=b, c=c,
            options=options, solver_options=solver_options, kktsolver=kktsolver)

    def _solve(self, configs: SolverConfigs):
        problem = self._create_problem(configs)
        sol = problem.solve()

        y = sol.get('x', None)
        if y is not None:
            y = np_array(y).flatten()

        result = {'y': y}
        status = sol['status']

        if status == 'optimal':
            result['optimal'] = True
        elif status == 'primal infeasible':
            result['infeasible'] = True
        elif status == 'dual infeasible':
            if self.is_feasible(y, tol_fsb_abs=configs.tol_fsb_abs, tol_fsb_rel=configs.tol_fsb_rel):
                result['unbounded'] = True
            result['inf_or_unb'] = True
        return result
