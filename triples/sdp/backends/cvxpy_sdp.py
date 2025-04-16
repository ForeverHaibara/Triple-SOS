from copy import deepcopy

from .backend import DualBackend
from .settings import SolverConfigs, SDPError

_CVXPY_SOLVER_CONFIGS = {
    'CLARABEL': {
        'max_iter': 'max_iters',
        'tol_gap_abs': 'tol_gap_abs',
        'tol_gap_rel': 'tol_gap_rel',
        'tol_feas': 'tol_fsb_abs',
    },
    'COPT': {
        'RelGap': 'tol_gap_rel',
        'AbsGap': 'tol_gap_abs',
        'FeasTol': 'tol_fsb_abs',
        'BarIterLimit': 'max_iters',
    },
    'MOSEK': {
        'mosek_params': {
            'MSK_IPAR_INTPNT_MAX_ITERATIONS': 'max_iters',
            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 'tol_gap_rel',
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 'tol_fsb_abs',
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 'tol_fsb_abs',
        }
    },
    'CVXOPT': {
        'max_iters': 'max_iters',
        'abstol': 'tol_gap_abs',
        'reltol': 'tol_gap_rel',
        'feastol': 'tol_fsb_abs',
        'kktsolver': 'robust'
    },
    'SDPA': {
        'maxIteration': 'max_iters',
        'epsilonStar': 'tol_gap_abs',
        'epsilonDash': 'tol_gap_rel',
    },
    'SCS': {
        # 'max_iters': 'max_iters', # SCS uses a different algorithm requiring more iters
        'eps': 'tol_gap_abs',
    }
}

def update_solver_options(problem, solver_options, configs: SolverConfigs):
    solver_options = deepcopy(solver_options)

    solver = solver_options.get('solver')
    if solver is None:
        cd_solvers = problem._find_candidate_solvers()
        problem._sort_candidate_solvers(cd_solvers)
        if len(cd_solvers['conic_solvers']):
            for _solver in cd_solvers['conic_solvers']:
                if _solver.upper() in _CVXPY_SOLVER_CONFIGS:
                    solver = _solver.upper()
                    break
    if solver is None:
        return solver_options
        # raise SDPError('No cvxpy conic solver found.')
    solver = solver.upper()

    solver_options['solver'] = solver
    def _form_params(dt, options, configs):
        for k, v in options.items():
            if isinstance(v, dict):
                dt[k] = _form_params(dt.get(k, {}), v, configs)
            elif k in dt:
                continue
            else:
                if v in configs._KEYS:
                    dt[k] = getattr(configs, v)
                else:
                    dt[k] = v
        return dt
    solver_options = _form_params(solver_options,
                                  _CVXPY_SOLVER_CONFIGS.get(solver, {}), configs)
    # print(solver_options)
    return solver_options


class DualBackendCVXPY(DualBackend):
    """
    CVXPY backend for SDP problems.
    CVXPY is a Python-embedded modeling language for convex optimization problems.

    Warnings:
    1. CVXPY reformulation of the SDP might be more difficult to solve and would
    be slower than calling the solver-specific backends.
    2. Low versions of CVXPY may use the SCS solver by default, which implements
    a different algorithm that would fail for some of the tests.

    Installation:
    pip install cvxpy

    Reference:
    [1] https://www.cvxpy.org/api_reference/cvxpy.html

    [2] https://www.cvxpy.org/tutorial/solvers/index.html
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
            constraints.append(cp.vec(S, order='C') == A @ y + b.flatten())
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
        solver_options = update_solver_options(problem, configs.solver_options, configs)
        obj = problem.solve(verbose=bool(configs.verbose), **solver_options)

        for var in problem.variables():
            if var.name() == 'y':
                y = var.value
        if y is not None:
            y = y.flatten()
        result = {'y': y}

        if problem.status in (s.OPTIMAL, s.OPTIMAL_INACCURATE):
            result['optimal'] = True
        elif problem.status in (s.INFEASIBLE, s.INFEASIBLE_INACCURATE):
            result['infeasible'] = True
        elif problem.status in (s.UNBOUNDED, s.UNBOUNDED_INACCURATE):
            result['unbounded'] = True
        elif problem.status in (s.INFEASIBLE_OR_UNBOUNDED,):
            result['inf_or_unb'] = True
        if problem.status in s.INACCURATE:
            result['inaccurate'] = True
        if problem.status in s.ERROR:
            result['error'] = True
        return result