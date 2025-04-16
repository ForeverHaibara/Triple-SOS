from numpy import array as np_array

from .backend import DualBackend
from .settings import SolverConfigs

# https://docs.mosek.com/latest/pythonfusion/parameters.html#doc-all-parameter-list
MOSEK_PARAMS = {
    'intpntMaxIterations': 'max_iters',
    'intpntCoTolRelGap': 'tol_gap_rel',
    'intpntCoTolDfeas': 'tol_fsb_abs',
    'intpntCoTolPfeas': 'tol_fsb_abs',
    'logIntpnt': 'verbose',
}

class mosek_problem:
    """Internal wrapper for mosek.fusion.Model which also holds the variable `y`."""
    M = None
    y = None
    def __init__(self, M, x):
        self.M = M
        self.y = y
    def solve(self):
        return self.M.solve()

class DualBackendMOSEK(DualBackend):
    """
    MOSEK backend for SDP problems.

    MOSEK solves SDP problems very efficiently, but it is a commercial software
    which requires a license. (But a free academic license is available.)

    Installation:
    pip install mosek

    Reference:
    [1] https://www.mosek.com/
    [2] https://docs.mosek.com/latest/pythonapi/intro_info.html
    [3] https://docs.mosek.com/latest/pythonfusion/tutorial-sdo-shared.html#doc-tutorial-sdo
    """
    _opt_isometric  = 'row'

    _dependencies = ('mosek',)
    @classmethod
    def is_available(cls) -> bool:
        try:
            from mosek.fusion import Model
            with Model("SDP") as M:
                _try_solve = M.solve() # checks if the license is available
        except:
            return False
        return True


    def _add_variables_to_model(self, M):
        from mosek.fusion import Domain, ObjectiveSense, Expr
        x = M.variable("y", self.dof, Domain.unbounded())
        for i, (A, b) in enumerate(zip(self.As, self.bs)):
            M.constraint("A%d"%i, Expr.add(Expr.mul(A, x), b), Domain.inSVecPSDCone(len(b)))

        if self.ineq_lhs.size > 0:
            M.constraint("geq", Expr.dot(ineq_lhs, x), Domain.GreaterThan(ineq_rhs))
        if self.eq_lhs.size > 0:
            M.constraint("eq", Expr.dot(eq_lhs, x), Domain.equalsTo(eq_rhs))

        M.objective(ObjectiveSense.Minimize, Expr.dot(self.c, x))
        return x

    def _create_problem(self, configs: SolverConfigs=None):
        """
        Create a problem instance of MOSEK. However, it is more recommended to use
        the "with" contextmanager to create a problem instance, which will do automatic
        rubbish collection.
        """
        if configs is None:
            configs = SolverConfigs()
        from mosek.fusion import Model, AccSolutionStatus
        M = Model("SDP")
        x = self._add_variables_to_model(M)
        M.acceptedSolutionStatus(AccSolutionStatus.Anything)
        for key, value in MOSEK_PARAMS.items():
            M.setSolverParam(key, getattr(configs, value))
        for key, value in configs.solver_options.items():
            M.setSolverParam(key, value)
        return mosek_problem(M, x)


    def _solve(self, configs: SolverConfigs):
        from mosek.fusion import Model, AccSolutionStatus
        from mosek.fusion import SolutionStatus, ProblemStatus
        result = {}
        with Model("SDP") as M:
            x = self._add_variables_to_model(M)
            M.acceptedSolutionStatus(AccSolutionStatus.Anything)
            for key, value in MOSEK_PARAMS.items():
                M.setSolverParam(key, getattr(configs, value))
            for key, value in configs.solver_options.items():
                M.setSolverParam(key, value)

            M.solve()
            # if M.getPrimalSolutionStatus() != SolutionStatus.Optimal:
            #     return
            y = x.level()
            if y is not None:
                y = np_array(y).flatten()
            result['y'] = y

            ss = M.getProblemStatus()
            ps = M.getPrimalSolutionStatus()
            if ps == SolutionStatus.Optimal:
                result['optimal'] = True
            elif ss == ProblemStatus.PrimalInfeasible:
                result['infeasible'] = True
            elif ss == ProblemStatus.DualInfeasible:
                result['unbounded'] = True
            elif ss == ProblemStatus.PrimalInfeasibleOrUnbounded:
                result['inf_or_unb'] = True
        return result