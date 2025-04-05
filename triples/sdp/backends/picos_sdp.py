from .backend import DualBackend

import numpy as np

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

    def _create_problem(self):
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
        return problem

    def _solve(self):
        problem = self._create_problem()
        problem.solve(verbosity = 0)
        value = problem.variables['y'].value
        if isinstance(value, (float, int)):
            value = [value]
        value = np.array(value).flatten()
        return value