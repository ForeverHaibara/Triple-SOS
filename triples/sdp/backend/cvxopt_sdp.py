from typing import Any

import numpy as np

from .backend import DualBackend

def to_square(x):
    if not hasattr(x, 'size'): # float
        return x
    n = int(round(np.sqrt(x.size)))
    return x.reshape((n, n))

class DualBackendCVXOPT(DualBackend):
    """
    CVXOPT backend for SDP problems.

    Installation:
    pip install cvxopt
    
    Reference:
    [1] https://cvxopt.org/userguide/coneprog.html#semidefinite-programming
    """
    _dependencies = ('cvxopt',)
    def __init__(self, dof) -> None:
        super().__init__(dof)
        self._c = None  # Objective function
        self._Gs = []   # List of matrix inequality constraints (Gx <= h)
        self._hs = []   # List of right-hand side inequalities (h)
        # self._A = []  # Equality constraint matrix (if needed) # this is for primal, not dual
        # self._b = []  # Equality constraint rhs (if needed)
        self.solution = None

    def _add_linear_matrix_inequality(self, x0: np.ndarray, extended_space: np.ndarray) -> np.ndarray:
        from cvxopt import matrix
        self._Gs.append(matrix(-extended_space))
        self._hs.append(matrix(to_square(x0)))
        return self._Gs[-1]

    def _add_constraint(self, constraint: np.ndarray, rhs: float = 0, operator='__ge__') -> np.ndarray:
        return super()._add_constraint(constraint, rhs, operator)

    def _set_objective(self, objective: np.ndarray) -> None:
        from cvxopt import matrix
        self._c = matrix(objective)

    def solve(self, solver_options = {}) -> np.ndarray:
        from cvxopt import solvers

        solvers.options['show_progress'] = False

        # configure kktsolver to handle ValueError: Rank(A) < p or Rank([P; A; G]) < n
        # https://ask.csdn.net/questions/1102440
        solvers.options['kktreg'] = 1e-9
        solvers.options.update(solver_options)
        sol = solvers.sdp(self._c, Gs=self._Gs, hs=self._hs, kktsolver='ldl')
        self.y = sol['x']
        self.solution = sol

        return np.array(sol['x']).flatten()[:-1]
