from typing import Any, Tuple

import numpy as np

from .backend import DualBackend
from .settings import SolverConfigs

class DualBackendSDPAP(DualBackend):
    """
    SDPA (for Python) backend for SDP problems.

    SDPA solves CLP (Conic Linear Programming) problems of the form:
    min c^T x: x >=_K 0, Ax >=J b where K, J are cones.

    Installation:
    pip install sdpa-python
    
    Reference:
    [1] https://sdpa.sourceforge.net/index.html
    [2] https://sdpa-python.github.io/docs/installation/
    """
    _dependencies = ('sdpap',)

    _opt_ineq_to_1d = False
    _opt_eq_to_ineq = False

    def _get_ABCKJ(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any, Any]:
        """
        Get the input arguments for sdpap.solve(), which are A, b, c, K, J.
        """
        A = np.vstack([self.eq_lhs, self.ineq_lhs] + self.As)
        b = np.concatenate([self.eq_rhs, self.ineq_rhs] + [-_ for _ in self.bs]).reshape((-1, 1))
        c = self.c

        from sdpap import SymCone
        K = SymCone(f = self.dof)
        J = SymCone(f = self.eq_rhs.shape[0], l = self.ineq_rhs.shape[0], s = tuple(self.mat_sizes))
        return A, b, c, K, J

    def _solve(self, configs: SolverConfigs):
        import sdpap
        option = {
            'print': 'display' if configs.verbose else 'no',
            'maxIteration': configs.max_iters,
            'epsilonStar': configs.tol_gap_abs,
            'epsilonDash': configs.tol_gap_rel,
        }
        option.update(configs.solver_options)
        y, _, sdpapinfo, timeinfo, sdpainfo = sdpap.solve(*self._get_ABCKJ(), option=option)

        # y might be a scipy sparse matrix, so convert it to numpy array
        # check format of y and convert it to numpy array
        if (not isinstance(y, np.ndarray)) and hasattr(y, 'toarray'):
            y = y.toarray()
        y = np.array(y).flatten()

        result = {'y': y}
        status = sdpapinfo['phasevalue']
        if status == 'pdOPT':
            result['optimal'] = True
        elif status in ('pINF', 'pdINF','dUNBD'):
            result['infeasible'] = True
        elif status in ('pUNBD', 'pFEAS_dINF'):
            result['unbounded'] = True
        # elif status == 'dUNBD':
        #     result['inf_or_unb'] = True
        return result