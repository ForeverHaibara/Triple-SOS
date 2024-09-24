from typing import Any, Tuple

import numpy as np

from .backend import SDPBackend

class SDPBackendSDPAP(SDPBackend):
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
    def __init__(self, dof) -> None:
        super().__init__(dof)
        self._As = []
        self._bs = []
        self._c = None
        self._J = []
        self.solution = None

    def _add_vector_variable(self, name: str, shape: int) -> Any:
        ...

    def _add_linear_matrix_inequality(self, name: str, x0: np.ndarray, extended_space: np.ndarray) -> np.ndarray:
        self._As.append(extended_space)
        self._bs.append(-x0)
        self._J.append(int(round(np.sqrt(extended_space.shape[0]))))
        return self._As[-1]

    def _add_constraint(self, constraint: np.ndarray, rhs: float = 0, operator='__ge__') -> None:
        return super()._add_constraint(constraint, rhs, operator)

    def _set_objective(self, objective: np.ndarray) -> None:
        self._c = objective

    @classmethod
    def is_available(cls) -> bool:
        try:
            import sdpap
            return True
        except ImportError:
            return False

    def _get_ABCKJ(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any, Any]:
        """
        Get the input arguments for sdpap.solve(), which are A, b, c, K, J.

        Since SDPAP is sensitive to ill-conditioned cases, we shall trim vars with zero coefficients.
        """
        A = np.vstack(self._As)
        b = np.concatenate(self._bs).reshape((-1, 1))
        c = self._c
        # trim vars with zero coefficients
        mask = np.any(A != 0, axis=0)
        A = A[:, mask]
        c = c[mask]

        from sdpap import SymCone
        K = SymCone(f = int(mask.sum()))
        J = SymCone(s = tuple(self._J))
        return A, b, c, K, J

    def _retrieve_y(self, y: np.ndarray) -> np.ndarray:
        """
        Since zero columns are trimmed, we need to recover the original y.
        """
        A = np.vstack(self._As)
        mask = np.any(A != 0, axis=0)
        y2 = []
        j = 0
        for m in mask:
            if m:
                y2.append(y[j])
                j += 1
            else:
                y2.append(0)
        return np.array(y2)

    def solve(self) -> np.ndarray:
        import sdpap
        option = {'print': 'no'}
        sol = sdpap.solve(*self._get_ABCKJ(), option=option)
        self.solution = sol

        # sol[0] might be a scipy sparse matrix, so convert it to numpy array
        # check format of sol[0] and convert it to numpy array
        y = sol[0]
        if hasattr(sol[0], 'toarray'):
            y = sol[0].toarray()
        y = np.array(y).flatten()
        y = self._retrieve_y(y)[:-1]
        self.y = y

        return y
    
