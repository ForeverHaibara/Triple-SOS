import numpy as np

from .backend import DualBackend

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

    def __init__(self, dof) -> None:
        super().__init__(dof)
        self._original_spaces = []
        self._original_bs = []
        self._As = []
        self._bs = []
        self._eqs = [] # TODO: implement linear constraints without converting to 1d SDP matrices
        self._eq_bs = []
        self._leqs = []
        self._leq_bs = []
        self._q = None
        # self._cones = []
        self.solution = None

    @classmethod
    def _convert_space_to_isometric(cls, space: np.ndarray) -> np.ndarray:
        n = int(round(np.sqrt(space.shape[0])))

        # multiply 2**.5 on off-diagonal entries
        space = space * (2**.5)
        space[np.arange(0,n**2,n+1), :] *= 2**-.5

        # extract the upper triangular part (note: column-major order)
        rows = np.array([i*n+j for j in range(n) for i in range(n) if i <= j])
        upper = space[rows, :]
        return upper

    def _add_linear_matrix_inequality(self, x0: np.ndarray, extended_space: np.ndarray) -> np.ndarray:
        self._original_spaces.append(extended_space)
        self._original_bs.append(x0)
        
        from scipy import sparse
        space = self._convert_space_to_isometric(-extended_space)
        space = sparse.csc_matrix(space)
        x0 = self._convert_space_to_isometric(x0.reshape((-1, 1))).flatten()

        self._As.append(space)
        self._bs.append(x0)
        return self._As[-1]

    def _add_constraint(self, constraint: np.ndarray, rhs: float = 0, operator='__ge__') -> None:
        return super()._add_constraint(constraint, rhs, operator)

    def _set_objective(self, objective: np.ndarray) -> None:
        self._q = objective

    def _get_PqAbcones(self):
        from scipy import sparse
        from clarabel import PSDTriangleConeT, ZeroConeT, NonnegativeConeT
        P = sparse.csc_matrix((self.dof + 1, self.dof + 1))
        q = self._q
        A = sparse.vstack(self._As + self._eqs + self._leqs).tocsc()
        b = np.concatenate(self._bs + self._eq_bs + self._leq_bs)
        cones = []

        for space in self._original_spaces:
            n = int(round(np.sqrt(space.shape[0])))
            cones.append(PSDTriangleConeT(n))
        if len(self._eqs) > 0:
            cones.append(ZeroConeT(len(self._eqs)))
        if len(self._leqs) > 0:
            cones.append(NonnegativeConeT(len(self._leqs)))
        return P, q, A, b, cones

    def _get_solver(self, solver_options={}):
        from clarabel import DefaultSolver, DefaultSettings
        settings = DefaultSettings()
        settings.verbose = False # set default: False
        for key, value in solver_options.items():
            setattr(settings, key, value)
        P, q, A, b, cones = self._get_PqAbcones()
        solver = DefaultSolver(P, q, A, b, cones, settings)

        return solver

    def solve(self, solver_options={}):
        solver = self._get_solver(solver_options)
        self.solution = solver.solve()
        return np.array(self.solution.x[:-1])