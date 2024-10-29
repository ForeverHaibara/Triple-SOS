import numpy as np

from .backend import DualBackend, PrimalBackend

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
        A = sparse.vstack(self._As + self._eqs + self._leqs, format='csc')
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


class PrimalBackendCLARABEL(PrimalBackend):
    def __init__(self, x0) -> None:
        super().__init__(x0)

        self._As = []
        self._eqs = []
        self._eq_bs = []
        self._leqs = []
        self._leq_bs = []

    @classmethod
    def _convert_space_to_isometric(cls, space: np.ndarray) -> np.ndarray:
        n = int(round(np.sqrt(space.shape[1])))

        # multiply 2**.5 on off-diagonal entries
        space = space * (2**.5)
        space[:, np.arange(0,n**2,n+1)] *= 2**-.5

        # extract the upper triangular part (note: column-major order)
        rows = np.array([i*n+j for j in range(n) for i in range(n) if i <= j])
        upper = space[:, rows]
        return upper

    @classmethod
    def _convert_vec_to_matrix(cls, vec: np.ndarray, mul=2**.5) -> np.ndarray:
        n = int(round((np.sqrt(1 + 8*vec.size) - 1)/2))
        rows = np.array([i*n+j for j in range(n) for i in range(n) if i <= j])

        vec = vec / (mul/2)
        mat = np.zeros(n**2)
        mat[rows] = vec
        mat[np.arange(0,n**2,n+1)] *= mul/2
        mat = mat.reshape((n,n))
        mat = (mat + mat.T) * .5
        return mat

    def _add_linear_matrix_equality(self, space: np.ndarray) -> None:
        self._spaces.append(space)
        
        from scipy import sparse
        space = self._convert_space_to_isometric(space)
        space = sparse.csc_matrix(space)

        self._As.append(space)
        return self._As[-1]

    def _add_constraint(self, constraint: np.ndarray, rhs: float = 0, operator = '__eq__') -> None:
        if operator == '__eq__':
            self._eqs.append(constraint)
            self._eq_bs.append(rhs)
        elif operator == '__le__':
            self._leqs.append(constraint)
            self._leq_bs.append(rhs)
        elif operator == '__ge__':
            self._leqs.append(-constraint)
            self._leq_bs.append(-rhs)

    def _get_PqAbcones(self):
        from scipy import sparse
        from clarabel import PSDTriangleConeT, ZeroConeT, NonnegativeConeT

        q = self.split_vector(self._objective)
        q = np.concatenate([self._convert_space_to_isometric(obj.reshape((1,-1))).flatten() for obj in q])

        P = sparse.csc_matrix((q.size, q.size))

        psd = [-sparse.eye(m*(m+1)//2, m*(m+1)//2, format='csc') for m in self._mat_size] + [sparse.csc_matrix((1, 1))]
        psd = sparse.block_diag(psd, format='csc')

        A = sparse.hstack(self._As + [sparse.csc_matrix(self._min_eigen_space.reshape((-1,1)))], format='csc')
        A = sparse.vstack([psd, A], format='csc')

        b = np.concatenate([np.zeros(psd.shape[0]), self.x0])

        cones = [PSDTriangleConeT(m) for m in self._mat_size]
        cones.append(ZeroConeT(1 + self.x0.size))

        dof = self.dof + 1
        if len(self._eqs):
            eqs = np.vstack([_.reshape((-1, dof)) for _ in self._eqs])
            eqs = self.split_vector(eqs)
            eqs = [self._convert_space_to_isometric(eq) for eq in eqs]
            eqs = sparse.csc_matrix(np.hstack(eqs))
            eq_b = np.concatenate([np.array(_).flatten() for _ in self._eq_bs])
            A = sparse.vstack([A, eqs], format='csc')
            b = np.concatenate([b, eq_b])

            # cones.append(ZeroConeT(eq_b.shape[0]))
            cones.pop()
            cones.append(ZeroConeT(1 + self.x0.size + eq_b.shape[0]))

        if len(self._leqs):
            leqs = np.vstack([_.reshape((-1, dof)) for _ in self._leqs])
            leqs = self.split_vector(leqs)
            leqs = [self._convert_space_to_isometric(leq) for leq in leqs]
            leqs = sparse.csc_matrix(np.hstack(leqs))
            leq_b = np.concatenate([np.array(_).flatten() for _ in self._leq_bs])
            A = sparse.vstack([A, leqs], format='csc')
            b = np.concatenate([b, leq_b])
            cones.append(NonnegativeConeT(leq_b.shape[0]))

        return P, q, A, b, cones

    def _get_solver(self, solver_options={}):
        from clarabel import DefaultSolver, DefaultSettings
        settings = DefaultSettings()
        settings.verbose = False
        for key, value in solver_options.items():
            setattr(settings, key, value)
        P, q, A, b, cones = self._get_PqAbcones()
        solver = DefaultSolver(P, q, A, b, cones, settings)

        return solver

    def solve(self, solver_options={}):
        solver = self._get_solver(solver_options)
        self.solution = solver.solve()
        solution = np.array(self.solution.x)
        solution = np.split(solution, np.cumsum([m*(m+1)//2 for m in self._mat_size] + [1]), axis=-1)[:-1]
        solution = [self._convert_vec_to_matrix(sol).flatten() for sol in solution]
        return self.restore_eigen(solution)