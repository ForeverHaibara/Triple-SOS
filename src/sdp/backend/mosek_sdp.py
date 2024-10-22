import numpy as np

from .backend import DualBackend

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
    """
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
        self.solution = None

    @classmethod
    def _convert_space_to_isometric(cls, space: np.ndarray) -> np.ndarray:
        n = int(round(np.sqrt(space.shape[0])))

        # multiply 2**.5 on off-diagonal entries
        space = space * (2**.5)
        space[np.arange(0,n**2,n+1), :] *= 2**-.5

        # extract the upper triangular part (note: row-major order)
        rows = np.array([i*n+j for i in range(n) for j in range(n) if i <= j])
        upper = space[rows, :]
        return upper

    def _add_linear_matrix_inequality(self, x0: np.ndarray, extended_space: np.ndarray) -> np.ndarray:
        self._original_spaces.append(extended_space)
        self._original_bs.append(x0)
        
        from mosek.fusion import Matrix
        space = self._convert_space_to_isometric(extended_space)
        space = Matrix.dense(space)
        x0 = (self._convert_space_to_isometric(x0.reshape((-1, 1))).flatten().tolist())

        self._As.append(space)
        self._bs.append(x0)
        return self._As[-1]

    def _add_constraint(self, constraint: np.ndarray, rhs: float = 0, operator='__ge__') -> None:
        return super()._add_constraint(constraint, rhs, operator)

    def _set_objective(self, objective: np.ndarray) -> None:
        self._q = objective.tolist()

    def _add_variables_to_model(self, M):
        from mosek.fusion import Model, Domain, ObjectiveSense, Expr, Variable
        x = M.variable("x", self.dof + 1, Domain.unbounded())
        for i in range(len(self._As)):
            A, b = self._As[i], self._bs[i]
            M.constraint("A%d"%i, Expr.add(Expr.mul(A, x), b), Domain.inSVecPSDCone(len(b)))

        for i in range(len(self._eqs)):
            A, b = self._eqs[i], self._eq_bs[i]
            M.constraint("eq%d"%i, Expr.dot(A, x), Domain.equalsTo(b))

        for i in range(len(self._leqs)):
            A, b = self._leqs[i], self._leq_bs[i]
            M.constraint("leq%d"%i, Expr.dot(A, x), Domain.lessThan(b))
        M.objective(ObjectiveSense.Minimize, Expr.dot(self._q, x))
        return x

    def solve(self, solve_options: dict = {}) -> np.ndarray:
        from mosek.fusion import Model, Domain, ObjectiveSense, Expr, Variable, Model
        with Model("SDP") as M:
            self.model = M
            x = self._add_variables_to_model(M)
            M.solve()
            self.solution = x.level()
        return np.array(self.solution).flatten()[:-1]