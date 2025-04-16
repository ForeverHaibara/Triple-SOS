from typing import List, Dict, Any

import numpy as np
from numpy import ndarray

from .settings import SDPResult, SolverConfigs

class SDPBackend:
    _dependencies = tuple()

    status = None

    @classmethod
    def is_available(cls) -> bool:
        for dep in cls._dependencies:
            try:
                __import__(dep)
            except ImportError:
                return False
        return True

    @classmethod
    def as_array(cls, x):
        return np.array(x).astype(np.float64)

    def is_feasible(self, x, tol_fsb_abs: float = 1e-8, tol_fsb_rel: float = 1e-8) -> bool:
        """
        Check if the solution `x` is feasible. It will not be checked twice
        if it has been checked by the backend solver.
        """
        raise NotImplementedError

    def _create_problem(self):
        """
        Some solvers formulate a Problem object and call `solve` method to solve the SDP problem,
        e.g., cvxpy. This method is used to wrap the construction of the Problem object for
        convenience.
        """
        raise NotImplementedError

    def _solve(self, configs: SolverConfigs) -> Dict[str, Any]:
        """
        Solve the SDP problem given sanitized configurations, e.g., verbosity, tolerances,
        solver options. The solvers should implement handling of these configurations.
        The output must be a numpy array of shape `(dof,)`. If the solver occurs errors like
        infeasibility or unboundedness, the status of `self` should be set correspondingly.
        """
        raise NotImplementedError

    def solve(self,
            verbose: int = SolverConfigs.verbose,
            max_iters: int = SolverConfigs.max_iters,
            tol_gap_abs: float = SolverConfigs.tol_gap_abs,
            tol_gap_rel: float = SolverConfigs.tol_gap_rel,
            tol_fsb_abs: float = SolverConfigs.tol_fsb_abs,
            tol_fsb_rel: float = SolverConfigs.tol_fsb_rel,
            solver_options: dict = SolverConfigs.solver_options,
        ) -> SDPResult:
        """
        Solve the SDP problem. The output must be a numpy array of shape `(dof,)`. If the solver
        occurs errors like infeasibility or unboundedness, the status of `self` should be set
        correspondingly.
        """
        raise NotImplementedError

    @classmethod
    def _convert_space_to_isometric(cls, space: np.ndarray, order='row') -> np.ndarray:
        ndims = 2
        if len(space.shape) == 1:
            ndims = 1
            space = space.reshape(-1, 1)

        n = int(round(np.sqrt(space.shape[0])))

        # multiply 2**.5 on off-diagonal entries
        space = space * (2**.5)
        space[np.arange(0,n**2,n+1), :] *= 2**-.5

        # extract the upper triangular part
        if order == 'row': # row major
            rows = np.array([i*n+j for i in range(n) for j in range(n) if i <= j])
        elif order == 'col': # column major
            rows = np.array([i*n+j for j in range(n) for i in range(n) if i <= j])
        else:
            raise ValueError('Order should be one of "row" or "col".')
        upper = space[rows, :]

        if ndims == 1: # flatten to vector
            upper = upper.flatten()
        return upper


class DualBackend(SDPBackend):
    """
    Configuration Flags (class variables):
    _opt_isometric        : If 'row' or 'col', store symmetric matrices in isometric form in given order.
    _opt_sparse           : If 'csc', 'csr' or 'coo', convert 2d matrices to scipy sparse format.
    _opt_ineq_to_1d       : If True, convert inequality constraints to 1D matrices.
    _opt_eq_to_ineq       : If True, convert equality constraints to two inequality constraints.
    """
    _opt_isometric  = False
    _opt_sparse     = False
    _opt_ineq_to_1d = True
    _opt_eq_to_ineq = True


    def __init__(self, As: List[ndarray], bs: List[ndarray], ineq_lhs: ndarray, ineq_rhs: ndarray,
                    eq_lhs: ndarray, eq_rhs: ndarray, c: ndarray):
        dof = c.shape[0]

        # store a copy of the original inputs
        self._As = As
        self._bs = bs
        self._ineq_lhs = ineq_lhs
        self._ineq_rhs = ineq_rhs
        self._eq_lhs   = eq_lhs
        self._eq_rhs   = eq_rhs

        if self._opt_eq_to_ineq:
            ineq_lhs = np.vstack((ineq_lhs, eq_lhs, -eq_lhs))
            ineq_rhs = np.concatenate((ineq_rhs, eq_rhs, -eq_rhs))
            eq_lhs, eq_rhs = np.zeros((0, dof)), np.zeros((0,))

        if self._opt_ineq_to_1d:
            As.extend([ineq_lhs[i:i+1,:] for i in range(ineq_lhs.shape[0])])
            ineq_rhs = -ineq_rhs
            bs.extend([ineq_rhs[i:i+1] for i in range(ineq_rhs.shape[0])])
            ineq_lhs, ineq_rhs = np.zeros((0, dof)), np.zeros((0,))

        if self._opt_isometric:
            As = [self._convert_space_to_isometric(A, order=self._opt_isometric) for A in As]
            bs = [self._convert_space_to_isometric(b, order=self._opt_isometric) for b in bs]

        if self._opt_sparse:
            if not self._opt_sparse in ('csc', 'csr', 'coo'):
                raise ValueError('DualBackend._opt_sparse must be one of "csc" or "csr" or "coo".')
            from scipy import sparse
            to_mat = getattr(sparse, self._opt_sparse + '_matrix')
            As = [to_mat(A) for A in As]
            ineq_lhs = to_mat(ineq_lhs)
            eq_lhs = to_mat(eq_lhs)

        self.As = As
        self.bs = bs
        self.ineq_lhs = ineq_lhs
        self.ineq_rhs = ineq_rhs
        self.eq_lhs   = eq_lhs
        self.eq_rhs   = eq_rhs
        self.c = c

    @property
    def dof(self) -> int:
        return self.c.shape[0]

    @property
    def mat_sizes(self) -> List[int]:
        shapes = np.array([A.shape[0] for A in self._As])
        # if not self._opt_isometric:
        return np.round(np.sqrt(shapes)).astype(int).tolist()
        # else:
        #     return np.round(np.sqrt(shapes*2+.25)).astype(int).tolist()

    def is_feasible(self, x, tol_fsb_abs: float = 1e-8, tol_fsb_rel: float = 1e-8) -> bool:
        if self._ineq_lhs.shape[0] > 0:
            v = self._ineq_lhs @ x - self._ineq_rhs
            minv = np.min(v)
            if minv < -tol_fsb_abs or minv < -tol_fsb_rel * max(1, np.max(np.abs(v))):
                return False
        if self._eq_lhs.shape[0] > 0:
            v = np.abs(self._eq_lhs @ x - self._eq_rhs)
            maxv = np.max(v)
            if maxv > tol_fsb_abs: # and
                return False
        for A, b, n in zip(self._As, self._bs, self.mat_sizes):
            if n <= 0:
                continue
            mat = A @ x + b
            mat = mat.reshape(n, n)
            eigs = np.linalg.eigvalsh(mat)
            min_eig, max_eig = np.min(eigs), np.max(eigs)
            if min_eig < -tol_fsb_abs and min_eig < -tol_fsb_rel * max(1, abs(max_eig)):
                return False
        return True

    def solve(self, **kwargs) -> SDPResult:
        configs = SolverConfigs(**kwargs)

        if self.dof == 0:
            # Most solvers would not allow a variable with shape (0,),
            # so we shall handle it separately.
            x = np.zeros((self.dof,), dtype=np.float64)
            if self.is_feasible(x, tol_fsb_abs=configs.tol_fsb_abs, tol_fsb_rel=configs.tol_fsb_rel):
                return SDPResult(y=x, optimal=True)
            return SDPResult(infeasible=True)
        return SDPResult(**self._solve(configs))


    @classmethod
    def from_sdpa_file(cls, filename: str, *args, **kwargs) -> 'DualBackend':
        with open(filename, 'r', *args, **kwargs) as f:
            s = f.read()
        return cls.from_sdpa_string(s)

    @classmethod
    def from_sdpa_string(cls, s: str) -> 'DualBackend':
        def read_sdpa(s):
            lines = s.strip().split('\n')
            for i in range(len(lines)):
                if not (lines[i].startswith('"') or lines[i].startswith("*")):
                    # discard lines of comments
                    break
            lines = lines[i:]                
            m = int(lines[0].split()[0])
            nBlock = int(lines[1].split()[0])
            block_sizes = list(map(abs, map(int, lines[2].strip().split())))
            c_vector = list(map(float, lines[3].strip().split()))

            entries = []
            for line in lines[4:]:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                k, i, r, c = map(int, parts[:4])
                val = float(parts[4])
                entries.append((k, i, r, c, val))

            matrices = [np.zeros((blk_size**2, m+1)) for blk_size in block_sizes]

            for (k, i, r, c, val) in entries:
                blk_size = block_sizes[k-1]
                matrices[i-1][blk_size*(r-1) + c-1, k] = val
                matrices[i-1][blk_size*(c-1) + r-1, k] = val
            bs = [-_[:,0] for _ in matrices]
            As = [_[:,1:] for _ in matrices]
            return As, bs, np.array(c_vector)

        As, bs, c = read_sdpa(s)
        return cls(As, bs,
                    np.zeros((0, c.shape[0])), np.zeros((0,)),
                    np.zeros((0, c.shape[0])), np.zeros((0,)), c)