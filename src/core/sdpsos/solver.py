from typing import List, Optional, Tuple, Callable, Dict, Any, Union
from contextlib import contextmanager, nullcontext, AbstractContextManager
from copy import deepcopy

import numpy as np
import sympy as sp

from .rationalize import rationalize, rationalize_and_decompose
from .ipm import (
    SDPConvergenceError, SDPNumericalError, SDPInfeasibleError, SDPRationalizeError
)
from .utils import (
    symmetric_matrix_from_upper_vec, upper_vec_of_symmetric_matrix,
    split_vector, solve_undetermined_linear, S_from_y
)


def _check_picos(verbose = False):
    """
    Check whether PICOS is installed.
    """
    try:
        import picos
    except ImportError:
        if verbose:
            print('Cannot import picos, please use command "pip install picos" to install it.')
        return False
    return True


class SDPResult():
    def __init__(self, sos, solution: Dict = None):
        self.sos = sos
        self.y = None
        self.S = None
        self.decompositions = None
        self.success = False
        if solution is not None:
            self.y = solution['y']
            self.S = solution['S']
            self.decompositions = solution['decompositions']
            self.success = True

    def __getitem__(self, key):
        return getattr(self, key)

    def as_dict(self):
        return {
            'sos': self.sos,
            'y': self.y,
            'S': self.S,
            'decompositions': self.decompositions,
            'success': self.success
        }


class SDPProblem():
    """
    Main class to solve rational SDP problems.
    """
    _has_picos = _check_picos()
    def __init__(self, x0, space, splits, keys = []):
        # unmasked x0, space, splits
        self._x0 = x0
        self._space = space
        self._splits = splits
        self.masked_rows = {}

        # masked x0, space, splits
        self.x0 = x0
        self.space = space
        self.splits = splits

        self.y = None
        self.S = None

        if keys is not None:
            if len(keys) != len(splits):
                raise ValueError("Length of keys and splits should be the same. But got %d and %d."%(len(keys), len(splits)))
            self.keys = keys
        else:
            self.keys = ['S_%d'%i for i in range(len(splits))]

        self.sos = self._construct_sos() if self._has_picos else None

        # record the numerical solutions
        self._ys = []

    @property
    def args(self):
        return self.x0, self.space, self.splits

    @property
    def dof(self):
        return self.space.shape[1]

    def _masked_dims(self, filter_zero = False):
        dims = {}
        for i in range(len(self.keys)):
            key = self.keys[i]
            split = self.splits[i]
            mask = self.masked_rows.get(key, [])
            k = round(np.sqrt(2 * (split.end - split.start) + .25) - .5)
            v = k - len(mask)
            if filter_zero and v == 0:
                continue
            dims[key] = v
        return dims

    def _not_none_keys(self):
        return list(self._masked_dims(filter_zero = True))


    def set_masked_rows(self, masks: Dict[str, List[int]] = {}) -> Dict[str, sp.Matrix]:
        """
        Sometimes the diagonal entries of S are zero. Or we set them to zero to
        reduce the degree of freedom. This function masks the corresponding rows.

        Parameters
        ----------
        masks : List[int]
            Indicates the indices of the rows to be masked.
        """
        # restore masked values to unmaksed values
        self.x0, self.space, self.splits = self._x0, self._space, self._splits
        self.masked_rows = {}

        if len(masks) == 0 or not any(_ for _ in masks.values()):
            return True

        # first compute y = x1 + space1 @ y1
        # => S = x0 + space @ x1 + space @ space1 @ y1
        perp_space = []
        tar_space = []
        lines = []

        for key, split in zip(self._not_none_keys(), self.splits):
            mask = masks.get(key, [])
            if not mask:
                continue
            n = self.Q[key].shape[1]
            for v, (i,j) in enumerate(upper_vec_of_symmetric_matrix(n, return_inds = True)):
                if i in mask or j in mask:
                    lines.append(v + split.start)

        tar_space = - self.x0[lines, :]
        perp_space = self.space[lines, :]

        # this might not have solution and raise an Error
        x1, space1 = solve_undetermined_linear(perp_space, tar_space)

        self.x0 += self.space @ x1
        self.space = self.space @ space1

        # remove masked rows
        not_lines = list(set(range(self.space.shape[0])) - set(lines))
        self.x0 = self.x0[not_lines, :]
        self.space = self.space[not_lines, :]
        self.masked_rows = deepcopy(masks)
        self.splits = split_vector(list(self._masked_dims().values()))

        return masks


    def pad_masked_rows(self, S: Union[Dict, sp.Matrix], key: str) -> sp.Matrix:
        """
        Pad the masked rows of S[key] with zeros.

        Returns
        ----------
        S : sp.Matrix
            The padded S.
        """
        if isinstance(S, dict):
            S = S[key]

        mask = self.masked_rows.get(key, [])
        if not mask:
            return S

        n = S.shape[0]
        m = n + len(mask)
        Z = sp.Matrix.zeros(m)
        # Z[:n, :n] = S
        not_masked = list(set(range(m)) - set(mask))

        for v1, r1 in enumerate(not_masked):
            for v2, r2 in enumerate(not_masked):
                Z[r1, r2] = S[v1, v2]
        return Z


    def S_from_y(self, y: Optional[sp.Matrix] = None) -> Dict[str, sp.Matrix]:
        """
        Given y, compute the symmetric matrices. This is useful when we want to see the
        symbolic representation of the SDP problem.

        This function does not register the result to self.S.
        """
        m = self.dof
        if y is None:
            y = sp.Matrix([sp.symbols('y_{%d}'%_) for _ in range(m)]).reshape(m, 1)
        elif not isinstance(y, sp.MatrixBase) or y.shape != (m, 1):
            raise ValueError('y must be a sympy Matrix of shape (%d, 1).'%m)

        Ss = S_from_y(y, *self.args)

        ret = {}
        for key, S in zip(self._not_none_keys(), Ss):
            ret[key] = S
        return ret



    def _construct_sos(self, reg = 0, constraints = None):        
        import picos

        # SDP should use numerical algorithm
        x0, space, splits = self.args
        x0_numer = np.array(x0).astype(np.float64).flatten()
        space_numer = np.array(space).astype(np.float64)

        sos = picos.Problem()
        y = picos.RealVariable('y', self.dof)
        for key, split in zip(self.keys, splits):
            x0_ = x0_numer[split]
            k = round(np.sqrt(2 * len(x0_) + .25) - .5)
            S = picos.SymmetricVariable(key, (k,k))
            sos.add_constraint(S >> reg)

            self._add_sdp_eq(sos, S, x0_, space_numer[split], y)

        for constraint in constraints or []:
            sos.add_constraint(constraint(sos))

        return sos, y

    def _add_sdp_eq(self, sos, S, x0, space, y):
        k = round(np.sqrt(2 * len(x0) + .25) - .5)
        x0_sym = symmetric_matrix_from_upper_vec(x0)
        space_sym = symmetric_matrix_from_upper_vec(space).reshape(k**2, -1)
        sos.add_constraint(S.vec == x0_sym + space_sym * y)


    def _nsolve_with_early_stop(
            self,
            max_iters: int = 50,
            min_iters: int = 10,
            verbose: bool = False
        ) -> Any:
        """
        Numerically solve the sdp with PICOS.

        Python package PICOS solving SDP problem with CVXOPT will oftentimes
        faces ZeroDivisionError. This is due to the iterations is large while
        working precision is not enough.

        This function is a workaround to solve this. It flexibly reduces the
        number of iterations and tries to solve the problem again until
        the problem is solved or the number of iterations is less than min_iters.

        Parameters
        ----------
        max_iters : int
            Maximum number of iterations. It cuts down to half if ZeroDivisionError is raised. Defaults to 50. 
        min_iters : int
            Minimum number of iterations. Return None if max_iters < min_iters. Defaults to 10.
        verbose : bool
            If True, print the number of iterations.

        Returns
        -------
        solution : Optional[picos.Problem]
            The solution of the SDP problem. If the problem is not solved,
            return None.
        """
        sos = self.sos
        # verbose = self.verbose

        sos.options.max_iterations = max_iters
        if verbose:
            print('Retry Early Stop SOS Max Iters = %d' % sos.options.max_iterations)

        try:
            solution = sos._strategy.execute()
            return solution # .primals[sos.variables['y']]
        except Exception as e:
            if isinstance(e, ZeroDivisionError):
                if max_iters // 2 >= min_iters and max_iters > 1:
                    return self._nsolve_with_early_stop(
                                sos, 
                                max_iters = max_iters // 2, 
                                min_iters = min_iters, 
                                verbose = verbose
                            )
            return None
        return None


    def _get_defaulted_objectives(self):
        """Get the default objectives of the SDP problem."""
        obj_key = 'S_minor' if 'S_minor' in self.sos.variables else 'S_major'
        objectives = [
            ('max', self.sos.variables[obj_key].tr),
            ('min', self.sos.variables[obj_key].tr),
            ('max', self.sos.variables[obj_key]|1)
        ]
        # x = np.random.randn(*sos.variables[obj_key].shape)
        # objectives.append(('max', lambda sos: sos.variables[obj_key]|x))
        return objectives


    def _nsolve_with_obj(
            self,
            objectives: List[Tuple[str, Union[Any, Callable]]],
            context: Optional[AbstractContextManager] = None
        ) -> Optional[np.ndarray]:
        """
        Numerically solve a SDP problem with multiple objectives.
        This returns a generator of ndarray.

        Parameters
        ---------- 
        objectives : Optional[List[Tuple[str, Union[Any, Callable]]]]
            Although it suffices to find one feasible solution, we might 
            use objective to find particular feasible solution that has 
            good rational approximant. This parameter takes in multiple objectives, 
            and the solver will try each of the objective. If still no 
            approximant is found, the final solution will average this 
            SOS solution and perform rationalization. Note that SDP problem is 
            convex so the convex combination is always feasible and not on the
            boundary.

            Example: 
            ```
            objectives = [
                ('max', lambda sos: sos.variables['S_major'].tr),
                ('max', lambda sos: sos.variables['S_major']|1)
            ]
            ```

        Yields
        ---------
        y: Optional[np.ndarray]
            Numerical solution y. Return None if y unfounded.
        """
        from picos.modeling.strategy import Strategy
        sos = self.sos

        if context is None:
            context = nullcontext()
        if objectives is None:
            objectives = self._get_defaulted_objectives()

        with context:
            for objective in objectives:
                # try each of the objectives
                max_or_min, obj = objective
                if isinstance(obj, Callable):
                    obj = obj(sos)
                sos.set_objective(max_or_min, obj)

                sos._strategy = Strategy.from_problem(sos)
                solution = self._nsolve_with_early_stop(max_iters = 50)

                if solution is not None:
                    try:
                        y = np.array(solution.primals[sos.variables['y']])
                    except KeyError:
                        raise SDPInfeasibleError("SDP problem numerically infeasible.")

                    self._ys.append(y)
                    yield y

                # NOTE: PICOS uses an isometric vectorization of symmetric matrices
                #       (off-diagonal elements are divided by sqrt(2))
                #       so if we need to convert it back, we had better use its API.
                # S0 = (SymmetricVectorization((6,6)).devectorize(cvxopt.matrix(list(solution.primals.values())[0])))

                yield None

    def _nsolve_with_rationalization(
            self,
            objectives: List[Tuple[str, Union[Any, Callable]]],
            context: Optional[AbstractContextManager] = None,
            **kwargs
        ):
        for y in self._nsolve_with_obj(objectives, context):
            if y is not None:
                ra = self.rationalize(y, **kwargs)
                if ra is not None:
                    return ra

    def rationalize(
            self,
            y: np.ndarray,
            try_rationalize_with_mask: bool = True,
            times: int = 0,
            check_pretty: bool = True
        ) ->  Optional[Tuple[sp.Matrix, List[Tuple[sp.Matrix, sp.Matrix, List[sp.Rational]]]]]:
        decomp = rationalize_and_decompose(y, *self.args,
            try_rationalize_with_mask=try_rationalize_with_mask, times=times, check_pretty=check_pretty
        )
        return decomp

    def rationalize_combine(
            self,
            ys: List[np.ndarray] = None,
            verbose: bool = False,
        ) ->  Optional[Tuple[sp.Matrix, List[Tuple[sp.Matrix, sp.Matrix, List[sp.Rational]]]]]:
            if ys is None:
                ys = self._ys

            if len(ys) == 0:
                return None

            y = np.array(ys).mean(axis = 0)

            S_numer = S_from_y(y, *self.args)
            if all(_.is_positive_definite for _ in S_numer):
                lcm, times = 1260, 5
            else:
                lcm = max(1260, sp.prod(set.union(*[set(sp.primefactors(_.q)) for _ in space])))
                times = int(10 / sp.log(lcm, 10).n(15) + 3)

            if verbose:
                print('Minimum Eigenvals = %s'%[min(map(lambda x:sp.re(x), _.eigenvals())) for _ in S_numer])

            decomp = rationalize_and_decompose(y, *self.args,
                try_rationalize_with_mask = False, lcm = 1260, times = times
            )
            return decomp


    def _solve_trivial(
            self,
            objectives: Optional[List[Tuple[str, Callable]]] = None
        ):
        return self._nsolve_with_rationalization(objectives)


    def _solve_relax(self):
        import picos
        from picos.constraints.con_lmi import LMIConstraint

        sos = self.sos
        obj_key = 'S_minor' if not 'S_major' in sos.variables else 'S_major'
        lamb = picos.RealVariable('lamb', 1)
        obj = sos.variables[obj_key]

        @contextmanager
        def restore_constraints(sos, obj, lamb):    
            for i, constraint in enumerate(sos.constraints):
                if isinstance(constraint, LMIConstraint) and obj in constraint.variables:
                    # remove obj >> 0
                    sos.remove_constraint(i)
                    break
            sos.add_constraint((obj - lamb * picos.I(obj.shape[0])) >> 0)
            sos.add_constraint(lamb >= 0)

            yield
            sos.remove_constraint(-1)
            sos.remove_constraint(-1)
            sos.set_objective('max', obj.tr)

        objectives = [('max', lambda sos: sos.variables['lamb'])]
        context = restore_constraints(sos, obj, lamb)
        return self._nsolve_with_rationalization(objectives, context)


    def _solve_partial_deflation(
            self,
            deflation_sequence: Optional[List[int]] = None,
            verbose: bool = False
        ):
        @contextmanager
        def restore_constraints(sos):
            constraints_num = len(sos.constraints)
            yield
            for i in range(len(sos.constraints) - 1, constraints_num - 1, -1):
                sos.remove_constraint(i)

        n = self.dof
        sos = self.sos
        if deflation_sequence is None:
            deflation_sequence = range(n)

        with restore_constraints(sos):
            for i in deflation_sequence:
                bounds = []
                objectives = [
                    ('max', lambda sos: sos.variables['y'][i]),
                    ('min', lambda sos: sos.variables['y'][i])
                ]
                cnt_ys = len(self._ys)
                ra = self._nsolve_with_rationalization(objectives, verbose = verbose)
                cnt_sol = len(self._ys) - cnt_ys

                if cnt_sol == 0 or isinstance(ra, tuple):
                    return ra
                elif cnt_sol < 2:
                    # not enough solutions
                    return None

                bounds = [self._ys[-2][i], self._ys[-1][i]]

                # fix == (max + min) / 2
                fix = (bounds[0] + bounds[1]) / 2
                eps = (bounds[0] - bounds[1]) / 2
                if eps <= 1e-7:
                    # this implies bounds[0] == bounds[1]
                    fix = rationalize(fix, reliable = True) if abs(fix) > 1e-7 else 0
                elif bounds[0] > round(fix) > bounds[1]:
                    fix = round(fix)
                else:
                    fix = rationalize(fix, rounding = eps * .8, reliable = False)

                if verbose:
                    print('Deflate y[%d] = %s Bounds = %s'%(i, fix, bounds))

                sos.add_constraint(sos.variables['y'][i] == float(fix))


    def _solve_degenerated(self):
        if self.dof == 0:
            decomp = rationalize_and_decompose(
                sp.Matrix([]).reshape(0,1), *self.args,
                check_pretty = False
            )
            return decomp


    def _solve_wrapped(
        self,
        method: str = 'partial deflation',
        allow_numer: bool = False,
        verbose: bool = False,
        **kwargs
    ):
        method = method.lower()
        if method == 'trivial':
            ra = self._solve_trivial(**kwargs)
        elif method == 'relax':
            ra = self._solve_relax(**kwargs)
        elif method == 'partial deflation':
            ra = self._solve_partial_deflation(verbose=verbose, **kwargs)
        else:
            raise ValueError("Method %s is not supported."%method)

        if ra is not None:
            return ra


        ra = self.rationalize_combine(verbose = verbose)
        if ra is not None:
            return ra

        if len(self._ys) > 0:
            if allow_numer:
                y = sp.Matrix(self._ys[-1])
                decomp = rationalize_and_decompose(y, *self.args,
                    try_rationalize_with_mask = False, times = 0, perturb = True, check_pretty = False
                )
                return decomp
            else:
                raise SDPRationalizeError(
                    "Failed to find a rational solution despite having a numerical solution."
                )

        return None

    def solve(
            self,
            method: str = 'partial deflation',
            allow_numer: bool = False,
            verbose: bool = False,
            **kwargs
        ) -> SDPResult:

        if self.dof == 0:
            solution = self._solve_degenerated()
        elif self._has_picos:
            solution = self._solve_wrapped(method = method, allow_numer = allow_numer, verbose = verbose, **kwargs)

        if solution is not None:
            solution = {
                'y': solution[0],
                'S': dict((key, S[0]) for key, S in zip(self.keys, solution[1])),
                'decompositions': dict((key, S[1:]) for key, S in zip(self.keys, solution[1]))
            }
            self.y = solution['y']
            self.S = solution['S']
        return SDPResult(self.sos, solution)