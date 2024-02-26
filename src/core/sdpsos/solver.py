from typing import List, Optional, Tuple, Callable, Dict, Any, Union, Generator
from contextlib import contextmanager, nullcontext, AbstractContextManager
from copy import deepcopy

import numpy as np
import sympy as sp

from .rationalize import rationalize, rationalize_and_decompose
from .ipm import (
    SDPConvergenceError, SDPNumericalError, SDPInfeasibleError, SDPRationalizeError
)
from .utils import (
    solve_undetermined_linear, S_from_y, Mat2Vec
)


Decomp = List[Tuple[sp.Matrix, sp.Matrix, List[sp.Rational]]]
PicosExpression = Any

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


def _decompose_matrix(
        M: sp.Matrix,
        variables: Optional[List[sp.Symbol]] = None
    ) -> Tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """
    Decomposes a symbolic matrix into the form vec(M) = x + A @ v
    where x is a constant vector, A is a constant matrix, and v is a vector of variables.

    Parameters
    ----------
    M : sp.Matrix
        The matrix to be decomposed.
    variables : List[sp.Symbol]
        The variables to be used in the decomposition. If None, it uses M.free_symbols.

    Returns
    ----------
    x : sp.Matrix
        The constant vector.
    A : sp.Matrix
        The constant matrix.
    v : sp.Matrix
        The vector of variables.
    """
    rows, cols = M.shape
    variables = list(M.free_symbols) if variables is None else variables
    variable_index = {var: idx for idx, var in enumerate(variables)}

    v = sp.Matrix(variables)
    x = sp.zeros(rows * cols, 1)

    A = sp.zeros(rows * cols, len(variables))

    for i in range(rows):
        for j in range(cols):
            expr = M[i, j]
            terms = sp.collect(expr, variables, evaluate=False)

            constant_term = terms.pop(sp.S.One, 0)  # Extract and remove constant term for x

            x[i * cols + j] = constant_term

            for term, coeff in terms.items():
                A[i * cols + j, variable_index[term]] = coeff  # Extract coefficients for A

    return x, A, v


class SDPProblem():
    """
    Class to solve rational SDP feasible problems, which is in the form of

        S_i = C_i + y_1 * A_i1 + y_2 * A_i2 + ... + y_n * A_in >> 0.
    
    where C, A_ij ... are known symmetric matrices, and y_i are free variables.

    It can be rewritten in the form of

        vec(S_i) = x_i + space_i @ y >> 0.

    And together they are vec([S_1, S_2, ...]) = [x_1, x_2, ...] + [space_1, space_2, ...] @ y
    where x_i and space_i are known. The problem is to find a rational solution y such that S_i >> 0.
    This is the standard form of our rational SDP feasible problem.
    """
    _has_picos = _check_picos(verbose = True)
    def __init__(self, x0, space, splits, keys = None, free_symbols = None):
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
        self.decompositions = None

        if keys is not None:
            if len(keys) != len(splits):
                raise ValueError("Length of keys and splits should be the same. But got %d and %d."%(len(keys), len(splits)))
            self.keys = keys
        else:
            self.keys = ['S_%d'%i for i in range(len(splits))]

        if free_symbols is not None:
            if len(free_symbols) != self.space.shape[1]:
                raise ValueError("Length of free_symbols and space should be the same. But got %d and %d."%(len(free_symbols), self.space.shape[1]))
            self.free_symbols_ = list(free_symbols)
        else:
            self.free_symbols_ = list(sp.Symbol('y_{%d}'%i) for i in range(self.space.shape[1]))
        self.free_symbols = self.free_symbols_

        self.sdp = None

        # record the numerical solutions
        self._ys = []

    @property
    def args(self):
        """
        Return the construction args for the object: (x0, space, splits).
        """
        return self.x0, self.space, self.splits

    @property
    def dof(self):
        """
        The degree of freedom of the SDP problem.
        """
        return self.space.shape[1]


    @classmethod
    def from_equations(cls, eq, vecP, splits, **kwargs) -> 'SDPProblem':
        x0, space = solve_undetermined_linear(eq, vecP)
        return SDPProblem(x0, space, splits, **kwargs)

    @classmethod
    def from_matrix(cls, S, **kwargs) -> 'SDPProblem':
        invalid_kwargs = ['splits', 'free_symbols']
        for name in invalid_kwargs:
            if name in kwargs:
                raise ValueError(f"Cannot specify {name} when constructing SDPProblem from matrix.")

        if isinstance(S, dict):
            kwargs['keys'] = list(S.keys())
            S = list(S.values())

        if isinstance(S, sp.Matrix):
            S = [S]

        free_symbols = set()
        for s in S:
            if not isinstance(s, sp.Matrix):
                raise ValueError("S must be a list of sp.Matrix or dict of sp.Matrix.")
            free_symbols |= set(s.free_symbols)
        splits = Mat2Vec.split_vector(S)

        free_symbols = list(free_symbols)


        #############   TOO SLOW BELOW   ################
        # params = {k: 0 for k in free_symbols}
        # def to_vec(params):
        #     mats = [s.subs(params) for s in S]
        #     upper_vecs = [sp.Matrix(list(upper_vec_of_symmetric_matrix(_))) for _ in mats]
        #     return sp.Matrix.vstack(*upper_vecs)

        # x0 = to_vec(params)
        # space = sp.Matrix.zeros(x0.shape[0], len(free_symbols))
        # for i in range(len(free_symbols)):
        #     params[free_symbols[i]] = 1
        #     xi = to_vec(params)
        #     xi = xi - x0
        #     space[:, i] = xi
        #     params[free_symbols[i]] = 0
        #############   TOO SLOW ABOVE   ################

        # alternative: use _decopose_matrix
        x0_list, space_list = [], []
        for s in S:
            x0, space, _ = _decompose_matrix(s, free_symbols)
            x0_list.append(x0)
            space_list.append(space)
        x0 = sp.Matrix.vstack(*x0_list)
        space = sp.Matrix.hstack(*space_list)

        return SDPProblem(x0, space, splits, free_symbols = free_symbols, **kwargs)


    def _masked_dims(self, filter_zero: bool = False) -> Dict[str, int]:
        """
        Compute the dimensions of each symmetric matrix after row-masking.

        Parameters
        ----------
        filter_zero : bool
            If filter_zero == True, then keys of dimension zero will be ignored.

        Returns
        ----------
        dims : Dict[str, int]
            Dimensions of the symmetric matrices after row-masking.
        """
        dims = {}
        for i in range(len(self.keys)):
            key = self.keys[i]
            split = self._splits[i]
            mask = self.masked_rows.get(key, [])
            k = round(np.sqrt(2 * (split.stop - split.start) + .25) - .5)
            v = k - len(mask)
            if filter_zero and v == 0:
                continue
            dims[key] = v
        return dims

    def _not_none_keys(self) -> List[str]:
        """
        Return keys that dim[key] > 0 after row-masking.

        Returns
        ----------
        keys : List[str]
            Keys that dim[key] > 0.
        """
        return list(self._masked_dims(filter_zero = True))


    def set_masked_rows(self,
            masks: Dict[str, List[int]] = {}
        ) -> Dict[str, sp.Matrix]:
        """
        Sometimes the diagonal entries of S are zero. Or we set them to zero to
        reduce the degree of freedom. This function masks the corresponding rows.

        Parameters
        ----------
        masks : List[int]
            Indicates the indices of the rows to be masked.

        Returns
        ----------
        masks : List[int]
            The input.
        """
        # restore masked values to unmaksed values
        self.x0, self.space, self.splits = self._x0, self._space, self._splits
        self.free_symbols = self.free_symbols_
        self._ys = []
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
            n = round(np.sqrt(2 * (split.stop - split.start) + .25) - .5)
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
        self.splits = Mat2Vec.split_vector(list(self._masked_dims().values()))

        self.free_symbols = list(sp.Symbol('y_{%d}'%i) for i in range(self.space.shape[1]))
        return masks


    def pad_masked_rows(self, 
            S: Union[Dict, sp.Matrix],
            key: str
        ) -> sp.Matrix:
        """
        Pad the masked rows of S[key] with zeros. This is an "inversed" 
        operation of the row-masking.

        Parameters
        ----------
        S : sp.Matrix
            Solved symmetric matrices after row-masking.
        key : str
            The key of the matrix. It is used to obtain the mask.

        Returns
        ----------
        S : sp.Matrix
            The restored S before row_masking.
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


    def S_from_y(self, 
            y: Optional[Union[sp.Matrix, np.ndarray, Dict]] = None
        ) -> Dict[str, sp.Matrix]:
        """
        Given y, compute the symmetric matrices. This is useful when we want to see the
        symbolic representation of the SDP problem.

        This function does not register the result to self.S.

        Parameters
        ----------
        y : Optional[Union[sp.Matrix, np.ndarray]]
            The generating vector. If None, it uses a symbolic vector.

        Returns
        ----------
        S : Dict[str, sp.Matrix]
            The symmetric matrices that SDP requires to be positive semidefinite.
        """
        m = self.dof
        if y is None:
            y = sp.Matrix(self.free_symbols).reshape(m, 1)
        elif isinstance(y, sp.MatrixBase):
            if y.shape != (m, 1):
                raise ValueError('y must be a matrix of shape (%d, 1).'%m)
        elif isinstance(y, np.ndarray):
            if y.size != m:
                raise ValueError('y must be a matrix of shape (%d, 1).'%m)
            y = sp.Matrix(y.flatten())
        elif isinstance(y, dict):
            y = sp.Matrix([y.get(v, v) for v in self.free_symbols]).reshape(m, 1)

        Ss = S_from_y(y, *self.args)

        ret = {}
        for key, S in zip(self._not_none_keys(), Ss):
            ret[key] = S
        return ret

    def as_params(self) -> Dict[sp.Symbol, sp.Rational]:
        """
        Return the free symbols and their values.
        """
        return dict(zip(self.free_symbols, self.y))


    def _construct_sdp(self,
            reg: float = 0,
            constraints: List[Union[PicosExpression, sp.Expr, Callable]] = []
        ):
        """
        Construct picos.Problem from self. The function
        is automatically called when __init__.

        Parameters
        ----------
        reg : float
            For symmetric matrix S, we require S >> reg * I.
        constraints : List[Union[PicosExpression, sp.Expr, Callable]]:
            Additional constraints.

            Example:
            ```
                constraints = [
                    lambda sdp: sdp.variables['y'][0] == 1
                ]
            ```

        Returns
        ---------
        sdp : picos.Problem
            Picos problem created. If there is no degree of freedom,
            return None.
        """
        if self.dof == 0:
            return None


        try:
            import picos

            # SDP should use numerical algorithm
            x0, space, splits = self.args
            x0_numer = np.array(x0).astype(np.float64).flatten()
            space_numer = np.array(space).astype(np.float64)

            sdp = picos.Problem()
            y = picos.RealVariable('y', self.dof)
            for key, split in zip(self._not_none_keys(), splits):
                x0_ = x0_numer[split]
                k = Mat2Vec.length_of_mat(x0_.shape[0])
                S = picos.SymmetricVariable(key, (k,k))
                sdp.add_constraint(S >> reg)

                self._add_sdp_eq(sdp, S, x0_, space_numer[split], y)

            for constraint in constraints or []:
                constraint = self._align_constraint(constraint)
                sdp.add_constraint(constraint)
        except Exception as e:
            raise e
            return None

        self.sdp = sdp
        return sdp

    def _add_sdp_eq(self, sdp, S, x0, space, y):
        """
        Helper function that add the constraint
        S.vec == x0 + space * y to the sdp.
        """
        sdp.add_constraint(S.vec == x0 + space * y)


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
        sdp = self.sdp
        # verbose = self.verbose

        sdp.options.max_iterations = max_iters
        if verbose:
            print('Retry Early Stop sdp Max Iters = %d' % sdp.options.max_iterations)

        try:
            solution = sdp._strategy.execute()
            return solution # .primals[sdp.variables['y']]
        except Exception as e:
            if isinstance(e, ZeroDivisionError):
                if max_iters // 2 >= min_iters and max_iters > 1:
                    return self._nsolve_with_early_stop(
                                max_iters = max_iters // 2, 
                                min_iters = min_iters, 
                                verbose = verbose
                            )
            return None
        return None


    def _get_defaulted_objectives(self):
        """
        Get the default objectives of the SDP problem.
        """
        obj_key = self._not_none_keys()[0]
        objectives = [
            ('max', self.sdp.variables[obj_key].tr),
            ('min', self.sdp.variables[obj_key].tr),
            ('max', self.sdp.variables[obj_key]|1)
        ]
        # x = np.random.randn(*sdp.variables[obj_key].shape)
        # objectives.append(('max', lambda sdp: sdp.variables[obj_key]|x))
        return objectives

    def _align_objective(
            self,
            objective: Tuple[str, Union[PicosExpression, sp.Expr, Callable]]
        ) -> Tuple[str, PicosExpression]:
        """
        Align the objective to an expression of sdp.variables.
        """
        indicator, x = objective
        if isinstance(x, Callable):
            x = x(self.sdp)
        elif isinstance(x, sp.Expr):
            f = sp.lambdify(self.free_symbols, x)
            x = f(*self.sdp.variables['y'])
        return indicator, x

    def _align_constraint(
        self,
        constraint: Union[PicosExpression, sp.Expr, Callable]
    ) -> PicosExpression:
        """
        Align the constraint to an expression of sdp.variables.
        """
        x = constraint
        if isinstance(x, Callable):
            x = x(self.sdp)
        elif isinstance(x, sp.core.relational.Relational):
            lhs = sp.lambdify(self.free_symbols, x.lhs)(*self.sdp.variables['y'])
            rhs = sp.lambdify(self.free_symbols, x.rhs)(*self.sdp.variables['y'])
            sym = {
                sp.GreaterThan: '__ge__',
                sp.StrictGreaterThan: '__ge__',
                sp.LessThan: '__le__',
                sp.StrictLessThan: '__le__',
                sp.Equality: '__eq__'
            }[x.__class__]
            x = getattr(lhs, sym)(rhs)
        return x


    def _nsolve_with_obj(
            self,
            objectives: List[Tuple[str, Union[PicosExpression, sp.Expr, Callable]]],
            context: Optional[AbstractContextManager] = None
        ) -> Generator[Optional[np.ndarray], None, None]:
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
            sdp solution and perform rationalization. Note that SDP problem is 
            convex so the convex combination is always feasible and not on the
            boundary.

            Example: 
            ```
            objectives = [
                ('max', lambda sdp: sdp.variables['S_major'].tr),
                ('max', lambda sdp: sdp.variables['S_major']|1)
            ]
            ```
        context : Optional[AbstractContextManager]
            Context that the SDP is solved in.

        Yields
        ---------
        y: Optional[np.ndarray]
            Numerical solution y. Return None if y unfound.
        """
        from picos.modeling.strategy import Strategy
        sdp = self.sdp

        if context is None:
            context = nullcontext()
        if objectives is None:
            objectives = self._get_defaulted_objectives()

        with context:
            for objective in objectives:
                # try each of the objectives
                sdp.set_objective(*self._align_objective(objective))

                sdp._strategy = Strategy.from_problem(sdp)
                solution = self._nsolve_with_early_stop(max_iters = 50)

                if solution is not None:
                    try:
                        y = np.array(solution.primals[sdp.variables['y']])
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
            objectives: List[Tuple[str, Union[PicosExpression, sp.Expr, Callable]]],
            context: Optional[AbstractContextManager] = None,
            **kwargs
        ) -> Optional[Tuple[sp.Matrix, Decomp]]:
        """
        Solve the SDP problem and returns the rational solution if any.

        Parameters
        ----------
        objectives : List[Tuple[str, Union[PicosExpression, sp.Expr, Callable]]]
            See details in self._nsolve_with_obj.
        context : Optional[AbstractContextManager]
            See details in self._nsolve_with_obj.
        kwargs : Any
            Keyword arguments that passed into self.rationalize.

        Returns
        ----------
        y, decompositions : Optional[Tuple[sp.Matrix, Decomp]]
            If the problem is solved, return the congruence decompositions `y, [(S, U, diag)]`
            So that each `S = U.T * diag(diag) * U` where `U` is upper triangular.
            Otherwise, return None.
        """
        for y in self._nsolve_with_obj(objectives, context):
            if y is not None:
                ra = self.rationalize(y, **kwargs)
                if ra is not None:
                    return ra

    def rationalize(
            self,
            y: np.ndarray,
            try_rationalize_with_mask: bool = True,
            times: int = 1,
            check_pretty: bool = True,
        ) -> Optional[Tuple[sp.Matrix, Decomp]]:
        """
        Rationalize a numerical vector y so that it produces a rational solution to SDP.

        Parameters
        ----------
        y : np.ndarray
            Numerical solution y.
        kwargs : Any
            Arguments that passed into rationalize_and_decompose.

        Returns
        ----------
        y, decompositions : Optional[Tuple[sp.Matrix, Decomp]]
            If the problem is solved, return the congruence decompositions `y, [(S, U, diag)]`
            So that each `S = U.T * diag(diag) * U` where `U` is upper triangular.
            Otherwise, return None.
        """
        decomp = rationalize_and_decompose(y, *self.args,
            try_rationalize_with_mask=try_rationalize_with_mask, times=times, check_pretty=check_pretty
        )
        return decomp

    def rationalize_combine(
            self,
            ys: List[np.ndarray] = None,
            verbose: bool = False,
        ) ->  Optional[Tuple[sp.Matrix, Decomp]]:
        """
        Linearly combine all numerical solutions [y] to produce a rational solution.

        Parameters
        ----------
        y : np.ndarray
            Numerical solution y.
        verbose : bool
            Whether to print out the eigenvalues of the combined matrix. Defaults
            to False.

        Returns
        ----------
        y, decompositions : Optional[Tuple[sp.Matrix, Decomp]]
            If the problem is solved, return the congruence decompositions `y, [(S, U, diag)]`
            So that each `S = U.T * diag(diag) * U` where `U` is upper triangular.
            Otherwise, return None.
        """
        if ys is None:
            ys = self._ys

        if len(ys) == 0:
            return None

        y = np.array(ys).mean(axis = 0)

        S_numer = S_from_y(y, *self.args)
        if all(_.is_positive_definite for _ in S_numer):
            lcm, times = 1260, 5
        else:
            lcm = max(1260, sp.prod(set.union(*[set(sp.primefactors(_.q)) for _ in self.space if isinstance(_, sp.Rational)])))
            times = int(10 / sp.log(lcm, 10).n(15) + 3)

        if verbose:
            print('Minimum Eigenvals = %s'%[min(map(lambda x:sp.re(x), _.eigenvals())) for _ in S_numer])

        decomp = rationalize_and_decompose(y, *self.args,
            try_rationalize_with_mask = False, lcm = 1260, times = times
        )
        return decomp


    def _solve_trivial(
            self,
            objectives: Optional[List[Tuple[str, Union[PicosExpression, sp.Expr, Callable]]]] = None,
            constraints: List[Union[PicosExpression, sp.Expr, Callable]] = []
        ) -> Optional[Tuple[sp.Matrix, Decomp]]:
        """
        Solve SDP numerically with given objectives and constraints.
        """
        # a context that imposes additional constraints
        if len(constraints):
            @contextmanager
            def restore_constraints(sdp, constraints):
                constraints_num = len(sdp.constraints)
                for constraint in constraints:
                    constraint = self._align_constraint(constraint)
                    sdp.add_constraint(constraint)
                yield
                for i in range(len(sdp.constraints) - 1, constraints_num - 1, -1):
                    sdp.remove_constraint(i)
            context = restore_constraints(self.sdp, constraints)
        else:
            context = nullcontext()
        return self._nsolve_with_rationalization(objectives, context)


    def _solve_relax(
            self
        ) -> Optional[Tuple[sp.Matrix, Decomp]]:
        """
        Solve SDP with such objective:
            S - l * I >= 0.
            max(l)
        """
        import picos
        from picos.constraints.con_lmi import LMIConstraint

        sdp = self.sdp
        obj_key = self._not_none_keys()[0]
        lamb = picos.RealVariable('lamb', 1)
        obj = sdp.variables[obj_key]

        @contextmanager
        def restore_constraints(sdp, obj, lamb):
            for i, constraint in enumerate(sdp.constraints):
                if isinstance(constraint, LMIConstraint) and obj in constraint.variables:
                    # remove obj >> 0
                    sdp.remove_constraint(i)
                    break
            sdp.add_constraint((obj - lamb * picos.I(obj.shape[0])) >> 0)
            sdp.add_constraint(lamb >= 0)

            yield
            sdp.remove_constraint(-1)
            sdp.remove_constraint(-1)
            sdp.set_objective('max', obj.tr)

        objectives = [('max', lambda sdp: sdp.variables['lamb'])]
        context = restore_constraints(sdp, obj, lamb)
        return self._nsolve_with_rationalization(objectives, context)


    def _solve_partial_deflation(
            self,
            deflation_sequence: Optional[List[int]] = None,
            verbose: bool = False
        ) -> Optional[Tuple[sp.Matrix, Decomp]]:
        """
        We use the following idea to generate a rational solution:
        1. Solve SDP with objectives = max(y[-1]) and min(y[-1]).
        2. Set y[-1] = (max + min) / 2 as a new constraint and solve SDP again.
        3. Repeat step 2 until the solution is rational.
        """
        @contextmanager
        def restore_constraints(sdp):
            constraints_num = len(sdp.constraints)
            yield
            for i in range(len(sdp.constraints) - 1, constraints_num - 1, -1):
                sdp.remove_constraint(i)

        n = self.dof
        sdp = self.sdp
        if deflation_sequence is None:
            deflation_sequence = range(n)

        with restore_constraints(sdp):
            for i in deflation_sequence:
                bounds = []
                objectives = [
                    ('max', lambda sdp: sdp.variables['y'][i]),
                    ('min', lambda sdp: sdp.variables['y'][i])
                ]
                cnt_ys = len(self._ys)
                ra = self._nsolve_with_rationalization(objectives)
                cnt_sol = len(self._ys) - cnt_ys

                if cnt_sol == 0 or isinstance(ra, tuple):
                    return ra
                elif cnt_sol < 2:
                    # not enough solutions
                    return None

                ra = self.rationalize_combine(verbose = verbose)
                if ra is not None:
                    return ra

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

                sdp.add_constraint(sdp.variables['y'][i] == float(fix))


    def _solve_degenerated(
            self
        ) -> Optional[Tuple[sp.Matrix, Decomp]]:
        """
        Solve the SDP if degree of freedom is zero.
        In this case it does not rely on any optimization package.
        """
        if self.dof == 0:
            decomp = rationalize_and_decompose(
                sp.Matrix([]).reshape(0,1), *self.args,
                check_pretty = False
            )
            return decomp


    def _solve_wrapped(
        self,
        method: str = 'trivial',
        allow_numer: int = 0,
        verbose: bool = False,
        **kwargs
    ) -> Optional[Tuple[sp.Matrix, Decomp]]:
        """
        Solve SDP with given method. Moreover, we try to make a convex combinations
        of all numerical solution to test whether it produces a rational solution.
        Finally, if allow_numer == 1, return one of the numerical solution if rationalization fails.
        If allow_numer == 2, force to return the first numerical solution.

        Parameters
        ----------
        method : str
            The method to solve the SDP problem. Currently supports:
            'partial deflation' and 'relax' and 'trivial'.
        allow_numer : int
            Whether to allow numerical solution. If 0, then the function will return None if
            the rational feasible solution does not exist. If 1, then the function will return a numerical solution
            if the rational feasible solution does not exist. If 2, then the function will return the first
            numerical solution  without any rationalization.
        verbose : bool
            If True, print the information of the solving process.

        Returns
        ----------
        y, decompositions : Optional[Tuple[sp.Matrix, Decomp]]
            If the problem is solved, return the congruence decompositions `y, [(S, U, diag)]`
            So that each `S = U.T * diag(diag) * U` where `U` is upper triangular.
            Otherwise, return None.
        """
        method = method.lower()

        num_sol = len(self._ys)

        if method == 'trivial':
            ra = self._solve_trivial(**kwargs)
        elif method == 'relax':
            ra = self._solve_relax(**kwargs)
        elif method == 'partial deflation':
            ra = self._solve_partial_deflation(verbose=verbose, **kwargs)
        else:
            raise ValueError("Method %s is not supported."%method)

        if allow_numer < 2:
            if ra is not None:
                return ra

            ra = self.rationalize_combine(verbose = verbose)
            if ra is not None:
                return ra

        if len(self._ys) > num_sol:
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
            method: str = 'trivial',
            allow_numer: int = 0,
            verbose: bool = False,
            **kwargs
        ) -> bool:
        """
        Interface for solving the SDP problem.

        Parameters
        ----------
        method : str
            The method to solve the SDP problem. Currently supports:
            'partial deflation' and 'relax' and 'trivial'.
        allow_numer : int
            Whether to allow numerical solution. If 0, then the function will return None if
            the rational feasible solution does not exist. If 1, then the function will return a numerical solution
            if the rational feasible solution does not exist. If 2, then the function will return the first
            numerical solution without any rationalization.
        verbose : bool
            If True, print the information of the solving process.

        Returns
        ----------
        bool
            Whether the problem is solved. If True, the result can be accessed by
            SDPProblem.y and SDPProblem.S and SDPProblem.decompositions.
        """

        if self.dof == 0:
            solution = self._solve_degenerated()
        elif self._has_picos:
            self._construct_sdp()
            solution = self._solve_wrapped(method = method, allow_numer = allow_numer, verbose = verbose, **kwargs)
        else:
            solution = None

        if solution is not None:
            self.y = solution[0]
            self.S = dict((key, S[0]) for key, S in zip(self.keys, solution[1]))
            self.decompositions = dict((key, S[1:]) for key, S in zip(self.keys, solution[1]))
        return (solution is not None)


class SDPProblemEmpty(SDPProblem):
    def __init__(self, *args, **kwargs):
        self.masked_rows = {}