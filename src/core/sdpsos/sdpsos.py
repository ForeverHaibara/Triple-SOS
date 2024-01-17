from copy import deepcopy
from typing import List, Optional, Union, Tuple, Dict, Callable

import sympy as sp

from .utils import (
    upper_vec_of_symmetric_matrix, solve_undetermined_linear, split_vector, S_from_y, 
    indented_print
)
from .solver import sdp_solver, SDPResult
from .manifold import RootSubspace, _REDUCE_KWARGS, coefficient_matrix, add_cyclic_constraints
from .solution import create_solution_from_M, SolutionSDP
from ...utils.basis_generator import arraylize, arraylize_sp
from ...utils.polytools import deg


class SDPProblem():
    """
    Helper class for SDPSOS. See details at SDPProblem.solve.

    Assume that a polynomial can be written in the form v^T @ M @ v.
    Sometimes there are implicit constraints that M = Q @ S @ Q.T where Q is a rational matrix.
    So we can solve the problem on S first and then restore it back to M.

    To summarize, it is about solving for S >> 0 such that
    eq @ vec(S) = vec(P) where P is determined by the target polynomial.
    """
    def __init__(self, 
            poly,
            manifold = None,
            verbose_manifold = True
        ):
        self.poly = poly
        self.poly_degree = deg(poly)

        info = {'major': None, 'minor': None, 'multiplier': None}
        self.Q = deepcopy(info)
        self.deg = deepcopy(info)
        self.M = deepcopy(info)

        self.S = deepcopy(info)
        self.decompositions = deepcopy(info)

        self.masked_rows = deepcopy(info)

        self.eq = deepcopy(info)
        self.vecP = None

        # unmasked S, x0, space, splits
        self.S_ = deepcopy(info)
        self.x0_ = None
        self.space_ = None
        self.splits_ = None

        # masked x0, space, splits
        self.x0 = None
        self.space = None
        self.splits = None

        self.sos = None
        self.y = None

        self.success = False

        if manifold is None:
            manifold = RootSubspace(poly)
        self.manifold = manifold
        if verbose_manifold:
            print(manifold)


    def _masked_dims(self, filter_zero = False):
        dims = {}
        for key, Q in self.Q.items():
            if Q is None:
                v = 0
            else:
                mask = self.masked_rows.get(key, [])
                v = Q.shape[1] - len(mask)
            if filter_zero and v == 0:
                continue
            dims[key] = v
        return dims


    def _not_none_keys(self):
        return list(self._masked_dims(filter_zero = True))

    def update(self, collection):
        if isinstance(collection, SDPResult):
            collection = collection.as_dict()
        for k, v in collection.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)


    def _compute_perp_space(self, minor = 0) -> Dict[str, sp.Matrix]:
        """
        Construct the perpendicular space of the problem.
        """
        self.Q = {'major': None, 'minor': None, 'multiplier': None}

        # positive indicates whether we solve the problem on R+ or R
        degree = self.poly_degree
        positive = not (degree % 2 == 0 and minor == 0)
        manifold = self.manifold
        self.deg['major'] = degree // 2
        self.Q['major'] = manifold.perp_space(minor = 0, positive = positive)

        if minor and degree > 2:
            self.deg['minor'] = degree // 2 - 1
            self.Q['minor'] = manifold.perp_space(minor = 1, positive = positive)

        return self.Q


    def _compute_equation(self, cyclic_constraint = True) -> Tuple[sp.Matrix, sp.Matrix]:
        """
        Construct the problem eq @ vec(S) = vec(M) where S is what we solved for.
        Return eq, vecM.
        """
        degree = self.poly_degree
        for key in self.Q.keys():
            Q = self.Q[key]
            if Q is not None and Q.shape[1] > 0:
                eq = coefficient_matrix(Q, self.deg[key], **_REDUCE_KWARGS[(degree%2, key)])
                self.eq[key] = eq
            else:
                self.Q[key] = None

        self.vecP = arraylize_sp(self.poly, cyc = False)
        
        if cyclic_constraint:
            add_cyclic_constraints(self)

        eq = sp.Matrix.hstack(*filter(lambda x: x is not None, self.eq.values()))
        return eq, self.vecP


    def _compute_subspace(self, eq, vecM) -> Tuple[sp.Matrix, sp.Matrix, List[slice]]:
        """
        Given eq @ vec(S) = vec(M), if we want to solve S, then we can
        see that S = x0 + space * y where x0 is a particular solution and y is arbitrary.
        """
        # we have eq @ vecS = vecM
        # so that vecS = x0 + space * y where x0 is a particular solution and y is arbitrary
        x0, space = solve_undetermined_linear(eq, vecM)
        splits = split_vector(list(self._masked_dims().values()))

        self.x0_ = x0
        self.x0 = x0
        self.space_ = space
        self.space = space
        self.splits_ = splits
        self.splits = splits
        return x0, space, splits

 
    def construct_problem(self, minor = 0, cyclic_constraint = True, verbose = True) -> Tuple[sp.Matrix, sp.Matrix, List[slice]]:
        """
        Construct the symbolic representation of the problem.
        """
        self.masked_rows = {}
        Q = self._compute_perp_space(minor = minor)
        eq, vecP = self._compute_equation(cyclic_constraint = cyclic_constraint)
        subspace = self._compute_subspace(eq, vecP)
        return subspace


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
        self.x0, self.space, self.splits = self.x0_, self.space_, self.splits_
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


    def solve(self,
            minor: bool = False,
            cyclic_constraint: bool = True,
            skip_construct_subspace: bool = False,
            method: str = 'trivial',
            reg: float = 0,
            constraints: Optional[List[Callable]] = None,
            objectives: Optional[List[Tuple[str, Callable]]] = None,
            allow_numer: bool = False,
            verbose: bool = False
        ) -> bool:
        """
        Solve a polynomial SOS problem with SDP.

        Parameters
        ----------
        minor : bool
            For a problem of even degree, if it holds for all real numbers, it might be in the 
            form of sum of squares. However, if it only holds for positive real numbers, then
            it might be in the form of \sum (...)^2 + \sum ab(...)^2. Note that we have an
            additional term in the latter, which is called the minor term. If we need to 
            add the minor term, please set `minor = True`.
        cyclic_constraint : bool
            Whether to add cyclic constraint the problem. This reduces the degree of freedom.
        skip_construct_subspace : bool
            Whether to skip the computation of the subspace. This is useful when we have
            already computed the subspace and want to solve the problem with different
            sdp configurations.
        method: str
            The method to solve the SDP problem. Currently supports:
            'partial deflation' and 'trivial'.
        reg : float
            We require `S[i]` to be positive semidefinite, but in practice
            we might want to add a small regularization term to make it
            positive definite >> reg * I.
        constraints : Optional[List[Callable]]
            Extra constraints of the SDP problem. This is not called when the problem is degenerated
            (when the degree of freedom is zero).

            Example:
            ```
            constraints = [
                lambda sos: sos.variables['y'][0] == 0,
            ]
            ```
        objectives : Optional[List[Tuple[str, Callable]]]
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
        allow_numer : bool
            Whether to allow numerical solution. If True, then the function will return numerical solution
            if the rational solution does not exist.
        verbose : bool
            If True, print the information of the solving process.

        Returns
        -------
        bool
            Whether the problem is solved successfully. It can also be accessed by `sdp_problem.success`.
        """
        self.success = False

        if not skip_construct_subspace:
            try:
                subspace = self.construct_problem(minor = minor, cyclic_constraint = cyclic_constraint, verbose = verbose)
            except:
                if verbose:
                    print('Linear system no solution. Please higher the degree by multiplying something %s.'%(
                        'or use the minor term' if not minor else ''
                    ))
                return False
        else:
            subspace = (self.x0, self.space, self.splits)


        if verbose:
            print('Matrix shape: %s'%(str(
                    {k: '{}/{}'.format(v, self.Q[k].shape[0])
                        for k, v in self._masked_dims(filter_zero = True).items()}
                ).replace("'", '')))

        # Main SOS solver
        sos_result = sdp_solver(
            *subspace,
            self._not_none_keys(),
            method = method,
            reg = reg,
            constraints = constraints,
            objectives = objectives,
            allow_numer = allow_numer,
            verbose = verbose
        )
        self.update(sos_result)

        if not sos_result.success:
            return False

        self.M = self.compute_M(self.S)

        return True


    def S_from_y(self, y: Optional[sp.Matrix] = None) -> Dict[str, sp.Matrix]:
        """
        Given y, compute the symmetric matrices. This is useful when we want to see the
        symbolic representation of the SDP problem.

        This function does not register the result to self.S.
        """
        if y is None:
            m = self.space.shape[1]
            y = sp.Matrix([sp.symbols('y_{%d}'%_) for _ in range(m)]).reshape(m, 1)
        elif not isinstance(y, sp.MatrixBase) or y.shape != (self.space.shape[1], 1):
            raise ValueError('y must be a sympy Matrix of shape (%d, 1).'%self.space.shape[1])

        Ss = S_from_y(y, self.x0, self.space, self.splits)

        ret = {}
        for key, S in zip(self._not_none_keys(), Ss):
            ret[key] = S
        return ret


    def compute_M(self, S: Dict[str, sp.Matrix]) -> Dict[str, sp.Matrix]:
        """
        Restore M = Q @ S @ Q.T from S.
        """
        M = {}
        self.S_ = {}
        for key in self._not_none_keys():
            self.S_[key] = self.pad_masked_rows(S, key)
            M[key] = self.Q[key] * self.S_[key] * self.Q[key].T
        return M


    def as_solution(self, 
            y: Optional[sp.Matrix] = None,
            decompose_method: str = 'raw',
            cyc: bool = True,
            factor_result: bool = True
        ):
        """
        Wrap the matrix form solution to a SolutionSDP object.
        Note that the decomposition of a quadratic form is not unique.

        Parameters
        ----------
        y : Optional[sp.Matrix]
            The y vector. If None, then we use the solution of the SDP problem.
        decompose_method : str
            One of 'raw' or 'reduce'. The default is 'raw'.
        cyc : bool
            Whether to convert the solution to a cyclic sum.
        factor_result : bool
            Whether to factorize the result. The default is True.
        """

        if y is None:
            S = self.S_
            M = self.M
            if (not S) or not any(S.values()):
                raise ValueError('The problem is not solved yet.')
        else:
            S = self.S_from_y(y)
            M = self.compute_M(S)

        return create_solution_from_M(
            poly = self.poly,
            S = S,
            Q = self.Q,
            M = M,
            decompose_method = decompose_method,
            cyc = cyc,
            factor_result = factor_result,
        )



def SDPSOS(
        poly: sp.Poly,
        minor: Union[List[bool], bool] = [False, True],
        degree_limit: int = 12,
        cyclic_constraint = True,
        method: str = 'trivial',
        allow_numer: bool = False,
        decompose_method: str = 'raw',
        factor_result: bool = True,
        verbose: bool = True,
        **kwargs
    ) -> Optional[SolutionSDP]:
    """
    Solve a polynomial SOS problem with SDP.

    Although the theory of numerical solution to sum of squares using SDP (semidefinite programming)
    is well established, there exists certain limitations in practice. One of the most major
    concerns is that we need accurate, rational solution rather a numerical one. One might argue 
    that SDP is convex and we could perturb a solution to get a rational, interior one. However,
    this is not always the case. If the feasible set of SDP is convex but not full-rank, then
    our solution might be on the boundary of the feasible set. In this case, perturbation does
    not work.

    To handle the problem, we need to derive the true low-rank subspace of the feasible set in advance
    and perform SDP on the subspace. Take Vasile's inequality as an example, s(a^2)^2 - 3s(a^3b) >= 0
    has four equality cases. If it can be written as a positive definite matrix M, then we have
    x'Mx = 0 at these four points. This leads to Mx = 0 for these four vectors. As a result, the 
    semidefinite matrix M lies on a subspace perpendicular to these four vectors. We can assume 
    M = QSQ' where Q is the nullspace of the four vectors x, so that the problem is reduced to find S.

    Hence the key problem is to find the root and construct such Q. Also, in our algorithm, the Q
    is constructed as a rational matrix, so that a rational solution to S converts back to a rational
    solution to M. We must note that the equality cases might not be rational as in Vasile's inequality.
    However, the cyclic sum of its permutations is rational. So we can use the linear combination of 
    x and its permutations, which would be rational, to construct Q. This requires knowledge of 
    algebraic numbers and minimal polynomials.

    For more flexible usage, please use
    ```
        sdp_problem = SDPProblem(poly)
        sdp_problem.solve(**kwargs)
        solution = sdp_problem.as_solution()
    ```

    Parameters
    ----------
    poly : sp.Poly
        Polynomial to be solved.
    minor : Union[List[bool], bool]
        For a problem of even degree, if it holds for all real numbers, it might be in the
        form of sum of squares. However, if it only holds for positive real numbers, then
        it might be in the form of \sum (...)^2 + \sum ab(...)^2. Note that we have an
        additional term in the latter, which is called the minor term. If we need to
        add the minor term, please set minor = True.
        The function also supports multiple trials. The default is [False, True], which
        first tries to solve the problem without the minor term.
    degree_limit : int
        The maximum degree of the polynomial to be solved. When the degree is too high,
        return None.
    cyclic_constraint : bool
        Whether to add cyclic constraint the problem. This reduces the degree of freedom.
    method : str
        The method to solve the SDP problem. Currently supports:
        'partial deflation' and 'relax' and 'trivial'.
    allow_numer : bool
        Whether to allow numerical solution. If True, then the function will return numerical solution
        if the rational solution does not exist.
    decompose_method : str
        One of 'raw' or 'reduce'. The default is 'raw'.
    factor_result : bool
        Whether to factorize the result. The default is True.
    verbose : bool
        If True, print the information of the problem.
    """
    degree = deg(poly)
    if degree > degree_limit or degree < 2:
        return None
    if not (poly.domain in (sp.polys.ZZ, sp.polys.QQ)):
        return None

    sdp_problem = SDPProblem(poly, verbose_manifold=verbose)

    if isinstance(minor, (bool, int)):
        minor = [minor]

    for minor_ in minor:
        if verbose:
            print('SDP Minor = %d:'%minor_)

        with indented_print(verbose = verbose):
            try:
                sdp_problem.solve(
                    minor = minor_,
                    cyclic_constraint = cyclic_constraint,
                    method = method,
                    allow_numer = allow_numer,
                    verbose = verbose
                )
                if sdp_problem.success:
                    # We can also pass in **M
                    return sdp_problem.as_solution(
                        decompose_method = decompose_method, 
                        factor_result = factor_result
                    )
                    if verbose:
                        print('Success.')
            except Exception as e:
                if verbose:
                    print(e)
            if verbose:
                print('Failed.')

    return None