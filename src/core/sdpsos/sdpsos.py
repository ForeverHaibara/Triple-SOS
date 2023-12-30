from copy import deepcopy
from typing import List, Optional, Union, Tuple, Dict

import sympy as sp

from .utils import solve_undetermined_linear, split_vector, indented_print
from .solver import sdp_solver
from .manifold import RootSubspace, LowRankHermitian, _REDUCE_KWARGS, add_cyclic_constraints
from .solution import create_solution_from_M, SolutionSDP
from ...utils.basis_generator import arraylize, arraylize_sp
from ...utils.polytools import deg


class SDPProblem():
    def __init__(self, 
            poly,
            manifold = None,
            verbose_manifold = True
        ):
        self.poly = poly
        self.poly_degree = deg(poly)
        degree = deg(poly)

        info = {'major': None, 'minor': None, 'multiplier': None}
        self.Q = deepcopy(info)
        self.deg = deepcopy(info)
        self.low_rank = deepcopy(info)
        self.M = deepcopy(info)
        self.eq = deepcopy(info)

        self.S = deepcopy(info)
        self.decompositions = deepcopy(info)

        self.vecM = None
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

    def _not_none_keys(self):
        return [key for key, value in self.Q.items() if value is not None]

    def update(self, collection):
        for k, v in collection.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)


    def _compute_equation(self, minor = 0, cyclic_constraint = True) -> Tuple[sp.Matrix, sp.Matrix]:
        """
        Construct the problem eq @ vec(S) = vec(M) where S is what we solved for.
        Return eq, vecM.
        """
        # positive indicates whether we solve the problem on R+ or R
        degree = self.poly_degree
        positive = not (degree % 2 == 0 and minor == 0)
        manifold = self.manifold
        self.deg['major'] = degree // 2
        self.Q['major'] = manifold.perp_space(minor = 0, positive = positive)

        if minor and degree > 2:
            self.deg['minor'] = degree // 2 - 1
            self.Q['minor'] = manifold.perp_space(minor = 1, positive = positive)

        for key in self.Q.keys():
            Q = self.Q[key]
            if Q is not None and Q.shape[1] > 0:
                low_rank = LowRankHermitian(Q)
                self.low_rank[key] = low_rank

                eq = low_rank.reduce(self.deg[key], **_REDUCE_KWARGS[(degree%2, key)])
                self.eq[key] = eq
            else:
                self.Q[key] = None

        self.vecM = arraylize_sp(self.poly, cyc = False)
        
        if cyclic_constraint:
            add_cyclic_constraints(self)

        eq = sp.Matrix.hstack(*filter(lambda x: x is not None, self.eq.values()))
        return eq, self.vecM


    def _compute_subspace(self, eq, vecM, minor = 0, verbose = True) -> Tuple[sp.Matrix, sp.Matrix, List[slice]]:
        """
        Given eq @ vec(S) = vec(M), if we want to solve S, then we can
        see that S = x0 + space * y where x0 is a particular solution and y is arbitrary.
        """
        # we have eq @ vecS = vecM
        # so that vecS = x0 + space * y where x0 is a particular solution and y is arbitrary
        x0, space = solve_undetermined_linear(eq, vecM)
        splits = split_vector(self.eq.values())

        self.x0 = x0
        self.space = space
        self.splits = splits     
        return x0, space, splits

 
    def construct_problem(self, minor = 0, cyclic_constraint = True, verbose = True) -> Tuple[sp.Matrix, sp.Matrix, List[slice]]:
        """
        Construct the symbolic representation of the problem.
        """
        eq, vecM = self._compute_equation(minor = minor, cyclic_constraint = cyclic_constraint)
        subspace = self._compute_subspace(eq, vecM, minor = minor, verbose = verbose)
        return subspace


    def solve(self,
            minor: bool = False,
            cyclic_constraint: bool = True,
            allow_numer: bool = False,
            verbose: bool = True
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

        try:
            subspace = self.construct_problem(minor = minor, cyclic_constraint = cyclic_constraint, verbose = verbose)
        except:
            if verbose:
                print('Linear system no solution. Please higher the degree by multiplying something %s.'%(
                    'or use the minor term' if not minor else ''
                ))
            return False


        if verbose:
            print('Matrix shape: %s'%(str(
                    {key: '{}/{}'.format(*self.Q[key].shape[::-1]) for key in self._not_none_keys()}
                ).replace("'", '')))

        # Main SOS solver
        sos_result = sdp_solver(
            *subspace,
            self._not_none_keys(), 
            reg = 0, 
            allow_numer = allow_numer, 
            verbose = verbose
        )
        if sos_result is None:
            return False

        self.update(sos_result)
        self.register_S(sos_result['S'], with_M = True)

        self.success = True
        return True


    def S_from_y(self, y: Optional[sp.Matrix] = None) -> Dict[str, sp.Matrix]:
        """
        Given y, compute the symmetric matrices. This is useful when we want to see the
        symbolic representation of the SDP problem.

        This function does not register the result to self.S.
        """
        if y is None:
            y = sp.Matrix([sp.symbols('y_{%d}'%_) for _ in range(self.space.shape[1])])
        elif not isinstance(y, sp.Matrix) or y.shape != (self.space.shape[1], 1):
            raise ValueError('y must be a sympy Matrix of shape (%d, 1).'%self.space.shape[1])

        vecS = self.x0 + self.space * y
        Ss = []
        for split in self.splits:
            S = LowRankHermitian(None, sp.Matrix(vecS[split])).S
            Ss.append(S)

        ret = {}
        for key, S in zip(self._not_none_keys(), Ss):
            ret[key] = S
        return ret


    def register_S(self, S: Dict[str, sp.Matrix], with_M = False) -> Dict[str, sp.Matrix]:
        """
        Given a solution S in the vector form, restore it back to symmetric matrices.
        """
        for key in self._not_none_keys():
            M = self.low_rank[key].construct_from_vector(S[key])
            self.S[key] = M.S
            if with_M:
                self.M[key] = M.M
        return self.S


    def as_solution(self, decompose_method = 'raw', factor_result = True):
        """
        Wrap the matrix form solution to a SolutionSDP object.
        Note that the decomposition of a quadratic form is not unique.
        """
        if not self.success:
            return None
        return create_solution_from_M(
            poly = self.poly, 
            M = self.M,
            Q = self.Q,
            decompositions = self.decompositions,
            method = decompose_method,
            factor_result = factor_result,
        )



def SDPSOS(
        poly: sp.Poly,
        minor: Union[List[bool], bool] = [False, True],
        degree_limit: int = 12,
        cyclic_constraint = True,
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