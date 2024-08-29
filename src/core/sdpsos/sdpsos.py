from time import time
from typing import Union, Optional, List, Tuple, Dict, Callable

import numpy as np
import sympy as sp
from sympy.combinatorics import Permutation, PermutationGroup, CyclicGroup, SymmetricGroup, AlternatingGroup

from .arithmetic import solve_column_separated_linear
from .manifold import RootSubspace
from .solver import SDPProblem
from .solution import SolutionSDP
from ..solver import homogenize
from ...utils.basis_generator import generate_expr, MonomialReduction, MonomialPerm, MonomialCyclic
from ...utils import arraylize_sp, Coeff, CyclicExpr


def _define_mapping(nvars: int, degree: int, monomials: List[Tuple[int, ...]], symmetry: MonomialReduction) -> Callable[[int, int], Tuple[int, int]]:
    m = sum(monomials)
    vec = generate_expr(nvars, (degree - m)//2, symmetry=symmetry.base())[1]
    dict_monoms = generate_expr(nvars, degree, symmetry=symmetry)[0]

    def mapping(i: int, j: int) -> Tuple[int, int]:
        monom = tuple(map(sum, zip(monomials, vec[i], vec[j])))
        std_ind = None
        v = 0
        for p in symmetry.permute(monom):
            ind = dict_monoms.get(p)
            if ind is not None:
                std_ind = ind
                v += 1
        return std_ind, v
    return mapping


def _get_monomial_list(nvars: int, d: int, symmetry: MonomialReduction) -> List[Tuple[int, ...]]:
    """
    Get all tuples (a1, ..., an) such that a1 + ... + an = d and every ai is 0 or 1.
    """
    def _get_monomial_list_recur(nvars: int, d: int) -> List[Tuple[int, ...]]:
        if d > nvars:
            return []
        elif nvars == 1:
            return [(d,)]
        else:
            return [(i,) + t for i in range(2) for t in _get_monomial_list_recur(nvars - 1, d - i) if d - i >= 0]
    ls = _get_monomial_list_recur(nvars, d)
    ls = list(filter(lambda x: symmetry.is_standard_monom(x), ls))
    return ls


def _form_sdp(monomials: List[Tuple[int, ...]], nvars: int, degree: int,
                rhs: sp.Matrix, symmetry: MonomialReduction, verbose: bool = False) -> SDPProblem:
    time0 = time()
    matrix_size = len(generate_expr(nvars, degree, symmetry=symmetry)[0])
    splits = {}

    eq_list = [[] for _ in range(matrix_size)]
    cnt = 0
    for monomial in monomials:
        # monomial vectors
        dict_monoms_half, inv_monoms_half = generate_expr(nvars, (degree - sum(monomial))//2, symmetry=symmetry.base())
        vector_size = len(inv_monoms_half)
        splits[str(monomial)] = vector_size

        mapping = _define_mapping(nvars, degree, monomial, symmetry=symmetry)

        # form the equation by coefficients
        for i in range(vector_size):
            for j in range(vector_size):
                # The following is equivalent to: eqmat[std_ind, cnt] = v
                std_ind, v = mapping(i, j)
                eq_list[std_ind].append((cnt, v))
                cnt += 1

    x0_equal_indices = _get_equal_entries(monomials, nvars, degree, symmetry)
    if verbose:
        print(f"Time for constructing coeff equations   : {time() - time0:.6f} seconds.")

    time0 = time()
    x0, space = solve_column_separated_linear(eq_list, rhs, x0_equal_indices, _cols = cnt)
    sdp = SDPProblem.from_full_x0_and_space(x0, space, splits)
    if verbose:
        print(f"Time for solving coefficient equations  : {time() - time0:.6f} seconds. Dof = {sdp.dof}")

    return sdp

def _constrain_nullspace(sdp: SDPProblem, monomials: List[Tuple[int, ...]], nullspaces: Optional[Union[List[sp.Matrix], RootSubspace]],
                         verbose: bool = False) -> SDPProblem:
    # constrain nullspace
    time0 = time()
    sdp = sdp.get_last_child()
    sdp.constrain_zero_diagonals()
    if verbose:
        print(f"Time for constraining zero diagonals    : {time() - time0:.6f} seconds. Dof = {sdp.get_last_child().dof}")

    time0 = time()
    if isinstance(nullspaces, RootSubspace):
        is_real = all(all(_ % 2 == 0 for _ in monomial) for monomial in monomials)
        nullspaces = [nullspaces.nullspace(m, real = is_real) for m in monomials]
    if isinstance(nullspaces, list):
        nullspaces = {str(m): n for m, n in zip(monomials, nullspaces)}
    if verbose:
        print(f"Time for computing nullspace            : {time() - time0:.6f} seconds.")
        time0 = time()

    if nullspaces is not None:
        sdp.constrain_nullspace(nullspaces, to_child=True)

    if verbose:
        print(f"Time for constraining nullspace         : {time() - time0:.6f} seconds. Dof = {sdp.get_last_child().dof}")
        time0 = time()
    return sdp

def _get_equal_entries(monomials: List[Tuple[int, ...]], nvars: int, degree: int, symmetry: MonomialReduction) -> List[List[int]]:
    bias = 0
    equal_entries = []
    for monomial in monomials:
        dict_monoms_half, inv_monoms_half = generate_expr(nvars, (degree - sum(monomial))//2, symmetry=symmetry.base())
        n = len(inv_monoms_half)

        permutes = symmetry.permute(monomial)
        if len(permutes) == 1 or any(p != monomial for p in permutes):
            for i in range(n):
                for j in range(i+1, n):
                    equal_entries.append([i*n+j+bias, j*n+i+bias])
        else:

            for i in range(n):
                m1 = inv_monoms_half[i]
                if not symmetry.is_standard_monom(m1):
                    continue
                for j in range(i, n):
                    m2 = inv_monoms_half[j]
                    s = set((i*n+j+bias, j*n+i+bias))
                    for p1, p2 in zip(symmetry.permute(m1), symmetry.permute(m2)):
                        i2, j2 = dict_monoms_half.get(p1), dict_monoms_half.get(p2)
                        # if i2 is not None and j2 is not None
                        s.add(i2*n+j2+bias)
                        s.add(j2*n+i2+bias)
                    equal_entries.append(list(s))
        bias += n**2
    return equal_entries



def _identify_symmetry(poly: sp.Poly, homogenizer: Optional[sp.Symbol] = None) -> PermutationGroup:
    """
    Identify the symmetry group of the polynomial heuristically.
    It only identifies very simple groups like complete symmetric and cyclic groups.
    TODO: Implement an algorithm to identify all symmetric groups.

    Reference
    ----------
    [1] https://cs.stackexchange.com/questions/64335/how-to-find-the-symmetry-group-of-a-polynomial
    """
    coeff = Coeff(poly)
    nvars = len(poly.gens)
    if coeff.is_symmetric():
        return SymmetricGroup(nvars)
    if coeff.is_cyclic():
        return CyclicGroup(nvars)
    if nvars > 3:
        alt = AlternatingGroup(nvars)
        if coeff.is_symmetric(alt):
            return alt

    if homogenizer is not None and nvars > 2:
        # check the symmetry of the polynomial before homogenization
        a = list(range(1, nvars - 1))
        a.append(0)
        a.append(nvars - 1)
        gen1 = Permutation(a)
        a = list(range(nvars))
        a[0], a[1] = a[1], a[0]
        gen2 = Permutation(a)
        G = PermutationGroup([gen1, gen2])
        if coeff.is_symmetric(G):
            return G

        G = PermutationGroup([gen1])
        if coeff.is_symmetric(G):
            return G

    return PermutationGroup()


class SOSProblem():
    """
    Helper class for SDPSOS. See details at SOSProblem.solve.

    Assume that a polynomial can be written in the form v^T @ M @ v.
    Sometimes there are implicit constraints that M = Q @ S @ Q.T where Q is a rational matrix.
    So we can solve the problem on S first and then restore it back to M.

    To summarize, it is about solving for S >> 0 such that
    eq @ vec(S) = vec(P) where P is determined by the target polynomial.
    """
    def __init__(
        self,
        poly: sp.Poly,
        symmetry: Optional[Union[MonomialReduction, PermutationGroup]] = None,
    ):
        self.poly = poly
        self._nvars = len(poly.gens)
        self._degree = poly.total_degree()

        if symmetry is None:
            symmetry = _identify_symmetry(poly)
        if not isinstance(symmetry, MonomialReduction):
            symmetry = MonomialPerm(symmetry)
        
        self._symmetry: MonomialReduction = MonomialPerm(symmetry) if not isinstance(symmetry, MonomialReduction) else symmetry
        self.manifold = RootSubspace(poly, symmetry=self._symmetry)
        self._sdp: SDPProblem = None

    @property
    def sdp(self) -> SDPProblem:
        """
        Return the SDP problem after rank reduction.
        """
        return self._sdp.get_last_child() if self._sdp is not None else None

    @property
    def dof(self) -> int:
        """
        Return the degree of freedom of the SDP problem after rank reduction.
        """
        return self.sdp.dof if self.sdp is not None else None

    @property
    def S(self) -> sp.Matrix:
        return self.sdp.S if self.sdp is not None else None

    @property
    def y(self) -> sp.Matrix:
        return self.sdp.y if self.sdp is not None else None


    def S_from_y(self, y: Optional[Union[sp.Matrix, np.ndarray, Dict]] = None) -> sp.Matrix:
        """
        Restore the solution to the original polynomial.
        """
        return self.sdp.S_from_y(y)

    def _construct_sdp(
        self,
        monomials: List[Tuple[int, ...]],
        nullspaces: Optional[Union[List[sp.Matrix], RootSubspace]] = None,
        register: bool = True,
        verbose: bool = False,
    ) -> SDPProblem:
        """
        Translate the current SOS problem to SDP problem.
        The problem is to find M1, M2, ..., Mn such that

            Poly = p1*(v1'M1v1) + p2*(v2'M2v2) + ...

        where vi are vectors of monomials like [a^2, b^2, c^2, ab, bc, ca]
        while pi are extra monomials like 1, a, ab or abc.
        M1, M2, ... are symmetric matrices and we want them to be positive semidefinite.

        Parameters
        ----------
        monomials : List[Tuple[int, ...]]
            A list of monomials. Each monomial is a tuple of integers.
            For example, (2, 0, 0) represents a^2 for a three-variable polynomial.
        nullspaces : Optional[List[sp.Matrix]]
            The nullspace for each matrix. If nullspace is None, it is skipped.
        register : bool
            Whether to register the SDP problem to the SOS problem. Default is True.
        verbose : bool
            Whether to print the progress. Default is False.

        Returns
        ----------
        SDPProblem
            The SDP problem to solve.
        """
        degree = self._degree

        for m in monomials:
            if (degree - sum(m)) % 2 != 0:
                raise ValueError(f"Degree of poly ({degree}) minus the degree of monomial {m} is not even.")


        symmetry = self._symmetry

        rhs = arraylize_sp(self.poly, symmetry=symmetry)
        sdp = _form_sdp(monomials, self._nvars, degree, rhs, symmetry, verbose=verbose)

        if register:
            self._sdp = sdp

        # constrain nullspace
        _constrain_nullspace(sdp, monomials, nullspaces, verbose=verbose)

        if verbose:
            sdp.print_graph()

        return sdp


    def _construct_sdp_by_default(
        self,
        monomials: List[Tuple[int, ...]],
        verbose: bool = False,
    ) -> SDPProblem:
        sdp = self._construct_sdp(monomials, nullspaces=self.manifold, verbose=verbose)
        self._sdp = sdp
        return sdp

    def solve(self, **kwargs) -> bool:
        """
        Solve the SOS problem. Keyword arguments are passed to SDPProblem.solve.
        """
        return self._sdp.solve(**kwargs)

    def as_solution(
        self,
        y: Optional[Union[sp.Matrix, np.ndarray, Dict]] = None,
    ) -> SolutionSDP:
        """
        Restore the solution to the original polynomial.
        """
        if y is not None:
            self.sdp.register_y(y)
        decomp = self._sdp.decompositions
        if decomp is None:
            return None
        return SolutionSDP.from_decompositions(self.poly, decomp, self._symmetry)



def SDPSOS(
        poly: sp.Poly,
        monomials_lists: Optional[List[List[Tuple[int, ...]]]] = None,
        degree_limit: int = 12,
        verbose: bool = False,
        method: str = "trivial",
        allow_numer: int = 0,
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
        sos_problem = SOSProblem(poly)
        sos_problem._construct_sdp_by_default(monomial)
        sos_problem.solve(**kwargs)
        solution = sos_problem.as_solution()
    ```

    Parameters
    ----------
    poly : sp.Poly
        Polynomial to be solved.
    monomials_lists : Optional[List[List[Tuple[int, ...]]]
        A list of lists. Each list contain monomials that are treated as nonnegative.
        Leave it None to use the default monomials.
    """
    nvars = len(poly.gens)
    degree = poly.total_degree()
    if degree > degree_limit or degree < 2 or nvars < 1:
        return None
    if not (poly.domain in (sp.ZZ, sp.QQ)):
        return None

    original_poly = poly
    poly, homogenizer = homogenize(poly)
    nvars = len(poly.gens)

    symmetry = MonomialPerm(_identify_symmetry(poly, homogenizer))

    if verbose:
        print(f'SDPSOS nvars = {nvars} degree = {degree}')
        if isinstance(symmetry, MonomialCyclic):
            print('Identified Symmetry = Cyclic Group')
        elif isinstance(symmetry, MonomialPerm):
            print('Identified Symmetry = %s' % str(symmetry.perm_group))

    sos_problem = SOSProblem(poly, symmetry=symmetry)

    if monomials_lists is None:
        monomials_lists_separated = [
            _get_monomial_list(nvars, d, symmetry) for d in range(degree % 2, min(nvars, degree) + 1, 2)
        ]
        monomials_lists = []
        accumulated_monomials = []        
        for monomials in monomials_lists_separated:
            accumulated_monomials.extend(monomials)
            monomials_lists.append(accumulated_monomials.copy())

    for monomials in monomials_lists:
        if verbose:
            print(f"Monomials = {monomials}")
        time0 = time()
        try:
            sdp = sos_problem._construct_sdp_by_default(monomials, verbose = verbose)
            if sos_problem.solve(allow_numer = allow_numer, verbose = verbose, method = method):
                if verbose:
                    print(f"Time for solving SDP{' ':20s}: {time() - time0:.6f} seconds. \033[32mSuccess\033[0m.")
                solution = sos_problem.as_solution()
                if homogenizer is not None:
                    solution = SolutionSDP(
                        problem = original_poly,
                        numerator = solution.solution.xreplace({homogenizer: 1}),
                        is_equal = solution.is_equal
                    )
                return solution
        except Exception as e:
            if verbose:
                print(f"Time for solving SDP{' ':20s}: {time() - time0:.6f} seconds. \033[31mFailed with exceptions\033[0m.")
                print(e)
            continue


    return None