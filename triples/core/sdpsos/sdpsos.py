from collections import defaultdict
from itertools import combinations
from time import time
from typing import Union, Optional, List, Tuple, Dict, Callable, Generator, Any

import numpy as np
import sympy as sp
from sympy import Poly, Expr, ZZ, QQ
from sympy.combinatorics import PermutationGroup
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.sdm import SDM

from .manifold import RootSubspace
from .solution import SolutionSDP
from ..shared import sanitize_input, sanitize_output, identify_symmetry_from_lists, clear_polys_by_symmetry
from ...sdp import SDPProblem
from ...sdp.arithmetic import solve_csr_linear
from ...utils import arraylize_sp, Coeff, CyclicExpr, generate_monoms, MonomialManager, Root, optimize_poly


def _define_mapping(nvars: int, degree: int, monomial: Tuple[int, ...], symmetry: MonomialManager, half: bool = True) -> Callable[[int, int], Tuple[int, int]]:
    """
    Given a gram matrix of standard monomials and a permutation group, we want
    to know how the entry a_{ij} contributes to the monomial of the product of two monomials.
    The function returns a mapping from (i, j) to (std_ind, v) where std_ind is the index of the monomial
    and v is the multiplicity of the monomial in the permutation group.
    """
    m = sum(monomial)
    codegree = (degree - m)//2 if half else degree - m
    vec = generate_monoms(nvars, codegree, symmetry=symmetry.base())[1]
    dict_monoms = generate_monoms(nvars, degree, symmetry=symmetry)[0]

    if half:
        def mapping(i: int, j: int) -> Tuple[int, int]:
            monom = tuple(map(sum, zip(monomial, vec[i], vec[j])))
            std_ind = None
            v = 0
            for p in symmetry.permute(monom):
                ind = dict_monoms.get(p)
                if ind is not None:
                    std_ind = ind
                    v += 1
            return std_ind, v
    else:
        def mapping(i: int) -> Tuple[int, int]:
            monom = tuple(ui + vi for ui, vi in zip(monomial, vec[i]))
            std_ind = None
            v = 0
            for p in symmetry.permute(monom):
                ind = dict_monoms.get(p)
                if ind is not None:
                    std_ind = ind
                    v += 1
            return std_ind, v
    return mapping


def _get_monomial_list(nvars: int, d: int, symmetry: MonomialManager) -> List[Tuple[int, ...]]:
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


def _form_sdp(ineq_constraints: List[Poly], eq_constraints: List[Poly], nvars: int, degree: int,
                rhs: sp.Matrix, symmetry: MonomialManager, verbose: bool = False) -> SDPProblem:
    time0 = time()
    matrix_size = len(generate_monoms(nvars, degree, symmetry=symmetry)[0])
    get_inv_half = lambda d: generate_monoms(nvars, d, symmetry=symmetry.base())[1]
    splits_ineq = {}
    splits_eq = {}

    domain = ZZ
    for ineq in ineq_constraints:
        domain = domain.unify(ineq.domain)
    for eq in eq_constraints:
        domain = domain.unify(eq.domain)

    # CSR (SDM) format, each column is the contribution of an entry to the poly
    eq_list = [[] for _ in range(matrix_size)]
    cnt = 0
    for ineq in ineq_constraints:
        d = ineq.total_degree()
        if d > degree or (degree - d) % 2 != 0:
            raise ValueError("The degree of the inequality constraint is invalid, received %d while the total degree is %d." % (d, degree))

        rep = ineq.rep.convert(domain)
        cur_cnt = cnt
        contribution = defaultdict(lambda : domain.zero)
        for monomial, const in rep.terms():
            cnt = cur_cnt
            inv_monoms_half = get_inv_half((degree - d)//2)
            vector_size = len(inv_monoms_half)
            splits_ineq[ineq] = vector_size

            mapping = _define_mapping(nvars, degree, monomial, symmetry)

            # form the equation by coefficients
            for i in range(vector_size):
                for j in range(vector_size):
                    # The following is equivalent to: eqmat[std_ind, cnt] += v*const
                    std_ind, v = mapping(i, j)
                    contribution[(std_ind, cnt)] += v*const
                    cnt += 1
        for (row, col), v in contribution.items():
            eq_list[row].append((col, v))

    ############################################
    #  Construct the equality constraints
    ############################################
    # For equality constraints we do not need PSD vars, but simply a vector of unbounded vars
    for eq in eq_constraints:
        d = eq.total_degree()
        if d > degree:
            continue
        
        inv_monoms_nonhalf = generate_monoms(nvars, degree - d, symmetry=symmetry.base())[1]
        vector_size = len(inv_monoms_nonhalf)
        splits_eq[eq] = vector_size

        rep = eq.rep.convert(domain)
        cur_cnt = cnt
        contribution = defaultdict(lambda : domain.zero)
        for monomial, const in rep.terms():
            cnt = cur_cnt
            mapping = _define_mapping(nvars, degree, monomial, symmetry, half=False)

            # form the equation by coefficients
            for i in range(vector_size):
                # The following is equivalent to: eqmat[std_ind, cnt] += v*const
                std_ind, v = mapping(i)
                contribution[(std_ind, cnt)] += v*const
                cnt += 1
        for (row, col), v in contribution.items():
            eq_list[row].append((col, v))

    eq_list = dict((i, dict(row)) for i, row in enumerate(eq_list))
    eq_list = sp.Matrix._fromrep(DomainMatrix.from_rep(SDM(eq_list, (matrix_size, cnt), domain)))

    x0_equal_indices = _get_equal_entries(ineq_constraints, eq_constraints, nvars, degree, symmetry)
    if verbose:
        print(f"Time for constructing coeff equations   : {time() - time0:.6f} seconds.")
        time0 = time()

    # Term sparsity:
    # diagonal entries of the PSD vars should be nonnegative
    # if a diagonal entry of a PSD var is zero, then the whole row is zero
    nonnegative_indices = []
    force_zeros = {}
    offset = 0
    for n in splits_ineq.values():
        for i, i0 in enumerate(range(offset, offset+n**2, n+1)):
            nonnegative_indices.append(i0)
            force_zeros[i0] = list(range(i0-i, i0-i+n))
        offset += n**2

    x0, space = solve_csr_linear(eq_list, rhs, x0_equal_indices,
                    nonnegative_indices=nonnegative_indices, force_zeros=force_zeros)
    splits_size = sum(n**2 for n in splits_ineq.values())
    sdp = SDPProblem.from_full_x0_and_space(x0[:splits_size,:], space[:splits_size,:],
            splits_ineq, constrain_symmetry=False)
    if verbose:
        print(f"Time for solving coefficient equations  : {time() - time0:.6f} seconds. Dof = {sdp.dof}")

    eq_vec = {}
    cnt = sum(n**2 for n in splits_ineq.values())
    for eq in eq_constraints:
        s = splits_eq[eq]
        eq_vec[eq] = (x0[cnt:cnt+s, :], space[cnt:cnt+s, :])
        cnt += s
    return sdp, eq_vec

def _get_equal_entries(ineq_constraints: List[Poly], eq_constraints: List[Poly],
        nvars: int, degree: int, symmetry: MonomialManager) -> List[List[int]]:
    offset = 0
    equal_entries = []
    perm_group = symmetry.perm_group
    for ineq in ineq_constraints:
        d = ineq.total_degree()
        if d > degree or (degree - d) % 2 != 0:
            continue

        dict_monoms_half, inv_monoms_half = generate_monoms(nvars, (degree - d)//2, symmetry=symmetry.base())
        n = len(inv_monoms_half)

        if not Coeff(ineq).is_symmetric(perm_group):
            for i in range(n):
                for j in range(i+1, n):
                    equal_entries.append([i*n+j+offset, j*n+i+offset])
        else:

            for i in range(n):
                m1 = inv_monoms_half[i]
                if not symmetry.is_standard_monom(m1):
                    continue
                for j in range(i, n):
                    m2 = inv_monoms_half[j]
                    s = set((i*n+j+offset, j*n+i+offset))
                    for p1, p2 in zip(symmetry.permute(m1), symmetry.permute(m2)):
                        i2, j2 = dict_monoms_half.get(p1), dict_monoms_half.get(p2)
                        # if i2 is not None and j2 is not None
                        s.add(i2*n+j2+offset)
                        s.add(j2*n+i2+offset)
                    equal_entries.append(list(s))
        offset += n**2

    for eq in eq_constraints:
        d = eq.total_degree()
        if d > degree:
            continue

        dict_monoms_nonhalf, inv_monoms_nonhalf = generate_monoms(nvars, degree - d, symmetry=symmetry.base())
        n = len(inv_monoms_nonhalf)

        if Coeff(eq).is_symmetric(perm_group):
            for i in range(n):
                m1 = inv_monoms_nonhalf[i]
                if not symmetry.is_standard_monom(m1):
                    continue
                s = set((i+offset,))
                for p1 in symmetry.permute(m1):
                    i2 = dict_monoms_nonhalf.get(p1)
                    s.add(i2+offset)
                equal_entries.append(list(s))
        offset += n

    return equal_entries

def _constrain_nullspace(sdp: SDPProblem, ineq_constraints: List[Poly], eq_constraints: List[Poly],
        nullspaces: Optional[Union[List[sp.Matrix], RootSubspace]], verbose: bool = False) -> SDPProblem:
    # constrain nullspace
    time0 = time()
    sdp = sdp.get_last_child()
    sdp.constrain_zero_diagonals()
    if verbose:
        print(f"Time for constraining zero diagonals    : {time() - time0:.6f} seconds. Dof = {sdp.get_last_child().dof}")

    time0 = time()
    if isinstance(nullspaces, RootSubspace):
        nullspaces = [nullspaces.nullspace(ineq, ineq_constraints, eq_constraints) for ineq in ineq_constraints]
    if isinstance(nullspaces, list):
        nullspaces = {ineq: n for ineq, n in zip(ineq_constraints, nullspaces)}
    if verbose:
        print(f"Time for computing nullspace            : {time() - time0:.6f} seconds.")
        time0 = time()

    # if deparametrize:
    #     sdp.deparametrize()

    if nullspaces is not None:
        sdp.constrain_nullspace(nullspaces, to_child=True)

    if verbose:
        print(f"Time for constraining nullspace         : {time() - time0:.6f} seconds. Dof = {sdp.get_last_child().dof}")
        time0 = time()

    return sdp


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
        poly: Poly,
        symmetry: Optional[Union[MonomialManager, PermutationGroup]] = None,
        # is_homogeneous: Optional[bool] = None,
        _check_symmetry: bool = True,
    ):
        """
        Construct the SOS problem.

        Parameters
        ----------
        poly : Poly
            The polynomial to perform SOS on.
        symmetry : PermutationGroup or MonomialManager
            The symmetry of the problem. It can be a sympy PermutationGroup object.        
        """
        self.poly = poly
        self._nvars = len(poly.gens)
        self._degree = poly.total_degree()

        is_homogeneous = True
        if is_homogeneous is None:
            is_homogeneous = poly.is_homogeneous
        elif is_homogeneous and not poly.is_homogeneous:
            raise ValueError("The polynomial is not homogeneous.")

        # if symmetry is None:
        #     symmetry = identify_symmetry_from_lists([[poly]])
        symmetry = MonomialManager(self._nvars, symmetry, is_homogeneous=is_homogeneous)

        if _check_symmetry and not Coeff(poly).is_symmetric(symmetry.perm_group):
            raise ValueError("The polynomial is not symmetric with respect to the symmetry group.")

        self._symmetry: MonomialManager = symmetry
        self.manifold = RootSubspace(poly, symmetry=self._symmetry)
        self._sdp: SDPProblem = None
        self._eqvec: Dict[Poly, Tuple[sp.Matrix, sp.Matrix]] = {}

    @property
    def is_homogeneous(self) -> bool:
        return self._symmetry.is_homogeneous

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

    def construct(
        self,
        ineq_constraints: List[Union[Poly, Expr]] = [],
        eq_constraints: List[Union[Poly, Expr]] = [],
        roots: Optional[List[Root]] = None,
        deparametrize: bool = True,
        verbose: bool = False,
    ) -> SDPProblem:
        """
        Construct the SDP problem with additional constraints.

        Parameters
        ----------
        ineq_constraints : List[Union[Poly, Expr]]
            Inequality constraints.
        eq_constraints : List[Union[Poly, Expr]]
            Equality constraints.
        roots : List[Root]
            The roots of the polynomial. Default is None.
        deparametrize : bool
            Whether to deparametrize the SDP problem if there
            exists linear free symbols in the coefficients. Default is True.
        verbose : bool
            Whether to print the progress. Default is False.

        Returns
        ----------
        SDPProblem
            The SDP problem to solve.
        """
        gens = self.poly.gens
        ineq_constraints = [Poly(ineq, *gens) for ineq in ineq_constraints]
        eq_constraints = [Poly(eq, *gens) for eq in eq_constraints]
        if self.is_homogeneous and (not all(eq.is_homogeneous for eq in eq_constraints) 
                            or not all(ineq.is_homogeneous for ineq in ineq_constraints)):
            raise ValueError("Expecting all constraints to be homogeneous.")
                            #  "Set is_homogeneous=False in initialization to allow non-homogeneous constraints.")

        degree = self._degree
        symmetry = self._symmetry
        rhs = arraylize_sp(self.poly, symmetry=symmetry)

        sdp, eqvec = _form_sdp(ineq_constraints, eq_constraints, self._nvars, degree, rhs, symmetry, verbose=verbose)
        self._sdp = sdp
        if deparametrize:
            sdp = sdp.deparametrize()
            sdp.clear_parents()
        self._sdp = sdp
        self._eqvec = eqvec

        if roots is None:
            if self.poly.domain.is_ZZ or self.poly.domain.is_QQ:
                roots = optimize_poly(self.poly, ineq_constraints, eq_constraints + [self.poly], return_type='root')
            else:
                # TODO: clean this
                from ..sdpsos.manifold import _findroot_binary
                roots = _findroot_binary(self.poly, symmetry=self._symmetry)
        self.manifold.roots = roots
        _constrain_nullspace(sdp, ineq_constraints, eq_constraints, self.manifold, verbose=verbose)

        if verbose:
            sdp.print_graph()

        return sdp

    def solve_obj(self, *args, **kwargs) -> Optional[sp.Matrix]:
        return self._sdp.solve_obj(*args, **kwargs)

    def solve(self, *args, **kwargs) -> Optional[sp.Matrix]:
        """
        Solve the SOS problem. Keyword arguments are passed to SDPProblem.solve.
        """
        return self._sdp.solve(*args, **kwargs)

    def as_solution(
        self,
        y: Optional[Union[sp.Matrix, np.ndarray, Dict]] = None,
        ineq_constraints: Optional[Dict[Poly, Expr]] = None,
        eq_constraints: Optional[Dict[Poly, Expr]] = None
    ) -> SolutionSDP:
        """
        Retrieve the solution to the original polynomial.

        Parameters
        ----------
        y: Optional[Union[sp.Matrix, np.ndarray, Dict]]
            The solution to the dual problem. Defaultedly it uses the solution from its own SDP problem.
            When specified, it will use the given solution.
        ineq_constraints: Optional[Dict[Poly, Expr]]
            Conversion of polynomial inequality constraints to sympy expression forms.
        eq_constraints: Optional[Dict[Poly, Expr]]
            Conversion of polynomial equality constraints to sympy expression forms.
        """
        if y is not None:
            self.sdp.register_y(y)
        y = self.sdp.y
        decomp = self._sdp.decompositions
        if decomp is None:
            return None
        eqvec = {eq: x + space * self._sdp.y for eq, (x, space) in self._eqvec.items()}
        return SolutionSDP.from_decompositions(self.poly, decomp, eqvec, self._symmetry,
                                ineq_constraints=ineq_constraints, eq_constraints=eq_constraints)


def _get_qmodule_list(poly: Poly, ineq_constraints: List[Tuple[Poly, Expr]],
        ineq_constraints_with_trivial: bool = True, preordering: str = 'linear-progressive') -> Generator[List[Tuple[Poly, Expr]], None, None]:
    _ACCEPTED_PREORDERINGS = ['none', 'linear', 'linear-progressive']
    if not preordering in _ACCEPTED_PREORDERINGS:
        raise ValueError("Invalid preordering method, expected one of %s, received %s." % (str(_ACCEPTED_PREORDERINGS), preordering))

    degree = poly.total_degree()
    poly_one = Poly(1, *poly.gens)

    def degree_filter(polys_and_exprs):
        return [pe for pe in polys_and_exprs if pe[0].total_degree() <= degree \
                    and (degree - pe[0].total_degree()) % 2 == 0]

    if preordering == 'none':
        if ineq_constraints_with_trivial:
            ineq_constraints = [(poly_one, sp.S.One)] + ineq_constraints
        ineq_constraints = degree_filter(ineq_constraints)
        yield ineq_constraints
        return

    linear_ineqs = []
    nonlin_ineqs = [(poly_one, sp.S.One)]
    for ineq, e in ineq_constraints:
        if ineq.is_linear:
            linear_ineqs.append((ineq, e))
        else:
            nonlin_ineqs.append((ineq, e))

    qmodule = nonlin_ineqs if ineq_constraints_with_trivial else nonlin_ineqs[1:]
    qmodule = degree_filter(qmodule)
    if 'progressive' in preordering:
        yield qmodule.copy()
    for n in range(1, len(linear_ineqs) + 1):
        has_additional = False
        for comb in combinations(linear_ineqs, n):
            mul = poly_one
            for c in comb:
                mul = mul * c[0]
            d = mul.total_degree()
            if d > degree:
                continue
            mul_expr = sp.Mul(*(c[1] for c in comb))
            for ineq, e in nonlin_ineqs:
                new_d = d + ineq.total_degree()
                if new_d <= degree and (degree - new_d) % 2 == 0:
                    qmodule.append((mul * ineq, mul_expr * e))
                    has_additional = True

        if has_additional and 'progressive' in preordering:
            yield qmodule.copy()
    if 'progressive' not in preordering:
        # yield them all at once
        yield qmodule


@sanitize_output()
@sanitize_input(homogenize=True, infer_symmetry=True, wrap_constraints=True)
def SDPSOS(
        poly: Poly,
        ineq_constraints: Union[List[Expr], Dict[Expr, Expr]] = {},
        eq_constraints: Union[List[Expr], Dict[Expr, Expr]] = {},
        symmetry: Optional[Union[MonomialManager, PermutationGroup]] = None,
        ineq_constraints_with_trivial: bool = True,
        preordering: str = 'linear-progressive',
        degree_limit: int = 12,
        verbose: bool = False,
        solver: Optional[str] = None,
        allow_numer: int = 0,
        solve_kwargs: Dict[str, Any] = {},
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
        sos_problem.construct(*args)
        sos_problem.solve(**kwargs)
        solution = sos_problem.as_solution()
    ```

    Parameters
    ----------
    poly: Poly
        The polynomial to perform SOS on.
    ineq_constraints: List[Poly]
        Inequality constraints to the problem. This assumes g_1(x) >= 0, g_2(x) >= 0, ...
        This is used to generate the quadratic module.
    eq_constraints: List[Poly]
        Equality constraints to the problem. This assumes h_1(x) = 0, h_2(x) = 0, ...
        This is used to generate the ideal (quotient ring).
    symmetry: PermutationGroup or MonomialManager
        The symmetry of the polynomial. When it is None, it will be automatically generated. 
        If we want to skip the symmetry generation algorithm, please pass in a MonomialManager object.
    ineq_constraints_with_trivial: bool
        Whether to add the trivial inequality constraint 1 >= 0. This is used to generate the
        quadratic module. Default is True.
    preordering: str
        The preordering method for extending the generators of the quadratic module. It can be
        'none', 'linear', 'linear-progressive'. Default is 'linear-progressive'.
    degree_limit: int
        The maximum degree of the polynomial to be considered. Default is 12.
    verbose: bool
        Whether to print the progress. Default is False.
    solver: str
        The numerical SDP solver to use. When set to None, it is automatically selected. Default is None.
    allow_numer: int
        Whether to allow numerical solution (still under development).
    """
    return _SDPSOS(poly, ineq_constraints=ineq_constraints, eq_constraints=eq_constraints,
                symmetry=symmetry, ineq_constraints_with_trivial=ineq_constraints_with_trivial,
                preordering=preordering, degree_limit=degree_limit, verbose=verbose,
                solver=solver, allow_numer=allow_numer, solve_kwargs=solve_kwargs)


def _SDPSOS(
        poly: Poly,
        ineq_constraints: Dict[Poly, Expr] = {},
        eq_constraints: Dict[Poly, Expr] = {},
        symmetry: MonomialManager = None,
        ineq_constraints_with_trivial: bool = True,
        preordering: str = 'linear-progressive',
        degree_limit: int = 12,
        verbose: bool = False,
        solver: Optional[str] = None,
        allow_numer: int = 0,
        solve_kwargs: Dict[str, Any] = {},
    ) -> Optional[SolutionSDP]:
    nvars = len(poly.gens)
    degree = poly.total_degree()
    if degree > degree_limit or degree < 1 or nvars < 1:
        return None
    if not (poly.domain in (sp.ZZ, sp.QQ)):
        return None

    if verbose:
        print(f'SDPSOS nvars = {nvars} degree = {degree}')
        print('Identified Symmetry = %s' % str(symmetry.perm_group).replace('\n', '').replace('  ',''))

    sos_problem = SOSProblem(poly, symmetry=symmetry)
    roots = None
    qmodule_list = _get_qmodule_list(poly, ineq_constraints.items(),
                        ineq_constraints_with_trivial=ineq_constraints_with_trivial, preordering=preordering)

    # odd_degree_vars = [i for i in range(nvars) if poly.degree(i) % 2 == 1]
    for qmodule in qmodule_list:
        qmodule = clear_polys_by_symmetry(qmodule, poly.gens, symmetry)

        if len(qmodule) == 0 and len(eq_constraints) == 0:
            continue
        # if the poly has odd degree on some var, but all monomials are even up to permutation,
        # then the poly is not SOS
        # unhandled_odd = len(odd_degree_vars) > 0 # init to True if there is any odd degree var
        # for i in odd_degree_vars:
        #     for i2 in symmetry.to_perm_group(nvars).orbit(i):
        #         if any(m[i2] % 2 == 1 for m in qmodule):
        #             unhandled_odd = False
        #             break
        #     if unhandled_odd:
        #         break
        # if unhandled_odd:
        #     continue

        if verbose:
            print(f"Qmodule = {[e[0] for e in qmodule]}\nIdeal   = {list(eq_constraints.keys())}")
        time0 = time()
        # now we solve the problem
        try:
            if roots is None:
                time1 = time()
                roots = optimize_poly(poly, list(ineq_constraints.keys()), [poly] + list(eq_constraints.keys()), return_type='root')
                if verbose:
                    print(f"Time for finding roots                  : {time() - time1:.6f} seconds.")
            
            sdp = sos_problem.construct([e[0] for e in qmodule], list(eq_constraints.keys()), roots=roots, verbose=verbose)
            if sos_problem.solve(verbose=verbose, solver=solver, allow_numer=allow_numer, kwargs=solve_kwargs) is not None:
                if verbose:
                    print(f"Time for solving SDP{' ':20s}: {time() - time0:.6f} seconds. \033[32mSuccess\033[0m.")
                solution = sos_problem.as_solution(ineq_constraints=dict(qmodule), eq_constraints=eq_constraints)
                return solution
        except Exception as e:
            if verbose:
                print(f"Time for solving SDP{' ':20s}: {time() - time0:.6f} seconds. \033[31mFailed with exceptions\033[0m.")
                print(e)
            continue


    return None