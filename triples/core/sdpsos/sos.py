from collections import defaultdict
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
from time import time
# from warnings import warn

from numpy import ndarray
from sympy.combinatorics import PermutationGroup
from sympy import Poly, Expr, Symbol, Mul, ZZ
from sympy.matrices import MutableDenseMatrix as Matrix

from .manifold import get_nullspace
from .solution import SolutionSDP
from ...utils import CyclicSum, verify_symmetry, MonomialManager, Root, optimize_poly
from ...sdp import SDPProblem


def _constrain_root_nullspace(sdp: SDPProblem, poly: Poly, ineq_constraints: Dict, eq_constraints: Dict,
        ineq_bases: Dict[Any, Any], eq_bases: Dict[Any, Any], degree: int,
        roots: Optional[List[Root]]=None, verbose: bool = False
    ) -> Tuple[SDPProblem, List[Root]]:
    # constrain nullspace
    # sdp = sdp.get_last_child()

    time0 = time()
    if roots is None:
        all_polys = list(ineq_constraints.values()) + list(eq_constraints.values()) + [poly]
        if all(p.domain.is_ZZ or p.domain.is_QQ for p in all_polys):
            roots = optimize_poly(poly,
                list(ineq_constraints.values()), list(eq_constraints.values()) + [poly], return_type='root')
        else:
            # TODO: clean this
            from ..sdpsos.manifold import _findroot_binary
            roots = _findroot_binary(poly)# symmetry=self._symmetry)
        if verbose:
            print(f"Time for finding roots num = {len(roots):<6d}     : {time() - time0:.6f} seconds.")
            time0 = time()
    else:
        roots = [Root(_) if not isinstance(_, Root) else _ for _ in roots]


    time0 = time()
    nullspaces = get_nullspace(poly, ineq_constraints, eq_constraints, ineq_bases, eq_bases,
                        degree=degree, roots=roots)
    if verbose:
        print(f"Time for computing nullspace            : {time() - time0:.6f} seconds.")
        time0 = time()

    new_sdp = sdp.constrain_nullspace(nullspaces, to_child=True)

    if verbose:
        print(f"Time for constraining nullspace         : {time() - time0:.6f} seconds. Dof = {sdp.get_last_child().dof}")
        time0 = time()

    return new_sdp, roots

def _define_mapping_psd(ineq: Poly, ineq_basis: MonomialManager, full_monomial_manager: MonomialManager,
        degree: int) -> Callable[[int, int], List[Tuple[int, Expr]]]:
    """
    Given a gram matrix of standard monomials and a permutation group, we want
    to know how the entry a_{ij} contributes to the monomial of the product of two monomials.
    The function returns a mapping from (i, j) to (std_ind, v) where std_ind is the index of the monomial
    and v is the multiplicity of the monomial in the permutation group.
    """
    ineqterms = ineq.rep.terms()
    var_terms = ineq_basis.inv_monoms((degree - ineq.total_degree())//2)
    dict_monoms = full_monomial_manager.dict_monoms(degree)
    zero, one = ineq.domain.zero, ineq.domain.one
    monom_add, permute = full_monomial_manager.add, full_monomial_manager.permute
    def mapping(i: int, j: int) -> Tuple[int, int]:
        vec = defaultdict(lambda: zero)
        m2, m3 = var_terms[i], var_terms[j]
        m2m3 = monom_add(m2, m3)
        for m1, v1 in ineqterms:
            monom = monom_add(m1, m2m3) # TODO: incorrect for nc polys
            std_ind = None
            v = zero
            for p in permute(monom):
                ind = dict_monoms.get(p)
                if ind is not None:
                    std_ind = ind
                    v += one
            # If std_ind is None, then the addition of monomials is not in
            # the monomial list of poly, which will cause an error when forming the matrix
            if std_ind is None:
                raise ValueError(f"Product of monomials {m1}, {m2} and {m3}"
                                    " is not expected in the monomial list of poly.")
            vec[std_ind] += v1*v
        return [(k, v) for k, v in vec.items() if v]
    return mapping

def _define_mapping_linear(eq: Poly, eq_basis: MonomialManager, full_monomial_manager: MonomialManager,
        degree: int) -> Callable[[int], List[Tuple[int, Expr]]]:
    eqterms = eq.rep.terms()
    var_terms = eq_basis.inv_monoms(degree - eq.total_degree())
    dict_monoms = full_monomial_manager.dict_monoms(degree)
    zero, one = eq.domain.zero, eq.domain.one
    monom_add, permute = full_monomial_manager.add, full_monomial_manager.permute
    def mapping(i: int) -> Tuple[int, int]:
        vec = defaultdict(lambda: zero)
        m2 = var_terms[i]
        for m1, v1 in eqterms:
            monom = monom_add(m1, m2)
            std_ind = None
            v = zero
            for p in permute(monom):
                ind = dict_monoms.get(p)
                if ind is not None:
                    std_ind = ind
                    v += one
            # If std_ind is None, then the addition of monomials is not in
            # the monomial list of poly, which will cause an error when forming the matrix
            if std_ind is None:
                raise ValueError(f"Product of monomials {m1} and {m2}"
                                    " is not expected in the monomial list of poly.")
            vec[std_ind] += v1*v
        return [(k, v) for k, v in vec.items() if v]
    return mapping


def _get_equal_entries(symmetry: MonomialManager, degree: int,
        ineq_constraints: Dict[Any, Poly], eq_constraints: Dict[Any, Poly],
        ineq_bases: Dict[Any, List[Tuple[int, ...]]], eq_bases: Dict[Any, List[Tuple[int, ...]]]) -> List[List[int]]:
    if symmetry.is_trivial:
        return []
    offset = 0
    equal_entries = []
    perm_group = symmetry.perm_group
    for key, ineq in ineq_constraints.items():
        codegree = (degree - ineq.total_degree())//2
        basis_dict = ineq_bases[key].dict_monoms(codegree)
        basis = ineq_bases[key].inv_monoms(codegree)
        n = len(basis)
        if not all(isinstance(_, (tuple, list)) for _ in basis):
            offset += n**2
            continue

        if verify_symmetry(ineq, perm_group):
            # TODO: it does not need to be fully symmetric,
            # partially symmetric is also acceptable
            for i in range(n):
                m1 = basis[i]
                if not symmetry.is_standard_monom(m1):
                    continue
                for j in range(i, n):
                    m2 = basis[j]
                    s = set((i*n+j+offset, j*n+i+offset))
                    for p1, p2 in zip(symmetry.permute(m1), symmetry.permute(m2)):
                        i2, j2 = basis_dict.get(p1), basis_dict.get(p2)
                        # if i2 is not None and j2 is not None
                        s.add(i2*n+j2+offset)
                        # s.add(j2*n+i2+offset)
                    equal_entries.append(list(s))
        offset += n**2

    for key, eq in eq_constraints.items():
        codegree = (degree - eq.total_degree())
        basis_dict = eq_bases[key].dict_monoms(codegree)
        basis = eq_bases[key].inv_monoms(codegree)
        n = len(basis)
        if not all(isinstance(_, (tuple, list)) for _ in basis):
            offset += n
            continue

        if verify_symmetry(eq, perm_group):
            for i in range(n):
                m1 = basis[i]
                if not symmetry.is_standard_monom(m1):
                    continue
                s = set((i+offset,))
                for p1 in symmetry.permute(m1):
                    i2 = basis_dict.get(p1)
                    s.add(i2+offset)
                equal_entries.append(list(s))
        offset += n

    return equal_entries


class SOSProblem():
    """
    Helper class for SDPSOS. See details at SOSProblem.solve.

    Assume that a polynomial can be written in the form v^T @ M @ v.
    Sometimes there are implicit constraints that M = Q @ S @ Q.T where Q is a rational matrix.
    So we can solve the problem on S first and then restore it back to M.

    To summarize, it is about solving for S >> 0 such that
    eq @ vec(S) = vec(P) where P is determined by the target polynomial.
    """
    poly: Poly
    gens: List[Symbol]
    _degree: int
    _symmetry: MonomialManager

    ineq_constraints: Dict[Any, Poly]
    eq_constraints: Dict[Any, Poly]
    _ineq_bases: Dict[Any, MonomialManager]
    _eq_bases: Dict[Any, MonomialManager]
    _ineq_codegrees: Dict[Any, int]
    _eq_codegrees: Dict[Any, int]

    _sdp: SDPProblem
    _eq_space: Dict[Any, Tuple[Matrix, Matrix]]
    _roots: List[Root]


    def __init__(self, poly: Poly, gens: Optional[Tuple[Symbol, ...]]=None):
        """
        Construct the SOS problem.

        Parameters
        ----------
        poly : Poly
            The polynomial to perform SOS on.
        gens : Optional[Tuple[Symbol,...]]
            The generators of the polynomial. Inferred from the polynomial if not specified.
        """
        if gens is None:
            if not isinstance(poly, Poly):
                raise ValueError("Generators must be specified when the polynomial is not a Poly object.")
            gens = poly.gens
        else:
            poly = Poly(poly, gens)
        self.poly = poly
        self.gens = gens

    @property
    def sdp(self) -> SDPProblem:
        """
        Return the root node of the constructed SDP problem.
        """
        return self._sdp

    @property
    def sdpp(self) -> SDPProblem:
        """
        Return the child node of the SDP problem.
        """
        return self.sdp.get_last_child() if self.sdp is not None else None

    def solve_obj(self, *args, **kwargs) -> Optional[Matrix]:
        """
        Solve the SOS problem. Arguments are passed to SDPProblem.solve.
        """
        return self.sdp.solve_obj(*args, **kwargs)

    def solve(self, *args, **kwargs) -> Optional[Matrix]:
        """
        Solve the SOS problem. Arguments are passed to SDPProblem.solve.
        """
        return self.sdp.solve(*args, **kwargs)

    def ineq_bases(self) -> Dict[Any, Matrix]:
        """
        Get the bases associated with each inequality constraint.
        """
        bases, gens = {}, self.gens
        for key, basis in self._ineq_bases.items():
            # TODO: what if the bases are not monomial tuples?
            basis = basis.inv_monoms(self._ineq_codegrees[key])
            bases[key] = [Mul(*(c**i for c, i in zip(gens, b))) for b in basis]
        return bases

    def eq_bases(self) -> Dict[Any, Matrix]:
        """
        Get the bases associated with each equality constraint.
        """
        bases, gens = {}, self.gens
        for key, basis in self._ineq_bases.items():
            basis = basis.inv_monoms(self._eq_codegrees[key])
            bases[key] = [Mul(*(c**i for c, i in zip(gens, b))) for b in basis]
        return bases


    def as_solution(
        self,
        ineq_constraints: Optional[Dict[Any, Expr]] = None,
        eq_constraints: Optional[Dict[Any, Expr]] = None,
        adjoint_operator: Optional[Callable[[Expr], Expr]] = None,
        trace_operator: Optional[Callable[[Expr], Expr]] = None,
    ) -> SolutionSDP:
        """
        Retrieve the solution to the original polynomial.

        Parameters
        ----------
        ineq_constraints: Optional[Dict[Any, Expr]]
            Conversion of polynomial inequality constraints to sympy expression forms.
        eq_constraints: Optional[Dict[Any, Expr]]
            Conversion of polynomial equality constraints to sympy expression forms.
        """
        decomp = self.sdp.decompositions
        if decomp is None:
            raise ValueError("The problem has not been solved yet.")
        if ineq_constraints is None:
            ineq_constraints = self.ineq_constraints
        if eq_constraints is None:
            eq_constraints = self.eq_constraints
        eqspace = {eq: x + space * self.sdp.y for eq, (x, space) in self._eq_space.items()}

        solution = SolutionSDP.from_decompositions(self.poly, decomp, eqspace,
            ineq_constraints = ineq_constraints,
            ineq_bases       = self._ineq_bases,
            ineq_codegrees   = self._ineq_codegrees,
            eq_constraints   = eq_constraints,
            eq_bases         = self._eq_bases,
            eq_codegrees     = self._eq_codegrees,
            cyclic_sum       = (lambda x: CyclicSum(x, self.gens, self._symmetry))\
                                    if self._symmetry is not None else (lambda x: x),
            adjoint_operator = adjoint_operator,
            trace_operator   = trace_operator,
        )

        # overwrite the constraints information in the form of dict((Poly, Expr))
        solution.ineq_constraints = {self.ineq_constraints[key]: expr for key, expr in ineq_constraints.items()}
        solution.eq_constraints = {self.eq_constraints[key]: expr for key, expr in eq_constraints.items()}
        return solution


    def construct(
        self,
        ineq_constraints: List[Union[Poly, Expr]] = [],
        eq_constraints: List[Union[Poly, Expr]] = [],
        symmetry: Optional[PermutationGroup] = None,
        degree: Optional[int] = None,
        roots: Optional[List[Root]] = None,
        term_sparsity: int = 1,
        deparametrize: bool = True,
        verbose: bool = False,
        ineq_bases: Optional[List[Tuple[int, ...]]] = None,
        eq_bases: Optional[List[Tuple[int, ...]]] = None,
    ) -> SDPProblem:
        """
        Construct the SDP problem given inequality and equality constraints.

        Parameters
        ----------
        ineq_constraints : List[Union[Poly, Expr]]
            List or dict of polynomial or sympy expression inequality constraints,
            G1, G2, ..., Gn >= 0.
            Used as the generators of quadratic modules of the SOS problem.
        eq_constraints : List[Union[Poly, Expr]]
            List or dict of polynomial or sympy expression equality constraints,
            H1, H2,..., Hm = 0.
            Used as the generators of the quotient ring / ideal of the SOS problem.
        symmetry : Optional[PermutationGroup]
            Sympy permutation group indicating the symmetry of the variables. If given, it assumes
        degree : Optional[int]
            Degree bound of the monomials. If not specified, it will be inferred from the constraints.
        roots : Optional[List[Root]]
            List of roots (zeros) of the polynomial. The roots must satisfy all given ineq and eq constraints.
            This helps reduce the degree of freedom but may be computationally expensive.
            If `roots=None`, roots will be automatically computed by heuristic methods.
            **THIS WILL BE SLOW IF THE PROBLEM IS LARGE.** TO SKIP ROOTS COMPUTATION, pass in an empty list.
        term_sparsity : int
            Level to exploit term sparsity.
            If 0, no term-sparsity is exploited.
            If 1, term-sparsity is exploited when solving the equation system.
        deparametrize : bool
            Whether to deparametrize the SDP if the polynomial has linear parameters in the coefficients.
        verbose : bool
            Whether to print the progress.
        
        Returns
        ----------
        SDPProblem
            The constructed root node of the SDP problem. It can also be accessed by `self.sdp`.
            The leaf node of the SDP problem can be accessed by `self.sdpp`. 

        Examples
        ----------
        """
        gens, poly = self.gens, self.poly
        nvars = len(gens)
        ineq_constraints = dict((ineq, ineq) for ineq in ineq_constraints)\
            if not isinstance(ineq_constraints, dict) else ineq_constraints
        eq_constraints = dict((eq, eq) for eq in eq_constraints)\
            if not isinstance(eq_constraints, dict) else eq_constraints
        ineq_constraints = {key: Poly(ineq, *gens) for key, ineq in ineq_constraints.items()}
        eq_constraints = {key: Poly(eq, *gens) for key, eq in eq_constraints.items()}

        all_polys = [self.poly] + list(ineq_constraints.values()) + list(eq_constraints.values())

        ###################################################################
        #       Basic operations: compute degrees & unify domains
        ###################################################################
        if degree is None:
            # TODO: what if the highest degree is odd?
            # degree = max(all_polys, key=lambda x: x.total_degree()).total_degree()
            degree = self.poly.total_degree()

        self._ineq_codegrees = {key: (degree - ineq_constraints[key].total_degree())//2
                                for key in ineq_constraints.keys()}
        self._eq_codegrees = {key: (degree - eq_constraints[key].total_degree())
                                for key in eq_constraints.keys()}
        # TODO: shall we raise an exception?
        # if any(_ < 0 for _ in self._ineq_codegrees.values())\
        #         or any(_ < 0 for _ in self._eq_codegrees.values()):
        #     raise ValueError("The working degree is too small to contain the constraints.")
        self._degree = degree


        domain = ZZ
        for p in all_polys[1:]:
            domain = domain.unify(p.domain)
        ineq_constraints = {key: ineq.set_domain(domain) for key, ineq in ineq_constraints.items()}
        eq_constraints = {key: eq.set_domain(domain) for key, eq in eq_constraints.items()}
        self.ineq_constraints = ineq_constraints
        self.eq_constraints = eq_constraints

        self._symmetry = symmetry

        ###################################################################
        #                       Infer bases for SDP
        ###################################################################
        # TODO: correlation sparsity + term sparsity
        homogeneous = all(p.is_homogeneous for p in all_polys)
        mg = MonomialManager(nvars, is_homogeneous=homogeneous)
        if ineq_bases is None:
            ineq_bases = {key: mg for key in ineq_constraints.keys()}

        if eq_bases is None:
            eq_bases = {key: mg for key in eq_constraints.keys()}

        self._ineq_bases = ineq_bases
        self._eq_bases = eq_bases

        ###################################################################
        #                  Get contribution func of bases
        ###################################################################
        monomial_manager = MonomialManager(nvars, perm_group=symmetry, is_homogeneous=homogeneous)
        ineq_mapping = {key: _define_mapping_psd(ineq_constraints[key], basis,
                            monomial_manager, degree) for key, basis in ineq_bases.items()}
        eq_mapping = {key: _define_mapping_linear(eq_constraints[key], basis,
                            monomial_manager, degree) for key, basis in eq_bases.items()}

        equal_indices = _get_equal_entries(monomial_manager, degree,
                            ineq_constraints, eq_constraints, ineq_bases, eq_bases)

        time0 = time()
        sdp, eq_space = SDPProblem.from_entry_contribution(
            monomial_manager.arraylize_sp(poly, degree=degree),
            {k: len(v.inv_monoms(self._ineq_codegrees[k])) for k, v in ineq_bases.items()}, ineq_mapping,
            {k: len(v.inv_monoms(self._eq_codegrees[k])) for k, v in eq_bases.items()}, eq_mapping,
            equal_indices=equal_indices, domain=domain
        )
        self._sdp = sdp
        self._eq_space = eq_space
        if verbose:
            print(f"Time for solving coefficient equations  : {time() - time0:.6f} seconds. Dof = {sdp.dof}")


        ###################################################################
        #            Apply transforms and facial reductions
        ###################################################################
        if deparametrize:
            sdp = sdp.deparametrize()
            sdp.clear_parents()
            self._sdp = sdp

        if term_sparsity:
            time0 = time()
            sdp.get_last_child().constrain_zero_diagonals()
            if verbose:
                print(f"Time for constraining zero diagonals    : {time() - time0:.6f} seconds. Dof = {sdp.get_last_child().dof}")
    
        _, roots = _constrain_root_nullspace(sdp, poly, ineq_constraints, eq_constraints,
            ineq_bases=ineq_bases, eq_bases=eq_bases, degree=degree, roots=roots, verbose=verbose)
        self._roots = roots

        if verbose:
            sdp.print_graph()

        return sdp
