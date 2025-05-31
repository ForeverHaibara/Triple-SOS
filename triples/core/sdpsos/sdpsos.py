from itertools import combinations
from time import time
from typing import Union, Optional, List, Tuple, Dict, Callable, Generator, Any

import numpy as np
import sympy as sp
from sympy import Poly, Expr
from sympy.combinatorics import PermutationGroup

from .sos import SOSPoly
from .solution import SolutionSDP
from ..preprocess import sanitize
from ..shared import clear_polys_by_symmetry
from ...utils import MonomialManager, optimize_poly, Root


class _lazy_iter:
    """A wrapper for recyclable iterators but initialized when first called."""
    def __init__(self, initializer: Callable[[], Any]):
        self._initializer = initializer
        self._iter = None
    def __iter__(self):
        if self._iter is None:
            self._iter = self._initializer()
        return iter(self._iter)


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


@sanitize(homogenize=False, infer_symmetry=True, wrap_constraints=True)
def SDPSOS(
        poly: Poly,
        ineq_constraints: Union[List[Expr], Dict[Expr, Expr]] = {},
        eq_constraints: Union[List[Expr], Dict[Expr, Expr]] = {},
        symmetry: Optional[Union[MonomialManager, PermutationGroup]] = None,
        roots: Optional[List[Root]] = None,
        ineq_constraints_with_trivial: bool = True,
        preordering: str = 'linear-progressive',
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
    roots: Optional[List[Root]]
        The roots of the polynomial satisfying constraints. When it is None, it will be automatically generated.
    ineq_constraints_with_trivial: bool
        Whether to add the trivial inequality constraint 1 >= 0. This is used to generate the
        quadratic module. Default is True.
    preordering: str
        The preordering method for extending the generators of the quadratic module. It can be
        'none', 'linear', 'linear-progressive'. Default is 'linear-progressive'.
    verbose: bool
        Whether to print the progress. Default is False.
    solver: str
        The numerical SDP solver to use. When set to None, it is automatically selected. Default is None.
    allow_numer: int
        Whether to allow numerical solution (still under development).
    """
    return _SDPSOS(poly, ineq_constraints=ineq_constraints, eq_constraints=eq_constraints,
                symmetry=symmetry, roots=roots, ineq_constraints_with_trivial=ineq_constraints_with_trivial,
                preordering=preordering, verbose=verbose,
                solver=solver, allow_numer=allow_numer, solve_kwargs=solve_kwargs)


def _SDPSOS(
        poly: Poly,
        ineq_constraints: Dict[Poly, Expr] = {},
        eq_constraints: Dict[Poly, Expr] = {},
        symmetry: MonomialManager = None,
        roots: Optional[List[Root]] = None,
        ineq_constraints_with_trivial: bool = True,
        preordering: str = 'linear-progressive',
        verbose: bool = False,
        solver: Optional[str] = None,
        allow_numer: int = 0,
        solve_kwargs: Dict[str, Any] = {},
    ) -> Optional[SolutionSDP]:
    nvars = len(poly.gens)
    degree = poly.total_degree()
    if degree < 1 or nvars < 1:
        return None
    is_hom = poly.is_homogeneous and \
        all(_.is_homogeneous for _ in ineq_constraints.keys()) and \
        all(_.is_homogeneous for _ in eq_constraints.keys())
    if not is_hom:
        degree = max([degree]\
            + [_.total_degree() for _ in ineq_constraints.keys()]\
            + [_.total_degree() for _ in eq_constraints.keys()])
    if not (poly.domain in (sp.ZZ, sp.QQ, sp.RR)):
        return None

    if verbose:
        print(f'SDPSOS nvars = {nvars} degree = {degree}')
        print('Identified Symmetry = %s' % str(symmetry.perm_group).replace('\n', '').replace('  ',''))

    # roots = None
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
                def _lazy_find_roots():
                    roots = optimize_poly(poly, list(ineq_constraints.keys()), [poly] + list(eq_constraints.keys()), return_type='root')
                    if verbose:
                        print(f"Time for finding roots num = {len(roots):<6d}     : {time() - time1:.6f} seconds.")
                    return roots
                roots = _lazy_iter(_lazy_find_roots)
 
            sos_problem = SOSPoly(poly, poly.gens, qmodule = [e[0] for e in qmodule], ideal = list(eq_constraints.keys()),
                                    symmetry = symmetry.perm_group, roots = roots, degree=degree)
            sdp = sos_problem.construct(verbose=verbose)

            if sos_problem.solve(verbose=verbose, solver=solver, allow_numer=allow_numer, kwargs=solve_kwargs) is not None:
                if verbose:
                    print(f"Time for solving SDP{' ':20s}: {time() - time0:.6f} seconds. \033[32mSuccess\033[0m.")
                solution = sos_problem.as_solution(qmodule=dict(enumerate([e[1] for e in qmodule])),
                                                    ideal=dict(enumerate(list(eq_constraints.values()))))
                return solution
        except Exception as e:
            if verbose:
                print(f"Time for solving SDP{' ':20s}: {time() - time0:.6f} seconds. \033[31mFailed with exceptions\033[0m.")
                print(f"{e.__class__.__name__}: {e}")
            continue


    return None