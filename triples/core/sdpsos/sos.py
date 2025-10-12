from typing import List, Tuple, Dict, Union, Optional, Callable, Any
from time import time
# from warnings import warn

from sympy.combinatorics import PermutationGroup
from sympy import Poly, Expr, Symbol
from sympy.matrices import MutableDenseMatrix as Matrix

from .algebra import PolyRing
from .abstract import AtomSOSElement, ArithmeticTimeout
from .manifold import constrain_root_nullspace
from .solution import SolutionSDP
from ...utils import CyclicSum, Root

CHECK_SYMMETRY = True


class SOSPoly(AtomSOSElement):
    """
    A class representing a sum-of-squares polynomial given a quadratic module
    and a ideal, i.e., a polynomial in the following form:

        F = sum(G[i] * SOS[i] for i in range(len(G))) + sum(H[i] * v[i] for i in range(len(H)))

    where SOS[i] is a sum-of-squares polynomial. G is the quadratic module and H is the ideal.
    Particularly, if G = [1] and H = [], then Poly is a pure sum-of-squares polynomial.

    Parameters
    ----------
    poly: Union[Expr, Poly]
        The target SymPy polynomial.
    gens: Tuple[Symbol,...]
        The generators (variables) of the polynomial.
    qmodule: List[Union[Poly, Expr]]
        A list of polynomials as the generators of the quadratic module.
    ideal: List[Union[Poly, Expr]]
        A list of polynomials as the generators of the ideal.
    degree: Optional[int]
        The degree truncation.
    symmetry: Optional[PermutationGroup]
        The symmetry group of the polynomial.
    roots: Optional[List[Root]]
        A list of roots of the polynomial. It can be a list of tuples.
        Each root must satisfy poly(*root) == 0 and ineq_constraints(*root) >= 0
        and eq_constraints(*root) == 0 where ineq_constraints and eq_constraints are the
        polynomials in qmodule and ideal and their permutations under the symmetry group,
        respectively. The roots are used for reducing the degree of freedom of the SDPProblem
        by facial reduction. If `roots=None`, it applies a heuristic approach to find some
        roots automatically, but it might be slow. Defaults to an empty list.


    Examples
    --------

    ## Proving nonnegativity

    Here is a simple tutorial on how to prove nonnegativity of a polynomial
    via sum of squares. Consider the Vasile's inequality:

        (a^2 + b^2 + c^2)^2 - 3*(a^3*b+b^3*c+c^3*a) >= 0.

    The inequality holds for all real numbers a, b, c and we want to prove it
    by sum of squares. To do so, we initialize the SOSPoly and solve it
    as follows.

        >>> from sympy.abc import a, b, c, u, v, w, x, y, z, t
        >>> sos = SOSPoly((a**2+b**2+c**2)**2-3*(a**3*b+b**3*c+c**3*a), (a,b,c), [1])
        >>> sos.solve()
        Matrix([
        [-3/2],
        [ 3/4],
        [-3/2],
        [-3/4],
        [ 3/4],
        [-3/2]])
    
    After the problem is solved successfully, the solution can be obtained by `sos.as_solution()`.
    It returns a Solution class object and the sympy expression can be accessed by `.solution`
    of the Solution object.

        >>> sol = sos.as_solution()
        >>> sol.solution  # a SymPy expression # doctest:+SKIP
        (2*a**2 - 3*a*b - b**2 + 3*b*c - c**2)**2/4 + 3*(a*b - 2*a*c - b**2 + b*c + c**2)**2/4

    ### Adding inequality or equality constraints

    The more general case of an SOSProblem is on a constrained set (semialgebraic set):
    G1(x), ..., Gn(x) >= 0, H1(x),..., Hm(x) = 0. And we expect the polynomial can be represented
    in the form of:

        f(x) = sum_i (Gi * SOS_i) + sum_j (Hj * Poly_j) >= 0.

    Each SOS_i is a sum of squares polynomial. The Gi are known as the generators of the quadratic module,
    while Hj are known as the generators of the ideal. The polynomials Gi and Hi should be
    passed in as the `qmodule` and `ideal` arguments, respectively.


    ### Setting degree truncation and symmetry group

    Consider the example: a,b,c >= 0 and abc=1, prove a^2+b^2+c^2-a-b-c>=0. The polynomial is completely
    symmetric with respect to all variables, which can be represented as a sympy SymmetricGroup(3) object.
    Suppose we want the polynomial to be in the form of Σ SOS1 + Σ a*b*SOS2 + Σ (a*b*c-1)*Poly,
    where Σ denotes a symmetric sum (6 terms) with respect to (a, b, c).
    Besides, we also consider polynomials of degree = 4, i.e., Σ (a*b*c-1)*Poly can be a degree-4 polynomial.
    Then we can initialize the instance as follows. Note that the SOS result might vary depending on the solver.

        >>> from sympy.combinatorics import SymmetricGroup
        >>> sos = SOSPoly(a**2+b**2+c**2-a-b-c, (a,b,c), [1,a*b], [a*b*c-1], degree=4, symmetry=SymmetricGroup(3))
        >>> solved = sos.solve()
        >>> sos.as_solution().solution  # doctest:+SKIP
        Σ(a*b*c - 1)*(-a/18 - b/18 - c/18 + 1/3) + (Σa*b*(c - 1)**2)/6 + 7*(Σ(a - 1)**2)/45
        + 7*(Σ(-a + 5*b - 4)**2)/1080 + (Σ(-a - b + 6*c - 4)**2)/216

    The cyclic sum can also be expanded by calling `doit`.

        >>> sos.as_solution().solution.doit()  # this expands the cyclic sum # doctest:+SKIP
        a*b*(c - 1)**2/3 + a*c*(b - 1)**2/3 + b*c*(a - 1)**2/3 + 14*(a - 1)**2/45 + 14*(b - 1)**2/45
        + 14*(c - 1)**2/45 + 6*(a*b*c - 1)*(-a/18 - b/18 - c/18 + 1/3) + 7*(-a + 5*b - 4)**2/1080
        + 7*(-a + 5*c - 4)**2/1080 + 7*(5*a - b - 4)**2/1080 + 7*(5*a - c - 4)**2/1080 + 7*(-b + 5*c - 4)**2/1080
        + 7*(5*b - c - 4)**2/1080 + (-a - b + 6*c - 4)**2/108 + (-a + 6*b - c - 4)**2/108 + (6*a - b - c - 4)**2/108
        >>> sos.as_solution().solution.expand()  # fully expand it to verify correctness
        a**2 - a + b**2 - b + c**2 - c

    Note that the original polynomial a^2+b^2+c^2-a-b-c has degree 2 and the constraint a*b*c-1 has degree 3,
    but we obtain a solution in degree 4. If `degree=4` is not set, the default degree will be 2, the degree
    of the target polynomial and there is no solution:

        >>> sos = SOSPoly(a**2+b**2+c**2-a-b-c, (a,b,c), [1,a*b], [a*b*c-1], symmetry=SymmetricGroup(3))
        >>> solved = sos.solve()  # doctest:+SKIP
        Traceback (most recent call last):
        ...
        ValueError: Linear system has no solution

    Apart from the complete symmetric group, other sympy PermutationGroup objects are supported as well.

    ### Nullstellensatz

    In "Moment and Polynomial Optimization" by Jiawang Nie, Example 2.6.7, an example
    is presented to show the set {(x,y): x-y^2+3>=0, y+x^2+2==0} is empty. This is done
    by finding polynomials g1, g2 and h1 such that -1 = g1 + (x-y^2+3)*g2 + (y+x^2+2)*h1,
    where g1, g2 >= 0 are sum-of-squares. If such polynomials are found, then -1 >= 0
    leads to an immediate contradiction, implying the set is empty. We need to set `degree = 2`
    so that it considers combinations of polynomials in degree 2.

        >>> sos = SOSPoly(-1, (x,y), [1, x-y**2+3], [y+x**2+2], degree=2)
        >>> solved = sos.solve()
        >>> sos.as_solution().solution  # doctest:+SKIP
        -10*x**2 + 2*x - 2*y**2 - 10*y + (10*x - 1)**2/10 + (2*y + 5)**2/2 - 68/5
        >>> sos.as_solution().solution.expand()  # fully expand it to verify correctness
        -1


    """
    ##########################################################
    # Manually set any variables below after construction
    # could be unsafe, and results in unexpected behaviors.
    ##########################################################

    gens: Tuple[Symbol,...]
    roots: List[Root]

    def __init__(self,
        poly: Union[Expr, Poly],
        gens: Tuple[Symbol,...],
        qmodule: List[Union[Poly, Expr]] = [],
        ideal: List[Union[Poly, Expr]] = [],
        degree: Optional[int] = None,
        symmetry: Optional[PermutationGroup] = None,
        roots: Optional[List[Root]] = [],
        # qmodule_bases: Optional[List[Tuple[int, ...]]] = None,
        # ideal_bases: Optional[List[Tuple[int, ...]]] = None,
    ):
        gens = tuple(gens)
        poly = Poly(poly, gens)
        self.poly = poly
        self.gens = gens

        # TODO: this is not neat enough. Also, what if a qmodule is associated with multiple
        # basis? e.g. CORR SPARSITY
        if not isinstance(qmodule, dict):
            qmodule = dict(enumerate(qmodule))
        if not isinstance(ideal, dict):
            ideal = dict(enumerate(ideal))
        self._qmodule = {k: Poly(v, gens) for k, v in qmodule.items()}
        self._ideal = {k: Poly(v, gens) for k, v in ideal.items()}
        
        if degree is None:
            # degree = max([self.poly] + list(self._qmodule.values()) + list(self._ideal.values()),
            #                 key=lambda _: _.total_degree()).total_degree()
            degree = self.poly.total_degree()
        else:
            degree = int(degree)
            if degree < 0 or degree < self.poly.total_degree():
                raise ValueError("The degree should be nonnegative and no smaller than the total degree of the polynomial.")

        is_homogeneous = self.poly.is_homogeneous \
            and all(_.is_homogeneous for _ in self._qmodule.values()) \
            and all(_.is_homogeneous for _ in self._ideal.values()) \
            and self.poly.total_degree() == degree
        self.algebra = PolyRing(len(gens), degree=degree, symmetry=symmetry, is_homogeneous=is_homogeneous)

        if symmetry is not None and not self.algebra.is_symmetric(poly, symmetry):
            raise ValueError("The polynomial is not symmetric under the given symmetry group.")


        self.roots = roots

    def _post_construct(self, verbose: bool = False, time_limit: Optional[Union[Callable, float]] = None, **kwargs):
        time_limit = ArithmeticTimeout.make_checker(time_limit)
        self.sdp.constrain_zero_diagonals(time_limit=time_limit)

        insert_prefix = lambda d: {(self, k): v for k, v in d.items()}

        _, roots = constrain_root_nullspace(self.sdp, self.poly,
            insert_prefix(self._qmodule), insert_prefix(self._ideal),
            ineq_bases=insert_prefix(self._qmodule_bases), eq_bases=insert_prefix(self._ideal_bases),
            degree=self.algebra.degree, roots=self.roots, symmetry=self.algebra.symmetry,
            verbose=verbose, time_limit=time_limit)
        self.roots = roots

    def as_solution(
        self,
        qmodule: Optional[Dict[Any, Expr]] = None,
        ideal: Optional[Dict[Any, Expr]] = None,
        adjoint_operator: Optional[Callable[[Expr], Expr]] = None,
        trace_operator: Optional[Callable[[Expr], Expr]] = None,
    ) -> SolutionSDP:
        """
        Retrieve the solution to the original polynomial.

        Parameters
        ----------
        qmodule: Optional[Dict[Any, Expr]]
            Conversion of polynomial inequality constraints to sympy expression forms.
        ideal: Optional[Dict[Any, Expr]]
            Conversion of polynomial equality constraints to sympy expression forms.
        """
        decomp = self.sdp.decompositions
        if decomp is None:
            raise ValueError("The problem has not been solved yet.")
        decomp = {k[1]: v for k, v in decomp.items() if k[0] is self} # remove the prefix

        qmodule = self._qmodule if qmodule is None else qmodule
        ideal = self._ideal if ideal is None else ideal
        if not isinstance(qmodule, dict):
            qmodule = dict(enumerate(qmodule))
        if not isinstance(ideal, dict):
            ideal = dict(enumerate(ideal))

        eqspace = {eq: x + space * self.sdp.y for eq, (x, space) in self._ideal_space.items()}

        solution = SolutionSDP.from_decompositions(self.poly, decomp, eqspace,
            qmodule          = qmodule,
            qmodule_bases    = self._qmodule_bases,
            ideal            = ideal,
            ideal_bases      = self._ideal_bases,
            state_operator   = (lambda x: CyclicSum(x, self.gens, self.algebra.symmetry))\
                                    if self.algebra.symmetry is not None else (lambda x: x),
            adjoint_operator = adjoint_operator,
        )

        # overwrite the constraints information in the form of dict((Poly, Expr))
        solution.ineq_constraints = {self._qmodule[key]: expr for key, expr in qmodule.items()}
        solution.eq_constraints = {self._ideal[key]: expr for key, expr in ideal.items()}
        return solution
