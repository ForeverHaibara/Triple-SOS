from typing import Union, Tuple, List, Dict, Optional, Any, Callable

from sympy import Expr, Poly, Symbol

from .abstract import AtomSOSElement
from .algebra import NCPolyRing
from .solution import SolutionSDP

def DEFAULT_ADJOINT(x):
    if x.is_Mul:
        return x.func(*x.args[::-1])
    elif x.is_Atom:
        return x
    return x.func(*(DEFAULT_ADJOINT(arg) for arg in x.args))

# def compress(monom):
#     m = []
#     pre = -1
#     cnt = 0
#     for x in monom:
#         if x == pre:
#             cnt += 1
#         else:
#             m.append((pre, cnt))
#             pre = x
#             cnt = 1
#     m.append((pre, cnt))
#     return tuple(m[1:]) # remove the first element (which is -1)


class SOHSPoly(AtomSOSElement):
    """
    SOHSPoly is an extension of SOSPoly to general noncommutative C-star algebra.
    It represents a sum-of-hermitian-squares polynomial.

    In a C-star algebra, there is an adjoint operator, "'". Given any element x,
    x'x is positive semidefinite. An element x is hermitian if x' = x.

    This is closely related to the definition in linear algebra. In the field of
    real numbers, X'X is a positive semidefinite matrix, and X is symmetric
    if X' = X. Only symmetric matrices have definitions for "positive semidefinite".

    The class takes in an expression that is symmetric (or hermitian), and assumes
    all given symbols are symmetric (or hermitian).


    Examples
    ---------

    ### Proving semidefiniteness

    Taking the example from the paper "Noncommutative Polynomial Optimization" by
    Bhardwaj, Klep and Margron, Example 3.2. We wish to show

        1+2*X+X**2+X*Y**2+2*Y**2+Y**2*X+Y*X**2*Y+Y**4

    is positive semidefinite by sum-of-hermitian-squares. Note that X and Y
    are hermitian but noncommutative, e.g., real symmetric matrices. To do so,
    we use the noncommutative algebra in SymPy:

        >>> from sympy import symbols
        >>> X, Y = symbols("X Y", commutative=False)
        >>> poly = 1+2*X+X**2+X*Y**2+2*Y**2+Y**2*X+Y*X**2*Y+Y**4
        >>> sohs = SOHSPoly(poly, (X, Y), [1])
        >>> sohs.solve()
        Matrix([
        [0],
        [0]])
        >>> sohs.as_solution().solution
        Y*X**2*Y + (1 + X + Y**2)**2
        >>> (poly - sohs.as_solution().solution).expand()  # check correctness
        0

    After solving, the solution can be accessed by calling `sohs.as_solution().solution`
    as above. It is also possible to pass in an adjoint operator (a callable function)
    that defines the adjoint of expressions:

        >>> sohs.as_solution(adjoint_operator=lambda x: x.adjoint()).solution
        (1 + adjoint(X) + adjoint(Y)**2)*(1 + X + Y**2) + adjoint(Y)*adjoint(X)*X*Y

    """
    def __init__(self,
        poly: Union[Expr, Poly],
        gens: Tuple[Symbol,...],
        qmodule: List[Union[Poly, Expr]] = [],
        ideal: List[Union[Poly, Expr]] = [],
        degree: Optional[int] = None,
    ):
        gens = tuple(gens)
        as_poly = NCPolyRing(len(gens), 0).as_poly
        poly = as_poly(poly, gens)
        self.poly = poly
        self.gens = gens

        # TODO: this is not neat enough. Also, what if a qmodule is associated with multiple
        # basis? e.g. CORR SPARSITY
        if not isinstance(qmodule, dict):
            qmodule = dict(enumerate(qmodule))
        if not isinstance(ideal, dict):
            ideal = dict(enumerate(ideal))
        self._qmodule = {k: as_poly(v, gens) for k, v in qmodule.items()}
        self._ideal = {k: as_poly(v, gens) for k, v in ideal.items()}

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
        self.algebra = NCPolyRing(len(poly.gens), degree=degree, is_homogeneous=is_homogeneous)

    def _post_construct(self, verbose: bool = False, **kwargs):
        self.sdp.constrain_zero_diagonals()

    def as_solution(
        self,
        qmodule: Optional[Dict[Any, Expr]] = None,
        ideal: Optional[Dict[Any, Expr]] = None,
        adjoint_operator: Optional[Callable[[Expr], Expr]] = DEFAULT_ADJOINT,
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
            # state_operator   = (lambda x: CyclicSum(x, self.gens, self.algebra.symmetry))\
            #                         if self.algebra.symmetry is not None else (lambda x: x),
            adjoint_operator = adjoint_operator,
        )

        # overwrite the constraints information in the form of dict((Poly, Expr))
        solution.ineq_constraints = {self._qmodule[key]: expr for key, expr in qmodule.items()}
        solution.eq_constraints = {self._ideal[key]: expr for key, expr in ideal.items()}
        return solution
