from time import perf_counter
from typing import Dict, TYPE_CHECKING

from sympy import Integer

from ..node import TransformNode
from ...sdp.arithmetic import ArithmeticTimeout

if TYPE_CHECKING:
    from sympy import Expr
    from ..sdpsos.sohs import SOHSPoly


class SolveNCPSD(TransformNode):
    """
    Prove a noncommutative operator is positive semidefinite.

    It assumes all symbols are hermitian.

    The current implementation uses `SOHSPoly` to solve the problem.
    See also examples in `test_noncommutative.py`

    The feature is still highly experimental and has limited performance.

    Examples
    --------
    The following provides examples to solve noncummutative PSD
    problems using the high-level interface `sum_of_squares`.

    Below is an example of solving a unconstrained noncommutative
    problem from [1], Lemma 5.27. We note that symbols are considered
    to be hermitian by default.

    >>> from triples import sum_of_squares
    >>> import sympy as sp
    >>> a,b,c = sp.symbols('a b c', commutative=False)
    >>> f = a**2-a*b-b*a+3*b**2-2*a*b*a+2*a*b**2*a-b*c-c*b
    >>> f = f+6*c**2+9*b**2*c+9*c*b**2-54*c*b*c+142*c*b**2*c
    >>> sol = sum_of_squares(f)

    >>> sol.solution # doctest: +SKIP
    3*a**2/109 + (-48*a + 109*b)**2/11336 + (-44*a + 101*b + 52*c)**2/7384
    + (-a + 2*a*b - c)*(-a + 2*b*a - c)/2
    + (a + 9*b - 27*c + 142*c*b)*(a + 9*b + 142*b*c - 27*c)/142
    >>> sol.as_eq().simplify()
    True

    Reference
    ---------
    [1] Jie Wang, "An Introduction to Polynomial Optimization", 2023.
    """
    def explore(self, configs):
        if self.state < 0:
            self.state = -1
            self.finished = True
            return

        problem = self.problem
        expr = problem.expr
        ineqs = problem.ineq_constraints
        eqs = problem.eq_constraints

        if eqs:
            # not implemented
            self.state = -1
            self.finished = True
            return

        from ..sdpsos.sohs import SOHSPoly
        sohs = SOHSPoly(expr, problem.free_symbols, [Integer(1)] + list(ineqs))
        verbose = configs["verbose"]
        time0 = perf_counter()
        try:
            sdp_sol = sohs.solve(verbose=verbose,
                       time_limit=configs["time_limit"])

            if sdp_sol is not None:
                self.state = -1
                self.finished = True
                if verbose:
                    print(f"Time for solving SDP{' ':20s}: {perf_counter() - time0:.6f}"
                            f" seconds. \033[32mSuccess\033[0m.")
                self._as_solution(
                    sohs,
                    qmodule=dict(enumerate([Integer(1)] + list(ineqs.values()))),
                    ideal=None,
                    configs=configs
                )
                return
        except Exception as e:
            if verbose:
                print(f"Time for solving SDP{' ':20s}: {perf_counter() - time0:.6f}"
                      f" seconds. \033[31mFailed with exceptions\033[0m.")
                print(f"{e.__class__.__name__}: {e}")
            if isinstance(e, (ArithmeticTimeout, MemoryError)):
                # do not try further
                pass
        self.state = -1
        self.finished = True


    def _as_solution(self,
        sohs: "SOHSPoly",
        qmodule: Dict[int, "Expr"],
        ideal: Dict[int, "Expr"],
        # poly_qmodule: Optional[Dict[int, "Expr"]] = None,
        configs: dict = {}
    ):
        solution = sohs.as_solution(
            qmodule=qmodule,
            ideal=ideal,
        ).solution

        self.problem.solution = solution
        return solution
