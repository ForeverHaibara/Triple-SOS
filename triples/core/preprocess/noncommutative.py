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
