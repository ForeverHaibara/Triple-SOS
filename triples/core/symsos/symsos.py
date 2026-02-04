from typing import Dict, List, Union, Optional

from sympy import Poly, Expr, Dummy

from .symmetric import UE3Real, UE3Positive, UE4Real
from .basic import prove_by_pivoting
from ..node import TransformNode
from ..preprocess import SolvePolynomial
from ..solution import Solution
from ...utils import verify_symmetry


class SymmetricSubstitution(TransformNode):
    """
    Apply symmetric substitution on the variables.
    """
    def explore(self, configs):
        if self.state != 0:
            return

        # check symmetry here
        poly = self.problem.expr
        if (not (3 <= len(poly.gens) <= 4)) or not poly.is_homogeneous:
            self.state = 1
            self.finished = True
            return None

        methods = [UE3Real, UE3Positive, UE4Real]
        # methods = ['real']
        # signs = self.problem.get_symbol_signs()
        # nonneg = {k: expr for k, (sgn, expr) in signs.items() if sgn == 1}
        # if all(i in nonneg for i in poly.gens):
        #     methods.append('positive')

        # dummys = [sp.Dummy("s") for _ in range(len(poly.gens))]
        dummys = [Dummy(_) for _ in 'xyzw']
        for method in methods:
            if method.nvars != len(poly.gens) or (not verify_symmetry(poly, method.symmetry)):
                continue

            applied = method.apply(self.problem, poly.gens, dummys[:len(poly.gens)])
            if applied is None:
                continue

            solver = SolvePolynomial(applied[0])
            self.children.append(solver)
            self.restorations[solver] = applied[1]

        self.state = -1
        if len(self.children) == 0:
            self.finished = True


# @sanitize(homogenize=True)
def SymmetricSOS(
    expr: Expr,
    ineq_constraints: Union[List[Expr], Dict[Expr, Expr]] = {},
    eq_constraints: Union[List[Expr], Dict[Expr, Expr]] = {},
    verbose: bool = False,
    time_limit: float = 3600.,
) -> Optional[Solution]:
    """
    Solve symmetric polynomial inequalities using special
    changes of variables. The algorithm is powerful but produces
    very complicated solutions.

    This SymmetricSOS solver uses SymmetricSubstitution in prior
    to solve problems.

    Parameters
    ----------
    expr: Expr
        The expression to perform SOS on.
    ineq_constraints: Union[List[Expr], Dict[Expr, Expr]]
        Inequality constraints to the problem. This assumes g_1(x) >= 0, g_2(x) >= 0, ...
    eq_constraints: Union[List[Expr], Dict[Expr, Expr]]
        Equality constraints to the problem. This assumes h_1(x) = 0, h_2(x) = 0, ...
    verbose: bool
        Whether to print information during the solving process. Defaults to False.
    time_limit: float
        The time limit (in seconds) for the solver. Defaults to 3600. When the time limit is
        reached, the solver is killed when it returns to the main loop.
        However, it might not be killed instantly if it is stuck in an internal function.

    Returns
    -----------
    Solution
        The solution of the problem.

    References
    -----------
    [1] 陈胜利. 不等式的分拆降维幂方法与可读证明. 哈尔滨工业大学出版社, 2016.

    [2] https://zhuanlan.zhihu.com/p/616532245

    [3] https://zhuanlan.zhihu.com/p/20969491385
    """
    from ..node import ProofTree, ProofNode
    problem = TransformNode.new_problem(expr, ineq_constraints, eq_constraints)

    def _explore_symsos(tree, node: ProofNode, configs):
        has_symsos = False
        for child in node.children:
            if isinstance(child, SymmetricSubstitution):
                child.explore(child.default_configs)
                if child.children:
                    has_symsos = True
        if has_symsos:
            # close other solvers
            for child in node.children:
                if not isinstance(child, SymmetricSubstitution):
                    child.finished = True

    # from ..structsos import StructuralSOSSolver
    # from ..sdpsos.sdpsos import SDPSOSSolver
    configs = {
        ProofTree: {
            "verbose": verbose,
            "time_limit": time_limit,
        },
        ProofNode: {"verbose": verbose},
        SolvePolynomial: {
            # "solvers": [SymmetricSubstitution, StructuralSOSSolver, SDPSOSSolver],
            "callback_after_explore": _explore_symsos,
        },
    }
    return problem.sum_of_squares(configs)
