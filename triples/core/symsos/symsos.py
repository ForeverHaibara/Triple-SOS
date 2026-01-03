from typing import Dict, Set, List, Union, Optional

from sympy import Poly, Expr, Dummy
from sympy.polys import ZZ, QQ

from .symmetric import UE3Real, UE3Positive
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
        if len(poly.gens) != 3 or not (poly.domain in (ZZ, QQ))\
                or not poly.is_homogeneous:
            self.state = 1
            self.finished = True
            return None

        methods = [UE3Real, UE3Positive]
        # methods = ['real']
        # signs = self.problem.get_symbol_signs()
        # nonneg = {k: expr for k, (sgn, expr) in signs.items() if sgn == 1}
        # if all(i in nonneg for i in poly.gens):
        #     methods.append('positive')

        # dummys = [sp.Dummy("s") for _ in range(len(poly.gens))]
        dummys = [Dummy(_) for _ in 'xyz']
        for method in methods:
            if method.nvars != len(poly.gens) or (not verify_symmetry(poly, method.symmetry)):
                continue

            applied = method.apply(self.problem, poly.gens, dummys)
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
) -> Optional[Solution]:
    """
    Solve symmetric polynomial inequalities using special
    changes of variables. The algorithm is powerful but produces
    EXTREMELY COMPLICATED solutions.

    Parameters
    ----------
    expr: Expr
        The expression to perform SOS on.
    ineq_constraints: Union[List[Expr], Dict[Expr, Expr]]
        Inequality constraints to the problem. This assumes g_1(x) >= 0, g_2(x) >= 0, ...
    eq_constraints: Union[List[Expr], Dict[Expr, Expr]]
        Equality constraints to the problem. This assumes h_1(x) = 0, h_2(x) = 0, ...

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
    # from ..structsos import StructuralSOS
    from ..sdpsos.sdpsos import SDPSOSSolver
    problem = TransformNode.new_problem(expr, ineq_constraints, eq_constraints)
    configs = {
        # TODO: ...
        SolvePolynomial: {'solvers': [SymmetricSubstitution, SDPSOSSolver]},
    }
    return problem.sum_of_squares(configs)
