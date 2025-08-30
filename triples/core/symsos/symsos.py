from typing import Dict, Set, List, Optional

import sympy as sp
from sympy.core.symbol import uniquely_named_symbol

from .symmetric import UE3Real
from .basic import prove_by_pivoting
from .representation import sym_transform, sym_representation_inv
from .solution import SolutionSymmetric
from ...utils import Coeff, verify_symmetry


from ..node import TransformNode
from ..preprocess import SolvePolynomial

def _nonnegative_vars(ineq_constraints: List[sp.Poly]) -> Set[sp.Symbol]:
    """
    Infer the nonnegativity of each variable from the inequality constraints.
    """
    nonnegative = set()
    for ineq in ineq_constraints:
        if ineq.is_monomial and ineq.total_degree() == 1 and ineq.LC() >= 0:
            nonnegative.update(ineq.free_symbols)
    return nonnegative


class SymmetricSubstitution(TransformNode):
    def explore(self, configs):
        if self.status != 0:
            return

        # check symmetricity here # and (1,1,1) == 0
        poly = self.problem.expr
        if (len(poly.gens) != 3 or not (poly.domain in (sp.ZZ, sp.QQ)))\
                or not poly.is_homogeneous:
            self.status = 1
            self.finished = True
            return None

        methods = [UE3Real]
        # methods = ['real']
        # signs = self.problem.get_symbol_signs()
        # nonneg = {k: expr for k, (sgn, expr) in signs.items() if sgn == 1}
        # if all(i in nonneg for i in poly.gens):
        #     methods.append('positive')

        # dummys = [sp.Dummy("s") for _ in range(len(poly.gens))]
        dummys = sp.symbols('x y z')

        for method in methods:
            if method.nvars != poly.gens or (not verify_symmetry(poly, method.symmetry)):
                continue

            applied = method.apply(self.problem, poly.gens, dummys)
            if applied is None:
                continue

            solver = SolvePolynomial(applied[0])
            self.children.append(solver)
            self.restorations[solver] = restoration

        self.status = 1
        if len(self.children) == 0:
            self.finished = True


# @sanitize(homogenize=True)
def SymmetricSOS(
        poly: sp.Poly,
        ineq_constraints: Dict[sp.Poly, sp.Expr] = {},
        eq_constraints: Dict[sp.Poly, sp.Expr] = {},
    ) -> Optional[SolutionSymmetric]:
    """
    Solve symmetric polynomial inequalities using special
    changes of variables. The algorithm is powerful but produces
    EXTREMELY COMPLICATED solutions.

    Parameters
    ----------
    poly: sp.Poly
        The polynomial to perform SOS on.
    ineq_constraints: List[sp.Poly]
        Inequality constraints to the problem. This assumes g_1(x) >= 0, g_2(x) >= 0, ...
    eq_constraints: List[sp.Poly]
        Equality constraints to the problem. This assumes h_1(x) = 0, h_2(x) = 0, ...

    Returns
    -----------
    SolutionSymmetricSimple
        The solution of the problem.

    References
    -----------
    [1] 陈胜利.不等式的分拆降维幂方法与可读证明.哈尔滨工业大学出版社,2016.

    [2] https://zhuanlan.zhihu.com/p/616532245
    
    [3] https://zhuanlan.zhihu.com/p/20969491385
    """
    # from ..structsos import StructuralSOS
    from ..sdpsos.sdpsos import SDPSOSSolver
    problem = TransformNode.new_problem(poly, ineq_constraints, eq_constraints)
    configs = {
        # TODO: ...
        SolvePolynomial: {'solvers': [SymmetricSubstitution, SDPSOSSolver]},
    }
    return problem.sum_of_squares(configs)
