from typing import List, TYPE_CHECKING
from .elimination import eliminate_power_constraints, resultant_elimination
from ..node import TransformNode

if TYPE_CHECKING:
    from ..node import ProofNode


class SolvePolynomial(TransformNode):
    """
    Solve a dense polynomial inequality. The target expression
    and its constraints are all converted and stored as sympy (dense)
    Poly class. However, the process of converting expressions to dense
    polynomials is inefficient for very large inputs.
    """
    default_configs = {
        "sqf": False,
        "homogenize": True,
        "remove_redundancy": True,
        "eliminate_power_constraints": True,
        "eliminate_binomial_constraints": True,
        "irrational_expr": False,
        "verbose": False,
    }
    def get_polynomial_solvers(self, configs) -> List["ProofNode"]:
        solvers = configs.get('solvers', None)
        if solvers is None:
            from ..progsos.progsos import ProgSOSSolver
            from ..structsos.structsos import StructuralSOSSolver
            from ..linsos.linsos import LinearSOSSolver
            from ..sdpsos.sdpsos import SDPSOSSolver
            from ..symsos.symsos import SymmetricSubstitution
            from ..pivoting.pivoting import Pivoting
            # from .reparam import Reparametrization
            solvers = [
                StructuralSOSSolver,
                ProgSOSSolver,
                LinearSOSSolver,
                SDPSOSSolver,
                SymmetricSubstitution,
                # Reparametrization,
                Pivoting
            ]
        return solvers

    def explore(self, configs):
        if self.state != 0 and len(self.children) == 0:
            # all children failed
            self.finished = True
            return

        problem = self.problem.polylize()

        sqf = 1
        if configs["sqf"]:
            problem, sqf = problem.sqr_free(
                problem_sqf=True, ineqs_sqf=False, eqs_sqf=False)

        problem = problem.remove_redundancy()

        power_restore = lambda x: x
        if configs["eliminate_power_constraints"]:
            new_problem, new_power_restore = eliminate_power_constraints(
                problem,
                irrational_expr=configs["irrational_expr"]
            )
            if new_problem is problem:
                # nothing changed
                pass
            else:
                sym0 = problem.identify_symmetry()
                sym1 = new_problem.identify_symmetry()
                if sym0.order() <= sym1.order():
                    problem = new_problem
                    power_restore = new_power_restore
                else:
                    # TODO: the elimination is not good enough
                    # and we had better preserve both
                    pass

        problem, res_restore = resultant_elimination(
            problem,
            **{
                key: configs[key]
                for key in [
                    "homogenize",
                    "eliminate_binomial_constraints",
                    "verbose",
                ]
            }
        )


        if problem.expr.total_degree() <= 0 and problem.expr.LC() >= 0:
            # nonnegative constant to prove
            self.solution = problem.expr.LC() * sqf**2
            self.finished = True
            return

        if configs["verbose"]:
            print(f"Processed problem at id {id(problem)}:\n"
                f"Vars        = {len(problem.gens)}\n"
                f"Degree      = {problem.expr.total_degree()}\n"
                f"Constraints = {len(problem.ineq_constraints)} ineqs +"
                f" {len(problem.eq_constraints)} eqs")

        solvers = self.get_polynomial_solvers(configs)
        self.children = [
            solver(problem,
                {"irrational_expr":configs["irrational_expr"]} \
                if "irrational_expr" in solver.default_configs else None
            ) for solver in solvers
        ]

        self.state = -1

        def composed_restoration(x):
            if x is None:
                return None
            return sqf**2 * power_restore(res_restore(x))

        self.restorations = dict.fromkeys(self.children, composed_restoration)

        if self.state != 0 and len(self.children) == 0:
            # check one more time if there are no children
            self.finished = True
