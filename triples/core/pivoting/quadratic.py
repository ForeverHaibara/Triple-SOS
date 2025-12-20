from typing import List, Dict, Tuple, Optional
from functools import partial

from sympy import Poly, Expr, Symbol, Dummy, Mul

from ..problem import InequalityProblem
from ..node import ProofNode
from ..preprocess.polynomial import SolvePolynomial
from ..preprocess.algebraic import CancelDenominator
from ..preprocess.signs import sign_sos

from ...utils import PSatz

def symmetry_ufs(symmetry) -> dict:
    orbits = symmetry.orbits()
    ufs = {}
    for i, orbit in enumerate(orbits):
        for j in orbit:
            ufs[j] = i
    return ufs

def dict_inject(x: dict, y: dict) -> dict:
    x = x.copy()
    x.update(y)
    return x

def _get_linear_symbol_bounds(ineqs: Dict[Poly, Expr], eqs: Dict[Poly, Expr],
    signs: Dict[Symbol, Tuple[Optional[int], Expr]], gen: Symbol):

    lb, lb_expr = None, None
    ub, ub_expr = None, None
    for eq in eqs:
        if eq.degree(gen) >= 1:
            return None
    for ineq, value in ineqs.items():
        d = ineq.degree(gen)
        if d >= 2:
            # cannot analyze its bound
            # TODO: we may factorize a quadratic expression
            return None
        if d == 1:
            ineq = ineq.as_poly(gen)
            k, b = ineq.LC(), ineq.TC()

            k_pos = sign_sos(k, signs)
            if k_pos is not None:
                if lb is not None:
                    # multiple lower bounds -> unknown
                    return None
                lb = -b/k
                lb_expr = value/k_pos
                continue
            k_neg = sign_sos(-k, signs)
            if k_neg is not None:
                if ub is not None:
                    # multiple upper bounds -> unknown
                    return None
                ub = -b/k
                ub_expr = value/k_neg
                continue
    return lb, lb_expr, ub, ub_expr


class PivotQuadratic(ProofNode):
    _info: dict
    def explore(self, configs):
        if self.state == 0:
            return self._explore_state_0(configs)
        elif self.state == 1 and len(self.children) == 0:
            self._restore_four_cases()
            self.state = 2
        if self.state == 2:
            return self._explore_state_2(configs)

    def _explore_state_0(self, configs):
        problem = self.problem
        self.state = 1

        _info = self._info
        A, B, C = _info["coeffs"]
        ineqs, eqs = _info["ineqs"], _info["eqs"]
        disc = 4*A*C - B**2
        _info["disc"] = disc
        margin = _info["margin"]

        lb, lb_expr, ub, ub_expr = self._info["bounds"]
        if lb is None or ub is None:
            # at least one side is infinity -> requires A >= 0
            pro_A = InequalityProblem.new(A, ineqs, eqs)
        elif lb is not None and ub is not None:
            # heuristically test whether A <= 0
            pro_A = InequalityProblem.new(-A, ineqs, eqs)
        node_A = SolvePolynomial(pro_A)
        _info["pro_A"] = pro_A
        _info["node_A"] = node_A

        if lb is None and ub is None:
            # sometimes proving C >= 0 is simpler than A >= 0
            pro_C = InequalityProblem.new(C, ineqs, eqs)
            node_C = SolvePolynomial(pro_C)
            _info["pro_C"] = pro_C
            _info["node_C"] = node_C

            # requires disc >= 0
            pro_disc = InequalityProblem.new(disc, ineqs, eqs)
            node_disc = SolvePolynomial(pro_disc)
            _info["pro_disc"] = pro_disc
            _info["node_disc"] = node_disc

            self.state = -1

        if lb is not None:
            # requires f(lb) >= 0
            pro_lb = InequalityProblem.new(margin.eval(lb), ineqs, eqs)
            node_lb = CancelDenominator(pro_lb)
            _info["pro_lb"] = pro_lb
            _info["node_lb"] = node_lb

            if ub is None:
                # test whether sym_axis <= lb
                pro_syml = InequalityProblem.new(2*A.as_expr()*lb + B.as_expr(), ineqs, eqs)
                node_syml = CancelDenominator(pro_syml)
                _info["pro_syml"] = pro_syml
                _info["node_syml"] = node_syml

        if ub is not None:
            # requires f(ub) >= 0
            pro_ub = InequalityProblem.new(margin.eval(ub), ineqs, eqs)
            node_ub = CancelDenominator(pro_ub)
            _info["pro_ub"] = pro_ub
            _info["node_ub"] = node_ub

            if lb is None:
                # test whether sym_axis >= ub
                pro_symu = InequalityProblem.new(-2*A.as_expr()*ub - B.as_expr(), ineqs, eqs)
                node_symu = CancelDenominator(pro_symu)
                _info["pro_symu"] = pro_symu
                _info["node_symu"] = node_symu


        for key, node in _info.items():
            if key.startswith("node_"):
                self.children.append(node)
                if isinstance(node, SolvePolynomial):
                    node.default_configs["sqf"] = True

    def _explore_state_2(self, configs):
        _info = self._info
        A, B, C = _info["coeffs"]
        ineqs, eqs = _info["ineqs"], _info["eqs"]
        disc = _info["disc"]
        lb, lb_expr, ub, ub_expr = _info["bounds"]

        if lb is None and ub is None:
            # should not reach here
            self.finished = True

        elif lb is not None and ub is None:
            if _info["pro_syml"].solution is None:
                # prove disc >= 0 given (-B/(2A)) >= lb
                SYM = Dummy("S")
                cond_disc = InequalityProblem.new(disc,
                    dict_inject(ineqs, {-2*A.as_expr()*lb - B.as_expr(): SYM}), eqs)
                node_disc = CancelDenominator(cond_disc)
                _info["pro_disc"] = cond_disc
                _info["node_disc"] = node_disc
                _info["dummy"] = SYM
                self.children.append(node_disc)
            else:
                self.finished = True

        elif lb is None and ub is not None:
            if _info["pro_symu"].solution is None:
                # prove disc >= 0 given (-B/(2A)) <= ub
                SYM = Dummy("S")
                cond_disc = InequalityProblem.new(disc,
                    dict_inject(ineqs, {2*A.as_expr()*ub + B.as_expr(): SYM}), eqs)
                node_disc = CancelDenominator(cond_disc)
                _info["pro_disc"] = cond_disc
                _info["node_disc"] = node_disc
                _info["dummy"] = SYM
                self.children.append(node_disc)
            else:
                self.finished = True

        else: # if lb is not None and ub is not None:
            if _info["pro_A"].solution is None:
                # prove disc >= 0 given A >= 0 and lb <= (-B/(2A)) and (-B/(2A)) <= ub
                ### TODO: not implemented
                pass
            self.finished = True

    def update(self, *args, **kwargs):
        if self.finished or self.state == 0:
            return
        self._restore_four_cases()
        if self.state == 1:
            if len(self.children) == 0 or all(_.finished for _ in self.children):
                self.state = 2
                self.children.clear()

    def _restore_four_cases(self):
        lb, lb_expr, ub, ub_expr = self._info["bounds"]
        if lb is None and ub is None:
            return self._restore_real()
        elif lb is not None and ub is None:
            return self._restore_lb()
        elif lb is None and ub is not None:
            return self._restore_ub()
        else:
            return self._restore_lb_ub()

    def _restore_real(self):
        _info = self._info
        failure = lambda x: x.finished and x.solution is None
        if failure(_info["node_disc"]) or (failure(_info["node_A"]) and failure(_info["node_C"])):
            self.finished = True
            return
        failure = lambda x: x.solution is None
        if failure(_info["node_disc"]) or (failure(_info["node_A"]) and failure(_info["node_C"])):
            return

        if _info["pro_A"].solution is not None:
            self.solution = _quadratic_real_A(
                _info["gen"], *_info["coeffs"],
                _info["pro_A"].solution, _info["pro_disc"].solution
            )
        elif _info["pro_C"].solution is not None:
            self.solution = _quadratic_real_C(
                _info["gen"], *_info["coeffs"],
                _info["pro_C"].solution, _info["pro_disc"].solution
            )

    def _restore_lb(self):
        _info = self._info
        if _info["pro_A"].solution is None and _info["node_A"].finished:
            self.finished = True
        elif _info["pro_lb"].solution is None and _info["node_lb"].finished:
            self.finished = True
        if _info["pro_A"].solution is None or _info["pro_lb"].solution is None:
            return

        if _info["pro_syml"].solution is not None:
            self.solution = _quadratic_lb_monotone(
                _info["gen"], *_info["coeffs"],
                _info["pro_A"].solution, 
                _info["pro_syml"].solution,
                _info["pro_lb"].solution,
                _info["bounds"]
            )
        elif _info.get("pro_disc") is not None \
                and _info["pro_disc"].solution is not None:
            self.solution = _quadratic_lb_full(
                _info["gen"], *_info["coeffs"],
                _info["pro_A"].solution, 
                _info["pro_lb"].solution,
                _info["pro_disc"].solution,
                _info["bounds"],
                self.problem.ineq_constraints,
                self.problem.eq_constraints,
                _info["dummy"]
            )

    def _restore_ub(self):
        _info = self._info
        if _info["pro_A"].solution is None and _info["node_A"].finished:
            self.finished = True
        if _info["pro_ub"].solution is None and _info["node_ub"].finished:
            self.finished = True
        if _info["pro_A"].solution is None or _info["pro_ub"].solution is None:
            return

        if _info["pro_symu"].solution is not None:
            self.solution = _quadratic_ub_monotone(
                _info["gen"], *_info["coeffs"],
                _info["pro_A"].solution, 
                _info["pro_symu"].solution,
                _info["pro_ub"].solution,
                _info["bounds"]
            )
        elif _info.get("pro_disc") is not None \
                and _info["pro_disc"].solution is not None:
            self.solution = _quadratic_ub_full(
                _info["gen"], *_info["coeffs"],
                _info["pro_A"].solution, 
                _info["pro_ub"].solution,
                _info["pro_disc"].solution,
                _info["bounds"],
                self.problem.ineq_constraints,
                self.problem.eq_constraints,
                _info["dummy"]
            )

    def _restore_lb_ub(self):
        _info = self._info
        if _info["pro_lb"].solution is None and _info["node_lb"].finished:
            self.finished = True
        if _info["pro_ub"].solution is None and _info["node_ub"].finished:
            self.finished = True
        if _info["pro_lb"].solution is None or _info["pro_ub"].solution is None:
            return

        if _info["pro_A"].solution is not None:
            self.solution = _quadratic_lb_ub_concave(
                _info["gen"], *_info["coeffs"],
                _info["pro_A"].solution, 
                _info["pro_lb"].solution,
                _info["pro_ub"].solution,
                _info["bounds"]
            )

    @classmethod
    def from_problem_gen(cls, problem: InequalityProblem, margin, bounds):
        gen = margin.gen
        other_gens = tuple([g for g in problem.gens if g != gen])
        domain = problem.expr.domain
        coeffs = [Poly(_.to_dict(), other_gens, domain=domain) 
                    for _ in margin.rep.all_coeffs()]

        # remove constraints involving the generator
        ineqs = {k.as_poly(other_gens, domain=domain): v
            for k, v in problem.ineq_constraints.items()
                 if gen not in problem._dtype_free_symbols(k)}
        eqs = {k.as_poly(other_gens, domain=domain): v
            for k, v in problem.eq_constraints.items()
                if gen not in problem._dtype_free_symbols(k)}

        lb, lb_expr, ub, ub_expr = bounds
        if lb is not None and ub is not None:
            # important to push in the constraint ub >= lb
            ineqs[ub - lb] = ub_expr + lb_expr

        node = PivotQuadratic(problem)
        node._info = {
            "gen": gen,
            "margin": margin,
            "coeffs": coeffs,
            "ineqs": ineqs,
            "eqs": eqs,
            "bounds": bounds,
        }
        return node


def pivoting_quadratic(problem: InequalityProblem, configs: dict) -> list:
    """
    Pivoting quadratic constraints.
    """
    poly = problem.expr
    lst = []

    # fs = constraint_free_symbols(problem)
    symmetry = configs["symmetry"]
    signs    = configs["signs"]
    ufs, tried_ufs = symmetry_ufs(symmetry), set()

    ineqs0, eqs0 = problem.ineq_constraints, problem.eq_constraints
    for gen in poly.gens:
        if poly.degree(gen) != 2:
            continue
        if gen in tried_ufs:
            continue
        tried_ufs.add(ufs.get(gen, -1))

        bounds = _get_linear_symbol_bounds(ineqs0, eqs0, signs, gen)
        if bounds is None:
            continue
        lb, lb_expr, ub, ub_expr = bounds


        other_gens = tuple([g for g in problem.gens if g != gen])
        poly = problem.expr
        margin = poly.as_poly(gen, domain=poly.domain[other_gens])

        if margin.rep.all_coeffs()[-1].is_zero:
            continue

        lst.append(PivotQuadratic.from_problem_gen(problem, margin, bounds))

    return lst


def _quadratic_real_A(x, A, B, C, a, disc):
    return ((2*A.as_expr()*x + B.as_expr())**2 + disc) / (4*a)


def _quadratic_real_C(x, A, B, C, c, disc):
    return ((2*C.as_expr() + B.as_expr()*x)**2 + disc*x**2) / (4*c)


def _quadratic_lb_monotone(x, A, B, C, a, syml, f_lb, bounds):
    lb, lb_expr, _, __ = bounds
    # syml = 2*A*lb + B
    # lb_expr = x - lb
    return a*(x - lb)**2 + syml*lb_expr + f_lb


def _quadratic_ub_monotone(x, A, B, C, a, symu, f_ub, bounds):
    _, __, ub, ub_expr = bounds
    # symu = -2*A*ub - B
    # ub_expr = ub - x
    return a*(x - ub)**2 + symu*ub_expr + f_ub


def _quadratic_lb_full(x, A, B, C, a, f_lb, disc, bounds, ineqs, eqs, SYM):
    preorder = [SYM] + list(ineqs.values())
    ideal = list(eqs.values())

    lb, lb_expr, _, __ = bounds

    NSYM = SYM # it stands for another symbol, but reusing it works as well

    # SYM = -2*A*lb - B
    # lb_expr = x - lb

    # proof when NSYM >= 0
    p1 = a*(x - lb)**2 + NSYM*lb_expr + f_lb

    # proof when SYM >= 0
    p2 = ((2*A.as_expr()*x + B.as_expr())**2 + disc) / (4*a)

    p1, p2 = [PSatz.from_sympy(_, preorder, ideal) for _ in [p1, p2]]

    if p1 is None or p2 is None:
        return None

    ps = p1.join(p2, 0)
    return ps.as_expr() if ps is not None else None


def _quadratic_ub_full(x, A, B, C, a, f_ub, disc, bounds, ineqs, eqs, SYM):
    preorder = [SYM] + list(ineqs.values())
    ideal = list(eqs.values())

    _, __, ub, ub_expr = bounds

    NSYM = SYM # it stands for another symbol, but reusing it works as well

    # SYM = 2*A*lb + B
    # ub_expr = ub - x

    # proof when NSYM >= 0
    p1 = a*(x - ub)**2 + NSYM*ub_expr + f_ub

    # proof when SYM >= 0
    p2 = ((2*A.as_expr()*x + B.as_expr())**2 + disc) / (4*a)

    p1, p2 = [PSatz.from_sympy(_, preorder, ideal) for _ in [p1, p2]]
    if p1 is None or p2 is None:
        return None

    ps = p1.join(p2, 0)
    return ps.as_expr() if ps is not None else None


def _quadratic_lb_ub_concave(x, A, B, C, a, f_lb, f_ub, bounds):
    lb, lb_expr, ub, ub_expr = bounds
    # a = -A
    # lb_expr = x - lb, ub_expr = ub - x
    return a*lb_expr*ub_expr + (f_lb*ub_expr + f_ub*lb_expr)/(lb_expr + ub_expr)
