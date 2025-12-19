from typing import List, Dict, Tuple, Optional
from functools import partial

from sympy import Poly, Expr, Symbol, Dummy

from ..problem import InequalityProblem
from ..preprocess.polynomial import SolvePolynomial
from ..preprocess.algebraic import CancelDenominator
from ..preprocess.signs import sign_sos

from ...utils import PSatz

def constraint_free_symbols(problem: InequalityProblem) -> set:
    symbols = set(problem.free_symbols)
    for ineq in problem.ineq_constraints:
        symbols -= problem._dtype_free_symbols(ineq)
    for eq in problem.eq_constraints:
        symbols -= problem._dtype_free_symbols(eq)
    return symbols

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


def pivoting_quadratic(problem: InequalityProblem, configs: dict) -> list:
    """
    Pivoting quadratic constraints.
    """
    poly = problem.expr
    lst = []

    # fs = constraint_free_symbols(problem)
    symmetry = problem.identify_symmetry()
    ufs, tried_ufs = symmetry_ufs(symmetry), set()

    signs = problem.get_symbol_signs()

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

        ineqs = {k.as_poly(other_gens, domain=poly.domain): v
            for k, v in ineqs0.items() if gen not in problem._dtype_free_symbols(k)}
        eqs = {k.as_poly(other_gens, domain=poly.domain): v
            for k, v in eqs0.items() if gen not in problem._dtype_free_symbols(k)}

        A, B, C = [Poly(_.to_dict(), other_gens, domain=poly.domain) 
                    for _ in margin.rep.all_coeffs()]
        if A.is_zero or C.is_zero:
            continue
        disc = 4*A*C - B**2


        LB, node_lb = None, None
        UB, node_ub = None, None
        if lb is not None:
            f_lb = margin.eval(lb)
            LB = InequalityProblem.new(f_lb, ineqs, eqs)
            node_lb = CancelDenominator(LB)
        if ub is not None:
            f_ub = margin.eval(ub)
            UB = InequalityProblem.new(f_ub, ineqs, eqs)
            node_ub = CancelDenominator(UB)

        # sym = -B/(2A)
        if lb is not None and ub is None:
            # (A >= 0) && (disc >= 0 | sym <= lb)
            SYM = Dummy("S")
            pro_A = InequalityProblem.new(A, ineqs, eqs)
            cond_disc = InequalityProblem.new(disc,
                            dict_inject(ineqs, {-2*A*lb - B: SYM}), eqs)
            node_a = SolvePolynomial(pro_A)
            node_cond_disc = CancelDenominator(cond_disc)
            lst.append({
                "children": [node_a, node_lb, node_cond_disc],
                "restoration": partial(_quadratic_lb_full,
                    gen, A, B, C, pro_A, LB, cond_disc, bounds, ineqs0, eqs0, SYM)
            })

        elif lb is None and ub is not None:
            # (A <= 0) && (disc >= 0 | sym >= ub)
            SYM = Dummy("S")
            pro_A = InequalityProblem.new(A, ineqs, eqs)
            cond_disc = InequalityProblem.new(disc,
                            dict_inject(ineqs, {2*A*ub + B: SYM}), eqs)
            node_a = SolvePolynomial(pro_A)
            node_cond_disc = CancelDenominator(cond_disc)
            lst.append({
                "children": [node_a, node_cond_disc],
                "restoration": partial(_quadratic_ub_full,
                    gen, A, B, C, UB, cond_disc, bounds, ineqs0, eqs0, SYM)
            })

        elif lb is not None and ub is not None:
            continue
            # ((A >= 0) && (disc >= 0 | sym <= lb | sym >= ub) | (A <= 0)
            AA = Dummy("A")
            cond_disc = InequalityProblem.new(disc, dict_inject(ineqs, {A: AA}), eqs)
            node_cond_disc = SolvePolynomial(cond_disc)
            lst.append({
                "children": [node_lb, node_ub, node_cond_disc],
                "restoration": partial(_quadratic_lb_ub_full, gen, A, LB, UB, cond_disc)
            })
        else: # over reals
            pros = [InequalityProblem.new(_, ineqs, eqs) for _ in [A, C, disc]]
            nodes = [SolvePolynomial(pro) for pro in pros]
            
            lst.append({
                "children": [nodes[0], nodes[2]],
                "restoration": partial(_quadratic_real_A, gen, A, B, pros[0], pros[2])
            })

            lst.append({
                "children": [nodes[1], nodes[2]],
                "restoration": partial(_quadratic_real_C, gen, C, B, pros[1], pros[2])
            })

    for pivot in lst:
        for node in pivot["children"]:
            if isinstance(node, SolvePolynomial):
                node.default_configs["sqf"] = True

    return lst


def _quadratic_real_A(gen, A, B, pro_A, pro_disc):
    a = pro_A.solution
    ndisc = pro_disc.solution
    return ((2*A.as_expr()*gen + B.as_expr())**2 + ndisc) / (4*a)


def _quadratic_real_C(gen, C, B, pro_C, pro_disc):
    c = pro_C.solution
    ndisc = pro_disc.solution
    return ((2*C.as_expr() + B.as_expr()*gen)**2 + ndisc*gen**2) / (4*c)


def _quadratic_lb_full(x, A, B, C, pro_A, LB, cond_disc, bounds, ineqs, eqs, SYM):
    preorder = [SYM] + list(ineqs.values())
    ideal = list(eqs.values())

    lb, lb_expr, _, __ = bounds

    NSYM = SYM # it stands for another symbol, but reusing it works as well

    # SYM = -2*A*lb - B
    # lb_expr = x - lb
    a, f_lb, ndisc = [_.solution for _ in [pro_A, LB, cond_disc]]

    # proof when NSYM >= 0
    p1 = a*(x - lb)**2 + NSYM*lb_expr + f_lb

    # proof when SYM >= 0
    p2 = ((2*A.as_expr()*x + B.as_expr())**2 + ndisc) / (4*a)

    p1, p2 = [PSatz.from_sympy(_, preorder, ideal) for _ in [p1, p2]]

    if p1 is None or p2 is None:
        return None

    ps = p1.join(p2, 0)
    return ps.as_expr() if ps is not None else None


def _quadratic_ub_full(x, A, B, C, pro_A, UB, cond_disc, bounds, ineqs, eqs, SYM):
    preorder = [SYM] + list(ineqs.values())
    ideal = list(eqs.values())

    _, __, ub, ub_expr = bounds

    NSYM = SYM # it stands for another symbol, but reusing it works as well

    # SYM = 2*A*lb + B
    # ub_expr = ub - x
    ub, ub_expr = bounds[1], x - ub
    a, f_ub, ndisc = [_.solution for _ in [pro_A, UB, cond_disc]]

    # proof when NSYM >= 0
    p1 = a*(x - ub)**2 + NSYM*ub_expr + f_ub

    # proof when SYM >= 0
    p2 = ((2*A.as_expr()*x + B.as_expr())**2 + ndisc) / (4*a)

    p1, p2 = [PSatz.from_sympy(_, preorder, ideal) for _ in [p1, p2]]
    if p1 is None or p2 is None:
        return None

    ps = p1.join(p2, 0)
    return ps.as_expr() if ps is not None else None