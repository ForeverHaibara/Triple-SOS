from typing import List, Dict, Tuple, Union, Optional

from sympy import Expr, Poly, Rational, Integer, Mul, Symbol, true

from ..problem import InequalityProblem
from ...utils import CyclicExpr

def sign_sos(expr: Expr, signs: Dict[Symbol, Tuple[int, Expr]]):
    """
    Very fast and simple nonnegativity check for a SymPy (commutative, real)
    expression instance given signs of symbols.
    """
    def is_nonneg_pow(x: Expr) -> bool:
        if isinstance(x.exp, Rational) and (int(x.exp.numerator) % 2 == 0 or
                int(x.exp.denominator) % 2 == 0):
            return True
        return False

    def prove_by_recur(expr: Expr) -> Optional[Tuple[Expr, bool]]:
        """
        Returns the proof `new_expr` such that expr == new_expr >= 0 and whether
        `new_expr` is not `expr`. The second argument tracks whether the expr
        has changed.
        """
        if isinstance(expr, Rational):
            if expr >= 0:
                return expr, False
            return None
        elif expr.is_Symbol:
            if signs.get(expr, (0, None))[0] == 1:
                v = signs[expr][1]
                return v, v != expr
            return None
        elif expr.is_Pow:
            if is_nonneg_pow(expr):
                return expr, False
            sol = prove_by_recur(expr.base)
            if sol is not None:
                v, changed = sol
                if changed:
                    return v ** expr.exp, True
                return expr, False
            return None
        elif expr.is_Add or expr.is_Mul:
            nonneg = []
            for arg in expr.args:
                nonneg.append(prove_by_recur(arg))
                if nonneg[-1] is None:
                    return None
            changed = any([_[1] for _ in nonneg])
            if changed:
                return expr.func(*[_[0] for _ in nonneg]), True
            return expr, False
        elif isinstance(expr, CyclicExpr):
            arg = expr.args[0]
            mulargs = []
            if arg.is_Pow:
                mulargs = [arg]
            elif arg.is_Mul:
                mulargs = arg.args
            def single(x):
                if x.is_Pow and is_nonneg_pow(x):
                    return True
                if isinstance(x, Rational) and x >= 0:
                    return True
                return False
            if len(mulargs) and all(single(_) for _ in mulargs):
                return expr, False

            # TODO: make it nicer
            # NOTE: calling doit(deep=False) to expand is not equivalent to generating
            # all permutations. E.g.
            # `CyclicProduct((a-b),(a,b,c,d),AlternatingGroup(4))`
            # is nonnegative after expanding. However, it is undetermined termwise.
            sol = prove_by_recur(expr.doit(deep=False))
            if sol is not None:
                return sol[0], True
            return None

        if len(expr.free_symbols) == 0:
            # e.g. (sqrt(2) - 1)
            sgn = (expr >= 0)
            if sgn in (true, True):
                return expr, False
            return None

    sol = prove_by_recur(expr)
    if sol is not None:
        return sol[0]


def get_symbol_signs(problem: InequalityProblem) -> Dict[Symbol, Tuple[int, Expr]]:
    """
    Infer the signs of each symbol in the problem given inequality
    and equality constraints. It can also be called by the class method
    `InequalityProblem.get_symbol_signs()`.

    The inference is heuristic and incomplete.

    Returns
    -------
    Dict[Symbol, Tuple[int, Expr]]
        A dictionary mapping each symbol to a tuple of its sign and a
        representative expression. The sign is 1 if the symbol is nonnegative,
        -1 if the symbol is nonpositive, and 0 if the symbol is zero.
        If 1 or -1, Expr is an nonnegative expression equal to the ABSOLUTE VALUE
        of the symbol. If 0, Expr is an expression of zero.
    """

    eq_constraints, ineq_constraints = problem.eq_constraints, problem.ineq_constraints

    fs0 = tuple(problem.free_symbols)
    signs = {s: (None, None) for s in fs0}
    for eq, expr in eq_constraints.items():
        if not isinstance(eq, Poly):
            eq = Poly(eq, *fs0)
        fs = eq.gens

        monoms = eq.monoms()
        if len(monoms) == 1:
            # x1**d1 * x2**d2 * ... == 0
            lm = monoms[0]
            c1 = eq.coeff_monomial(lm)
            nnz = [i for i in range(len(lm)) if lm[i] != 0]
            if len(nnz) == 1:
                signs[fs[nnz[0]]] = (0, (expr / c1)**Rational(1, lm[nnz[0]]))
        elif len(monoms) == 2:
            # c1 * x1**d1 * x2**d2 * ... + c2* x1**e1 * x2**e2 * ... == expr >= 0
            lm = monoms[0]
            tm = monoms[1]
            nnz1 = [i for i in range(len(lm)) if lm[i] != 0]
            nnz2 = [i for i in range(len(tm)) if tm[i] != 0]
            # if len(nnz1) == 0 or len(nnz2) == 0 or \
            #     (len(nnz1) == 1 and len(nnz2) == 1 and nnz1[0] == nnz2[0]):
            #     d1 = lm[nnz1[0]] if len(nnz1) else 0
            #     d2 = tm[nnz2[0]] if len(nnz2) else 0
            #     if (d1 - d2) % 2 == 1:
            #         # e.g. x^5 - 3*x^2 = 0  => x = 0 or x = 3^(1/3)
            #         c1 = eq.coeff_monomial(lm)
            #         c2 = eq.coeff_monomial(tm)
            #         sgn = bool(c1 > 0)^bool(c2 > 0)
            #         sgn = 1 if sgn else -1
            #         expr2 = sgn*(-sgn * c2 / c1)**Rational(1, d1 - d2)
            #         signs[fs[nnz1[0]]] = (sgn, expr2)


    for ineq, expr in ineq_constraints.items():
        if not isinstance(ineq, Poly):
            ineq = Poly(ineq, *fs0)
        fs = ineq.gens

        monoms = ineq.monoms()
        c1 = 0
        if len(monoms) == 1:
            c1 = ineq.coeff_monomial(monoms[0])

        # if len(monoms) == 2:
        #     # c1 * t1 + c2 * t2 = expr >= 0
        #     # equivalent to (c1/(c2*sgn))*(t1/t2) = expr/(c2*sgn*t2) - sgn >= 0
        #     m1, m2 = monoms
        #     c1 = ineq.coeff_monomial(m1)
        #     c2 = ineq.coeff_monomial(m2)
        #     monom_diff = tuple([m1[i] - m2[i] for i in range(len(m1))])
        #     sgn = 1 if bool(c2 > 0) else -1
        #     c1 = c1 / (sgn * c2)
        #     expr = expr / (sgn * c2 * Mul(*[fs[i]**m2[i] for i in range(len(m2))])) - sgn
        #     monoms = [monom_diff]
        #     # now it goes to the len(monoms) == 1 case
        if len(monoms) == 1:
            # c1 * x1**d1 * x2**d2 * ... >= 0
            lm = monoms[0]
            odds = [i for i in range(len(lm)) if lm[i] % 2 == 1]  
            if len(odds) == 1:
                sgn = 1 if bool(c1 > 0) else -1
                other = Mul(sgn*c1, *[fs[i]**lm[i] for i in range(len(lm)) if i != odds[0]])
                signs[fs[odds[0]]] = (sgn, sgn*(expr / other)**Rational(1, lm[odds[0]]))
    return signs