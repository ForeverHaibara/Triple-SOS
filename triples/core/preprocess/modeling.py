from functools import wraps
from typing import List, Dict, Tuple, Union

import sympy as sp
from sympy import Expr, Poly, Symbol, Rational, Integer, Min, Max, Abs, Pow

class _unique_symbol_generator:
    def __init__(self, symbols: Tuple[Symbol,...]):
        self.symbols = set(symbols)
        self.symbol_indices = {}
    def __call__(self, prefix: str = 's') -> Symbol:
        i = self.symbol_indices.get(prefix, 0) + 1
        while True:
            if prefix + str(i) in self.symbols:
                i += 1
            else:
                self.symbol_indices[prefix] = i
                s = Symbol(prefix + str(i))
                self.symbols.add(s)
                return s


class FormulationFailure(Exception):
    pass

class _replacement_rule:
    @classmethod
    def _replace_Pow(cls, x: Pow, gen: _unique_symbol_generator):
        if isinstance(x.exp, Integer):
            return None
        elif len(x.base.free_symbols) == 0 and len(x.exp.free_symbols) == 0:
            return None
        elif isinstance(x.exp, Rational):
            p, q = x.exp.numerator, x.exp.denominator
            if q == 1:
                return None

            z = gen('z')
            ineqs = {x.base: x.base, z: x} if q % 2 == 0 else {}
            eqs = {x.base**p - z**q: Integer(0)} if p > 0 else {z**q * x.base**(-p) - Integer(1): Integer(0)}
            return z, ineqs, eqs

        else:
            raise FormulationFailure

    @classmethod
    def _replace_Min(cls, x: Min, gen: _unique_symbol_generator):
        if len(x.args) == 0:
            return None
        elif len(x.args) == 1:
            return x, {}, {}
        z = gen('Min')
        ineqs = {i - z: i - x for i in x.args}
        eqs = {sp.prod(i - z for i in x.args): Integer(0)}
        return z, ineqs, eqs

    @classmethod
    def _replace_Max(cls, x: Max, gen: _unique_symbol_generator):
        if len(x.args) == 0:
            return None
        elif len(x.args) == 1:
            return x, {}, {}
        z = gen('Max')
        ineqs = {z - i: z - x for i in x.args}
        eqs = {sp.prod(z - i for i in x.args): Integer(0)}
        return z, ineqs, eqs

    @classmethod
    def _replace_Abs(cls, x: Abs, gen: _unique_symbol_generator):
        z = gen('Abs')
        return z, {z: x}, {z**2 - x.args[0]**2: Integer(0)}

REPLACEMENT = {
    Pow: _replacement_rule._replace_Pow,
    Abs: _replacement_rule._replace_Abs,
    Min: _replacement_rule._replace_Min,
    Max: _replacement_rule._replace_Max,
}



def replace_expr(expr: Expr, gen: _unique_symbol_generator, rule: Dict=REPLACEMENT):
    if isinstance(expr, Poly):
        return expr, {}, {}, {}
    expr = expr.doit()
    ineqs, eqs, inverse = {}, {}, {}
    def _recur_replace(expr):
        has_changed = False
        if not expr.is_Atom:
            args = [None] * len(expr.args)
            for i, arg in enumerate(expr.args):
                new, changed = _recur_replace(arg)
                args[i] = new
                has_changed = has_changed or changed
            if has_changed:
                expr = expr.func(*args)
 
        f = rule.get(expr.__class__)
        if f is not None:
            has_changed = True
            new = f(expr, gen)
            if new is not None:
                inverse[new[0]] = expr
                expr = new[0]
                ineqs.update(new[1])
                eqs.update(new[2])

        return expr, has_changed
    expr, has_changed = _recur_replace(expr)
    return expr, ineqs, eqs, inverse


def handle_general_expr(

):
    """
    Decorator factory to handle general sympy expressions and convert them to algebraic
    or rational expressions. For example, the expression Abs(x) is converted to a symbol "y"
    with constraints y >= 0 and y**2 - x**2 = 0.

    Noncommutative: (TODO)

    Commutative:
        Complex variables (TODO)

        Pow, Min, Max, Abs

        Piecewise (TODO)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(poly: Expr,
                ineq_constraints: Dict[Expr, Expr] = {},
                eq_constraints: Dict[Expr, Expr] = {}, *args, **kwargs):
            if isinstance(poly, Poly):
                pass

            symbols = set(poly.free_symbols).union(
                *[ineq.free_symbols for ineq in ineq_constraints],
                *[eq.free_symbols for eq in eq_constraints]
            )
            symbol_gen = _unique_symbol_generator(symbols)

            # get all expressions that need to be replaced
            new_ineqs = {}
            new_eqs = {}
            inverse = {}

            new_poly, ineqs, eqs, inv = replace_expr(poly, symbol_gen)
            new_ineqs.update(ineqs)
            new_eqs.update(eqs)
            inverse.update(inv)

            for ineq, expr in ineq_constraints.items():
                new_ineq, ineqs, eqs, inv = replace_expr(ineq, symbol_gen)
                new_ineqs[new_ineq] = expr
                new_ineqs.update(ineqs)
                new_eqs.update(eqs)
                inverse.update(inv)

            for eq, expr in eq_constraints.items():
                new_eq, ineqs, eqs, inv = replace_expr(eq, symbol_gen)
                new_eqs[new_eq] = expr
                new_ineqs.update(ineqs)
                new_eqs.update(eqs)
                inverse.update(inv)

            if len(inverse) and float(kwargs.get('verbose', 0)) > 0:
                print('Problem Reformulation')
                print(f'Goal         : {new_poly}')
                print(f'Inequalities : {new_ineqs}')
                print(f'Equalities   : {new_eqs}')
                print(f'Replacement  : {inverse}')

            sol = func(new_poly, new_ineqs, new_eqs, *args, **kwargs)
            if sol is not None:
                sol.problem = poly
                sol.ineq_constraints = ineq_constraints
                sol.eq_constraints = eq_constraints
                if len(inverse):
                    sol.solution = sol.solution.xreplace(inverse)
            return sol
        return wrapper
    return decorator