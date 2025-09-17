from typing import List, Dict, Tuple, Union, Set, Callable, Optional

import sympy as sp
from sympy import (Expr, Poly, Symbol, Rational, Integer,
    Min, Max, Abs, Pow,
    sin, cos, tan, cot, sec, csc,
)
from sympy.utilities.iterables import iterable

from .algebraic import CancelDenominator
from ..node import ProofNode

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



class ModelingHelper:
    Trigs = {sin, cos, tan, cot, sec, csc}
    def __init__(self, poly: Expr, ineq_constraints: Dict[Expr, Expr], eq_constraints: Dict[Expr, Expr]):
        self.poly = poly
        self.ineq_constraints = ineq_constraints
        self.eq_constraints = eq_constraints


        self.symbols = set(poly.free_symbols).union(
            *[ineq.free_symbols for ineq in ineq_constraints.keys()],
            *[ineq.free_symbols for ineq in ineq_constraints.values()],
            *[eq.free_symbols for eq in eq_constraints.keys()],
            *[eq.free_symbols for eq in eq_constraints.values()]
        )
        self.symbol_gen = _unique_symbol_generator(self.symbols)

        self._prepare_replace_trigs()

    def __iter__(self):
        return iter([self.poly] + list(self.ineq_constraints) + list(self.eq_constraints))

    def __len__(self):
        return len(self.poly) + len(self.ineq_constraints) + len(self.eq_constraints)

    def find(self, classes: Union[Callable, Set]):
        collection = []
        if iterable(classes):
            def recur_find(expr):
                if expr.__class__ in classes:
                    collection.append(expr)
                if not expr.is_Atom:
                    for _ in expr.args:
                        recur_find(_)                    
        else:
            def recur_find(expr):
                if classes(expr):
                    collection.append(expr)
                if not expr.is_Atom:
                    for _ in expr.args:
                        recur_find(_)
        for expr in self:
            recur_find(expr)
        return collection

    def _replace_Pow(self, x: Pow):
        if isinstance(x.exp, Integer):
            return None
        elif len(x.base.free_symbols) == 0 and len(x.exp.free_symbols) == 0:
            return None
        elif isinstance(x.exp, Rational):
            p, q = x.exp.numerator, x.exp.denominator
            if q == 1:
                return None

            z = self.symbol_gen('z')
            ineqs = {x.base: x.base, z: x} if q % 2 == 0 else {}
            eqs = {x.base**p - z**q: Integer(0)} if p > 0 else {z**q * x.base**(-p) - Integer(1): Integer(0)}
            return z, ineqs, eqs

        else:
            raise FormulationFailure

    def _replace_Min(self, x: Min):
        if len(x.args) == 0:
            return None
        elif len(x.args) == 1:
            return x, {}, {}
        z = self.symbol_gen('Min')
        ineqs = {i - z: i - x for i in x.args}
        eqs = {sp.prod(i - z for i in x.args): Integer(0)}
        return z, ineqs, eqs

    def _replace_Max(self, x: Max):
        if len(x.args) == 0:
            return None
        elif len(x.args) == 1:
            return x, {}, {}
        z = self.symbol_gen('Max')
        ineqs = {z - i: x - i for i in x.args}
        eqs = {sp.prod(z - i for i in x.args): Integer(0)}
        return z, ineqs, eqs

    def _replace_Abs(self, x: Abs):
        z = self.symbol_gen('Abs')
        return z, {z: x}, {z**2 - x.args[0]**2: Integer(0)}

    def _get_replacement_rule(self):     
        return {
            Pow: self._replace_Pow,
            Abs: self._replace_Abs,
            Min: self._replace_Min,
            Max: self._replace_Max,
            sin: self._replace_sin,
            cos: self._replace_cos,
            tan: self._replace_tan,
            cot: self._replace_cot,
            sec: self._replace_sec,
            csc: self._replace_csc,
        }

    def _prepare_replace_trigs(self):
        """Create a replacement rule for all trignometric variables."""
        self._trigs = self.find(self.Trigs)
        if len(self._trigs) == 0:
            return

        args = []
        atoms = {}
        for x in self._trigs:
            x_args = []
            for arg in sp.Add.make_args(x.args[0].expand()):
                arg = arg.as_content_primitive()
                if arg[1].could_extract_minus_sign():
                    arg = -arg[0], -arg[1]
                x_args.append(arg)

                if len(arg[1].free_symbols):
                    if arg[1] not in atoms:
                        atoms[arg[1]] = arg[0]
                    else:
                        coeff = atoms[arg[1]]
                        atoms[arg[1]] = sp.gcd(arg[0].numerator, coeff.numerator) \
                                        / sp.lcm(arg[0].denominator, coeff.denominator)
            args.append(x_args)

        cosines = {atom: self.symbol_gen('cos') for atom in atoms}
        sines = {atom: self.symbol_gen('sin') for atom in atoms}

        class ComplexVar:
            def __init__(self, x, y):
                self.x, self.y = x, y
            def __mul__(self, other):
                if isinstance(other, ComplexVar):
                    return ComplexVar(self.x * other.x - self.y * other.y, self.x * other.y + self.y * other.x)
            def __pow__(self, n):
                if n == 0:
                    return ComplexVar(Integer(1), Integer(0))
                if n == 1:
                    return self
                if n % 2 == 0:
                    return (self * self) ** (n // 2)
                if n % 2 == 1:
                    return self * ((self * self) ** ((n - 1) // 2))
            def conjugate(self):
                return ComplexVar(self.x, -self.y)

        def eiz(arg):
            mul = ComplexVar(Integer(1), Integer(0))
            for coeff, atom in arg:
                if atom in atoms:
                    coeff = coeff / atoms[atom]
                    if not isinstance(coeff, Integer):
                        raise FormulationFailure
                    coeff = int(coeff)
                    if coeff > 0:
                        mul = mul * (ComplexVar(cosines[atom], sines[atom])**coeff)
                    elif coeff < 0:
                        mul = mul * (ComplexVar(cosines[atom], -sines[atom])**(-coeff))
                else:
                    # atom is a constant, e.g., pi/6
                    mul = mul * ComplexVar(cos(coeff*atom), sin(coeff*atom))
            return mul

        trigs_real_and_imag = {}
        for x, expansion in zip(self._trigs, args):
            z = eiz(expansion)
            trigs_real_and_imag[x] = z.x, z.y

        self._trigs_var_cosine = {atoms[atom]*atom: cosines[atom] for atom in cosines}
        self._trigs_var_sine   = {atoms[atom]*atom: sines[atom] for atom in sines}
        self._trigs_real_and_imag = trigs_real_and_imag
        # print('Trigs var cosine =', self._trigs_var_cosine)
        # print('Trigs var sine =', self._trigs_var_sine)
        # print('Trigs real and imag =', self._trigs_real_and_imag)

    def _replace_cos(self, expr: cos):
        return self._trigs_real_and_imag[expr][0], {}, {}, {}

    def _replace_sin(self, expr: sin):
        return self._trigs_real_and_imag[expr][1], {}, {}, {}

    def _replace_tan(self, expr: tan):
        return self._trigs_real_and_imag[expr][1] / self._trigs_real_and_imag[expr][0], {}, {}, {}

    def _replace_cot(self, expr: cot):
        return self._trigs_real_and_imag[expr][0] / self._trigs_real_and_imag[expr][1], {}, {}, {}

    def _replace_sec(self, expr: sec):
        return Integer(1) / self._trigs_real_and_imag[expr][0], {}, {}, {}

    def _replace_csc(self, expr: csc):
        return Integer(1) / self._trigs_real_and_imag[expr][1], {}, {}, {}

    def replace_expr(self, expr: Expr, rule: Optional[Dict]=None):
        if isinstance(expr, Poly):
            return expr, {}, {}, {}
        if rule is None:
            rule = self._get_replacement_rule()
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
                new = f(expr)
                if new is not None:
                    inverse[new[0]] = expr.xreplace(inverse)
                    expr = new[0]
                    ineqs.update(new[1])
                    eqs.update(new[2])

            return expr, has_changed
        expr, has_changed = _recur_replace(expr)
        return expr, ineqs, eqs, inverse

    def formulate(self):
        # get all expressions that need to be replaced
        new_ineqs = {}
        new_eqs = {}
        inverse = {}

        # trignometric
        if len(self._trigs) > 0:
            for key in self._trigs_var_cosine.keys():
                c, s = self._trigs_var_cosine[key], self._trigs_var_sine[key]
                new_eqs[c**2 + s**2 - Integer(1)] = Integer(0)
                # new_ineqs[Integer(1) + c] = 2*cos(key/2)**2
                # new_ineqs[Integer(1) - c] = 2*sin(key/2)**2
                # new_ineqs[Integer(1) + s] = 2*cos(sp.pi/4 - key/2)**2
                # new_ineqs[Integer(1) - s] = 2*sin(sp.pi/4 - key/2)**2
                new_ineqs[Integer(1) + c] = (sin(key)**2 + (cos(key) + 1)**2)/2
                new_ineqs[Integer(1) - c] = (sin(key)**2 + (cos(key) - 1)**2)/2
                new_ineqs[Integer(1) + s] = (cos(key)**2 + (sin(key) + 1)**2)/2
                new_ineqs[Integer(1) - s] = (cos(key)**2 + (sin(key) - 1)**2)/2
                inverse[c] = cos(key)
                inverse[s] = sin(key)


        rule = self._get_replacement_rule()
        new_poly, ineqs, eqs, inv = self.replace_expr(self.poly, rule)
        new_ineqs.update(ineqs)
        new_eqs.update(eqs)
        inverse.update({k: v.xreplace(inverse) for k, v in inv.items()})

        for ineq, expr in self.ineq_constraints.items():
            new_ineq, ineqs, eqs, inv = self.replace_expr(ineq, rule)
            new_ineqs[new_ineq] = expr
            new_ineqs.update(ineqs)
            new_eqs.update(eqs)
            inverse.update({k: v.xreplace(inverse) for k, v in inv.items()})

        for eq, expr in self.eq_constraints.items():
            new_eq, ineqs, eqs, inv = self.replace_expr(eq, rule)
            new_eqs[new_eq] = expr
            new_ineqs.update(ineqs)
            new_eqs.update(eqs)
            inverse.update({k: v.xreplace(inverse) for k, v in inv.items()})
        return new_poly, new_ineqs, new_eqs, inverse


class ReformulateAlgebraic(ProofNode):
    inverse = None
    def explore(self, configs):
        if self.status == 0:
            problem = self.problem
            expr, ineq_constraints, eq_constraints = problem.expr, problem.ineq_constraints, problem.eq_constraints
            helper = ModelingHelper(expr, ineq_constraints, eq_constraints)
            new_expr, new_ineqs, new_eqs, inverse = helper.formulate()

            # print('Problem Reformulation')
            # print(f'Goal         : {new_expr}')
            # print(f'Inequalities : {new_ineqs}')
            # print(f'Equalities   : {new_eqs}')
            # print(f'Replacement  : {inverse}')

            self.children = [
                CancelDenominator(
                    self.new_problem(new_expr, new_ineqs, new_eqs)
                )
            ]
            self.inverse = inverse
            self.status = -1


    def update(self, *args, **kwargs):
        if self.children and self.children[0].problem.solution is not None:
            self.register_solution(self.children[0].problem.solution.xreplace(self.inverse))
