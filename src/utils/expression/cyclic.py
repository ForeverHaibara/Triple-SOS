from typing import List

from numbers import Number

import sympy as sp
from sympy.core.parameters import global_parameters
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.combinatorics import Permutation, PermutationGroup, CyclicGroup, SymmetricGroup
from sympy.printing.latex import LatexPrinter
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence_traditional, PRECEDENCE

from ..basis_generator import MonomialCyclic

def _leading_symbol(expr):
    if isinstance(expr, sp.Symbol):
        return expr
    if hasattr(expr, '_leading_symbol'):
        return expr._leading_symbol
    for arg in expr.args:
        result = _leading_symbol(arg)
        if result is not None:
            return result
    return None

def is_cyclic_expr(expr, symbols, perm):
    symbols = _std_seq(symbols, perm)
    if expr.free_symbols.isdisjoint(symbols):
        return True
    if hasattr(expr, '_eval_is_cyclic') and expr._eval_is_cyclic(symbols, perm):
        return True
    if isinstance(expr, (sp.Add, sp.Mul)):
        return all(is_cyclic_expr(arg, symbols, perm) for arg in expr.args)
    if isinstance(expr, sp.Pow) and expr.args[1].is_constant():
        return is_cyclic_expr(expr.args[0], symbols, perm)
    return False

def _std_seq(symbols, perm):
    ind = sorted(list(range(len(symbols))), key = lambda i: symbols[i].name)
    inv_ind = [0] * len(symbols)
    for i, j in enumerate(ind):
        inv_ind[j] = i
    p = min(map(lambda x: x(inv_ind), perm.generate()))
    # sorted_symbols = [symbols[i] for i in ind]
    # return tuple(sorted_symbols[i] for i in p)
    return tuple(symbols[ind[i]] for i in p)


class CyclicExpr(sp.Expr):
    """
    Represent cyclic expressions. The printing style for __str__ and __repr__
    can be configured by global variable CyclicExpr.PRINT_WITH_PARENS and CyclicExpr.PRINT_FULL.

    Every cyclic expression is defaultedly assumed to be cyclic with respect to (a, b, c).
    """
    PRINT_WITH_PARENS = False
    PRINT_FULL = False
    is_commutative = True
    def __new__(cls, expr, *args, evaluate = None):
        if evaluate is None:
            evaluate = global_parameters.evaluate

        if len(args) == 0:
            symbols, perm = sp.symbols('a b c'), None
        elif len(args) == 1:
            symbols, perm = args[0], None
        elif len(args) == 2:
            symbols, perm = args
        else:
            raise ValueError("Invalid arguments.")
        if not all(isinstance(_, sp.Symbol) for _ in symbols):
            raise ValueError("Second argument should be a tuple of sympy symbols.")

        if perm is None:
            perm = CyclicGroup(len(symbols))
        if evaluate:
            symbols = _std_seq(symbols, perm)
        symbols = sympify(tuple(symbols))


        if evaluate:
            expr0 = expr
            for translation in cls._generate_all_translations(symbols, perm):
                expr2 = expr0.xreplace(translation)
                if expr.compare(expr2) > 0:
                    expr = expr2

            return cls._eval_simplify_(expr, symbols, perm)

        obj = sp.Expr.__new__(cls, expr, symbols, perm)
        return obj

    @classmethod
    def _eval_simplify_(cls, *args, **kwargs):
        return sp.Expr.__new__(cls, *args, **kwargs)

    @property
    def symbols(self):
        return self.args[1]

    @property
    def perm(self):
        return self.args[2]

    def _eval_is_cyclic(self, symbols, perm):
        return self.symbols == symbols and self.perm == perm

    @classmethod
    def _generate_all_translations(cls, symbols, perm):
        for p in perm._elements:
            yield dict(zip(symbols, p(symbols)))

    def doit(self, **hints):
        # expand the cyclic expression
        perm = self.args[2]
        new_args = [None] *  perm.order()
        symbols = self.symbols
        for i, translation in enumerate(self._generate_all_translations(symbols, perm)):
            new_args[i] = self.args[0].xreplace(translation)
        expr = self.base_func(*new_args)
        if hints.get("deep", True):
            return expr.doit(**hints)
        return expr

    def _eval_expand_basic(self, **hints):
        return self.doit(**hints)

    def subs(self, *args, **kwargs):
        """
        Substitutes old for new in an expression after sympifying args. This will perform self.doit(),
        and the cyclic properties are not preserved. To preserve cyclic property, see xreplace.

        Examples
        ========

        >>> CyclicSum(a*(b-c)**2).subs({a:3, b:2, c:1})
        12
        >>> CyclicSum(a*(b-c)**2).subs({a:y+z, b:z+x, c:x+y})
        (-x + y)**2*(x + y) + (x - z)**2*(x + z) + (-y + z)**2*(y + z)
        >>> CyclicSum((x*a+y*b-CyclicSum(b))**2).subs({a:a*b, b:b*c, c:c*a}, simultaneous=True)
        (-a*b + a*c*y - a*c + b*c*x - b*c)**2 + (a*b*x - a*b - a*c + b*c*y - b*c)**2 + (a*b*y - a*b + a*c*x - a*c - b*c)**2

        See also
        ========

        xreplace
        """
        return super().subs(*args, **kwargs)
    
    def _eval_subs(self, old, new):
        return self.doit()._subs(old, new)

    def xreplace(self, *args, **kwargs):
        """
        Substitutes old for new in an expression after sympifying args. This does not perform self.doit().
        If the replacement rule is cyclic, then the cyclic expression will be preserved.

        Examples
        ========
        >>> CyclicSum.PRINT_FULL = True

        If the replacement rule is cyclic with respect to its permutation group,
        then the cyclic expression is preserved.

        >>> (CyclicSum(a*(b-c)**2)).xreplace({a:a*b, b:b*c, c:c*a})
        CyclicSum(a*b*(-a*c + b*c)**2, (a, b, c), PermutationGroup([
            (0 1 2)]))

        Xreplace also supports the case when the rule is not cylic with respect to its permutation group,
        but translating symbols to new symbols.

        >>> (CyclicSum(a**2*(b+CyclicSum(a)))).xreplace({a:x, b:y, c:z})
        CyclicSum(x**2*(y + CyclicSum(x, (x, y, z), PermutationGroup([
            (0 1 2)]))), (x, y, z), PermutationGroup([
            (0 1 2)]))

        When the replacements are not symbols, yet not cyclic with respect to its permutation group,
        an error will be raised. Use subs() to expand the cyclic expression and perform substitutions instead.

        >>> SymmetricSum(a**2, (a, b, c)).xreplace({a:a*b, b:b*c, c:c*a}) # doctest: +Raises(ValueError)
        """
        return super().xreplace(*args, **kwargs)

    def _xreplace(self, rule):
        arg0 = self.args[0]._xreplace(rule)
        if set(rule.keys()) == set(self.symbols):
            x = self.symbols[0]
            for translation in self._generate_all_translations(self.symbols, self.perm):
                y = translation[x]
                if rule[y] != rule[x].subs(translation, simultaneous=True):
                    break
            else:
                return self.func(arg0[0], *self.args[1:]), arg0[1]
        arg1 = self.args[1]._xreplace(rule)
        return self.func(arg0[0], arg1[0], self.args[2]), arg0[1] or arg1[1]


class CyclicSum(CyclicExpr):

    precedence = PRECEDENCE['Mul']
    base_func = sp.Add

    def __new__(cls, expr, *args, **kwargs):
        obj = CyclicExpr.__new__(cls, expr, *args, **kwargs)
        return obj

    @classmethod
    def str_latex(cls, printer, expr):
        s = printer._print(expr.args[0])
        if precedence_traditional(expr.args[0]) < cls.precedence:
            s = printer._add_parens(s)
        return r'\sum_{\mathrm{cyc}} ' + s

    @classmethod
    def str_str(cls, printer, expr):
        if cls.PRINT_FULL:
            return f"{cls.__name__}({', '.join(printer._print(arg) for arg in expr.args)})"
        s = printer._print(expr.args[0])
        if cls.PRINT_WITH_PARENS or precedence_traditional(expr.args[0]) < cls.precedence:
            s = '(%s)'%s
        return 'Σ' + s

    @classmethod
    def _eval_degenerate(cls, expr, symbols):
        return expr * len(symbols)

    @classmethod
    def _eval_simplify_(cls, expr, symbols, perm):
        if isinstance(expr, Number) or expr.free_symbols.isdisjoint(symbols):
            return cls._eval_degenerate(expr, symbols)

        if isinstance(expr, sp.Mul):
            cyc_args = []
            uncyc_args = []
            symbol_degrees = {}

            for arg in expr.args:
                if is_cyclic_expr(arg, symbols, perm):
                    cyc_args.append(arg)

                # elif isinstance(arg, sp.Symbol) and arg in symbols:
                #     # e.g. CyclicSum(a**2 * b * c) = CyclicSum(a) * CyclicProduct(a)
                #     symbol_degrees[arg] = 1
                # elif isinstance(arg, sp.Pow):
                #     arg2 = arg.args[0] 
                #     if isinstance(arg2, sp.Symbol) and arg2 in symbols and arg.args[1].is_constant():
                #         symbol_degrees[arg2] = arg.args[1]
                else:
                    uncyc_args.append(arg)
            
            # if len(symbol_degrees) == len(symbols):
            #     # all symbols appear at least once
            #     base = min(symbol_degrees.values())
            #     cyc_args.append(CyclicProduct(symbols[0] ** base, *symbols))

            if len(cyc_args) > 0:
                obj0 = cls(expr.func(*uncyc_args), symbols, perm)
                obj = sp.Mul(obj0, *cyc_args)
                return obj
        return sp.Expr.__new__(cls, expr, symbols, perm)

    def as_content_primitive(self, radical=False, clear=True):
        c, p = self.args[0].as_content_primitive(radical=radical, clear=clear)
        if c is S.One:
            return S.One, self
        return c, CyclicSum(p, *self.args[1:])



class CyclicProduct(CyclicExpr):

    precedence = PRECEDENCE['Mul']
    base_func = sp.Mul

    def __new__(cls, expr, *args, **kwargs):
        obj = CyclicExpr.__new__(cls, expr, *args, **kwargs)
        return obj

    @classmethod
    def str_latex(cls, printer, expr):
        s = printer._print(expr.args[0])
        if precedence_traditional(expr.args[0]) < cls.precedence:
            s = printer._add_parens(s)
        return r'\prod_{\mathrm{cyc}} ' + s

    @classmethod
    def str_str(cls, printer, expr):
        if cls.PRINT_FULL:
            return f"{cls.__name__}({', '.join(printer._print(arg) for arg in expr.args)})"

        s = printer._print(expr.args[0])
        if cls.PRINT_WITH_PARENS or precedence_traditional(expr.args[0]) < cls.precedence:
            s = '(%s)'%s
        return '∏' + s

    @classmethod
    def _eval_degenerate(cls, expr, symbols):
        return expr ** len(symbols)

    @classmethod
    def _eval_simplify_(cls, expr, symbols, perm):
        if isinstance(expr, Number) or expr.free_symbols.isdisjoint(symbols):
            return cls._eval_degenerate(expr, symbols)

        if isinstance(expr, sp.Mul):
            cyc_args = list(filter(lambda x: is_cyclic_expr(x, symbols, perm), expr.args))
            cyc_args = [arg ** len(symbols) for arg in cyc_args]
            uncyc_args = list(filter(lambda x: not is_cyclic_expr(x, symbols, perm), expr.args))
            obj0 = sp.Mul(*[cls(arg, symbols, perm) for arg in uncyc_args])
            obj = sp.Mul(obj0, *cyc_args)
            return obj
        return sp.Expr.__new__(cls, expr, symbols, perm)

    def as_content_primitive(self, radical=False, clear=True):
        c, p = self.args[0].as_content_primitive(radical=radical, clear=clear)
        if c is S.One:
            return S.One, self
        return c ** len(self.symbols), CyclicProduct(p, *self.args[1:])


def SymmetricSum(expr, symbols, **kwargs):
    """
    Shortcut to represent the symmetric sum of an expression with respect to all given symbols.
    """
    return CyclicSum(expr, symbols, SymmetricGroup(len(symbols)), **kwargs)

def SymmetricProduct(expr, symbols, **kwargs):
    """
    Shortcut to represent the symmetric product of an expression with respect to all given symbols.
    """
    return CyclicProduct(expr, symbols, SymmetricGroup(len(symbols)), **kwargs)


setattr(LatexPrinter, '_print_CyclicSum', lambda self, expr: CyclicSum.str_latex(self, expr))
setattr(LatexPrinter, '_print_CyclicProduct', lambda self, expr: CyclicProduct.str_latex(self, expr))

setattr(StrPrinter, '_print_CyclicSum', lambda self, expr: CyclicSum.str_str(self, expr))
setattr(StrPrinter, '_print_CyclicProduct', lambda self, expr: CyclicProduct.str_str(self, expr))

setattr(MonomialCyclic, 'cyclic_sum', lambda self, expr, gens: CyclicSum(expr, *gens))


if __name__ == '__main__':
    a,b,c,d = sp.symbols('a b c d')
    print(CyclicProduct(sp.S(4)*d))
    x = CyclicSum(sp.S(4)/7*b*c*(a*a-b*b+sp.S(23)/11*(a*b-a*c)+(b*c-a*b))**2) + CyclicProduct((a-b)**2)
    print(x.as_numer_denom())
    print(sp.latex(x))
    print(x.as_content_primitive())
    print(x.doit())