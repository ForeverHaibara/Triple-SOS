from typing import List

from numbers import Number

import sympy as sp
from sympy.core.parameters import global_parameters
from sympy.core.singleton import S
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

def is_cyclic_expr(expr, symbols):
    if expr.free_symbols.isdisjoint(symbols):
        return True
    if hasattr(expr, '_eval_is_cyclic') and expr._eval_is_cyclic(symbols):
        return True
    if isinstance(expr, (sp.Add, sp.Mul)):
        return all(is_cyclic_expr(arg, symbols) for arg in expr.args)
    if isinstance(expr, sp.Pow) and expr.args[1].is_constant():
        return is_cyclic_expr(expr.args[0], symbols)
    return False


class CyclicExpr(sp.Expr):
    PRINT_WITH_PARENS = False
    is_commutative = True
    def __new__(cls, expr, *symbols, evaluate = None):
        if evaluate is None:
            evaluate = global_parameters.evaluate

        if isinstance(symbols, tuple) and len(symbols) == 0:
            # default symbols
            symbols = sp.symbols('a b c')
        elif isinstance(symbols, str):
            symbols = sp.symbols(symbols)
        elif hasattr(symbols, '__len__') and len(symbols) == 1:
            symbols = symbols[0]

        if evaluate:
            symbol = _leading_symbol(expr)
            if symbol is not None and symbol in symbols:
                index = symbols.index(symbol)
                translation = cls._eval_get_translation(symbols, index)
                expr = cls._eval_symbol_translation(expr, translation)

            return cls._eval_simplify_(expr, symbols)

        obj = sp.Expr.__new__(cls, expr, *symbols)
        return obj

    @property
    def symbols(self):
        return self.args[1:]

    @property
    def _leading_symbol(self):
        return self.args[1]

    def _eval_is_cyclic(self, symbols):
        return self.symbols == symbols

    @classmethod
    def _eval_get_translation(cls, symbols, index):
        return {symbols[(i + index) % len(symbols)]: symbols[i] for i in range(len(symbols))}

    @classmethod
    def _eval_symbol_translation(cls, expr, translation):
        if isinstance(expr, sp.Symbol):
            return translation.get(expr, expr)
        if len(expr.args) == 0:
            return expr
        args = []
        for arg in expr.args:
            args.append(cls._eval_symbol_translation(arg, translation))
        return expr.func(*args)

    def doit(self):
        # expand the cyclic expression
        new_args = [None] * len(self.symbols)
        new_args[0] = self.args[0]
        for i in range(1, len(self.symbols)):
            translation = self._eval_get_translation(self.symbols, i)
            new_args[i] = self._eval_symbol_translation(self.args[0], translation)
        return self.base_func(*new_args).doit()

    def _xreplace(self, rule):
        """
        Xreplace will not replace the symbols.
        """
        return self.func(self.args[0].xreplace(rule), *self.symbols), True


class CyclicSum(CyclicExpr):

    precedence = PRECEDENCE['Mul']
    base_func = sp.Add

    def __new__(cls, expr, *symbols, evaluate = None):
        obj = CyclicExpr.__new__(cls, expr, *symbols, evaluate = evaluate)
        return obj

    @classmethod
    def str_latex(cls, printer, expr):
        s = printer._print(expr.args[0])
        if precedence_traditional(expr.args[0]) < cls.precedence:
            s = printer._add_parens(s)
        return r'\sum_{\mathrm{cyc}} ' + s

    @classmethod
    def str_str(cls, printer, expr):
        s = printer._print(expr.args[0])
        if cls.PRINT_WITH_PARENS or precedence_traditional(expr.args[0]) < cls.precedence:
            s = '(%s)'%s
        return 'Σ' + s

    @classmethod
    def _eval_degenerate(cls, expr, symbols):
        return expr * len(symbols)

    @classmethod
    def _eval_simplify_(cls, expr, symbols):
        if isinstance(expr, Number) or expr.free_symbols.isdisjoint(symbols):
            return cls._eval_degenerate(expr, symbols)

        if isinstance(expr, sp.Mul):
            cyc_args = []
            uncyc_args = []
            symbol_degrees = {}

            for arg in expr.args:
                if is_cyclic_expr(arg, symbols):
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
                obj0 = cls(expr.func(*uncyc_args), *symbols)
                obj = sp.Mul(obj0, *cyc_args)
                return obj
        return sp.Expr.__new__(cls, expr, *symbols)

    def as_content_primitive(self, radical=False, clear=True):
        c, p = self.args[0].as_content_primitive(radical=radical, clear=clear)
        if c is S.One:
            return S.One, self
        return c, CyclicSum(p, *self.symbols)



class CyclicProduct(CyclicExpr):

    precedence = PRECEDENCE['Mul']
    base_func = sp.Mul

    def __new__(cls, expr, *symbols, evaluate = None):
        obj = CyclicExpr.__new__(cls, expr, *symbols, evaluate = evaluate)
        return obj

    @classmethod
    def str_latex(cls, printer, expr):
        s = printer._print(expr.args[0])
        if precedence_traditional(expr.args[0]) < cls.precedence:
            s = printer._add_parens(s)
        return r'\prod_{\mathrm{cyc}} ' + s

    @classmethod
    def str_str(cls, printer, expr):
        s = printer._print(expr.args[0])
        if cls.PRINT_WITH_PARENS or precedence_traditional(expr.args[0]) < cls.precedence:
            s = '(%s)'%s
        return '∏' + s

    @classmethod
    def _eval_degenerate(cls, expr, symbols):
        return expr ** len(symbols)

    @classmethod
    def _eval_simplify_(cls, expr, symbols):
        if isinstance(expr, Number) or expr.free_symbols.isdisjoint(symbols):
            return cls._eval_degenerate(expr, symbols)

        if isinstance(expr, sp.Mul):
            cyc_args = list(filter(lambda x: is_cyclic_expr(x, symbols), expr.args))
            cyc_args = [arg ** len(symbols) for arg in cyc_args]
            uncyc_args = list(filter(lambda x: not is_cyclic_expr(x, symbols), expr.args))
            obj0 = sp.Mul(*[cls(arg, *symbols) for arg in uncyc_args])
            obj = sp.Mul(obj0, *cyc_args)
            return obj
        return sp.Expr.__new__(cls, expr, *symbols)

    def as_content_primitive(self, radical=False, clear=True):
        c, p = self.args[0].as_content_primitive(radical=radical, clear=clear)
        if c is S.One:
            return S.One, self
        return c ** len(self.symbols), CyclicProduct(p, *self.symbols)


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
    print( x.as_content_primitive())
    print(x.doit())