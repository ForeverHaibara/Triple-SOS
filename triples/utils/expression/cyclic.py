from importlib import import_module
from numbers import Number
from typing import List, Tuple, Union, Callable

import sympy as sp
from sympy.core.cache import cacheit
from sympy.core import sympify, S, Mul, Add, Symbol, Expr
from sympy.core.containers import Dict
from sympy.core.numbers import zoo, nan
from sympy.core.parameters import global_parameters
from sympy.combinatorics import Permutation, PermutationGroup, CyclicGroup, SymmetricGroup
from sympy.polys import cancel
from sympy.printing.latex import LatexPrinter
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence_traditional, PRECEDENCE
from sympy.simplify import signsimp


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
    if isinstance(expr, (Add, Mul)):
        return all(is_cyclic_expr(arg, symbols, perm) for arg in expr.args)
    if isinstance(expr, sp.Pow) and expr.args[1].is_constant():
        return is_cyclic_expr(expr.args[0], symbols, perm)
    return False

def _std_seq(symbols, perm_group):
    ind = sorted(list(range(len(symbols))), key = lambda i: symbols[i].name)
    inv_ind = [0] * len(symbols)
    for i, j in enumerate(ind):
        inv_ind[j] = i
    p = min(map(lambda x: x(inv_ind), perm_group.generate()))
    # sorted_symbols = [symbols[i] for i in ind]
    # return tuple(sorted_symbols[i] for i in p)
    return tuple(symbols[ind[i]] for i in p)

def _func_perm(func, expr, symbols, perm_group):
    new_args = [None] *  perm_group.order()
    symbols = symbols
    for i, translation in enumerate(CyclicExpr._generate_all_translations(symbols, perm_group)):
        new_args[i] = expr.xreplace(translation)
    expr = func(*new_args)
    return expr

def _is_same_dict(d1, d2, simpfunc=signsimp):
    if len(d1) != len(d2):
        return False
    d1 = dict((simpfunc(k), simpfunc(v)) for k, v in d1.items())
    d2 = dict((simpfunc(k), simpfunc(v)) for k, v in d2.items())
    for k, v in d1.items():
        if k not in d2 or (not simpfunc(d2[k]) == simpfunc(v)):
            return False
    return True

@cacheit
def _is_perm_invariant_dict(symbols, perm_group, d):
    """
    Check whether the dictionary is invariant under the permutation group.
    """
    for perm_dict in CyclicExpr._generate_all_translations(symbols, perm_group, full=False):
        # checking only generators is sufficient
        # perm_symbols = tuple(perm_dict[s] for s in rule.keys())
        perm_symbols = [sympify(s).subs(perm_dict, simultaneous=True) for s in d.keys()]
        permed_rule = [sympify(d[s]).subs(perm_dict, simultaneous=True) for s in d.keys()]
        permed_rule = dict(zip(perm_symbols, permed_rule))
        if not _is_same_dict(permed_rule, d):
            return False
    return True

def _project_perm_group(perm_group, inds):
    """
    Project the permutation group (especially stabilizers) to given indices,
    to reduce the degree of the permutation group.

    Examples
    ========
    >>> _project_perm_group(SymmetricGroup(5).stabilizer(3), [0,1,2,4])
    PermutationGroup([
        (3)(0 1),
        (0 3),
        (0 3 2 1),
        (0 1 2 3),
        (3)(1 2)])
    """
    projs = []
    mapping = dict(zip(inds, range(len(inds))))
    for p in perm_group.generators:
        q = []
        for i in p.array_form:
            v = mapping.get(i)
            if v is not None:
                q.append(v)
        projs.append(Permutation(q))
    return PermutationGroup(projs)


class CyclicExpr(sp.Expr):
    """
    Represent cyclic expressions. The printing style for __str__ and __repr__
    can be configured by global variable CyclicExpr.PRINT_WITH_PARENS and CyclicExpr.PRINT_FULL.

    This is a base class for CyclicSum and CyclicProduct, and should not be used directly.
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
        if perm.degree < 2 or perm.is_trivial:
            return expr

        if evaluate:
            symbols = _std_seq(symbols, perm)
        symbols = sympify(tuple(symbols))

        # set is_Atom to True to skip sympy simplification
        # symbols.is_Atom = True
        # perm.is_Atom = True

        if evaluate:
            expr = sympify(expr)
            expr0 = expr
            for translation in cls._generate_all_translations(symbols, perm, full=True):
                # find the simplest form up to permutation
                # expr2 = signsimp(expr0.xreplace(translation)) # signsimp is unstable
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
    def perm(self) -> PermutationGroup:
        return self.args[2]

    def _eval_is_cyclic(self, symbols, perm):
        return self.symbols == symbols and self.perm == perm

    @classmethod
    def _generate_all_translations(cls, symbols, perm, full=True):
        """
        Generate all possible translations of the symbols according to the permutation group.

        Parameters
        ==========
        symbols : tuple
            The symbols to be translated.
        perm : PermutationGroup
            The permutation group.
        full : bool
            If True, generate all possible translations. If False, generate only the generators.
        """
        for p in (perm.elements if full else perm.generators):
            yield dict(zip(symbols, p(symbols)))

    def doit(self, **hints):
        """
        Expand the cyclic expression.
        """
        expr = _func_perm(self.base_func, self.args[0], self.symbols, self.perm)
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
        >>> CyclicSum(a*x,(a,b,c,x,y,z),PermutationGroup(Permutation([(0,1,2),(3,4,5)]))).xreplace({x:u,y:v,z:w})
        CyclicSum(a*u, (a, b, c, u, v, w), PermutationGroup([
            (0 1 2)(3 4 5)]))

        When the replacements are not symbols, yet not cyclic with respect to its permutation group,
        the expression will be expanded.

        >>> SymmetricSum(a**2, (a, b, c)).xreplace({a:a*b, b:b*c, c:c*a})
        2*a**2*b**2 + 2*a**2*c**2 + 2*b**2*c**2
        """
        return super().xreplace(*args, **kwargs)

    def _xreplace(self, rule):
        if not isinstance(rule, (dict, Dict)):
            # might be a sympy Transform object
            return _func_perm(self.base_func, self.args[0], self.symbols, self.perm)._xreplace(rule)

        def fs(x):
            if hasattr(x, 'free_symbols'): return x.free_symbols
            return set()

        rule_vars = set.union(set(), *(fs(_) for _ in rule.keys()))
        if len(rule_vars.intersection(self.free_symbols)) == 0:
            # nothing has changed
            return self, False

        # first substitute the expression
        arg0, changed0 = self.args[0]._xreplace(rule)

        if all(signsimp(k) == signsimp(v) for k, v in rule.items()):
            # identical replacement, e.g. signsimp
            return self.func(arg0, self.symbols, self.perm), changed0

        self_vars = set(self.symbols)
        rule_vars = set.union(rule_vars, *(fs(_) for _ in rule.values()))
        changed_vars = rule_vars.intersection(self_vars)
        if len(changed_vars) == 0:
            # no symbol is changed, so we can preserve the cyclic property
            return self.func(arg0, self.args[1], self.args[2]), changed0

        # Case A. if we are replacing symbols to symbols
        for var in changed_vars:
            if not (var in rule and isinstance(rule[var], Symbol)):# and ((rule[var] not in self_vars) or rule[var] is var)):
                # the replacement is not cyclic with respect to the permutation group
                break
        else:
            # check other rule are not related to these symbols
            other_rule = [fs(k) | fs(v) for k, v in rule.items() if not k in changed_vars]
            if not any(_.intersection(self_vars) for _ in other_rule):
                new_symbols = tuple(rule.get(s, s) for s in self.symbols)
                if len(set(new_symbols)) == len(new_symbols): # distinct
                    return self.func(arg0, new_symbols, self.perm), changed0

        # Case B. if we are replacing symbols to f(symbols)...
        # we check whether the symmetry of the replacement rule agrees with the permutation group
        if _is_perm_invariant_dict(self.symbols, self.perm, rule):
            return self.func(arg0, self.symbols, self.perm), changed0

        # # fall back to the default implementation
        if len(changed_vars) >= len(self_vars) - 1:
            # every symbol is changed, so we can't preserve the cyclic property
            return _func_perm(self.base_func, self.args[0], self.symbols, self.perm)._xreplace(rule)
        else:
            # partial change, compute the stabilized subgroup
            changed_inds = tuple(i for i, s in enumerate(self.symbols) if s in changed_vars)
            unchanged_inds = tuple(i for i, s in enumerate(self.symbols) if s not in changed_vars)

            if len(unchanged_inds) < 2:
                return _func_perm(self.base_func, self.args[0], self.symbols, self.perm)._xreplace(rule)

            stab = self.perm.pointwise_stabilizer(list(changed_inds))
            stab_proj = _project_perm_group(stab, unchanged_inds)
            if stab_proj.is_trivial:
                return _func_perm(self.base_func, self.args[0], self.symbols, self.perm)._xreplace(rule)


            changed_vars = tuple(self.symbols[i] for i in changed_inds)
            unchanged_vars = tuple(self.symbols[i] for i in unchanged_inds)

            orbits = {changed_inds}
            moving_perms = [self.perm.identity]
            for p in self.perm.elements:
                # get where the changed variables are permuted to
                q = tuple(p.array_form[i] for i in changed_inds)
                if not (q in orbits):
                    moving_perms.append(p)
                    orbits.add(q)

            # sum up with respect to moving perms
            other_vars = tuple(s for s in rule.keys() if s not in self.symbols)
            other_dict = dict((s, rule.get(s, s)) for s in other_vars)
            new_args = []
            for p in moving_perms:
                trans = dict((self.symbols[i], rule.get(s, s)) for (s, i) in zip(self.symbols, p.array_form))
                trans.update(other_dict)
                new_args.append(self.func(self.args[0]._xreplace(trans)[0], unchanged_vars, stab_proj))
            return self.base_func(*new_args), True

        arg1, changed1 = self.args[1]._xreplace(rule)
        return self.func(arg0, arg1, self.args[2]), changed0 or changed1


    @property
    def is_cyclic_group(self):
        return self.perm.is_cyclic and self.perm.order() == self.perm.degree

    @property
    def is_symmetric_group(self):
        return self.perm.is_symmetric

    @property
    def is_alternating_group(self):
        return self.perm.is_alternating

    @property
    def is_dihedral_group(self):
        # is_dihedral is supported after https://github.com/sympy/sympy/pull/24384, sympy version 1.12
        return hasattr(self.perm, 'is_dihedral') and self.perm.is_dihedral and self.perm.order() == self.perm.degree * 2

class CyclicSum(CyclicExpr):
    """
    Represent cyclic sums.

    Examples
    ========    
    >>> CyclicSum.PRINT_FULL = True

    Every CyclicSum object is defined by an expression, a tuple of symbols, and a permutation group.
    >>> expr = CyclicSum(a*(b-c)**2, (a, b, c), PermutationGroup(Permutation([1,2,0]))); expr
    CyclicSum(a*(b - c)**2, (a, b, c), PermutationGroup([
        (0 1 2)]))

    Sums are simplified by choosing the lexicographically smallest representation of the summand
    and checking nested symmetries.
    >>> CyclicSum(z*y**2, (x, y, z), SymmetricGroup(3))
    CyclicSum(x*y**2, (x, y, z), PermutationGroup([
        (0 1 2),
        (2)(0 1)]))
    >>> CyclicSum(a*b*CyclicSum(a, (a, b, c), SymmetricGroup(3)), (a, b, c), SymmetricGroup(3))
    (CyclicSum(a, (a, b, c), PermutationGroup([
        (0 1 2),
        (2)(0 1)])))*(CyclicSum(a*b, (a, b, c), PermutationGroup([
        (0 1 2),
        (2)(0 1)])))
    >>> CyclicSum(1, (a, b, c, d), SymmetricGroup(4))
    24

    SymPy expressions containing cyclic sums can be expanded by calling doit().
    >>> expr.doit()
    a*(b - c)**2 + b*(-a + c)**2 + c*(a - b)**2

    When the permutation group is not specified, it is assumed to be the cyclic group.
    >>> CyclicSum(a*(b-c+d)**2, (a, b, c, d)).doit()
    a*(b - c + d)**2 + b*(a + c - d)**2 + c*(-a + b + d)**2 + d*(a - b + c)**2

    When neither the symbols nor the permutation group is specified, it assumes
    the cyclic sum is with respect to (a, b, c) and the cyclic group.
    >>> CyclicSum(a**3*b**2*c).doit()
    a**3*b**2*c + a**2*b*c**3 + a*b**3*c**2
    """

    precedence = PRECEDENCE['Mul']
    base_func = Add

    def __new__(cls, expr, *args, **kwargs):
        obj = CyclicExpr.__new__(cls, expr, *args, **kwargs)
        return obj

    @classmethod
    def _str_latex(cls, printer, expr):
        s = printer._print(expr.args[0])
        if precedence_traditional(expr.args[0]) < cls.precedence:
            s = printer._add_parens(s)
        cyc = r'\sum '
        if expr.is_cyclic_group:
            cyc = r'\sum_{\mathrm{cyc}} '
        elif expr.is_symmetric_group:
            cyc = r'\sum_{\mathrm{sym}} '
        elif expr.is_alternating_group:
            cyc = r'\sum_{\mathrm{alt}} '
        elif expr.is_dihedral_group:
            cyc = r'\sum_{\mathrm{dih}} '

        return cyc + s

    @classmethod
    def _str_str(cls, printer, expr):
        if cls.PRINT_FULL:
            return f"{cls.__name__}({', '.join(printer._print(arg) for arg in expr.args)})"
        s = printer._print(expr.args[0])
        if cls.PRINT_WITH_PARENS or precedence_traditional(expr.args[0]) < cls.precedence:
            s = '(%s)'%s
        return 'Σ' + s

    @classmethod
    def _eval_degenerate(cls, expr, perm):
        return expr * perm.order()

    @classmethod
    def _eval_simplify_(cls, expr, symbols, perm):
        if isinstance(expr, Number) or expr.free_symbols.isdisjoint(symbols):
            return cls._eval_degenerate(expr, perm)

        if isinstance(expr, Mul):
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
                obj = Mul(obj0, *cyc_args)
                return obj
        return sp.Expr.__new__(cls, expr, symbols, perm)

    def as_content_primitive(self, radical=False, clear=True):
        c, p = self.args[0].as_content_primitive(radical=radical, clear=clear)
        if c is S.One:
            return S.One, self
        return c, CyclicSum(p, *self.args[1:])



class CyclicProduct(CyclicExpr):
    """
    Represent cyclic products.

    Examples
    ========
    >>> CyclicProduct.PRINT_FULL = True

    Every CyclicProduct object is defined by an expression, a tuple of symbols, and a permutation group.
    >>> expr = CyclicProduct((a + b - c), (a, b, c), PermutationGroup(Permutation([1,2,0]))); expr
    CyclicProduct(a + b - c, (a, b, c), PermutationGroup([
        (0 1 2)]))

    Products are simplified by choosing the lexicographically smallest representation of the expression
    and checking nested symmetries.
    >>> CyclicProduct((y**2 + z), (x, y, z), SymmetricGroup(3))
    CyclicProduct(x + y**2, (x, y, z), PermutationGroup([
        (0 1 2),
        (2)(0 1)]))
    >>> CyclicProduct(a*(b - c)**2, (a, b, c), SymmetricGroup(3))
    (CyclicProduct(a, (a, b, c), PermutationGroup([
        (0 1 2),
        (2)(0 1)])))*(CyclicProduct((a - b)**2, (a, b, c), PermutationGroup([
        (0 1 2),
        (2)(0 1)])))
    >>> CyclicProduct(2, (a, b, c, d), SymmetricGroup(4))
    16777216

    SymPy expressions containing cyclic products can be expanded by calling doit().
    >>> expr.doit()
    (-a + b + c)*(a - b + c)*(a + b - c)

    When the permutation group is not specified, it is assumed to be the cyclic group.
    >>> CyclicProduct(a*(b-c+d)**2, (a, b, c, d)).doit()
    a*b*c*d*(-a + b + d)**2*(a - b + c)**2*(a + c - d)**2*(b - c + d)**2

    When neither the symbols nor the permutation group is specified, it assumes
    the cyclic product is with respect to (a, b, c) and the cyclic group.
    >>> CyclicProduct(a**3 + b**2 + c).doit()
    (a + b**3 + c**2)*(a**2 + b + c**3)*(a**3 + b**2 + c)
    """

    precedence = PRECEDENCE['Mul']
    base_func = Mul

    def __new__(cls, expr, *args, **kwargs):
        obj = CyclicExpr.__new__(cls, expr, *args, **kwargs)
        return obj

    @classmethod
    def _str_latex(cls, printer, expr):
        s = printer._print(expr.args[0])
        if precedence_traditional(expr.args[0]) < cls.precedence:
            s = printer._add_parens(s)

        cyc = r'\prod '
        if expr.is_cyclic_group:
            cyc = r'\prod_{\mathrm{cyc}} '
        elif expr.is_symmetric_group:
            cyc = r'\prod_{\mathrm{sym}} '
        elif expr.is_alternating_group:
            cyc = r'\prod_{\mathrm{alt}} '
        elif expr.is_dihedral_group:
            cyc = r'\prod_{\mathrm{dih}} '

        return cyc + s

    @classmethod
    def _str_str(cls, printer, expr):
        if cls.PRINT_FULL:
            return f"{cls.__name__}({', '.join(printer._print(arg) for arg in expr.args)})"

        s = printer._print(expr.args[0])
        if cls.PRINT_WITH_PARENS or precedence_traditional(expr.args[0]) < cls.precedence:
            s = '(%s)'%s
        return '∏' + s

    @classmethod
    def _eval_degenerate(cls, expr, perm):
        return expr ** perm.order()

    @classmethod
    def _eval_simplify_(cls, expr, symbols, perm):
        if isinstance(expr, Number) or expr.free_symbols.isdisjoint(symbols):
            return cls._eval_degenerate(expr, perm)

        if isinstance(expr, Mul):
            cyc_args = list(filter(lambda x: is_cyclic_expr(x, symbols, perm), expr.args))
            cyc_args = [arg ** len(symbols) for arg in cyc_args]
            uncyc_args = list(filter(lambda x: not is_cyclic_expr(x, symbols, perm), expr.args))
            obj0 = Mul(*[cls(arg, symbols, perm) for arg in uncyc_args])
            obj = Mul(obj0, *cyc_args)
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


setattr(LatexPrinter, '_print_CyclicSum', lambda self, expr: CyclicSum._str_latex(self, expr))
setattr(LatexPrinter, '_print_CyclicProduct', lambda self, expr: CyclicProduct._str_latex(self, expr))

setattr(StrPrinter, '_print_CyclicSum', lambda self, expr: CyclicSum._str_str(self, expr))
setattr(StrPrinter, '_print_CyclicProduct', lambda self, expr: CyclicProduct._str_str(self, expr))


if sp.__version__ < '1.14':
    def radsimp(expr, symbolic=True, max_terms=4):
        r"""
        Rationalize the denominator by removing square roots.

        Fix sympy.radsimp on non-Expr sympy objects for sympy < 1.14.
        See details in https://github.com/sympy/sympy/pull/26720
        """
        from collections import defaultdict
        from sympy.core.expr import Expr
        from sympy.core.exprtools import Factors, gcd_terms
        from sympy.core.function import _mexpand, expand_mul
        from sympy.core.sorting import ordered
        from sympy.core.symbol import symbols
        from sympy.core.mul import _unevaluated_Mul
        from sympy.functions import sqrt, log
        from sympy.simplify.radsimp import numer, denom, fraction, rad_rationalize
        from sympy.simplify.simplify import signsimp
        from sympy.simplify.sqrtdenest import sqrtdenest

        syms = symbols("a:d A:D")
        def _num(rterms):
            # return the multiplier that will simplify the expression described
            # by rterms [(sqrt arg, coeff), ... ]
            a, b, c, d, A, B, C, D = syms
            if len(rterms) == 2:
                reps = dict(list(zip([A, a, B, b], [j for i in rterms for j in i])))
                return (
                sqrt(A)*a - sqrt(B)*b).xreplace(reps)
            if len(rterms) == 3:
                reps = dict(list(zip([A, a, B, b, C, c], [j for i in rterms for j in i])))
                return (
                (sqrt(A)*a + sqrt(B)*b - sqrt(C)*c)*(2*sqrt(A)*sqrt(B)*a*b - A*a**2 -
                B*b**2 + C*c**2)).xreplace(reps)
            elif len(rterms) == 4:
                reps = dict(list(zip([A, a, B, b, C, c, D, d], [j for i in rterms for j in i])))
                return ((sqrt(A)*a + sqrt(B)*b - sqrt(C)*c - sqrt(D)*d)*(2*sqrt(A)*sqrt(B)*a*b
                    - A*a**2 - B*b**2 - 2*sqrt(C)*sqrt(D)*c*d + C*c**2 +
                    D*d**2)*(-8*sqrt(A)*sqrt(B)*sqrt(C)*sqrt(D)*a*b*c*d + A**2*a**4 -
                    2*A*B*a**2*b**2 - 2*A*C*a**2*c**2 - 2*A*D*a**2*d**2 + B**2*b**4 -
                    2*B*C*b**2*c**2 - 2*B*D*b**2*d**2 + C**2*c**4 - 2*C*D*c**2*d**2 +
                    D**2*d**4)).xreplace(reps)
            elif len(rterms) == 1:
                return sqrt(rterms[0][0])
            else:
                raise NotImplementedError

        def ispow2(d, log2=False):
            if not d.is_Pow:
                return False
            e = d.exp
            if e.is_Rational and e.q == 2 or symbolic and denom(e) == 2:
                return True
            if log2:
                q = 1
                if e.is_Rational:
                    q = e.q
                elif symbolic:
                    d = denom(e)
                    if d.is_Integer:
                        q = d
                if q != 1 and log(q, 2).is_Integer:
                    return True
            return False

        def handle(expr):
            # Handle first reduces to the case
            # expr = 1/d, where d is an add, or d is base**p/2.
            # We do this by recursively calling handle on each piece.
            from sympy.simplify.simplify import nsimplify

            if expr.is_Atom:
                return expr
            elif not isinstance(expr, Expr):
                return expr.func(*[handle(a) for a in expr.args])

            n, d = fraction(expr)

            if d.is_Atom and n.is_Atom:
                return expr
            elif not n.is_Atom:
                n = n.func(*[handle(a) for a in n.args])
                return _unevaluated_Mul(n, handle(1/d))
            elif n is not S.One:
                return _unevaluated_Mul(n, handle(1/d))
            elif d.is_Mul:
                return _unevaluated_Mul(*[handle(1/d) for d in d.args])

            # By this step, expr is 1/d, and d is not a mul.
            if not symbolic and d.free_symbols:
                return expr

            if ispow2(d):
                d2 = sqrtdenest(sqrt(d.base))**numer(d.exp)
                if d2 != d:
                    return handle(1/d2)
            elif d.is_Pow and (d.exp.is_integer or d.base.is_positive):
                # (1/d**i) = (1/d)**i
                return handle(1/d.base)**d.exp

            if not (d.is_Add or ispow2(d)):
                return 1/d.func(*[handle(a) for a in d.args])

            # handle 1/d treating d as an Add (though it may not be)

            keep = True  # keep changes that are made

            # flatten it and collect radicals after checking for special
            # conditions
            d = _mexpand(d)

            # did it change?
            if d.is_Atom:
                return 1/d

            # is it a number that might be handled easily?
            if d.is_number:
                _d = nsimplify(d)
                if _d.is_Number and _d.equals(d):
                    return 1/_d

            while True:
                # collect similar terms
                collected = defaultdict(list)
                for m in Add.make_args(d):  # d might have become non-Add
                    p2 = []
                    other = []
                    for i in Mul.make_args(m):
                        if ispow2(i, log2=True):
                            p2.append(i.base if i.exp is S.Half else i.base**(2*i.exp))
                        elif i is S.ImaginaryUnit:
                            p2.append(S.NegativeOne)
                        else:
                            other.append(i)
                    collected[tuple(ordered(p2))].append(Mul(*other))
                rterms = list(ordered(list(collected.items())))
                rterms = [(Mul(*i), Add(*j)) for i, j in rterms]
                nrad = len(rterms) - (1 if rterms[0][0] is S.One else 0)
                if nrad < 1:
                    break
                elif nrad > max_terms:
                    # there may have been invalid operations leading to this point
                    # so don't keep changes, e.g. this expression is troublesome
                    # in collecting terms so as not to raise the issue of 2834:
                    # r = sqrt(sqrt(5) + 5)
                    # eq = 1/(sqrt(5)*r + 2*sqrt(5)*sqrt(-sqrt(5) + 5) + 5*r)
                    keep = False
                    break
                if len(rterms) > 4:
                    # in general, only 4 terms can be removed with repeated squaring
                    # but other considerations can guide selection of radical terms
                    # so that radicals are removed
                    if all(x.is_Integer and (y**2).is_Rational for x, y in rterms):
                        nd, d = rad_rationalize(S.One, Add._from_args(
                            [sqrt(x)*y for x, y in rterms]))
                        n *= nd
                    else:
                        # is there anything else that might be attempted?
                        keep = False
                    break
                from sympy.simplify.powsimp import powsimp, powdenest

                num = powsimp(_num(rterms))
                n *= num
                d *= num
                d = powdenest(_mexpand(d), force=symbolic)
                if d.has(S.Zero, nan, zoo):
                    return expr
                if d.is_Atom:
                    break

            if not keep:
                return expr
            return _unevaluated_Mul(n, 1/d)

        if not isinstance(expr, Expr):
            return expr.func(*[radsimp(a, symbolic=symbolic, max_terms=max_terms) for a in expr.args])

        coeff, expr = expr.as_coeff_Add()
        expr = expr.normal()
        old = fraction(expr)
        n, d = fraction(handle(expr))
        if old != (n, d):
            if not d.is_Atom:
                was = (n, d)
                n = signsimp(n, evaluate=False)
                d = signsimp(d, evaluate=False)
                u = Factors(_unevaluated_Mul(n, 1/d))
                u = _unevaluated_Mul(*[k**v for k, v in u.factors.items()])
                n, d = fraction(u)
                if old == (n, d):
                    n, d = was
            n = expand_mul(n)
            if d.is_Number or d.is_Add:
                n2, d2 = fraction(gcd_terms(_unevaluated_Mul(n, 1/d)))
                if d2.is_Number or (d2.count_ops() <= d.count_ops()):
                    n, d = [signsimp(i) for i in (n2, d2)]
                    if n.is_Mul and n.args[0].is_Number:
                        n = n.func(*n.args)

        return coeff + _unevaluated_Mul(n, 1/d)


    _radsimp_module = import_module(".simplify.radsimp", "sympy")
    _old_radsimp = _radsimp_module.radsimp
    setattr(_radsimp_module, 'radsimp', radsimp)
    setattr(import_module(".simplify", "sympy"), 'radsimp', radsimp)
    setattr(sp, 'radsimp', radsimp)

try:
    # local hijack, representing cyclic sums and products using new classes
    from ..monomials import MonomialManager
    setattr(MonomialManager, 'cyclic_sum', lambda self, expr, gens=None: CyclicSum(expr, gens, self.perm_group))
except ImportError:
    pass


def _get_rewriting_replacement(symbols: Tuple[Symbol], perm_group: PermutationGroup) -> Callable[[Expr], Expr]:
    """
    Get a replacement function to convert all cyclic expressions to the default cyclic group.
    The replacement function can be used as sympy.Expr.replace(lambda x: isinstance(x, CyclicExpr), replacement))
    This is an internal function. Use `rewrite_symmetry` instead.
    """
    # if perm_group.is_trivial:
    #     return lambda x: x.doit()
    def replacement(x: Expr) -> Expr:
        if not x.has(CyclicExpr):
            return x
        if not isinstance(x, CyclicExpr):
            # we do not need to call recursion because sympy.Expr.replace will do it
            return x # .func(*[replacement(_) for _ in x.args])
        if x.args[1] == symbols:
            # if x.is_cyclic_group:
            #     return x
            # elif x.is_symmetric_group:
            #     a, b, c = symbols
            #     v = (signsimp(x.args[0]) + signsimp(x.args[0].xreplace({a:b,b:a}))).together()
            #     v = replacement(v)
            #     return x.func(v, x.args[1], perm_group)

            if x.args[2] == perm_group:
                # 1. if the expression is already with respect to the default cyclic group
                return x
            elif x.args[2].is_subgroup(perm_group):
                # 2. check whether the expression is symmetric with respect to the given permutation group
                # e.g. CyclicSum(a*(b-c)**2, (a,b,c), CyclicGroup(3)) is also symmetric with respect to SymmetricGroup(3)
                expr = x.doit(deep=False)
                expr2 = x.func(x.args[0], x.args[1], perm_group).doit(deep=False)
                mul = perm_group.order() // x.args[2].order()
                if isinstance(x, CyclicSum):
                    if signsimp(mul * expr - expr2) == 0:
                        # we only check signsimp rather than mul * expr == expr2
                        return x.func(x.args[0], x.args[1], perm_group) / mul
                # elif isinstance(x, CyclicProduct):
                #     if signsimp(expr**mul - expr2) == 0:
                #         return x.func(x.args[0], x.args[1], perm_group) ** (1/mul)
            elif perm_group.is_subgroup(x.args[2]):
                # 3. check whether the given permutation group is a subgroup of the expression's permutation group
                transversals = x.args[2].coset_transversal(perm_group)
                translations = [dict(zip(x.args[1], p(x.args[1]))) for p in transversals]
                trans_perm_group = [dict(zip(x.args[1], p(x.args[1]))) for p in perm_group.elements]
                exprs = [x.args[0].xreplace(t) for t in translations]
                for i, expr in enumerate(exprs):
                    for t in trans_perm_group:
                        # find the simplest form up to permutation
                        expr2 = (expr.xreplace(t))
                        if expr.compare(expr2) > 0:
                            expr = expr2
                    exprs[i] = expr
                merged_expr = x.base_func(*(expr for expr in exprs)).together()
                merged_expr = x.func(merged_expr, x.args[1], perm_group)
                return merged_expr

        return x.doit(deep=False)
    return replacement

def rewrite_symmetry(expr: Expr, symbols: Tuple[Symbol], perm_group: PermutationGroup) -> Expr:
    """
    Rewrite the expression heuristically with respect to the given permutation group.
    After rewriting, it is expected all cyclic expressions are expanded or in the given permutation group.
    This avoids the ambiguity of the cyclic expressions.

    The process is done inplace and is not reversible. For reversible transformation,
    call `rewrite_symmetry` on the solution expression directly or make a copy of the solution.

    Parameters
    ----------
    symbols : Tuple[sp.Symbol]
        The symbols that the permutation group acts on.
    perm_group : PermutationGroup
        Sympy permutation group object.

    Returns
    ----------
    expr : Expr
        The rewritten expression.

    Examples
    ----------
    >>> from sympy.combinatorics import CyclicGroup, DihedralGroup, SymmetricGroup, PermutationGroup
    >>> from sympy.abc import a, b, c, d
    >>> expr = CyclicSum((a*c-b*d)**2, (a,b,c,d), DihedralGroup(4)) + CyclicProduct((a-b)**2, (a,b,c,d), SymmetricGroup(4))
    >>> rewritten = rewrite_symmetry(expr, (a,b,c,d), CyclicGroup(4))
    >>> rewritten
    (∏(a - b)**4)*(∏(a - c)**4)*(∏(a - d)**4) + 2*(Σ(a*c - b*d)**2)
    >>> rewritten.find(PermutationGroup)
    {PermutationGroup([
     (0 1 2 3)])}
    """
    return expr.replace(lambda x: isinstance(x, CyclicExpr), _get_rewriting_replacement(symbols, perm_group))