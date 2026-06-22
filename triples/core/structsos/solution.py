from warnings import warn

warn("SolutionStructural is deprecated, please use Solution instead.",
     stacklevel=2, category=DeprecationWarning)

from typing import Optional, Callable

from sympy import Expr, Symbol, Add, Mul, Pow, Integer, Rational, Function

from ..solution import Solution
from ...utils import CyclicSum, CyclicProduct

class _rewriting_exception(Exception): ...


class SolutionStructural(Solution):
    method = 'StructuralSOS'
    _verified = True # ...?
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self._verified:
            self._is_equal = bool(self.as_eq().simplify())

    @classmethod
    def _extract_nonnegative_exprs(cls, expr: Expr, func_name: str = "_G", extra_checker: Optional[Callable] = None):
        """
        Raw output of StructuralSOS might assume nonnegativity of some symbols,
        we extract these symbols and replace them with _F(x) for further processing.
        This is not intended to be used by end users.
        """
        # extract symbol constraints from dict
        # mapping = dict((s.as_expr(), v) for s, v in ineq_constraints.items() if \
        #             isinstance(s, Symbol) or (isinstance(s, Poly) and s.is_monomial and len(s.free_symbols) == 1 and s.LC() > 0))
        # mapping.update(dict((s.as_expr(), v) for s, v in eq_constraints.items() if s.is_monomial and len(s.free_symbols) == 1 and s.LC() > 0))

        # TODO: Handle symbols that represent zero?
        func = Function(func_name)
        if extra_checker is None:
            extra_checker = lambda x: None
        def dfs(arg):
            checked = extra_checker(arg)
            if checked is not None:
                return checked
            if isinstance(arg, Expr):
                if len(arg.free_symbols) == 0:
                    # constants might be Add, etc., e.g. 1+sqrt(2)
                    # however, using .is_constant() is very slow
                    if arg < 0:
                        raise _rewriting_exception
                elif isinstance(arg, Symbol):
                    return func(arg)
                    # v = mapping.get(arg)
                    # if v is not None:
                    #     return v
                    # raise _rewriting_exception
                elif isinstance(arg, (Add, Mul)):
                    return arg.func(*(dfs(_) for _ in arg.args))
                elif isinstance(arg, Pow):
                    base, exp = arg.as_base_exp()
                    if isinstance(exp, Integer):
                        if exp % 2 == 0:
                            return arg
                        elif exp == -1:
                            return 1 / dfs(base)
                        elif exp > 0:
                            return dfs(base)*Pow(base, exp - 1, evaluate=False)
                        else:
                            return Pow(base, exp + 1, evaluate=False) / dfs(base)
                    elif isinstance(base, Rational):
                        if exp.p % 2 == 0:
                            return arg
                    return dfs(base)**exp
                elif isinstance(arg, (CyclicSum, CyclicProduct)):
                    base = arg.args[0]
                    def is_pow2(x):
                        if isinstance(x, Pow):
                            if isinstance(x.exp, Rational) and x.exp.p % 2 == 0:
                                return True
                        #     elif isinstance(x.base, Symbol) and mapping.get(x.base) is not None:
                        #         return True
                        # elif isinstance(x, Symbol) and mapping.get(x) is not None:
                        #     return True
                        elif len(x.free_symbols) == 0 and x >= 0:
                            return True
                        return False
                    if is_pow2(base): # easy case where we do not need to expand
                        return arg
                    elif isinstance(base, Mul) and all(is_pow2(_) for _ in base.args):
                        return arg
                    # ensure each arg is nonnegative by expanding
                    each_arg = [dfs(_) for _ in arg.doit(deep=False).args]
                    assert len(each_arg) >= 0
                    return arg.func(dfs(base), *arg.args[1:])
            return arg

        try:
            new_expr = dfs(expr)
        except _rewriting_exception:
            return None
        return new_expr
