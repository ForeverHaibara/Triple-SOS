import sympy as sp

from ...utils.expression import poly_get_factor_form
from ...utils.polytools import deg


class RootTangent():
    __slots__ = ('expr', 'poly', 'degree', '_length')

    def __init__(self, expr):
        self.expr = expr
        self.poly = expr.doit().as_poly(*sp.symbols('a b c'))
        self.degree = deg(self.poly)
        self._length = len(self.poly.coeffs())

    def __str__(self):
        return str(self.expr)

    def __repr__(self):
        return f'RootTangent({self.__str__()})'

    def __eq__(self, other):
        return self.poly == other.poly

    def __hash__(self):
        return hash(self.poly)

    def __len__(self):
        return self._length

    def __call__(self, *args, **kwargs):
        return self.poly(*args, **kwargs)

    def as_expr(self):
        return self.expr.as_expr()

    def as_poly(self):
        return self.poly

    def subs(self, *args, **kwargs):
        return self.expr.doit().subs(*args, **kwargs)

    def as_factor_form(self, remove_minus_sign = False):
        s = poly_get_factor_form(self.poly)
        if remove_minus_sign and s.startswith('-'):
            s = s[1:]
        return s

def root_tangents(rootsinfo):
    """
    Deprecated function for finding tangents.
    Please use the new function `src.core.linsos.root_tangents` instead.
    """
    import warnings
    warnings.warn(
        'This function is deprecated. Please use the new function `src.core.linsos.root_tangents` instead.',
        DeprecationWarning
    )
    return []
