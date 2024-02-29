import sympy as sp

from ...utils import convex_hull_poly, findroot_resultant

class RootSubspace():
    def __init__(self, poly: sp.Expr) -> None:
        self.poly = poly
        self._degree = poly.degree()
        self._nvars = 3
        self._is_cyc = True

        self.convex_hull = convex_hull_poly(poly)[0]
        self.roots = findroot_resultant(poly)
        self.roots = [r for r in self.roots if not r.is_corner]

    def nullspace(self, monomial):
        """
        Compute the nullspace of the polynomial.
        """
        funcs = [
            self._nullspace_hull,
            self._nullspace_roots
        ]
        nullspaces = []
        for func in funcs:
            n_ = func(monomial)
            if isinstance(n_, list):
                nullspaces.extend(n_)
            elif isinstance(n_, sp.Matrix):
                nullspaces.append(n_)
        return sp.Matrix.hstack(*nullspaces)


    def _nullspace_hull(self, monomial):
        return []

    def _nullspace_roots(self, monomial):
        d = (self._degree - sum(monomial)) // 2
        nullspaces = []
        for root in self.roots:
            nullspaces.append(root.span(d))
        return nullspaces
