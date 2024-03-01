from typing import Optional, Dict, List, Tuple

import sympy as sp

from ...utils import convex_hull_poly, findroot_resultant, generate_expr

def _hull_space(
        nvars: int,
        degree: int,
        convex_hull: Dict[Tuple[int, ...], bool],
        monomial: Tuple[int, ...]
    ) -> Optional[sp.Matrix]:
    """
    For example, s(ab(a-b)2(a+b-3c)2) does not have a^6,
    so in the positive semidefinite representation, the entry (a^3,a^3) of M is zero.
    This requires that Me_i = 0 where i is the index of a^3.
    """
    if convex_hull is None:
        return None

    half_degree = (degree - sum(monomial)) // 2
    dict_monoms = generate_expr(nvars, half_degree)[0]

    def onehot(i: int) -> List[int]:
        v = [0] * len(dict_monoms)
        v[i] = 1
        return v

    space = []
    for key, value in convex_hull.items():
        if value:
            continue

        # value == False: not in convex hull
        rest_monom = tuple(key[i] - monomial[i] for i in range(nvars))
        if any(r % 2 for r in rest_monom):
            continue
        rest_monom = tuple(r // 2 for r in rest_monom)

        space.append(onehot(dict_monoms[rest_monom]))

    return sp.Matrix(space).T


class RootSubspace():
    def __init__(self, poly: sp.Expr) -> None:
        self.poly = poly
        self._degree = poly.total_degree()
        self._nvars = len(poly.gens)
        self._is_cyc = True

        self.convex_hull = convex_hull_poly(poly)[0] if self._nvars == 3 else None
        self.roots = findroot_resultant(poly) if self._nvars == 3 else []
        self.roots = [r for r in self.roots if not r.is_corner]

    def nullspace(self, monomial, real: bool = False) -> sp.Matrix:
        """
        Compute the nullspace of the polynomial.

        Parameters
        ----------
        monomial : Tuple[int, ...]
            The monomial to compute the nullspace for.

        real : bool
            Whether prove the inequality on R^n or R+^n. If R+^n, it ignores nonpositive roots.
        """
        half_degree = (self._degree - sum(monomial))
        if half_degree % 2 != 0:
            raise ValueError(f"Degree of the polynomial ({self._degree}) minus the degree of the monomial {monomial} must be even.")

        funcs = [
            self._nullspace_hull,
            lambda x: self._nullspace_roots(x, real = real)
        ]
        nullspaces = []
        for func in funcs:
            n_ = func(monomial)
            if isinstance(n_, list) and len(n_) and isinstance(n_[0], sp.MatrixBase):
                nullspaces.extend(n_)
            elif isinstance(n_, sp.MatrixBase) and n_.shape[0] * n_.shape[1] > 0:
                nullspaces.append(n_)

        nullspaces = list(filter(lambda x: x.shape[0] * x.shape[1] > 0, nullspaces))
        return sp.Matrix.hstack(*nullspaces)

    def _nullspace_hull(self, monomial):
        return _hull_space(self._nvars, self._degree, self.convex_hull, monomial)


    def _nullspace_roots(self, monomial, real: bool = False):
        d = (self._degree - sum(monomial)) // 2
        nullspaces = []
        for root in self.roots:
            if not real and any(_ < 0 for _ in root.root):
                continue
            nullspaces.append(root.span(d))
        return nullspaces

    def __str__(self) -> str:
        return f"RootSubspace(poly={self.poly})"

    def __repr__(self) -> str:
        return f"<RootSubspace nvars={self._nvars} degree={self._degree} roots={self.roots}>"