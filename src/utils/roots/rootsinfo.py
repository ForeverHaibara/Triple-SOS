from typing import Optional, Callable, List, Union

import sympy as sp

from .roots import Root
from .tangents import root_tangents, RootTangent

class RootsInfo():
    """
    A class to record the rootsinfo of a polynomial.
    """
    PRINT_PRECISION = 8
    PRINT_MAXLEN = 20
    def __init__(self,
            poly: sp.Poly = None,
            roots: List[Root] = [],
            strict_roots: List[Root] = [],
            tangents: List[RootTangent] = [],
            reg: float = 1e-7,
            approximate_roots: bool = True,
            with_tangents: Union[bool, Callable] = False,
        ):
        """
        Parameters
        ----------
        poly: sympy.Poly
            The polynomial.
        roots: List[Root]
            The roots of the polynomial. Local minima are also included.
        strict_roots: List[Root]
            The strict roots of the polynomial. Local minima are not included.
        tangents: List[RootTangent]
            The tangents of the polynomial.
        reg: float
            The tolerance for strict roots.
        approximate_roots: bool
            If True, approximate the roots if detected.
        with_tangents: bool | Callable
            If True, generate tangents for each root and store them in self.tangents.
        """
        self.poly = poly

        roots = [Root(r) if not isinstance(r, Root) else r for r in roots]
        strict_roots = [Root(r) if not isinstance(r, Root) else r for r in strict_roots]

        if approximate_roots:
            roots = [r.approximate() if hasattr(r, 'approximate') else r for r in roots]
            strict_roots = [r.approximate() if hasattr(r, 'approximate') else r for r in strict_roots]

        self.roots = roots
        self.strict_roots = strict_roots
        self.tangents = tangents
        self.reg = reg
        self.is_centered_ = None
        self.tangents = self.generate_tangents(with_tangents)
        self.filter_tangents()

    @property
    def normal_roots(self):
        """
        Return local minima but not strict roots.
        """
        return list(set(self.roots) ^ set(self.strict_roots))

    def __str__(self):
        return 'RootsInfo:[\n  Tolerance = %s\n  Strict Roots = %s\n  Normal Roots = %s\n]'%(
                self.reg, self.strict_roots, self.normal_roots)

    def __repr__(self):
        return self.__str__()

    @property
    def gui_description(self):
        s = 'Local Minima Approx:'
        if len(self.roots) == 0:
            return s

        def formatter(root):
            if hasattr(root, 'n'):
                return root.n(self.PRINT_PRECISION)
            elif hasattr(root, '__round__'):
                return round(root, self.PRINT_PRECISION)
            return root

        for root in self.roots:
            a, b, c = root
            for i in range(3):
                if c == 0:
                    a, b, c = b, c, a
            a, b = a/c, b/c
            value = float(root.eval(self.poly, rational = True))
            if abs(value) < sp.S(10)**(-15):
                value = sp.S(0)
            s += f'\n({formatter(a)},{formatter(b)},1) = {formatter(value)}'
        return s

    @property
    def nvars(self) -> int:
        return len(self.poly.gens)

    @property
    def is_centered(self) -> bool:
        if self.is_centered_ is None:
            self.is_centered_ = self.poly(*([1] * self.nvars)) == 0
        return self.is_centered_

    def nonborder_roots(self) -> List[Root]:
        """
        Return roots that a, b, c are nonzero.
        """
        return [r for r in self.strict_roots if not r.is_border]

    def nontrivial_roots(self, tolerance: float = 1e-5) -> List[Root]:
        """
        Return roots that a, b, c are distinct and also nonzero.
        """
        return [r for r in self.strict_roots if r.is_nontrivial]

    def has_nontrivial_roots(self) -> bool:
        return len(self.nontrivial_roots()) > 0

    def filter_tangents(self, tangents: Optional[List[RootTangent]] = None, tolerance: float = 1e-6) -> List[RootTangent]:
        """
        Remove tangents that are not zero at nontrivial roots.

        Parameters
        ----------
        tangents: list of RootTangent
            If None, use self.tangents.
        tolerance: float
            If abs(tangent(root)) < tolerance, then the tangent is removed.

        Returns
        ----------
        filtered_tangents: list of RootTangent
            The filtered tangents. When tangents = None, the function uses
            self.tangents and also modifies self.tangents inplace.
        """
        if tangents is None:
            tangents = self.tangents
        if len(tangents) == 0:
            return tangents

        nontrivial_roots = self.nontrivial_roots()
        if len(nontrivial_roots) == 0:
            return tangents

        filtered_tangents = []
        a, b, c = self.poly.gens
        for t in tangents:
            for r in nontrivial_roots:
                s = sum(r)/3
                u, v, w = r[0]/s, r[1]/s, r[2]/s
            if all(abs(t(u,v,w)) < tolerance for r in nontrivial_roots):
                filtered_tangents.append(t)

        if tangents is self.tangents: # pointer
            self.tangents = filtered_tangents

        return filtered_tangents

    def generate_tangents(self, with_tangents: Union[bool, Callable] = True) -> List[RootTangent]:
        """
        Generate tangents for each root.

        Parameters
        ----------
        with_tangents: bool | Callable
            The function to generate tangents. The function should take a
            RootsInfo object as input and return a list of RootTangent objects.
        """
        if with_tangents is None or with_tangents is False:
            return []
        if with_tangents is True:
            with_tangents = root_tangents
        
        return with_tangents(self)

    def sort_tangents(self) -> List[RootTangent]:
        """
        Sort the tangents of self by length of strings.
        """
        tangents = sorted(self.tangents, key = lambda t: len(t))
        self.tangents = tangents
        return tangents