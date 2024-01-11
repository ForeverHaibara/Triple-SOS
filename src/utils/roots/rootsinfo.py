from typing import Callable, List, Union

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
            with_tangents: Union[bool, Callable] = False
        ):
        """
        Parameters
        ----------
        poly: sympy.Poly
            The polynomial.
        roots: list of Root
            The roots of the polynomial. Local minima are also included.
        strict_roots: list of Root
            The strict roots of the polynomial. Local minima are not included.
        tangents: list of RootTangent
            The tangents of the polynomial.
        reg: float
            The tolerance for strict roots.
        with_tangents: bool | Callable
            If True, generate tangents for each root and store them in self.tangents.
        """
        self.poly = poly
        self.roots = roots
        self.strict_roots = strict_roots
        self.tangents = tangents
        self.reg = reg
        self.is_centered_ = None
        if with_tangents is not False:
            self.generate_tangents(with_tangents if isinstance(with_tangents, Callable) else root_tangents)
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

        c = self.poly.gens[2]
        poly_reduced = self.poly.subs(c, 1)
        for root in self.roots:
            # NOTE: we use rational to evaluate for two reasons:
            # 1. to avoid numerical error
            # 2. rational is always faster than float by experiment
            a, b = root[0], root[1]
            value = poly_reduced(sp.Rational(a), sp.Rational(b))
            if abs(value) < sp.S(10)**(-15):
                value = sp.S(0)
            s += f'\n({formatter(a)},{formatter(b)},1) = {formatter(value)}'
        return s

    @property
    def is_centered(self):
        if self.is_centered_ is None:
            self.is_centered_ = self.poly(1,1,1) == 0
        return self.is_centered_

    @property
    def nonborder_roots(self):
        """
        Return roots that a, b, c are nonzero.
        """
        return [r for r in self.strict_roots if r[0] != 0 and r[1] != 0]

    @property
    def nontrivial_roots(self):
        """
        Return roots that a, b, c are distinct and also nonzero.
        """
        roots = self.nonborder_roots
        TOL = 1e-5
        def check(r):
            # r[2] == 1
            return abs(r[0] - r[1]) > TOL and abs(r[1] - 1) > TOL and abs(r[0] - 1) > TOL
        return [r for r in roots if check(r)]

    @property
    def has_nontrivial_roots(self):
        return len(self.nontrivial_roots) > 0

    def filter_tangents(self, tangents = None, tolerance = 1e-6):
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

        nontrivial_roots = self.nontrivial_roots
        if len(nontrivial_roots) == 0:
            return tangents

        filtered_tangents = []
        a, b, c = self.poly.gens
        for t in tangents:
            if all(abs(t.subs({a: r[0], b: r[1], c: 1})) < tolerance for r in nontrivial_roots):
                filtered_tangents.append(t)

        if tangents is self.tangents: # pointer
            self.tangents = filtered_tangents

        return filtered_tangents

    def generate_tangents(self, func: Callable = root_tangents):
        """
        Generate tangents for each root and store them in self.tangents.

        Parameters
        ----------
        func: Callable
            The function to generate tangents. The function should take a
            RootsInfo object as input and return a list of RootTangent objects.
        """
        self.tangents = func(self)
        return self.tangents

    def sort_tangents(self):
        """
        Sort the tangents by length of strings.
        """
        tangents = sorted(self.tangents, key = lambda t: len(t))
        self.tangents = tangents
        return tangents