from numbers import Number

import sympy as sp

from .tangents import root_tangents

class RootsInfo():
    PRINT_PRECISION = 8
    PRINT_MAXLEN = 20
    def __init__(self, poly = None, roots = [], strict_roots = [], tangents = [], reg = 1e-7, with_tangents = False):
        self.poly = poly
        self.roots = roots
        self.strict_roots = strict_roots
        self.tangents = tangents
        self.reg = reg
        self.is_centered_ = None
        if with_tangents:
            self.generate_tangents()
        self.filter_tangents()

    @property
    def normal_roots(self):
        return list(set(self.roots) ^ set(self.strict_roots))

    def __str__(self):
        return 'RootsInfo:[\n  Tolerance = %s\n  Strict Roots = %s\n  Normal Roots = %s\n]'%(
                self.reg, self.strict_roots, self.normal_roots)

    def __repr__(self):
        return self.__str__()

    @property
    def gui_description(self):
        s = 'Local Minima Approx:'
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
        Return roots that a, b, c are nonzero
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

    def filter_tangents(self, tangents = None):
        if tangents is None:
            tangents = self.tangents
        if len(tangents) == 0:
            return tangents

        filtered_tangents = []
        TOL = 1e-6
        nontrivial_roots = self.nontrivial_roots
        a, b, c = self.poly.gens
        for t in tangents:
            if all(abs(t.subs({a: r[0], b: r[1], c: 1})) < TOL for r in nontrivial_roots):
                filtered_tangents.append(t)

        if tangents == self.tangents: # pointer
            self.tangents = filtered_tangents

        return filtered_tangents

    def generate_tangents(self):
        self.tangents = root_tangents(self)
        return self.tangents

    def sort_tangents(self):
        tangents = sorted(self.tangents, key = lambda t: len(t))
        self.tangents = tangents
        return tangents