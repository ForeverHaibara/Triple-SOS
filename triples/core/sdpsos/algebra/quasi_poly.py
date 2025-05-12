from typing import Any

from sympy import Poly
from sympy.polys.constructor import construct_domain

from .state_algebra import StateAlgebra

class QuasiPoly:
    algebra: StateAlgebra

    @classmethod
    def new(cls, rep, *gens, algebra=None):
        """Construct :class:`Poly` instance from raw representation. """
        obj = object.__new__(cls)
        obj.rep = rep
        obj.gens = gens
        obj.algebra = algebra
        return obj

    @property
    def is_zero(f):
        return f.rep.is_zero
    def __bool__(f):
        return not f.is_zero

    @classmethod
    def from_dict(cls, rep, *gens, **kwargs):
        domain = kwargs.get('domain', None)
        algebra = kwargs.pop('algebra', None)
        if domain is None:
            domain, rep = construct_domain(rep, **kwargs)
        else:
            for monom, coeff in rep.items():
                rep[monom] = domain.convert(coeff)

        return cls.new(QuasiSMP(algebra, domain, rep), *gens, algebra=algebra)

class QuasiSMP(dict):
    algebra: StateAlgebra
    dom: Any
    def __init__(self, algebra, domain, init=None):
        super().__init__(init)
        self.algebra = algebra
        self.dom = domain

    @classmethod
    def from_dict(cls, rep, level, domain):
        return cls({monom: domain.convert(coeff) for monom, coeff in rep.items()})

    def terms(self):
        return list(self.items())

    @property
    def is_zero(f):
        return not f

    def __add__(p1, p2):
        if not p2:
            return p1.copy()
        ring = p1.ring
        if ring.is_element(p2):
            p = p1.copy()
            get = p.get
            zero = ring.domain.zero
            for k, v in p2.items():
                v = get(k, zero) + v
                if v:
                    p[k] = v
                else:
                    del p[k]
            return p
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__radd__(p1)
            else:
                return NotImplemented

        try:
            cp2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            p = p1.copy()
            if not cp2:
                return p
            zm = ring.zero_monom
            if zm not in p1.keys():
                p[zm] = cp2
            else:
                if p2 == -p[zm]:
                    del p[zm]
                else:
                    p[zm] += cp2
            return p