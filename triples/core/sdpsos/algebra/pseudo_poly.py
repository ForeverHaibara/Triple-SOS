from typing import Any

from sympy import Poly, Expr
from sympy.polys.constructor import construct_domain

from .state_algebra import StateAlgebra

class PseudoPoly:
    """
    Polynomials on general algebra. It provides some interfaces similar to :class:`sympy.Poly`.
    """
    @classmethod
    def new(cls, rep, *gens):
        """Construct :class:`Poly` instance from raw representation. """
        obj = object.__new__(cls)
        obj.rep = rep
        obj.gens = gens
        return obj

    def per(f, rep, gens=None, remove=None):
        if gens is None:
            gens = f.gens
        if remove is not None:
            gens = gens[:remove] + gens[remove + 1:]
            if not gens:
                return f.rep.dom.to_sympy(rep)
        return f.__class__.new(rep, *gens)

    @property
    def algebra(self):
        return self.rep.algebra
    @property
    def domain(self):
        return self.get_domain()
    def get_domain(f):
        return f.rep.dom
    def set_domain(f, dom):
        rep = f.rep.convert(dom)
        return f.new(rep, *f.gens)
    @property
    def is_zero(f):
        return f.rep.is_zero
    def __bool__(f):
        return not f.is_zero

    @property
    def is_homogeneous(f):
        return f.rep.is_homogeneous
    @property
    def is_linear(f):
        return f.rep.is_linear
    @property
    def is_quadratic(f):
        return f.rep.is_quadratic


    def terms(f, order=None):
        return [(m, f.rep.dom.to_sympy(c)) for m, c in f.rep.terms(order=order)]
    def coeffs(f, order=None):
        return [f.rep.dom.to_sympy(c) for m, c in f.rep.coeffs(order=order)]
    def monoms(f, order=None):
        return f.rep.monoms(order=order)
    def as_expr(f):
        return f.algebra.as_expr(f)
    def primitive(f):
        cont, result = f.rep.primitive()
        return f.rep.dom.to_sympy(cont), f.per(result)
    @property
    def free_symbols(self):
        symbols = set(self.gens)
        # gens = self.gens
        # for i in range(len(gens)):
        #     for monom in self.monoms():
        #         if monom[i]:
        #             symbols |= gens[i].free_symbols
        #             break
        return symbols | self.free_symbols_in_domain
    @property
    def free_symbols_in_domain(self):
        domain, symbols = self.rep.dom, set()
        if domain.is_Composite:
            for gen in domain.symbols:
                symbols |= gen.free_symbols
        elif domain.is_EX or domain.is_EXRAW:
            for coeff in self.coeffs():
                symbols |= coeff.free_symbols
        return symbols

    def total_degree(f):
        return f.rep.total_degree()

    @classmethod
    def from_dict(cls, rep, *gens, **kwargs):
        domain = kwargs.get('domain', None)
        algebra = kwargs.pop('algebra', None)
        if domain is None:
            domain, rep = construct_domain(rep, **kwargs)
        else:
            for monom, coeff in rep.items():
                rep[monom] = domain.convert(coeff)

        return cls.new(PseudoSMP(algebra, domain, rep), *gens, algebra=algebra)


class PseudoSMP(dict):
    algebra: StateAlgebra
    dom: Any
    def __init__(self, algebra, domain, init=None):
        super().__init__(init)
        self.algebra = algebra
        self.dom = domain

    @classmethod
    def from_dict(cls, rep, level, domain, algebra=None):
        return cls(algebra, domain, {monom: domain.convert(coeff) for monom, coeff in rep.items()})

    def per(f, rep):
        return f.__class__(f.algebra, f.dom, rep)

    def terms(self, order=None):
        return list(self.items())
    def coeffs(self, order=None):
        return list(self.values())
    def monoms(self, order=None):
        return list(self.keys())
    def total_degree(f):
        d = f.algebra.total_degree
        return max((d(monom) for monom in f.keys()), default=0)

    def primitive(f):
        def _quo_ground(f, c, K):
            if not f: return f
            if K.is_Field:
                return [ K.quo(cf, c) for cf in f ]
            else:
                return [ cf // c for cf in f ]
        def _content(f, K):
            if not f:
                return K.zero
            cont = K.zero
            if K.is_QQ:
                for c in f:
                    cont = K.gcd(cont, c)
            else:
                for c in f:
                    cont = K.gcd(cont, c)
                    if K.is_one(cont):
                        break
            return cont
        c = _content(list(f.values()), f.dom)
        rep = _quo_ground(list(f.values()), c, f.dom)
        return c, f.per({k: v for k, v in zip(f.keys(), rep) if v})


    @property
    def is_zero(f):
        return not f

    @property
    def is_homogeneous(f):
        if not f: return True
        d = f.algebra.total_degree
        d0 = d(next(iter(f.keys())))
        return all(d(monom) == d0 for monom in f.keys())
    @property
    def is_linear(f):
        d = f.algebra.total_degree
        return all(d(monom) <= 1 for monom in f.keys())
    @property
    def is_quadratic(f):
        d = f.algebra.total_degree
        return all(d(monom) <= 2 for monom in f.keys())

    def convert(self, domain):
        return PseudoSMP(self.algebra, domain, {monom: domain.convert(coeff) for monom, coeff in self.items()})

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