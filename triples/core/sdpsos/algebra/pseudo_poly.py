from collections import defaultdict
from typing import List, Any

from sympy import Poly, Expr, Symbol, Add, Mul, Integer, sympify, EX
from sympy.polys.constructor import construct_domain
from sympy.matrices.expressions import MatPow

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
    def as_expr(f, **kwargs):
        return f.algebra.as_expr(f, **kwargs)
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

    def __add__(p1, p2):
        return p1.per(p1.rep + p2.rep)
    def __sub__(p1, p2):
        return p1.per(p1.rep - p2.rep)
    def __neg__(p1):
        return p1.per(-p1.rep)
    def __mul__(p1, p2):
        return p1.per(p1.rep * p2.rep)
    def __pow__(p1, n):
        return p1.per(p1.rep ** n)
    def state(p1):
        return p1.per(p1.rep.state())


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
    def zero(f):
        return f.per({})

    @property
    def one(f):
        return f.per({f.algebra.gen_monom(None): f.dom.one})

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
        # assume p1.ring == p2.ring and p1.domain == p2.domain
        zero = p1.dom.zero
        p = dict(p1.copy())
        for m, c in p2.items():
            if m in p:
                p[m] += c
            else:
                p[m] = c
        for m, c in list(p.items()):
            if c == zero:
                del p[m]
        return p1.per(p)

    def __sub__(p1, p2):
        # assume p1.ring == p2.ring and p1.domain == p2.domain
        zero = p1.dom.zero
        p = dict(p1.copy())
        for m, c in p2.items():
            if m in p:
                p[m] -= c
            else:
                p[m] = -c
        for m, c in list(p.items()):
            if c == zero:
                del p[m]
        return p1.per(p)

    def __neg__(p1):
        p = {m: -c for m, c in p1.items()}
        return p1.per(p)

    def __mul__(p1, p2):
        # assume p1.ring == p2.ring and p1.domain == p2.domain
        zero = p1.dom.zero
        alg = p1.algebra
        p = defaultdict(lambda: zero)
        for m1, c1 in p1.items():
            for m2, c2 in p2.items():
                m3, c3 = alg.mul((m1, c1), (m2, c2))
                p[m3] += c3
        for m, c in list(p.items()):
            if c == zero:
                del p[m]
        return p1.per(p)

    def __pow__(p1, n):
        if n < 0:
            raise ValueError("Pseudo-polynomials do not support negative exponents.")
        r = p1.one
        b = p1
        e = n
        while e > 0:
            if e % 2: r *= b
            b *= b
            e //= 2
        return r

    def state(p1):
        alg = p1.algebra
        zero = p1.dom.zero
        p = {}
        for m, c in p1.terms():
            m, c = alg.s((m, c))
            if m in p:
                p[m] += c
            else:
                p[m] = c
        for m, c in list(p.items()):
            if c == zero:
                del p[m]
        return p1.per(p) 


def convert_expr_to_pseudo_poly(algebra: StateAlgebra, expr: Expr, gens: List[Symbol],
        state_operator=None, **domain_kwargs) -> Poly:

    expr = sympify(expr)
    gens_dict = {v: i for i, v in enumerate(gens)}

    # domain = EX
    class _domain_cls:
        one = Integer(1)
        zero = Integer(0)
    domain = _domain_cls

    def _recur_build(x) -> PseudoSMP:
        if x.is_Integer or x.is_Rational or x.is_Float:
            return PseudoSMP(algebra, domain, {algebra.gen_monom(None): x})

        i = gens_dict.get(x, None)
        if i is not None:
            return PseudoSMP(algebra, domain, {algebra.gen_monom(i): domain.one})
        elif x.is_Add or x.is_Mul:
            args = [_recur_build(arg) for arg in x.args]
            arg0 = args[0]
            if x.is_Add:
                for i in range(1, len(args)):
                    arg0 = arg0 + args[i]
            elif x.is_Mul:
                for i in range(1, len(args)):
                    arg0 = arg0 * args[i]
            return arg0
        elif x.is_Pow or isinstance(x, MatPow):
            if x.exp.is_Integer and x.exp >= 0:
                return _recur_build(x.args[0]) ** x.args[1]
            raise ValueError
        elif state_operator is not None and isinstance(x, state_operator):
            return _recur_build(x.args[0]).state()

        if len(x.free_symbols) == 0:
            return PseudoSMP(algebra, domain, {algebra.gen_monom(None): x})

    smp = _recur_build(expr)
    monoms, coeffs = smp.monoms(), smp.coeffs()
    domain, coeffs = construct_domain(coeffs, **domain_kwargs)
    rep = PseudoSMP(algebra, domain, dict(zip(monoms, coeffs)))
    return PseudoPoly.new(rep, *gens)


