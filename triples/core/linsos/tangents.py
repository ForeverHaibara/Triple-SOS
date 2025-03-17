from typing import List, Tuple

from mpmath import pslq as mp_pslq
import sympy as sp
from sympy import Expr
from sympy.simplify.sqrtdenest import _sqrt_match

from ...utils import (
    Root, RootRational, find_nearest_root,
    CyclicSum, CyclicProduct, Coeff
)
from ...utils import rationalize as _utils_rationalize

a, b, c = sp.symbols('a b c')

def pslq(coeffs: List[float], *args, **kwargs) -> List[sp.Rational]:
    """Wrapper of mpmath.pslq."""
    coeffs = [_.n(20) if isinstance(_, Expr) else _ for _ in coeffs]
    return mp_pslq(coeffs, *args, **kwargs)

def rationalize(v: float, **kwargs) -> sp.Rational:
    """Wrapper of _utils_rationalize."""
    return _utils_rationalize(v.n(20), **kwargs) if not isinstance(v, sp.Rational) else v

class _option():
    """
    Helper class for storing options for generating tangents.
    """
    _key_words_default = {
        'cyc': True,
        'sym': False,
        'centered': True,
        'strict': True
    }
    __slots__ = _key_words_default.keys()
    def __init__(self, *args, **kwargs):
        for key, value in self._key_words_default.items():
            setattr(self, key, kwargs.get(key, value))


def near(a, b, tol = 1e-4):
    """Check whether a and b are close enough."""
    return abs(a - b) < tol

def near2(a, b, N = 1000):
    """
    Check whether a is a nice rational number close to b.
    Return True if |a - b| < 1/q^3 and q < N.

    Recall Dirichlet's approximation theorem:
    For any irrational number x, there exist infinitely many rational numbers p/q such that
    |x - p/q| < 1/q^2. For rational numbers, there are only finitely many such p/q.
    """
    f = lambda x: 1 / x**3 if x > 21 else 1e-3
    if isinstance(a, sp.Rational):
        if a.q > N:
            return False
        tol = f(a.q)
    elif isinstance(b, sp.Rational):
        if b.q > N:
            return False
        tol = f(b.q)
    else:
        raise ValueError(f"Both {a} and {b} are not rational.")
    return near(a, b, tol)

def rl(v: float, type: int = 0):
    """
    Wrapper of rationalize(v, **kwargs).

    Parameters
    ----------
    v : float
        The float to be rationalized.
    type : int
        When type == 0, use {reliable=True}
        When type == 1, use {rounding=1e-2, reliable=False}
    """
    if type == 0:
        return rationalize(v, reliable=True)
    elif type == 1:
        return rationalize(v, rounding=1e-2, reliable=False)
    raise ValueError(f"Invalid type {type}.")

def _inner_product(x, y):
    """Inner product of two vectors."""
    return sum(x[i] * y[i] for i in range(len(x)))

def _standard_border_root(root):
    """If root.is_border, return x such that it is equivalent to (x,1,0)."""
    for i in range(3):
        if root[i] == 0:
            return root[(i+1)%3] / root[(i+2)%3]
    raise ValueError(f"Root {root} is not on the border.")

def _standard_symmetric_root(root):
    """If root.is_symmetric, return x such that it is equivalent to (x,1,1)."""
    for i in range(3):
        if root[i] == root[(i+1)%3]:
            return root[(i+2)%3] / root[(i+1)%3]
    raise ValueError(f"Root {root} is not on the symmetric axis.")


def root_tangents(poly: sp.Poly, roots: List[Root]) -> List[Expr]:
    """
    Generate tangents for linear programming.
    """
    nvars = len(poly.gens)
    if nvars != 3:
        return []

    is_hom = poly.is_homogeneous
    is_cyc = Coeff(poly).is_cyclic()
    if not is_hom:
        return []
    is_sym = is_cyc and Coeff(poly).is_symmetric()
    is_centered = any(_.is_centered for _ in roots)
    strict_roots = roots
    normal_roots = []

    tangents = []
    for is_strict, roots in [(True, strict_roots), (False, normal_roots)]:
        tangents += _tangents_helper_ternary.root_tangents(
            roots,
            poly.gens,
            _option(
                cyc = is_cyc,
                sym = is_sym,
                centered = is_centered,
                strict = is_strict
            )
        )
        if len(tangents) > 0:
            break

    return tangents


class _tangents_helper_ternary():
    """Helper class for generating tangents."""
    @classmethod
    def root_tangents(cls, roots: List[Root], gens: Tuple[sp.Symbol], option: _option) -> List[Expr]:
        if not option.cyc:
            helper = _tangents_helper_ternary_acyclic
        elif option.sym:
            helper = _tangents_helper_ternary_symmetric
        else:
            helper = _tangents_helper_ternary_cyclic

        tangents = []

        if option.strict and len(roots) > 1: # and any(_.is_nontrivial for _ in roots):
            # We should handle multiple roots at the same time.
            if option.cyc:
                tangents = _tangents_helper_ternary_mixed.root_tangents(roots, option)
            else:
                tangents = _tangents_helper_ternary_acyclic_mixed.root_tangents(roots, option)

        if len(tangents) == 0:
            for root in roots:
                if root.is_centered:
                    continue
                tangents += helper.root_tangents(root, option)

        tangents = [_ for _ in tangents if _ is not None]
        tangents = [_.together().as_coeff_Mul()[1] for _ in tangents]
        tangents = [_.xreplace(dict(zip((a,b,c), gens))) for _ in tangents]
        # tangents = [Expr(_, gens) for _ in tangents]
        return tangents


class _tangents_helper_ternary_cyclic():
    """Helper class for generating tangents for cyclic polynomials."""
    @classmethod
    def root_tangents(cls, root: Root, option: _option) -> List[sp.Expr]:
        tangents = []
        if root.is_border:
            return cls._tangents_border(root, option)
        else:
            tangents += cls._tangents_quadratic(root, option)
            tangents += cls._tangents_cubic(root, option)
        return tangents

    @classmethod
    def _tangents_quadratic(cls, root: Root, option: _option) -> List[sp.Expr]:
        u0, v0 = root.uv()
        u, v = rl(u0), rl(v0)
        if not (near2(u, u0) and near2(v, v0)):
            if option.strict:
                return cls._tangents_irrational_uv(root, option)
            u, v = rl(u0,1), rl(v0,1)

        tangents = [
            a**2 - b**2 + u*(a*b - a*c) + v*(b*c - a*b),
            # a**2 + (-u - v)*a*b + (-u + 2*v)*a*c + b**2 + (2*u - v)*b*c - 2*c**2
        ]
        if (not option.centered) and option.strict:
            diamond = lambda a, b, c: (u + v - 1)*a**2 + (-v**2 + v - 1)*a*b + (-u**2 + u - 1)*a*c + (u*v - 1)*b*c
            tangents += [
                diamond(a, b, c),
                (v**2 - v + 1)*diamond(a,b,c) + (u*v - 1)*diamond(b,c,a),
                (u*v - 1)*diamond(a,b,c) + (u**2 - u + 1)*diamond(b,c,a),
            ]
        tangents = [_.expand() for _ in tangents]
        return tangents

    @classmethod
    def _tangents_cubic(cls, root: Root, option: _option) -> List[sp.Expr]:
        u0, v0 = root.uv()
        u, v = rl(u0), rl(v0)
        if not (near2(u, u0) and near2(v, v0)):
            if option.strict:
                return []
            p = (u0*u0 + v0) / (u0*v0 - 1)
            q = (v0*v0 + u0) / (u0*v0 - 1)
            p, q = rl(p, 1), rl(q, 1)
            u, v = rl(u0,1), rl(v0,1)
            
            inverse_quad = a**2*c - b**2*c - p*(a**2*b - a*b*c) + q*(a*b**2 - a*b*c)
            trapezoid = a**3 + a**2*b*(u - v - 1) - a**2*c*u + a*b**2*(v - 1) + a*c**2*v + b**2*c*(1 - u) + b*c**2*(u - v + 1) - c**3

            return [inverse_quad, trapezoid]

        if u*v == 1: # or u <= 0 or v <= 0:
            return []
        p = (u*u + v) / (u*v - 1)
        q = (v*v + u) / (u*v - 1)

        inverse_quad = a**2*c - b**2*c - p*(a**2*b - a*b*c) + q*(a*b**2 - a*b*c)

        umbrella = _inner_product(
            [1 - u*v, u**2*v - u, u**3 - u**2*v - u**2 + u*v + u - v, -u**3 - 1, u**2 + v],
            [a**2*c, a*b**2, a*b*c, b**2*c, b*c**2]
        )

        scythe = _inner_product(
            [1 - u*v, u**3 - u**2*v - u**2 + u*v**2 + u*v + u - 2*v, u*v - 1, -u**3 + u**2*v - u*v**2 - u + v - 1, u**2 - u*v + v + 1],
            [a**2*c, a*b*c, b**3, b**2*c, b*c**2]
        )

        tangents = [inverse_quad, umbrella, scythe]

        if option.centered:
            knife = _inner_product(
                [1 - u*v, -u**3 + u**2*v - u*v - u, u**3 - u**2 + u*v + u + v**2 - v, u**2 + v, -u**2*v + u*v - v**2 - 1],
                [a**2*c, a*b**2, a*b*c, b**3, b**2*c]
            )

            trapezoid = a**3 + a**2*b*(u - v - 1) - a**2*c*u + a*b**2*(v - 1) + a*c**2*v + b**2*c*(1 - u) + b*c**2*(u - v + 1) - c**3

            tangents += [knife, trapezoid]
        else:
            tangents += [
                CyclicSum((u*v-1)*a**2*b - ((u-v)*(u*v+u+v-2)+v**3+1)/3*a*b*c),
                CyclicSum((u*v-1)*a**2*c - ((v-u)*(u*v+u+v-2)+u**3+1)/3*a*b*c),
            ]

        return tangents

    @classmethod
    def _tangents_irrational_uv(cls, root: Root, option: _option) -> List[sp.Expr]:
        """
        Example: root of 24abcs(a)3+s(a2b)s(a)3-243abcs(a2b) near (1.48962108725722, 1.05209446747076, 0.458284445272021)
        has u = 3sqrt(3) - 4 and v = 2.
        """
        u, v = root.uv()

        res1 = pslq([u**2, u, 1])
        res2 = pslq([v**2, v, 1])
        x = sp.symbols('x')
        if res1 is None or res2 is None:
            return []
        res1, res2 = sp.Poly.from_list(res1, x), sp.Poly.from_list(res2, x)
        u_, v_ = find_nearest_root(res1, u), find_nearest_root(res2, v)
        m1, m2 = _sqrt_match(u_), _sqrt_match(v_)
        if m1[2] == 0 and m2[2] != 0:
            m1[2] = m2[2]
        elif m2[2] == 0 and m1[2] != 0:
            m2[2] = m1[2]
        elif (m1[2] == 0 and m2[2] == 0) or (m1[2] != m2[2]):
            return []

        u1, u2, v1, v2, k = m1[0], m1[1], m2[0], m2[1], m1[2]
        terms = [
            ((3, 0, 0), v2),
            ((2, 1, 0), -2*u1*u2 + u1*v2 + u2*v1 - 2*v1*v2),
            ((2, 0, 1), -u1*v2 + u2*v1 + v2),
            ((1, 2, 0),
            -k*u2**2*v2 + k*u2*v2**2 - k*v2**3 - u1**2*v2 + 2*u1*u2*v1 - u2*v1**2 + u2 + v1**2*v2 - v2),
            ((1, 1, 1),
            -k*u2**3 + 2*k*u2**2*v2 - 2*k*u2*v2**2 + k*v2**3 + u1**2*u2 - 2*u1*u2*v1 + 2*u1*v1*v2 + u1*v2 - u2*v1 - v1**2*v2),
            ((1, 0, 2), -u1*v2 + u2*v1 - u2),
            ((0, 3, 0), u1*v2 - u2*v1),
            ((0, 2, 1),
            k*u2**3 - k*u2**2*v2 + k*u2*v2**2 - u1**2*u2 + u1**2*v2 - 2*u1*v1*v2 + u2*v1**2 + u2 - v2),
            ((0, 1, 2), 2*u1*u2 - u1*v2 - u2*v1 + 2*v1*v2),
            ((0, 0, 3), -u2)
        ]

        part = lambda a, b, c: sum(v * a**i*b**j*c**k for (i, j, k), v in terms)
        tangent = (u1*u2*v2 - u2**2*v1 + v2**2)*part(a,b,c) + (-u1*v2**2 + u2**2 + u2*v1*v2)*part(b,c,a) + ((u1*v2-u2*v1)**2+u2*v2)*part(c,a,b)

        cyc_sum = part(a,b,c) + part(b,c,a) + part(c,a,b)
        cyc_diff = part(a,b,c) - part(b,c,a)

        cyc_sum2 = b * tangent + c * tangent.xreplace({a:b, b:c, c:a}) + a * tangent.xreplace({a:c, b:a, c:b})
        cyc_diff2 = b * tangent - c * tangent.xreplace({a:b, b:c, c:a})

        tangents = [tangent, cyc_sum, cyc_diff, cyc_sum2, cyc_diff2]
        tangents = [_.expand() for _ in tangents]
        return tangents


    @classmethod
    def _tangents_border(cls, root: Root, option: _option) -> List[sp.Expr]:
        x = _standard_border_root(root)
        if (not x.is_real) or x == 0:
            return []
        if x == 1:
            return [a + b - c]
        if isinstance(x, sp.Rational):
            # return outer product of [x,1,0] and [1,0,x]
            return [x*a - x**2*b - c, a**2 - b**2 + 1/x*(a*b - a*c) + x*(b*c - a*b)]

        if True:
            # test quadratic root
            res = pslq([x**2, x, 1])
            if res:
                u, v, w = res
                return [w*u*a**2 + w*v*a*b + w**2*b**2 + u**2*c**2 + u*v*a*c - (w*(u+v+w)+u*(u+v))*b*c]

        return []


class _tangents_helper_ternary_symmetric(_tangents_helper_ternary):
    """Helper class for generating tangents for symmetric polynomials."""
    @classmethod
    def root_tangents(cls, root: Root, option: _option) -> List[sp.Expr]:
        tangents = []
        if root.is_border:
            return cls._tangents_border(root, option)
        elif root.is_symmetric:
            return cls._tangents_symmetric(root, option)
        return tangents

    @classmethod
    def _tangents_border(cls, root: Root, option: _option) -> List[sp.Expr]:
        x = _standard_border_root(root)
        if (not x.is_real) or x == 0:
            return []
        if x == 1:
            return [a + b - c]
        if x > 1:
            # x > 1 is equivalent to x < 1, which should be already handled
            return []
        v0 = x + 1/x
        v = rl(v0)
        if near(v, v0) and option.strict:
            return [
                a**2 + b**2 + c**2 - v*a*c - v*b*c + (2*v - 3)*a*b,
                a**2 - v*a*b + b**2
            ]
        return []

    @classmethod
    def _tangents_symmetric(cls, root: Root, option: _option) -> List[sp.Expr]:
        x = _standard_symmetric_root(root)
        if not x.is_real:
            return []
        if x == 0:
            # this should not happen, because it is handled in _tangents_border
            return [a + b - c]

        tangents = []
        if isinstance(x, sp.Rational):
            tangents = [
                a + b - (x + 1)*c,
                x*a*b + x*a*c - (x + 1)*b*c,
                a**2 + a*b*(-2*x - 2) + a*c*(x + 1) + b**2 + b*c*(x + 1) - 2*c**2
            ]

        if len(tangents) == 0 and option.strict:
            res = pslq([x**2, x, 1])
            if res:
                u, v, w = res
                def _tangent_t(t):
                    return (-a**2 + a*b + a*c - b*c)*t + (-a**2 + b**2 + c**2)*u + b*c*v + a**2*w
                return [_tangent_t(_) for _ in [-v, u + w]]

        return tangents



class _tangents_helper_ternary_mixed(_tangents_helper_ternary):
    """Helper class for generating tangents over multiple roots."""
    @classmethod
    def root_tangents(cls, roots: List[Root], option: _option) -> List[sp.Expr]:
        roots = [root for root in roots if not (root.is_centered or root.is_corner)]
        nontrivial_roots = [root for root in roots if root.is_nontrivial]
        symmetric_roots = [root for root in roots if (root.is_symmetric and not root.is_border)]
        border_roots = [root for root in roots if root.is_border]

        tangents = []
        if nontrivial_roots and border_roots and (not symmetric_roots):
            tangents = cls._tangents_nontrivial_border(nontrivial_roots, border_roots, option)
        elif symmetric_roots and border_roots and (not nontrivial_roots):
            tangents = cls._tangents_symmetric_border(symmetric_roots, border_roots, option)

        return tangents

    @classmethod
    def _tangents_nontrivial_border(cls, nontrivial_roots: List[Root], border_roots: List[Root], option: _option) -> List[sp.Expr]:
        if len(nontrivial_roots) == 1:
            u0, v0 = nontrivial_roots[0].uv()
            u, v = rl(u0), rl(v0)
            if not (near2(u, u0) and near2(v, v0) and u*v != 1):
                return []

            methods = [
                None,
                cls._tangents_nontrivial_1_border_1,
                cls._tangents_nontrivial_1_border_2,
            ]
            for i in range(len(border_roots), 3):
                tangents = methods[i](u, v, border_roots)
                if tangents:
                    return tangents

        return []


    @classmethod
    def _tangents_nontrivial_1_border_1(cls, u: sp.Rational, v: sp.Rational, border_roots: List[Root]) -> List[sp.Expr]:
        x = _standard_border_root(border_roots[0])
        if x == 0 or not x.is_real:
            return []
        if not isinstance(x, sp.Rational):
            return []

        def quad(a, b, c):
            return a**2 - b**2 + u*(a*b - a*c) + v*(b*c - a*b)

        def _tangent_t(t):
            # (tb+zc)(a2-b2+u(ab-ac)+v(bc-ab))+(ra+tb+c)(b2-c2+u(bc-ab)+v(ca-bc))
            # {z: t*(u*x - 1)/(-v + x) + (u*x - v*x + x**2 - 1)/(-v*x + x**2), r: t*(-v + x)/(u*x - 1)}
            if t is sp.zoo: return None
            m, n = u*x - 1, x - v
            basis = (m + x*n) * quad(a,b,c) + (x*n) * quad(b,c,a)
            if t == 0:
                return basis
            if m == 0 or n == 0: return None
            diff = (m*n*b + m**2*c) * quad(a,b,c) + (n**2*a + m*n*b) * quad(b,c,a)
            if t is sp.oo:
                return diff
            return diff * t + basis * c

        t3 = (-u*x - v*x + x**2 + 1)/(x*(u*v - 1))
        t4 = -(u*x - 1)*(u*x - v*x + x**2 - 1)/(x*(u**2*x**2 - 2*u*x + v**3 - 2*v**2*x + v*x**2 + 1))
        tangents = [_tangent_t(_) for _ in (0, sp.oo, t3, t4)]

        if u != v:
            def _tangent_t2(t):
                # (tb+zc)(a2-b2+u(ab-ac)+v(bc-ab))+(ra+tb+c)(b2-c2+u(bc-ab)+v(ca-bc))
                # {z: (-u + x)/(v*x - 1)*t + (-u*x + v*x + x**2 - 1)/(v*x - 1), r: (-v + x)/(u*x - 1)*t}
                m, n = u*x - 1, v*x - 1
                if t is sp.zoo or t == 0: return None
                basis = (x*m + n) * quad(a,b,c) + (x*n) * quad(b,c,a)
                diff = (m*n*b + (x-u)*m*c) * quad(a,b,c) + ((x-v)*n*a + m*n*b) * quad(b,c,a)
                if t is sp.oo:
                    return diff
                return diff * t + basis * c

        t5 = (-u + x)/(u*v - 1)
        tangents += [_tangent_t2(_) for _ in (sp.oo, t5)]
        tangents = [_.expand() for _ in tangents if _ is not None]
        return tangents

    @classmethod
    def _tangents_nontrivial_1_border_2(cls, u: sp.Rational, v: sp.Rational, border_roots: List[Root]) -> List[sp.Expr]:
        """
        Examples
        ---------
        s((4a+b)(a-b)2(a-2b)2)-s(a(a2-bc)(4a2-27bc))

        s((b2-a2-9c2+4ab+4bc+16ca)(a2-b2-ab+2bc-ca)2)
        """
        res = None
        if len(border_roots) == 2:
            x, y = [_standard_border_root(border_roots[i]) for i in range(2)]
            if isinstance(x, sp.Rational) and isinstance(y, sp.Rational):
                res = [sp.S(1), -x-y, x*y]
            elif isinstance(x, sp.Rational) or isinstance(y, sp.Rational):
                return []
            if res is None:
                res = pslq([x**2, x, 1])
        if res is None:
            return []
        c2, c1, c0 = res

        # tangent must be a linear combination of vertex1 and vertex2
        # tangent = x1/vertex2[1] * vertex2 + x6/vertex1[-1] * vertex1
        vertex1 = [
            sp.S(0),
            -c0*u**2 - c1*u - c2,
            c0 + c1*v + c2*v**2,
            -c0*u - c1 - c2*v,
            -c0*u**2 + c0*u*v - c0 - c1*u - c2,
            c2*(u*v - 1)
        ]

        vertex2 = [
            c0*c2*(u*v - 1)*(c0*u + c1 + c2*v),
            c2**3 + (2*c0*u**2 - c0*u*v + c0 + 2*c1*u)*c2**2 + (c0**2*u**4 - c0**2*u**3*v + c0**2*u**2*v**2 + c0**2*u**2 - 2*c0**2*u*v + c0**2 + 2*c0*c1*u**3 - c0*c1*u**2*v + c0*c1*u*v**2 + c0*c1*u - c0*c1*v + c1**2*u**2)*c2 + c0**2*c1*u**2*v - c0**2*c1*u + c0*c1**2*u*v - c0*c1**2,
            -(c0*u + c1 + c2*v)*c2*(c0*u + c1 + c2*v),
            (c0*u**2 + c1*u + c2)*c2*(c0*u + c1 + c2*v),
            (-u*v + 1)*c0**3 + (c1*u**2*v - c1*u*v**2 - c1*u + c1*v + c2*u**4 - 2*c2*u**3*v + 2*c2*u**2*v**2 + 2*c2*u**2 - c2*u*v**3 - 4*c2*u*v + c2*v**2 + 2*c2)*c0**2 + (c1**2*u*v - c1**2 + 2*c1*c2*u**3 - 2*c1*c2*u**2*v + c1*c2*u*v**2 + 2*c1*c2*u - c1*c2*v + 2*c2**2*u**2 - 2*c2**2*u*v + 2*c2**2)*c0 + c1**2*c2*u**2 + 2*c1*c2**2*u + c2**3,
            sp.S(0)
        ]

        def quad(a, b, c):
            return a**2 - b**2 + u*(a*b - a*c) + v*(b*c - a*b)

        def _tangent_t(t):
            if t is sp.zoo: return None
            x1, x2, x3, x4, x5, x6 = [(1-t)*v1 + t*v2 for v1, v2 in zip(vertex1, vertex2)]
            return ((x1 * a + x2 * b + x3 * c) * quad(a,b,c) + (x4 * a + x5 * b + x6 * c) * quad(b,c,a)).expand()

        tangents = [quad(a, b, c), _tangent_t(0)]
        s1, r1 = sum(vertex1[:3]), sum(vertex1[3:])
        s2, r2 = sum(vertex2[:3]), sum(vertex2[3:])
        if s1*r2 == s2*r1 and s1 != s2:
            # In this case there exists a solution that x1+x2+x3 == x4+x5+x6 == 0,
            # this makes CyclicSum(a*tangent**2) at (1,1,1) has zero hessian.
            tangents.append(_tangent_t(s1 / (s1 - s2)))
        if s1 != s2 or r1 != r2:
            k1, k2 = vertex1[4] - vertex1[1], vertex2[4] - vertex2[1]
            tangents += [_tangent_t(_) for _ in set([sp.S(1), k1/(k1-k2)])]

        return tangents

    @classmethod
    def _tangents_symmetric_border(cls, symmetric_roots: List[Root], border_roots: List[Root], option: _option) -> List[sp.Expr]:
        if not option.sym:
            return []
        if len(border_roots) == 1:
            # For symmetric poly with only 1 root on the border, it must be (1,1,0)
            # and it is degenerated.
            return []
        if len(border_roots) > 3:
            # Too complicated to handle.
            return []
        def _get_border_root_r(root):
            x = _standard_border_root(root)
            if x == 0 or x == 1:
                return None
            r = x + 1/x
            r0 = rl(r)
            return r0 if near(r, r0) else None

        r = list(filter(None, map(_get_border_root_r, border_roots)))
        if len(r) == 0:
            return []
        r = r[0]

        if len(symmetric_roots) == 1:
            # s(bc(b-c)^2(b+c-4a)^2)+s(bc(a-b)(a-c)(a-3b)(a-3c))-10p(a-b)^2
            x = _standard_symmetric_root(symmetric_roots[0])
            if isinstance(x, sp.Rational):
                v = (r + x*(r - x) - 2)/x
                return [
                    a**2 + b**2 + c**2 - r*a*(b+c) + v*b*c, 
                    a + b - (x + 1)*c,
                    x*a*b + x*a*c - (x + 1)*b*c
                ]

        return []


class _tangents_helper_ternary_acyclic(_tangents_helper_ternary):
    """Helper class for generating tangents for acyclic polynomials."""
    @classmethod
    def root_tangents(cls, root: Root, option: _option) -> List[sp.Expr]:
        tangents = []
        if root.is_border:
            return cls._tangents_border(root, option)
        else:
            tangents += cls._tangents_nontrivial(root, option)
        return tangents

    @classmethod
    def _tangents_border(cls, root: Root, option: _option) -> List[sp.Expr]:
        if (not isinstance(root, RootRational)) or root.is_symmetric:
            return []
        x, y, z = root.root
        # line passing through (1,1,1) and its perpendicular line
        line = (y - z)*a + (z - x)*b + (x - y)*c
        perp_line = (x*y + x*z - y**2 - z**2)*a + (-x**2 + x*y + y*z - z**2)*b + (-x**2 + x*z - y**2 + y*z)*c

        return [line, perp_line]

    @classmethod
    def _tangents_nontrivial(cls, root: Root, option: _option) -> List[sp.Expr]:
        if not option.strict:
            return []
        xyz = [rl(_) for _ in root.root]
        if all(near2(xyz[i], root.root[i]) for i in range(len(root))):
            return cls._tangents_rational(*xyz, option)
        return []

    @classmethod
    def _tangents_rational(cls, x, y, z, option: _option) -> List[sp.Expr]:
        # line passing through (1,1,1) and its perpendicular line
        line = (y - z)*a + (z - x)*b + (x - y)*c
        perp_line = (x*y + x*z - y**2 - z**2)*a + (-x**2 + x*y + y*z - z**2)*b + (-x**2 + x*z - y**2 + y*z)*c

        # conic passing through vertices and (1,1,1)
        conic1 = x*(y - z)*b*c + y*(z - x)*c*a + z*(x - y)*a*b

        # conic passing through midpoints and (1,1,1)
        conic2 = (-x*y + x*z + y**2 - z**2)*a**2 + (x**2 - x*z - y**2 + y*z)*a*b + (-x**2 + x*y - y*z + z**2)*a*c + (-x**2 + x*y - y*z + z**2)*b**2 + (-x*y + x*z + y**2 - z**2)*b*c + (x**2 - x*z - y**2 + y*z)*c**2

        if x == y or y == z or z == x:
            return [perp_line]
        return [line, perp_line, conic1, conic2]


class _tangents_helper_ternary_acyclic_mixed(_tangents_helper_ternary):
    """Helper class for generating tangents over multiple roots."""
    @classmethod
    def root_tangents(cls, roots: List[Root], option: _option) -> List[sp.Expr]:
        roots = [root for root in roots if not (root.is_centered or root.is_corner)]
        nontrivial_roots = [root for root in roots if root.is_nontrivial]
        symmetric_roots = [root for root in roots if (root.is_symmetric and not root.is_border)]
        border_roots = [root for root in roots if (root.is_border and not root.is_symmetric)]

        tangents = []
        # if nontrivial_roots and border_roots and (not symmetric_roots):
        #     tangents = cls._tangents_nontrivial_border(nontrivial_roots, border_roots, option)
        # elif symmetric_roots and border_roots and (not nontrivial_roots):
        #     tangents = cls._tangents_symmetric_border(symmetric_roots, border_roots, option)
        if len(border_roots) > 1:
            return cls._tangents_multiple_border(border_roots, option)

        return tangents

    @classmethod
    def _tangents_nontrivial_border(nontrivial_roots, border_roots, option):
        return []

    @classmethod
    def _tangents_multiple_border(cls, border_roots: List[Root], option: _option) -> List[sp.Expr]:
        if len(border_roots) < 2 or len(border_roots) > 3:
            return []
        sides = [r.root.index(0) for r in border_roots]
        if len(sides) != len(set(sides)):
            return [] # not implemented

        tangents = []

        for i in range(len(border_roots)):
            x1, y1, z1 = border_roots[i].root
            line = (y1 - z1)*a + (z1 - x1)*b + (x1 - y1)*c
            tangents.append(line)
            for j in range(i+1, len(border_roots)):
                x2, y2, z2 = border_roots[j].root
                tangents.append((y1*z2 - y2*z1)*a + (-x1*z2 + x2*z1)*b + (x1*y2 - x2*y1)*c)

        return tangents