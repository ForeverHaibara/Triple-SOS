import sympy as sp
from mpmath import pslq

from ...utils import (
    RootsInfo, Root, RootRational, RootTangent,
    rationalize,
    verify_is_symmetric, CyclicSum, CyclicProduct
)


a, b, c = sp.symbols('a b c')

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


def root_tangents(
        rootsinfo: RootsInfo,
    ):
    """
    Generate tangents for linear programming.
    """
    is_centered = rootsinfo.is_centered
    is_sym = verify_is_symmetric(rootsinfo.poly)
    strict_roots = rootsinfo.strict_roots
    normal_roots = rootsinfo.normal_roots

    tangents = []
    for roots in [strict_roots, normal_roots]:
        tangents += _tangents_helper.root_tangents(
            roots,
            is_sym = is_sym,
            is_centered = is_centered
        )
        if len(tangents) > 0:
            break

    tangents = [_.normalize() for _ in tangents]
    return tangents


class _tangents_helper():
    """Helper class for generating tangents."""
    @classmethod
    def root_tangents(cls, roots, is_sym = False, is_centered = True):
        if is_sym:
            helper = _tangents_helper_symmetric
        else:
            helper = _tangents_helper_cyclic

        tangents = []
        for root in roots:
            if root.is_centered:
                continue
            tangents += helper.root_tangents(root, is_centered = is_centered)

        tangents = [_.together().as_coeff_Mul()[1] for _ in tangents]
        tangents = [RootTangent(_) for _ in tangents]
        return tangents


class _tangents_helper_cyclic():
    """Helper class for generating tangents for cyclic polynomials."""
    @classmethod
    def root_tangents(cls, root, is_centered = True):
        tangents = []
        if root.is_border:
            return cls._tangents_border(root, is_centered = is_centered)
        else:
            tangents += cls._tangents_quadratic(root, is_centered = is_centered)
            tangents += cls._tangents_cubic(root, is_centered = is_centered)
        return tangents

    @classmethod
    def _tangents_quadratic(cls, root, is_centered = True):
        u0, v0 = root.uv()
        u, v = rl(u0), rl(v0)
        if not (near2(u, u0) and near2(v, v0)):
            u, v = rl(u0,1), rl(v0,1)
        tangents = [
            a**2 - b**2 + u*(a*b - a*c) + v*(b*c - a*b),
            # a**2 + (-u - v)*a*b + (-u + 2*v)*a*c + b**2 + (2*u - v)*b*c - 2*c**2
        ]
        if not is_centered:
            tangents.append(
                (u + v - 1)*a**2 + (-v**2 + v - 1)*a*b + (-u**2 + u - 1)*a*c + (u*v - 1)*b*c
            )
        return tangents

    @classmethod
    def _tangents_cubic(cls, root, is_centered = True):
        u0, v0 = root.uv()
        u, v = rl(u0), rl(v0)
        if not (near2(u, u0) and near2(v, v0)):
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

        knife = _inner_product(
            [1 - u*v, -u**3 + u**2*v - u*v - u, u**3 - u**2 + u*v + u + v**2 - v, u**2 + v, -u**2*v + u*v - v**2 - 1],
            [a**2*c, a*b**2, a*b*c, b**3, b**2*c]
        )

        trapezoid = a**3 + a**2*b*(u - v - 1) - a**2*c*u + a*b**2*(v - 1) + a*c**2*v + b**2*c*(1 - u) + b*c**2*(u - v + 1) - c**3

        tangents = [inverse_quad, umbrella, scythe, knife, trapezoid]

        if not is_centered:
            tangents += [
                CyclicSum((u*v-1)*a**2*b - ((u-v)*(u*v+u+v-2)+v**3+1)/3*a*b*c),
                CyclicSum((u*v-1)*a**2*c - ((v-u)*(u*v+u+v-2)+u**3+1)/3*a*b*c),
            ]

        return tangents

    @classmethod
    def _tangents_border(cls, root, is_centered = True):
        x = _standard_border_root(root)
        if (not x.is_real) or x == 0:
            return []
        if x == 1:
            return [a + b - c]
        if isinstance(x, sp.Rational):
            # return outer product of [x,1,0] and [1,0,x]
            return [x*a - x**2*b - c]

        return []


class _tangents_helper_symmetric(_tangents_helper):
    """Helper class for generating tangents for symmetric polynomials."""
    @classmethod
    def root_tangents(cls, root, is_centered = True):
        tangents = []
        if root.is_border:
            return cls._tangents_border(root, is_centered = is_centered)
        elif root.is_symmetric:
            return cls._tangents_symmetric(root, is_centered = is_centered)
        return tangents

    @classmethod
    def _tangents_border(cls, root, is_centered = True):
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
        if near(v, v0):
            tangents = [a**2 + b**2 + c**2 - v*a*c - v*b*c + (2*v - 3)*a*b]
        return tangents

    @classmethod
    def _tangents_symmetric(cls, root, is_centered = True):
        x = _standard_symmetric_root(root)
        if not x.is_real:
            return []
        if x == 0:
            # this should not happen, because it is handled in _tangents_border
            return [a + b - c]

        tangents = []
        if isinstance(x, sp.Rational):
            tangents += [
                a + b - (x + 1)*c,
                x*a*b + x*a*c - (x + 1)*b*c,
                a**2 + a*b*(-2*x - 2) + a*c*(x + 1) + b**2 + b*c*(x + 1) - 2*c**2
            ]
        return tangents