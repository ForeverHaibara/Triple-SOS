from sympy.abc import a, b, c, d, u, v, x, y, z
from sympy import Poly, Symbol, Rational, ZZ, QQ, ring, field, cbrt, sqrt, asin
from sympy.polys.rings import PolyElement
from sympy.polys.fields import FracElement

from ..dispatch import (
    _dtype_is_homogeneous, _dtype_homogenize
)

def test_dispatch_names():
    # import all singledispatch functions from ..dispatch
    import importlib
    dispatch = importlib.import_module('..dispatch', __package__)

    funcs = [getattr(dispatch, f) for f in dir(dispatch) if f.startswith('_dtype_')]
    funcs = [f for f in funcs if callable(f) and hasattr(f, 'registry')]

    for f in funcs:
        name = f.__name__[7:] # strip "_dtype_"
        for g in f.registry.values():
            assert name in g.__name__,\
                f"{g.__name__} does not match the name of {f.__name__} as a singledispatch registry"

    # open the dispatch.py to check no data type is registered twice
    # TODO: use a more robust way to check the registry?
    import os
    dispatch_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../dispatch.py')
    with open(dispatch_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    registry = set()
    for line in lines:
        if line.startswith('@_dtype_') and '.register(' in line:
            line = line.rstrip()
            if line in registry:
                assert False, f"{line} is registered twice in {dispatch_path}"
            registry.add(line)


def test_is_homogeneous():
    rng = ring((a,b,c), QQ)[0]
    fld = field((a,b,c), ZZ)[0]
    exprs = [
        sqrt(0),
        x,
        (a - b)**2,
        a*x**2 + b*x*y + c*y**2 + (u + 2*v)*(x + y)*(z - x),
        a/b/2 + (2*a + b)/(b - c) - 3,
        Poly(0, (x,)),
        Poly(a*x**2 + x*y + u**2*y**2 + (z - 3)*x*y, (x, y)),
        cbrt((a**3 + b**3)/3) - (a + b)/3,
        sqrt(x + y) - sqrt(x),
        (a + sqrt(a*b) + b)/3 - (a - b)/(2*asin((a-b)/(a+b))),
        rng(0),
        rng((a + 2*b)*(c**2 - 2*a*b + a**2) + b**3/2 - (b + c)*a**2*10/4),
        fld(1/a + 1/(a + 1) - (2*a + 1)/a/(a + 1)),
        fld(a/(b + 2*c)/4 - 5*(b + 2*a)*(c + 3*a)/(a**2 + b**2 - 4*b*c) + 3),
    ]
    for expr in exprs:
        assert _dtype_is_homogeneous(expr), f"{expr} is homogeneous, but asserted not"

    exprs = [
        a**2 - 2*a + 1,
        (a + 2)/(b + 1),
        Poly(a*x**2 + x*y + u**2*y**2 + (z - 3)*x*y, (a, x, y)),
        sqrt(a) - a,
        rng((a - 2)/3*(b + 2) + c**2 - 4*a*b/3),
        fld(a/(b + 1) - b/(a + 1)),
    ]
    for expr in exprs:
        assert not _dtype_is_homogeneous(expr), f"{expr} is not homogeneous, but asserted homogeneous"


def test_homogenize():
    s = Symbol("s", real=True, positive=True)
    rng = ring((a,b,c), QQ)[0]
    fld = field((a,b,c), ZZ)[0]
    rng2 = ring((a,b,c,s), QQ)[0]
    fld2 = field((a,b,c,s), ZZ)[0]
    exprs = [
        (2*a + b**2 - (c + 4)*x**2, 2*a*s**2 + b**2*s - (c + 4*s)*x**2),
        (a/b - (b**2 + 1)/(c + 1) - Rational(2,3), a*s/b - (b**2 + s**2)/(c + s) - Rational(2,3)*s),
        (Poly(a - b + c, (a, b, c)), Poly(a - b + c, (a, b, c, s))),
        (Poly(x**2 - b + a**3, (x, b)), Poly(x**2 - b*s + a**3*s**2, (x, b, s))),
        (rng(a**3 - 4*b + c*b), rng2(a**3 - 4*b*s**2 + c*b*s)),
        (fld((a - 1)/(b**2*a - 2*c*b + 3)), fld2((a - s)/(b**2*a - 2*c*b*s + 3*s**3))),
        ((a**2 + b**2 + 1)**2 - 3*(sqrt(a**3*b) + sqrt(b**3) + sqrt(a)),
            (a**2 + b**2 + s**2)**2 - 3*(sqrt(a)*sqrt(s**7) + sqrt(b**3)*sqrt(s**5) + sqrt(a**3*b)*s**2))
    ]
    for before, after in exprs:
        z = _dtype_homogenize(before, s)
        assert z == after, f"{before} homogenized to {z}, but expected {after}"
