import numpy as np
import sympy as sp
from sympy.abc import a,b,c,d,x,y,z,w

from ..num_extrema import NumerFunc

def _verify(expr, wrapped, free_symbols, point, tol=1e-10, op='__add__'):
    """Verify whether the computation of function values and gradients at a point
    by a NumerFunc object is correct."""
    pointd = dict(zip(free_symbols, point))
    f_expected = float(expr.xreplace(pointd))
    g_expected = np.array([float(expr.diff(_).xreplace(pointd)) for _ in free_symbols])
    f_computed = wrapped(point)
    g_computed = wrapped.g(point)
    assert abs(f_expected - f_computed) < tol, f'{op} failed: {f_expected} != {f_computed}'
    assert np.abs(g_expected - g_computed).max() < tol, f'{op} failed: {g_expected}!= {g_computed}'
    return True

def test_numer_func_operations():
    embedding = {a: a, b: a*c/(a+5*c+2), c: c}
    p1 = (a**2+b**2+c**2)**2 - 3*(a**3*b+b**3*c+c**3*a)
    p2 = p1.xreplace(embedding)
    free_symbols = (c, a)

    point = np.array([3.1415926, 4.869869869])

    wrapped_embedding = NumerFunc.wrap([a,embedding[b],c], free_symbols)
    wrapped = NumerFunc.wrap(p1, (a,b,c)).compose(wrapped_embedding)
    _verify(p2, wrapped, free_symbols, point, op='wrap')

    # test operations
    p3 = 10/(a**3*(a-b)*(a-c) + b**3*(b-c)*(b-a) + c**3*(c-a)*(c-b))
    p4_ = p3.xreplace(embedding)
    wrapped2_ = NumerFunc.wrap(p3, (a,b,c)).compose(wrapped_embedding)

    for wrapped2, p4 in [(1.414, 1.414), (wrapped2_, p4_)]:
        for op in ['add', 'sub', 'mul', 'truediv', 'pow',
                    'radd', 'rsub', 'rmul', 'rtruediv']:
            op = f'__{op}__'
            wrapped_op = getattr(wrapped, op)(wrapped2)
            expr_op = getattr(p2, op)(p4)
            _verify(expr_op, wrapped_op, free_symbols, point, op=op)

def test_numer_func_permutation():
    # test permutation
    embedding = {
        x: a**3*b**2*c + b**3*c**2*a + c**2*a**2*b,
        y: a**4*b**3*c + b**4*c**3*a + c**3*a**3*b,
        z: a**5*b**4*c + b**5*c**4*a + c**4*a**4*b,
        w: a**6*b**5*c + b**6*c**5*a + c**5*a**5*b,
    }
    p1 = (x*y*z + y*z*w + z*w*x)/(x*w + y*z + 2)
    wrapped_embedding = NumerFunc.wrap([embedding[_] for _ in [x,y,z,w]], [a,b,c])
    wrapped = NumerFunc.wrap(p1, (x,y,z,w)).compose(wrapped_embedding)
    point = np.array([-3**.25, 0.984807753012, 1.23456789])
    _verify(p1.xreplace(embedding), wrapped, (a,b,c), point, op='wrap')

    # p2(a_, b_, c_) = p1(b_, c_, a_)
    p2 = p1.xreplace(embedding).xreplace({a:b, b:c, c:a})
    perm = [1,2,0]
    assert abs(p2.subs(dict(zip((a,b,c), point))) - \
            p1.xreplace(embedding).subs(dict(zip((a,b,c), point[perm])))) < 1e-10
    _verify(p2, wrapped.permute(perm), (a,b,c), point, op='permute')

    p2 = p1.xreplace(embedding).xreplace({a:a, b:c, c:b})
    perm = [0,2,1]
    assert abs(p2.subs(dict(zip((a,b,c), point))) - \
            p1.xreplace(embedding).subs(dict(zip((a,b,c), point[perm])))) < 1e-10
    _verify(p2, wrapped.permute(perm), (a,b,c), point, op='permute')

def test_numer_func_composition():
    p1 = (a**2+b**2+c**2)**2 - 3*(a**3*b+b**3*c+c**3*a)
    mat = [[2**.5,3**.5],[-7**(1/3), 9**(1/3)],[-1.234,-2.345]], [.7,-.6,-.5]
    f = NumerFunc.wrap(p1, (a,b,c)).compose_affine(*mat)
    point = np.array([3.1415926535, -2.71828])
    expr = p1.xreplace(dict(zip((a,b,c), sp.Matrix(mat[0])*sp.Matrix([x,y]) + sp.Matrix(mat[1]))))
    _verify(expr, f, (x,y), point, op='compose_affine')