import numpy as np
import sympy as sp
from sympy.abc import a,b,c,d,x,y,z,w

from ..num_extrema import NumerFuncWrapper


def test_numer_func_wrapper():
    embedding = {b: a*c/(a+5*c+2)}
    p1 = (a**2+b**2+c**2)**2 - 3*(a**3*b+b**3*c+c**3*a)
    p2 = p1.xreplace(embedding)
    free_symbols = (c, a)
    wrapper = NumerFuncWrapper((a,b,c),
        free_symbols=free_symbols, embedding=embedding)

    point = np.array([3.1415926, 4.869869869])
    pointd = dict(zip(free_symbols, point))
    f_expected = float(p2.xreplace(pointd))
    g_expected = np.array([float(p2.diff(_).xreplace(pointd)) for _ in free_symbols])
    f_computed = wrapper.wrap(p1)(point)
    g_computed = wrapper.wrap(p1, jac=True)(point)

    assert abs(f_expected - f_computed) < 1e-10
    assert np.abs(g_expected - g_computed).max() < 1e-10