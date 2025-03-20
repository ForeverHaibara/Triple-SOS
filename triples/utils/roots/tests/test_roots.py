import sympy as sp
from sympy import sqrt, cbrt, sin, pi, I, Poly, Rational, CRootOf
from sympy.abc import a, b, c, x
from sympy.combinatorics import SymmetricGroup
from sympy.testing.pytest import raises

from ..roots import Root

def test_as_vec_and_span():
    # ZZ, QQ, AA, ZZ_I, QQ_I, RR, CC, EX, EXRAW
    assert sp.Eq(Root((1,1,1)).span(3, diff=(1,0,0)), sp.Matrix([3,2,2,1,1,1,0,0,0,0]))

    assert sp.Eq(Root((3,7,-11,13)).as_vec(2),
            sp.Matrix([9, 21, -33, 39, 49, -77, 91, 121, -143, 169]))
    assert Root((3,7,-11,13)).span(2) == Root((3,7,-11,13)).as_vec(2)

    assert sp.Eq(Root((Rational(1,5), Rational(3,7))).as_vec(3),
        sp.Matrix(list(map(Rational, ('1/125','3/175','9/245','27/343')))))
    assert Root((Rational(1,5), Rational(3,7))).span(3) == Root((Rational(1,5), Rational(3,7))).as_vec(3)

    poly = Poly([81, -81, 18, -1], x)
    r2, r1, r0 = CRootOf(poly, 2), CRootOf(poly, 1), CRootOf(poly, 0)
    root = Root((r2, r1, r0))
    span = root.span(2)
    assert span.shape == (6, 3) and \
        not any(span.T * sp.Matrix([1,-3,-2,-1,5,0])) and\
        not any(span.T * sp.Matrix([0,-2,5,1,-3,-1]))
    assert abs(root.as_vec(2, numer=True) - [0.5075, 0.1437, 0.06121, 0.04068, 0.01733, 0.007383]).max() < 1e-2
    assert abs(root.as_vec(2, numer=True) - list(root.as_vec(2, numer=False).n(6))).max() < 1e-4

    assert sp.Eq(Root((1 + sqrt(2), 1 - sqrt(2))).span(3),
            sp.Matrix([[7, 5], [-1, -1], [-1, 1], [7, -5]]))

    assert sp.Eq(Root((1,1+2*I,2)).span(2),
            sp.Matrix(2,6,[1,1,2,-3,2,4,0,2,0,4,4,0]).T)

    # assert sp.Eq(Root((Rational(-2,5), 2+sp.I, 3-7*sp.I)).span(2),

    assert sp.Eq(Root((1,1.5,2)).as_vec(2),
            sp.Matrix([1.0,1.5,2.0,2.25,3.0,4.0]))
    assert Root((1, 1.5, 2)).span(2) == Root((1, 1.5, 2)).as_vec(2)

    assert Root((-7, 1+1j, 2)).span(5, sym=True, hom=False) == Root((-7, 1+1j, 2)).as_vec(5, sym=True, hom=False)

    assert Root((a, a/b)).as_vec(2) == sp.Matrix([a**2, a**2/b, a**2/b**2])
    assert Root((a, a/b)).span(2) == Root((a, a/b)).as_vec(2)

    assert sp.Eq(Root((1,2,3,4)).span(3, diff=(0,0,0,1), normalize=True) * 4**2*3,
            Root((1,2,3,4)).span(3, diff=(0,0,0,1), normalize=False))

def test_cyclic_sum():
    assert Root((sqrt(2),)).cyclic_sum((5,)) == 4*sqrt(2)
    assert Root((0,) * 10).cyclic_sum((0,)*10) == 10
    assert Root((3, 4+I, 1)).cyclic_sum((1, 2, 3)) == 621 + 474*I
    assert Root((3, 4+I, 1)).cyclic_sum((1, 2, 3), standardize=True) == \
        Rational('213593907/75418890625') - 20176074*I/75418890625
    assert Root((Rational(1, 5), sqrt(2)+1, 10)).cyclic_sum((2, 2, 2)) == 36 + 24*sqrt(2)
    assert Root((a,b,c)).cyclic_sum((3,-2,5)) == a**3*c**5/b**2 + b**3*a**5/c**2 + c**3*b**5/a**2

    assert raises(ZeroDivisionError, lambda: Root((1,0,Rational(2,5))).cyclic_sum((-1,2,3)))

    assert Root((4, sqrt(2)-1, 7-3*sqrt(2), -1)).cyclic_sum((3,0,1,2), SymmetricGroup(4)) == -41142+28924*sqrt(2)

def test_uv():
    assert Root((1/sin(pi/9),1/sin(2*pi/9),-1/sin(4*pi/9))).uv() == (0, 1)
    assert Root((sin(4*pi/7)**2, sin(2*pi/7)**2, sin(pi/7)**2)).uv() == (1, 2) # Vasile inequality
    assert Root((a,a,1), domain=sp.EX).uv() == (1+1/a, 1+1/a)

    uv = Root((0.8180504-0.4869j,0.,3.14159+2.71828j)).uv()
    assert abs((uv[0]*uv[1] - 1).n(8)) < 1e-6

    assert Root.from_uv(3, 3) == Root((Rational(1,4), Rational(1,4), Rational(1,2)))
    assert Root.from_uv(-2, 7).uv() == (-2, 7)
    assert Root.from_uv(5, 3).uv() == (5, 3)
    assert Root.from_uv(5.4869, 4.321).domain.is_RR
    assert abs(Root.from_uv(5.4869, 4.321).uv()[0] - 5.4869) < 1e-8
    assert abs(Root.from_uv(Rational(-36,511) + 373*sqrt(2)/511, Rational(114,511) + 437*sqrt(2)/511)[0]\
               - (9 - 6*2**.5)) < 1e-8
    raises(ValueError, lambda: Root.from_uv(-1, (3 + sqrt(2))/(1 + sqrt(2)) - 2*sqrt(2)))

    # slow
    assert Root.from_uv((2*sqrt(7)+1)/3,(2+sqrt(7))/3).eval(
        Poly(a**3 + a**2*b - a*b**2 - 3*a*c**2 - b**3 + 2*b**2*c + b*c**2, a, b, c)) == 0