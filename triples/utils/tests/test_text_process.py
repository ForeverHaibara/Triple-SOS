import sympy as sp
from sympy import Poly, QQ, ZZ, sympify, sqrt, cbrt, Rational, Float, Symbol
from sympy.abc import a, b, c, d, e, x, y, z
from sympy.combinatorics import CyclicGroup, SymmetricGroup, DihedralGroup, PermutationGroup, Permutation

from ..expressions import CyclicSum, CyclicProduct, SymmetricSum, SymmetricProduct
from ..text_process import pl, degree_of_expr


def test_preprocess_test():
    strings = [
        '(a2)2',
        'p(2+sqrt(2))  ',
        ' s(a2-ab)   s(a3-a2b-a b 2+abc) -s(a(a-b)2(a-c)2) ',
        's((a 2-b /2c  ) 2(x2a-bx+c*5+ab-2)2--x/13s(a+2b)2)',
        'p ((( a(2/3)+1.4s{[a-b]2*c2/5} )))',
        'p(a+b)s(1/(asqrt(2)+b**3^2s(a)))',
        '756s(a/s((a-b)2(9xa+b/ab-c/3a)^(-2/3))-b801c)',
        '(a/x-s(a*2b-b/2))3/(5-x)',
    ]
    targets = [
        a**4,
        (2 + sqrt(2))**3,
        CyclicSum(a**2-a*b, (a,b,c))*CyclicSum(a**3-a**2*b-a*b**2+a*b*c, (a,b,c)) - CyclicSum(a*(a-b)**2*(a-c)**2, (a,b,c)),
        CyclicSum(x/13*CyclicSum(a + 2*b, (a,b,c))**2 + (a**2 - b*c/2)**2*(a*x**2 - b*x + 5*c + a*b - 2)**2, (a,b,c)),
        CyclicProduct(a*2/3 + 1.4*CyclicSum((a - b)**2*c**2/5, (a,b,c)), (a,b,c)),
        CyclicProduct(a + b,(a,b,c)) * CyclicSum(1/(a*sqrt(2) + b**9*CyclicSum(a,(a,b,c))), (a,b,c)),
        756*CyclicSum(a/CyclicSum((a-b)**2*(9*x*a + b/a*b - c/3*a)**Rational(-2,3), (a,b,c)) - b**801*c, (a,b,c)),
        (a/x - CyclicSum(a*2*b - b/2, (a,b,c)))**3/(5 - x),
    ]
    for s, t in zip(strings, targets):
        p1 = pl(s, return_type='expr')
        assert p1 == t

        p1 = pl(s)
        if p1 is not None:
            p2 = Poly(t, a, b, c, extension=True)
            assert (p1.domain.is_Exact and p1.gens == p2.gens and (p1 - p2).is_zero)\
                or all(abs(_) < 1e-9 for _ in (p1 - p2).coeffs())

    strings_gens_perms = [
        ('( 1/5s((a2b)))', (a,c,b), PermutationGroup(Permutation([1,0,2]))),
        ('p(a)+s(a(a-b){a-c}(a-d))', (d,a,c,b), SymmetricGroup(4)),
        ('s(as[xa(a-b)/32-(a^2/5-d/3c+2/7)2])', (x,a,b,d), DihedralGroup(4)),
    ]
    targets = [
        CyclicSum(a**2*b/5, (a,c,b), PermutationGroup(Permutation([1,0,2]))),
        SymmetricProduct(a,(a,b,c,d)) + SymmetricSum(a*(a-b)*(a-c)*(a-d), (d,a,c,b)),
        CyclicSum(a*CyclicSum(x*a*(a-b)/32 - (a**2/5 - d/3*c + Rational(2,7))**2, (x,a,b,d), DihedralGroup(4)),
                  (x,a,b,d), DihedralGroup(4)),
    ]
    for (s, gens, perms), t in zip(strings_gens_perms, targets):
        p1 = pl(s, gens=gens, symmetry=perms, return_type='expr')
        assert p1 == t

        p1 = pl(s, gens=gens, symmetry=perms)
        p2 = Poly(t, *gens)
        assert p1.gens == p2.gens and (p1 - p2).is_zero

    strings = [
        '-(3a2-5b+c)/sqrt(5)',
        'sqrt(2)a/(a-b)/(a-c)+sqrt(2)b/((b-c)(b-a))+sqrt(2)c/((c-a)(c-b))',
        '(s((a+b-c)2(a-b)2)/2+s(ab(a-b)2))/5/s(3a)',
        's(a**2/(a-b)/-5/7*2)-4s(b/a)'
    ]
    targets = [
        (-(3*a**2-5*b+c)/sqrt(5), Rational(1,1)),
        (Rational(0,1), Rational(1,1)),
        (CyclicSum(a*(a-b)*(a-c),(a,b,c))/15, Rational(1,1)),
        (-2*(a**4*b**2*c + 70*a**4*b**2 - a**4*b*c**2 - 70*a**4*b*c + a**3*b**3*c - 70*a**3*b**3\
           - a**3*b**2*c**2 + a**3*b*c**3 + 140*a**3*b*c**2 - 70*a**3*c**3 - a**2*b**4*c\
           - a**2*b**3*c**2 + 140*a**2*b**3*c - a**2*b**2*c**3 - 210*a**2*b**2*c**2 + a**2*b*c**4\
           + 70*a**2*c**4 + a*b**4*c**2 - 70*a*b**4*c + a*b**3*c**3 - a*b**2*c**4 + 140*a*b**2*c**3\
           - 70*a*b*c**4 + 70*b**4*c**2 - 70*b**3*c**3), 35*a*b*c*(a - b)*(a - c)*(b - c))
    ]
    for s, t in zip(strings, targets):
        print(s, pl(s, return_type='text'))
        p1 = pl(s, return_type='frac')
        assert (p1[0] - Poly(t[0].doit(), (a,b,c), extension=True)).is_zero and\
                (p1[1] - Poly(t[1].doit(), (a,b,c), extension=True)).is_zero

    assert pl('2/5b31e-4a-1e-8-s(ex+.2e5)',(e,x,a), scientific_notation=True,return_type='expr')\
        == Rational(2,5)*b**Float('31e-4')*a - Float('1e-8') - CyclicSum(e*x + Float('.2e5'), (e,x,a))

    x1, x3, xx, xx2 = Symbol('x1'), Symbol('x3'), Symbol('xx'), Symbol('xx2')
    assert pl('y5x1^3/4s(x*xxxb-c)-2x3s(xa-b)/xx2', preserve_patterns=['xx','x','sqrt'],return_type='expr')\
        == y**5*x1**3/4*CyclicSum(x*xx*x*b-c,(a,b,c)) - 2*x3*CyclicSum(x*a-b,(a,b,c))/xx2

    ab, cbrty2 = Symbol('ab'), Symbol('cbrty2')
    assert pl('ab-acbcbrt(3)/ab*ba+3cbrty2/5',preserve_patterns=('ab','cbrt','cbrty'),return_type='expr')\
        == ab - a*c*b*cbrt(3)/ab*b*a + 3*cbrty2/5

    # test variables with extra assumptions
    a2, b2, c2 = [Symbol(_, real=True) for _ in 'abc']
    assert pl('s(a2)2-3s(a3b)', (a2,b2,c2), return_type='expr')\
        == CyclicSum(a2**2, (a2,b2,c2))**2 - 3*CyclicSum(a2**3*b2, (a2,b2,c2))


def test_degree_of_expr():
    strings_gens_perms_degrees = [
        ('s(a(a-b)(a-c))-s(a(a-b)(a-c))', (a,b,c), CyclicGroup(3), 3),
        ('4/7s(a2-ab)2-s(a2)+p(as(a)-b)-3/5s(a5)', (a,b,c), CyclicGroup(3), 6),
        ('p(a)(s(a2)2-s(a4+2a2b2))', (a,b,c), CyclicGroup(3), 7),
        ('p(s(a2c2-b2d2))2', (a,b,c,d), SymmetricGroup(4), 192),
        ('s(xy(a2+b))-s(xya2)-s(xyb)', (a,b,c), SymmetricGroup(3), 2),
        ('s(a2-ab)s(a(a-b)(a-c))-s(a(a-b)2(a-c)2)', (a,b,c), CyclicGroup(3), 5),
        ('s(a/(b+c))p(a+b)-s(a3+a2b+ab2+abc)', (a,b,c), CyclicGroup(3), 3),
        ('(r2+r+1)s((x-y)2(x+y-tz)2)-s(((y-z)(y+z-tx)-r(x-y)(x+y-tz))2)', (x,y,z), CyclicGroup(3), 4),
    ]
    for s, g, p, degree in strings_gens_perms_degrees:
        e = pl(s, g, p, return_type='expr',parse_expr_kwargs={'evaluate':False})
        assert degree_of_expr(e, g) == degree, \
            f"failed for {s}, expected {degree}, got {degree_of_expr(e, g)}"
