from sympy.abc import a, b, c, x, y
from sympy import sqrt

from ..linsos import LinearSOS
from ....testing.doctest_parser import solution_checker

def test_linsos_tangents():
    p1 = (a**2+b**2+c**2-a*b-b*c-c*a) - 2*(a+b+c) + 4 + 2*a*b*c
    sol = LinearSOS(p1, [a,b,c], [], roots=[], augment_tangents=False,
            tangents=[(a+b-2)**2,((a+b+c-3)**2)], basis_limit=2000)
    assert sol is not None
    solution_checker(sol, p1, [a,b,c], [])
    assert sol.solution.doit().has((a+b-2)**2) or\
           sol.solution.doit().has((a+b+c-3)**2)

    p2 = (a**2+b**2+c**2)**2 - 3*(a**3*b+b**3*c+c**3*a)
    sol = LinearSOS(p2, [], [], roots=[], augment_tangents=False,
            tangents=[(a**2 - 3*a*b + b**2 + 3*a*c - 2*c**2)**2], basis_limit=2000)
    assert sol is not None
    solution_checker(sol, p2, [], [])

    p3 = sqrt(41) - 5*x - 4*y
    sol = LinearSOS(p3, [], [x**2 + y**2 - 1], roots=[], augment_tangents=False,
            tangents=[(sqrt(41)*x - 5)**2, (sqrt(41)*y - 4)**2], basis_limit=2000)
    assert sol is not None
    solution_checker(sol, p3, [a,b,c], [])
