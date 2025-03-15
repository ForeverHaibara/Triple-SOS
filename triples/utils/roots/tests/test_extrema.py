import sympy as sp
from sympy.abc import a,b,c,d,x,y,z,w

from ..extrema import optimize_poly
from ....utils import CyclicSum

class ExtremaProblems():
    def collect(self):
        return {k: getattr(self, k) for k in dir(self) if k.startswith('problem')}

    def problem_pqr(self):
        return (-x, [a,b,c], [a+b+c-5,a*b+b*c+c*a-3,a*b*c-x]),\
            [(13/3,1/3,1/3,13/27), (1/3,13/3,1/3,13/27), (1/3,1/3,13/3,13/27)]
    def problem1003(self):
        poly = sp.S(4)/3-y+x
        return (poly, [], [x**2*(1-x)-(1-y)*y**2-(1-x)*(1-y)]), [(-5/9, 7/9)]
    def problem1007(self):
        poly = sp.S(4)/9 - CyclicSum(x*y*(1-x),(x,y,z))
        return (poly, [x,y,z,16-9*x,16-9*y,16-9*z], []), [(0, 1/2, 16/9), (1/2, 16/9, 0), (2/3, 2/3, 2/3), (16/9, 0, 1/2)]
    def problem_2003(self):
        # https://artofproblemsolving.com/community/c6h3378194
        poly = 7*(a+b)*(b+c)*(c+d)*(d+a)-240*a*b*c*d
        return (poly, [a,b,c,d,a-b-7*c-d], [d-1,poly]), [(1,0,0,1), (5,1,3/7,1)]
    def problem_beale(self):
        # https://www.sfu.ca/~ssurjano/beale.html
        poly = (sp.S(3)/2 - x + x*y)**2 + (sp.S(225)/100 - x + x*y**2)**2 + (sp.S(2625)/1000 - x + x*y**3)**2
        return (poly, [], []), [(3, 0.5)]
    def problem_camel6(self):
        # https://www.sfu.ca/~ssurjano/camel6.html
        poly = 4*x**2 - sp.S(21)/10*x**4 + x**6/3 + x*y - 4*y**2 + 4*y**4
        return (poly, [], []), [(-0.0898420131, 0.712656403), (0.0898420131, -0.712656403)]

def test_extrema():
    problems = ExtremaProblems().collect()
    for func in problems.values():
        (poly, ineqs, eqs), solutions = func()
        result = optimize_poly(poly, ineqs, eqs)
        assert len(result) == len(solutions)

        result_dict = set(tuple(_.n(4) for _ in v) for v in result)
        solution_dict = set(tuple(sp.Float(_).n(4) for _ in v) for v in solutions)
        assert result_dict == solution_dict

def test_extrema_max_different():
    assert optimize_poly((a-2*b+2)**4+(b-c-1)**2+(c+2*a-3*d-1)**2+(d-a+3)**4,max_different=4)\
        == [(16,9,8,13)]