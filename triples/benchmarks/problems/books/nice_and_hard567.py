from ..problem_set import ProblemSet, mark
from sympy.abc import a,b,c,d,e,p,q,x,y,z
from sympy import symbols, Rational, Add, Mul, sqrt, cbrt, Abs

CyclicSum = lambda x, y: Add(*[x.xreplace(
    dict(zip(y,[y[(i+j)%len(y)] for j in range(len(y))]))) for i in range(len(y))])
CyclicProduct = lambda x, y: Mul(*[x.xreplace(
    dict(zip(y,[y[(i+j)%len(y)] for j in range(len(y))]))) for i in range(len(y))])
c3s = lambda x: CyclicSum(x, (a,b,c))
c3p = lambda x: CyclicProduct(x, (a,b,c))
c4s = lambda x: CyclicSum(x, (a,b,c,d))
c4p = lambda x: CyclicProduct(x, (a,b,c,d))
c5s = lambda x: CyclicSum(x, (a,b,c,d,e))
c5p = lambda x: CyclicProduct(x, (a,b,c,d,e))

class NiceAndHard567(ProblemSet):
    """
    "567 Nice And Hard Inequalities" by Nguyen Duy Tung
    """

    def problem_nah567_001_p1(self):
        return c3s(a/b) - c3s(sqrt((a**2 + 1)/(b**2 + 1))), [a,b,c], []

    def problem_nah567_001_p2(self):
        return c4s((a**2 - b*d)/(b + 2*c + d)), [a,b,c,d], []

    def problem_nah567_002(self):
        return Rational(3, 8) - c3s(a*b/(1 - c**2)), [a,b,c], [a + b + c - 1]

    def problem_nah567_003(self):
        return 1 + c3s(a*b**2)/c3s(a*b)/c3s(a) - 4*cbrt(c3p(a**2+a*b+b*c))/c3s(a)**2, [a,b,c], []

    @mark(mark.nvars)
    def problem_nah567_004(self):
        ...

    def problem_nah567_005(self):
        return c3s(cbrt((a**2 + b*c)/(b**2 + c**2))) - 9 * cbrt(a*b*c)/c3s(a), [a,b,c], []

    def problem_nah567_006(self):
        return c3s(1/(2*a**2 + b*c)) - 2/c3s(a*b), [a,b,c], []

    def problem_nah567_007(self):
        return c3s(1/(2*a**2 + b*c)) + 1/(a*b + b*c + c*a) - 12/c3s(a)**2, [a,b,c], []

    def problem_nah567_008(self):
        constraint = 16*(a + b + c) - (1/a + 1/b + 1/c)
        return Rational(8, 9) - c3s(1/((a + b + sqrt(2*(a + c)))**3)), [a,b,c, constraint], []

    def problem_nah567_009(self):
        expr = CyclicSum((x**3 + 1)/sqrt(x**4 + y + z), (x,y,z)) - 2*sqrt(x*y + y*z + z*x)
        return expr, [x,y,z], [x*y*z - 1]

    def problem_nah567_010(self):
        return sqrt(5) - c3p(a**2 - b**2), [a,b,c], [a + b + c - sqrt(5)]

    def problem_nah567_011(self):
        return Rational(3, 5) - c3s(1/(3 + a**2 + b**2)), [a,b,c], [c3s(a) - 3]

    def problem_nah567_012(self):
        return c3s(1/(4*a**2 - b*c + 1)) - 1, [a,b,c], [c3s(a*b) - 1]

    def problem_nah567_013(self):
        return c3s(1/(4*a**2 - b*c + 2)) - 1, [a,b,c], [c3s(a*b) - 1]

    def problem_nah567_014(self):
        return c3s(1/a)*c3s(1/(1 + a)) - 9/(1 + c3p(a)), [a,b,c], []

    def problem_nah567_015(self):
        return Rational(3, 2)*sqrt(c3p(a+1)) - c3s(a*(b+1)), [a,b,c], []

    def problem_nah567_016(self):
        return c3s(1/(a**2 + b**2)) - 10/c3s(a)**2, [a,b,c], []

    def problem_nah567_017(self):
        return c3s(a**2) - c3s(a**2*(b+c)**2/(a**2+3*b*c)), [a,b,c], []

    @mark(mark.nvars)
    def problem_nah567_018(self):
        ...

    def problem_nah567_019(self):
        return c3s((1 - a*b)/(7 - 3*a*c)) - Rational(1, 3), [a,b,c], [c3s(a**2) - 1]

    @mark(mark.noimpl)
    def problem_nah567_020(self):
        ...

    def problem_nah567_021(self):
        expr = 1/(a + x**2*(b*y + c*z)) + 1/(b + y**2*(c*z + a*x)) \
            + 1/(c + z**2*(a*y + b*x)) - 3/(a + b + c)
        return expr, [a,b,c,x,y,z,b+c-2*a], [x*y*z - 1]

    def problem_nah567_022(self):
        return CyclicSum(1/((1 + x**2)*(1 + x**7)), (x,y,z)) - Rational(3,4), [x,y,z], [x*y*z - 1]

    def problem_nah567_023(self):
        return 3/sqrt(2) - c3s(a/sqrt(a + b)), [a,b,c], [3*c3s(a**2) + c3s(a*b) - 12]

    def problem_nah567_024(self):
        return 8*c3s(a)**2/(3*c3p(a+b)**2) - c3s(1/((a**2 + b*c)*(b + c)**2)), [a,b,c], []

    def problem_nah567_025(self):
        return c3s(1/(Rational(8,5)*a**2 + b*c)) - Rational(9,4), [a,b,c], [c3s(a*b) - 1]

    def problem_nah567_026(self):
        return c3s(a/(b**2 + c**2)) - c3s(a)/c3s(a*b) - c3p(a)*c3s(a)/(c3s(a**3)*c3s(a*b)), [a,b,c], []

    def problem_nah567_027(self):
        return c3s(a**2*(b+c)/(b**2 + b*c + c**2)) - 2*c3s(a**2)/c3s(a), [a,b,c], []

    @mark(mark.noimpl)
    def problem_nah567_028(self):
        ...

    def problem_nah567_029(self):
        return c3s(a**2*(b + c)/(b**2 + b*c + c**2)) - 2*sqrt(c3s(a**3)/c3s(a)), [a,b,c], []

    def problem_nah567_030(self):
        return 1 - (2*sqrt(c3s(a**2*b)) + c3s(a*b)), [a,b,c], [c3s(a) - 1]

    def problem_nah567_031(self):
        return c3s(a**2*b**2/(c**3*(a**2 + b**2))) - sqrt(3)/2, [a,b,c], [c3s(a**2*b**2) - c3p(a**2)]

    def problem_nah567_032_p1(self):
        expr = x + y + z - x*y*z
        return expr - 1, [x,y,z], [x**2 + y**2 + z**2 - 1]

    def problem_nah567_032_p2(self):
        expr = x + y + z - x*y*z
        return 8*sqrt(3)/9 - expr, [x,y,z], [x**2 + y**2 + z**2 - 1]

    def problem_nah567_033_p1(self):
        return c3s(a*b)/2 - c3s(a**3*(b+c-a)/(a**2 + b*c)), [a,b,c], []

    def problem_nah567_033_p2(self):
        return 3*c3p(a)*c3s(a)/2/c3s(a*b) - c3s(a**3*(b+c-a)/(a**2 + b*c)), [a,b,c], []

    def problem_nah567_033_p3(self):
        return c3s((a**3 + b*c)/(a**2 + b*c)) - 2, [a,b,c], [c3s(a) - 1]

    def problem_nah567_034(self):
        return 6*c3s(a**2) - c3s((a**3+b*c)/(a**2+b*c)), [a,b,c], [c3s(a) - 1]

    def problem_nah567_035_p1(self):
        expr = 2/x**2 + 1/x + y*(y + 1/x + 2)
        return (1 + sqrt(2))/2 - expr, [], [x**2*y**2 + 2*y*x + 1]

    def problem_nah567_035_p2(self):
        expr = 2/x**2 + 1/x + y*(y + 1/x + 2)
        return expr - (1 - sqrt(2))/2, [], [x**2*y**2 + 2*y*x + 1]

    @mark(mark.skip)
    def problem_nah567_036(self):
        x1, x2, x3, x4, y1, y2, y3, y4 = symbols('x1 x2 x3 x4 y1 y2 y3 y4')
        expr = 2*(a/b + b/a + c/d + d/c) - ((a*y1 + b*y2 + c*y3 + d*y4)**2 + (a*x4 + b*x3 + c*x2 + d*x1)**2)
        return expr, [a,b,c,d], [a*b + c*d - 1, x1**2+y1**2-1, x2**2+y2**2-1, x3**2+y3**2-1, x4**2+y4**2-1]

    @mark(mark.geom)
    def problem_nah567_037(self):
        return 27*c**4 - c4p(-a + b + c + d), [a,b-a,c-b,d-c,a+b+c-d], []

    def problem_nah567_038(self):
        return c3s(1/(a+b-c)) - c3s(1/a), [a,b,c,b+c-a,c+a-b,a+b-c], []

    def problem_nah567_039(self):
        return c3s(a**3) + 3*a*b*c - (c3s(a**2*b)**2/c3s(a*b**2) + c3s(a*b**2)**2/c3s(a**2*b)), [a,b,c], []

    def problem_nah567_040(self):
        return CyclicSum((x + 1/y - 1)*(y + 1/z - 1), (x,y,z)) - 3, [x,y,z], []

    def problem_nah567_041(self):
        return c3s(a)**2/(2*c3s(a*b)) - c3s(a**2/(a**2 + b*c)), [a,b,c], []

    def problem_nah567_042(self):
        return a**3/(b**2 - b*c + c**2) + (b**3 + c**3)/a**2 - sqrt(2), [a,b,c], [a**2 + b**2 + c**2 - 1]

    def problem_nah567_043(self):
        S = sqrt(2*c3s(a**2*b**2) - c3s(a**4))/4
        lhs = c3s((a*b*(b**2+c**2-a**2)*(a**2+c**2-b**2))/(8*c**2*S*(a**2+b**2-c**2)))
        return lhs, [a**2+b**2-c**2, b**2+c**2-a**2, c**2+a**2-b**2, a,b,c]

    def problem_nah567_044(self):
        return c3s(a/(b + c)) + c3s(a**2*b)/c3s(a*b**2) - Rational(5,2), [a,b,c], []

    def problem_nah567_045(self):
        return c3s(1/(a + b)**2) - (3*sqrt(3*a*b*c*c3s(a))*c3s(a)**3)/(4*c3s(a*b)**3), [a,b,c], []

    def problem_nah567_046(self):
        return Rational(2,3)*c3s(1/(a**2 + b*c)) - 1/c3s(a*b) - 2/c3s(a**2), [a,b,c], []

    def problem_nah567_047(self):
        return c3s(1/(2*a**2 + b*c)) - 1/c3s(a*b) - 2/c3s(a**2), [a,b,c], []

    def problem_nah567_048(self):
        return 2*c3s(1/(a**2 + 8*b*c)) - 1/c3s(a*b) - 2/c3s(a**2), [a,b,c], []

    def problem_nah567_049(self):
        return Rational(5,3)*c3s(1/(4*a**2 + b*c)) - 2/c3s(a*b) - 1/c3s(a**2), [a,b,c], []

    def problem_nah567_050(self):
        return 30*c5s(a**4) - 7*c3s(a**2)**2, [a,b,c,d,e], [a + b + c + d + e]

    def problem_nah567_051(self):
        return sqrt(27 + c3s(a)*c3s(1/a))/2 - c3s(a*(b + c)/(a**2 + b*c)), [a,b,c], []

    def problem_nah567_052(self):
        return sqrt(c3s(sqrt(a))*c3s(1/sqrt(a))) - c3s(sqrt(a*(b + c)/(a**2 + b*c))), [a,b,c], []

    def problem_nah567_053(self):
        return c3s(a)**3/(3*a*b*c) + c3s(a*b**2)/c3s(a**3) - 10, [a,b,c], []

    def problem_nah567_054(self):
        expr = c3s(sqrt(b*c)*sqrt(2*b**2 + 2*c**2 - a**2)) - c3s(a*sqrt(2*b**2 + 2*c**2 - a**2))
        return expr, [a+b-c, b+c-a, c+a-b]

    def problem_nah567_055(self):
        expr = c3p(a**2 + 3) - Rational(64,27)*sqrt(a*b*c)*(a*b + b*c + c*a)
        return expr, [a,b,c], []

    def problem_nah567_056(self):
        return c3s((a + b)/(c*sqrt(a**2 + b**2))) - (3*sqrt(6))/sqrt(c3s(a**2)), [a,b,c], []

    def problem_nah567_057(self):
        return c3s(a**2) - c3s(a**2*b**2), [a,b,c], [c3s(a**2) + a*b*c - 4]

    def problem_nah567_058(self):
        return c3s(b**2/(a + b**2)) - Rational(3,4), [a,b,c], [a + b + c - 1]

    @mark(mark.noimpl)
    def problem_nah567_059(self):
        """Transcendental"""
        return c3s(a**(b + c)) - 1, [a,b,c], []

    def problem_nah567_060(self):
        expr = 3 - Abs(c3s(a**3/b) - c3s(a**3/c))
        return expr, [a+b-c, b+c-a, c+a-b], [a + b + c - 2]

    def problem_nah567_061(self):
        expr = CyclicSum( x*(y + z)**2/(2*x + y + z), (x,y,z)) - sqrt(3*x*y*z*(x + y + z))
        return expr, [x,y,z], []

    def problem_nah567_062(self):
        return 9 - (7*a + 5*b + 12*a*b), [6 - (9*a**2 + 8*a*b + 7*b**2)], []

    def problem_nah567_063(self):
        return x*y*z + y*z + z*x + x*y - 4, [x,y,z], [x + y + z - (1/x + 1/y + 1/z)]

    def problem_nah567_064(self):
        return Rational(1,32) - c3p(a**2 + b**2), [a,b,c], [c3s(a) - 1]

    def problem_nah567_065(self):
        return c3s(a/(2*a - b + c)) - Rational(3,2), [b+c-a, c+a-b, a+b-c], []

    def problem_nah567_066(self):
        return 4*(1 + c4s(1/a)) - (3 + 5/(1+a) + 7/(1+a+b) + 9/(1+a+b+c) + 36/(1+a+b+c+d)), [a,b,c,d], []

    def problem_nah567_067(self):
        return c3s(a) - 3, [a,b,c], [9 + 3*a*b*c - 4*c3s(a*b)]

    def problem_nah567_068(self):
        return c4s(a**2/(b**2+3)) - 1, [a,b,c,d], [c4s(a) - 4]

    def problem_nah567_069(self):
        return 8*c3p(a**3 + b) - 125*c3p(a + b), [a-2,b-2,c-2]

    def problem_nah567_070(self):
        return c3s(1/sqrt(2*a**2 + a*b + b*c)) - 9/(2*c3s(a)), [a,b,c], []

    def problem_nah567_071(self):
        return (3*c3s(a**2))/c3s(a)**2 - c3s(a/(a + 2*b)), [a,b,c], []

    def problem_nah567_072(self):
        return c4s((a**4 + b**4)/((a + b)*(a**2 + a*b + b**2))) - c4s(a**2)/c4s(a), [a,b,c,d], []

    def problem_nah567_073(self):
        return c3s(a*sqrt(a**2 + 4*b**2 + 4*c**2)) - c3s(a)**2, [a,b,c], []

    def problem_nah567_074(self):
        return c3s((b**2 + c**2)/(a**2 + b*c)) - 2*c3s(a/(b + c)), [a,b,c], []

    def problem_nah567_075(self):
        return c3s(sqrt(2*(b + c)/a)) - (27*c3p(a+b))/(4*c3s(a)*c3s(a*b)), [a,b,c], []

    def problem_nah567_076_p1(self):
        return c3s(1/(a*sqrt(2*(a**2 + b*c)))) - 9/(2*c3s(a*b)), [a,b,c], []

    @mark(mark.nvars)
    def problem_nah567_076_p2(self):
        ...

    def problem_nah567_077(self):
        return c3s((a**3 + 2*a*b*c)/(a**3 + (b + c)**3)) - 1, [a,b,c], []

    def problem_nah567_078(self):
        return sqrt(3*c3s(a/b)) + 2*sqrt(c3s(a*b)/c3s(a**2)) - 5, [a,b,c], []

    def problem_nah567_079(self):
        return c3s((b*c)/(1 + a**2)) - Rational(3,2), [b+c-a,c+a-b,a+b-c], [c3s(a**2) - 3]

    def problem_nah567_080(self):
        return 2 + 2/c4p(a) - c4s(1/a), [a,b,c,d], [c4s(a**2) - 4]

    def problem_nah567_081(self):
        return cbrt(3*c3s((2*a**2 + b*c))**2 ) - c3s(cbrt((a**2 + a*b + b**2)**2)), [a,b,c], []

    def problem_nah567_082(self):
        return c3s(1/(a**2 + b*c)) - (c3s(1/(a**2 + 2*b*c)) + c3s(a*b)/(2*c3s(a**2*b**2))), [a,b,c], []

    def problem_nah567_083(self):
        return c3s(a**2*(b+c)/(b**2+c**2)) - c3s(a), [a,b,c], []

    def problem_nah567_084(self):
        return Rational(3,4) - c3s(a/(3*a+b**2)), [a,b,c], [c3s(a)-3]

    def problem_nah567_085(self):
        return 1 - c4s(a*b*(b+c)), [a-b,b-c,c-d,d],[c4s(a)-2]

    def problem_nah567_086(self):
        return CyclicSum(2*(p+q)/((y+z)*(p*y+q*z)), (x,y,z)) - 9/(x*y+y*z+z*x), [x,y,z,p,q], []
