from ..problem_set import ProblemSet, mark
from sympy.abc import a,b,c,d,e,p,q,r,s,u,v,x,y,z,w
from sympy import symbols, Rational, Add, Mul, sqrt, cbrt, sin, cos, pi, Abs, Min, Max

CyclicSum = lambda x, y: Add(*[x.xreplace(
    dict(zip(y,[y[(i+j)%len(y)] for j in range(len(y))]))) for i in range(len(y))])
CyclicProduct = lambda x, y: Mul(*[x.xreplace(
    dict(zip(y,[y[(i+j)%len(y)] for j in range(len(y))]))) for i in range(len(y))])
c3s = lambda x: CyclicSum(x, (a,b,c))
c3p = lambda x: CyclicProduct(x, (a,b,c))
c4s = lambda x: CyclicSum(x, (a,b,c,d))
c4p = lambda x: CyclicSum(x, (a,b,c,d))
c5s = lambda x: CyclicSum(x, (a,b,c,d,e))
c5p = lambda x: CyclicProduct(x, (a,b,c,d,e))

class MathematicalInequalities(ProblemSet):
    """
    "Mathematical Inequalities" by Vasile Cirtoaje
    """

class MathematicalInequalitiesVol1(MathematicalInequalities):
    """Symmetric Polynomial Inequalities"""
    def problem_vasile_p12001(self):
        return 27-a**3-b**3-c**3-d**3, [], [a**2+b**2+c**2+d**2-9]

    def problem_vasile_p12002(self):
        return -c3p(2*a**2+b*c), [], [a+b+c]

    def problem_vasile_p12003(self):
        return 9*c3p(a+b)-8*c3s(a)*c3s(a*b), [a+b,b+c,c+a], []

    def problem_vasile_p12004(self):
        return c3p(3*a**2+1)-64, [], [c3s(a*b)-3]

    def problem_vasile_p12005(self):
        fx = lambda x: 1 - x + x**2
        return 3*fx(a)*fx(b) - 2*fx(a*b), [], []

    def problem_vasile_p12006(self):
        return 3*c3p(1-a+a**2) - (1+c3p(a)+c3p(a)**2), [], []

    def problem_vasile_p12007(self):
        return c3s(a**2)**3-c3s(a)*c3s(a*b)*c3s(a**3), [], []

    def problem_vasile_p12008(self):
        return 2*c3p(a**2+b**2)-(c3s(a*b*(a+b))-2*c3p(a))**2, [], []

    def problem_vasile_p12009(self):
        return c3p(a**2+1)-2*c3s(a*b), [], []

    def problem_vasile_p12010(self):
        return c3p(a**2+1)-Rational(5,16)*(c3s(a)+1)**2, [], []

    def problem_vasile_p12011_q1(self):
        return c3s(a**6)-3*c3p(a**2)+2*c3p(a**2+b*c), [], []

    def problem_vasile_p12011_q2(self):
        return c3s(a**6)-3*c3p(a**2)-c3p(a**2-2*b*c), [], []

    def problem_vasile_p12012(self):
        return 2*c3s(a**6)/3+c3s(a**3*b**3)+c3p(a)*c3s(a**3), [], []

    def problem_vasile_p12013(self):
        return 4*c3p(a**2+a*b+b**2)-c3p((a-b)**2), [], []

    def problem_vasile_p12014(self):
        return c3p(a**2+a*b+b**2)-3*c3s(a**2*b)*c3s(a*b**2), [], []

    def problem_vasile_p12015(self):
        return 4*c3p(a+1/a)-9*c3s(a), [], [c3p(a)]

    def problem_vasile_p12016_q1(self):
        return c3s(a**2)*c3s(a*b)**2 - c3p(a**2+2*b*c), [], []

    def problem_vasile_p12016_q2(self):
        return c3s(a)**2*c3s(a**2*b**2) - c3p(2*a**2+b*c), [], []

    def problem_vasile_p12017(self):
        return c3s(a)**6-27*c3p(a**2+2*b*c), [], [c3s(a*b)]

    def problem_vasile_p12018(self):
        return c3p(a**2+2*b*c)+2, [], [c3s(a**2)-2]

    def problem_vasile_p12019(self):
        return 3*c3s(a**4)+c3s(a**2)+6-6*c3s(a**3), [], [c3s(a)-3]

    def problem_vasile_p12020(self):
        return 3*c3s(a**2)+2*c3s(a)-5*c3s(a*b), [], [c3p(a)-1]

    def problem_vasile_p12021(self):
        return c3s(a**2)+6-Rational(3,2)*c3s(a+1/a), [], [c3p(a)-1]

    def problem_vasile_p12022(self):
        return c3p(1+a**2)+8*c3p(a)-c3p(1+a)**2/4, [], []

    def problem_vasile_p12023(self):
        return c3s(a**12)-2049*c3p(a**4)/8, [], [c3s(a)]

    def problem_vasile_p12024(self):
        return c3s(a**2)+2*c3p(a)+4-2*c3s(a)-c3s(a*b), [c3p(a)], []

    def problem_vasile_p12025_q1(self):
        return c3s(1/a**2-1/a), [a+3,b+3,c+3], [a+b+c-3]

    def problem_vasile_p12025_q2(self):
        return c3s((1-a)/(1+a)**2), [a+7,b+7,c+7], [a+b+c-3]

    def problem_vasile_p12026(self):
        return c3s(a**6)-3*c3p(a**2)-c3p((a-b)**2)/2, [], []

    def problem_vasile_p12027(self):
        return (c3s(a**2)/3)**3 - c3p(a**2) - c3p((a-b)**2)/16, [], []

    def problem_vasile_p12028(self):
        return c3s(a**2)**3 - 108*c3p(a**2)/5 - 2*c3p((a-b)**2), [], []

    def problem_vasile_p12029(self):
        return 2*c3p(a**2+b**2) - c3p((a-b)**2), [], []

    def problem_vasile_p12030(self):
        return 32*c3p(a**2+b*c) + 9*c3p((a-b)**2), [], []

    def problem_vasile_p12031(self):
        return c3s(a**4*(b-c)**2) - c3p((a-b)**2)/2, [], []

    def problem_vasile_p12032(self):
        return c3s(a**2*(b-c)**4) - c3p((a-b)**2)/2, [], []

    def problem_vasile_p12033(self):
        return c3s(a**2*(b**2-c**2)**2) - c3p((a-b)**2)*3/8, [], []

    def problem_vasile_p12034_q1(self):
        return c3p(a**2+a*b+b**2)-3*c3s(a)**2, [], [c3s(a*b)-3]

    def problem_vasile_p12034_q2(self):
        return c3p(a**2+a*b+b**2)-3*c3s(a**2)/2, [], [c3s(a*b)-3]

    def problem_vasile_p12035(self):
        return c3p(a**2+a*b+b**2) - 3*c3s(a*b)*c3s(a**2*b**2), [], []

    def problem_vasile_p12036(self):
        return c3p(a**2+a*b+b**2) - 3*c3s(a*b)**3, [], [-c3p(a)]

    def problem_vasile_p12037(self):
        return c3p(a**2+a*b+b**2) - 3*c3p(a**2+b**2)/8, [], []

    def problem_vasile_p12038(self):
        return 2*c3p(a**2+b**2) - c3p(a**2-a*b+b**2), [], []

    def problem_vasile_p12039(self):
        return 9*c3p(1+a**4) - 8*(1+c3p(a)+c3p(a**2))**2, [], []

    def problem_vasile_p12040(self):
        return 2*c3p(1+a**2) - c3p(1+a)*(1+c3p(a)), [], []

    def problem_vasile_p12041(self):
        return 3*c3p(a**2-a*b+b**2) - c3s(a**3*b**3), [], []

    def problem_vasile_p12042(self):
        return c3s((b**2-b*c+c**2)/a**2 + 2*a**2/(b*c)) - c3s(a)*c3s(1/a), [], []

    def problem_vasile_p12043_q1(self):
        return 1 - c3p(a) + c3p(b+c-a), [a,b,c,1-a,1-b,1-c], []

    def problem_vasile_p12043_q2(self):
        return 4 - c3p(a) + c3p(b+c-a), [a+1,b+1,c+1, 1-a,1-b,1-c], []

    def problem_vasile_p12044_q1(self):
        return 1 - c3s(a**2*(a-b)*(a-c)), [a,b,c,1-a,1-b,1-c], []

    def problem_vasile_p12044_q2(self):
        return 4 - c3s(a**2*(a-b)*(a-c)), [a+1,b+1,c+1,1-a,1-b,1-c], []

    def problem_vasile_p12045(self):
        return c3s(a**2)-3-(2+sqrt(3))(c3s(a)-3), [], [c3s(a*b)-c3p(a)-2]

    def problem_vasile_p12046(self):
        return c3p(a**2+b**2) + 12*c3p(a**2) - 30, [], [c3p(a+b)-10]

    def problem_vasile_p12047(self):
        return c3p(a**2+a*b+b**2) + 12*c3p(a**2) - 15, [], [c3p(a+b)-5]

    def problem_vasile_p12048_q1(self):
        fx = lambda x: 1-x**2
        return Max(fx(a),fx(b),fx(c)), [], [c3s(a)-1, c3s(a**3)-25]

    def problem_vasile_p12048_q2(self):
        fx = lambda x: (2-x)*(x-1)
        return Max(fx(a),fx(b),fx(c)), [], [c3s(a)-1, c3s(a**3)+11]

    def problem_vasile_p12049(self):
        return (a - Rational(5,4))*(a-2), [], [c3s(a)-2, c3s(a**3)-2]

    def problem_vasile_p12050(self):
        return c3s((a-b)*(a-c)*(a-x*b)*(a-x*c)), [], []

    def problem_vasile_p12051(self):
        return c3p(b+c-a)**2 - c3p(b**2+c**2-a**2), [], []

    def problem_vasile_p12052(self):
        return c3s(a**2*(a-b)*(a-c)) - c3p(a-b)**2/c3s(a**2+b*c), [], []

    @mark(mark.noimpl, mark.quant)
    def problem_vasile_p12053(self):
        ...

    def problem_vasile_p12054(self):
        return (c3s(a*b)-3)**2-27*(c3p(a)-1), [], [c3s(a)-3]

    def problem_vasile_p12055(self):
        return c3s(a*b)**2+9-18*c3p(a), [], [c3s(a)-3]

    def problem_vasile_p12056(self):
        return c3p(a)+10-2*c3s(a), [], [c3s(a**2)-9]

    def problem_vasile_p12057(self):
        return c3s(a**2)+3-2*c3s(a*b), [], [c3s(a)+c3p(a)-4]

    def problem_vasile_p12058(self):
        return 4*c3s(a**2)+9-7*c3s(a*b), [], [c3s(a*b)-3*c3p(a)]

    def problem_vasile_p12059(self):
        return c3p(a**2+1)-c3p(a+1), [], [c3s(a)-3]

    @mark(mark.noimpl, mark.quant)
    def problem_vasile_p12060(self):
        ...

    def problem_vasile_p12061(self):
        return c3s(10*a**4+64*a**2*b**2-33*a*b*(a**2+b**2)), [], []

    def problem_vasile_p12062(self):
        return 3*c3s(a**4)+33-14*c3s(a**2), [], [c3s(a)-3]

    def problem_vasile_p12063(self):
        return 12 - c3s(a**4+3*a*b), [], [c3s(a**2)-3]

    def problem_vasile_p12064(self):
        A, C = x, y
        B = 2*C - 1 - A
        return c3s(a**4+A*a**2*b**2+B*a**2*b*c-C*a*b*(a**2+b**2)),\
            [1+A-C**2], []

    def problem_vasile_p12065(self):
        return 1 - c3s(a*b*(a**2-a*b+b**2-c**2)), [], [c3s(a**2)-2]

    def problem_vasile_p12066(self):
        return c3s((a+b)**4) - 4*c3s(a**4)/7, [], []

    def problem_vasile_p12067(self):
        p_ = c3s(a)
        q_ = c3s(a*b)
        r_ = c3p(a)
        return (3-p_)*r_ + (p_**2+q_**2-p_*q_)/3 - q_, [], []

    def problem_vasile_p12068(self):
        return Rational(3,4) - c3s(a*b*(a+b))/c3p(a**2+1), [], []

    def problem_vasile_p12069(self):
        return c3p(a+1/a-1) + 2 - c3s(a)*c3s(1/a)/3, [], [c3p(a)]

    def problem_vasile_p12070(self):
        return c3p(a**2 + Rational(1,2)) - c3p(a + b - Rational(1,2)), [], []

    def problem_vasile_p12071(self):
        return c3s(a*(a-1)/(8*a**2+9)), [], [c3s(a)-3]

    def problem_vasile_p12072(self):
        return c3s((a-11)*(a-1)/(2*a**2+1)), [], [c3s(a)-3]

    def problem_vasile_p12073(self):
        return c3p(a**2+2) - 9*c3s(a*b), [], []

    def problem_vasile_p12074(self):
        return 4*c3s(a**4)+11*c3p(a)*c3s(a)-45, [], [c3s(a*b)-3]

    @mark(mark.noimpl, mark.quant)
    def problem_vasile_p12075(self):
        ...

    def problem_vasile_p12076_q1(self):
        return 5*c3p(a**2+b**2)-8, [], [c3s(a*b)+1]

    def problem_vasile_p12076_q2(self):
        return c3p(a**2+a*b+b**2)-1, [], [c3s(a*b)+1]

    def problem_vasile_p12077_q1(self):
        return c3s(a**2*(a-b)*(a-c)*(a+2*b)*(a+2*c)) + c3p(a-b)**2, [], []

    def problem_vasile_p12077_q2(self):
        return c3s(a**2*(a-b)*(a-c)*(a-4*b)*(a-4*c)) + 7*c3p(a-b)**2, [], []

    def problem_vasile_p12078(self):
        return c3p(a**2+2*b*c) + c3p(a-b)**2, [], []

    def problem_vasile_p12079(self):
        return c3p(2*a**2+5*a*b+2*b**2) + c3p(a-b)**2, [], []

    def problem_vasile_p12080(self):
        return c3p(a**2+2*a*b/3+b**2) - 64*c3p(a**2+b*c)/27, [], []

    def problem_vasile_p12081(self):
        return c3s(a**2*(a-b)*(a-c)) - 2*c3p(a-b)**2/c3s(a**2), [], []

    def problem_vasile_p12082(self):
        return c3s((a-b)*(a-c)*(a-2*b)*(a-2*c)) - 8*c3p(a-b)**2/c3s(a**2), [], []

    def problem_vasile_p12083(self):
        return c3s((a**2+3*b*c)/(b**2+c**2)), [], []

    def problem_vasile_p12084(self):
        return c3s((a**2+6*b*c)/(b**2-b*c+c**2)), [], []

    def problem_vasile_p12085(self):
        return c3s((4*a**2+23*b*c)/(b**2+c**2)), [], []

    def problem_vasile_p12086(self):
        return 20*c3s(a**6) + 43*c3p(a)*c3s(a**3) - 189, [], [c3s(a*b)-3]

    def problem_vasile_p12087(self):
        return 4*c3s((a**2+b*c)*(a-b)*(a-c)*(a-3*b)*(a-3*c)) - 7*c3p(a-b)**2, [], []

    def problem_vasile_p12088(self):
        return 4*c3s(b*c*(a-b)*(a-c)*(a-x*b)*(a-x*c)) + c3p(a-b)**2, [c3s(a*b)], []

    def problem_vasile_p12089(self):
        return c3s(a**2*b+a*b**2)**2 - 4*c3s(a*b)*c3s(a**2*b**2), [], []

    def problem_vasile_p12090(self):
        return c3s((a-1)*(a-25))/(a**2+23), [], [c3s(a)-3]

    def problem_vasile_p12091(self):
        return c3s((b+c)**2/a**2) - 2, [], []

    def problem_vasile_p12092_q1(self):
        return c3p(a**2+1) - 8/sqrt(3)*Abs(c3p(a-b)), [], []

    def problem_vasile_p12092_q2(self):
        return c3p(a**2-a+1) - Abs(c3p(a-b)), [], []

    def problem_vasile_p12093(self):
        return c3p(1-a+a**2) - 1, [], [c3s(a)-3]

    def problem_vasile_p12094(self):
        return c3s(a*(a-4)/(a**2+2)), [], [c3s(a)]

    def problem_vasile_p12095(self):
        return c4p(1-a+a**2) - ((1+c4p(a))/2)**2, [], []

    def problem_vasile_p12096(self):
        return c4p(a+1/a) - c4s(a)*c4s(1/a), [], [c4p(a)]

    def problem_vasile_p12097(self):
        return 16-c4s(a**3), [], [c4s(a)-4, c4s(a)-7]

    def problem_vasile_p12098(self):
        return 7*c4s(a**2)**2 - 12*c4s(a**4), [], [c4s(a)]

    def problem_vasile_p12099(self):
        return c4s(a**2)**3 - 3*c4s(a**3)**2, [], [c4s(a)]

    def problem_vasile_p12100(self):
        return c4p(1+a**2) - c4s(a)**2, [], [c4p(a)-1]

    def problem_vasile_p12101(self):
        return 4 - c4s(a**2*b**2*c**2), [], [c4s(a**2)-4]

    def problem_vasile_p12102(self):
        return c4s((1-a)**4 - a**4), [], [c4s(a**2)-1]

    def problem_vasile_p12103(self):
        fx = lambda x: (1-x)/(1-x+x**2)
        return Add(*[fx(_) for _ in [a,b,c,d]]), [_+Rational(1,2) for _ in [a,b,c,d]], [c4s(a)-4]

    @mark(mark.skip)
    def problem_vasile_p12104(self):
        fx = lambda x: (1-x)/(1+x+x**2)
        return Add(*[fx(_) for _ in [a,b,c,d,e]]), [_+3 for _ in [a,b,c,d,e]], [a+b+c+d+e-5]

    def problem_vasile_p12105(self):
        return 30*(a**4+b**4+c**4+d**4+e**4) - 7*(a**2+b**2+c**2+d**2+e**2)**2, [], [a+b+c+d+e]

    @mark(mark.noimpl, mark.nvars)
    def problem_vasile_p12106(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_vasile_p12107(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_vasile_p12108(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_vasile_p12109(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_vasile_p12110(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_vasile_p12111(self):
        ...


    def problem_vasile_p13001(self):
        return c3s(a**2)+2*c3p(a)+1-2*c3s(a*b), [a,b,c], []

    def problem_vasile_p13002(self):
        return c3s(a**2)+x*c3p(a)+(2*x+3)-(x+2)*c3s(a*b), [a,b,c,x,sqrt(2)-x], []

    def problem_vasile_p13003(self):
        return c3p(a)*c3s(a)+2*c3s(a**2)+3-4*c3s(a*b), [a,b,c], []

    def problem_vasile_p13004(self):
        return c3s(a*(b**2+c**2))+3-3*c3s(a*b), [a,b,c], []

    def problem_vasile_p13005(self):
        return (c3s(a**2)/3)**3 - c3p(a**2) - c3p((a-b)**2), [a,b,c], []

    def problem_vasile_p13006(self):
        return (c3s(a)-3)*(c3s(a*b)-3) - 3*(c3p(a)-1)*(c3s(a)-c3s(a*b)), [a,b,c], []

    def problem_vasile_p13007_q1(self):
        return c3s(a**3+a*b-5*a)+9, [a,b,c], []

    def problem_vasile_p13007_q2(self):
        return c3s(a**3+4*a*b-11*a)+18, [a,b,c], []

    def problem_vasile_p13008_q1(self):
        return c3s(a**3)+c3p(a)+8-4*c3s(a), [a,b,c], []

    def problem_vasile_p13008_q2(self):
        return 4*c3s(a**3)+15*c3p(a)+54-27*c3s(a), [a,b,c], []

    def problem_vasile_p13009(self):
        return c3s(a*b-a**2*b**2), [a,b,c], [c3s(a-a**2)]

    def problem_vasile_p13010(self):
        return c3p(a**2+2*b*c)-c3s(a*b)**3, [a,b,c], []

    def problem_vasile_p13011(self):
        return c3p(2*a**2+b*c)-c3s(a*b)**3, [a,b,c], []

    def problem_vasile_p13012_q1(self):
        return c3p(a+b) - c3p(a**2+b**2), [a,b,c], [c3s(a)-2]

    def problem_vasile_p13012_q2(self):
        return 2 - c3p(a**2+b**2), [a,b,c], [c3s(a)-2]

    def problem_vasile_p13013(self):
        return 2 - c3p(a**3+b**3), [a,b,c], [c3s(a)-2]

    def problem_vasile_p13014(self):
        return 2 - c3p(a**3+b**3), [a,b,c], [c3s(a**2)-2]

    def problem_vasile_p13015(self):
        return 36 - c3p(3*a**2-2*a*b+3*b**2), [a,b,c], [c3s(a)-2]

    def problem_vasile_p13016(self):
        return 3 - c3p(a**2-4*a*b+b**2), [a,b,c], [c3s(a)-3]

    def problem_vasile_p13017(self):
        return c3p(a) + 12/c3s(a*b) - 5, [a,b,c], [c3s(a)-3]

    def problem_vasile_p13018(self):
        return 5*c3s(a) + 3/c3p(a) - 18, [a,b,c], [c3s(a**2)-3]

    def problem_vasile_p13019(self):
        return 12+9*c3p(a)-7*c3s(a*b), [a,b,c], [c3s(a**2)-3]

    def problem_vasile_p13020(self):
        return 21+18*c3p(a)-13*c3s(a*b), [a,b,c], [c3s(a**2)-3]

    def problem_vasile_p13021(self):
        return c3p(2-a*b)-1, [a,b,c], [c3s(a**2)-3]

    def problem_vasile_p13022(self):
        return c3s(a/3)**5 - c3s(a**2)/3, [a,b,c], [c3p(a)-1]

    @mark(mark.skip)
    def problem_vasile_p13023(self):
        return c3s(a**3+a**(-3))+21-3*c3s(a)*c3s(1/a), [a,b,c], [c3p(a)-1]

    def problem_vasile_p13024(self):
        return c3s(a**2-a*b) - Rational(9,4)*(c3s(a)-3), [a,b,c], [c3p(a)-1]

    def problem_vasile_p13025(self):
        return c3s(a**2+a-2*a*b), [a,b,c], [c3p(a)-1]

    def problem_vasile_p13026(self):
        return c3s(a**2+15*a*b-16*a), [a,b,c], [c3p(a)-1]

    def problem_vasile_p13027(self):
        return 2/c3s(a)+Rational(1,3)-3/c3s(a*b), [a,b,c], [c3p(a)-1]

    def problem_vasile_p13028(self):
        return c3s(a*b)+6/c3s(a)-5, [a,b,c], [c3p(a)-1]

    def problem_vasile_p13029(self):
        return cbrt(c3p(1+a)) - (4*(1+c3s(a)))**Rational(1,4), [a,b,c], [c3p(a)-1]

    def problem_vasile_p13030(self):
        return c3s(a**6)-3*c3p(a**2)-18*c3p(a**2-b*c), [a,b,c], []

    def problem_vasile_p13031(self):
        return c3s(1/a**2-a**2), [a,b,c], [c3s(a)-3]

    def problem_vasile_p13032(self):
        return c3s(a**3)+7*c3p(a)-10, [a,b,c], [c3s(a*b)-3]

    def problem_vasile_p13033(self):
        return 3 - c3s(a**4*b**4), [a,b,c], [c3s(a**3)-3]

    def problem_vasile_p13034(self):
        return c3p(a+1)**2 - 4*c3s(a)*c3s(a*b) - 28*c3p(a), [a,b,c], []

    def problem_vasile_p13035(self):
        return 1+8*c3p(a)-9*Min(a,b,c), [a,b,c], [c3s(a)-3]

    def problem_vasile_p13036(self):
        return 1+4*c3p(a)-5*Min(a,b,c), [a,b,c], [c3s(a**2)-3]

    def problem_vasile_p13037(self):
        return c3p(1-a) + (sqrt(3)-1)**3, [a,b,c], [c3s(a)-c3p(a)]

    def problem_vasile_p13038(self):
        return 1 - c3p(a**2+b*c), [a,b,c], [c3s(a)-2]

    def problem_vasile_p13039(self):
        return c3s(a)**6 - c3p(8*a**2+b*c), [a,b,c], []

    def problem_vasile_p13040(self):
        return c3s(a) - c3p(a) - 2, [a,b,c], [c3s(a**2*b**2)-3]

    def problem_vasile_p13041(self):
        return c3p(a**2+3) - 192, [a,b,c], [c3s(a)-5]

    def problem_vasile_p13042(self):
        return c3s(a**2)+c3p(a)+2-c3s(a)-c3s(a*b), [a,b,c], []

    def problem_vasile_p13043(self):
        return c3s(a**3*(b+c)*(a-b)*(a-c)) - 3*c3p((a-b)**2), [a,b,c], []

    def problem_vasile_p13044(self):
        return c3s(a)+4*c3p(a)-2*c3s(a*b), [a,b,c,1-a,1-b,1-c], []

    def problem_vasile_p13045(self):
        return c3s(a**2*b**2-a*b), [_ - Rational(2,3) for _ in [a,b,c]], [c3s(a) - 3]

    def problem_vasile_p13046(self):
        return c3s(1/a - a**2), [a,1-a,b-1,c-b], [c3s(a)-3]

    def problem_vasile_p13047(self):
        return c3s(1/a**2 - a**2), [a,1-a,b-1,c-b], [c3s(a-1/a)]

    @mark(mark.noimpl, mark.nvars)
    def problem_vasile_p13048(self):
        ...

    def problem_vasile_p13049_q1(self):
        E = c3s(a*(a-b)*(a-c))
        return c3s(a)*E - c3s(a*b*(a-b)**2), [a,b,c], []

    def problem_vasile_p13049_q2(self):
        E = c3s(a*(a-b)*(a-c))
        return 2*c3s(1/a)*E - c3s((a-b)**2), [a,b,c], []

    @mark(mark.noimpl)
    def problem_vasile_p13050(self):
        """Comparison of the tightness of inequalities is not well-defined."""
        ...

    def problem_vasile_p13051(self):
        return c3s(sqrt(a) - a*b), [a,b,c], [c3p(a+b)-8]

    def problem_vasile_p13052(self):
        ub = 4+3*sqrt(2)
        return 9*c3s(a*b)*c3s(a**2) - c3s(a)**4, [a-1,b-1,c-1,ub-a,ub-b,ub-c], []

    def problem_vasile_p13053_q1(self):
        return c3s(a**2)+12-5*c3s(a*b), [a,b,c], [c3s(a)+c3p(a)-4]

    def problem_vasile_p13053_q2(self):
        return 3*c3s(a**2)+13*c3s(a*b)-48, [a,b,c], [c3s(a)+c3p(a)-4]

    def problem_vasile_p13054(self):
        return c3s(a*b)-1-2*c3p(a), [b+c-a,c+a-b,a+b-c], [c3s(a**2)-3]

    @mark(mark.noimpl)
    def problem_vasile_p13055(self):
        """Wrong problem: Given a,b,c to be the sides of a triangle, a^2+b^2+c^2=3,
        prove that a^2b^2+b^2c^2+c^2a^2>=ab+bc+ca.
        A counterexample is (a,b,c) = (1.1, 1.1, sqrt(0.58)).
        """
        # return c3s(a**2*b**2-a*b), [b+c-a,c+a-b,a+b-c], [c3s(a**2)-3]

    def problem_vasile_p13056(self):
        return c3s(1/a)+Rational(41,6)-3*c3s(a**2), [b+c-a,c+a-b,a+b-c], [c3s(a)-3]

    @mark(mark.noimpl, mark.quant)
    def problem_vasile_p13057(self):
        ...

    def problem_vasile_p13058(self):
        return 9/c3p(a)+16-75/c3s(a*b), [a,b,c], [c3s(a)-3]

    def problem_vasile_p13059(self):
        return 8*c3s(1/a)+9-10*c3s(a**2), [a,b,c], [c3s(a)-3]

    def problem_vasile_p13060(self):
        return 7*c3s(a**2)+8*c3s(a**2*b**2)+4*c3p(a**2)-49, [a,b,c], [c3s(a)-3]

    def problem_vasile_p13061(self):
        return (c3s(a**3)+c3p(a))**2 - 2*c3p(a**2+b**2), [a,b,c], []

    def problem_vasile_p13062(self):
        return c3s(a*b*(a+b))**2 - 4*c3s(a*b)*c3s(a**2*b**2), [a,b,c], []

    def problem_vasile_p13063(self):
        return 4*c3s(a**3)+7*c3p(a)+125-48*c3s(a), [a,b,c], [c3s(a*b)-3]

    def problem_vasile_p13064(self):
        return c3s(a*sqrt(a)) + 4*c3p(a) - 2*c3s(a*b), [a,b,c,1-a,1-b,1-c], []

    def problem_vasile_p13065(self):
        return c3s(a*sqrt(a)) - 3*(c3s(a*b)-c3p(a))/2, [a,b,c,1-a,1-b,1-c], []

    def problem_vasile_p13066(self):
        return 3*c3s(a*sqrt(a)) + 500*c3p(a)/81 - 5*c3s(a*b), [a,b,c], []

    def problem_vasile_p13067(self):
        return c3s(a)-2-c3p(a), [b+c-a,c+a-b,a+b-c], [c3s(a**2)-3]

    @mark(mark.noimpl, mark.quant)
    def problem_vasile_p13068(self):
        ...

    def problem_vasile_p13069(self):
        return 4*c3s(a**4)+45-19*c3s(a**2), [a,b,c], [c3s(a)-3]

    def problem_vasile_p13070(self):
        return c3s(a*(a-b)*(a-c)*(a-x*b)*(a-x*c)), [a,b,c,2-x], []

    def problem_vasile_p13071(self):
        return c3s((b+c)*(a-b)*(a-c)*(a-x*b)*(a-x*c)), [a,b,c], []

    def problem_vasile_p13072(self):
        return c3s(a*(a-2*b)*(a-2*c)*(a-5*b)*(a-5*c)), [a,b,c], []

    def problem_vasile_p13073(self):
        return 10*c3s(a**2*b**2)-c3s(a**4)-9*c3p(a)*c3s(a), [b+c-a,c+a-b,a+b-c], []

    def problem_vasile_p13074(self):
        return 5*c3s(a*b*(a**2+b**2))-3*c3s(a**4)-7*c3p(a)*c3s(a), [b+c-a,c+a-b,a+b-c], []

    def problem_vasile_p13075(self):
        return -c3s((b**2+c**2-6*b*c)/a + 4*a), [b+c-a,c+a-b,a+b-c], []

    @mark(mark.noimpl, mark.quant)
    def problem_vasile_p13076(self):
        ...

    def problem_vasile_p13077(self):
        return c3s(a*(b+c)*(a-b)*(a-c)*(a-2*b)*(a-2*c)) - c3p((a-b)**2), [a,b,c], []

    def problem_vasile_p13078_q1(self):
        return c3s(a*(a-b)*(a-c)*(a-x*b)*(a-x*c)) + 4*(x-2)*c3p((a-b)**2)/c3s(a), [a,b,c,x-2,6-x], []

    def problem_vasile_p13078_q2(self):
        return c3s(a*(a-b)*(a-c)*(a-x*b)*(a-x*c)) + (x+2)**2/4*c3p((a-b)**2)/c3s(a), [a,b,c,x-6], []

    def problem_vasile_p13079(self):
        return c3p(3*a**2+2*a*b+3*b**2) - 8*c3p(a**2+3*b*c), [a,b,c], []

    @mark(mark.skip)
    def problem_vasile_p13080(self):
        return (x+2) - c3p(a**2+x*a*b+b**2), [a,b,c,x+Rational(2,3),Rational(11,8)-x], [c3s(a)-2]

    def problem_vasile_p13081(self):
        return 4 - c3p(2*a**2+b*c), [a,b,c], [c3s(a)-2]

    def problem_vasile_p13082(self):
        return c3s((a-b)*(a-c)*(a-2*b)*(a-2*c)) - 5*c3p((a-b)**2)/c3s(a*b), [a,b,c], []

    @mark(mark.noimpl, mark.quant)
    def problem_vasile_p13083(self):
        ...

    def problem_vasile_p13084(self):
        return (c3s(a)-3)*(c3s(1/a)-3)+c3p(a)+1/c3p(a)-2, [a,b,c], []

    def problem_vasile_p13085_q1(self):
        return 3*(c3s(a*b)-Rational(2,3))/7 - sqrt(2*c3s(a)/3-1), [a,b,c], [c3p(a)-1]

    def problem_vasile_p13085_q2(self):
        return (c3s(a*b)-3) - 46*(sqrt(c3s(a)-2) - 1)/27, [a,b,c], [c3p(a)-1]

    def problem_vasile_p13086(self):
        return c3s(a*b) + 50/(c3s(a)+5) - Rational(37,4), [a,b,c], [c3p(a)-1]

    def problem_vasile_p13087_q1(self):
        return (c3s(a)-3)**2 + 1 - c3s(a**2)/3, [a,b,c], [c3p(a)-2]

    def problem_vasile_p13087_q2(self):
        return c3s(a**2) + 3*(3-c3s(a))**2 - 3, [a,b,c], [c3p(a) - Rational(1,2)]

    def problem_vasile_p13088(self):
        return 4*c3s(b*c/a) + 9*c3p(a) - 21, [a,b,c], [c3s(a)-3]

    def problem_vasile_p13089(self):
        return c3s(a**2)+c3p(a)-4, [a,b,c], [c3s(a*b)-c3p(a)-2]

    def problem_vasile_p13090(self):
        return c3s(((b+c)/a - 2-sqrt(2))**2) - 6, [a,b,c], []

    def problem_vasile_p13091(self):
        return 2*c3s(a**3)+9*c3s(a*b)+39-24*c3s(a), [a,b,c], []

    def problem_vasile_p13092(self):
        return c3s(a**3)-3-Abs(c3p(a-b)), [a,b,c], [c3s(a**2)-3]

    def problem_vasile_p13093(self):
        return c3s(a**4-a**2*b**2) - 2*Abs(c3s(a**3*b-a*b**3)), [a,b,c], []

    def problem_vasile_p13094(self):
        return c3s(a**4)-c3p(a)*c3s(a) - 2*sqrt(2)*Abs(a**3*b-a*b**3), [a,b,c], []

    def problem_vasile_p13095(self):
        return c3s((1-a)/(1+a+a**2)), [a+5,b+5,c+5], [c3s(a)-3]

    def problem_vasile_p13096(self):
        return c3s((1-a)/(1-x*a)**2), [a,b,c,x - Rational(4,3)], [c3s(a)-3]

    def problem_vasile_p13097(self):
        return c4p(1-a) - c4p(a), [a,b,c,d], [c4s(a**2)-1]

    def problem_vasile_p13098(self):
        return c4p(a-1) - (x-1)**4, [a,b,c,d], [c4s(1/a**2) - 4/x**2]

    @mark(mark.skip)
    def problem_vasile_p13099(self):
        return c4p((1+a**3)/(1+a**2)) - (1+c4p(a))/2, [a,b,c,d], []

    def problem_vasile_p13100(self):
        return c4p(a+1/a-1) + 3 - c4s(1/a), [a,b,c,d], [c4s(a)-4]

    def problem_vasile_p13101(self):
        return 4*c4s(a)+15*c4s(a*b*c)-c4s(a)**3, [a,b,c,d], []

    def problem_vasile_p13102(self):
        return 1 + 2*c4s(a*b*c) - 9*Min(a,b,c,d), [a,b,c,d], [c4s(a)-4]

    def problem_vasile_p13103(self):
        return 5*c4s(a**2) - c4s(a**3) - 16, [a,b,c,d], [c4s(a)-4]

    def problem_vasile_p13104(self):
        return 3*c4s(a**2) + 4*c4p(a) - 16, [a,b,c,d], [c4s(a)-4]

    def problem_vasile_p13105(self):
        """There is a typo in the book."""
        return 64 + 44*c4p(a) - 27*c4s(a*b*c), [a,b,c,d], [c4s(a)-4]

    @mark(mark.skip)
    def problem_vasile_p13106(self):
        return (1 - c4p(a))*(c4s(a**2 - 1/a**2)), [a,b,c,d], [c4s(a-1/a)]

    def problem_vasile_p13107(self):
        return c4p(1-a)*c4s(1/a) - Rational(81,16), [a,b,c,d], [c4s(a)-1]

    def problem_vasile_p13108(self):
        return c4s(a**2) - Rational(7,4), [a,b,c,d], [c4s(a)-2, c4s(a**3)-2]

    def problem_vasile_p13109(self):
        return c4p(1+2*a) - c4p(5-2*a), [a,b,c,d,4-a,4-b,4-c,4-d], [c4p(a)-1]

    def problem_vasile_p13110(self):
        k_ = c4s(a)*c4s(1/a)
        return b+c-a, [a,b,c,d,(1+sqrt(10))**2-k_], []

    @mark(mark.skip)
    def problem_vasile_p13111(self):
        k_ = c4s(a)*c4s(1/a)
        tri = lambda x,y,z: Min(x+y-z,y+z-x,z+x-y)
        return Max(tri(a,b,c),tri(b,c,d),tri(c,d,a),tri(d,a,b)),\
            [a,b,c,d,Rational(119,16)-k_], []

    def problem_vasile_p13112(self):
        k_ = c4s(a)**2/c4s(a**2)
        return b+c-a, [a,b,c,d,k_-Rational(11,3)], []

    @mark(mark.skip)
    def problem_vasile_p13113(self):
        k_ = c4s(a)**2/c4s(a**2)
        tri = lambda x,y,z: Min(x+y-z,y+z-x,z+x-y)
        return Max(tri(a,b,c),tri(b,c,d),tri(c,d,a),tri(d,a,b)),\
            [a,b,c,d,k_-Rational(49,15)], []

    def problem_vasile_p13114_q1(self):
        return 4*c5s(a**4) - c5s(a**2)**2, [a,b,c,d,e], [a+b+c-3*(d+e)]

    def problem_vasile_p13114_q2(self):
        return -12*c5s(a**4) + 7*c5s(a**2)**2, [a,b,c,d,e], [a+b+c-(d+e)]

    def problem_vasile_p13115(self):
        return 31*c5s(a**2)-150-c5s(a**4), [a,b,c,d,e], [c5s(a)-5]

    def problem_vasile_p13116(self):
        return 5 - c5p(a)*c5s(a**4), [a,b,c,d,e], [a,b,c,d,e], [c5s(a)-5]

    @mark(mark.skip)
    def problem_vasile_p13117(self):
        return c5s(1/a) + 20/c5s(a**2) - 9, [a,b,c,d,e], [c5s(a)-5]

    def problem_vasile_p13118(self):
        return c5p(a+1/a) + 68 - 4*c5s(a)*c5s(1/a), [a-1,b-1,c-1,d-1,e-1], []

    def problem_vasile_p13119(self):
        return Rational(1,36) - a*b*c*x*y*z, [a,b,c,x,y,z],\
            [(a+b+c)*(x+y+z)-4, (a**2+b**2+c**2)*(x**2+y**2+z**2)-4]

    @mark(mark.noimpl, mark.nvars)
    def problem_vasile_p13120(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_vasile_p13121(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_vasile_p13122(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_vasile_p13123(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_vasile_p13124(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_vasile_p13125(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_vasile_p13126(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_vasile_p13127(self):
        ...