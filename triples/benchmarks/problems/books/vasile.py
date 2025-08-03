from ..problem_set import ProblemSet, mark
from sympy.abc import a,b,c,d,e,f,k,m,n,p,q,r,s,u,v,x,y,z,w
from sympy import symbols, prod, Rational, Add, Mul, exp, sqrt, cbrt, sin, cos, pi, Abs, Min, Max

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
        return 4*c3p(a+1/a)-9*c3s(a), [c3p(a)], []

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
        return c3p(a**2+a*b+b**2) - 3*c3s(a*b)**3, [a,b,-c], []

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
        return c3s(a**2)-3-(2+sqrt(3))*(c3s(a)-3), [], [c3s(a*b)-c3p(a)-2]

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
        return c3p(a+1/a-1) + 2 - c3s(a)*c3s(1/a)/3, [c3p(a)], []

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
        return 16-c4s(a**3), [], [c4s(a)-4, c4s(a**2)-7]

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

    @mark(mark.skip)
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
        return 4*c4s(a**3)+15*c4s(a*b*c)-c4s(a)**3, [a,b,c,d], []

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
        return 5 - c5p(a)*c5s(a**4), [a,b,c,d,e], [c5s(a)-5]

    @mark(mark.skip)
    def problem_vasile_p13117(self):
        return c5s(1/a) + 20/c5s(a**2) - 9, [a,b,c,d,e], [c5s(a)-5]

    @mark(mark.skip)
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


class MathematicalInequalitiesVol2(MathematicalInequalities):
    """Symmetric Rational and Irrational Inequalities"""
    def problem_vasile_p21001(self):
        return 1/(1+a)**2 + 1/(1+b)**2 - 1/(1+a*b), [a,b], []

    def problem_vasile_p21002(self):
        return c3s((a**2 - b*c)/(3*a + b + c)), [a,b,c], []

    def problem_vasile_p21003(self):
        return 3 - c3s((4*a**2 - b**2 - c**2)/(a*(b + c))), [a,b,c], []

    def problem_vasile_p21004_p1(self):
        return c3s(1/(a**2+b*c)) - 3/c3s(a*b), [a,b,c], []

    def problem_vasile_p21004_p2(self):
        return c3s(1/(2*a**2+b*c)) - 2/c3s(a*b), [a,b,c], []

    def problem_vasile_p21004_p3(self):
        return c3s(1/(a**2+2*b*c)) - 2/c3s(a*b), [a,b,c], []

    def problem_vasile_p21005(self):
        return c3s(a*(b+c)/(a**2+b*c)) - 2, [a,b,c], []

    def problem_vasile_p21006(self):
        return c3s(a**2/(b**2+c**2)) - c3s(a/(b+c)), [a,b,c], []

    def problem_vasile_p21007(self):
        return c3s(1/(b+c)) - c3s(a/(a**2+b*c)), [a,b,c], []

    def problem_vasile_p21008(self):
        return c3s(1/(b+c)) - c3s(2*a/(3*a**2+b*c)), [a,b,c], []

    def problem_vasile_p21009_p1(self):
        return c3s(a/(b+c)) - Rational(13,6) + Rational(2,3)*c3s(a*b)/c3s(a**2), [a,b,c], []

    def problem_vasile_p21009_p2(self):
        return c3s(a/(b+c)) - Rational(3,2) - (sqrt(3)-1)*(1 - c3s(a*b)/c3s(a**2)), [a,b,c], []

    def problem_vasile_p21010(self):
        return (c3s(a)/c3s(a*b))**2 - c3s(1/(a**2+2*b*c)), [a,b,c], []

    def problem_vasile_p21011(self):
        return c3s(a**2*(b+c)/(b**2+c**2)) - c3s(a), [a,b,c], []

    def problem_vasile_p21012(self):
        return 3*c3s(a**2)/c3s(a) - c3s((a**2+b**2)/(a+b)), [a,b,c], []

    def problem_vasile_p21013(self):
        return c3s(1/(a**2+a*b+b**2)) - 9/c3s(a)**2, [a,b,c], []

    def problem_vasile_p21014(self):
        return Rational(1,3) - c3s(a**2/(2*a+b)/(2*a+c)), [a,b,c], []

    def problem_vasile_p21015_p1(self):
        return 1/c3s(a) - c3s(a/(2*a+b)/(2*a+c)), [a,b,c], []

    def problem_vasile_p21015_p2(self):
        return 1/c3s(a) - c3s(a**3/(2*a**2+b**2)/(2*a**2+c**2)), [a,b,c], []

    def problem_vasile_p21016(self):
        return c3s(1/(a+2*b)/(a+2*c)) - 1/c3s(a)**2 - 2/c3s(a*b)/3, [a,b,c], []

    def problem_vasile_p21017_p1(self):
        return c3s(1/(a-b)**2) - 4/c3s(a*b), [a,b,c], []

    def problem_vasile_p21017_p2(self):
        return c3s(1/(a**2-a*b+b**2)) - 3/c3s(a*b), [a,b,c], []

    def problem_vasile_p21017_p3(self):
        return c3s(1/(a**2+b**2)) - 5/c3s(a*b)/2, [a,b,c], []

    def problem_vasile_p21018(self):
        return c3s((a**2+b**2)*(a**2+c**2)/(a+b)/(a+c)) - c3s(a**2), [a,b,c], []

    def problem_vasile_p21019(self):
        return 1 - c3s(1/(a**2+b+c)), [a,b,c], [c3s(a)-3]

    def problem_vasile_p21020(self):
        return c3s((a**2-b*c)/(a**2+3)), [a,b,c], [c3s(a)-3]

    def problem_vasile_p21021(self):
        return c3s((1-b*c)/(5+2*a)), [a,b,c], [c3s(a)-3]

    def problem_vasile_p21022(self):
        return Rational(3,4) - c3s(1/(a**2+b**2+2)), [a,b,c], [c3s(a)-3]

    def problem_vasile_p21023(self):
        return Rational(1,2) - c3s(1/(4*a**2+b**2+c**2)), [a,b,c], [c3s(a)-3]

    def problem_vasile_p21024(self):
        return 1 - c3s(b*c/(a**2+1)), [a,b,c], [c3s(a)-2]

    def problem_vasile_p21025(self):
        return Rational(1,4) - c3s(b*c/(a+1)), [a,b,c], [c3s(a)-1]

    def problem_vasile_p21026(self):
        return 3/c3p(a)/11 - c3s(1/(a*(2*a**2+1))), [a,b,c], [c3s(a)-1]

    def problem_vasile_p21027(self):
        return 1 - c3s(1/(a**3+b+c)), [a,b,c], [c3s(a)-3]

    def problem_vasile_p21028(self):
        return c3s(a**2/(1+b**3+c**3)) - 1, [a,b,c], [c3s(a)-3]

    def problem_vasile_p21029(self):
        return Rational(3,5) - c3s(1/(6-a*b)), [a,b,c], [c3s(a)-3]

    def problem_vasile_p21030(self):
        return Rational(1,3) - c3s(1/(2*a**2+7)), [a,b,c], [c3s(a)-3]

    def problem_vasile_p21031(self):
        return Rational(3,4) - c3s(1/(a**2+3)), [a-b,b-1,1-c,c], [c3s(a)-3]

    def problem_vasile_p21032(self):
        return c3s(1/(2*a**2+3)) - Rational(3,5), [a,b,c], [c3s(a)-3]

    def problem_vasile_p21033(self):
        return c3s(1/(a**2+2)) - 1, [a-1,1-b,b-c,c], [c3s(a)-3]

    def problem_vasile_p21034(self):
        return c3s(1/(a+b)) - c3s(a)/6 - 3/c3s(a), [a,b,c], [c3s(a*b)-3]

    def problem_vasile_p21035(self):
        return c3s(1/(a**2+1)) - Rational(3,2), [a,b,c], [c3s(a*b)-3]

    def problem_vasile_p21036(self):
        return c3s(a**2/(a**2+b+c)) - 1, [a,b,c], [c3s(a*b)-3]

    def problem_vasile_p21037_p1(self):
        return 3 - c3s((b*c+4)/(a**2+4)), [a,b,c], [c3s(a*b)-3]

    def problem_vasile_p21037_p2(self):
        return c3s((b*c+2)/(a**2+2)) - 3, [a,b,c], [c3s(a*b)-3]

    def problem_vasile_p21038(self):
        return 3/(1+k) - c3s(1/(a+k)), [a,b,c,k-2-sqrt(3)], [c3s(a*b)-3]

    def problem_vasile_p21039(self):
        return 3 - c3s(a*(b+c)/(1+b*c)), [a,b,c], [c3s(a**2)-3]

    def problem_vasile_p21040(self):
        return 3 - c3s((a**2+b**2)/(a+b)), [a,b,c], [c3s(a**2)-3]

    def problem_vasile_p21041(self):
        return 7*c3s(a)/6 - 2 - c3s(a*b/(a+b)), [a,b,c], [c3s(a**2)-3]

    def problem_vasile_p21042_p1(self):
        return Rational(3,2) - c3s(1/(3-a*b)), [a,b,c], [c3s(a**2)-3]

    @mark(mark.skip)
    def problem_vasile_p21042_p2(self):
        return 3/(sqrt(6)-1) - c3s(1/(sqrt(6)-a*b)), [a,b,c], [c3s(a**2)-3]

    @mark(mark.skip)
    def problem_vasile_p21043(self):
        return c3s(1/(1+a**5)) - Rational(3,2), [a,b,c], [c3s(a**2)-3]

    def problem_vasile_p21044(self):
        return c3s(1/(a**2+a+1)) - 1, [a,b,c], [c3p(a)-1]

    def problem_vasile_p21045(self):
        return 3 - c3s(1/(a**2-a+1)), [a,b,c], [c3p(a)-1]

    def problem_vasile_p21046(self):
        return c3s((3+a)/(1+a)**2) - 3, [a,b,c], [c3p(a)-1]

    def problem_vasile_p21047(self):
        return c3s((7-6*a)/(2+a**2)) - 1, [a,b,c], [c3p(a)-1]

    @mark(mark.skip)
    def problem_vasile_p21048(self):
        return c3s(a**6/(1+2*a**5)) - 1, [a,b,c], [c3p(a)-1]

    def problem_vasile_p21049(self):
        return Rational(1,2) - c3s(a/(a**2+5)), [a,b,c], [c3p(a)-1]

    def problem_vasile_p21050(self):
        return c3s(1/(1+a)**2) + 2/c3p(1+a) - 1, [a,b,c], [c3p(a)-1]

    def problem_vasile_p21051(self):
        return 3/c3s(a) - 2/c3s(a*b) - 1/c3s(a**2), [a,b,c], [c3s(1/(a+b)) - Rational(3,2)]

    def problem_vasile_p21052_p1(self):
        return c3s(a/(b+c)) - Rational(51,28), [], [7*c3s(a**2)-11*c3s(a*b)]

    def problem_vasile_p21052_p2(self):
        return 2 - c3s(a/(b+c)), [], [7*c3s(a**2)-11*c3s(a*b)]

    def problem_vasile_p21053(self):
        return c3s(1/(a**2+b**2)) - 10/c3s(a)**2, [a,b,c], []

    def problem_vasile_p21054(self):
        return c3s(1/(a**2-a*b+b**2)) - 3/Max(a*b,b*c,c*a), [a,b,c], []

    def problem_vasile_p21055(self):
        return c3s(a*(2*a+b+c)/(b**2+c**2)) - 6, [a,b,c], []

    def problem_vasile_p21056(self):
        return c3s(a**2*(b+c)**2/(b**2+c**2)) - 2*c3s(a*b), [a,b,c], []

    def problem_vasile_p21057(self):
        return 3*c3s(a/(b**2-b*c+c**2)) + 5*c3s(c/(a*b)) - 8*c3s(1/a), [a,b,c], []

    def problem_vasile_p21058_p1(self):
        return 2*c3p(a)*c3s(1/(a+b)) + c3s(a**2) - 2*c3s(a*b), [a,b,c], []

    def problem_vasile_p21058_p2(self):
        return 3*c3s(a**2)/2/c3s(a) - c3s(a**2/(a+b)), [a,b,c], []

    def problem_vasile_p21059_p1(self):
        return c3s((a**2-b*c)/(b**2+c**2)) + 3*c3s(a*b)/c3s(a**2) - 3, [a,b,c], []

    def problem_vasile_p21059_p2(self):
        return c3s(a**2/(b**2+c**2)) + c3s(a*b)/c3s(a**2) - Rational(5,2), [a,b,c], []

    def problem_vasile_p21059_p3(self):
        return c3s((a**2+b*c)/(b**2+c**2)) - c3s(a*b)/c3s(a**2) - 2, [a,b,c], []

    def problem_vasile_p21060(self):
        return c3s(a**2/(b**2+c**2)) - c3s(a)**2/(2*c3s(a*b)), [a,b,c], []

    def problem_vasile_p21061(self):
        return c3s(2*a*b/(a+b)**2) + c3s(a**2)/c3s(a*b) - Rational(5,2), [a,b,c], []

    def problem_vasile_p21062(self):
        return c3s(a*b/(a+b)**2) + Rational(1,4) - c3s(a*b)/c3s(a**2), [a,b,c], []

    def problem_vasile_p21063(self):
        return c3s(a*b)/c3s(a**2) + Rational(5,4) -  c3s(3*a*b/(a+b)**2), [a,b,c], []

    def problem_vasile_p21064_p1(self):
        return c3s((a**3+a*b*c)/(b+c)) - c3s(a**2), [a,b,c], []

    def problem_vasile_p21064_p2(self):
        return c3s((a**3+2*a*b*c)/(b+c)) - c3s(a)**2/2, [a,b,c], []

    def problem_vasile_p21064_p3(self):
        return c3s((a**3+3*a*b*c)/(b+c)) - 2*c3s(a*b), [a,b,c], []

    def problem_vasile_p21065(self):
        return c3s((a**3+3*a*b*c)/(b+c)**2) - c3s(a), [a,b,c], []

    def problem_vasile_p21066_p1(self):
        return c3s((a**3+3*a*b*c)/(b+c)**3) - Rational(3,2), [a,b,c], []

    def problem_vasile_p21066_p2(self):
        return c3s((3*a**3+13*a*b*c)/(b+c)**3) - 6, [a,b,c], []

    def problem_vasile_p21067_p1(self):
        return c3s(a**3/(b+c)) + c3s(a*b) - 3*c3s(a**2)/2, [a,b,c], []

    def problem_vasile_p21067_p2(self):
        return c3s((2*a**2+b*c)/(b+c)) - 9*c3s(a**2)/2/c3s(a), [a,b,c], []

    def problem_vasile_p21068(self):
        return c3s(a*(b+c)/(b**2+b*c+c**2)) - 2, [a,b,c], []

    def problem_vasile_p21069(self):
        return c3s(a*(b+c)/(b**2+b*c+c**2)) - 2 - 4*c3p((a-b)/(a+b))**2, [a,b,c], []

    def problem_vasile_p21070(self):
        return c3s((a*b-b*c+c*a)/(b**2+c**2)) - Rational(3,2), [a,b,c], []

    def problem_vasile_p21071(self):
        return c3s((a*b+(k-1)*b*c+c*a)/(b**2+k*b*c+c**2)) - 3*(k+1)/(k+2), [a,b,c,k+2], []

    @mark(mark.skip)
    def problem_vasile_p21072(self):
        return 3/(k+2) - c3s((3*b*c-a*(b+c))/(b**2+k*b*c+c**2)), [a,b,c,k+2], []

    def problem_vasile_p21073(self):
        return c3s((a*b+1)/(a**2+b**2)) - Rational(4,3), [a,b,c], [c3s(a*b)-3]

    def problem_vasile_p21074(self):
        return c3s((5*a*b+1)/(a+b)**2) - 2, [a,b,c], [c3s(a*b)-3]

    def problem_vasile_p21075(self):
        return c3s((a**2-b*c)/(2*b**2-3*b*c+2*c**2)), [a,b,c], []

    def problem_vasile_p21076(self):
        return c3s((2*a**2-b*c)/(b**2-b*c+c**2)) - 3, [a,b,c], []

    def problem_vasile_p21077(self):
        return c3s(a**2/(2*b**2-b*c+2*c**2)) - 1, [a,b,c], []

    def problem_vasile_p21078(self):
        return c3s(1/(4*b**2-b*c+4*c**2)) - 9/c3s(a**2)/7, [a,b,c], []

    def problem_vasile_p21079(self):
        return c3s((2*a**2+b*c)/(b**2+c**2)) - Rational(9,2), [a,b,c], []

    def problem_vasile_p21080(self):
        return c3s((2*a**2+3*b*c)/(b**2+b*c+c**2)) - 5, [a,b,c], []

    def problem_vasile_p21081(self):
        return c3s((2*a**2+5*b*c)/(b+c)**2) - Rational(21,4), [a,b,c], []

    @mark(mark.skip)
    def problem_vasile_p21082(self):
        return c3s((2*a**2+(2*k+1)*b*c)/(b**2+k*b*c+c**2)) - 3*(2*k+3)/(k+2), [a,b,c,k+2], []

    @mark(mark.skip)
    def problem_vasile_p21083(self):
        return 3/(k+2) - c3s((3*b*c-2*a**2)/(b**2+k*b*c+c**2)), [a,b,c,k+2], []

    def problem_vasile_p21084(self):
        return c3s((a**2+16*b*c)/(b**2+c**2)) - 10, [a,b,c], []

    def problem_vasile_p21085(self):
        return c3s((a**2+128*b*c)/(b**2+c**2)) - 46, [a,b,c], []

    def problem_vasile_p21086(self):
        return c3s((a**2+64*b*c)/(b+c)**2) - 18, [a,b,c], []

    @mark(mark.skip)
    def problem_vasile_p21087(self):
        return c3s((a**2*(b+c)+k*a*b*c)/(b**2+k*b*c+c**2)) - c3s(a), [a,b,c,k+1], []

    @mark(mark.skip)
    def problem_vasile_p21088(self):
        return c3s((a**3+(k+1)*a*b*c)/(b**2+k*b*c+c**2)) - c3s(a), [a,b,c,k+Rational(3,2)], []

    @mark(mark.noimpl)
    def problem_vasile_p21089(self):
        return c3s((2*a**k-b**k-c**k)/(b**2-b*c+c**2)), [a,b,c,k], []

    def problem_vasile_p21090_p1(self):
        return c3s((b+c-a)/(b**2-b*c+c**2)) - 2*c3s(a)/c3s(a**2), [b+c-a,c+a-b,a+b-c], []

    def problem_vasile_p21090_p2(self):
        return c3s((a**2-2*b*c)/(b**2-b*c+c**2)), [b+c-a,c+a-b,a+b-c], []

    def problem_vasile_p21091(self):
        return Rational(1,3) - c3s(a**2/(5*a**2+(b+c)**2)), [a,b,c], []

    def problem_vasile_p21092(self):
        return c3s((b**2+c**2-a**2)/(2*a**2+(b+c)**2)) - Rational(1,2), [a,b,c], []

    @mark(mark.skip)
    def problem_vasile_p21093(self):
        return 3/k - c3s((3*a**2-2*b*c)/(k*a**2+(b-c)**2)), [a,b,c,k], []

    @mark(mark.skip)
    def problem_vasile_p21094_p1(self):
        return c3s(a/(a**2+k*b*c)) - 9/c3s(a)/(1+k), [a,b,c,k-3-sqrt(7)], []

    @mark(mark.skip)
    def problem_vasile_p21094_p2(self):
        return c3s(a/(k*a**2+b*c)) - 9/c3s(a)/(1+k), [a,b,c,k-3-sqrt(7)], []

    def problem_vasile_p21095(self):
        return c3s(1/(2*a**2+b*c)) - 6/c3s(a**2+a*b), [a,b,c], []

    def problem_vasile_p21096(self):
        return c3s(1/(22*a**2+5*b*c)) - 1/c3s(a)**2, [a,b,c], []

    def problem_vasile_p21097(self):
        return c3s(1/(2*a**2+b*c)) - 8/c3s(a)**2, [a,b,c], []

    def problem_vasile_p21098(self):
        return c3s(1/(a**2+b*c)) - 12/c3s(a)**2, [a,b,c], []

    def problem_vasile_p21099_p1(self):
        return c3s(1/(a**2+2*b*c)) - 1/c3s(a**2) - 2/c3s(a*b), [a,b,c], []

    def problem_vasile_p21099_p2(self):
        return c3s(a*(b+c)/(a**2+2*b*c)) - 1 - c3s(a*b)/c3s(a**2), [a,b,c], []

    def problem_vasile_p21100_p1(self):
        return c3s(a)/c3s(a*b) - c3s(a/(a**2+2*b*c)), [a,b,c], []

    def problem_vasile_p21100_p2(self):
        return 1 + c3s(a**2)/c3s(a*b) - c3s(a*(b+c)/(a**2+2*b*c)), [a,b,c], []

    def problem_vasile_p21101_p1(self):
        return c3s(a/(2*a**2+b*c)) - c3s(a)/c3s(a**2), [a,b,c], []

    def problem_vasile_p21101_p2(self):
        return c3s((b+c)/(2*a**2+b*c)) - 6/c3s(a), [a,b,c], []

    def problem_vasile_p21102(self):
        return c3s(a*(b+c)/(a**2+b*c)) - c3s(a)**2/c3s(a**2), [a,b,c], []

    def problem_vasile_p21103(self):
        return c3s((b**2+c**2+sqrt(3)*b*c)/(a**3+k*b*c)) - 3*(2+sqrt(3))/(1+k), [a,b,c,k]

    def problem_vasile_p21104(self):
        return c3s(1/(a**2+b**2)) + 8/c3s(a**2) - 6/c3s(a*b), [a,b,c], []

    def problem_vasile_p21105(self):
        return 2 - c3s(a*(b+c)/(a**2+2*b*c)), [b+c-a,c+a-b,a+b-c], []

    def problem_vasile_p21106(self):
        return c3s((a**2-b*c)/(2*a**2+b**2+c**2)), [], []

    def problem_vasile_p21107(self):
        return Rational(3,2) - c3s((3*a**2-b*c)/(2*a**2+b**2+c**2)), [a,b,c], []

    def problem_vasile_p21108(self):
        return c3s((b+c)**2/(4*a**2+b**2+c**2)) - 2, [a,b,c], []

    def problem_vasile_p21109_p1(self):
        return 3/c3s(a*b)/5 - c3s(1/(11*a**2+2*b**2+2*c**2)), [a,b,c], []

    def problem_vasile_p21109_p2(self):
        return 1/c3s(a**2)/2 + 1/c3s(a*b) - c3s(1/(4*a**2+b**2+c**2)), [a,b,c], []

    def problem_vasile_p21110(self):
        return c3s(sqrt(a)/(b+c)) - Rational(3,2), [a,b,c], [c3s(a*b)-3]

    def problem_vasile_p21111(self):
        return c3s(1/(2+a)) - c3s(1/(1+b+c)), [a,b,c,c3s(a*b)-3], []

    def problem_vasile_p21112_p1(self):
        return -c3s((a**2-b*c)/(3*a**2+b**2+c**2)), [b+c-a,c+a-b,a+b-c], []

    def problem_vasile_p21112_p2(self):
        return -c3s((a**4-b**2*c**2)/(3*a**4+b**4+c**4)), [b+c-a,c+a-b,a+b-c], []

    def problem_vasile_p21113(self):
        return c3s(b*c/(4*a**2+b**2+c**2)) - Rational(1,2), [b+c-a,c+a-b,a+b-c], []

    def problem_vasile_p21114(self):
        return 9/c3s(a*b)/2 - c3s(1/(b**2+c**2)), [b+c-a,c+a-b,a+b-c], []

    @mark(mark.skip)
    def problem_vasile_p21115_p1(self):
        return Abs(c3s((a+b)/(a-b))) - 5, [b+c-a,c+a-b,a+b-c], []

    @mark(mark.skip)
    def problem_vasile_p21115_p2(self):
        return Abs(c3s((a**2+b**2)/(a**2-b**2))) - 3, [b+c-a,c+a-b,a+b-c], []

    def problem_vasile_p21116(self):
        return c3s((b+c)/a) + 3 - 6*c3s(a/(b+c)), [b+c-a,c+a-b,a+b-c], []

    def problem_vasile_p21117(self):
        return c3s((3*a*(b+c)-2*b*c)/((b+c)*(2*a+b+c))) - Rational(3,2), [a,b,c]

    def problem_vasile_p21118(self):
        return c3s((a*(b+c)-2*b*c)/((b+c)*(3*a+b+c))), [a,b,c], []

    @mark(mark.skip)
    def problem_vasile_p21119(self):
        return c3s((a**5-a**2)/(a**5+b**2+c**2)), [a,b,c,c3s(a**2)-3], []

    def problem_vasile_p21120(self):
        return c3s(a**2/(b+c)) - Rational(3,2), [a,b,c], [c3s(a**2)-c3s(a**3)]

    def problem_vasile_p21121_p1(self):
        return 1 - c3s(a/(b*c+2)), [a,b,c,1-a,1-b,1-c], []

    def problem_vasile_p21121_p2(self):
        return 1 - c3s(a*b/(b*c*2+1)), [a,b,c,1-a,1-b,1-c], []

    def problem_vasile_p21122(self):
        return 5*(1-c3s(a*b))*c3s(1/(1-a*b)) + 9, [a,b,c], [c3s(a)-2]

    def problem_vasile_p21123(self):
        return 3 - c3s((2-a**2)/(2-b*c)), [a,b,c], [c3s(a)-2]

    def problem_vasile_p21124(self):
        return c3s((3+5*a**2)/(3-b*c)) - 12, [a,b,c], [c3s(a)-3]

    def problem_vasile_p21125(self):
        return c3s((a**2+m)/(3-2*b*c)) - 3*(4+9*m)/19,\
            [a,b,c, m+Rational(1,7), Rational(7,8)-m], [c3s(a)-2]

    def problem_vasile_p21126(self):
        return c3s((47-7*a**2)/(1+b*c)) - 60, [a,b,c], [c3s(a)-3]

    def problem_vasile_p21127(self):
        return Rational(57,2) - c3s((26-7*a**2)/(1+b*c)), [a,b,c], [c3s(a)-3]

    def problem_vasile_p21128(self):
        return 3 - c3s((5*a*(b+c)-6*b*c)/(a**2+b**2+c**2+b*c)), [a,b,c], []

    def problem_vasile_p21129_p1(self):
        x_ = c3s(a**2)/c3s(a*b)
        return c3s(a/(b+c)) + Rational(1,2) - (x_ + 1/x_), [a,b,c], []

    def problem_vasile_p21129_p2(self):
        x_ = c3s(a**2)/c3s(a*b)
        return  6*c3s(a/(b+c)) - (5*x_ + 4/x_), [a,b,c], []

    def problem_vasile_p21129_p3(self):
        x_ = c3s(a**2)/c3s(a*b)
        return c3s(a/(b+c)) - Rational(3,2) - (x_ - 1/x_)/3, [a,b,c], []

    def problem_vasile_p21130(self):
        return 9/c3s(a)**2/5 - c3s(1/(a**2+7*b**2+7*c**2)), [], []

    def problem_vasile_p21131(self):
        return Rational(3,5) - c3s(b*c/(3*a**2+b**2+c**2)), [], []

    def problem_vasile_p21132_p1(self):
        return Rational(3,4) - c3s(1/(2+b**2+c**2)), [], [c3s(a)-3]

    def problem_vasile_p21132_p2(self):
        return Rational(1,6) - c3s(1/(8+5*(b**2+c**2))), [], [c3s(a)-3]

    def problem_vasile_p21133(self):
        return Rational(4,3) - c3s((a+b)*(a+c)/(a**2+4*(b**2+c**2))), [], []

    def problem_vasile_p21134(self):
        return 1/c3s(a*b)/2 - c3s(1/(b+c)/(7*a+b+c)), [a,b,c], []

    def problem_vasile_p21135(self):
        return 9/c3s(a*b)/10 - c3s(1/(b**2+c**2+4*a*(b+c))), [a,b,c], []

    def problem_vasile_p21136(self):
        return 9/c3s(a*b)/2 - c3s(1/(3-a*b)), [a,b,c], [c3s(a)-3]

    def problem_vasile_p21137(self):
        return Rational(3,8) - c3s(b*c/(a**2+a+6)), [a,b,c], [c3s(a)-3]

    def problem_vasile_p21138(self):
        return c3s(1/(8*a**2-2*b*c+21)) - Rational(1,9), [a,b,c], [c3s(a*b)-3]

    def problem_vasile_p21139_p1(self):
        return c3s((a**2+b*c)/(b**2+c**2)) - c3s(a)**2/c3s(a**2), [], []

    def problem_vasile_p21139_p2(self):
        return c3s((a**2+3*b*c)/(b**2+c**2)) - 6*c3s(a*b)/c3s(a**2), [], []

    def problem_vasile_p21140(self):
        return c3s(a*(b+c)/(b**2+c**2)) - Rational(3,10), [c3s(a*b)], []

    def problem_vasile_p21141(self):
        return 1/(c3s(a)-3) + 1/(c3p(a)-1) - 4/(c3s(a*b)-3), [a,b,c], [c3p(a)-1]

    def problem_vasile_p21142(self):
        return 27*c3p(a)/2 - c3s((4*b**2-a*c)*(4*c**2-a*b)/(b+c)), [a,b,c], []

    def problem_vasile_p21143(self):
        return c3s(a/(3*a+b*c)) - Rational(2,3), [a,b,c], [c3s(a)-3]

    def problem_vasile_p21144_p1(self):
        return c3s(a/(b+c)) - Rational(19,12), [a,b,c], [c3s(a)*c3s(1/a)-10]

    def problem_vasile_p21144_p2(self):
        return Rational(5,3) - c3s(a/(b+c)), [a,b,c], [c3s(a)*c3s(1/a)-10]

    def problem_vasile_p21145_p1(self):
        return c3s(a/(2*a+b*c)) - Rational(9,10), [a,b,c], [c3s(a)-3]

    def problem_vasile_p21145_p2(self):
        return 1 - c3s(a/(2*a+b*c)), [a,b,c], [c3s(a)-3]

    def problem_vasile_p21146(self):
        return c3s(a**3)/c3s(a**2) - c3s(a**3/(2*a**2+b*c)), [a,b,c], []

    def problem_vasile_p21147(self):
        return c3s(a**3/(4*a**2+b*c)) - c3s(a)/5, [a,b,c], []

    def problem_vasile_p21148(self):
        return c3s(1/(2+a)**2) - 3/(6+c3s(a*b)), [a,b,c], []

    def problem_vasile_p21149(self):
        return c3s(1/(1+3*a)) - 3/(3+c3p(a)), [a,b,c], []

    def problem_vasile_p21150(self):
        return c3p(k + 2*a*b/(a**2+b**2)) - (k-1)*(k**2-1), [k-1,3-k], []

    def problem_vasile_p21151(self):
        return c3s(1/a**2) +3*c3s(1/(a-b)**2) - 4*c3s(1/(a*b)), [], []

    @mark(mark.skip)
    def problem_vasile_p21152(self):
        A, B, C = a/b+b/a+k, b/c+c/b+k, c/a+a/c+k
        return 1/(k+2) + 4/(A+B+C-(k+2)) - (1/A+1/B+1/C), [a,b,c,k+2,4-k], []

    def problem_vasile_p21153(self):
        return c3s(1/(b**2+b*c+c**2)) - c3s(1/(2*a**2+b*c)), [a,b,c], []

    def problem_vasile_p21154(self):
        return c3s(1/(2*a*b+1)) - c3s(1/(a**2+2)), [a,b,c], [c3s(a)-3]

    def problem_vasile_p21155(self):
        return c3s(1/(a*b+2)) - c3s(1/(a**2+2)), [a,b,c], [c3s(a)-4]

    def problem_vasile_p21156_p1(self):
        return 1 - c3s(a*b)/c3s(a**2) - c3p(a-b)**2/c3p(a**2+b**2), [a,b,c], []

    def problem_vasile_p21156_p2(self):
        return 1 - c3s(a*b)/c3s(a**2) - c3p(a-b)**2/c3p(a**2-a*b+b**2), [a,b,c], []

    def problem_vasile_p21157(self):
        return c3s(1/(a**2+b**2)) - 45/c3s(8*a**2+2*a*b), [a,b,c], []

    def problem_vasile_p21158(self):
        return c3s((a**2-7*b*c)/(b**2+c**2)), [], []

    def problem_vasile_p21159(self):
        return c3s((b+c)**2/a**2) - 2 - 10*c3s(a)**2/3/c3s(a**2), [], []

    def problem_vasile_p21160(self):
        return c3s((a**2-4*b*c)/(b**2+c**2)) + 9*c3s(a*b)/c3s(a**2) - Rational(9,2), [a,b,c], []

    def problem_vasile_p21161(self):
        return c3s(a**2)/c3s(a*b) - 1 - 9*c3p(a-b)**2/c3p(a+b)**2, [a,b,c], []

    def problem_vasile_p21162(self):
        return c3s(a**2)/c3s(a*b) - 1 - (1+sqrt(2))**2*c3p(a-b)**2/c3p(a**2+b**2), [a,b,c], []

    def problem_vasile_p21163(self):
        return c3s(2/(a+b)) - c3s(5/(3*a+b+c)), [a,b,c], []

    def problem_vasile_p21164_p1(self):
        return c3s((8*a**2+3*b*c)/(b**2+b*c+c**2)) - 11, [a,b,c], []

    def problem_vasile_p21164_p2(self):
        return c3s((8*a**2-5*b*c)/(b**2-b*c+c**2)) - 9, [a,b,c], []

    def problem_vasile_p21165(self):
        return c3s((4*a**2+b*c)/(4*b**2+7*b*c+4*c**2)) - 1, [], []

    def problem_vasile_p21166(self):
        return c3s(1/(a-b)**2) - 27/c3s(a**2-a*b)/4, [], []

    def problem_vasile_p21167(self):
        return c3s(1/(a**2-a*b+b**2)) - 14/c3s(a**2)/3, [], []

    def problem_vasile_p21168_p1(self):
        return c3s(a/(b+c)) - Rational(3,2), [c3s(a*b)], []

    def problem_vasile_p21168_p2(self):
        return c3s(a/(b+c)) - 2, [c3s(a*b), -a*b], []

    def problem_vasile_p21169(self):
        return c3s(a/(7*a+b+c)) - c3s(a*b)/c3s(a)**2, [a,b,c], []

    def problem_vasile_p21170(self):
        return c3s(a**2/(4*a**2+5*b*c)) - Rational(1,3), [b+c-a,c+a-b,a+b-c], []

    def problem_vasile_p21171(self):
        return c3s(1/(7*a**2+b**2+c**2)) - 3/c3s(a)**2, [b+c-a,c+a-b,a+b-c], []

    @mark(mark.skip)
    def problem_vasile_p21172(self):
        return 3*(k+3)/(k+2) - c3s((a*(b+c)+(k+1)*b*c)/(b**2+k*b*c+c**2)), [b+c-a,c+a-b,a+b-c,k+2], []

    @mark(mark.skip)
    def problem_vasile_p21173(self):
        return 3*(4*k+11)/(k+2) - c3s((2*a**2+(4*k+9)*b*c)/(b**2+k*b*c+c**2)), [b+c-a,c+a-b,a+b-c,k+2]

    @mark(mark.skip)
    def problem_vasile_p21174(self):
        return c3s(1/(1+a)) - 3/(1+c3p(a)**Rational(1,3)), [a-b,b-c,c-d,d], [c4p(a)-1]

    def problem_vasile_p21175(self):
        return 1 - c4s(1/(1+a*b+b*c+c*a)), [a,b,c,d], [c4p(a)-1]

    def problem_vasile_p21176(self):
        return c4s(1/(1+a)**2) - 1, [a,b,c,d], [c4p(a)-1]

    @mark(mark.skip)
    def problem_vasile_p21177(self):
        return c4s(1/(3*a-1)**2) - 1, [a,b,c,d], [c4p(a)-1]

    @mark(mark.skip)
    def problem_vasile_p21178(self):
        return c4s(1/(1+a+a**2+a**3)) - 1, [a,b,c,d], [c4p(a)-1]

    def problem_vasile_p21179(self):
        return c4s(1/(1+a+2*a**2)) - 1, [a,b,c,d], [c4p(a)-1]

    def problem_vasile_p21180(self):
        return c4s(1/a) + 9/c4s(a) - Rational(25,4), [a,b,c,d], [c4p(a)-1]

    def problem_vasile_p21181(self):
        return 4 - c4s((a-1)**2/(3*a**2+1)), [], [c4s(a)]

    def problem_vasile_p21182(self):
        return c4s((1-a)/(1+a)**2), [a+5,b+5,c+5,d+5], [c4s(a)-4]

    @mark(mark.noimpl)
    def problem_vasile_p21183(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p21184(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p21185(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p21186(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p21187(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p21188(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p21189(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p21190(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p21191(self):
        ...

    def problem_vasile_p22001(self):
        return sqrt(c3s(6*a**2-3*a*b)) - c3s(sqrt(a**2-a*b+b**2)), [a,b,c], []

    def problem_vasile_p22002(self):
        return 3*sqrt(c3s(a**2)/2) - c3s(sqrt(a**2-a*b+b**2)), [a,b,c], []

    def problem_vasile_p22003(self):
        return c3s(sqrt(a**2+b**2-2*a*b/3)) - 2*sqrt(c3s(a**2)), [a,b,c], []

    def problem_vasile_p22004(self):
        return c3s(sqrt(a**2+a*b+b**2)) - sqrt(c3s(4*a**2+5*a*b)), [a,b,c], []

    def problem_vasile_p22005(self):
        return sqrt(c3s(5*a**2+4*a*b)) - c3s(sqrt(a**2+a*b+b**2)), [a,b,c], []

    def problem_vasile_p22006(self):
        return 2*sqrt(c3s(a**2))+sqrt(c3s(a*b)) - c3s(sqrt(a**2+a*b+b**2)), [a,b,c], []

    def problem_vasile_p22007(self):
        return sqrt(c3s(a**2))+2*sqrt(c3s(a*b))-c3s(sqrt(a**2+2*b*c)), [a,b,c], []

    def problem_vasile_p22008(self):
        return c3s(1/sqrt(a**2+2*b*c)) - 1/sqrt(c3s(a**2)) - 2/sqrt(c3s(a*b)), [a,b,c], []

    def problem_vasile_p22009(self):
        return 2*sqrt(c3s(a**2))+sqrt(c3s(a*b))-c3s(sqrt(2*a**2+b*c)), [a,b,c], []

    def problem_vasile_p22010(self):
        k_ = sqrt(3)-1
        return 3*sqrt(3) - c3s(sqrt(a*(a+k_*b)*(a+k_*c))), [a,b,c], [c3s(a)-3]

    def problem_vasile_p22011(self):
        return c3s(sqrt(a*(2*a+b)*(2*a+c))) - 9, [a,b,c], [c3s(a)-3]

    def problem_vasile_p22012(self):
        return c3s(sqrt(b**2+c**2+a*(b+c))) - 6, [a,b,c], [c3s(a)-3]

    def problem_vasile_p22013_p1(self):
        return c3s(sqrt(a*(3*a**2+a*b*c))) - 6, [a,b,c], [c3s(a)-3]

    def problem_vasile_p22013_p2(self):
        return c3s(sqrt(3*a**2+a*b*c)) - 3*sqrt(3+c3p(a)), [a,b,c], [c3s(a)-3]

    def problem_vasile_p22014(self):
        return c3s(a*sqrt((a+2*b)*(a+2*c))) - 9, [a,b,c], [c3s(a*b)-3]

    def problem_vasile_p22015(self):
        return c3s(sqrt(a+(b-c)**2)) - sqrt(3), [a,b,c], [c3s(a)-1]

    def problem_vasile_p22016(self):
        return c3s(sqrt(a*(b+c)/(a**2+b*c))) - 2, [a,b,c], []

    @mark(mark.skip)
    def problem_vasile_p22017(self):
        return c3s(1/(a**2+25*a+1)**Rational(1,3)) - 1, [a,b,c], [c3p(a)-1]

    def problem_vasile_p22018(self):
        return 3*c3s(a)/2 - c3s(sqrt(a**2+b*c)), [a,b,c], []

    def problem_vasile_p22019(self):
        return c3s(sqrt(a**2+9*b*c)) - 5*sqrt(c3s(a*b)), [a,b,c], []

    def problem_vasile_p22020(self):
        return c3s(sqrt((a**2+4*b*c)*(b**2+4*c*a))) - 5*c3s(a*b), [a,b,c], []

    def problem_vasile_p22021(self):
        return c3s(sqrt((a**2+9*b*c)*(b**2+9*c*a))) - 7*c3s(a*b), [a,b,c], []

    def problem_vasile_p22022(self):
        return c3s(a)**2 - c3s(sqrt((a**2+b**2)*(b**2+c**2))), [a,b,c], []

    def problem_vasile_p22023(self):
        return c3s(sqrt((a**2+a*b+b**2)*(b**2+b*c+c**2))) - c3s(a)**2, [a,b,c], []

    def problem_vasile_p22024(self):
        return c3s(sqrt((a**2+7*a*b+b**2)*(b**2+7*b*c+c**2))) - 7*c3s(a*b), [a,b,c], []

    def problem_vasile_p22025(self):
        return 13*c3s(a)**2/12 - c3s(sqrt((a**2+7*a*b/9+b**2)*(b**2+7*b*c/9+c**2))), [a,b,c], []

    def problem_vasile_p22026(self):
        return 61*c3s(a)**2/60 - c3s(sqrt((a**2+a*b/3+b**2)*(b**2+b*c/3+c**2))), [a,b,c], []

    def problem_vasile_p22027(self):
        return c3s(a/sqrt(4*b**2+b*c+4*c**2)) - 1, [a,b,c], []

    def problem_vasile_p22028(self):
        return c3s(a/sqrt(b**2+b*c+c**2)) - c3s(a)/sqrt(c3s(a*b)), [a,b,c], []

    def problem_vasile_p22029(self):
        return c3s(a)/sqrt(c3s(a*b)) - c3s(a/sqrt(a**2+2*b*c)), [a,b,c], []

    def problem_vasile_p22030(self):
        return c3s(a**3)+3*c3p(a) - c3s(a**2*sqrt(a**2+3*b*c)), [a,b,c], []

    def problem_vasile_p22031(self):
        return 1 - c3s(a/sqrt(4*a**2+5*b*c)), [a,b,c], []

    def problem_vasile_p22032(self):
        return c3s(a*sqrt(4*a**2+5*b*c)) - c3s(a)**2, [a,b,c], []

    def problem_vasile_p22033(self):
        return c3s(a*sqrt(a**2+3*b*c)) - 2*c3s(a*b), [a,b,c], []

    def problem_vasile_p22034(self):
        return c3s(a)**2 - c3s(a*sqrt(a**2+8*b*c)), [a,b,c], []

    def problem_vasile_p22035(self):
        return c3s((a**2+2*b*c)/sqrt(b**2+b*c+c**2)) - 3*sqrt(c3s(a*b)), [a,b,c], []

    @mark(mark.noimpl)
    def problem_vasile_p22036(self):
        return c3s(a**k)/c3s(a) - c3s(a**(k+1)/(2*a**2+b*c)), [a,b,c,k-1], []

    def problem_vasile_p22037_p1(self):
        return c3s((a**2-b*c)/sqrt(3*a**2+2*b*c)), [a,b,c], []

    def problem_vasile_p22037_p2(self):
        return c3s((a**2-b*c)/sqrt(8*a**2+(b+c)**2)), [a,b,c], []

    @mark(mark.skip)
    def problem_vasile_p22038(self):
        return c3s((a**2-b*c)/sqrt(k*a**2+b**2+c**2)), [a,b,c,k,2*sqrt(2)+1-k], []

    def problem_vasile_p22039(self):
        return c3s((a**2-b*c)*sqrt(b+c)), [a,b,c],[ ]

    def problem_vasile_p22040(self):
        return c3s((a**2-b*c)*sqrt(a**2+4*b*c)), [a,b,c], []

    def problem_vasile_p22041(self):
        return c3s(a**3/(a**3+(b+c)**3)) - 1, [a,b,c], []

    def problem_vasile_p22042(self):
        return sqrt(c3s(a)*c3s(1/a)) - 1 - sqrt(1+sqrt(c3s(a**2)*c3s(1/a**2))), [a,b,c], []

    def problem_vasile_p22043(self):
        return 5 + sqrt(2*c3s(a**2)*c3s(1/a**2)-2) - c3s(a)*c3s(1/a), [a,b,c], []

    def problem_vasile_p22044(self):
        return 2*(1+c3p(a)) + sqrt(2*c3p(1+a**2)) - c3p(1+a), [], []

    def problem_vasile_p22045(self):
        return c3s(sqrt((a**2+b*c)/(b**2+c**2))) - 2 - 1/sqrt(2), [a,b,c], []

    def problem_vasile_p22046(self):
        return c3s(sqrt(a*(2*a+b+c))) - sqrt(12*c3s(a*b)), [a,b,c], []

    def problem_vasile_p22047(self):
        return c3s(a*sqrt((4*a+5*b)*(4*a+5*c))) - 27, [a,b,c], [c3s(a)-3]

    def problem_vasile_p22048(self):
        return c3s(a*sqrt((a+3*b)*(a+3*c))) - 12, [a,b,c], [c3s(a*b)-12]

    def problem_vasile_p22049(self):
        return c3s(sqrt(2+7*a*b)) - 3*sqrt(3*c3s(a*b)), [a,b,c], [c3s(a**2)-3]

    def problem_vasile_p22050_p1(self):
        return c3s(sqrt(a*(b+c)*(a**2+b*c))) - 6, [a,b,c], [c3s(a*b)-3]

    def problem_vasile_p22050_p2(self):
        return c3s(a*(b+c)*sqrt(a**2+2*b*c)) - 6*sqrt(3), [a,b,c], [c3s(a*b)-3]

    def problem_vasile_p22050_p3(self):
        return c3s(a*(b+c)*sqrt((a+2*b)*(a+2*c))) - 18, [a,b,c], [c3s(a*b)-3]

    def problem_vasile_p22051(self):
        return c3s(a*sqrt(b*c+3)) - 6, [a,b,c], [c3s(a*b)-3]

    def problem_vasile_p22052_p1(self):
        return c3s((b+c)*sqrt(b**2+c**2+7*b*c)) - 18, [a,b,c], [c3s(a)-3]

    def problem_vasile_p22052_p1(self):
        return 12*sqrt(3) - c3s((b+c)*sqrt(b**2+c**2+10*b*c)), [a,b,c], [c3s(a)-3]

    def problem_vasile_p22053(self):
        return c3s(sqrt(a+4*b*c)) - 4*sqrt(c3s(a*b)), [a,b,c], [c3s(a)-2]

    def problem_vasile_p22054(self):
        return c3s(sqrt(a**2+b**2+7*a*b)) - 5*sqrt(c3s(a*b)), [a,b,c], []

    def problem_vasile_p22055(self):
        return c3s(sqrt(a**2+b**2+5*a*b)) - sqrt(21*c3s(a*b)), [a,b,c], []

    def problem_vasile_p22056(self):
        return c3s(a*sqrt(a**2+5)) - sqrt(Rational(2,3))*c3s(a)**2, [a,b,c], [c3s(a*b)-3]

    def problem_vasile_p22057(self):
        return c3s(a*sqrt(2+3*b*c)) - c3s(a)**2, [a,b,c], [c3s(a**2)-1]

    def problem_vasile_p22058_p1(self):
        return c3s(a*sqrt((2*a+b*c)/3)) - 3, [a,b,c], [c3s(a)-3]

    def problem_vasile_p22058_p2(self):
        return c3s(a*sqrt(a*(1+b+c)/3)) - 3, [a,b,c], [c3s(a)-3]

    def problem_vasile_p22059(self):
        return c3s(sqrt(8*(a**2+b*c)+9)) - 15, [a,b,c], [c3s(a)-3]

    def problem_vasile_p22060(self):
        return c3s(sqrt(a**2+b*c+k)) - 3*sqrt(2+k), [a,b,c,k-Rational(9,8)], [c3s(a)-3]

    def problem_vasile_p22061(self):
        return c3s(sqrt(a**3+2*b*c)) - 3*sqrt(3), [a,b,c], [c3s(a)-3]

    def problem_vasile_p22062(self):
        return c3s(sqrt(a**2+b*c)/(b+c)) - 3*sqrt(2)/2, [a,b,c], []

    def problem_vasile_p22063(self):
        return c3s(sqrt(b*c+4*a*(b+c))/(b+c)) - Rational(9,2), [a,b,c], []

    def problem_vasile_p22064(self):
        return c3s(a*sqrt(a**2+3*b*c)/(b+c)) - c3s(a), [a,b,c], []

    def problem_vasile_p22065(self):
        return c3s(sqrt(2*a*(b+c)/((2*b+c)*(b+2*c)))) - 2, [a,b,c], []

    def problem_vasile_p22066_p1(self):
        return 1 - c3s(sqrt(b*c/(3*a**2+6))), [a,b,c], [c3s(a*b)-3]

    def problem_vasile_p22066_p2(self):
        return c3s(sqrt(b*c/(6*a**2+3))) - 1, [a,b,c], [c3s(a*b)-3]

    @mark(mark.noimpl)
    def problem_vasile_p22067(self):
        return c3s(a**k*(b+c)) - 6, [a,b,c,k-1], [c3s(a*b)-3]

    @mark(mark.noimpl)
    def problem_vasile_p22068(self):
        return 2 - c3s(a**k*(b+c)), [a,b,c,k-2,3-k], [c3s(a)-2]

    @mark(mark.noimpl)
    def problem_vasile_p22069(self):
        return c3s((b**m+c**m)/(b**n+c**n)*(b+c-2*a)), [a,b,c,m-n,n], []

    def problem_vasile_p22070(self):
        return c3s(sqrt(a**2-a+1)) - c3s(a), [a,b,c], [c3p(a)-1]

    def problem_vasile_p22071(self):
        return c3s(sqrt(16*a**2+9)) - 4*c3s(a) - 3, [a,b,c], [c3p(a)-1]

    def problem_vasile_p22072(self):
        return 5*c3s(a) + 24 - c3s(sqrt(25*a**2+144)), [a,b,c], [c3p(a)-1]

    def problem_vasile_p22073_p1(self):
        return c3s(sqrt(a**2+3)) - c3s(a) - 3, [a,b,c], [c3s(a*b)-3]

    def problem_vasile_p22073_p2(self):
        return c3s(sqrt(a+b)) - sqrt(4*c3s(a)+6), [a,b,c], [c3s(a*b)-3]

    def problem_vasile_p22074(self):
        return c3s(sqrt((5*a**2+3)*(5*b**2+3))) - 24, [a,b,c], [c3s(a)-3]

    def problem_vasile_p22075(self):
        return c3s(sqrt(a**2+1)) - sqrt((4*c3s(a**2)+42)/3), [a,b,c], [c3s(a)-3]

    def problem_vasile_p22076_p1(self):
        return c3s(sqrt(a**2+3)) - sqrt(2*c3s(a**2)+30), [a,b,c], [c3s(a)-3]

    def problem_vasile_p22076_p2(self):
        return c3s(sqrt(3*a**2+1)) - sqrt(2*c3s(a**2)+30), [a,b,c], [c3s(a)-3]

    def problem_vasile_p22077(self):
        return 105 - c3s(sqrt(32*a**2+3)*(32*b**2+3)), [a,b,c], [c3s(a)-3]

    def problem_vasile_p22078(self):
        return c3s(Abs((b+c)/a - 3)) - 2, [a,b,c], []

    def problem_vasile_p22079(self):
        return c3s(Abs((b+c)/a)) - 2, [], []

    def problem_vasile_p22080_p1(self):
        x_, y_, z_ = 2*a/(b+c), 2*b/(c+a), 2*c/(a+b)
        return x_+y_+z_+sqrt(x_*y_)+sqrt(y_*z_)+sqrt(z_*x_)-6, [a,b,c], []

    def problem_vasile_p22080_p2(self):
        x_, y_, z_ = 2*a/(b+c), 2*b/(c+a), 2*c/(a+b)
        return sqrt(x_)+sqrt(y_)+sqrt(z_)-sqrt(8+x_*y_*z_), [a,b,c], []

    def problem_vasile_p22081(self):
        return c3s(sqrt(1+24*2*a/(b+c))) - 15, [a,b,c], []

    def problem_vasile_p22082(self):
        return 3 - c3s(sqrt(7*a/(a+3*b+3*c))), [a,b,c], []

    def problem_vasile_p22083(self):
        return 3*2**Rational(1,3) - c3s((a**2*(b**2+c**2))**Rational(1,3)), [a,b,c], [c3s(a)-3]

    def problem_vasile_p22084(self):
        return c3s(1/(a+b)) - 1/c3s(a) - 2/sqrt(c3s(a*b)), [a,b,c], []

    def problem_vasile_p22085(self):
        return 1/sqrt(3*a*b+1) + Rational(1,2) - 1/sqrt(3*a+1) - 1/sqrt(3*b+1), [a-1,b-1], []

    def problem_vasile_p22086(self):
        return c3s(1/sqrt(3*a+1)) - Rational(3,2), [a-1,1-b,b-c,c], [c3p(a)-1]

    @mark(mark.noimpl)
    def problem_vasile_p22087(self):
        return 3 - c3p(a)**k*c3s(a**2), [a,b,c,k-1/sqrt(2)], [c3s(a)-3]

    @mark(mark.skip)
    def problem_vasile_p22088_p1(self):
        p_ = c3s(a)
        q_ = c3s(a*b)
        w_ = sqrt(p_**2-3*q_)
        g_ = sqrt((2*p_-2*w_)/3) + 2*sqrt((2*p_+w_)/3)
        return c3s(sqrt(a+b)) - g_, [a,b,c], []

    @mark(mark.skip)
    def problem_vasile_p22088_p2(self):
        p_ = c3s(a)
        q_ = c3s(a*b)
        w_ = sqrt(p_**2-3*q_)
        h_ = sqrt((2*p_+2*w_)/3) + 2*sqrt((2*p_-w_)/3)
        return h_ - c3s(sqrt(a+b)), [a,b,c,4*q_-p_**2], []

    @mark(mark.skip)
    def problem_vasile_p22088_p3(self):
        p_ = c3s(a)
        q_ = c3s(a*b)
        w_ = sqrt(p_**2-3*q_)
        h_ = sqrt(p_)+sqrt(p_+sqrt(q_))
        return h_ - c3s(sqrt(a+b)), [a,b,c,p_**2-4*q_], []

    @mark(mark.skip)
    def problem_vasile_p22089(self):
        return c4s(sqrt(1-a)) - c4s(sqrt(a)), [a,b,c,d], [c4s(a**2)-1]

    def problem_vasile_p22090(self):
        A = c4s(a)*c4s(1/a)-16
        B = c4s(a**2)*c4s(1/a**2)-16
        return A + 2 - sqrt(B+4), [a,b,c,d], []

    @mark(mark.noimpl)
    def problem_vasile_p22091(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p22092(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p22093(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p22094(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p22095(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p22096(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p22097(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p22098(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p22099(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p22100(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p22101(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p22102(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p22103(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p22104(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p22105(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p22106(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p22107(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p22108(self):
        ...



class MathematicalInequalitiesVol3(MathematicalInequalities):
    """Cyclic and Noncyclic Inequalities"""
    def problem_vasile_p31001(self):
        return c3s(1/(a*(a+2*b))) - 3/c3s(a*b), [a,b,c], []

    def problem_vasile_p31002(self):
        return 4-c3s(a*b**2), [a,b,c], [c3s(a)-3]

    def problem_vasile_p31003_q1(self):
        return c3s(a*b**2)+6-3*c3s(a), [a-1,b-1,c-1], []

    def problem_vasile_p31003_q2(self):
        return 2*c3s(a*b**2)+3-3*c3s(a*b), [a-1,b-1,c-1], []

    def problem_vasile_p31004(self):
        return c3s(a/(b**2+2*c))-1, [a,b,c], [c3s(a)-3]

    def problem_vasile_p31005(self):
        return c3s((a-1)/(b+1)), [a,b,c,c3s(a)-3], []

    def problem_vasile_p31006(self):
        return c3s(1/(2*a*b**2+1))-1, [a,b,c], [c3s(a)-3]

    def problem_vasile_p31007(self):
        return Rational(3,5)-c3s(a*b/(9-4*b*c)), [a,b,c], [c3s(a)-3]

    def problem_vasile_p31008_q1(self):
        return c3s(a**2/(2*a+b**2))-1, [a,b,c], [c3s(a)-3]

    def problem_vasile_p31008_q2(self):
        return c3s(a**2/(a+2*b**2))-1, [a,b,c], [c3s(a)-3]

    def problem_vasile_p31009(self):
        return 1-c3s(1/(a+b**2+c**3)), [a,b,c], [c3s(a)-3]

    def problem_vasile_p31010(self):
        return c3s((1+a**2)/(1+b+c**2))-2, [a,b,c], []

    def problem_vasile_p31011(self):
        return Rational(1,3) - c3s(a/(4*a+4*b+c)), [a,b,c], []

    def problem_vasile_p31012(self):
        return c3s((a+b)/(a+7*b+c)) - Rational(2,3), [a,b,c], []

    def problem_vasile_p31013(self):
        return c3s((2*a+b)/(2*a+c)) - 3, [a,b,c], []

    def problem_vasile_p31014(self):
        return c3s((5*a+b)/(a+c)) - 9, [a,b,c], []

    def problem_vasile_p31015(self):
        return 3*c3s(a**2)/c3s(a) - c3s(a*(a+b)/(a+c)), [a,b,c], []

    def problem_vasile_p31016(self):
        return c3s((a**2-b*c)/(4*a**2+b**2+4*c**2)), [], []

    def problem_vasile_p31017_p1(self):
        return c3s(a*(a+b)**3), [], []

    def problem_vasile_p31017_p1(self):
        return c3s(a*(a+b)**5), [], []
        
    def problem_vasile_p31018(self):
        return 3*c3s(a**4)+4*c3s(a**3*b), [], []

    def problem_vasile_p31019(self):
        return c3s((a-b)*(3*a+b)/(a**2+b**2)), [a,b,c], []

    def problem_vasile_p31020(self):
        return 1 - c3s(1/(1+a+b**2)), [a,b,c], [a*b*c-1]

    def problem_vasile_p31021(self):
        return c3s(a/(a+1)/(b+2)) - Rational(1,2), [a,b,c], [a*b*c-1]

    def problem_vasile_p31022(self):
        return c3p(a+2*b)-27, [a,b,c], [c3s(a*b)-3]

    def problem_vasile_p31023(self):
        return 1 - c3s(a/(a+a**3+b)), [a,b,c], [c3s(a*b)-3]

    def problem_vasile_p31024(self):
        return c3s(1/(a+2*b))-1, [a,b,c,a-b,b-c], [c3s(a*b)-3]

    def problem_vasile_p31025(self):
        return c3s(a/(4*b**2+5)) - Rational(1,3), [a,b,c,1-a,1-b,1-c], []

    def problem_vasile_p31026(self):
        return c3s(a/(a+b)) - Rational(7,5), [3*a-1,3*b-1,3*c-1,3-a,3-b,3-c], []

    def problem_vasile_p31027(self):
        return c3s(3/(a+2*b)-2/(a+b)),\
            [sqrt(2)*a-1,sqrt(2)*b-1,sqrt(2)*c-1,a-sqrt(2),b-sqrt(2),c-sqrt(2)], []

    def problem_vasile_p31028_p1(self):
        return 4 - c3s(a*b**2) - c3p(a), [a,b,c], [c3s(a)-3]

    def problem_vasile_p31028_p2(self):
        return 1 - c3s(a/(4-b)), [a,b,c], [c3s(a)-3]

    def problem_vasile_p31028_p3(self):
        return 12 - c3s(a*b**3) - c3s(a*b)**2, [a,b,c], [c3s(a)-3]

    def problem_vasile_p31028_p4(self):
        return 1 - c3s(a*b**2/(1+a+b)), [a,b,c], [c3s(a)-3]

    def problem_vasile_p31029(self):
        return 4*c3p(a)/(c3s(a*b**2)+c3p(a)) + c3s(a**2)/c3s(a*b) - 2, [a,b,c], []

    def problem_vasile_p31030(self):
        return c3s(1/(a*b**2+8)) - Rational(1,3), [a,b,c], [c3s(a)-3]

    def problem_vasile_p31031(self):
        return Rational(3,4) - c3s(a*b/(b*c+3)), [a,b,c], [c3s(a)-3]

    def problem_vasile_p31032(self):
        return 9 - c3s(a*b)*c3s(a*b**2), [a,b,c], [c3s(a)-3]

    def problem_vasile_p31033_p1(self):
        return 2 + c3p(a) - c3s(a*b**2), [a,b,c], [c3s(a**2)-3]

    def problem_vasile_p31033_p2(self):
        return 1 - c3s(a/(b+2)), [a,b,c], [c3s(a**2) - 3]

    def problem_vasile_p31034(self):
        return 3 - c3s(a**2*b**3), [a,b,c], [c3s(a**2)-3]

    def problem_vasile_p31035(self):
        return c3s(a**4*b**2) + 4 - c3s(a**3*b**3), [a,b,c], [c3s(a)-3]

    def problem_vasile_p31036_p1(self):
        return c3s(a/(b**2+3)) - Rational(3,4), [a,b,c], [c3s(a)-3]

    def problem_vasile_p31036_p2(self):
        return c3s(a/(b**2+1)) - Rational(3,2), [a,b,c], [c3s(a)-3]

    def problem_vasile_p31037(self):
        x_ = a+1/b-1
        y_ = b+1/c-1
        z_ = c+1/a-1
        return x_*y_+y_*z_+z_*x_ - 3, [a,b,c], []

    def problem_vasile_p31038(self):
        return c3s((a-1/b-sqrt(2))**2) - 6, [a,b,c], [c3p(a)-1]

    def problem_vasile_p31039(self):
        return c3s(Abs(1+a-1/b))-2, [a,b,c], [c3p(a)-1]

    def problem_vasile_p31040(self):
        return c3s(Abs(1+a/(b-c)))-2, [a,b,c], []

    def problem_vasile_p31041(self):
        return c3s((2*a-1/b-Rational(1,2))**2) - Rational(3,4), [a,b,c], [c3p(a)-1]

    def problem_vasile_p31042(self):
        x_ = a+1/b-Rational(5,4)
        y_ = b+1/c-Rational(5,4)
        z_ = c+1/a-Rational(5,4)
        return x_*y_+y_*z_+z_*x_ - Rational(27,16), [a-b,b-c,c], []

    def problem_vasile_p31043(self):
        E = c3p(a+1/a-sqrt(3))
        F = c3p(a+1/b-sqrt(3))
        return E-F, [a,b,c], []

    def problem_vasile_p31044(self):
        return c3s(b/a) - Rational(17,4), [a,b,c], [c3s(a/b)-5]

    def problem_vasile_p31045_p1(self):
        return 1 + c3s(a/b) - 2*sqrt(1+c3s(b/a)), [a,b,c], []

    def problem_vasile_p31045_p2(self):
        return 1 + 2*c3s(a/b) - sqrt(1+16*c3s(b/a)), [a,b,c], []

    def problem_vasile_p31046(self):
        return c3s(a**2/b**2+15*b/a-16*a/b), [a,b,c], []

    def problem_vasile_p31047_p1(self):
        return c3s(a/b-a), [a,b,c], [c3p(a)-1]

    def problem_vasile_p31047_p2(self):
        return c3s(a/b) - Rational(3,2)*(c3s(a)-1), [a,b,c], [c3p(a)-1]

    def problem_vasile_p31047_p3(self):
        return c3s(a/b) + 2 - Rational(5,3)*c3s(a), [a,b,c], [c3p(a)-1]

    def problem_vasile_p31048_p1(self):
        return c3s(a/b) - 2 - 3/c3s(a*b), [a,b,c], [c3s(a**2)-3]

    def problem_vasile_p31048_p2(self):
        return c3s(a/b) - 9/c3s(a), [a,b,c], [c3s(a**2)-3]

    def problem_vasile_p31049(self):
        return 6*c3s(a/b)+5*c3s(a*b)-33, [a,b,c], [c3s(a**2)-3]

    def problem_vasile_p31050_p1(self):
        return 6*c3s(a/b)+3-7*c3s(a**2), [a,b,c], [c3s(a)-3]

    def problem_vasile_p31050_p2(self):
        return c3s(a/b-a**2), [a,b,c], [c3s(a)-3]

    def problem_vasile_p31051(self):
        return c3s(a/b)+2-14*c3s(a**2)/c3s(a)**2, [a,b,c], []

    def problem_vasile_p31052(self):
        x_ = 3*a+1/b
        y_ = 3*b+1/c
        z_ = 3*c+1/a
        return x_*y+y_*z_+z_*x_-48, [a,b,c], [c3s(a)-3]

    def problem_vasile_p31053(self):
        return c3s((a+1)/b) - 2*c3s(a**2), [a,b,c], [c3s(a)-3]

    def problem_vasile_p31054(self):
        return c3s(a**2/b)+3-2*c3s(a**2), [a,b,c], [c3s(a)-3]

    def problem_vasile_p31055(self):
        return c3s(a**3/b)+2*c3s(a*b)-3*c3s(a**2), [a,b,c], []

    def problem_vasile_p31056_p1(self):
        return c3s(a**2/b)-3, [a,b,c], [c3s(a**4)-3]

    def problem_vasile_p31056_p2(self):
        return c3s(a**2/(b+c))-Rational(3,2), [a,b,c], [c3s(a**4)-3]

    def problem_vasile_p31057(self):
        return c3s(a**2/b) - 3*c3s(a**3)/c3s(a**2), [a,b,c], []

    def problem_vasile_p31058(self):
        return c3s(a**2/b)+c3s(a) - 2*sqrt(c3s(a**2)*c3s(a/b)), [a,b,c], []

    def problem_vasile_p31059(self):
        return c3s(a/b)+32*c3s(a/(a+b))-51, [a,b,c], []

    def problem_vasile_p31060_p1(self):
        K = 1
        return c3s(a/b)-3 - K*(c3s(a/(b+c))-Rational(3,2)), [a,b,c], []

    def problem_vasile_p31060_p2(self):
        K = 27
        return c3s(a/b)-3 + K*(c3s(a/(2*a+b)) - 1), [a,b,c], []

    def problem_vasile_p31061(self):
        return 8*c3s(a/b) - 5*c3s(b/a) - 9, [2*a-1,2*b-1,2*c-1,2-a,2-b,2-c], []

    def problem_vasile_p31062(self):
        return c3s(a/b) - c3s(2*a/(b+c)), [a,b,c,c-b,b-a], []

    def problem_vasile_p31063_p1(self):
        return c3s(a/b) - c3s(a**Rational(3,2)), [a,b,c,c-b,b-a], [a*b*c-1]

    @mark(mark.noimpl)
    def problem_vasile_p31063_p2(self):
        return c3s(a/b) - c3s(a**sqrt(3)), [c-b,b-1,1-a,a], [a*b*c-1]

    def problem_vasile_p31064(self):
        return c3s(1/((k+1)*a+b)) - c3s(1/(k*a+b+c)), [k,a,b,c], []

    def problem_vasile_p31065_p1(self):
        return sqrt(c3s(a)) - c3s(a/sqrt(2*a+b)), [a,b,c], []

    def problem_vasile_p31065_p2(self):
        return c3s(a/sqrt(a+2*b)) - sqrt(c3s(a)), [a,b,c], []

    def problem_vasile_p31066(self):
        return 3 - c3s(a*sqrt((a+2*b)/3)), [a,b,c], [c3s(a)-3]

    def problem_vasile_p31067(self):
        return 5 - c3s(a*sqrt(1+b**3)), [a,b,c], [c3s(a)-3]

    def problem_vasile_p31068_p1(self):
        return c3s(sqrt(a/(b+3))) - Rational(3,2), [a,b,c], [c3p(a)-1]

    def problem_vasile_p31068_p2(self):
        return c3s(cbrt(a/(b+7))) - Rational(3,2), [a,b,c], [c3p(a)-1]

    def problem_vasile_p31069(self):
        return c3s((1+4*a/(a+b))**2) - 27, [a,b,c], []

    def problem_vasile_p31070(self):
        return 3 - c3s(sqrt(2*a/(a+b))), [a,b,c], []

    def problem_vasile_p31071(self):
        return 1 - c3s(a/(4*a+5*b)), [a,b,c], []

    def problem_vasile_p31072(self):
        return 1 - c3s(a/sqrt(4*a**2+a*b+4*b**2)), [a,b,c], []

    def problem_vasile_p31073_p1(self):
        return c3s(sqrt(a/(3*b+c))) - Rational(3,2), [a,b,c], []

    def problem_vasile_p31073_p2(self):
        return c3s(sqrt(a/(2*b+c))) - 8**Rational(1,4), [a,b,c], []

    def problem_vasile_p31074(self):
        return c3s(sqrt(a/(a+b+7*c))) - 1, [a,b,c], []

    def problem_vasile_p31075_p1(self):
        return c3s(1/(a+b)/(3*a+b)) - Rational(3,8), [a,b,c], [c3s(a*b)-3]

    def problem_vasile_p31075_p2(self):
        return c3s(1/(2*a+b)**2) - Rational(1,3), [a,b,c], [c3s(a*b)-3]

    def problem_vasile_p31076(self):
        return c3s(a**4)+15*c3s(a**3*b)-Rational(47,4)*c3s(a**2*b**2), [a,b,c], []

    def problem_vasile_p31077(self):
        return 27 - c3s(a**3*b), [a,b,c], [c3s(a)-4]

    def problem_vasile_p31078(self):
        return c3s(a**4) - Rational(82,27)*c3s(a**3*b),\
            [a,b,c], [c3s(a**2)-Rational(10,3)*c3s(a*b)]

    def problem_vasile_p31079(self):
        return c3s(a**3/(2*a**2+b**2)) - c3s(a)/3, [a,b,c], []

    def problem_vasile_p31080(self):
        return c3s(a**4/(a**3+b**3)) - c3s(a)/3, [a,b,c], []

    def problem_vasile_p31081_p1(self):
        return 3*c3s(a**2/b)+4*c3s(b/a**2)-7*c3s(a**2), [a,b,c], [c3p(a)-1]

    def problem_vasile_p31081_p2(self):
        return 8*c3s(a**3/b)+5*c3s(b/a**3)-13*c3s(a**3), [a,b,c], [c3p(a)-1]

    def problem_vasile_p31082(self):
        return c3s(a**2)/c3s(a*b) - c3s(a*b/(b**2+b*c+c**2)), [a,b,c], []

    def problem_vasile_p31083(self):
        return c3s((a-b)/(b*(2*b+c))), [a,b,c], []

    def problem_vasile_p31084_p1(self):
        return c3s((a**2+6*b*c)/(a*b+2*b*c))-7, [a,b,c], []

    def problem_vasile_p31084_p2(self):
        return c3s((a**2+7*b*c)/(a*b+b*c))-12, [a,b,c], []

    def problem_vasile_p31085_p1(self):
        return c3s(a**2)/c3s(a) - c3s(a*b/(2*b+c)), [a,b,c], []

    def problem_vasile_p31085_p2(self):
        return 3*c3s(a**2)/(2*c3s(a)) - c3s(a*b/(b+c)), [a,b,c], []
  
    def problem_vasile_p31085_p1(self):
        return c3s(a**2)/(3*c3s(a)) - c3s(a*b/(4*b+5*c)), [a,b,c], []

    def problem_vasile_p31086_p1(self):
        return c3s(a)**2 - c3s(a*sqrt(b**2+8*c**2)), [a,b,c], []

    def problem_vasile_p31086_p2(self):
        return c3s(a**2+a*b) - c3s(a*sqrt(b**2+3*c**2)), [a,b,c], []

    def problem_vasile_p31087_p1(self):
        return c3s(1/a/sqrt(a+2*b)) - sqrt(3/c3p(a)), [a,b,c], []

    def problem_vasile_p31087_p2(self):
        return c3s(1/a/sqrt(a+8*b)) - sqrt(1/c3p(a)), [a,b,c], []

    def problem_vasile_p31088(self):
        return sqrt(c3s(a)/3) - c3s(a/sqrt(5*a+4*b)), [a,b,c], []

    def problem_vasile_p31089_p1(self):
        return c3s(a/sqrt(a+b)) - c3s(sqrt(a)/sqrt(2)), [a,b,c], []

    @mark(mark.skip)
    def problem_vasile_p31089_p2(self):
        return c3s(a/sqrt(a+b)) - (27*c3s(a*b)/4)**Rational(1,4), [a,b,c], []

    def problem_vasile_p31090(self):
        return c3s(sqrt(3*a+b**2)) - 6, [a,b,c], [c3s(a)-3]

    def problem_vasile_p31091(self):
        return c3s(sqrt(a**2+b**2+2*b*c)) - 2*c3s(a), [a,b,c], [c3s(a)-3]

    def problem_vasile_p31092(self):
        return c3s(sqrt(a**2+b**2+7*b*c)) - 3*sqrt(3*c3s(a*b)), [a,b,c], []

    def problem_vasile_p31093(self):
        return c3s(a**4)+5*c3s(a**3*b)-6*c3s(a**2*b**2), [a,b,c], []

    def problem_vasile_p31094(self):
        return c3s(a**5-a**4*b)-2*c3p(a)*(c3s(a**2-a*b)), [a,b,c], []

    def problem_vasile_p31095(self):
        return c3s(a**2)**2 - 3*c3s(a**3*b), [], []

    def problem_vasile_p31096(self):
        return c3s(a**4+a*b**3-2*a**3*b), [], []

    def problem_vasile_p31097(self):
        return c3s(a**2/(a*b+2*c**2)) - 1, [a,b,c], []

    def problem_vasile_p31098(self):
        return c3s(a/(a*b+1)) - Rational(3,2), [a,b,c], [c3s(a)-3]

    def problem_vasile_p31099(self):
        return Rational(3,2) - c3s(a/(3*a+b**2)), [a,b,c], [c3s(a)-3]

    def problem_vasile_p31100(self):
        return c3s(a/(b**2+c)) - Rational(3,2), [a,b,c], [c3s(a)-3]

    def problem_vasile_p31101(self):
        return c3s(a*sqrt(a+b)) - 3*sqrt(2), [a,b,c], [c3s(a)-3]

    def problem_vasile_p31102(self):
        return c3s(a/(2*b**2+c)) - 1, [a,b,c], [c3s(a)-3]

    def problem_vasile_p31103(self):
        return c3s(a**3/(a+b**5)) - Rational(3,2), [a,b,c], [c3s(a**2)-3]

    def problem_vasile_p31104(self):
        return c3s(a/(1+b)) - Rational(3,2), [a,b,c], [c3s(a**2)-3]

    def problem_vasile_p31105(self):
        return c3s(a**2*b)+9-4*c3s(a), [], [c3s(a**2)-3]

    def problem_vasile_p31106(self):
        return c3s((1-a)*(1-a*b)), [], [c3s(a**2)-3]

    def problem_vasile_p31107(self):
        return 1 - c3s(1/(a**2+b+1)), [a,b,c], [c3s(a-a*b)]

    @mark(mark.noimpl)
    def problem_vasile_p31108(self):
        return c3s(a**x/b**y) - 3, [a,b,c,y,x-y], [c3s(a**(x+y))-3]

    def problem_vasile_p31109_p1(self):
        return c3s(1/(4*a)+1/(a+b)-3/(3*a+b)), [a,b,c], []

    def problem_vasile_p31109_p2(self):
        return c3s(1/(4*a)+1/(a+3*b)-2/(3*a+b)), [a,b,c], []

    def problem_vasile_p31110(self):
        return c3s(a**5/b)-3, [a,b,c], [c3s(a**6)-3]

    def problem_vasile_p31111(self):
        return 1/c3s(a*b)/4 -  c3s(1/(a+2*b+3*c)**2), [a,b,c], []

    def problem_vasile_p31112(self):
        return Rational(5,4) - c3s(a*(1-b**2)), [a,b,c,1-a,1-b,1-c], []

    def problem_vasile_p31113_p1(self):
        return c3s(a**2*b)-c3p(a)-2, [c-b,b-1,1-a,a], [c3s(a)-3]

    def problem_vasile_p31113_p2(self):
        return c3s(a**2*b)-3, [c-b,b-1,1-a,a], [c3s(a*b)-3]

    def problem_vasile_p31114(self):
        return c3s(a**4)-Rational(17,8)*c3s(a**3*b), [a,b,c], [c3s(a**2)-Rational(5,2)*c3s(a*b)]

    def problem_vasile_p31115_p1(self):
        return 2*c3s(a**3*b) - c3s(a**2*b**2) - c3p(a)*c3s(a),\
            [a,b,c], [c3s(a**2)-Rational(5,2)*c3s(a*b)]

    def problem_vasile_p31115_p2(self):
        return 11*c3s(a**4) - 17*c3s(a**3*b) - 129*c3p(a)*c3s(a),\
            [a,b,c], [c3s(a**2)-Rational(5,2)*c3s(a*b)]

    def problem_vasile_p31115_p3(self):
        return (14+sqrt(102))/8*c3s(a**2*b**2) - c3s(a**3*b),\
            [a,b,c], [c3s(a**2)-Rational(5,2)*c3s(a*b)]

    def problem_vasile_p31116(self):
        k_ = (1 + sqrt(21+8*sqrt(7)))/2
        return c3s(a**2) - k_*c3s(a*b), [-c3s(a**3*b)]

    def problem_vasile_p31117(self):
        k_ = (-1 + sqrt(21+8*sqrt(7)))/2
        return c3s(a**2) + k_*c3s(a*b), [c3s(a**3*b)]

    @mark(mark.skip)
    def problem_vasile_p31118_p1(self):
        alpha = (1+13*k-5*k**2-2*(1-k)*(1+2*k)*sqrt(7*(1-k)/(1+2*k)))/27
        return c3s(a**3*b)/c3s(a**2)**2 - alpha, [2*k+1,1-k], [k*c3s(a**2)-c3s(a*b)]

    @mark(mark.skip)
    def problem_vasile_p31118_p2(self):
        beta = (1+13*k-5*k**2+2*(1-k)*(1+2*k)*sqrt(7*(1-k)/(1+2*k)))/27
        return beta - c3s(a**3*b)/c3s(a**2)**2, [2*k+1,1-k], [k*c3s(a**2)-c3s(a*b)]

    def problem_vasile_p31119(self):
        return c3s((a**2+3*a*b)/(b+c)**2) - 3, [a,b,c], []

    def problem_vasile_p31120(self):
        return c3s((a**2*b+1)/(a*(b+1))) - 3, [a,b,c], []

    def problem_vasile_p31121(self):
        return c3s(sqrt(a**3+3*b)) - 6, [a,b,c], [c3s(a)-3]

    def problem_vasile_p31122(self):
        return c3s(sqrt(a/(a+6*b+2*b*c)))-1, [a,b,c], [c3p(a)-1]

    def problem_vasile_p31123(self):
        return c3s(a**2/(4*a+b**2)) - Rational(3,5), [a,b,c], [c3s(a)-3]

    def problem_vasile_p31124(self):
        return c3s(a)**2/(3*c3s(a*b)) - c3s((a**2+b*c)/(a+b)), [a,b,c], []

    def problem_vasile_p31125(self):
        return 3*sqrt(2)-c3s(sqrt(a*b**2+b*c**2)), [a,b,c], [c3s(a)-3]

    def problem_vasile_p31126(self):
        return c3s((a+1/b)**2) - 6*(c3s(a)-1), [a,b,c], [c3p(a)-1]

    def problem_vasile_p31127(self):
        return 3+1/c3p(a)-12/c3s(a**2*b), [a,b,c], [c3s(a)-3]

    def problem_vasile_p31128(self):
        return 24/c3s(a**2*b)+1/c3p(a)-9, [a,b,c], [c3s(a)-3]

    def problem_vasile_p31129(self):
        return c3s(a/(a+b)) - c3s(a)/(c3s(a)-cbrt(c3p(a))), [a,b,c]

    def problem_vasile_p31130(self):
        return 3*sqrt(3) - c3s(a*sqrt(b**2+b+1)), [a,b,c], [c3s(a)-3]

    def problem_vasile_p31131(self):
        return 1/(12*c3p(a)) - c3s(1/(b*(a+2*b+3*c)**2)), [a,b,c], []

    def problem_vasile_p31132_p1(self):
        return c3s((a**2+9*b)/(b+c)) - 15, [a,b,c], [c3s(a)-3]

    def problem_vasile_p31132_p2(self):
        return c3s((a**2+3*b)/(a+b)) - 6, [a,b,c], [c3s(a)-3]

    def problem_vasile_p31133(self):
        return c3s(a**2/b)-3, [a,b,c], [c3s(a**5)-3]

    @mark(mark.noimpl, mark.quant)
    def problem_vasile_p31134(self):
        return

    def problem_vasile_p31135(self):
        return c3s(a**3)-3*c3p(a) - sqrt(9+6*sqrt(3))*c3p(a-b), [a,b,c], []

    def problem_vasile_p31136(self):
        return c3s(a/(3*a+b-c))-1, [b+c-a,c+a-b,a+b-c], []

    def problem_vasile_p31137(self):
        return -c3s((a**2-b**2)/(a**2+b*c)), [b+c-a,c+a-b,a+b-c], []

    def problem_vasile_p31138(self):
        return c3s(a**2*(a+b)*(b-c)), [b+c-a,c+a-b,a+b-c], []

    def problem_vasile_p31139(self):
        return c3s(a**2*(b/c-1)), [b+c-a,c+a-b,a+b-c], []

    def problem_vasile_p31140_p1(self):
        return c3s(a**3*b)-c3s(a**2*b**2), [b+c-a,c+a-b,a+b-c], []

    def problem_vasile_p31140_p2(self):
        return 3*c3s(a**3*b)-c3s(a*b)*c3s(a**2), [b+c-a,c+a-b,a+b-c], []

    def problem_vasile_p31140_p3(self):
        return c3s(a**3*b)/3-c3s(a/3)**4, [b+c-a,c+a-b,a+b-c], []

    def problem_vasile_p31141(self):
        return 2*c3s(a**2/b**2) - c3s(b**2/a**2) - 3, [b+c-a,c+a-b,a+b-c], []

    def problem_vasile_p31142(self):
        return -c3s(a**2/(a**2-b**2)), [c-b,b-a,a,b+c-a,c+a-b,a+b-c], []

    def problem_vasile_p31143(self):
        return c3s(a/b)+3-2*c3s((a+b)/(b+c)), [b+c-a,c+a-b,a+b-c], []

    @mark(mark.noimpl)
    def problem_vasile_p31144(self):
        return c3s(a**k*b*(a-b)), [b+c-a,c+a-b,a+b-c,k-2], []

    @mark(mark.noimpl)
    def problem_vasile_p31145(self):
        return 3*c3s(a**k*b)-c3s(a)*c3s(a**(k-1)*b), [b+c-a,c+a-b,a+b-c,k-2]

    def problem_vasile_p31146(self):
        return c4s(a/(3+b))-1, [a,b,c,d], [c4s(a)-4]

    def problem_vasile_p31147(self):
        return c4s(a/(1+b**2))-2, [a,b,c,d], [c4s(a)-4]

    def problem_vasile_p31148(self):
        return 4 - c4s(a**2*b*c), [a,b,c,d], [c4s(a)-4]

    def problem_vasile_p31149(self):
        return 16 - c4s(a*(b+c)**2), [a,b,c,d], [c4s(a)-4]

    def problem_vasile_p31150(self):
        return c4s((a-b)/(b+c)), [a,b,c,d], []

    def problem_vasile_p31151_p1(self):
        return c4s((a-b)/(a+2*b+c)), [a,b,c,d], []

    def problem_vasile_p31151_p2(self):
        return 1 - c4s(a/(2*a+b+c)), [a,b,c,d], []

    def problem_vasile_p31152(self):
        return c4s(1/(a*(a+b))) - 2, [a,b,c,d], [c4p(a)-1]

    @mark(mark.skip)
    def problem_vasile_p31153_p1(self):
        return c4s(1/(a*(1+b))) - 16/(1+8*sqrt(c4p(a))), [a,b,c,d], []

    @mark(mark.skip)
    def problem_vasile_p31153_p2(self):
        return 1/a/(1+b)+1/b/(1+a)+1/c/(1+d)+1/d/(1+c) - 16/(1+8*sqrt(c4p(a))), [a,b,c,d], []

    def problem_vasile_p31154_p1(self):
        return 3*c4s(a)-2*c4s(a*b)-4, [a,b,c,d], [c4s(a**2)-4]

    def problem_vasile_p31154_p2(self):
        return c4s(a)-4-(2-sqrt(2))*(c4s(a*b)-4), [a,b,c,d], [c4s(a**2)-4]

    def problem_vasile_p31155_p1(self):
        return c4p(a+1/b)-c4s(a)*c4s(1/a), [a-1,b-1,c-1,d-1], []

    def problem_vasile_p31155_p2(self):
        return c4p(a+1/b)-c4s(a)*c4s(1/a), [a,b,c,d], [c4p(a)-1]

    def problem_vasile_p31156(self):
        return c4s((1+a/(a+b))**2) - 7, [a,b,c,d], []

    def problem_vasile_p31157(self):
        return c4s((a**2-b*d)/(b+2*c+d)), [a,b,c,d], []

    def problem_vasile_p31158(self):
        return 4 - c4s(sqrt(2*a/(a+b))), [d-c,c-b,b-a,a], []

    def problem_vasile_p31159_p1(self):
        x_, y_, z_, t_ = a/(b+c), b/(c+d), c/(d+a), d/(a+b)
        return 1 - sqrt(x_*z_) - sqrt(y_*t_), [a,b,c,d], []

    def problem_vasile_p31159_p2(self):
        x_, y_, z_, t_ = a/(b+c), b/(c+d), c/(d+a), d/(a+b)
        return x_+y_+z_+t_+4*(x_*z_+y_*t_)-4, [a,b,c,d], []

    def problem_vasile_p31160(self):
        return c4p(1+2*a/(b+c)) - 9, [a,b,c,d] ,[]

    @mark(mark.skip)
    def problem_vasile_p31161(self):
        return c4p(1+k*a/(b+c)) - (1+k)**2, [a,b,c,d,k], []

    def problem_vasile_p31162(self):
        return c4s(1/(a*b)) - c4s(a**2), [a,b,c,d], [c4s(a)-4]

    def problem_vasile_p31163(self):
        return c4s(a**2/(a+b+c)**2) - Rational(4,9), [a,b,c,d], []

    def problem_vasile_p31164(self):
        return 4 - c4s(a*b*(b+c)), [a,b,c,d], [c4s(a)-3]

    def problem_vasile_p31165(self):
        return 1 - c4s(a*b*(b+c)), [a-b,b-c,c-d,d], [c4s(a)-2]

    def problem_vasile_p31166(self):
        return 4*(1+k) - c4s(a*b*(b+k*c)), [a,b,c,d,k-Rational(37,27)], [c4s(a)-4]

    def problem_vasile_p31167(self):
        return 2*c4s(a/b) - 4 - c4s(a/c), [d-c,c-b,b-a,a], []

    def problem_vasile_p31168(self):
        return c4s(a/b) - c4s(a*b), [d-c,c-b,b-a,a], [c4p(a)-1]

    def problem_vasile_p31169(self):
        return 4 + c4s(a/b) - 2*c4s(a), [d-c,c-b,b-a,a], [c4p(a)-1]

    @mark(mark.noimpl, mark.quant)
    def problem_vasile_p31170(self):
        ...

    def problem_vasile_p31171(self):
        return 1 + 4/c5p(a) - c5s(a/b), [a,b,c,d,e], [c5s(a)-5]

    def problem_vasile_p31172_p1(self):
        return (sqrt(5)-1)/4 - c5s(a*b)/c5s(a**2), [], [c5s(a)]

    def problem_vasile_p31172_p1(self):
        return (sqrt(5)+1)/4 + c5s(a*b)/c5s(a**2), [], [c5s(a)]

    def problem_vasile_p31173(self):
        return c5s(a**2/(b+c+d)) - Rational(5,3), [a,b,c,d,e], [c5s(a**2)-5]

    @mark(mark.skip)
    def problem_vasile_p31174(self):
        return Rational(729,2) - c5p(a**2+b**2), [a,b,c,d,e], [c5s(a)-5]

    def problem_vasile_p31175(self):
        return c5s((a-b)/(b+c)), [a-1,b-1,c-1,d-1,e-1,5-a,5-b,5-c,5-d,5-e], []

    @mark(mark.skip)
    def problem_vasile_p31176(self):
        return (a-b)/(b+c)+(b-c)/(c+d)+(c-d)/(d+e)+(d-e)/(e+f)+(e-f)/(f+a)+(f-a)/(a+b),\
            [_-1 for _ in [a,b,c,d,e,f]] + [3-_ for _ in [a,b,c,d,e,f]], []

    @mark(mark.noimpl, mark.nvars)
    def problem_vasile_p31177(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_vasile_p31178(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_vasile_p31179(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_vasile_p31180(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_vasile_p31181(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_vasile_p31182(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_vasile_p31183(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_vasile_p31184(self):
        ...

    def problem_vasile_p32001(self):
        return 4 - (a*b + c)*(a*c + b), [a, b, c], [a + b + c - 3]

    def problem_vasile_p32002(self):
        return a**3 + b**3 + c**3 - 3*a*b*c - Rational(1,4)*(b + c - 2*a)**3, [a, b, c], []

    def problem_vasile_p32003_p1(self):
        return a**3 + b**3 + c**3 - 3*a*b*c - 2*(2*b - a - c)**3, [a - b, b - c, c], []

    def problem_vasile_p32003_p2(self):
        return a**3 + b**3 + c**3 - 3*a*b*c - (a - 2*b + c)**3, [a - b, b - c, c], []

    def problem_vasile_p32004_p1(self):
        return a**3 + b**3 + c**3 - 3*a*b*c - 3*(a**2 - b**2)*(b - c), [a - b, b - c, c], []

    def problem_vasile_p32004_p2(self):
        return a**3 + b**3 + c**3 - 3*a*b*c - Rational(9,2)*(a - b)*(b**2 - c**2), [a - b, b - c, c], []

    def problem_vasile_p32005(self):
        return a**6 + b**6 + c**6 - 3*a**2*b**2*c**2 - 2*(b**4 + c**4 + 4*b**2*c**2)*(b - c)**2, [a - b, a - c, a, b, c], []

    def problem_vasile_p32006(self):
        return a**2 + b**2 + c**2 - (9*a*b*c)/(a + b + c) - Rational(5,3)*(b - c)**2, [a - b, a - c, a, b, c], []

    def problem_vasile_p32007(self):
        return 1/((a + b)**2) + 1/((a + c)**2) + 16/((b + c)**2) - 6/(a*b + b*c + c*a), [a, b, c], []

    def problem_vasile_p32008(self):
        return 1/((a + b)**2) + 1/((a + c)**2) + 2/((b + c)**2) - Rational(5,2)/(a*b + b*c + c*a), [a, b, c], []

    def problem_vasile_p32009(self):
        return (a + b)**3*(a + c)**3 - 4*a**2*b*c*(2*a + b + c)**2, [a, b, c], []

    def problem_vasile_p32010_p1(self):
        return (a/b + b/c + 1/a - a - b - 1), [a, b, c], [a*b*c - 1]

    def problem_vasile_p32010_p2(self):
        return (a/b + b/c + 1/a - sqrt(3*(a**2 + b**2 + 1))), [a, b, c], [a*b*c - 1]

    @mark(mark.noimpl)
    def problem_vasile_p32011(self):
        return (a**(a/b) * b**(b/c) * c**(c/a) - 1), [a, b, c, a*b*c - 1], []

    def problem_vasile_p32012(self):
        return 4 - a*b**2*c**3, [a, b, c, b - a, c - b], [a*b + b*c + c*a - 3]

    def problem_vasile_p32013(self):
        return Rational(1,3) - a*b**2*c**2, [b - a, c - b, a, b, c], [a*b + b*c + c*a - Rational(5,3)]

    def problem_vasile_p32014_p1(self):
        return Rational(9,8) - a*b**2*c, [a, b, c, b - a, c - b], [a*b + b*c + c*a - 3]

    def problem_vasile_p32014_p2(self):
        return 2 - a*b**4*c, [a, b, c, b - a, c - b], [a*b + b*c + c*a - 3]

    def problem_vasile_p32014_p3(self):
        return 2 - a**2*b**3*c, [a, b, c, b - a, c - b], [a*b + b*c + c*a - 3]

    def problem_vasile_p32015(self):
        return a*b**2*c**3 - 1, [a, b, c, b - a, c - b], [a + b + c - (1/a + 1/b + 1/c)]

    def problem_vasile_p32016(self):
        return (1 - b)*(1 - a*b**3*c), [a, b, c, b - a, c - b], [a + b + c - a*b*c - 2]

    def problem_vasile_p32017(self):
        return b - 1/(a + c - 1), [a, b, c, b - a, c - b], [a + b + c - (1/a + 1/b + 1/c)]

    def problem_vasile_p32018_p1(self):
        return ((a - b)**2)/(a**2 + b**2) + ((a - c)**2)/(a**2 + c**2) - ((b - c)**2)/(2*(b + c)**2), [], []

    def problem_vasile_p32018_p2(self):
        return ((a + b)**2)/(a**2 + b**2) + ((a + c)**2)/(a**2 + c**2) - ((b - c)**2)/(2*(b + c)**2), [], []

    def problem_vasile_p32019_p1(self):
        return ((a - b)**2)/(a**2 + b**2) + ((a - c)**2)/(a**2 + c**2) - ((b - c)**2)/((b + c)**2), [b*c], []

    def problem_vasile_p32019_p2(self):
        return ((a + b)**2)/(a**2 + b**2) + ((a + c)**2)/(a**2 + c**2) - ((b - c)**2)/((b + c)**2), [b*c], []

    @mark(mark.skip)
    def problem_vasile_p32020(self):
        return (Abs(a - b)**3)/(a**3 + b**3) + (Abs(a - c)**3)/(a**3 + c**3) - (Abs(b - c)**3)/(b + c)**3, [a, b, c], []

    def problem_vasile_p32021(self):
        return (b + c)**2/(4*(b - c)**2) - (a*b)/(a + b)**2 - (a*c)/(a + c)**2, [a, b, c], []

    def problem_vasile_p32022(self):
        return (3*b*c + a**2)/(b**2 + c**2) - (3*a*b - c**2)/(a**2 + b**2) - (3*a*c - b**2)/(a**2 + c**2), [a, b, c], []

    def problem_vasile_p32023(self):
        return a*b*c - (b + c - a)*(c + a - b)*(a + b - c) - (a*b*(a - b)**2)/(a + b), [a, b, c], []

    def problem_vasile_p32024_p1(self):
        return a*b*c - (b + c - a)*(c + a - b)*(a + b - c) - (2*a*b*(a - b)**2)/(a + b), [a - b, b - c, c], []

    def problem_vasile_p32024_p2(self):
        return a*b*c - (b + c - a)*(c + a - b)*(a + b - c) - (27*b*(a - b)**4)/(4*a**2), [a - b, b - c, c], []

    def problem_vasile_p32025(self):
        return c3s(a**2*(a - b)*(a - c)) - a**2*b**2*((a - b)/(a + b))**2, [a, b, c], []

    def problem_vasile_p32026(self):
        return 8 - a*b**2 - b*c**2 - 2*c*a**2, [a, b, c], [a + b + c - 3]

    def problem_vasile_p32027(self):
        return 4 - a*b**2 - b*c**2 - Rational(3,2)*a*b*c, [a, b, c], [a + b + c - 3]

    def problem_vasile_p32028(self):
        return 20 - a*b**2 - b*c**2 - 2*a*b*c, [a, b, c], [a + b + c - 5]

    def problem_vasile_p32029(self):
        return a**3 + b**3 + c**3 - a**2*b - b**2*c - c**2*a - Rational(8,9)*(a - b)*(b - c)**2, [a, b, c], []

    def problem_vasile_p32030_p1(self):
        return c3s(a**2*(a - b)*(a - c)) - 4*a**2*b**2*((a - b)/(a + b))**2, [a - b, b - c, c], []

    def problem_vasile_p32030_p2(self):
        return c3s(a**2*(a - b)*(a - c)) - (27*b*(a - b)**4)/(4*a), [a - b, b - c, c], []

    def problem_vasile_p32031(self):
        return (a/b + b/c + c/a) - 3 - 2*(a - c)**2/(a + c)**2, [a, b, c], []

    def problem_vasile_p32032(self):
        return (a/b + b/c + c/a) - 3 - (a - c)**2/(a*b + b*c + c*a), [a, b, c], []

    def problem_vasile_p32033(self):
        return (a/b + b/c + c/a) - 3 - 4*(a - c)**2/(a + b + c)**2, [a, b, c], []

    def problem_vasile_p32034(self):
        return (a/b + b/c + c/a) - 3 - 3*(b-c)**2/(a*b+b*c+c*a), [a-b, b-c, c], []

    def problem_vasile_p32035(self):
        return (a**2/b + b**2/c + c**2/a) - a - b - c - 4*(a - c)**2/(a + b + c), [a, b, c], []

    def problem_vasile_p32036(self):
        return (a**2/b + b**2/c + c**2/a) - a - b - c - 6*(b - c)**2/(a + b + c), [a - b, b - c, c], []

    def problem_vasile_p32037(self):
        return (a**2/b + b**2/c + c**2/a - 5*(a - b)), [a - b, b - c, c], []

    def problem_vasile_p32038(self):
        return (a/(b + c) + b/(c + a) + c/(a + b) - Rational(3,2) - 27*(b - c)**2/(16*(a + b + c)**2)), [a, b, c], []

    def problem_vasile_p32039(self):
        return (a/(b + c) + b/(c + a) + c/(a + b) - Rational(3,2) - 9*(b - c)**2/(4*(a + b + c)**2)), [a, b, c, b - a, c - a], []

    def problem_vasile_p32040(self):
        return (a/(b + c) + b/(c + a) + c/(a + b) - Rational(3,2) - (b - c)**2/(2*(b + c)**2)), [a, b, c], []

    def problem_vasile_p32041(self):
        return (a/(b + c) + b/(c + a) + c/(a + b) - Rational(3,2) - (b - c)**2/(4*b*c)), [b - a, c - a, a], []

    def problem_vasile_p32042_p1(self):
        return 1 - ( (a*b + b*c + c*a)/(a**2 + b**2 + c**2) + 2*(b - c)**2/(3*(b**2 + c**2)) ), [b-a, c-a, a], []

    def problem_vasile_p32042_p2(self):
        return 1 - ( (a*b + b*c + c*a)/(a**2 + b**2 + c**2) + (a - b)**2/(2*(a**2 + b**2)) ), [b-a, c-a, a], []

    def problem_vasile_p32043_p1(self):
        return 1 - ( (a*b + b*c + c*a)/(a**2 + b**2 + c**2) + (b - c)**2/(2*(a*b + b*c + c*a)) ), [a - b, a - c, b, c], []

    def problem_vasile_p32043_p2(self):
        return 1 - ( (a*b + b*c + c*a)/(a**2 + b**2 + c**2) + 2*(b - c)**2/(a + b + c)**2 ), [a - b, a - c, b, c], []   

    def problem_vasile_p32044_p1(self):
        return (a**2 + b**2 + c**2)/(a*b + b*c + c*a) - 1 - 4*(b - c)**2/(3*(b + c)**2), [b - a, c - a, a], []

    def problem_vasile_p32044_p2(self):
        return (a**2 + b**2 + c**2)/(a*b + b*c + c*a) - 1 - (a - b)**2/(a + b)**2, [b - a, c - a, a], []

    def problem_vasile_p32045(self):
        return (a**2 + b**2 + c**2)/(a*b + b*c + c*a) - 1 - 9*(a - c)**2/(4*(a + b + c)**2), [a, b, c], []

    def problem_vasile_p32046(self):
        return 1/sqrt(a**2 - a*b + b**2) + 1/sqrt(b**2 - b*c + c**2) + 1/sqrt(c**2 - c*a + a**2) - 6/(b + c), [b - a, c - a, a], []

    def problem_vasile_p32047(self):
        return (4 - 2*sqrt(2)) - a*c, [a - 1, 1 - b, b - c, c], [a*b + b*c + c*a - a*b*c - 2]

    def problem_vasile_p32048(self):
        return 2 - a**4*(b**4 + c**4), [b - a, c - b, a], [a + b + c - 3]

    def problem_vasile_p32049(self):
        return a**2 + b**2 + c**2 - a - b - c - Rational(5,8)*(a - c)**2, [a, b, c], [a*b + b*c + c*a - 3]

    def problem_vasile_p32050(self):
        return (a**3 + b**3 + c**3)/(a + b + c) - 1 - Rational(5,9)*(a - c)**2, [a, b, c], [a*b + b*c + c*a - 3]

    def problem_vasile_p32051_p1(self):
        return (a**3 + b**3 + c**3)/(a + b + c) - 1 - Rational(7,9)*(a - b)**2, [a - b, b - c, c], [a*b + b*c + c*a - 3]

    def problem_vasile_p32051_p2(self):
        return (a**3 + b**3 + c**3)/(a + b + c) - 1 - Rational(2,3)*(b - c)**2, [a - b, b - c, c], [a*b + b*c + c*a - 3]

    def problem_vasile_p32052(self):
        return a**4 + b**4 + c**4 - a**2 - b**2 - c**2 - Rational(11,4)*(a - c)**2, [a, b, c], [a*b + b*c + c*a - 3]

    def problem_vasile_p32053_p1(self):
        return a**4 + b**4 + c**4 - a**2 - b**2 - c**2 - Rational(11,3)*(a - b)**2, [a - b, b - c, c], [a*b + b*c + c*a - 3]

    def problem_vasile_p32053_p2(self):
        return a**4 + b**4 + c**4 - a**2 - b**2 - c**2 - Rational(10,3)*(b - c)**2, [a - b, b - c, c], [a*b + b*c + c*a - 3]

    @mark(mark.noimpl)
    def problem_vasile_p32054(self):
        ...

    def problem_vasile_p32055(self):
        return 8 + a/c - 3*(a+b+c), [a-b, b-c, c], [a*b*c-1]

    def problem_vasile_p32056(self):
        return (a + b - c)*(a**2*b - b**2*c + c**2*a) - (a*b - b*c + c*a)**2, [a - b, b - c, c], []

    def problem_vasile_p32057_p1(self):
        return (a + b + c - 3*(a*b*c)**Rational(1,3)) - (a - c)**2/(2*(a + c)), [a - b, b - c, c], []

    def problem_vasile_p32057_p2(self):
        return (2*(a - c)**2)/(a + 5*c) - (a + b + c - 3*(a*b*c)**Rational(1,3)), [a - b, b - c, c], []

    def problem_vasile_p32058_p1(self):
        return (a + b + c + d - 4*(a*b*c*d)**Rational(1,4)) - (a - d)**2/(a + 3*d), [a - b, b - c, c - d, d], []

    def problem_vasile_p32058_p2(self):
        return (3*(a - d)**2)/(a + 5*d) - (a + b + c + d - 4*(a*b*c*d)**Rational(1,4)), [a - b, b - c, c - d, d], []

    def problem_vasile_p32059_p1(self):
        return (a + b + c - 3*(a*b*c)**Rational(1,3)) - 3*(a - b)**2/(5*a + 4*b), [a - b, b - c, c], []

    def problem_vasile_p32059_p2(self):
        return (a + b + c - 3*(a*b*c)**Rational(1,3)) - 64*(a - b)**2/(7*(11*a + 24*b)), [a - b, b - c, c], []

    def problem_vasile_p32060_p1(self):
        return (a + b + c - 3*(a*b*c)**Rational(1,3)) - 3*(b - c)**2/(4*b + 5*c), [a - b, b - c, c], []

    def problem_vasile_p32060_p2(self):
        return (a + b + c - 3*(a*b*c)**Rational(1,3)) - 25*(b - c)**2/(7*(3*b + 11*c)), [a - b, b - c, c], []

    def problem_vasile_p32061(self):
        return (a + b + c - 3*(a*b*c)**Rational(1,3)) - 3*(a - c)**2/(4*(a + b + c)), [a - b, b - c, c], []

    def problem_vasile_p32062_p1(self):
        return a**6 + b**6 + c**6 - 3*a**2*b**2*c**2 - 12*a**2*c**2*(b - c)**2, [a - b, b - c, c], []

    def problem_vasile_p32062_p2(self):
        return a**6 + b**6 + c**6 - 3*a**2*b**2*c**2 - 10*a**3*c*(b - c)**2, [a - b, b - c, c], []

    @mark(mark.skip)
    def problem_vasile_p32063_p1(self):
        E = (k*a + b + c)*(k/a + 1/b + 1/c)
        F = (k*a**2 + b**2 + c**2)*(k/a**2 + 1/b**2 + 1/c**2)
        return sqrt(( F - (k-2)**2 )/(2*k)) + 2 - ( E - (k-2)**2 )/(2*k), [a, b, c, k - 1], []

    @mark(mark.skip)
    def problem_vasile_p32063_p2(self):
        E = (k*a + b + c)*(k/a + 1/b + 1/c)
        F = (k*a**2 + b**2 + c**2)*(k/a**2 + 1/b**2 + 1/c**2)
        return sqrt(( F - k**2 )/(k + 1)) + 2 - ( E - k**2 )/(k + 1), [a, b, c, 1 - k, k], []

    def problem_vasile_p32064(self):
        return a/(2*b + 6*c) + b/(7*c + a) + 25*c/(9*a + 8*b) - 1, [a, b, c], []

    def problem_vasile_p32065(self):
        return 1/(a + b) + 1/(b + c) + 1/(c + a) - 55/(12*(a + b + c)), [a, b, c, 1/a - 1/b - 1/c], []

    def problem_vasile_p32066(self):
        return 1/(a**2 + b**2) + 1/(b**2 + c**2) + 1/(c**2 + a**2) - 189/(40*(a**2 + b**2 + c**2)),\
            [a, b, c, 1/a - 1/b - 1/c], []

    def problem_vasile_p32067(self):
        return a**3*(b + c) + b*c*(b**2 + c**2) - a*(b**3 + c**3), [b + c - a, a + b - c, a + c - b], []

    def problem_vasile_p32068(self):
        return ((a + b)**2)/(2*a*b + c**2) + ((a + c)**2)/(2*a*c + b**2) - ((b + c)**2)/(2*b*c + a**2),\
            [b + c - a, a + b - c, a + c - b], []

    def problem_vasile_p32069(self):
        return (a + b)/(a*b + c**2) + (a + c)/(a*c + b**2) - (b + c)/(b*c + a**2), [b + c - a, a + b - c, a + c - b], []

    def problem_vasile_p32070(self):
        return (b*(a + c))/(a*c + b**2) + (c*(a + b))/(a*b + c**2) - (a*(b + c))/(b*c + a**2),\
            [b + c - a, a + b - c, a + c - b], []

    def problem_vasile_p32071(self):
        return (a + b)*(c + d) - 2*(a*b + c*d), [a, b, c, d], [a**2 - a*b + b**2 - c**2 + c*d - d**2]

    @mark(mark.noimpl)
    def problem_vasile_p32072_p1(self):
        E = exp(1)
        return 2*a**a - a**b - b**a, [a, b - a, Rational(1, E) - a, 1 - b], []

    @mark(mark.noimpl)
    def problem_vasile_p32072(self):
        E = exp(1)
        return 2*b**b - a**b - b**a, [a, b - a, b - Rational(1, E), 1 - b], []

    @mark(mark.noimpl)
    def problem_vasile_p32073(self):
        return 2*b**(2*b) - a**(2*b) - b**(2*a), [a, b - a, b - Rational(1,2)], []

    @mark(mark.noimpl)
    def problem_vasile_p32074_p1(self):
        return 1 + (a - b)/sqrt(a) - a**(b - a), [a - b, b], []

    @mark(mark.noimpl)
    def problem_vasile_p32074_p2(self):
        return a**(a - b) + 3*(a - b)/(4*sqrt(a)) - 1, [a - b, b], []

    def problem_vasile_p32075(self):
        return a*x**2 + b*y**2 + c*z**2 + x*y*z - 4*a*b*c, [a,b,c,x,y,z], [x + y + z - a - b - c]

    def problem_vasile_p32076(self):
        return x*(3*x + a)/(b*c) + y*(3*y + b)/(c*a) + z*(3*z + c)/(a*b) - 12, [a,b,c,x,y,z], [x + y + z - a - b - c]

    @mark(mark.noimpl)
    def problem_vasile_p32077(self):
        ...

    def problem_vasile_p32078_p1(self):
        return ((b+c)*x+(c+a)*y+(a+b)*z)**2 - 4*(a*b+b*c+c*a)*(x*y+y*z+z*x), [a*b+b*c+c*a], []

    def problem_vasile_p32078_p2(self):
        return ((b+c)*x+(c+a)*y+(a+b)*z)**2 - 4*(a+b+c)*(a*y*z+b*z*x+c*x*y), [a,b,c], []

    @mark(mark.noimpl)
    def problem_vasile_p32079(self):
        ...

    @mark(mark.skip)
    def problem_vasile_p32080_p1(self):
        return x + y + z - sqrt(4*(a + b + c + sqrt(a*b) + sqrt(b*c) + sqrt(c*a)) + 3*(a*b*c)**Rational(1,3)), [],\
            [a/(y*z) + b/(z*x) + c/(x*y) - 1]

    @mark(mark.skip)
    def problem_vasile_p32080_p2(self):
        return x + y + z - (sqrt(a + b) + sqrt(b + c) + sqrt(c + a)), [], [a/(y*z) + b/(z*x) + c/(x*y) - 1]

    def problem_vasile_p32081(self):
        return (y*a**2 + z*b**2 + x*c**2)*(z*a**2 + x*b**2 + y*c**2) - (x*y + y*z + z*x)*(a**2*b**2 + b**2*c**2 + c**2*a**2),\
            [b + c - a, a + c - b, a + b - c], []

    def problem_vasile_p32082(self):
        return 6*(a**2 + b**2 + c**2 + d**2) + (a + b + c + d)**2 - 12*(a*b + b*c + c*d), [], []

    def problem_vasile_p32083(self):
        return 1/(a**2 + a*b) + 1/(b**2 + b*c) + 1/(c**2 + c*d) + 1/(d**2 + d*a) - 4/(a*c + b*d), [a,b,c,d], []

    def problem_vasile_p32084(self):
        return 4 - a**3*b*c*d, [a - b, b - c, c - d, d], [a*b + b*c + c*d + d*a - 3]

    def problem_vasile_p32085(self):
        return 2 - a*c*d, [a - b, b - c, c - d, d], [a*b + b*c + c*d + d*a - 6]

    def problem_vasile_p32086(self):
        return 4 - a*b*d, [a - b, b - c, c - d, d], [a*b + b*c + c*d + d*a - 9]

    def problem_vasile_p32087(self):
        return 3*c + 5 - 2*b - 4*d, [a - b, b - c, c - d, d], [a**2 + b**2 + c**2 + d**2 - 10]

    def problem_vasile_p32088(self):
        return (a + b + c + d)**2 - 8*(a*c + b*d), [a - b, b - c, c - d, d], []

    def problem_vasile_p32089(self):
        return 4 + a/b + b/c + c/d + d/a - 2*(a*c + b*c + b*d + a*d), [a, b - a, c - b, d - c], [a*b*c*d - 1]

    def problem_vasile_p32090_p1(self):
        return 2*(b + c) - (a + d), [a - b, b - c, c - d, d], [3*(a**2 + b**2 + c**2 + d**2) - (a + b + c + d)**2]

    def problem_vasile_p32090_p2(self):
        return (7 + 2*sqrt(6))/5 - (a+c)/(b+d), [a - b, b - c, c - d, d], [3*(a**2 + b**2 + c**2 + d**2) - (a + b + c + d)**2]

    def problem_vasile_p32090_p3(self):
        return (3 + sqrt(5))/2 - (a+c)/(c+d), [a - b, b - c, c - d, d], [3*(a**2 + b**2 + c**2 + d**2) - (a + b + c + d)**2]

    def problem_vasile_p32091(self):
        return a - b - 3*c - (2*sqrt(3) - 1)*d, [a - b, b - c, c - d, d], [2*(a**2 + b**2 + c**2 + d**2) - (a + b + c + d)**2]

    def problem_vasile_p32092(self):
        lhs = (a + b + c + d) - 4*(a*b*c*d)**Rational(1,4)
        return lhs - Rational(3,2)*(sqrt(b) - 2*sqrt(c) + sqrt(d))**2, [a - b, b - c, c - d, d], []

    def problem_vasile_p32093_p1(self):
        lhs = (a + b + c + d) - 4*(a*b*c*d)**Rational(1,4)
        return lhs - Rational(2,9)*(3*sqrt(b) - 2*sqrt(c) - sqrt(d))**2, [a - b, b - c, c - d, d], []

    def problem_vasile_p32093_p2(self):
        lhs = (a + b + c + d) - 4*(a*b*c*d)**Rational(1,4)
        return lhs - Rational(1,5)*(3*sqrt(b) - sqrt(c) - 2*sqrt(d))**2, [a - b, b - c, c - d, d], []

    def problem_vasile_p32093_p3(self):
        lhs = (a + b + c + d) - 4*(a*b*c*d)**Rational(1,4)
        return lhs - Rational(3,8)*(sqrt(b) - 3*sqrt(c) + 2*sqrt(d))**2, [a - b, b - c, c - d, d], []

    def problem_vasile_p32093_p4(self):
        lhs = (a + b + c + d) - 4*(a*b*c*d)**Rational(1,4)
        return lhs - Rational(1,2)*(2*sqrt(b) - 3*sqrt(c) + sqrt(d))**2, [a - b, b - c, c - d, d], []

    def problem_vasile_p32093_p5(self):
        lhs = (a + b + c + d) - 4*(a*b*c*d)**Rational(1,4)
        return lhs - Rational(1,6)*(2*sqrt(b) + sqrt(c) - 3*sqrt(d))**2, [a - b, b - c, c - d, d], []

    def problem_vasile_p32093_p6(self):
        lhs = (a + b + c + d) - 4*(a*b*c*d)**Rational(1,4)
        return lhs - Rational(4,3)*(sqrt(b) - sqrt(d))**2, [a - b, b - c, c - d, d], []

    def problem_vasile_p32094(self):
        return (sqrt(3)/2) - (a*b + b*c + c*d + d*e)/(a**2 + b**2 + c**2 + d**2 + e**2), [], []

    def problem_vasile_p32095(self):
        return (a + b + c + d + e + f)**2 - 8*(a*c + b*d + c*e + d*f), [a - b, b - c, c - d, d - e, e - f, f], []

    @mark(mark.skip)
    def problem_vasile_p32096(self):
        ai = symbols('a1:9')
        return sum(ai) - 8*prod(ai)**Rational(1,8) - 3*(sqrt(ai[5])-sqrt(ai[6]))**2,\
            [ai[i]-ai[i+1] for i in range(7)] + [ai[-1]], []

    @mark(mark.noimpl)
    def problem_vasile_p32097(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p32098(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p32099(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p32100(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p32101(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p32102(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p32103(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p32104(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p32105(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p32106(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p32107(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p32108(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p32109(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p32110(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p32111(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p32112(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p32113(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p32114(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p32115(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p32116(self):
        ...

    @mark(mark.noimpl)
    def problem_vasile_p32117(self):
        ...

    def problem_vasile_p32118(self):
        return c3s((1-a)/(3+a**2)), [a-1,1-b,b-c,c], [a*b*c-1]

    @mark(mark.noimpl)
    def problem_vasile_p32119(self):
        ...