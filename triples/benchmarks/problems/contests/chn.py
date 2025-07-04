from ..problem_set import ProblemSet, mark
from sympy.abc import a,b,c,d,e,p,q,r,s,u,v,x,y,z,w
from sympy import symbols, Rational, Add, Mul, sqrt, cbrt, sin, cos, pi, Abs, Min, Max

class CMOProblems(ProblemSet):
    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_1986_p1(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_1986_p3(self):
        ...

    @mark(mark.noimpl, mark.geom)
    def problem_CMO_1986_p4(self):
        ...

    @mark(mark.noimpl, mark.geom)
    def problem_CMO_1987_p5(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_1988_p1(self):
        ...

    @mark(mark.noimpl, mark.geom)
    def problem_CMO_1988_p2(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_1988_p3(self):
        ...

    def problem_CMO_1988_p4_q1(self):
        return b+c-a, [a,b,c,(a**2+b**2+c**2)**2-2*(a**4+b**4+c**4)], []

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_1988_p4_q2(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_1989_p2(self):
        ...

    @mark(mark.noimpl, mark.quant)
    def problem_CMO_1990_p3(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_1992_p1(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_1992_p2(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_1993_p2(self):
        ...

    @mark(mark.noimpl, mark.geom)
    def problem_CMO_1993_p3(self):
        ...

    @mark(mark.geom)
    def problem_CMO_1994_p1(self):
        """AE:BD:DF:FC = a:b:c:d. Then
        [AEG]:[EFG]:[AGD]:[FDG]=a^2:a*c:a*c:c^2,
        [BEH]:[EHF]:[BHC]:[FCH]=b^2:b*d:b*d:d^2,
        Also, [AEFD]:[BEFC]=(a+c):(b+d),
        Hence [EFH]:[EGF]:[ABCD]=a*c*(b+d):b*d*(a+c):(a+c)*(b+d)*(a+b+c+d).
        """
        EFH, EGF, ABCD = a*c*(b+d), b*d*(a+c), (a+c)*(b+d)*(a+b+c+d)
        return ABCD/4 - EFH - EGF, [a,b,c,d], []

    @mark(mark.noimpl, mark.nvars, mark.quant)
    def problem_CMO_1994_p4(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_1995_p1(self):
        ...
        
    @mark(mark.skip)
    def problem_CMO_1995_p3(self):
        summands = [2394*10**6]
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    summands.append(
                        -Abs(k*(x+y-10*i)*((3*x-6*y-36*j)*(19*x+95*y-95*k))))
        return Add(*summands), [], []

    @mark(mark.noimpl)
    def problem_CMO_1995_p5(self):
        """Involving integer variables."""
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_1996_p5(self):
        ...

    @mark(mark.geom)
    def problem_CMO_1996_p6(self):
        dist2 = lambda u,v: ((u[0]-v[0])**2+(u[1]-v[1])**2).expand()
        X,Y,Z = (a,0), (0,sqrt(3)*b), (1-c, sqrt(3)*c)
        return Max(dist2(X,Y),dist2(Y,Z),dist2(Z,X)) - Rational(3,7),\
            [a,b,c,1-a,1-b,1-c], []
    
    @mark(mark.skip, mark.nvars)
    def problem_CMO_1997_p1(self):
        xi = symbols('x1:1998')
        return 189548 - Add(*[_**12 for _ in xi]),\
            [_ + 1/sqrt(3) for _ in xi] + [sqrt(3) - _ for _ in xi],\
            [Add(*xi, 318*sqrt(3))]

    @mark(mark.noimpl, mark.recur, mark.nvars)
    def problem_CMO_1997_p6(self):
        ...

    @mark(mark.skip, mark.geom)
    def problem_CMO_1998_p5(self):
        # Matrix(5,5,[0,1,1,1,1,1,0,c**2,b**2,x**2,1,c**2,0,a**2,y**2,1,b**2,a**2,0,z**2,1,z**2,y**2,x**2,0]).det()
        cayley_menger = -a**4*x**2 - a**4*z**2 - 2*a**2*b**2*c**2 + a**2*b**2*x**2 + 2*a**2*b**2*y**2 \
            + a**2*b**2*z**2 + 2*a**2*c**2*x**2 + 2*a**2*c**2*z**2 + a**2*x**4 - 2*a**2*x**2*z**2 \
            + a**2*z**4 - 2*b**4*y**2 + b**2*c**2*x**2 + 2*b**2*c**2*y**2 + b**2*c**2*z**2 - b**2*x**4 \
            + 2*b**2*x**2*y**2 - 2*b**2*y**4 + 2*b**2*y**2*z**2 - b**2*z**4 - c**4*x**2 - c**4*z**2 \
            + c**2*x**4 - 2*c**2*x**2*z**2 + c**2*z**4
        return y*z*c+z*x*b+x*y*a-a*b*c,\
            [a,b,c,x,y,z,a**2+b**2-c**2,b**2+c**2-a**2,c**2+a**2-b**2], [cayley_menger]

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_1998_p6(self):
        ...

    def problem_CMO_1999_p5(self):
        fx = (x-u)*(x-v)*(x-w)
        a_ = -(u+v+w)
        return fx + Rational(1,27)*(x-a_)**3, [x,u,v,w], []

    @mark(mark.geom)
    def problem_CMO_2000_p1(self):
        S_ = sqrt((a+b+c)*(a+b-c)*(b+c-a)*(c+a-b))/4
        R_ = a*b*c/(4*S_)
        r_ = S_/((a+b+c)/2)
        return (a+b-(R_ + r_)*2), [c-b,b-a,a,a+b-c,a**2+b**2-c**2]

    @mark(mark.noimpl, mark.geom)
    def problem_CMO_2001_p1(self):
        ...

    @mark(mark.noimpl, mark.geom)
    def problem_CMO_2002_p4(self):
        ...

    @mark(mark.noimpl, mark.nvars, mark.quant)
    def problem_CMO_2002_p6(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_2003_p3(self):
        ...

    @mark(mark.skip)
    def problem_CMO_2003_p6(self):
        x1,x2,x3,x4 = symbols('x1:5')
        y1,y2,y3,y4 = symbols('y1:5')
        lhs = (a*y1+b*y2+c*y3+d*y4)**2 + (a*x4+b*x3+c*x2+d*x1)**2
        rhs = 2*((a**2+b**2)/(a*b)+(c**2+d**2)/(c*d))
        return rhs-lhs, [],\
            [xi**2+yi**2-1 for xi,yi in zip((x1,x2,x3,x4),(y1,y2,y3,y4))] + [a*b+c*d-1]

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_2004_p4(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_2004_p5(self):
        ...

    @mark(mark.skip)
    def problem_CMO_2005_p1(self):
        c1,c2,c3,c4 = tuple(map(cos, symbols('t1:5')))
        s1,s2,s3,s4 = tuple(map(sin, symbols('t1:5')))
        return (1+s1*s2*s3*s4+c1*c2*c3*c4)*2-(s1**2+s2**2+s3**2+s4**2),\
            [c1**2*c2**2-(s1*s2-x)**2, c3**2*c4**2-(s3*s4-x)**2], []

    @mark(mark.noimpl, mark.recur, mark.nvars)
    def problem_CMO_2005_p4(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_2006_p1(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_2006_p5(self):
        ...

    @mark(mark.skip)
    def problem_CMO_2007_p1(self):
        A,B,C = (p,q), (u,v), (x,y)
        AC = (p*x-q*y, q*x+p*y)
        BC = (u*x-v*y, v*x+u*y)
        norm2 = lambda z: (z[0]**2+z[1]**2)
        m_, n_ = norm2(A+B), norm2(A-B)
        return Max(norm2(AC+B), norm2(A+BC)) - m_*n_/sqrt(m_**2+n_**2)

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_2007_p5(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_2008_p3(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_2009_p4(self):
        ...

    @mark(mark.noimpl, mark.quant)
    def problem_CMO_2010_p3(self):
        """Complex numbers a,b,c satisfy: any complex |z|<=1
        must have |a*z^2+b*z+c|<=1. Show that |bc|<=3*sqrt(3)/16."""
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_2011_p1(self):
        """Nonconvex QCQP, cyclic."""
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_2011_p5(self):
        """n>=4, a1,...,an,b1,...,bn>=0, sum(ai)=sum(bi)>0,
        show that sum(ai*(ai+bi))/sum(bi*(ai+bi))<=n-1."""
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_2012_p4(self):
        """Nonconvex QCQP, symmetric."""
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_2014_p1(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_2015_p1(self):
        """LP"""
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_2016_p6(self):
        """n>=2, a<=x1,...,xn<=b, show that
        (x1^2/x2+x2^2/x3+...+xn^2/x1) / (x1+...+xn) <= M
        where
        M = (a^2-a*b+b^2)/(a*b) when n is even,
        M = ((n-1)/2*(a^3+b^3)+a^2*b)/(a*b*((k+1)*a+k*b)) when n is odd.
        """
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_2017_p6(self):
        ...

    def problem_CMO_2018_p1_lb(self):
        return (a+b)*(b+c)*(c+d)*(d+e)*(e+a)+512, [a+1,b+1,c+1,d+1,e+1], [a+b+c+d+e-5]

    def problem_CMO_2018_p1_ub(self):
        return 288-(a+b)*(b+c)*(c+d)*(d+e)*(e+a), [a+1,b+1,c+1,d+1,e+1], [a+b+c+d+e-5]

    @mark(mark.noimpl, mark.nvars, mark.geom)
    def problem_CMO_2018_p6(self):
        ...

    @mark(mark.skip, mark.nvars)
    def problem_CMO_2019_p1_q1(self):
        """LP"""
        ai = symbols('a1:41')
        a_, b_, c_, d_ = ai[9], ai[19], ai[29], ai[39]
        cons = [1 - Abs(ai[i]-ai[(i+1)%40]) for i in range(40)]
        return 10 - a_ - b_ - c_ - d_, cons, []

    @mark(mark.skip, mark.nvars)
    def problem_CMO_2019_p1_q2(self):
        """Nonconvex QCQP"""
        ai = symbols('a1:41')
        a_, b_, c_, d_ = ai[9], ai[19], ai[29], ai[39]
        cons = [1 - Abs(ai[i]-ai[(i+1)%40]) for i in range(40)]
        return Rational(425,8) - a_*b_ - c_*d_, cons, []

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_2020_p1(self):
        ...

    @mark(mark.noimpl, mark.quant)
    def problem_CMO_2021_p2(self):
        ...

    @mark(mark.noimpl)
    def problem_CMO_2022_p1(self):
        """Although it is a recursion, it can be written as a finite-var ineq."""
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_2023_p2(self):
        """PSD SDP"""
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CMO_2024_p6(self):
        ...


class ChinaHighSchoolMathLeague2(ProblemSet):
    """China National High School Mathematics League."""
    @mark(mark.noimpl, mark.nvars)
    def problem_CNHSML_2001_p2(self):
        ...

    def problem_CNHSML_2002_p2(self):
        a_ = -(x+y+z)
        b_ = x*y+y*z+z*x
        c_ = -x*y*z
        l_= y - x
        return 3*sqrt(3)/2 - (2*a_**3 - 9*a_*b_ + 27*c_)/l_**3, [l_, z-(x+y)/2], []

    @mark(mark.noimpl)
    def problem_CNHSML_2004_p2(self):
        ...

    def problem_CNHSML_2005_p2(self):
        return x**2/(1+x)+y**2/(1+y)+z**2/(1+z)-Rational(1,2), [a,b,c,x,y,z], [c*y+b*z-a, a*z+c*x-b, b*x+a*y-c]

    @mark(mark.noimpl, mark.geom)
    def problem_CNHSML_2008_p1(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CNHSML_2008_p3(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CNHSML_2009_p2(self):
        """Show that -1<sum_{k=1}^n (k/(k^2+1)) - ln n <= 1/2."""
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CNHSML_2009_p4(self):
        """LP"""
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CNHSML_2010_p3(self):
        ...

    def problem_CNHSML_2011B_p3(self):
        return 6 - (a+b+c), [a-1,b-1,c-1], [a*b*c+2*a**2+2*b**2+2*c**2+c*a-c*b-4*a+4*b-c-28]

    def problem_CNHSML_2012B_p3(self):
        A = (1+sqrt(5))/2
        x1 = sqrt(x + 1)
        return -Abs(x1 - A) + Abs(x - A)/A, [], []

    @mark(mark.noimpl, mark.geom)
    def problem_CNHSML_2012A_p3(self):
        ...

    def problem_CNHSML_2013B_p3_lb(self):
        return sqrt(6-x**2)+sqrt(6-y**2)+sqrt(6-z**2)-sqrt(6)-sqrt(2), [x,y,z], [x**2+y**2+z**2-10]

    def problem_CNHSML_2013B_p3_ub(self):
        return 2*sqrt(6)-(sqrt(6-x**2)+sqrt(6-y**2)+sqrt(6-z**2)), [x,y,z], [x**2+y**2+z**2-10]

    @mark(mark.noimpl)
    def problem_CNHSML_2014B_p2(self):
        """Compute the range of a so that f(x)=sqrt(ax+4) has 3 intersections with f^(-1)(x)."""
        ...

    @mark(mark.noimpl, mark.geom)
    def problem_CNHSML_2014B_p4(self):
        ...

    def problem_CNHSML_2014A_p1(self):
        return Rational(1,4)+sqrt(a*b*c)/2-a*b-b*c-c*a, [a*b*c], [a+b+c-1]

    def problem_CNHSML_2015B_p1(self):
        return ((a-b*c)**2+(b-c*a)**2+(c-a*b)**2)/((a-b)**2+(b-c)**2+(c-a)**2) - Rational(1,2), [a,b,c], []

    @mark(mark.noimpl, mark.nvars)
    def problem_CNHSML_2015A_p1(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CNHSML_2016B_p1(self):
        ...

    @mark(mark.skip, mark.nvars)
    def problem_CNHSML_2016A_p1(self):
        ai = symbols('a1:2017')
        return Rational(1,4**2016) - Mul(*[ai[i]-ai[(i+1)%2016]**2 for i in range(2016)]),\
            [9*ai[i]-11*ai[i+1]**2 for i in range(2015)], []

    @mark(mark.skip)
    def problem_CNHSML_2017B_p1(self):
        d_ = Max(Abs(a),Abs(b),Abs(c))
        return Abs((1+a)*(1+b)*(1+c)) - (1-d_**2), [], [a+b+c]

    @mark(mark.noimpl, mark.quant)
    def problem_CNHSML_2018B_p1(self):
        """f(x) = ax+b+9/x. Show that there exists x in [1,9] that |f(x)|>=2."""
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CNHSML_2018A_p1(self):
        """a1,...,an,b1,...,bn,A,B>0. And ai<=bi, ai<=A, (b1*...*bn)/(a1*...*an) <= B/A.
        Show that ((b1+1)*...*(bn+1))/((a1+1)*...*(an+1)) <= (B+1)/(A+1)."""
        ...

    @mark(mark.skip, mark.nvars)
    def problem_CNHSML_2019B_p1(self):
        ai = symbols('a0:101')
        cons = [ai[i] - ai[101-i] for i in range(1,51)]
        xi_pow = [1] + [(k*ai[k+1]/Add(*ai[1:k+1]))**k for k in range(1,100)]
        return 1 - Mul(*xi_pow), list(ai[1:]) + cons, []

    @mark(mark.noimpl, mark.nvars)
    def problem_CNHSML_2019A_p2(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CNHSML_2020B_p3(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CNHSML_2020A_p2(self):
        ...

    @mark(mark.skip)
    def problem_CNHSML_2021B_p3(self):
        fx = lambda x: x/sqrt(2 - x**4)
        return fx(a)+fx(b)+fx(c)+fx(d)-2, [a,b,c,d,2-a**4,2-b**4,2-c**4,2-d**4], [a**3+b**3+c**3+d**3-2]

    @mark(mark.noimpl, mark.nvars)
    def problem_CNHSML_2021A2_p2(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CNHSML_2021A2_p3(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_CNHSML_2021A1_p3(self):
        ...

    @mark(mark.skip, mark.nvars)
    def problem_CNHSML_2022B_p2(self):
        """LP, cyclic"""
        xi = symbols('x1:2023')
        cons = [a,b-a] + [_ - a for _ in xi] + [b - _ for _ in xi]
        obj = Add(*[Abs(xi[i]-xi[(i+1)%2022]) for i in range(2022)]) / Add(*xi)
        return 2*(b-a)/(b+a) - obj, cons, []

    @mark(mark.skip, mark.nvars)
    def problem_CNHSML_2022A2_p3_q1(self):
        """LP"""
        ai = symbols('a1:10')
        S = Add(*[Min(ai[i],ai[(i+1)%9])*(i+1) for i in range(9)])
        return 6 - S, ai, [Add(*ai) - 1]

    @mark(mark.skip, mark.nvars)
    def problem_CNHSML_2022A2_p3_lb(self):
        """LP"""
        ai = symbols('a1:10')
        S = Add(*[Min(ai[i],ai[(i+1)%9])*(i+1) for i in range(9)])
        T = Add(*[Max(ai[i],ai[(i+1)%9])*(i+1) for i in range(9)])
        return T - Rational(36,5), ai, [Add(*ai) - 1]

    def problem_CNHSML_2022A1_p1_lb(self):
        P = (a-b)*(b-c)*(c-d)
        return P + Rational(1,54), [a-b,c-d], [Abs(a)+2*Abs(b)+3*Abs(c)+4*Abs(d)-1]

    def problem_CNHSML_2022A1_p1_ub(self):
        P = (a-b)*(b-c)*(c-d)
        return Rational(1,324) - P, [a-b,c-d], [Abs(a)+2*Abs(b)+3*Abs(c)+4*Abs(d)-1]

    @mark(mark.noimpl, mark.nvars)
    def problem_CNHSML_2022A_p3(self):
        """Integer"""
        ...

    @mark(mark.skip, mark.nvars)
    def problem_CNHSML_2023B_p3(self):
        ai = symbols('a1:2024')
        obj = [-Abs(ai[i]-ai[j]) for i in range(2023) for j in range(i+1,2023)]
        obj += [1/_ for _ in ai] + [10**6]
        return obj, list(ai) + [1 - _ for _ in ai], []

    @mark(mark.noimpl, mark.nvars)
    def problem_CNHSML_2023A_p3(self):
        ...

    @mark(mark.skip, mark.nvars)
    def problem_CNHSML_2023A_p4(self):
        n = 2023
        a_ = 1 + Rational(1,10**4)
        zij = symbols(f'z0:{n**2}')
        cons = [_ - 1 for _ in zij] + [a_ - _  for _ in zij]
        xi = Add(*[zij[i*n:i*n+i] for i in range(n)]) # row sums
        yi = Add(*[zij[i:n*(n-1)+i:n] for i in range(n)]) # col sums
        M = (Rational(n-1,2)*a + Rational(n+1,2))**n / (n**n * a_**(Rational(n-1,2)))
        # M = (1011*a + 1012)**2023/(2023**2023 * a**1011)
        return M - Mul(*yi, *[1/_ for _ in xi]), cons, []