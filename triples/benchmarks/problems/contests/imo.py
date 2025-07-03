from ..problem_set import ProblemSet, mark
from sympy.abc import a,b,c,d,e,p,q,r,s,x,y,z,w
from sympy import symbols, Rational, Add, sqrt, cbrt, sin, cos, pi, Abs, Min, Max

class IMOProblems(ProblemSet):
    @mark(mark.noimpl, mark.geom)
    def problem_IMO_1974_p2(self):
        ...

    def problem_IMO_1974_p5_lb(self):
        return a/(d+a+b) + b/(a+b+c) + c/(b+c+d) + d/(c+d+a) - 1, [a,b,c,d], []

    def problem_IMO_1974_p5_ub(self):
        return 2 - (a/(d+a+b) + b/(a+b+c) + c/(b+c+d) + d/(c+d+a)), [a,b,c,d], []

    @mark(mark.noimpl, mark.nvars)
    def problem_IMO_1975_p1(self):
        ...

    @mark(mark.noimpl, mark.recur)
    def problem_IMO_1976_p6(self):
        ...

    @mark(mark.noimpl)
    def problem_IMO_1977_p2(self):
        ...

    @mark(mark.noimpl, mark.quant)
    def problem_IMO_1977_p4(self):
        """1 - acos(x) - bsin(x) - Acos(2x) - Bsin(2x) >= 0 holds for all x,
        show that a^2+b^2 <= 2, A^2+B^2 <= 1."""
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_IMO_1978_p5(self):
        """sum (ak/k^2) >= sum(1/k) if a1,...,an are distinct integers."""
        ...

    @mark(mark.noimpl)
    def problem_IMO_1979_p5(self):
        """Compute the range of a so that there exists x1,...,x5 >= 0 that
        sum(k*xk) = a, sum(k^3*xk) = a^2, sum(k^5*xk) = a^3.
        """
        ...

    @mark(mark.skip, mark.geom)
    def problem_IMO_1981_p1(self):
        return a/x+b/y+c/z-a/r-b/r-c/r, [a,b,c,x,y,z,r], [a*x+b*y+c*z-a*r-b*r-c*r]

    @mark(mark.noimpl, mark.recur, mark.quant)
    def problem_IMO_1982_p3(self):
        ...

    def problem_IMO_1983_p6(self):
        return a**2*b*(a-b)+b**2*c*(b-c)+c**2*a*(c-a), [a,b,c,a+b-c,b+c-a,c+a-b], []

    def problem_IMO_1984_p1_lb(self):
        return y*z+z*x+x*y-2*x*y*z, [x,y,z], [x+y+z-1]

    def problem_IMO_1984_p1_ub(self):
        return Rational(7,27)-(y*z+z*x+x*y-2*x*y*z), [x,y,z], [x+y+z-1]

    @mark(mark.noimpl, mark.geom, mark.nvars)
    def problem_IMO_1984_p5(self):
        ...

    @mark(mark.noimpl, mark.nvars, mark.quant)
    def problem_IMO_1987_p3(self):
        ...

    @mark(mark.noimpl, mark.geom)
    def problem_IMO_1988_p5(self):
        ...

    @mark(mark.noimpl, mark.geom)
    def problem_IMO_1989_p2(self):
        ...

    @mark(mark.noimpl, mark.geom)
    def problem_IMO_1989_p4(self):
        """ABCD is a convex polyhedron. AB=AD+BC. P is inside ABCD,
        and dist(P,CD)=h, AP=h+AD, BP=h+BC. Show that
        1/sqrt(h)>=1/sqrt(AD)+1/sqrt(BC)."""
        ...

    @mark(mark.geom)
    def problem_IMO_1991_p1_lb(self):
        return (b+c)/(a+b+c)*(c+a)/(a+b+c)*(a+b)/(a+b+c) - Rational(1,4),\
            [a,b,c,b+c-a,c+a-b,a+b-c], []

    @mark(mark.geom)
    def problem_IMO_1991_p1_ub(self):
        return Rational(8,27) - ((b+c)/(a+b+c)*(c+a)/(a+b+c)*(a+b)/(a+b+c)),\
            [a,b,c,b+c-a,c+a-b,a+b-c], []

    @mark(mark.skip, mark.geom)
    def problem_IMO_1991_p5(self):
        return Rational(1,2) - Min(sin(x),sin(y),sin(z)),\
            [sin(x), sin(y), sin(z), sin(a-x), sin(b-y), sin(c-z), sin(a), sin(b), sin(c)],\
            [sin(x)*sin(y)*sin(z) - sin(a-x)*sin(b-y)*sin(c-z), a+b+c-pi]

    @mark(mark.noimpl, mark.nvars, mark.geom)
    def problem_IMO_1992_p5(self):
        ...

    @mark(mark.skip, mark.geom)
    def problem_IMO_1993_p4(self):
        # Matrix(5,5,[0,1,1,1,1,1,0,c**2,b**2,x**2,1,c**2,0,a**2,y**2,1,b**2,a**2,0,z**2,1,z**2,y**2,x**2,0]).det()
        cayley_menger = -a**4*x**2 - a**4*z**2 - 2*a**2*b**2*c**2 + a**2*b**2*x**2 + 2*a**2*b**2*y**2 \
            + a**2*b**2*z**2 + 2*a**2*c**2*x**2 + 2*a**2*c**2*z**2 + a**2*x**4 - 2*a**2*x**2*z**2 \
            + a**2*z**4 - 2*b**4*y**2 + b**2*c**2*x**2 + 2*b**2*c**2*y**2 + b**2*c**2*z**2 - b**2*x**4 \
            + 2*b**2*x**2*y**2 - 2*b**2*y**4 + 2*b**2*y**2*z**2 - b**2*z**4 - c**4*x**2 - c**4*z**2 \
            + c**2*x**4 - 2*c**2*x**2*z**2 + c**2*z**4
        area = lambda a1,b1,c1: ((a1+b1+c1)*(a1+b1-c1)*(c1+a1-b1)*(b1+c1-a1))**Rational(1,2)/4
        m_ =  lambda a1,b1,c1: area(a1,b1,c1) * 2 / Max(a1,b1,c1)
        return m_(a,y,z) + m_(b,z,x) + m_(c,x,y) - m_(a,b,c), [a,b,c,x,y,z], [cayley_menger]

    @mark(mark.noimpl, mark.nvars, mark.quant)
    def problem_IMO_1994_p1(self):
        ...

    def problem_IMO_1995_p2(self):
        return 1/(a**3*(b+c)) + 1/(b**3*(c+a)) + 1/(c**3*(a+b)) - Rational(3,2), [a,b,c], [a*b*c-1]

    @mark(mark.skip, mark.nvars)
    def problem_IMO_1995_p4(self):
        xi = symbols('x0:1996')
        return 2**997 - xi[0], xi, [xi[i-1]+2/xi[i-1]-2*xi[i]-1/xi[i] for i in range(1,1996)]

    @mark(mark.noimpl, mark.geom)
    def problem_IMO_1995_p5(self):
        ...

    @mark(mark.noimpl, mark.geom)
    def problem_IMO_1996_p5(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_IMO_1997_p3(self):
        ...

    @mark(mark.noimpl, mark.geom)
    def problem_IMO_1998_p5(self):
        """Proving an angle is acute."""
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_IMO_1999_p2(self):
        """Given n>=2, find smallest c such that
        c(sum(xi))^4 - sum_{i!=j}(xi*xj*(xi^2+xj^2)) holds for xi >= 0."""
        ...

    def problem_IMO_2000_p2(self):
        return 1 - (a-1+1/b)*(b-1+1/c)*(c-1+1/a), [a,b,c], [a*b*c-1]

    @mark(mark.skip)
    def problem_IMO_2001_p2(self):
        return a/sqrt(a**2+8*b*c)+b/sqrt(b**2+8*c*a)+c/sqrt(c**2+8*a*b)-1, [a,b,c], []

    @mark(mark.noimpl, mark.nvars, mark.geom)
    def problem_IMO_2002_p6(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_IMO_2003_p5(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_IMO_2004_p4(self):
        ...

    @mark(mark.skip)
    def problem_IMO_2005_p3(self):
        return (x**5-x**2)/(x**5+y**2+z**2)+(y**5-y**2)/(y**5+z**2+x**2)\
            +(z**5-z**2)/(z**5+x**2+y**2), [x,y,z,x*y*z-1], []

    @mark(mark.noimpl, mark.geom)
    def problem_IMO_2006_p1(self):
        ...

    def problem_IMO_2006_p3(self):
        return 9*sqrt(2)/32*(a**2+b**2+c**2)**2 - \
            Abs(a*b*(a**2-b**2)+b*c*(b**2-c**2)+c*a*(c**2-a**2)), [], []

    @mark(mark.noimpl, mark.nvars, mark.geom)
    def problem_IMO_2006_p6(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_IMO_2007_p1(self):
        ...

    def problem_IMO_2008_p2(self):
        return x**2/(x-1)**2+y**2/(y-1)**2+z**2/(z-1)**2-1, [x,y,z], [x*y*z-1]

    @mark(mark.noimpl, mark.nvars)
    def problem_IMO_2012_p2(self):
        ...

    @mark(mark.skip)
    def problem_IMO_2020_p2(self):
        # transcendental
        return 1-(a+2*b+3*c+4*d)*a**a*b**b*c**c*d**d, [a-b,b-c,c-d,d], [a+b+c+d-1]

    @mark(mark.noimpl, mark.nvars)
    def problem_IMO_2021_p2(self):
        ...


class IMOSLProblems(ProblemSet):
    @mark(mark.noimpl, mark.nvars)
    def problem_IMOSLA_2003_p6(self):
        ...

    @mark(mark.skip)
    def problem_IMOSLA_2004_p5(self):
        return 1/(a*b*c)-(cbrt(1/a+6*b)+cbrt(1/b+6*c)+cbrt(1/c+6*a)), [a,b,c], [a*b+b*c+c*a-1]

    @mark(mark.noimpl, mark.nvars)
    def problem_IMOSLA_2004_p7(self):
        ...

    def problem_IMOSLA_2005_p3(self):
        return p*q-r*s-2, [p-q,q-r,r-s], [p+q+r+s-9, p**2+q**2+r**2+s**2-21]

    @mark(mark.noimpl, mark.nvars)
    def problem_IMOSLA_2006_p2(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_IMOSLA_2006_p4(self):
        ...

    @mark(mark.skip)
    def problem_IMOSLA_2006_p5(self):
        return 3-(sqrt(b+c-a)/(sqrt(b)+sqrt(c)-sqrt(a))+\
                sqrt(c+a-b)/(sqrt(c)+sqrt(a)-sqrt(b))+\
                sqrt(a+b-c)/(sqrt(a)+sqrt(b)-sqrt(c))), [a,b,c,a+b-c,b+c-a,c+a-b], []

    @mark(mark.noimpl, mark.nvars)
    
    @mark(mark.noimpl, mark.nvars)
    def problem_IMOSLA_2007_p2(self):
        """x^n+y^n=1, x,y>=0, n is positive integer. Show that ..."""
        ...

    @mark(mark.skip, mark.nvars)
    def problem_IMOSLA_2007_p6(self):
        ai = symbols('a1:101')
        return Rational(12,25) - Add(*(ai[i]**2*ai[(i+1)%100] for i in range(100))),\
            ai, [Add(*[aii**2 for aii in ai]) - 1]

    def problem_IMOSLA_2008_p5(self):
        return b/a+c/b+d/c+a/d-a-b-c-d, [a,b,c,d,a+b+c+d-a/b-b/c-c/d-d/a], [a*b*c*d-1]

    def problem_IMOSLA_2008_p7(self):
        return (a-b)*(a-c)/(a+b+c)+(b-c)*(b-d)/(b+c+d)+(c-d)*(c-a)/(c+d+a)+(d-a)*(d-b)/(d+c+b),\
            [a,b,c,d], []

    def problem_IMOSLA_2009_p2(self):
        return Rational(3,16)-1/(2*a+b+c)**2-1/(2*b+c+a)**2-1/(2*c+a+b)**2,\
            [a,b,c], [1/a+1/b+1/c-a-b-c]

    @mark(mark.skip)
    def problem_IMOSLA_2009_p4(self):
        return sqrt(2)*(sqrt(a+b)+sqrt(b+c)+sqrt(c+a))-3-\
            (sqrt((a**2+b**2)/(a+b))+sqrt((b**2+c**2)/(b+c))+sqrt((c**2+a**2)/(c+a))),\
                [a,b,c, 3*a*b*c-a*b-b*c-c*a], []

    def problem_IMOSLA_2010_p2_lb(self):
        nt = lambda n_: a**n_+b**n_+c**n_+d**n_
        return 4*nt(3)-nt(4)-36, [], [nt(1)-6, nt(2)-12]

    def problem_IMOSLA_2010_p2_ub(self):
        nt = lambda n_: a**n_+b**n_+c**n_+d**n_
        return 48-(4*nt(3)-nt(4)), [], [nt(1)-6, nt(2)-12]

    @mark(mark.skip, mark.nvars)
    def problem_IMOSLA_2010_p3(self):
        xi = symbols('x1:101')
        return Rational(25,2) - Add(*(xi[i]*xi[(i+2)%100] for i in range(100))),\
            [1-xi[i]-xi[(i+1)%100]-xi[(i+2)%100] for i in range(100)], []

    @mark(mark.skip)
    def problem_IMOSLA_2010_p8(self):
        S = a+c+e
        T = b+d+f
        return 2*S*T - sqrt(3*(S+T)*(S*(b*d+b*f+d*f) + T*(a*c+a*e+c*e))),\
            [f-e,e-d,d-c,c-b,b-a,a], []

    def problem_IMOSLA_2011_p7(self):
        return a/(b+c-a)**2+b/(c+a-b)**2+c/(a+b-c)**2-3/(a*b*c)**2,\
            [a,b,c, Min(a+b,b+c,c+a)-sqrt(2)], [a**2+b**2+c**2-3]

    @mark(mark.noimpl, mark.recur)
    def problem_IMOSLA_2013_p4(self):
        ...
    
    @mark(mark.noimpl, mark.recur)
    def problem_IMOSLA_2015_p1(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_IMOSLA_2015_p3(self):
        ...

    def problem_IMOSLA_2016_p1(self):
        return 1 + ((a+b+c)/3)**2 - cbrt((a**2+1)*(b**2+1)*(c**2+1)),\
            [a,b,c,Min(a*b,b*c,c*a)-1]

    @mark(mark.skip)
    def problem_IMOSLA_2016_p2(self):
        a1, a2, a3, a4, a5 = symbols('a1:6')
        ff = lambda x: Rational(1,2) - Abs(x)
        return Max(ff(a1/a2-a3/a4), ff(a2/a3-a4/a5), ff(a1/a4-a2/a5)),\
            [a5-a4,a4-a3,a3-a2,a2-a1,a1], []

    @mark(mark.noimpl, mark.nvars)
    def problem_IMOSLA_2016_p3(self):
        """For odd n>=3, given real numbers |ak|+|bk|=1,
        show that there exists e1,...,en in {-1,1} such that
        |sum(ek*ak)| + |sum(ek*bk)| <= 1."""
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_IMOSLA_2016_p8(self):
        """Let 0=x0<x1<...<xn, show that sum(1/(x_i-x_{i-1})) >= 4/9*sum((i+1)/x_i)."""
        ...

    @mark(mark.noimpl, mark.nvars)    
    def problem_IMOSLA_2017_p1(self):
        """Let 1/a1+..+1/an=k, a1,...,an>=1, x>0, show
        that a1*...*an*(x+1)^k - (x+a1)*...*(x+an) <= 0."""
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_IMOSLA_2017_p5(self):
        ...

    @mark(mark.noimpl, mark.nvars, mark.recur)
    def problem_IMOSLA_2017_p7(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_IMOSLA_2018_p4(self):
        ...

        
    def problem_IMOSLA_2018_p7(self):
        return 8/cbrt(7) - (cbrt(a/(b+7))+cbrt(b/(c+7))+cbrt(c/(d+7))+cbrt(d/(a+7))),\
            [a,b,c,d], [a+b+c+d-100]

    @mark(mark.skip, mark.nvars)
    def problem_IMOSLA_2019_p2(self):
        ui = symbols('u1:2020')
        a_, b_ = Min(*ui), Max(*ui)
        return Rational(-1,2019) - a_*b_, [], [Add(*ui), Add(*[_**2 for _ in ui]) - 1]

    @mark(mark.noimpl, mark.nvars)
    def problem_IMOSLA_2019_p4(self):
        ...

    @mark(mark.skip)
    def problem_IMOSLA_2020_p1(self):
        N = symbols('N')
        return N/2*(x-1)**2 + x - ((x**(2*N) + 1)/2)**(1/N), [N-1], []

    def problem_IMOSLA_2020_p3(self):
        return a/b+b/c+c/d+d/a-8, [a,b,c,d], [(a+c)*(b+d)-a*c-b*d]

    @mark(mark.noimpl, mark.nvars)
    def problem_IMOSLA_2020_p7(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_IMOSLA_2021_p3(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_IMOSLA_2021_p5(self):
        """n>=2, a1+...+an=1, ak>=0, show sum(ak/(1-ak)*(a1+..+a_{k-1})^2) < 1/3."""
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_IMOSLA_2021_p7(self):
        """n>=1, x0,...x_{n+1}>=0, x_{i}*x_{i+1}-x_{i-1}^2>=1 for i=1,...,n,
        show that x0+...+x_{n+1} > (2*n/3)^(3/2)."""
        ...

    def problem_IMOSLA_2022_p1(self):
        con = lambda x,y,z: x + z - x*z - y**2
        # a,b,c,d,e = symbols('a2020:2025')
        return 1-c, [a,b,c,d,e, con(a,b,c), con(b,c,d), con(c,d,e)], []

    @mark(mark.noimpl, mark.nvars)
    def problem_IMOSLA_2022_p4(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_IMOSLA_2023_p5(self):
        ...

    @mark(mark.noimpl, mark.nvars)
    def problem_IMOSLA_2023_p7(self):
        ...

    