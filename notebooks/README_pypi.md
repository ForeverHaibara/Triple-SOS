# Triples

Triples is an automatic inequality proving software developed by ForeverHaibara, based on the Python SymPy library. It focuses on generating readable proofs of inequalities through sum of squares (SOS). The program offers both a graphical user interface and a code interface to facilitate the exploration of Olympiad-level algebraic inequalities.

Homepage: [https://github.com/ForeverHaibara/Triple-SOS](https://github.com/ForeverHaibara/Triple-SOS)

To install, use:
```
pip install triples
```

## Code Usage

```py
from triples import sum_of_squares
import sympy as sp
a, b, c, x, y, z = sp.symbols("a b c x y z")
```

### Sum of Squares

Given a sympy polynomial, the `sum_of_squares` solver  will return a Solution-class object if it succeeds. It returns None if it fails (but it does not mean the polynomial is not a sum of squares or positive semidefinite).

**Example 1** $a,b,c\in\mathbb{R}$, prove:  $\left(a^2+b^2+c^2\right)^2\geq 3\left(a^3b+b^3c+c^3a\right)$.

```py
>>> sol = sum_of_squares((a**2 + b**2 + c**2)**2 - 3*(a**3*b + b**3*c + c**3*a))
>>> sol.solution # this should be a sympy expression
(Σ(a**2 - a*b - a*c - b**2 + 2*b*c)**2)/2
>>> sol.solution.doit() # this expands the cyclic sums
(-a**2 + 2*a*b - a*c - b*c + c**2)**2/2 + (a**2 - a*b - a*c - b**2 + 2*b*c)**2/2 + (-a*b + 2*a*c + b**2 - b*c - c**2)**2/2
```

<br>

If there are inequality or equality constraints, send them as a list of sympy expressions to `ineq_constraints` and `eq_constraints`.

**Example 2** $a,b,c\in\mathbb{R}_+$, prove:  $a(a-b)(a-c)+b(b-c)(b-a)+c(c-a)(c-b)\geq 0$.

```py
>>> sol = sum_of_squares(a*(a-b)*(a-c) + b*(b-c)*(b-a) + c*(c-a)*(c-b), ineq_constraints = [a,b,c])
>>> sol.solution
((Σ(a - b)**2*(a + b - c)**2)/2 + Σa*b*(a - b)**2)/(Σa)
```

<br>

If you want to track the inequality and equality constraints, you can send in a dict containing the alias of the constraints.

**Example 3** $a,b,c\in\mathbb{R}_+$ and $abc=1$, prove: $\sum \frac{a^2}{2+a}\geq 1$.

```py
>>> sol = sum_of_squares(((a+2)*(b+2)*(c+2)*(a**2/(2+a)+b**2/(2+b)+c**2/(2+c)-1)).cancel(), ineq_constraints=[a,b,c], eq_constraints={a*b*c-1:x})
>>> sol.solution
x*(Σ(2*a + 13))/6 + Σa*(b - c)**2 + (Σa*b*(c - 1)**2)/6 + 5*(Σ(a - 1)**2)/6 + 7*(Σ(a - b)**2)/12
>>> sol.solution.doit()
a*b*(c - 1)**2/3 + a*c*(b - 1)**2/3 + a*(-b + c)**2 + a*(b - c)**2 + b*c*(a - 1)**2/3 + b*(-a + c)**2 + b*(a - c)**2 + c*(-a + b)**2 + c*(a - b)**2 + x*(4*a + 4*b + 4*c + 78)/6 + 7*(-a + b)**2/12 + 7*(-a + c)**2/12 + 5*(a - 1)**2/3 + 7*(a - b)**2/12 + 7*(a - c)**2/12 + 7*(-b + c)**2/12 + 5*(b - 1)**2/3 + 7*(b - c)**2/12 + 5*(c - 1)**2/3
>>> F = sp.Function("F")
>>> sol = sum_of_squares(((a+2)*(b+2)*(c+2)*(a**2/(2+a)+b**2/(2+b)+c**2/(2+c)-1)).cancel(), {a: F(a), b: F(b), c: F(c)}, {a*b*c-1:x})
>>> sol.solution
x*(Σ(2*F(a) + 13))/6 + Σ(a - b)**2*F(c) + (Σ(a - 1)**2*F(b)*F(c))/6 + 5*(Σ(a - 1)**2)/6 + 7*(Σ(a - b)**2)/12
```


## Warnings

* The `triples` package uses dynamic patches to fix some known bugs in the SymPy library for older versions of SymPy, and may slightly affect the behaviour of SymPy. This includes a fix for `radsimp` at [https://github.com/sympy/sympy/pull/26720](https://github.com/sympy/sympy/pull/26720) and setting `CRootOf.is_algebraic = True` at [https://github.com/sympy/sympy/pull/26806](https://github.com/sympy/sympy/pull/26806).