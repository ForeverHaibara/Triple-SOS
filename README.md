# Triple-SOS

Triple-SOS 是由 forever豪3 开发的开源且**具备图形化界面**的自动**多项式**不等式配方器。

Triple-SOS is an open-source symbolic polynomial sum-of-square solver with Graphic User Interface, developed by ForeverHaibara.

在线体验 Online Server:

* **Hugging Face**      [https://huggingface.co/spaces/ForeverHaibara/Ternary-Inequality-Prover](https://huggingface.co/spaces/ForeverHaibara/Ternary-Inequality-Prover)
* **AIStudio**               [https://aistudio.baidu.com/application/detail/37245](https://aistudio.baidu.com/application/detail/37245)
* **AIStudio Backup** [https://aistudio.baidu.com/application/detail/13542](https://aistudio.baidu.com/application/detail/13542)

## 快速开始 Quick Start

本程序有两种启动方式。一种是 Flask，另一种是 Gradio。

Two versions of backend are supported. One is Flask and the other is Gradio.

### Flask 启动

1. 安装依赖: Install Dependencies

```
pip install sympy
pip install numpy
pip install scipy
pip install flask
pip install flask_cors
pip install flask_socketio
pip install picos
```

2. 控制台中运行 `python web_main.py` 启动后端。 Run `python web_main.py` to launch the backend.
3. 浏览器打开 `triples.html` 即可使用。 Open `triples.html` in your browser.

### Gradio 启动

1. 安装依赖: Install Dependencies

   注意：目前 gradio 3.44.4 是已知唯一正常支持 LaTeX 的版本。Warning: currently gradio 3.44.4 is the only known version that supports LaTeX display.

```
pip install sympy
pip install numpy
pip install scipy
pip install gradio==3.44.4
pip install pillow
pip install cvxopt
```

2. 控制台中运行 `python gradio.app.py` 启动后端。 Run `python gradio.app.py` to launch the backend.
3. 浏览器打开控制台中显示的地址。 Open the link displayed in the console using the browser.

输入关于 a,b,c 的齐次轮换式。注: 幂符号 ^ 可以省略，函数 s 与 p 分别表示轮换和与轮换积，例如 `s(a2)` 表示 `a^2+b^2+c^2`。

Input a cyclic and homogeneous polynomial with respect to variables a,b,c. The exponential symbol ^ can be omitted. Function s(...) and p(...) stands for cyclic sum and cyclic product. For instance, inputting `s(a2)` means `a^2+b^2+c^2`.

![image](notebooks/triple_sos_example.png)

## 代码调用 Code Usage


```py
>>> from src.core import sum_of_square
>>> import sympy as sp
>>> a, b, c, x, y, z = sp.symbols("a b c x y z")
```

### Sum of Square

Given a polynomial, the sum-of-square solver will return a sympy expression of the sum of squares if it succeeds. It returns None if it fails (but it does not mean the polynomial is not a sum of square or positive semidefinite).
 

**Example 1** $a,b,c\in\mathbb{R}$, prove:  $\left(a^2+b^2+c^2\right)^2\geq 3\left(a^3b+b^3c+c^3a\right)$.
```py
>>> sol = sum_of_square(((a**2 + b**2 + c**2)**2 - 3*(a**3*b + b**3*c + c**3*a)).as_poly(a, b, c))
>>> sol.solution # this should be a sympy expression
(Σ(a**2 - a*b - a*c - b**2 + 2*b*c)**2)/2
>>> sol.solution.doit() # this expands the cyclic sums
(-a**2 + 2*a*b - a*c - b*c + c**2)**2/2 + (a**2 - a*b - a*c - b**2 + 2*b*c)**2/2 + (-a*b + 2*a*c + b**2 - b*c - c**2)**2/2
```

<br>


If there are inequality or equality constraints, send them as a list of sympy expressions to `ineq_constraints` and `eq_constraints`.

**Example 2** $a,b,c\in\mathbb{R}_+$, prove:  $a(a-b)(a-c)+b(b-c)(b-a)+c(c-a)(c-b)\geq 0$.
```py
>>> sol = sum_of_square((a*(a-b)*(a-c) + b*(b-c)*(b-a) + c*(c-a)*(c-b)).as_poly(a, b, c), ineq_constraints = [a,b,c])
>>> sol.solution
((Σ(a - b)**2*(a + b - c)**2)/2 + Σa*b*(a - b)**2)/(Σa)
```

<br>

If you want to track the inequality and equality constraints, you can send in a dict containing the alias of the constraints.

**Example 3** $a,b,c\in\mathbb{R}_+$ and $abc=1$, prove: $\sum \frac{a^2}{2+a}\geq 1$.
```py
>>> sol = sum_of_square(((a+2)*(b+2)*(c+2)*(a**2/(2+a)+b**2/(2+b)+c**2/(2+c)-1)).cancel(), ineq_constraints=[a,b,c], eq_constraints={a*b*c-1:x})
>>> sol.solution
x*(Σa)/3 + 13*x + Σa*(b - c)**2 + (Σa*b*(c - 1)**2)/6 + 5*(Σ(a - 1)**2)/6 + 7*(Σ(a - b)**2)/12
>>> sol.solution.doit()
a*b*(c - 1)**2/3 + a*c*(b - 1)**2/3 + a*(-b + c)**2 + a*(b - c)**2 + b*c*(a - 1)**2/3 + b*(-a + c)**2 + b*(a - c)**2 + c*(-a + b)**2 + c*(a - b)**2 + x*(2*a + 2*b + 2*c)/3 + 13*x + 7*(-a + b)**2/12 + 7*(-a + c)**2/12 + 5*(a - 1)**2/3 + 7*(a - b)**2/12 + 7*(a - c)**2/12 + 7*(-b + c)**2/12 + 5*(b - 1)**2/3 + 7*(b - c)**2/12 + 5*(c - 1)**2/3

>>> F = sp.Function("F")
>>> sol = sum_of_square(((a+2)*(b+2)*(c+2)*(a**2/(2+a)+b**2/(2+b)+c**2/(2+c)-1)).cancel(), {a: F(a), b: F(b), c: F(c)}, {a*b*c-1:x})
>>> sol.solution
x*(ΣF(a))/3 + 13*x + Σ(a - b)**2*F(c) + (Σ(a - 1)**2*F(b)*F(c))/6 + 5*(Σ(a - 1)**2)/6 + 7*(Σ(a - b)**2)/12
```



## 讨论交流 Discussion

配方器的主要函数在 `src.core.sum_of_square.sum_of_square`。配方器基于多种算法尝试配方，核心思想与系数阵紧密相关。但其无法保证 100% 配出，程序还在不断改进，三元齐次不等式的研究也在不断发展。进一步交流可加入 QQ 群 875413273。

The main function of the sum-of-square solver is `src.core.sum_of_square.sum_of_square`. The solver uses a mixed strategy with tight relationship of coefficient triangle. Still it does not guarantee to solve all problems.

## 算法 Algorithm

### StructuralSOS

StructuralSOS 可以解决已知的构型的不等式，例如四次以下的三元齐次轮换不等式。

StructuralSOS solves inequality in known forms. For example, ternary homogeneous cyclic inequalities with degree no greater than four are completely solved.

### LinearSOS

LinearSOS 待定多个基，利用线性规划求得一组非负系数，可以自动升次。

LinearSOS constructs several bases and find a group of nonnegative coefficients by linear programming. It will automatically lift the degree of the polynomial.

### SymmetricSOS

SymmetricSOS 针对三元齐次对称不等式，利用特殊换元，可以解决一部分取等在对称轴或边界的对称不等式问题。

SymmetricSOS is designed for ternary homogeneous symmetric inequality. It uses special substitution to solve inequalities with equality cases on the symmetry axis or boundary.

### SDPSOS

SDPSOS 先找到多选式的零点，并将问题转化为低秩有理的SDP问题求解。适合取等非常多的问题。暂时不支持自动升次，一些多项式无法不升次地表示为平方和。

SDPSOS first finds the roots of the polynomial and then transforms the problem into a low-rank rational SDP problem. It is suitable for problems with many equality cases. Currently it does not support automatic degree lifting, and some polynomials cannot be represented as sum of squares without lifting the degree.