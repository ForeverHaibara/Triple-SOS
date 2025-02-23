# Triples

Triples 是由 forever豪3 开发的基于 Python 的 SymPy 库的多项式不等式自动证明程序。其专注于通过配方法生成不等式的可读证明。它同时提供直观的图形界面与代码接口，助力代数不等式的探索。

Triples is an automatic inequality proving software developed by ForeverHaibara, based on the Python SymPy library. It focuses on generating readable proofs of inequalities through sum of squares (SOS). The program offers both a graphical user interface and a code interface to facilitate the exploration of Olympiad-level algebraic inequalities.


在线体验 Online Servers:

* **Hugging Face**      [https://huggingface.co/spaces/ForeverHaibara/Ternary-Inequality-Prover](https://huggingface.co/spaces/ForeverHaibara/Ternary-Inequality-Prover)
* **AIStudio**               [https://aistudio.baidu.com/application/detail/37245](https://aistudio.baidu.com/application/detail/37245)
* **AIStudio Backup** [https://aistudio.baidu.com/application/detail/13542](https://aistudio.baidu.com/application/detail/13542)

## 快速开始 Quick Start

本程序图形化界面有两种启动方式。一种是 Flask，另一种是 Gradio。

Two graphical backends are supported. One is Flask and the other is Gradio.

### Flask 启动

1. 安装依赖: Install Dependencies

```
pip install sympy
pip install numpy
pip install scipy
pip install picos
pip install flask
pip install flask_cors
pip install flask_socketio
```

2. 控制台中运行 `python web_main.py` 启动后端。 Run `python web_main.py` to launch the backend.
3. 浏览器打开 `triples.html` 即可使用。 Open `triples.html` in your browser.

### Gradio 启动

1. 安装依赖: Install Dependencies

   注意：目前 gradio 3.44.4 是已知唯一正常支持 LaTeX 的版本。Warning: gradio 3.44.4 is the only known version that supports LaTeX display with pretty line breaks.

```
pip install sympy
pip install numpy
pip install scipy
pip install cvxopt
pip install gradio==3.44.4
pip install pillow
```

2. 控制台中运行 `python gradio.app.py` 启动后端。 Run `python gradio.app.py` to launch the backend.
3. 浏览器打开控制台中显示的地址。 Open the link displayed in the console using the browser.

输入关于 a,b,c 的齐次式。注: 幂符号 ^ 可以省略，函数 s 与 p 分别表示轮换和与轮换积，例如 `s(a2)` 表示 `a^2+b^2+c^2`。

Input a homogeneous polynomial with respect to variables a,b,c. The exponential symbol ^ can be omitted. Functions s(...) and p(...) stand for cyclic sum and cyclic product, respectively. For instance, inputting `s(a2)` means `a^2+b^2+c^2`.

![image](notebooks/triple_sos_example.png)

修改左下角的 Generators 可以可视化四元齐次多项式。

Configure the Generators in the bottom left corner to visualize quaternary homogeneous polynomials.

![image](notebooks/triple_sos_example2.png)

## 代码调用 Code Usage

Note: Flask or Gradio is not required for code usage.


```py
>>> from triples.core import sum_of_squares
>>> import sympy as sp
>>> a, b, c, x, y, z = sp.symbols("a b c x y z")
```

### Sum of Squares

Given a sympy polynomial, the `sum_of_squares` solver  will return a Solution-class object if it succeeds. It returns None if it fails (but it does not mean the polynomial is not a sum of square or positive semidefinite).
 

**Example 1** $a,b,c\in\mathbb{R}$, prove:  $\left(a^2+b^2+c^2\right)^2\geq 3\left(a^3b+b^3c+c^3a\right)$.
```py
>>> sol = sum_of_squares(((a**2 + b**2 + c**2)**2 - 3*(a**3*b + b**3*c + c**3*a)).as_poly(a, b, c))
>>> sol.solution # this should be a sympy expression
(Σ(a**2 - a*b - a*c - b**2 + 2*b*c)**2)/2
>>> sol.solution.doit() # this expands the cyclic sums
(-a**2 + 2*a*b - a*c - b*c + c**2)**2/2 + (a**2 - a*b - a*c - b**2 + 2*b*c)**2/2 + (-a*b + 2*a*c + b**2 - b*c - c**2)**2/2
```

<br>


If there are inequality or equality constraints, send them as a list of sympy expressions to `ineq_constraints` and `eq_constraints`.

**Example 2** $a,b,c\in\mathbb{R}_+$, prove:  $a(a-b)(a-c)+b(b-c)(b-a)+c(c-a)(c-b)\geq 0$.
```py
>>> sol = sum_of_squares((a*(a-b)*(a-c) + b*(b-c)*(b-a) + c*(c-a)*(c-b)).as_poly(a, b, c), ineq_constraints = [a,b,c])
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



## 讨论交流 Discussions

配方器的主要函数在 `triples.core.sum_of_squares.sum_of_squares`。配方器基于多种算法尝试配方，但其无法保证 100% 配出，程序还在不断改进，三元齐次不等式的研究进一步交流可加入 QQ 群 875413273。

The main function of the sum-of-square solver is `triples.core.sum_of_squares.sum_of_squares`. The solver uses a mixed strategy. It does not guarantee to solve all problems.

## 算法 Algorithms

### StructuralSOS

StructuralSOS 核心思想与系数阵紧密相关，可以解决已知的构型的不等式，例如四次以下的三元齐次轮换不等式。

StructuralSOS solves inequality in known forms. For example, ternary homogeneous cyclic inequalities with degree no greater than four are completely solved.

### LinearSOS

LinearSOS 待定多个基，利用线性规划求得一组非负系数，可以自动升次。

LinearSOS constructs several bases and find a group of nonnegative coefficients by linear programming. It will automatically lift the degree of the polynomial.

### SymmetricSOS

SymmetricSOS 针对三元齐次对称不等式，利用特殊换元解决问题。

SymmetricSOS is designed for ternary homogeneous symmetric inequality. It uses a special substitution to solve the problems.

### SDPSOS

SDPSOS 将问题转化为低秩有理的SDP问题求解。暂时不支持自动升次，一些多项式无法不升次地表示为平方和。

SDPSOS transforms the problem into a low-rank rational SDP problem. It does not support automatic degree lifting in the current, and some polynomials cannot be represented as sum of squares without lifting the degree.