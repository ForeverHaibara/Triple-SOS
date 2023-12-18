# Triple-SOS

Triple-SOS 是由 forever豪3 开发的开源且**具备图形化界面**的自动**三元齐次轮换**不等式配方器。 

Triple-SOS is an open-source sum-of-square solver with GUI for three-variable cyclic homogeneous polynomials, developed by ForeverHaibara.

在线体验 Online Server:
https://aistudio.baidu.com/application/detail/13542 

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
pip install picos
```

2. 控制台中运行 `python web_main.py` 启动后端。 Run `python web_main.py` to launch the backend.

3. 浏览器打开 `triples.html` 即可使用。 Open `triples.html` in your browser.

### Gradio 启动

1. 安装依赖: Install Dependencies
   
   注意：目前 gradio==3.44.4 是已知唯一正常支持 LaTeX 的版本。Warning: currently gradio==3.44.4 is the only known version that supports LaTeX display.
```
pip install sympy
pip install numpy
pip install scipy
pip install gradio==3.44.4
pip install pillow
pip install picos
```

2.  控制台中运行 `python main.gradio.py` 启动后端。 Run `python main.gradio.py` to launch the backend.

3.  浏览器打开控制台中显示的地址。 Open the link displayed in the console using the browser.


输入关于 a,b,c 的齐次轮换式。注: 幂符号 ^ 可以省略，函数 s 与 p 分别表示轮换和与轮换积，例如 s(a2) 表示 a^2+b^2+c^2。

Input a cyclic and homogeneous polynomial with respect to variables a,b,c. The exponential symbol ^ can be omitted. Function s(...) and p(...) stands for cyclic sum and cyclic product. For instance, inputting s(a2) means a^2+b^2+c^2.

![image](https://raw.githubusercontent.com/ForeverHaibara/Triple-SOS/main/notebooks/triple_sos_example.png)

## 讨论交流 Discussion

配方器的主要函数在 `src.core.sum_of_square.sum_of_square`。配方器基于多种算法尝试配方，核心思想与系数阵紧密相关。但其无法保证 100% 配出，程序还在不断改进，三元齐次不等式的研究也在不断发展。进一步交流可加入 QQ 群 875413273。

The main function of the sum-of-square solver is `src.core.sum_of_square.sum_of_square`. The solver uses a mixed strategy with tight relationship of coefficient triangle. Still it does not guarantee to solve all problems.

## 算法 Algorithm

### StructuralSOS

StructuralSOS 可以解决已知的构型的不等式，例如四次以下的不等式。

StructuralSOS solves inequality in known forms. For example, cyclic, homogeneous inequality with degree no greater than four is completely solved.

### LinearSOS

LinearSOS 待定多个基，利用线性规划求得一组非负系数。暂时不支持对实数的配方。可以自动升四次。

LinearSOS construct several basis and find a group of nonnegative coefficients by linear programming. It currently does not support sum-of-square for real numbers. It will automatically higher the degree by at most four degrees.

### SymmetricSOS
  
SymmetricSOS 利用特殊换元，可以解决一部分取等在对称轴或边界的对称不等式问题。支持对实数的配方。

SymmetricSOS uses a special change of variables. It can solve some inequalities with equality cases on the symmetric axis or on the border. It supports sum-of-square for real numbers.

### SDPSOS
  
SDPSOS 先找到多选式的零点，并将问题转化为低秩有理的SDP问题求解。适合取等非常多的问题。支持对实数的配方。暂时不能自动升次，可以手动乘 s(a), s(a2-ab), s(a2) 等尝试。

SDPSOS finds the root of the polynomial first and converts the problem into low-rank rational SDP. It works well when there are many equality cases. It supports sum-of-square for real numbers. It currently does not higher the degree automatically. It is suggested to multiply s(a), s(a2-ab), s(a2) to take a try if no direct solution is available.
