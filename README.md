# Triple-SOS

Triple-SOS 是由 forever豪3 开发的开源且**具备图形化界面**的自动**三元齐次轮换**不等式配方器。


## 快速开始

1. 安装依赖: 
```
pip install sympy
pip install numpy
pip install scipy
pip install matplotlib
pip install flask
pip install flask_cors
```

2. 控制台中运行 `python web_main.py` 启动后端。

3. 浏览器打开 `triples.html` 即可使用。

输入关于 a,b,c 的齐次轮换式。注: 幂符号 ^ 可以省略，函数 s 与 p 分别表示轮换和与轮换积，例如 s(a2) 表示 a^2+b^2+c^2。

![image](https://github.com/ForeverHaibara/Triple-SOS/blob/main/notebooks/triple_sos_example.png?raw=true)

## 讨论交流

配方器基于多种算法尝试配方，核心思想与系数阵紧密相关。但其无法保证 100% 配出，程序还在不断改进，三元齐次不等式的研究也在不断发展。进一步交流可加入 QQ 群 875413273。
