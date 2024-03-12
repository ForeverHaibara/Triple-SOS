Triple_SOS机器配方程序安装说明
该说明文档大致可以帮助不会用python的人使用上Triple_SOS

作者：forever豪3
时间：2024年3月8日
系数阵交流群：875413273
GitHub: https://github.com/ForeverHaibara/Triple-SOS 
测试用系统：Windows 10 64位

安装方式: 
分为两种情况 A, B:

A. 我已经安装过原先网页版的配方器

    将新的文件替换掉原来的即可。
    注1：如果是浏览器收藏了以前的triples.html，可能要重新收藏新的triples.html。
    注2：可以打开cmd，输入下面一行，可以稍微加速SDPSOS。
    pip install gmpy2
    
    
B. 我没有安装过原先的配方器
    安装 python-3.7 

    打开 cmd, 依次输入：
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

    pip install sympy
    pip install numpy
    pip install scipy
    pip install flask
    pip install flask_cors
    pip install picos


运行方式:
    每次运行都要做以下两步, 顺序不限:
    1. 右键文件夹里的 web_main.py 打开方式选择 python, 出现一个黑框不要关闭(可以最小化)
    2. 双击打开文件夹里的 triples.html (提示:可以在浏览器里收藏这个网页哦~)

更新：
    1. 可以配非轮换的三元不等式了，虽然可能比较弱。
    2. 大大加强了线性规划LinearSOS。
    3. 修改了很多底层代码，比如说SDPSOS。（其实利用代码可以调用SDPSOS配多元不等式了。）
    4. 如果还安装了gmpy2：pip install gmpy2可以加速一下SDPSOS。
