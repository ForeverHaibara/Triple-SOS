Triple_SOS机器配方程序安装说明
该说明文档大致可以帮助不会用python的人使用上Triple_SOS

作者：forever豪3
时间：20230608
系数阵交流群：875413273
GitHub: https://github.com/ForeverHaibara/Triple-SOS 
测试用系统：Windows 10 64位

安装方式: 
分为两种情况 A, B:

A. 我已经安装过原先非网页版的配方器

    打开 cmd, 输入:
    pip install flask 
    pip install flask_cors
    
B. 我没有安装过原先的配方器
    安装 python-3.7 

    打开 cmd, 依次输入：
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

    pip install sympy
    pip install numpy
    pip install scipy
    pip install matplotlib
    pip install flask
    pip install flask_cors


运行方式:
    每次运行都要做以下两步, 顺序不限:
    1. 右键文件夹里的 web_main.py 打开方式选择 python, 出现一个黑框不要关闭(可以最小化)
    2. 双击打开文件夹里的 triple.html (提示:可以在浏览器里收藏这个网页哦~)