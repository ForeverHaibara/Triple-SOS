# author: https://github.com/ForeverHaibara 

#import imp
from sum_of_square import *

# dependencies:
# pip install sympy
# pip install scipy
# pip install numpy

example = 2

if example == 1:
    # Example 1
    #----------------------------------------------------------
    # be careful with '\\' and '\'
    s = r'$$\frac{s(a2(a-b)(4b+c))}{2}$$'

    print(PreprocessText(s,retText=True))

    print(PreprocessText(s))

    print('-'*30 + 'SOS' + '-'*30)

    result = SOS(s,[],maxiter=5000,roots=[],tangent_points=[],
                updeg=7,precision=8,show_tangents=True)

else:
    # Example 2
    #----------------------------------------------------------
    
    s = 's(ab(a-b)4)-8p(a)s(a3-2a2b-2a2c+3p(a))'

    print(PreprocessText(s,retText=True))

    print(PreprocessText(s))

    print('-'*30 + 'SOS' + '-'*30)

    # specify a tangent here !
    result = SOS(s,tangents=['(a+b-c)2-3ab'],maxiter=5000,roots=[],tangent_points=[],
                updeg=7,precision=8,show_tangents=True)