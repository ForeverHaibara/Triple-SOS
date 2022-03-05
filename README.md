# Triple-SOS
Automatic sum of square representation calculator.

Now it only supports cyclic, homogenous and 3-variable (a,b,c) polynomial with domain R+. For example, (a^2+b^2+c^2)^2 - 3*(a^3b+b^3c+c^3a). 

For a quick start, just run the **example.py** !

## Requirements

* numpy
* scipy
* sympy

## SOS_Manager

Packed in **sos_manager.py**, SOS_Manager is a class designed for OOP (object oriented programming). It prepares some data in the memory or cache and avoids repeated work when processing a large number of polynomials. For example, the manager halves the execution time on a 12-degree polynomial the second time, simply because it has preloaded the information necessary for 12-degree polynomials. 

Check out sos_manager.py and try! 

## graphics_main

The file **graphics_main.py** requires PySide6 to build a GUI program, with which one can easily work with GUI rather codes.
![fig1](https://user-images.githubusercontent.com/69423537/156883000-496843aa-dd68-4c4d-9462-451f84fcaea2.png)


## Algorithms

The alogrithm behind the sum of square calculator is complicated and elaborate. Instead of turning to semidefinite programming, the core of this alogrithm lies in constructing 

a^i * b^j * c^k * (a-b)^2m * (b-c)^2n * (c-a)^2p * g(a,b,c) ^ 2l

By computing suitable nonnegative coefficients that sum up to the original polynomial, one can obtain a SOS representation. This can be done by linear programming in scipy.

The additional term, g(a,b,c) in the formula above is called 'tangent' where chances are born. If the given polynomial has some nontrivial roots, the program will search these roots by gradient decreasing in advance and generate some possible tangents accordingly.

For example, if f(0.643104,0.198062,1) = 0 is founded, then one of the tangents automatically generated is g(a,b,c) = a^2 + b^2 - ab - bc + 2bc. This is because the function 
g has the property that g(0.643104,0.198062,1) = 0, a key that highly possibly leads to the success.
