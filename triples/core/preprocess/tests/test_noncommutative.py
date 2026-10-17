from sympy import symbols
from sympy.matrices.expressions import MatrixSymbol, Identity

from ...sum_of_squares import sum_of_squares

def test_noncommutative():
    X, Y = symbols("X Y", commutative=False)
    poly = 1+2*X+X**2+X*Y**2+2*Y**2+Y**2*X+Y*X**2*Y+Y**4
    sol = sum_of_squares(poly)
    assert sol is not None
    assert (sol.solution - poly).expand() == 0

    # test support for matrix variables and identity matrix
    X, Y = [MatrixSymbol(s, 3, 3) for s in 'XY']
    I = Identity(3)
    poly = I+2*X+X**2+X*Y**2+2*Y**2+Y**2*X+Y*X**2*Y+Y**4
    sol = sum_of_squares(poly)
    assert sol is not None
    assert (sol.solution - poly).expand() == I - I # Matrix zero
