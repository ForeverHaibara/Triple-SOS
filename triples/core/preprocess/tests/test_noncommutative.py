from sympy import symbols

from ...sum_of_squares import sum_of_squares

def test_noncommutative():
    X, Y = symbols("X Y", commutative=False)
    poly = 1+2*X+X**2+X*Y**2+2*Y**2+Y**2*X+Y*X**2*Y+Y**4
    sol = sum_of_squares(poly, verbose=1)
    assert sol is not None
    assert (sol.solution - poly).expand() == 0
