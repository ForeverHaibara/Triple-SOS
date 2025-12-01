from ..univariate import prove_univariate

from sympy import Poly, Rational, sympify, sqrt, prod, E
from sympy.abc import x, y

def test_univariate_correctness():
    # real
    polys = [0, Rational(2,5), x**2, x**4,
        (5*x**2 - 4*x + 1)/3,
        3*x**4 - 9*x**3/2 + x**2 + x/2 + Rational(1,10),
        x**6 + 2*x**5 - 2*x**4 - 5*x**3 + 3*x + 2,
    ]
    for poly in polys:
        proof = prove_univariate(poly)
        assert proof is not None, f"Failed for {poly} (-oo < x < oo)."
        assert (Poly(proof, x) - Poly(poly, x)).is_zero, f"Proof of {poly} incorrect: {proof}."

    # R+
    polys = [0, Rational(2,5), x, 3*x/4 + 2, x**2, x**2/2 + 3*x + 2, x**3/2,
        2*x**3/3 - x**2 - x + Rational(8,5), 2*x**4 + x**3,
        2*x**4 + x**3 - x**2 + Rational(1,20),
        x**5/3 - x**4/2 - x**3 + 21*x**2/10 - x + Rational(1,6)
    ]
    for poly in polys:
        proof = prove_univariate(poly, (0, None))
        assert proof is not None, f"Failed for {poly} (x > 0)."
        assert (Poly(proof, x) - Poly(poly, x)).is_zero, f"Proof of {poly} incorrect: {proof}."

        proof = prove_univariate(poly, (Rational(1,6), None))
        assert proof is not None, f"Failed for {poly} (x > 1/6)."
        assert (Poly(proof, x) - Poly(poly, x)).is_zero, f"Proof of {poly} incorrect: {proof} (x >= 1/6)."

        proof = prove_univariate(poly, (Rational(1,6), Rational(11,2)))
        assert proof is not None, f"Failed for {poly} (1/6 < x < 11/2)."
        assert (Poly(proof, x) - Poly(poly, x)).is_zero, f"Proof of {poly} incorrect: {proof} (1/6 <= x <= 11/2)."

    for poly in polys:
        poly = sympify(poly).xreplace({x: -2 - y}).expand()
        proof = prove_univariate(poly, (None, -2))
        assert proof is not None, f"Failed for {poly} (-oo < y < -2)."
        assert (Poly(proof, y) - Poly(poly, y)).is_zero, f"Proof of {poly} incorrect: {proof}."

    polys_intervals = [
        (-3*x**4/2 + x**3 + 2*x**2 - x + Rational(1, 4), (-Rational(-11,10), Rational(6,5))),
        ((x**4 + 2*x + 1), (-Rational(1,2), 2)),
    ]
    for poly, interval in polys_intervals:
        proof = prove_univariate(poly, interval)
        assert proof is not None, f"Failed for {poly} {interval[0]} < x < {interval[1]}."
        assert (Poly(proof, x) - Poly(poly, x)).is_zero, f"Proof of {poly} incorrect: {proof}."


def test_univariate_domain():
    # RR
    polys_intervals = [
        (1.2*x - 2.3, (1.92, None)),
        (-1.5*x**5 - 2*x**4 + 4*x**3 + 3*x**2 - 3.3*x + 0.9, (-5, Rational(6,5))),
    ]
    for poly, interval in polys_intervals:
        proof = prove_univariate(poly, interval)
        assert proof is not None, f"Failed for {poly} {interval[0]} < x < {interval[1]}."
        assert [abs(_) < 10**-14 for _ in (Poly(proof, x) - Poly(poly, x)).all_coeffs()],\
            f"Proof of {poly} incorrect: {proof}."

    # extension
    polys_intervals = [
        (-x/(1 + sqrt(2)) + sqrt(2), (-3-sqrt(3), 1)),
        ((sqrt(2)*x**2 - (1 + sqrt(6))*x + 5), (None, None)),
        (((x**2 - sqrt(6)*x + 1)**2*(7*x**4 - sqrt(6)*x**3 - x**2 + x/2 + Rational(1,8))).expand(),
            (-Rational(1,4), 10)),
        (x**6 - (E+1)*x**3 + E*x**2/2 + E*x/2 + 1/(3*E**2), (0, None))
    ]
    for poly, interval in polys_intervals:
        proof = prove_univariate(poly, interval)
        assert proof is not None, f"Failed for {poly} {interval[0]} < x < {interval[1]}."
        assert (proof - poly).factor() == 0,\
            f"Proof of {poly} incorrect: {proof}."


def test_univariate_stability():
    # Wilkinson polynomial
    poly = 1 + prod([(x-i)**2 for i in range(12)])
    assert prove_univariate(poly) is not None
