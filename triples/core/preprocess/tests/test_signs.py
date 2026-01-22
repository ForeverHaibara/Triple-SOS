from sympy.abc import a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z
from sympy import Poly, Function, Rational, fraction

F, G = Function('F'), Function('G')

import pytest

from ..signs import _prove_poly, sign_sos
from ...problem import InequalityProblem

class InferSignProblems:
    """
    Each of the problem must return a tuple of (ineq_constraints, eq_constraints, signs)
    """
    @classmethod
    def collect(cls):
        return {k: getattr(cls, k) for k in dir(cls) if k.startswith('problem')}

    @classmethod
    def problem_basic_ineqs1(cls):
        ineqs = {
            a: F(a),
            c*3: F(c),
            3*g - 1: F(g),
            d*-4/5: F(d),
            -b: F(b),
            e + 2: F(e),
            -5*f + 3: F(f),
            -2*h/3 - 4: F(h),
        }
        signs = {
            a: (1, F(a)),
            b: (-1, F(b)),
            g: (1, F(g)/3 + Rational(1,3)),
            c: (1, F(c)/3),
            d: (-1, F(d)*5/4),
            e: (None, None),
            f: (None, None),
            h: (-1, F(h)*3/2 + 6),
        }
        return ineqs, {}, signs

    @classmethod
    def problem_basic_eqs1(cls):
        eqs = {
            a: F(a),
            c*3: F(c),
            d*-4/5: F(d),
            -b: F(b),
            4*e/3 + 2: F(e),
            -5*f + Rational(1,2): F(f),
        }
        signs = {
            a: (0, F(a)),
            b: (0, -F(b)),
            c: (0, F(c)/3),
            d: (0, F(d)*-5/4),
            e: (-1, (2 - F(e))*3/4),
            f: (1, (-F(f) + Rational(1,2))/5)
        }
        return {}, eqs, signs

    @classmethod
    def problem_relation1(cls):
        ineqs = {
            5*c - b: x,
            a - 3: y,
            b - 2*a: z,
            d + b - a: u,
            2*c - d*7: v,
            e + 2: F(e),
            -3*e - c**2*(a + 3): w,
            (-a - 3 - 2*b**3)*g - 4*b - c**3/2 + f*2 - a*f**2: z,
        }
        eqs = {
            2*f + 1: r,
        }
        b_ = z + 2*(y + 3)
        c_ = (x + z + 2*(y + 3))/5
        signs = {
            a: (1, y + 3),
            b: (1, b_),
            c: (1, c_),
            d: (None, None),
            e: (-1, (w + c**2*(y + 6))/3),
            f: (-1, (1 - r)/2),
            g: (-1, (z + (y + 3)*f**2 - r + 1 + c_/2*c**2 + 4*b_)/(2*b_*b**2 + y + 6))
        }
        return ineqs, eqs, signs

    @classmethod
    def problem_relation2(cls):
        ineqs = {
            a*b - 2: x,
            b + 3: y,
            b*c - 4*a + 2: z,
            -(b**2 + 2 - b*e)*d - 4*a - b: r
        }
        eqs = {
            4*b - 1: u,
            a*e + a**3 + b + 2: v,
        }
        a_ = (x + 2)/(u + 1)*4
        b_ = (u + 1)/4
        e_ = (-v + a_*a**2 + b_ + 2)/a_
        signs = {
            a: (1, a_),
            b: (1, b_),
            c: (None, None),
            d: (-1, (r + 4*a_ + b_)/(b**2 + 2 + b_ * e_)),
            e: (-1, e_),
        }
        return ineqs, eqs, signs


@pytest.mark.parametrize("problem", InferSignProblems.collect().values(),
    ids=InferSignProblems.collect().keys())
def test_infer_signs(problem):
    ineqs, eqs, signs0 = problem()
    pro = InequalityProblem(Rational(0), ineqs, eqs)
    signs1 = pro.get_symbol_signs()

    assert set(signs1.keys()) == set(signs0.keys())

    for key in signs1.keys():
        sign0, expr0 = signs0[key]
        sign1, expr1 = signs1[key]
        assert sign0 == sign1, f"got signs[{key}] = {signs1[key]}, expected {signs0[key]}"
        if sign0 is not None:
            assert fraction((expr0 - expr1).together())[0].expand() == 0,\
                f"got signs[{key}] = {signs1[key]}, expected {signs0[key]}"


def test_infer_signs_empty():
    pro = InequalityProblem(Poly(0, a, b), {}, {})
    assert pro.get_symbol_signs() == {a: (None, None), b: (None, None)}


def test_prove_poly_by_signs():
    cases = [
        (
            3*a**3*(2 - b) - b**3*c + 2*c**5/3,
            {b: (-1, r), a: (1, a), c: (1, u)}
        ),
        (
            3*a*b*(c + 2*a*b**5) + a**4*(b**2 + c**2)/5 - c**3*b + 2*(a**2 + b*c)*b**2,
            {c: (0, v), a: (None, None)}
        ),
        (
            4*(a**3*b + b*c) + b**2*(c + 2)/3 + (4 - a - b)*(1 - a - b)*(a*b + 2),
            {a: (-1, u), b: (-1, v), c: (0, r)}
        ),
        (
            a**2 - a*b + b**2,
            {a: (0, a), b: (0, v), c: (-1, r)}
        ),
        (
            (3*a*(a - b)*(a - 2*b + 3 - c**2)**2*(a**3 - 4*b + 1)**3/4),
            {a: (1, u), b: (-1, v)}
        )
    ]
    for ind, (poly, signs) in enumerate(cases):
        need_factor = (ind >= 4)
        proof = _prove_poly(poly.as_poly(a, b, c), signs, factor=need_factor)
        assert proof is not None, f"Case {ind}: failed to establish the nonnegativity of {poly} given {signs}."

        # extract nonnegative symbols from "signs"
        new_signs = {g: (None, None) for g in poly.free_symbols}
        new_signs.update({e: (1 if s else 0, e) for s, e in signs.values() if s is not None})

        valid_proof = _prove_poly(proof.as_poly(), new_signs, factor=need_factor)
        assert valid_proof is not None, f"Case {ind}: failed to validate the proof {poly} == {proof} given {new_signs}."
        assert (valid_proof - proof).expand() == 0, f"Case {ind}: wrong sign_sos solution {proof} != {valid_proof}."

        diff = (poly - proof).xreplace({
            g: e if s >= 0 else -e for g, (s, e) in signs.items() if s is not None})
        # diff = diff.xreplace({
        #     e: 0 for g, (s, e) in signs.items() if s == 0})
        assert diff.expand() == 0, f"Case {ind}: wrong sign_sos solution {poly} != {proof}."
