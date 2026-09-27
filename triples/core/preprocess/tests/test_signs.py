from sympy.abc import a,b,c,d,e,f,g,h,r,u,v,w,x,y,z
from sympy import Add, Mul, Poly, Pow, Function, Rational, fraction, sqrt
from sympy.combinatorics import CyclicGroup, SymmetricGroup

F, G = Function('F'), Function('G')

import pytest

from ..signs import _prove_poly, _prove_by_recur, _SignProver, sign_sos
from ...problem import InequalityProblem
from ....utils.expressions import CyclicExpr, CyclicSum, CyclicProduct

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
        new_signs = dict.fromkeys(poly.free_symbols, (None, None))
        new_signs.update({e: (1 if s else 0, e) for s, e in signs.values() if s is not None})

        valid_proof = _prove_poly(proof.as_poly(), new_signs, factor=need_factor)
        assert valid_proof is not None, f"Case {ind}: failed to validate the proof {poly} == {proof} given {new_signs}."
        assert (valid_proof - proof).expand() == 0, f"Case {ind}: wrong sign_sos solution {proof} != {valid_proof}."

        diff = (poly - proof).xreplace({
            g: e if s >= 0 else -e for g, (s, e) in signs.items() if s is not None})
        # diff = diff.xreplace({
        #     e: 0 for g, (s, e) in signs.items() if s == 0})
        assert diff.expand() == 0, f"Case {ind}: wrong sign_sos solution {poly} != {proof}."


SIGN_SOS_TEST_CASES = [
    (a*b, {a: (-1, F(a)), b: (-1, F(b))}, F(a)*F(b)),
    (-a-b, {a: (-1, F(a)), b: (-1, F(b))}, F(a)+F(b)),
    ((a+b)*(c+d), {s: (-1, F(s)) for s in (a, b, c, d)},
        (F(a)+F(b))*(F(c)+F(d))),
    (-a**3, {a: (-1, F(a))}, a**2*F(a)),
    (1/(a*b), {a: (-1, F(a)), b: (-1, F(b))}, 1/(F(a)*F(b))),
    (u*z, {z: (0, G(z))}, u*G(z)),
    (u*z**2, {z: (0, G(z))}, u*z*G(z)),
    (u*(z**2+a**2), {z: (0, G(z)), a: (0, G(a))},
        u*(z*G(z)+a*G(a))),
    (-z**2, {z: (0, G(z))}, -z*G(z)),
    (a**2-z**2, {z: (0, G(z))}, a**2-z*G(z)),
    ((a+z)*b, {a: (-1, F(a)), b: (-1, F(b)), z: (0, G(z))},
        (F(a)-G(z))*F(b)),
    (u*a*b, {a: (0, G(a)), b: (0, G(b))}, u*G(a)*G(b)),
    (-a, {a: (-1, None)}, -a),
    (a, {a: (1, None)}, a),
    (u*z, {z: (0, None)}, u*z),
    (F(a)*b, {F(a): (-1, u), b: (-1, v)}, u*v),
    ((sqrt(2)-1)*(sqrt(3)-1), {}, (sqrt(2)-1)*(sqrt(3)-1)),
    ((1-sqrt(2))*(1-sqrt(3)), {}, (1-sqrt(2))*(1-sqrt(3))),
]

@pytest.mark.parametrize("expr, signs, expected", SIGN_SOS_TEST_CASES)
def test_sign_sos_signed_certificates(expr, signs, expected):
    proof = sign_sos(expr, signs)
    assert proof is not None
    assert (proof-expected).expand() == 0
    # Restore the witnesses without discarding equality constraints.
    restore = {v: (-s if sign == -1 else s)
        for s, (sign, v) in signs.items() if v is not None and sign is not None}
    assert fraction((proof.xreplace(restore)-expr).together())[0].expand() == 0


@pytest.mark.parametrize("expr, signs", [
    (a*b, {a: (-1, F(a)), b: (1, F(b))}),
    (a+b, {a: (-1, F(a)), b: (1, F(b))}),
    (a*b, {a: (None, None), b: (-1, F(b))}),
    (a**Rational(1, 3), {a: (-1, F(a))}),
    (1/z, {z: (0, G(z))}),
    (z**-2, {z: (0, G(z))}),
])
def test_sign_sos_unknown(expr, signs):
    assert sign_sos(expr, signs) is None


def test_sign_sos_cyclic_certificates():
    signs = {s: (-1, F(s)) for s in (a, b, c)}
    expr = -CyclicSum(a*(b-c)**2)
    proof = sign_sos(expr, signs)
    assert isinstance(proof, CyclicSum)
    assert (proof.doit().xreplace({F(s): -s for s in signs})-expr.doit()).expand() == 0

    # There are six factors, not three, and the representative is nonpositive.
    expr = CyclicProduct(a, (a, b, c), SymmetricGroup(3), evaluate=False)
    proof = sign_sos(expr, signs)
    assert isinstance(proof, CyclicProduct)
    assert proof.args[0] == F(a)
    assert (proof.doit().xreplace({F(s): -s for s in signs})-expr.doit()).expand() == 0
    expr = CyclicProduct(a, (a, b, c), CyclicGroup(3), evaluate=False)
    assert sign_sos(expr, signs) is None


def test_sign_sos_cyclic_zero_and_asymmetric_signs():
    expr = CyclicProduct(a, (a, b, c), evaluate=False)
    proof = sign_sos(expr, {a: (0, G(a))})
    assert proof == b*c*G(a)
    proof = sign_sos(expr, {a: (-1, u), b: (-1, v), c: (1, w)})
    assert proof == u*v*w

    signs = {s: (0, G(s)) for s in (a, b, c)}
    expr = x*CyclicSum(a**2, (a, b, c), evaluate=False)
    proof = sign_sos(expr, signs)
    assert proof == x*CyclicSum(a*G(a), (a, b, c), evaluate=False)
    assert (proof.doit().xreplace({G(s): s for s in signs})-expr.doit()).expand() == 0

    expr = CyclicSum(a, (a, b, c), evaluate=False)
    assert sign_sos(expr, {a: (1, u), b: (1, v), c: (1, w)}) == u+v+w
    # Composite keys must have invariant directions as well as witnesses.
    expr = CyclicProduct(F(a), (a, b, c), evaluate=False)
    signs = {F(a): (-1, F(a)), F(b): (1, F(b)), F(c): (1, F(c))}
    assert sign_sos(expr, signs) is None


def test_sign_sos_cyclic_representative_only(monkeypatch):
    expr = CyclicSum(a*(b-c)**2, (a, b, c), evaluate=False)
    original = CyclicExpr._generate_all_translations

    def translations(symbols, group, full=True):
        assert not full, "An invariant proof must not enumerate the whole group."
        return original(symbols, group, full=False)

    monkeypatch.setattr(CyclicExpr, '_generate_all_translations', staticmethod(translations))
    proof = sign_sos(expr, {s: (1, F(s)) for s in (a, b, c)})
    assert isinstance(proof, CyclicSum)
    assert proof.args[0] == F(a)*(b-c)**2
    assert _prove_by_recur(expr, {s: (1, s) for s in (a, b, c)}) == (expr, False)


def test_sign_sos_preserves_square_and_witnesses():
    expr = (a-b)**2
    prover = _SignProver({a: (1, F(a)), b: (1, F(b))})
    assert prover.prove(expr) == (expr, False)
    assert (expr.base, False) not in prover.results
    assert sign_sos(a, {a: (1, F(a))}) == F(a)
    # Even mutually referring witnesses are terminal certificates.
    assert sign_sos(a+b, {a: (1, F(b)), b: (1, F(a))}) == F(a)+F(b)


def test_sign_sos_deep_expression_and_shared_nodes():
    # test a highly nested expression
    expr = a
    for i in range(1200):
        expr = Add(expr, 1, evaluate=False) if i % 2 else Pow(expr, 3, evaluate=False)
        # SymPy itself hashes recursively; warm each new node bottom-up.
        hash(expr)
    assert sign_sos(expr, {a: (1, a)}) is expr
    assert sign_sos(expr, {}) is None

    expr = z
    for i in range(1200):
        expr = Add(expr, z, evaluate=False) if i % 2 else Pow(expr, 2, evaluate=False)
        hash(expr)
    expr = Mul(u, expr, evaluate=False)
    assert sign_sos(expr, {z: (0, z)}) is expr

    shared = Add(a, b, evaluate=False)
    expr = Add(shared, Mul(2, shared, evaluate=False), evaluate=False)
    prover = _SignProver({a: (1, F(a)), b: (1, F(b))})
    assert prover.prove(expr)[0] == 3*(F(a)+F(b))
    size = len(prover.results)
    assert prover.prove(expr)[0] == 3*(F(a)+F(b))
    assert len(prover.results) == size
