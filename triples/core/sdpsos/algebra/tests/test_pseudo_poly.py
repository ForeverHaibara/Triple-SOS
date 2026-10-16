import pytest

from sympy import (
    EX, QQ, RR, ZZ, Float, Function, Integer, MatrixSymbol, Poly, Rational,
    S, expand, pi, sin, sqrt, symbols,
)
from sympy.combinatorics import Permutation, PermutationGroup, SymmetricGroup
from sympy.polys.polyerrors import ExactQuotientFailed, GeneratorsError, PolynomialError

from ..pseudo_poly import PseudoPoly, PseudoSMP, convert_expr_to_pseudo_poly
from ...algebra import IdealBasis, MixedMomentAlgebra, NCPolyRing, PolyRing, QModuleBasis
from ...manifold import complete_constraints_by_symmetry


@pytest.fixture(params=['commutative', 'noncommutative', 'moments'])
def algebra_gens(request):
    """Exercise exponent tuples, words, and mixed moment monomials."""
    kind = request.param
    gens = symbols('x y z', commutative=kind != 'noncommutative')
    if kind == 'commutative':
        algebra = PolyRing(3, 3, is_homogeneous=False)
    elif kind == 'noncommutative':
        algebra = NCPolyRing(3, 3, is_homogeneous=False)
    else:
        algebra = MixedMomentAlgebra(3, 3, 2, is_homogeneous=False)
    return algebra, gens


def _assert_domain_elements(poly):
    assert poly.rep.dom == poly.domain
    assert all(poly.domain.of_type(coeff) for coeff in poly.rep.values())


def test_pseudo_smp(algebra_gens):
    algebra, _ = algebra_gens
    one, x, y = [algebra.gen_monom(i) for i in (None, 0, 1)]
    xx = algebra.mul((x, ZZ.one), (x, ZZ.one))[0]
    xy = algebra.mul((x, ZZ.one), (y, ZZ.one))[0]
    yx = algebra.mul((y, ZZ.one), (x, ZZ.one))[0]
    p = PseudoSMP.from_dict({x: 2, one: -4}, 2, ZZ, algebra=algebra)
    q = PseudoSMP(algebra, ZZ, {y: ZZ(3), one: ZZ(4)})
    original = p.to_dict()

    assert p.algebra is algebra
    assert p.dom == ZZ
    assert p.to_dict() == {x: ZZ(2), one: ZZ(-4)}
    assert p.terms() == [(x, ZZ(2)), (one, ZZ(-4))]
    assert p.coeffs() == [ZZ(2), ZZ(-4)]
    assert p.monoms() == [x, one]
    assert p.per({y: ZZ.one}) == {y: ZZ.one}
    assert p.per({}).algebra is algebra
    assert p.per({}).dom == ZZ

    zero = PseudoSMP(algebra, ZZ)
    assert zero == p.zero == {}
    assert zero.is_zero and not zero
    assert PseudoSMP(algebra, ZZ, {x: ZZ.zero}).is_zero
    assert p.one == {one: ZZ.one}
    assert zero.total_degree() == 0
    assert zero.is_homogeneous and zero.is_linear and zero.is_quadratic
    assert not p.is_zero and p.total_degree() == 1
    assert not p.is_homogeneous and p.is_linear and p.is_quadratic
    homogeneous = PseudoSMP(algebra, ZZ, {xx: ZZ(2), xy: ZZ(3)})
    assert homogeneous.is_homogeneous
    assert not homogeneous.is_linear and homogeneous.is_quadratic
    assert not (p ** 3).is_quadratic

    assert p + q == {x: ZZ(2), y: ZZ(3)}
    assert p - q == {x: ZZ(2), y: ZZ(-3), one: ZZ(-8)}
    assert p + (-p) == p - p == zero
    assert -p == {x: ZZ(-2), one: ZZ(4)}
    assert p * q == {xy: ZZ(6), x: ZZ(8), y: ZZ(-12), one: ZZ(-16)}
    assert q * p == {yx: ZZ(6), x: ZZ(8), y: ZZ(-12), one: ZZ(-16)}
    assert p * p.one == p.one * p == p
    assert p * zero == zero * p == zero
    assert p ** 0 == zero ** 0 == p.one
    assert p ** 1 == p
    assert p ** Integer(3) == p * p * p
    assert p ** 2 == {xx: ZZ(4), x: ZZ(-16), one: ZZ(16)}
    assert zero ** 3 == zero
    with pytest.raises(ValueError, match='negative'):
        p ** -1
    for exponent in (Rational(1, 2), 1.5, symbols('n')):
        with pytest.raises(TypeError):
            p ** exponent

    content, primitive = p.primitive()
    assert content == ZZ(2)
    assert primitive == {x: ZZ.one, one: ZZ(-2)}
    assert zero.primitive() == (ZZ.zero, zero)
    rational = PseudoSMP.from_dict({x: Rational(2, 3), one: Rational(4, 9)}, 2, QQ, algebra=algebra)
    content, primitive = rational.primitive()
    assert content == QQ(2, 9)
    assert primitive == {x: QQ(3), one: QQ(2)}
    assert rational.monic() == {x: QQ.one, one: QQ(2, 3)}
    assert p.monic() == {x: ZZ.one, one: ZZ(-2)}
    assert all(ZZ.of_type(c) for c in p.monic().values())
    with pytest.raises(ExactQuotientFailed):
        PseudoSMP(algebra, ZZ, {x: ZZ(2), one: ZZ.one}).monic()
    assert zero.monic() == zero

    converted = p.convert(QQ)
    assert converted.dom == QQ and converted.algebra is algebra
    assert all(QQ.of_type(c) for c in converted.values())
    assert converted == p and hash(converted) == hash(p)
    assert p.to_dict() == original and p.dom == ZZ
    assert hash(p) == hash(PseudoSMP(algebra, ZZ, dict(reversed(p.terms()))))
    assert hash(p) == hash(p)
    assert p.state() == PseudoSMP(algebra, ZZ, dict(algebra.s(term) for term in p.terms()))
    assert zero.state() == zero


def test_pseudo_poly(algebra_gens):
    algebra, gens = algebra_gens
    x, y, z = gens
    one, mx, my = [algebra.gen_monom(i) for i in (None, 0, 1)]
    source = {mx: Integer(2), one: Integer(-4), my: S.Zero}
    original = source.copy()
    p = PseudoPoly.from_dict(source, *gens, algebra=algebra)
    q = PseudoPoly.from_dict(source, *gens, algebra=algebra, domain='QQ')

    assert source == original
    assert all(source[monom] is coefficient for monom, coefficient in original.items())
    assert p.rep == {mx: ZZ(2), one: ZZ(-4)}
    assert p.gens == gens and p.algebra is algebra
    assert p.domain == p.get_domain() == ZZ
    assert p.terms() == [(mx, Integer(2)), (one, Integer(-4))]
    assert p.coeffs() == [Integer(2), Integer(-4)]
    assert p.monoms() == [mx, one]
    assert p.as_expr() == 2*x - 4
    assert p.LC() == 2
    assert isinstance(p.LC(), Integer)
    assert p.free_symbols == set(gens)
    assert p.free_symbols_in_domain == set()
    assert bool(p) and not p.is_zero
    assert p.is_linear and p.is_quadratic and not p.is_homogeneous
    assert p.total_degree() == 1
    assert q.domain == QQ and all(QQ.of_type(c) for c in q.rep.values())
    assert p.set_domain(QQ) == q
    assert p.domain == ZZ
    assert p.to_field() == q
    assert q.to_field() is q
    assert p.monic().domain == q.monic().domain == QQ
    assert p.monic().as_expr() == q.monic().as_expr() == x - 2
    content, primitive = p.primitive()
    assert content == Integer(2) and primitive.as_expr() == x - 2

    zero = p - p
    assert zero.is_zero and not zero
    assert zero.rep == {} and zero.as_expr() == 0
    assert zero.terms() == zero.monoms() == zero.coeffs() == []
    assert zero.LC() == 0 and zero.total_degree() == 0
    assert zero.is_homogeneous and zero.is_linear and zero.is_quadratic
    assert zero.primitive() == (S.Zero, zero)
    assert zero.monic().is_zero and zero.monic().domain == QQ
    assert (p + p).as_expr() == 4*x - 8
    assert (-p).as_expr() == 4 - 2*x
    assert expand((p * p).as_expr()) == expand((2*x - 4)**2)
    assert expand((p ** 3).as_expr()) == expand((2*x - 4)**3)
    assert (p ** 0).as_expr() == 1
    with pytest.raises(ValueError):
        p ** -1
    assert p.state().rep == p.rep.state()

    assert p.per(p.rep) == p
    assert p.per(p.rep, gens=(z, y, x)).gens == (z, y, x)
    assert p.per(p.rep, remove=1).gens == (x, z)
    univariate = PseudoPoly.new(p.rep, x)
    assert univariate.per(ZZ(3), remove=0) == Integer(3)
    assert PseudoPoly.new(p.rep, *gens) == p
    assert p != zero and p != q
    assert p != p.per(p.rep, gens=(z, y, x))
    other_algebra = NCPolyRing(3, 3)
    assert p != PseudoPoly.new(PseudoSMP(other_algebra, ZZ, p.rep), *gens)
    assert len({p, zero, q, PseudoPoly.new(p.rep, *gens)}) == 3
    assert hash(p) == hash(PseudoPoly.new(p.rep, *gens))
    reordered_terms = PseudoPoly.from_dict(dict(reversed(p.terms())), *gens, algebra=algebra)
    assert reordered_terms == p and hash(reordered_terms) == hash(p)

    t = symbols('t')
    for domain, coefficient in ((ZZ.poly_ring(t), t + 1), (QQ.frac_field(t), 1/t), (EX, sin(t))):
        parameter_poly = PseudoPoly.from_dict({mx: coefficient}, *gens, algebra=algebra, domain=domain)
        assert parameter_poly.free_symbols_in_domain == {t}
        assert parameter_poly.free_symbols == set(gens) | {t}
        assert parameter_poly.coeffs() == [coefficient]
        assert parameter_poly.as_expr() == coefficient*x
    content, primitive = PseudoPoly.from_dict({mx: 2*(t + 1), one: 4*(t + 1)},
        *gens, algebra=algebra, domain=ZZ.poly_ring(t)).primitive()
    assert content == 2*t + 2
    assert primitive.as_expr() == x + 2


def test_convert_expr_to_pseudo_poly(algebra_gens):
    algebra, gens = algebra_gens
    x, y, _ = gens

    def convert(expr, **kwargs):
        # convert expr to PseudoPoly and assert domain elements
        conv = convert_expr_to_pseudo_poly(algebra, expr, gens, **kwargs)
        _assert_domain_elements(conv)
        return conv

    for expr in (0, 1, -3, Rational(2, 3), Float('1.25'), pi, sqrt(2), x, x*y*x, (x + y + 1)**3):
        p = convert(expr)
        assert isinstance(p, PseudoPoly)
        assert p.gens == gens and p.algebra is algebra
        assert expand(p.as_expr()) == expand(expr)
    assert convert(0).is_zero and convert(0).rep == {}
    assert convert(sin(1)).as_expr() == sin(1)
    assert convert('2').as_expr() == 2
    assert convert(x - x).is_zero
    assert convert(Rational(1, 2)*x).domain == QQ
    assert convert(Float('1.25')*x).domain == RR
    assert convert(sqrt(2)*x, extension=True).domain == QQ.algebraic_field(sqrt(2))
    assert convert(x, domain=QQ).domain == QQ
    assert convert(x, domain=QQ).rep[algebra.gen_monom(0)] == QQ.one
    assert convert(x**3 + y*x*y - 2*x*y).as_expr() == x**3 + y*x*y - 2*x*y
    assert convert(x**3 + y*x*y - 2*y*x).as_expr() == x**3 + y*x*y - 2*y*x

    t = symbols('t')
    for expr in (t*x + t**2, x/t, sin(t)*x, sqrt(t)*x):
        p = convert(expr)
        assert p.free_symbols_in_domain == {t}
        assert p.as_expr() == expr
    for expr in (x**-1, sqrt(x), x**t, sin(x), 1/(x + 1)):
        with pytest.raises(PolynomialError):
            convert(expr)


def test_convert_noncommutative_and_matrix_powers():
    x, y = symbols('x y', commutative=False)
    algebra = NCPolyRing(2, 4, is_homogeneous=False)
    p = algebra.as_poly(x*y - y*x, (x, y))
    assert p.rep == {((0, 1), (1, 1)): ZZ.one, ((1, 1), (0, 1)): -ZZ.one}
    assert p.state() == p
    assert expand((p ** 2).as_expr()) == expand((x*y - y*x)**2)
    a, b = MatrixSymbol('A', 2, 2), MatrixSymbol('B', 2, 2)
    matrix_poly = algebra.as_poly((a*b)**2, (a, b))
    assert matrix_poly.rep == {((0, 1), (1, 1), (0, 1), (1, 1)): ZZ.one}
    with pytest.raises(PolynomialError):
        algebra.as_poly(a**-1, (a, b))


def test_pseudo_poly_reorder(algebra_gens):
    algebra, gens = algebra_gens
    x, y, z = gens
    p = convert_expr_to_pseudo_poly(algebra, x**2*y + 2*y*z + 3*x, gens)
    for reordered_gens in ((y, x, z), (y, z, x), (z, x, y)):
        # test PseudoPoly.reorder does not change the value
        # of the expression directly
        reordered = p.reorder(*reordered_gens)
        expected = convert_expr_to_pseudo_poly(algebra, p.as_expr(), reordered_gens)
        assert reordered.gens == reordered_gens
        assert reordered.rep == expected.rep
        assert reordered.as_expr() == p.as_expr()
        assert reordered.algebra is algebra and reordered.domain == p.domain
        assert reordered.reorder(*gens) == p
    assert p.reorder() == p
    assert p.reorder(z, x, y).reorder() == p
    for invalid_gens in ((x, y), (x, x, z), (x, y, symbols('t'))):
        with pytest.raises(GeneratorsError):
            p.reorder(*invalid_gens)

    if isinstance(algebra, MixedMomentAlgebra):
        state = Function('s')
        expr = x*state(y)*state(x*z)**2 + state(x)*state(y*z)
        p = algebra.as_poly(expr, gens, state_operator=state)
        for reordered_gens in ((y, z, x), (z, x, y)):
            reordered = p.reorder(*reordered_gens)
            expected = algebra.as_poly(expr, reordered_gens, state_operator=state)
            assert reordered == expected
            assert reordered.as_expr(state_operator=state) == expr
            assert reordered.reorder(*gens) == p


def test_sos_basis_symmetry(algebra_gens):
    algebra, gens = algebra_gens
    x, y, z = gens
    cycle = Permutation([1, 2, 0])
    swap = Permutation([1, 0, 2])
    group = SymmetricGroup(3)
    if isinstance(algebra, (PolyRing, NCPolyRing)):
        algebra = type(algebra)(3, 3, is_homogeneous=False, symmetry=group)
    else:
        algebra.symmetry = group
    constant = convert_expr_to_pseudo_poly(algebra, S.One, gens)
    monoms = [algebra.gen_monom(i) for i in range(3)]
    assert set(algebra.permute(monoms[0])) == set(monoms)
    basis = QModuleBasis(algebra, constant, monoms)
    symmetry, action = basis.get_symmetry_representation()
    assert symmetry.order() == 6
    assert action(cycle) == [2, 0, 1]
    assert action(cycle * swap) == [action(cycle)[i] for i in action(swap)]
    assert {frozenset(entries) for entries in basis.get_equal_entries()} == {
        frozenset((0, 4, 8)), frozenset((1, 2, 3, 5, 6, 7)),
    }

    p = convert_expr_to_pseudo_poly(algebra, x**2*y + 2*y*z, gens)
    cyclic_group = PermutationGroup(cycle)
    orbit = complete_constraints_by_symmetry([p], cyclic_group)
    expected_exprs = (x**2*y + 2*y*z, z**2*x + 2*x*y, y**2*z + 2*z*x)
    assert set(orbit) == {
        convert_expr_to_pseudo_poly(algebra, expr, gens) for expr in expected_exprs
    }
    assert len(complete_constraints_by_symmetry(orbit, cyclic_group)) == 3
    assert complete_constraints_by_symmetry([], group) == []
    assert complete_constraints_by_symmetry([p]) == [p]
    if isinstance(algebra, PolyRing):
        native = Poly(p.as_expr(), *gens)
        assert {poly.as_expr() for poly in complete_constraints_by_symmetry([native], cyclic_group)} == {
            expand(expr) for expr in expected_exprs
        }

    symmetric_poly = convert_expr_to_pseudo_poly(algebra, x + y + z, gens)
    asymmetric_poly = convert_expr_to_pseudo_poly(algebra, x + y, gens)
    assert algebra.is_symmetric(symmetric_poly, group)
    assert not algebra.is_symmetric(asymmetric_poly, cycle)
    for basis_class in (QModuleBasis, IdealBasis):
        assert basis_class(algebra, symmetric_poly, monoms).symmetry.order() == 6
        partial_basis = basis_class(algebra, asymmetric_poly, monoms)
        assert partial_basis.symmetry.order() == 2
        assert partial_basis.symmetry.contains(swap)
        if basis_class is IdealBasis:
            assert {frozenset(entries) for entries in partial_basis.get_equal_entries()} == {
                frozenset((0, 1)), frozenset((2,)),
            }

    if isinstance(algebra, MixedMomentAlgebra):
        state = Function('s')
        p = algebra.as_poly(x*state(y), gens, state_operator=state)
        orbit = complete_constraints_by_symmetry([p], cyclic_group)
        assert {poly.as_expr(state_operator=state) for poly in orbit} == {
            x*state(y), z*state(x), y*state(z),
        }
        partial = algebra.as_poly(state(x) + state(y), gens, state_operator=state)
        assert QModuleBasis(algebra, partial, monoms).symmetry.order() == 2


def test_commutative_arithmetic_matches_sympy():
    # test PseudoPoly behaves as Poly
    x, y = symbols('x y')
    algebra = PolyRing(2, 4, is_homogeneous=False)
    exprs = (0, Rational(2, 3), x - 2*y + 1, (x + y)**2, 2*x**3 - y**2)
    for expr in exprs:
        p = convert_expr_to_pseudo_poly(algebra, expr, (x, y), domain=QQ)
        reference = Poly(expr, x, y, domain=QQ)
        assert p.rep.to_dict() == {m: c for m, c in reference.rep.to_dict().items() if c}
        assert p.as_expr() == reference.as_expr()
        assert algebra.as_expr(reference) == reference.as_expr()
        assert p.monic().as_expr() == reference.monic().as_expr()
        content, primitive = p.primitive()
        expected_content, expected_primitive = reference.primitive()
        assert content == expected_content
        assert primitive.as_expr() == expected_primitive.as_expr()
        for other in exprs:
            q = convert_expr_to_pseudo_poly(algebra, other, (x, y), domain=QQ)
            assert (p + q).as_expr() == expand(expr + other)
            assert (p - q).as_expr() == expand(expr - other)
            assert (p * q).as_expr() == expand(expr * other)
