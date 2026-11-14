"""
This module contains heuristic, experimental, imcomplete facial reduction algorithm
for SDP via computing the equality cases of the original SOS problem.
"""
from collections import defaultdict
from itertools import islice, product
from typing import (
    Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple,
    Union, Optional, Callable, Any, TYPE_CHECKING
)
from sympy import QQ
from sympy.external.gmpy import MPQ
from sympy.matrices import MutableDenseMatrix as Matrix
from sympy.polys.rings import ring

from .algebra import PseudoPoly
from ...sdp.arithmetic import ArithmeticTimeout, rep_matrix_from_list
from ...utils import Root, MonomialManager
from ...utils.roots.roots import _algebraic_field_coeffs

if TYPE_CHECKING:
    from sympy import Poly
    from sympy.combinatorics import PermutationGroup
    from sympy.polys.rings import PolyElement

    from .algebra import SOSBasis


_Monomial = Tuple[int, ...]
_Coefficients = Mapping[_Monomial, MPQ]
_GramBlock = Tuple[Any, _Coefficients, Sequence[_Monomial]]
_Jet = Tuple[Any, Tuple[MPQ, ...]]
_SignedJet = Tuple[Any, bool, Tuple[MPQ, ...]]
_JetLayers = Tuple[Tuple[_SignedJet, ...], ...]


def _order(poly: "PolyElement") -> int:
    """Return the order at zero of a nonzero univariate ring element."""
    return min(poly)[0]


def _parametrization_jets(
    target: _Coefficients,
    equalities: Sequence[_Coefficients],
    blocks: Sequence[_GramBlock],
    parametrization: Sequence["PolyElement"],
    permutations: Sequence[Sequence[int]] = (),
) -> _JetLayers:
    """Prepare successive orders of the Gram terms on one parametrization.

    The coordinates are polynomials in a common univariate ring over QQ,
    with the base point at parameter zero. A nonzero Gram term has leading
    order d + 2*j and the sign of its weight. Here d is the weight order,
    and j is the first basis coefficient outside the Gram kernel.
    """
    ring = parametrization[0].ring
    cache: Dict[_Monomial, PolyElement] = {}

    def monomial(powers: _Monomial) -> "PolyElement":
        if powers not in cache:
            value = ring.one
            for x, e in zip(parametrization, powers):
                if e:
                    value *= x**e
            cache[powers] = value
        return cache[powers]

    def restrict(poly: _Coefficients, perm: Sequence[int] = ()) -> "PolyElement":
        return sum((c * monomial(tuple(m[i] for i in perm) if perm else m)
                    for m, c in poly.items()), ring.zero)

    if any(restrict(h) for h in equalities):
        return ()
    rhs = restrict(target)
    rhs_order = _order(rhs) if rhs else float('inf')
    # Restrict each orbit term separately: an invariant SOS sum can contain
    # weights of both signs even when its representative is positive.
    terms = []
    for key, weight, basis in blocks:
        seen = set()
        for perm in permutations or ((),):
            g = restrict(weight, perm)
            if not g:
                continue
            d = _order(g)
            if d >= rhs_order:
                continue
            monoms = tuple(tuple(m[i] for i in perm) if perm else m for m in basis)
            signature = (d, g[(d,)] > 0, monoms)
            if signature not in seen:
                seen.add(signature)
                terms.append((key, *signature))

    # The target is a fixed term on the other side. Work strictly below its
    # first nonzero order, where its coefficients cannot participate.
    if not terms:
        return ()
    by_order = defaultdict(list)
    for key, d, positive, monoms in terms:
        evaluated = [monomial(m) for m in monoms]
        count = max((max(p)[0] for p in evaluated if p), default=-1) + 1
        if rhs:
            count = min(count, (int(rhs_order)-d+1)//2)
        for j in range(count):
            column = tuple(p.get((j,), QQ.zero) for p in evaluated)
            if any(column):
                by_order[d+2*j].append((key, positive, column))

    return tuple(tuple(by_order[order]) for order in sorted(by_order))


def scan_jet_layers(layers: _JetLayers, survives: Callable[[_Jet], bool]) -> Iterator[Tuple[bool, _Jet]]:
    """Yield proven directions and the first layer that can still cancel."""
    for layer in layers:
        active = [(key, positive, column) for key, positive, column in layer
                  if survives((key, column))]
        mixed = len({positive for _, positive, _ in active}) == 2
        for key, _, column in active:
            yield mixed, (key, column)
        if mixed:
            break


def _collect_line_jets(
    target: "Poly",
    inequalities: Mapping[Any, "Poly"],
    bases: Mapping[Any, "SOSBasis"],
    ideal: Iterable["Poly"],
    *,
    roots: Sequence[Root],
    symmetry: Optional["PermutationGroup"],
    check_time: Callable[[], None],
) -> List[_JetLayers]:
    nvars = len(target.gens)
    if nvars > 4 or not roots:
        return []
    equalities = complete_constraints_by_symmetry(list(ideal), symmetry)
    if not all(p.domain.is_ZZ or p.domain.is_QQ for p in [
            target, *inequalities.values(), *equalities]):
        return []

    def coefficients(poly: "Poly") -> _Coefficients:
        return poly.set_domain(QQ).rep.to_dict()

    _, t = ring('t', QQ)
    target_coeffs = coefficients(target)
    eq_coeffs = [coefficients(p) for p in equalities]
    blocks = [(k, coefficients(p), bases[k]._basis)
              for k, p in inequalities.items() if len(bases[k])]
    permutations = ([tuple(p(i) if i < p.size else i for i in range(nvars))
                     for p in symmetry.elements] if symmetry is not None else [])
    lines: Dict[_JetLayers, None] = {}
    points = dict.fromkeys(
        tuple(QQ.convert(p, root.domain) for p in root.rep)
        for root in roots if root.is_Rational and root.rep is not None)
    directions = list(product((-1, 0, 1), repeat=nvars))
    for point in islice(points, 12):
        check_time()
        if target.eval(dict(zip(target.gens, map(QQ.to_sympy, point)))) != 0:
            continue
        stabilizer = [p for p in permutations if tuple(point[i] for i in p) == point]
        for direction in directions:
            # The line test already uses the full orbit of every Gram term.
            if any(tuple(direction[i] for i in p) < direction for p in stabilizer):
                continue
            check_time()
            parametrization = tuple(p + v*t for p, v in zip(point, direction))
            layers = _parametrization_jets(
                target_coeffs, eq_coeffs, blocks, parametrization, permutations)
            if layers:
                lines[layers] = None
    return list(lines)


def _root_span(root: Root, basis: "SOSBasis") -> Matrix:
    """Evaluate the basis and extract rational coefficient vectors."""
    values = [root._single_power_monomial(monom) for monom in basis._basis]
    if root.is_Rational:
        return rep_matrix_from_list(values, len(values), domain=root.domain)
    return _algebraic_field_coeffs(values, root.domain)


def _findroot_binary(poly: "Poly", symmetry: "PermutationGroup" = None) -> List[Root]:
    """
    Very easy implementation to find binary roots of the polynomial.
    """
    symmetry = MonomialManager(len(poly.gens), perm_group=symmetry)
    roots = set()
    for root in product([0, 1], repeat=len(poly.gens)):
        # root = RootRational(root)
        # if root.subs(poly, poly.gens) == 0:
        #     roots.append(root)
        if poly(*root) == 0:
            # roots.append(RootRational(root))
            roots.add(symmetry.standard_monom(root))
    roots = set(roots)
    all_zero = tuple([0] * len(poly.gens))
    if all_zero in roots:
        roots.remove(all_zero)
    roots = [Root(root) for root in roots]
    return roots


def get_sos_jet_data(
    poly: "Poly",
    ineq_constraints: Dict[Any, "Poly"],
    eq_constraints: Dict[Any, "Poly"],
    ineq_bases: Dict[Any, "SOSBasis"],
    *,
    roots: Sequence[Root] = (),
    perm_group: Optional["PermutationGroup"] = None,
    time_limit: Optional[Union[Callable, float]]=None,
) -> Tuple[Dict[Any, Matrix], List[_JetLayers]]:
    """Return point kernels and the ordered Gram coefficients along each line.

    Point evaluation also supports algebraic roots. The line data can be
    rescanned after imposing kernels without expanding the polynomials again.
    """
    time_limit = ArithmeticTimeout.make_checker(time_limit)
    if not roots:
        return {}, []
    lines = _collect_line_jets(
        poly, ineq_constraints, ineq_bases,
        eq_constraints.values(),
        roots=roots, symmetry=perm_group, check_time=time_limit
    )

    # A zero of the target alone does not justify point kernels when
    # nonzero weights of opposite signs can cancel.
    inequalities = complete_constraints_by_symmetry(
        [g for k, g in ineq_constraints.items() if len(ineq_bases[k])], perm_group)
    equalities = complete_constraints_by_symmetry(
        list(eq_constraints.values()), perm_group)
    point_roots = []
    for root in roots:
        time_limit()
        if not (root.is_Rational or root.domain.is_AlgebraicField):
            continue
        if root.eval(poly) != 0 or any(root.eval(h) != 0 for h in equalities):
            continue
        values = [root.eval(g) for g in inequalities]
        if (all(v.is_nonnegative is True for v in values)
                or all(v.is_nonpositive is True for v in values)):
            point_roots.append(root)

    nullspaces = {}
    for key, weight in ineq_constraints.items():
        basis = ineq_bases[key]
        if not len(basis):
            continue
        columns: Dict[Tuple, None] = {}
        for root in point_roots:
            time_limit()
            if root.eval(weight) != 0:
                span = _root_span(root, basis)
                columns.update(dict.fromkeys(tuple(col) for col in span.T.tolist() if any(col)))
        if columns:
            nullspaces[key] = Matrix(list(columns)).T
    return nullspaces, lines


def complete_constraints_by_symmetry(polys: List[Union["Poly", "PseudoPoly"]], symmetry=None):
    """
    Complete the constraints in the orbit of the symmetry group.
    Returns a list of polynomials.
    """
    if symmetry is None or len(polys) == 0 or symmetry.is_trivial:
        return list(polys)
    rep_set = set()
    polylize, gens = polys[0].__class__.new, polys[0].gens
    for poly in polys:
        rep = poly.rep
        rep_set.add(rep)
        pm = poly.algebra.permute_monom if isinstance(poly, PseudoPoly) else None
        for perm in symmetry.elements:
            if pm is not None:
                reorder = rep.per({pm(m, perm): c for m, c in rep.items()})
            else:
                reorder = poly.reorder(*[gens[i] for i in perm._array_form]).rep
            if rep == reorder:
                continue
            rep_set.add(reorder)
    return [polylize(rep, *gens) for rep in rep_set]
