from typing import Generator, Dict, Tuple, List, Optional

from sympy import Poly, Expr, Symbol, Mul

from .basis import LinearBasis, quadratic_difference, _callable_expr
from ...utils import MonomialManager, generate_monoms, clear_polys_by_symmetry


class LinearBasisMultiplier(LinearBasis):
    """
    For example, if we want to find

        CyclicSum(x0 * a**2 + x1 * a*b) * f(a,b,c) = CyclicSum(y1 * g1(a,b,c) + y2 * g2(a,b,c) + ...)

    then it is equivalent to

        RHS + x0 * CyclicSum(-a**2) * f(a,b,c) + x1 * CyclicSum(-a*b) * f(a,b,c) = 0.

    This converts the problem to a usual linear programming by adding the basis
    CyclicSum(-a**2)*f and CyclicSum(-a*b)*f to the linear programming.
    """
    def __init__(self, poly: Poly, multiplier: _callable_expr):
        self.poly = poly
        self._tangent = multiplier
    @property
    def multiplier(self) -> Expr:
        return self._tangent(self.poly.gens)
    def nvars(self) -> int:
        return len(self.poly.gens)
    def as_poly(self, symbols) -> Poly:
        poly = (self.poly * (-self._tangent(self.poly.gens, poly=True)))
        poly.gens = symbols
        return poly
    def as_expr(self, symbols) -> Expr:
        return (self.poly.as_expr() * self.multiplier).xreplace(dict(zip(self.poly.gens, symbols)))

    @classmethod
    def from_expr(cls, poly: Poly, expr: Expr, p: Optional[Poly] = None) -> 'LinearBasisMultiplier':
        return cls(poly, _callable_expr.from_expr(expr, poly.gens, p))

def lift_degree(
    poly: Poly,
    ineq_constraints: Dict[Poly, Expr],
    symmetry: MonomialManager,
    degree_limit: int = 1000,
    lift_degree_limit: int = 4
) -> Generator[Dict, None, None]:
    """
    Hilbert's problem has shown that not every positive polynomial can be written as a sum of squares.
    However, we can write it as sum of rational functions. As a result, we can write
    f(a,b,c) = g(a,b,c) / h(a,b,c) where g and h are both positive. In other words,
    f(a,b,c) * h(a,b,c) = g(a,b,c).

    In practice, we can try out h(a,b,c) = Sum(a), h(a,b,c) = Sum(a^2-ab + xab) and so on.
    This `lift_degree` function would generate the h(a,b,c) and associated information.

    Parameters
    ----------
    poly: Poly
        The target polynomial.
    ineq_constraints: Dict[Poly, Expr]
        Inequality constraints added to the problem.
    symmetry: MonomialManager
        The symmetry of the polynomial.
    degree_limit: int
        When the degree of f(a,b,c) * h(a,b,c) is larger than this limit,
        we stop to save computation resources.
    lift_degree_limit: int
        The degree of h(a,b,c) is at most this limit.

    Yields
    ----------
    Dict containing following items:
        poly: Poly
            The f(a,b,c) * h(a,b,c).
        multiplier: Expr
            The multiplier h(a,b,c).
        basis: List[LinearBasisMultiplier]
            The additional basis to be added to the linear programming.
            See details in `LinearBasisMultiplier`.
        degree: int
            The degree of f(a,b,c) * h(a,b,c).
        add_degree: int
            The degree of h(a,b,c).
    """
    n = poly.total_degree()
    nvars = len(poly.gens)
    symbols = poly.gens
    n_plus = 0

    while n + n_plus <= degree_limit and n_plus <= lift_degree_limit:
        multipliers = _get_multipliers(ineq_constraints, symbols, n_plus, symmetry=symmetry)
        basis = [LinearBasisMultiplier.from_expr(poly, e, p) for p, e in multipliers]

        if len(basis) > 0:
            yield {
                'basis': basis,
                'degree': n + n_plus,
                'add_degree': n_plus
            }

        n_plus += 1



def _get_multipliers(ineq_constraints: Dict[Poly, Expr], symbols: Tuple[Symbol, ...], n_plus: int,
                        symmetry: MonomialManager, preordering: str ='linear') -> Dict[Poly, Expr]:
    if preordering == 'linear':
        ineq_constraints = [(k, v) for k, v in ineq_constraints.items() if k.is_linear
                            and k.total_degree() == 1]
    else:
        raise ValueError(f"Preordering {preordering} is not supported.")
        ineq_constraints = ineq_constraints.items()

    multipliers = None
    if n_plus == 2:
        multipliers = quadratic_difference(symbols)
        multipliers = [(Poly(e, *symbols), e) for e in multipliers]
        def cross_mul(x):
            return [(x[i][0] * x[j][0], x[i][1] * x[j][1]) for i in range(len(x)) for j in range(i+1, len(x))]
        multipliers.extend(cross_mul(ineq_constraints))

    # TODO: this should be more carefully considered???
    if multipliers is None:
        # default case
        multipliers = []
        poly_one = Poly(1, *symbols)
        n_constraints = len(ineq_constraints)
        for power in generate_monoms(len(ineq_constraints), n_plus)[1]:
            mul_poly = poly_one
            for i in range(n_constraints):
                if power[i] > 0:
                    mul_poly *= ineq_constraints[i][0]**power[i]
            mul_expr = Mul(*(ineq_constraints[i][1]**power[i] for i in range(n_constraints)))
            multipliers.append((mul_poly, mul_expr))

    if n_plus > 2 and n_plus % 2 == 0:
        for power in generate_monoms(len(symbols), n_plus//2)[1]:
            mul_poly = Poly({tuple(p*2 for p in power): 1}, *symbols)
            mul_expr = Mul(*(si**(pi*2) for si, pi in zip(symbols, power)))
            multipliers.append((mul_poly, mul_expr))

    multipliers = clear_polys_by_symmetry(multipliers, symbols, symmetry)

    return multipliers
