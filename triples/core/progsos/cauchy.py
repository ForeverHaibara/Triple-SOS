from typing import List, Dict, Tuple, Optional, Iterator, Union, Any, TYPE_CHECKING
from time import perf_counter

from sympy import Rational, Mul, Add, Dummy, QQ
from sympy.combinatorics.named_groups import SymmetricGroup
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.rings import PolyElement
from sympy.polys.fields import FracElement
from sympy import MutableDenseMatrix as Matrix
try:
    from sympy.external.gmpy import lcm
except ImportError:
    from math import gcd
    from functools import reduce
    lcm = lambda *args: reduce(lambda x, y: x*y//gcd(x, y), args, 1)

from ..node import TransformNode
from ..dispatch import (
    _dtype_make_reorder_func, _dtype_is_homogeneous
)
from ..preprocess.signs import sign_sos
from ..preprocess.polynomial import SolvePolynomial
from ..problem import InequalityProblem
from ..complexity import ProblemComplexity
from ...utils.polytools import marginalize
from ...utils.monomials import _identify_symmetry_from_action, MonomialManager
from ...utils.roots import Root
from ...utils.expressions import CyclicSum
from ...sdp import ConeSDPBuilder, ArithmeticTimeout

if TYPE_CHECKING:
    from sympy import Expr, Poly, Symbol
    from sympy.combinatorics.perm_groups import PermutationGroup
    from sympy.external.gmpy import MPQ
    from sympy.polys.rings import PolyRing
    from sympy.polys.fields import FracField
    # from ..problem import InequalityProblem


def rem_deg(a: int, b: int) -> int:
    """Requires `b > 0`. If `a < 0`, returns `-a`.
    If `a > 0`, returns `v` such that `(a + v)%b == 0`
    and `0 <= v < b`."""
    if a < 0:
        return -a
    r = a % b
    return b - r if r else r


def increment_poly(f: "Poly") -> "Poly":
    """
    Returns `f(a1 + x1, ..., an + xn) - f(a1, ..., an)`
    as a polynomial in (2n) variables.
    """
    new_gens = [Dummy('_%s'%g) for g in f.gens]
    p = f.as_expr().xreplace({g: (g + d) for g, d in zip(f.gens, new_gens)})
    q = p.as_poly(*f.gens, *new_gens, domain = f.domain)
    n = len(f.gens)
    tail = (0,)*n
    f = f.from_dict({
        m + tail: c for m, c in f.rep.terms()
    }, *q.gens, domain=f.domain)
    return q - f


def _total_degree(f) -> int:
    if not f:
        return 0
    if isinstance(f, PolyElement):
        return max(map(sum, f.itermonoms()))
    elif isinstance(f, FracElement):
        return _total_degree(f.numer) - _total_degree(f.denom)
    return f.total_degree()


def _get_symbols_constrained_once(
    problem: "InequalityProblem"
) -> Dict["Symbol", Tuple["Poly", "Expr", int]]:
    """
    Get symbols that are constrained once in the problem.

    Returns
    -------
    Dict["Symbol", Tuple["Poly", "Expr", int]]
        * Symbol: the symbol that is constrained once
        * Poly: the constraint polynomial
        * Expr: the constraint expression
        * int: 1 if the constraint is an equality, 0 if it is an inequality
    """
    res = {}
    for eq, v in problem.eq_constraints.items():
        fs = eq.free_symbols
        if len(fs) == 1:
            # this symbol has a univariate constraint
            # and we should discard the symbol
            eq, v = None, None

        for s in fs:
            if s not in res:
                res[s] = (eq, v, 1)
            else:
                # occurs more than once
                res[s] = (None, None, 1)

    for ineq, v in problem.ineq_constraints.items():
        fs = ineq.free_symbols
        if len(fs) > 1:
            for s in fs:
                if s not in res:
                    res[s] = (ineq, v, 0)
                else:
                    # occurs more than once
                    res[s] = (None, None, 0)
    return {k: v for k, v in res.items() if v[0] is not None}


def _get_symbol_radical(poly: "Poly", i: int) -> Tuple[Any, int]:
    """
    Given `poly == 0`, infer `x**r = expr`
    where `x` is the `i`-th generator.
    """
    # do not use marginalization, because we need to
    # ensure no other generators are mixed in the polynomial

    num_terms = len(poly.monoms())
    if num_terms != 2:
        return None, 0

    if poly.rep.LC() != poly.domain.one:
        poly = poly.monic()

    (m1, _), (m2, v2) = poly.rep.terms()
    d1, d2 = m1[i], m2[i]

    if not (sum(m1) == m1[i] and sum(m2) == m2[i]):
        return None, 0

    # x**(d1 - d2) + v2 == 0
    return -v2, d1 - d2


def _clear_elements_by_symmetry(elems, G: "PermutationGroup", action) -> Tuple[list, list]:
    from collections import Counter
    counter = Counter(elems)
    order = int(G.order())
    reps, stabs = [], []
    while counter:
        k, num = next(iter(counter.items()))
        orbit = {k}
        reps.append(k)
        for perm in G.elements:
            v = action(k, perm)
            orbit.add(v)

        for e in orbit:
            # check all elements in the orbit
            # has the same multiplicity
            if counter.get(e, 0) != num:
                raise ValueError(f"Element {e} has multiplicity"
                            f" {counter.get(e, 0)} instead of {num}")
            del counter[e]
        stabs.append(order // len(orbit)) # size of stabilizer

    return reps, stabs


def _eval_in(f, x):
    if isinstance(f, PolyElement):
        return f.parent().dom.to_sympy(f.evaluate(list(enumerate(x))))
    elif isinstance(f, FracElement):
        return _eval_in(f.numer, x) / _eval_in(f.denom, x)
    elif isinstance(f, RadicalMonomial):
        s = 1
        for t, v in f:
            s *= _eval_in(t, x)**v
        return s


def _cauchy_ge_residual(a: list, b: list, r: int = 2,
    A = None, AB = None
):
    """
    Prove that
    ```
    (sum a[i])**r * sum (a[i] * b[i]**(r+1)) - (sum (a[i]*b[i]))**(r+1)
    ```
    is nonnegative by direct sum-of-squares.
    """
    n = len(a)
    if n != len(b):
        raise ValueError("a and b must have the same length")

    s = []
    if A is None:
        A = sum(a)
    if AB is None:
        AB = sum(ai * bi for ai, bi in zip(a, b))
    for k in range(r):
        prefix = A**k*AB**(r-1-k)
        v = prefix * sum(a[i]*a[j]*(b[i] - b[j])**2*
                            sum(b[i]**(k-t)*b[j]**t for t in range(k+1))
                            for i in range(n) for j in range(i+1, n))
        s.append(v)
    return Add(*s)



class RadicalMonomDomain(Domain):
    dom: Union["PolyRing", "FracField"]
    exp_dom: Domain = QQ
    def __init__(self, dom: Union["PolyRing", "FracField"], exp_dom: Domain = QQ):
        self.dom = dom
        self.exp_dom = exp_dom

    @property
    def zero(self):
        return RadicalMonomial([], self)

    @property
    def one(self):
        return RadicalMonomial([(self.dom.one, self.exp_dom.one)], self)

    def __str__(self):
        return f"RadicalMonomDomain({self.dom}, {self.exp_dom})"


class RadicalMonomial(DomainElement):
    """
    Represents `prod([f**v for f, v in self])` where `f` are
    polynomial or rational functions, and `v` are rational numbers.

    When the tuple is empty, the monomial is zero (not one).

    Used internally.
    """
    __slots__ = ("args", "domain")

    args: Tuple[Tuple[Union[PolyElement, FracElement], "MPQ"]]
    domain: RadicalMonomDomain

    def __init__(self, args, domain):
        args = RadicalMonomial._canonicalize(domain, args)
        self.args = tuple(args)
        self.domain = domain

    def __hash__(self):
        return hash(self.args)

    def __eq__(self, other):
        if not isinstance(other, RadicalMonomial):
            return NotImplemented
        return self.args == other.args

    def __str__(self):
        return str(self.args)

    def __repr__(self):
        return repr(self.args)

    def __len__(self):
        return len(self.args)

    def __bool__(self):
        return bool(self.args)

    def __getitem__(self, i):
        return self.args[i]

    def __contains__(self, item):
        return item in self.args


    @staticmethod
    def _canonicalize(domain, args):
        def check(x, r):
            if isinstance(x, (PolyElement, FracElement)):
                if x == x.parent().one:
                    return False
            return True

        dt = {}
        is_field = domain.dom.is_Field
        for f, r in args:
            p, q = int(r.numerator), int(r.denominator)
            if p == 0:
                continue
            if (not is_field) and (p < 0):
                p, q = -p, -q

            # dt[QQ(1, q)] = dt.get(QQ(1, q), 1) * f**p
            dt[QQ(p, q)] = dt.get(QQ(p, q), 1) * f
        args = [(v, k) for k, v in sorted(dt.items()) if check(v, k)]
        if not args:
            args = [(domain.dom.one, domain.exp_dom.one)]
        return args

    def __iter__(self) -> Iterator[Tuple[Union[PolyElement, FracElement], "MPQ"]]:
        return iter(self.args)


    @classmethod
    def new(cls, arg, domain: RadicalMonomDomain):
        return cls(arg, domain) # TODO: use a faster constructor

    def per(self, arg):
        return self.new(arg, self.domain)

    def __neg__(self):
        negone = (-self.domain.dom.one, self.domain.exp_dom.one)
        return RadicalMonomial(self + (negone,), self.domain)

    def __mul__(self, other):
        if isinstance(other, RadicalMonomial):
            return RadicalMonomial(self.args + other.args, self.domain)
        x = self.domain.dom(other)
        if x == self.domain.dom.one:
            return self
        if x == self.domain.dom.zero:
            return self.domain.zero
        return RadicalMonomial(self.args + ((x, self.domain.exp_dom.one),), self.domain)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, RadicalMonomial):
            return RadicalMonomial(self.args + (-other.args), self.domain)
        x = self.domain.dom(other)
        if x == self.domain.dom.one:
            return self
        if x == self.domain.dom.zero:
            return self.domain.zero
        return RadicalMonomial(self.args + ((x, -self.domain.exp_dom.one),), self.domain)

    def __pow__(self, exp):
        return self.per([(f, p*exp) for f, p in self])

    def inv_power_content(self):
        return lcm(*[r.denominator for f, r in self])

    def power_primitive(self):
        p = self.inv_power_content()
        return p, self**p

    def prod(self):
        p = self.inv_power_content()
        m = self.domain.dom.one
        for f, v in self:
            m = m * f**int(v * p)
        return p, m

    def total_degree(self) -> "MPQ":
        if not self:
            return self.domain.exp_dom.zero
        return sum(_total_degree(f) * r for f, r in self)

    def to_ring(self):
        """
        If self.domain.dom is a FracField, convert it to PolyRing
        by using negative powers.
        """
        if not self.domain.dom.is_Field:
            return self
        rng = self.domain.dom.get_ring()
        arg = [(f.numer, p) for f, p in self] + [(f.denom, -p) for f, p in self]
        dom = RadicalMonomDomain(rng, self.domain.exp_dom)
        return RadicalMonomial(arg, dom)

    def rem_degrees(self, n: int):
        """Should only be used when all powers are integral."""
        return [rem_deg(int(d), n) for _, d in self]

    def quo_degrees(self, n: int):
        """Should only be used when all powers are integral."""
        return [(int(d) + rem_deg(int(d), n))//n for _, d in self]

    def to_rem_element(self, n: int):
        if not self:
            return self.domain.dom.zero
        x = self.domain.dom.one
        for (f, _), d in zip(self, self.rem_degrees(n)):
            x = x * f**d
        return x

    def to_quo_element(self, n: int):
        if not self:
            return self.domain.dom.zero
        x = self.domain.dom.one
        for (f, _), d in zip(self, self.quo_degrees(n)):
            x = x * f**d
        return x

    def to_sympy(self):
        if not self:
            return Rational(0)
        to_sp = self.domain.dom.to_sympy
        exp_to_sp = self.domain.exp_dom.to_sympy
        return Mul(*[to_sp(f)**exp_to_sp(p) for f, p in self])

    def to_rem_sympy(self, n: int):
        if not self:
            return Rational(0)
        to_sp = self.domain.dom.to_sympy
        exp_to_sp = self.domain.exp_dom.to_sympy
        ds = self.rem_degrees(n)
        return Mul(*[to_sp(f)**exp_to_sp(p) for (f, _), p in zip(self, ds)])

    def to_quo_sympy(self, n: int):
        if not self:
            return Rational(0)
        to_sp = self.domain.dom.to_sympy
        exp_to_sp = self.domain.exp_dom.to_sympy
        ds = self.quo_degrees(n)
        return Mul(*[to_sp(f)**exp_to_sp(p) for (f, _), p in zip(self, ds)])

    @property
    def is_power_positive(self):
        return all(p > 0 for f, p in self)

    @property
    def is_homogeneous(self):
        return all(_dtype_is_homogeneous(f) for f, p in self)


class RadicalProblem(InequalityProblem):
    """
    An inequality problem in the form of
    ```
    Σ_i (prod_j (f_{ij}(x)^{p_ij})) >= 0
    ```
    where `p_ij` are rational numbers, and `f_{ij}`
    are polynomial or rational functions.
    """
    _terms: Dict[RadicalMonomial, "Expr"]
    _separated_terms: Optional[Dict[Optional[int], List[RadicalMonomial]]] = None

    auxiliary_symbols: Dict["Symbol", Tuple["Poly", "Expr", int]]

    def terms(self):
        return self._terms

    def separate_sides(self) -> Dict[Optional[int], List[RadicalMonomial]]:
        """
        Split data by positive, negative and undetermined terms.
        """
        if self._separated_terms is not None:
            return self._separated_terms

        signs = self.get_symbol_signs()
        terms = {1: [], -1: [], 0: [], None: []}
        for term in self.terms():
            sgn = 1
            for f, p in term:
                if p.numerator%2 != 0 and p.denominator%2 != 0:
                    f_proof = sign_sos(f, signs)
                    if f_proof is None:
                        f_proof = sign_sos(-f, signs)
                        if f_proof is None:
                            sgn = None
                            break
                        else:
                            sgn = -sgn
            terms[sgn].append(term)
        self._separated_terms = terms
        return terms

    def side_sign(self) -> int:
        sep = self.separate_sides()
        if len(sep[None]) == 0 and len(sep[0]) == 0:
            if len(sep[-1]) == 1 and sep[1]:
                return 1
            if len(sep[1]) == 1 and sep[-1]:
                return -1
        return 0


def as_radical_problem(problem: InequalityProblem) -> Optional[RadicalProblem]:
    """
    Convert a problem to a radical problem.

    Returns
    -------
    Optional[RadicalProblem]
        The radical problem.
    """
    info = _get_symbols_constrained_once(problem)
    if len(info) < 2:
        # TODO:
        return None
    if len(info) == len(problem.gens):
        # all symbols are constrained once -> degenerated
        return None

    # check if all symbols are nonnegative
    signs = problem.get_symbol_signs()
    if not all(signs.get(s, (0, 0))[0] == 1 for s in info.keys()):
        return None

    aux = info

    # check it is linear with respect to the symbols
    # TODO:
    # 1. remove nuisance symbols
    expr: "Poly" = problem.expr

    if not all(i[2] == 1 for i in info.values()):
        # only handle equality constraints now
        return None
    if not all(i[1] == 0 for i in info.values()):
        # requires only zero constraints now
        return None


    elim = list(info.keys())
    expr = marginalize(expr, *elim)
    info = {s: (marginalize(args[0], *elim).monic(), *args[1:])
            for s, args in info.items()}

    # unify domains
    dom = expr.domain
    for args in info.values():
        dom = dom.unify(args[0].domain)

    expr = expr.set_domain(dom)
    info = {s: (args[0].set_domain(dom), *args[1:])
            for s, args in info.items()}
    rads = {s: _get_symbol_radical(args[0], i)
            for i, (s, args) in enumerate(info.items())}

    if any(v is None for v, r in rads.values()):
        return None

    terms = []
    for monom, coeff in expr.rep.terms():
        term = [(coeff, QQ(1)), *((v, QQ(d, r))
                    for d, (v, r) in zip(monom, rads.values()))]
        alias = dom.to_sympy(coeff)
        alias = Mul(alias, *[g**i for g, i in zip(expr.gens, monom)])
        terms.append((term, alias))

    new_problem = RadicalProblem.new(
        problem.expr, problem.ineq_constraints, problem.eq_constraints)

    rdom = RadicalMonomDomain(dom, QQ)
    new_problem._terms = {RadicalMonomial(term, rdom): alias
                                for term, alias in terms}
    # TODO: it should collect the terms
    new_problem.auxiliary_symbols = aux

    return new_problem


class CauchySolver(TransformNode):
    """
    Try to solve a problem using Cauchy-Schwarz inequality.

    The solution of this function involves radicals. Only
    use it when `irrational_expr` is True.

    Highly Experimental. Use with caution.
    """
    default_configs = {
        "lift_degree_limit": 4,
        "sample_density": 12,
    }
    radical_problem: Optional[RadicalProblem] = None
    def explore(self, configs):
        if self.state == 0:
            self.state = 1
            self.radical_problem = self.get_standard_form()

            if self.radical_problem is None:
                self.state = -1
                self.finished = True
            return

        if self.state == 1:
            self.state = 2
            sgn = self.radical_problem.side_sign()
            if sgn == 1:
                result = self.solve_cauchy_ge(configs)
                if result is not None:
                    new_problem, restore = result
                    new_node = SolvePolynomial(new_problem)
                    self.children.append(new_node)
                    self.restorations[new_node] = restore


        self.state = -1

    def _evaluate_complexity(self) -> ProblemComplexity:
        # Fast in most cases
        if self.state == 0:
            return ProblemComplexity(0.01, 1.)
        return ProblemComplexity(5., .9)

    def get_standard_form(self):
        """
        Get the standard form from the original problem.

        The standard form should be:
        `Σ (...)^r (>=/<=) (...)^s`
        """
        return as_radical_problem(self.problem)

    @staticmethod
    def construct_cauchy_sdp_ge(
        poly: Union[PolyElement, FracElement],
        gens: List["Symbol"],
        modules: List[RadicalMonomial],
        lhs_power: int,
        degree: int = 1,
        symmetry: Optional["PermutationGroup"] = None,
        configs: Optional[Dict[str, Any]] = None,
    ):
        """
        Given modules `F1`, ..., `Fn`, try to find polynomials
        `G1`, ..., `Gn` such that (*) holds:

        ```
        (Σ (Fi * Gi))**(r+1) >= poly**r * (Σ Fi**r * Gi**(r+1)) # (*)
        ```

        If (*) holds, then it implies
        ```
        Σ Fi**(1/r) >= poly
        ```
        by (generalized) Cauchy-Schwarz inequality. To solve for
        the coefficients of `Gi`, we sample enough points to
        require the constraint (*) holds at these points, and then
        solve the feasibility problem by formulating it as a cone
        program.
        """


        modules = [m.to_ring() for m in modules]

        module_rem_degs = [m.rem_degrees(lhs_power + 1) for m in modules]
        module_quo_degs = [m.quo_degrees(lhs_power + 1) for m in modules]

        nvars = len(gens)

        hom = _dtype_is_homogeneous(poly) and all(
            m.is_homogeneous for m in modules)
        # if not hom:
        #     # not implemented
        #     return None
        if hom:
            poly_degree = _total_degree(poly)
            if not all(m.total_degree() == poly_degree for m in modules):
                # the degree does not match
                return None

        mg0 = MonomialManager(nvars, symmetry, is_homogeneous=hom)
        mg0base = mg0.base()

        p = lhs_power
        def action(x, perm):
            if isinstance(x, RadicalMonomial):
                return x.per([(action(v, perm), r) for v, r in x])
            return _dtype_make_reorder_func(x, gens)(~perm)

        symms = [_identify_symmetry_from_action(
            [[module]], symmetry, action) for module in modules]
        mgs = [MonomialManager(nvars, symm,
                               is_homogeneous=hom) for symm in symms]
        projs = [mg.proj_matrix(degree) for mg in mgs]

        dofs = [len(mg.inv_monoms(degree)) for mg in mgs]

        builder = ConeSDPBuilder(sum(dofs))

        def sample(x):
            affs = []
            ws = []
            As = []

            for i, module in enumerate(modules):
                rhsx = _eval_in(poly, x)
                aff = Matrix.zeros(dofs[i], 1)
                trans = []
                for perm in symmetry.elements:
                    x2 = [x[j] for j in perm._array_form]
                    mv = [_eval_in(t, x2) for t, _ in module]
                    if any(t == 0 and d < 0 for t, d in zip(mv, module_rem_degs[i])):
                        # prevent division by zero
                        return

                    qmv = 1 # product of mv
                    for t, d in zip(mv, module_quo_degs[i]):
                        qmv *= t**d
                    rmv = 1
                    for t, d in zip(mv, module_rem_degs[i]):
                        rmv *= t**d

                    rt = Root(x2).as_vec(degree, symmetry=mg0base)
                    rt = projs[i] * rt

                    aff = aff + qmv * rt
                    ws.append(rhsx**p * rmv)

                    trans.append(rt)
                affs.append(aff)

                A = Matrix.hstack(*trans).T
                As.append(A)

            aff = Matrix.vstack(*affs)
            A = Matrix.diag(*As)
            ws = Matrix(ws)
            # normalize for numerical stability
            builder.add_pnorm_cone(A, aff/10**2, p + 1, ws/10**(2*(p+1)))

        sample_density = configs.get("sample_density", 12)
        points = mg0.inv_monoms(sample_density)
        for x in points:
            sample(list(x))

        sdp = builder.build()
        return sdp, mgs


    def solve_cauchy_ge(self, configs: dict = {}):
        """
        Solve `Σ (...)**(1/lhs_power) >= RHS**rhs_power`.

        See also `construct_cauchy_sdp_ge`.
        """
        verbose = configs.get("verbose", False)
        start_time = perf_counter()

        degree = 2
        elim_vars = list(self.radical_problem.auxiliary_symbols.keys())
        gens = [g for g in self.problem.gens if g not in elim_vars]

        if len(gens) == 0:
            return

        def action(x, perm):
            if isinstance(x, RadicalMonomial):
                return x.per([(action(v, perm), r) for v, r in x])
            return _dtype_make_reorder_func(x, gens)(~perm)

        sep = self.radical_problem.separate_sides()
        lhs = sep[1]

        if len(sep[-1][0]) != 1:
            # not implemented
            return None

        rhs_power, rhs = sep[-1][0].prod()
        if rhs_power != 1:
            # TODO: not implemented
            return None
        rhs = -rhs

        # compute lcm of powers
        lhs_power = int(lcm(*[term.inv_power_content() for term in lhs]))
        if lhs_power == 1:
            # degenerated
            return None
        lhs = [term**lhs_power for term in lhs]


        G = SymmetricGroup(len(gens))
        G = _identify_symmetry_from_action(
            [lhs, [rhs]], G, action)

        # collect expressions by symmetry
        modules, stabs = _clear_elements_by_symmetry(lhs, G, action)
        modules: List[RadicalMonomial]

        modules = [m / s**lhs_power for m, s in zip(modules, stabs)]
        modules = [m.to_ring() for m in modules]
        if verbose:
            print("Identified Symmetry = %s" % \
                    str(G).replace('\n', '').replace('  ',''))
            print("Modules   =", modules, "\nStability =", stabs)

        result = self.construct_cauchy_sdp_ge(
            rhs, gens, modules, lhs_power,
            degree=degree, symmetry=G, configs=configs
        )
        if result is not None:
            sdp, mgs = result

        codegrees = [degree] * len(modules) # TODO
        dofs = [len(mg.inv_monoms(d)) for mg, d in zip(mgs, codegrees)]
        dof = sum(dofs)

        # add a nonhomogeneous constraint
        sdp._x0_and_space['z'] = (
            Matrix([[-1]]), Matrix([1]*dof+[0]*(sdp.dof-dof)).T)

        if verbose:
            sdp.print_graph(short = 2)

        if configs.get("time_limit") is not None:
            time_limit = configs["time_limit"] - (perf_counter() - start_time)
            if time_limit < 0:
                self.finished = True
                return

        if sdp.dof == 0: # can it happen?
            self.finished = True
            return

        val = None
        try:
            val = sdp.solve_obj(
                [0]*sdp.dof,
                time_limit=time_limit,
                verbose=True if float(verbose) > 2 else False
            )[:dof,:]
        except Exception as e:
            if verbose:
                print(e)
            if isinstance(e, ArithmeticTimeout):
                self.finished = True
                return
            if hasattr(e, 'y') and e.y is not None:
                val = Matrix(e.y[:dof])

        if verbose:
            print("Found a numerical solution...")

        if val is None:
            return None
        val = (val * 24).applyfunc(round) / 24
        if val.is_zero_matrix:
            return None

        # TODO: project to equality cases
        multipliers = self.get_multipliers_ge(
            val, gens, mgs, dofs, degree
        )

        new_problem = self.get_new_problem_ge(
            self.problem, rhs, gens, elim_vars, modules, multipliers,
            lhs_power, symmetry=G
        )

        restore = self.get_restoration_ge(
            self.problem, rhs, gens, modules, multipliers,
            lhs_power, symmetry=G
        )
        return new_problem, restore


    @staticmethod
    def get_multipliers_ge(
        y,
        gens: List["Symbol"],
        mgs: List[MonomialManager],
        dofs: List[int],
        degree: int,
    ):
        codegrees = [degree] * len(mgs)
        muls = []
        cnt = 0
        for mg, dof, codgree in zip(mgs, dofs, codegrees):
            muls.append(mg.invarraylize(y[cnt:cnt+dof, :], gens, codgree))
            cnt += dof
        return muls


    @staticmethod
    def get_new_problem_ge(
        problem: InequalityProblem,
        poly: PolyElement,
        gens: List["Symbol"],
        elim_gens: List["Symbol"],
        modules: List[RadicalMonomial],
        multipliers: List["Poly"],
        lhs_power: int,
        rhs_power: int = 1,
        symmetry: Optional["PermutationGroup"] = None,
    ):
        assert rhs_power == 1

        def action(x, perm):
            if isinstance(x, RadicalMonomial):
                return x.per([(action(v, perm), r) for v, r in x])
            return _dtype_make_reorder_func(x, gens)(~perm)

        def cyc_sum(x):
            x1 = x.zero
            if symmetry is not None:
                for perm in symmetry.elements:
                    x1 = x1 + action(x, perm)
            return x1

        def to_poly(x):
            # return Poly.from_dict(dict(x), *gens, domain=x.parent().dom)
            return x.parent().to_sympy(x).as_poly(gens)

        fg_terms = [
           g * to_poly(f.to_quo_element(lhs_power + 1)) for g, f in zip(multipliers, modules)
        ]
        pow_of_sum = cyc_sum(sum(fg_terms))**(lhs_power + 1)

        f_pow_g_pow_terms = [
            to_poly(f.to_rem_element(lhs_power + 1)) * g**(lhs_power + 1)
                for g, f in zip(multipliers, modules)
        ]
        sum_of_pow = to_poly(poly)**lhs_power * cyc_sum(sum(f_pow_g_pow_terms))
        new_expr = pow_of_sum - sum_of_pow
        _, __, ineqs, eqs = problem.separate_constraints(elim_gens)
        new_problem = problem.new(
            new_expr.as_poly(problem.gens), ineqs, eqs).remove_redundancy()

        return new_problem


    @staticmethod
    def get_restoration_ge(
        problem,
        poly,
        gens: List["Symbol"],
        modules: List[RadicalMonomial],
        multipliers: List["Poly"],
        lhs_power: int,
        symmetry: Optional["PermutationGroup"] = None,
    ):
        """
        Get the function that restores the solution
        from the transformed problem to the original problem.
        """
        G = symmetry
        def to_poly(x):
            # return Poly.from_dict(dict(x), *gens, domain=x.parent().dom)
            return x.parent().to_sympy(x).as_poly(gens)

        def action(x, perm):
            if isinstance(x, RadicalMonomial):
                return x.per([(action(v, perm), r) for v, r in x])
            return _dtype_make_reorder_func(x, gens)(~perm)

        def restore(x: Optional["Expr"]) -> Optional["Expr"]:
            if x is None:
                return None
            rhs = to_poly(poly)
            lhs = (problem.expr + rhs).as_expr()
            rhs = rhs.as_expr()
            multiplier = sum([
                module.to_rem_sympy(lhs_power + 1)\
                    *mul.as_expr()**(lhs_power + 1)
                        for mul, module in zip(multipliers, modules)])
            if (G is not None) and not G.is_trivial:
                multiplier = CyclicSum(multiplier, gens, G)

            a_list, b_list = [], []
            exp = Rational(1, lhs_power)

            A_list, B_list = [], []

            def compute_b(m: RadicalMonomial):
                # m.to_rem_sympy(lhs_power + 1) / m.to_sympy()**exp)**Rational(1, lhs_power+1)
                rems = m.rem_degrees(lhs_power + 1)
                ds = [(r - exp*d)/(lhs_power + 1) for r, (_, d) in zip(rems, m)]
                val = Rational(1)
                to_sp = m.domain.dom.to_sympy
                for (f, _), d in zip(m, ds):
                    val *= to_sp(f)**d
                return val

            for m, mul in zip(modules, multipliers):
                A_list.append((m.to_sympy())**exp)
                B_list.append(mul.as_expr() * compute_b(m))
                m0, mul0 = m, mul
                for perm in G.elements:
                    m = action(m0, perm)
                    mul = action(mul0, perm)
                    a_list.append((m.to_sympy())**exp)
                    b_list.append(mul.as_expr() * compute_b(m))

            if (G is not None) and not G.is_trivial:
                A = CyclicSum(sum(A_list), gens, G)
                AB = CyclicSum(sum(ai * bi for ai, bi in zip(A_list, B_list)), gens, G)
            else:
                A, AB = sum(A_list), sum(ai * bi for ai, bi in zip(A_list, B_list))

            res = _cauchy_ge_residual(a_list, b_list, r=lhs_power, A=A, AB=AB)
            return (x + res) / ((lhs + rhs) * multiplier)

        return restore
