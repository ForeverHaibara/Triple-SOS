from typing import List, Dict, Tuple, Optional, Iterator, Union, Any, TYPE_CHECKING

from sympy import Rational, Add, QQ
from sympy.combinatorics.named_groups import SymmetricGroup
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.rings import PolyElement
from sympy.polys.fields import FracField, FracElement
from sympy import MutableDenseMatrix as Matrix
from sympy.external.gmpy import lcm

from ..node import TransformNode
from ..dispatch import _dtype_make_reorder_func
from ..preprocess.signs import sign_sos
from ..preprocess.polynomial import SolvePolynomial
from ...utils.polytools import marginalize
from ...utils.monomials import _identify_symmetry_from_action, MonomialManager
from ...utils.roots import Root
from ...utils.expressions import CyclicSum
from ..problem import InequalityProblem
from ...sdp import ConeSDPBuilder

if TYPE_CHECKING:
    from sympy import Expr, Poly, Symbol
    from sympy.combinatorics.perm_groups import PermutationGroup
    from sympy.external.gmpy import MPQ
    from sympy.polys.rings import PolyRing
    # from ..problem import InequalityProblem


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
    seen = set()
    ret = []
    stabs = []
    for e in elems:
        if e in seen:
            continue

        stab = 0
        seen.add(e)
        for perm in G.elements:
            v = action(e, perm)
            if v == e:
                stab += 1
            else:
                seen.add(v)
        ret.append(e)
        stabs.append(stab)
    return ret, stabs


def _eval_in(f, x):
    if isinstance(f, PolyElement):
        return f.parent().dom.to_sympy(f.evaluate(list(enumerate(x))))
    elif isinstance(f, FracElement):
        return _eval_in(f.numer, x) / _eval_in(f.denom, x)


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
    dom: Union["PolyRing", FracField]
    exp_dom: Domain = QQ
    def __init__(self, dom: Union["PolyRing", FracField], exp_dom: Domain = QQ):
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


class RadicalMonomial(DomainElement, list):
    """
    Represents `prod([f**v for f, v in self])` where `f` are
    polynomial or rational functions, and `v` are rational numbers.
    """
    domain: RadicalMonomDomain
    def __init__(self, arg, domain: RadicalMonomDomain):
        self.domain = domain
        arg = self._canonicalize(arg)
        super().__init__(arg)

    def _canonicalize(self, arg):
        def check(x, r):
            if isinstance(x, (PolyElement, FracElement)):
                if x == x.parent().one:
                    return False
            return True

        dt = {}
        for f, r in arg:
            p, q = int(r.numerator), int(r.denominator)
            if p == 0:
                continue
            if p < 0:
                p, q = -p, -q
            # TODO: separate powers
            dt[QQ(1, q)] = dt.get(QQ(1, q), 1) * f**p
        arg = [(v, k) for k, v in dt.items() if check(v, k)]
        if not arg:
            arg = [(self.domain.dom.one, self.domain.exp_dom.one)]
        return arg


    @classmethod
    def new(cls, arg, domain: RadicalMonomDomain):
        return cls(arg, domain) # TODO: use a faster constructor

    def per(self, arg):
        return self.new(arg, self.domain)

    def __iter__(self) -> Iterator[Tuple[Union[PolyElement, FracElement], "MPQ"]]:
        return super().__iter__()

    def __neg__(self):
        negone = (-self.domain.dom.one, self.domain.exp_dom.one)
        return RadicalMonomial(self + [negone], self.domain)

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

    def total_degree(self):
        if not self:
            return self.domain.exp_dom.zero
        def d(f):
            if f.is_zero:
                return 0
            if isinstance(f, PolyElement):
                return max(map(sum, f.itermonoms()))
            elif isinstance(f, FracElement):
                return d(f.numer) - d(f.denom)
            return f.total_degree()
        return sum(d(f) * r for f, r in self)

    def to_ring(self):
        """
        If self.domain.dom is a FracField, convert it to PolyRing
        by using negative powers.
        """
        if not isinstance(self.domain.dom, FracField):
            return self
        rng = self.domain.dom.get_ring()
        arg = [(f.numer, p) for f, p in self] + [(f.denom, -p) for f, p in self]
        dom = RadicalMonomDomain(rng, self.domain.exp_dom)
        return RadicalMonomial(arg, dom)

    @property
    def is_power_positive(self):
        return all(p > 0 for f, p in self)


class RadicalProblem(InequalityProblem):
    """
    An inequality problem in the form of
    ```
    Σ_i (prod_j (f_{ij}(x)^{p_ij})) >= 0
    ```
    where `p_ij` are rational numbers, and `f_{ij}`
    are polynomial or rational functions.
    """
    _terms: List[RadicalMonomial]
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
    aux = info

    # check it is linear with respect to the symbols
    # TODO:
    # 1. remove nuisance symbols
    expr: "Poly" = problem.expr

    if not all(i[2] == 1 for i in info.values()):
        # only handle equality constraints now
        return None


    elim = list(info.keys())
    expr = marginalize(expr, *elim)
    info = {s: (marginalize(args[0], *elim), *args[1:])
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
        terms.append(term)

    new_problem = RadicalProblem.new(
        problem.expr, problem.ineq_constraints, problem.eq_constraints)

    rdom = RadicalMonomDomain(dom, QQ)
    new_problem._terms = [RadicalMonomial(term, rdom) for term in terms]

    new_problem.auxiliary_symbols = aux

    return new_problem


class CauchySolver(TransformNode):
    """
    Try to solve a problem using Cauchy-Schwarz inequality.

    Highly Experimental. DO NOT USE.
    """
    default_configs = {
        "lift_degree_limit": 4,
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


    def get_standard_form(self):
        """
        Get the standard form from the original problem.

        The standard form should be:
        `Σ (...)^r (>=/<=) (...)^s`
        """
        return as_radical_problem(self.problem)

    @staticmethod
    def construct_cauchy_ge_sdp(
        poly: Union[PolyElement, FracElement],
        gens: List["Symbol"],
        modules: List[PolyElement],
        lhs_power: int = 2,
        degree: int = 1,
        symmetry: Optional["PermutationGroup"] = None
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
        nvars = len(gens)
        hom = True # TODO: check this
        mg0 = MonomialManager(nvars, symmetry, is_homogeneous=hom)
        mg0base = mg0.base()

        p = lhs_power
        def action(x, perm):
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
                    mv = _eval_in(module, x2)
                    rt = Root(x2).as_vec(degree, symmetry=mg0base)
                    rt = projs[i] * rt
                    aff = aff + mv * rt
                    ws.append((rhsx * mv)**p)

                    trans.append(rt)
                affs.append(aff)

                A = Matrix.hstack(*trans).T
                As.append(A)

            aff = Matrix.vstack(*affs)
            A = Matrix.diag(*As)
            ws = Matrix(ws)
            builder.add_pnorm_cone(A, aff/10**2, p + 1, ws/10**(2*(p+1)))

        points = mg0.inv_monoms(12) # TODO: no magic number
        for x in points:
            sample(list(x))

        sdp = builder.build()
        return sdp, mgs


    def solve_cauchy_ge(self, configs: dict = {}):
        """
        Solve `Σ (...)^r >= (...)^s`
        """
        verbose = configs.get("verbose", False)
        degree = 2
        elim_vars = list(self.radical_problem.auxiliary_symbols.keys())
        gens = [g for g in self.problem.gens if g not in elim_vars]

        if len(gens) == 0:
            return

        def action(x, perm):
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
        lhs_power = lcm(*[term.inv_power_content() for term in lhs])
        if lhs_power == 1:
            # degenerated
            return None
        lhs = [term**lhs_power for term in lhs]


        data = [term.prod()[1] for term in lhs]
        G = SymmetricGroup(len(gens))
        G = _identify_symmetry_from_action(
            [data, [rhs]], G, action)
        # TODO: check multiplicity

        # collect expressions by symmetry
        modules, stabs = _clear_elements_by_symmetry(data, G, action)
        modules = [module / stab**lhs_power for module, stab in zip(modules, stabs)]
        if verbose:
            print("Identified Symmetry = %s" % \
                    str(G).replace('\n', '').replace('  ',''))
            print("Modules   =", modules, "\nStability =", stabs)

        sdp, mgs = self.construct_cauchy_ge_sdp(
            rhs, gens, modules, degree=degree, symmetry=G
        )
        codegrees = [degree] * len(modules) # TODO
        dofs = [len(mg.inv_monoms(d)) for mg, d in zip(mgs, codegrees)]
        dof = sum(dofs)

        # add a nonhomogeneous constraint
        sdp._x0_and_space['z'] = (
            Matrix([[-1]]), Matrix([1]*dof+[0]*(sdp.dof-dof)).T)

        val = None
        try:
            val = sdp.solve_obj([0]*sdp.dof)[:dof,:]
        except Exception as e:
            if verbose:
                print(e)
            if hasattr(e, 'y') and e.y is not None:
                val = Matrix(e.y[:dof])

        if verbose:
            print(val)

        if val is None:
            return None
        val = (val * 24).applyfunc(round) / 24
        if val.is_zero_matrix:
            return None

        def cyc_sum(x):
            x1 = x.zero
            for perm in G.elements:
                x1 = x1 + action(x, perm)
            return x1
        def to_poly(x):
            # return Poly.from_dict(dict(x), *gens, domain=x.parent().dom)
            return x.parent().to_sympy(x).as_poly(gens)
        muls = []
        cnt = 0
        for mg, dof, codgree in zip(mgs, dofs, codegrees):
            muls.append(mg.invarraylize(val[cnt:cnt+dof, :], gens, codgree))
            cnt += dof

        pow_of_sum = cyc_sum(sum([
            mul * to_poly(module) for mul, module in zip(muls, modules)]))**(lhs_power + 1)
        sum_of_pow = to_poly(rhs)**lhs_power * cyc_sum(sum([
            to_poly(module)**lhs_power*mul**(lhs_power+1) for mul, module in zip(muls, modules)]))
        new_expr = pow_of_sum - sum_of_pow
        _, __, ineqs, eqs = self.problem.separate_constraints(elim_vars)
        new_problem = self.problem.new(
            new_expr.as_poly(self.problem.gens), ineqs, eqs).remove_redundancy()

        rhs0 = rhs
        def restore(x):
            if x is None:
                return None
            rhs = to_poly(rhs0)
            lhs = (self.problem.expr + rhs).as_expr()
            rhs = rhs.as_expr()
            multiplier = sum([to_poly(module).as_expr()**lhs_power*mul.as_expr()**(lhs_power+1)
                              for mul, module in zip(muls, modules)])
            if not G.is_trivial:
                multiplier = CyclicSum(multiplier, gens, G)

            a_list, b_list = [], []
            exp = Rational(1, lhs_power)

            A_list, B_list = [], []

            for m, mul in zip(modules, muls):
                A_list.append((m.parent().to_sympy(m))**exp)
                B_list.append(mul.as_expr() * (m.parent().to_sympy(m))**exp)
                m0, mul0 = m, mul
                for perm in G.elements:
                    m = action(m0, perm)
                    mul = action(mul0, perm)
                    a_list.append((m.parent().to_sympy(m))**exp)
                    b_list.append(mul.as_expr() * (m.parent().to_sympy(m))**exp)

            if not G.is_trivial:
                A = CyclicSum(sum(A_list), gens, G)
                AB = CyclicSum(sum(ai * bi for ai, bi in zip(A_list, B_list)), gens, G)
            else:
                A, AB = sum(A_list), sum(ai * bi for ai, bi in zip(A_list, B_list))

            res = _cauchy_ge_residual(a_list, b_list, r=lhs_power, A=A, AB=AB)
            return (x + res) / ((lhs + rhs) * multiplier)

        return new_problem, restore
