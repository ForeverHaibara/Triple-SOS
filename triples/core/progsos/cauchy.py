from typing import List, Dict, Tuple, Optional, Any, TYPE_CHECKING

from sympy import Poly, Rational, Add
from sympy.combinatorics.named_groups import SymmetricGroup
from sympy.polys.rings import PolyElement
from sympy.polys.fields import FracElement
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
    from sympy import Expr, Symbol
    from sympy.combinatorics.perm_groups import PermutationGroup

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


def _get_symbol_radical(poly: Poly, i: int) -> Tuple[Any, int]:
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


def _cauchy_ge_residual(a: list, b: list, r: int = 2):
    n = len(a)
    if n != len(b):
        raise ValueError("a and b must have the same length")
    s = []
    A = sum(a)
    S = sum(ai * bi for ai, bi in zip(a, b))
    for i in range(n):
        for j in range(i+1, n):
            T = sum(A**k*S**(r-1-k)*sum(
                b[i]**(k-t)*b[j]**t for t in range(k+1)) for k in range(r))
            s.append(a[i]*a[j]*(b[i] - b[j])**2*T)
    return Add(*s)


class RadicalProblem(InequalityProblem):
    """
    An inequality problem in the form of
    ```
    Σ_i (prod_j (f_{ij}(x)^{p_ij})) >= 0
    ```
    where `p_ij` are rational numbers, and `f_{ij}`
    are polynomial or rational functions.
    """
    _terms: List[List[Tuple[Any, Rational]]]
    _separated_terms: Optional[Dict[Optional[int], List[Tuple[Any, Rational]]]]

    auxiliary_symbols: Dict["Symbol", Tuple["Poly", "Expr", int]]

    def terms(self):
        return self._terms

    def canonicalize_terms(self):
        self._separated_terms = None

        def check(x, r):
            if isinstance(x, (PolyElement, FracElement)):
                if x == x.parent().one:
                    return False
            return True

        for i, term in enumerate(self._terms):
            if len(term) == 1:
                continue
            dt = {}
            for f, r in term:
                p, q = int(r.numerator), int(r.denominator)
                # TODO: separate powers
                dt[Rational(1, q)] = dt.get(Rational(1, q), 1) * f**p
            term = [(v, k) for k, v in dt.items() if check(v, k)]
            if term:
                self._terms[i] = term
        return self

    def separate_sides(self) -> Dict[Optional[int], List[Tuple[Any, Rational]]]:
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
    # 1. relax it to monomials / quadratic forms
    # 2. remove nuisance symbols
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
        term = [(coeff, Rational(1)), *((v, Rational(d, r))
                    for d, (v, r) in zip(monom, rads.values()))]
        terms.append(term)

    new_problem = RadicalProblem.new(
        problem.expr, problem.ineq_constraints, problem.eq_constraints)

    new_problem._terms = terms
    new_problem.canonicalize_terms()
    
    new_problem.auxiliary_symbols = aux

    return new_problem


class CauchySolver(TransformNode):
    """
    Try to solve a problem using Cauchy-Schwarz inequality.
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
        poly,
        gens,
        modules,
        degree = 1,
        symmetry=None,
    ):
        nvars = len(gens)
        hom = True 
        mg0 = MonomialManager(nvars, symmetry, is_homogeneous=hom)
        mg0base = mg0.base()
  
        p = 2
        def action(x, perm):
            return _dtype_make_reorder_func(x, gens)(~perm)

        symms = [_identify_symmetry_from_action([[module]], symmetry, action) for module in modules]
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
            builder.add_pnorm_cone(A, aff/10**2, p + 1, Matrix(ws)/10**(2*(p+1)))

        points = mg0.inv_monoms(12) # TODO: no magic number
        for x in points:
            sample([_ for _ in x])

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

        rhs, rhs_power = sep[-1][0][0]
        rhs = -rhs
        if rhs_power != 1:
            # TODO: not implemented
            return None

        # compute lcm of powers
        lhs_power = 1
        for term in lhs:
            for _, p in term:
                lhs_power = lcm(lhs_power, int(p.denominator))
        lhs_power = int(lhs_power)
        if lhs_power == 1:
            # degenerated
            return None
        lhs = [[(k, int(r * lhs_power)) for k, r in term] for term in lhs]

        def prod(ls):
            v = 1
            for l, p in ls:
                v = v * l**p
            return v

        # TODO: compute the prod of each term
        data = [prod(term) for term in lhs]
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
        codegrees = [degree] * len(modules)
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
            if e.y is not None:
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

            # a_list = [m.parent().to_sympy(m) for m in modules]
            # b_list = [mul.as_expr() for mul in muls]
            a_list, b_list = [], []
            exp = Rational(1, lhs_power)
            for perm in G.elements:
                for m, mul in zip(modules, muls):
                    m = action(m, perm)
                    mul = action(mul, perm)
                    a_list.append((m.parent().to_sympy(m))**exp)
                    b_list.append(mul.as_expr() * (m.parent().to_sympy(m))**exp)

            res = _cauchy_ge_residual(a_list, b_list, r=lhs_power)
            return (x + res) / ((lhs + rhs) * multiplier)

        return new_problem, restore
