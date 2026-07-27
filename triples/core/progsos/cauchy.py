from typing import List, Tuple, TYPE_CHECKING

from sympy import Poly, Rational, Add
from sympy.combinatorics.named_groups import SymmetricGroup
from sympy.polys.rings import PolyElement
from sympy.polys.fields import FracElement
from sympy import MutableDenseMatrix as Matrix

from ..node import TransformNode
from ..dispatch import _dtype_make_reorder_func
from ..preprocess.signs import sign_sos
from ..preprocess.polynomial import SolvePolynomial
from ...utils.polytools import marginalize
from ...utils.monomials import _identify_symmetry_from_action, MonomialManager
from ...utils.roots import Root
from ...utils.expressions import CyclicSum
from ...sdp import ConeSDPBuilder

if TYPE_CHECKING:
    from sympy import Expr, Symbol
    from sympy.combinatorics.perm_groups import PermutationGroup


def _get_symbols_constrained_once(problem) -> List[Tuple["Symbol", "Poly", "Expr", int]]:
    res = {}
    for eq, v in problem.eq_constraints.items():
        fs = eq.free_symbols
        if len(fs) == 1:
            # this symbols has a univariate constraint
            # and we should skip it
            eq, v = None, None

        for s in fs:
            if s not in res:
                res[s] = (s, eq, v, 1)
            else:
                # occurs more than once
                res[s] = (s, None, None, 1)

    for ineq, v in problem.ineq_constraints.items():
        fs = ineq.free_symbols
        if len(fs) > 1:
            for s in fs:
                if s not in res:
                    res[s] = (s, ineq, v, 0)
                else:
                    # occurs more than once
                    res[s] = (s, None, None, 0)
    return [v for v in res.values() if v[1] is not None]


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
            T = sum(A**k*S**(r-1-k)*sum(b[i]**(k-t)*b[j]**t for t in range(k+1)) for k in range(r))
            s.append(a[i]*a[j]*(b[i] - b[j])**2*T)
    return Add(*s)


class CauchySolver(TransformNode):
    """
    Try to solve a problem using Cauchy-Schwarz inequality.
    """
    default_configs = {
        "lift_degree_limit": 4,
    }
    std_form = None
    def explore(self, configs):
        if self.state == 0:
            self.state = 1
            self.std_form = self.get_standard_form()

        if self.std_form is None:
            self.state = -1
            self.finished = True
            return

        if self.state == 1:
            self.state = 2
            if self.std_form["sgn"] == 1:
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
        problem = self.problem
        info = _get_symbols_constrained_once(problem)
        if len(info) < 2:
            return None
        if len(info) == len(problem.gens):
            # all symbols are constrained once -> degenerated
            return None

        # check it is linear with respect to the symbols
        # TODO:
        # 1. relax it to monomials / quadratic forms
        # 2. remove nuisance symbols
        expr: "Poly" = problem.expr
        if not all(expr.degree(i[0]) == 1 for i in info):
            # TODO: check linearity
            return None

        if not all(i[3] == 1 for i in info):
            # only handle equality constraints now
            return None

        signs = problem.get_symbol_signs()
        sgn = 1

        expr = marginalize(expr, *[i[0] for i in info])
        lc = expr.LC()
        if sign_sos(lc, signs) is None:
            expr = -expr
            sgn = -1

        expr_sparse = expr.rep.to_dict()

        lhs = []

        # TODO: separate constraints handling and expression handling

        for idx, (x, constraint, value, is_eq) in enumerate(info):
            if value != 0:
                # not implemented
                return None

            poly = marginalize(constraint, *[i[0] for i in info])
            if len(poly.monoms()) != 2:
                continue

            monic = poly
            if poly.rep.LC() != poly.domain.one:
                monic = poly.monic()
            (d1, _), (d2, v2) = monic.rep.terms()
            if not (sum(d1) == d1[idx] and sum(d2) == d2[idx]):
                return None
            d1, d2 = d1[idx], d2[idx]

            # x**(d1 - d2) = -v2
            monom = (0,)*idx + (1,) + (0,)*(len(info) - idx - 1)
            cf = expr_sparse.get(monom, expr.domain.zero)

            dom = expr.domain.unify(monic.domain)
            cf = dom.convert_from(cf, expr.domain)
            v2 = dom.convert_from(v2, monic.domain)

            if (d1 - d2)%2 == 0 and sign_sos(dom.to_sympy(cf), signs) is None:
                return None

            lhs.append((monom, cf, -v2*cf**(d1 - d2), Rational(1, d1 - d2)))

        rhs = -expr_sparse.get((0,)*len(info), expr.domain.zero)

        return {
            "info": info,
            "lhs": lhs,
            "rhs": (rhs, Rational(1)),
            "sgn": sgn,
        }

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


    def solve_cauchy_ge(self, configs={}):
        """
        Solve `Σ (...)^r >= (...)^s`
        """
        verbose = configs.get("verbose", False)
        degree = 2
        elim_vars = {i[0] for i in self.std_form["info"]}
        gens = [g for g in self.problem.gens if g not in elim_vars]

        if len(gens) == 0:
            return

        def action(x, perm):
            return _dtype_make_reorder_func(x, gens)(~perm)
        from sympy.combinatorics import CyclicGroup
        data = [term[2] for term in self.std_form["lhs"]]
        G = SymmetricGroup(len(gens))
        G = _identify_symmetry_from_action(
            [data, [self.std_form["rhs"][0]]], G, action)
        # TODO: check multiplicity

        # collect expressions by symmetry
        modules, stabs = _clear_elements_by_symmetry(data, G, action)
        modules = [module / stab**2 for module, stab in zip(modules, stabs)]
        if verbose:
            print("Identified Symmetry = %s" % \
                    str(G).replace('\n', '').replace('  ',''))
            print("Modules   =", modules, "\nStability =", stabs)

        rhs = self.std_form["rhs"][0]

        sdp, mgs = self.construct_cauchy_ge_sdp(
            rhs, gens, modules, degree=degree, symmetry=G
        )
        codegrees =  [degree] * len(modules)
        dofs = [len(mg.inv_monoms(d)) for mg, d in zip(mgs, codegrees)]
        dof = sum(dofs)

        # add a nonhomogeneous constraint
        sdp._x0_and_space['z'] = (Matrix([[-1]]), Matrix([1]*dof+[0]*(sdp.dof-dof)).T)

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

        new_expr = cyc_sum(sum([mul * to_poly(module) for mul, module in zip(muls, modules)]))**(2 + 1)\
                - to_poly(rhs)**2 * cyc_sum(
                    sum([to_poly(module)**2*mul**3 for mul, module in zip(muls, modules)]))
        _, __, ineqs, eqs = self.problem.extract_constraints(elim_vars)
        new_problem = self.problem.new(
            new_expr.as_poly(self.problem.gens), ineqs, eqs).remove_redundancy()

        def restore(x):
            if x is None:
                return None
            rhs = to_poly(self.std_form["rhs"][0])
            lhs = (self.problem.expr + rhs).as_expr()
            rhs = rhs.as_expr()
            multiplier = sum([to_poly(module).as_expr()**2*mul.as_expr()**3
                              for mul, module in zip(muls, modules)])
            if not G.is_trivial:
                multiplier = CyclicSum(multiplier, gens, G)

            # a_list = [m.parent().to_sympy(m) for m in modules]
            # b_list = [mul.as_expr() for mul in muls]
            a_list, b_list = [], []
            for perm in G.elements:
                for m, mul in zip(modules, muls):
                    m = action(m, perm)
                    mul = action(mul, perm)
                    a_list.append((m.parent().to_sympy(m))**Rational(1,2))
                    b_list.append(mul.as_expr() * (m.parent().to_sympy(m))**Rational(1,2))

            res = _cauchy_ge_residual(a_list, b_list, r=2)
            return (x + res) / ((lhs + rhs) * multiplier)

        return new_problem, restore
