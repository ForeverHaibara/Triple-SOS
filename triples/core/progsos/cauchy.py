from typing import List, Tuple, TYPE_CHECKING

from sympy import Rational
from sympy.combinatorics.named_groups import SymmetricGroup
from sympy.polys.rings import PolyElement
from sympy.polys.fields import FracElement
from sympy import MutableDenseMatrix as Matrix

from ..node import TransformNode
from ..dispatch import _dtype_make_reorder_func
from ..preprocess.signs import sign_sos
from ...utils.polytools import marginalize
from ...utils.monomials import _identify_symmetry_from_action, MonomialManager
from ...utils.roots import Root
from ...sdp import ConeSDPBuilder

if TYPE_CHECKING:
    from sympy import Poly, Expr, Symbol
    from sympy.combinatorics.permutations import PermutationGroup


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


def _clear_elements_by_symmetry(elems, G: "PermutationGroup", action) -> list:
    seen = set()
    ret = []
    for e in elems:
        if e in seen:
            continue

        for perm in G.elements:
            v = action(e, perm)
            seen.add(v)
        ret.append(e)
    return ret


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
                self.solve_cauchy_ge()
                return


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

        for idx, (x, constraint, value, is_eq) in enumerate(info):
            if value != 0:
                # not implemented
                return None

            poly = marginalize(constraint, *[i[0] for i in info])
            if len(poly.monoms()) != 2:
                continue

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

            lhs.append((-v2*cf**(d1 - d2), Rational(1, d1 - d2)))

        rhs = -expr_sparse.get((0,)*len(info), expr.domain.zero)

        return {
            "info": info,
            "lhs": lhs,
            "rhs": (rhs, Rational(1)),
            "sgn": sgn,
        }


    def solve_cauchy_ge(self):
        """
        Solve `Σ (...)^r >= (...)^s`
        """
        degree = 1
        elim_vars = {i[0] for i in self.std_form["info"]}
        gens = [g for g in self.problem.gens if g not in elim_vars]

        if len(gens) == 0:
            return

        def action(x, perm):
            return _dtype_make_reorder_func(x, gens)(perm)

        data = [x for x, _ in self.std_form["lhs"]]
        G = SymmetricGroup(len(gens))
        G = _identify_symmetry_from_action(
            [data, [self.std_form["rhs"][0]]], G, action)

        # TODO: check homogeneity of the problem
        hom = True

        # collect expressions by symmetry
        modules = _clear_elements_by_symmetry(data, G, action)
        rhs = self.std_form["rhs"][0]
        mg = MonomialManager(len(gens), G, is_homogeneous=hom)
        dofs = [len(mg.inv_monoms(degree))] * len(modules)
        dof = sum(dofs)
        stabs = [mg.arraylize_sp(mg.invarraylize([1] * dof, gens, degree),
                                 degree, expand_cyc=True) for dof in dofs]

        def _eval_in(f, x):
            if isinstance(f, PolyElement):
                return int(f.evaluate(list(enumerate(x))))
            elif isinstance(f, FracElement):
                return _eval_in(f.numer, x) / _eval_in(f.denom, x)


        builder = ConeSDPBuilder(dof)
        def sample(x):
            affs = []
            ws = []
            As = []

            for i, module in enumerate(modules):
                rhsx = _eval_in(rhs, x)
                aff = Matrix.zeros(dofs[i], 1)
                trans = []
                for p in G.elements:
                    x2 = [x[i] for i in p._array_form]
                    mv = _eval_in(module, x2)
                    rt = Root(x2).as_vec(degree, symmetry=mg)
                    aff = aff + mv * rt
                    ws.append(rhsx**2 * mv**2)
                    trans.append(rt.multiply_elementwise(stabs[i]))
                affs.append(aff)

                A = Matrix.hstack(*trans).T
                As.append(A)

            aff = Matrix.vstack(*affs)
            A = Matrix.hstack(*As)
            builder.add_pnorm_cone(A, aff, 3, Matrix(ws))

        points = mg.inv_monoms(12)
        for x in points:
            sample([_ + 1 for _ in x])
        sdp = builder.build()

        sdp._x0_and_space['z'] = (Matrix([[-1]]), Matrix([1]*dof+[0]*(sdp.dof-dof)).T)
        val = sdp.solve_obj([0]*sdp.dof)
        return val
