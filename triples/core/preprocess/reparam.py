"""
This is still highly experimental. Do not use this.
"""

from typing import Tuple, Union, List, Callable

from sympy import Poly, Expr, Integer, Mul, Dummy, gcd

from ..node import ProofNode, TransformNode

def _solver(problem) -> ProofNode:
    from .polynomial import SolvePolynomial
    return SolvePolynomial(problem)

def _perfect_power(x: int, l: int) -> Union[Tuple[Expr, int], bool]:
    l = int(l)
    if l == 1:
        return (x, l)
    if x == 1 or x == 0:
        return (x, l)
    if x == -1:
        return (x, l) if l % 2 == 1 else False
    from sympy import perfect_power
    z = perfect_power(x, [l])
    if not z:
        return False
    return (z[0], int(z[1]))


class Reparametrization(TransformNode):
    """
    Warning: This is still in development. Avoid using this.
    See `elimination.py` instead.

    Eliminate some of the equality constraints by reparametrizing the variables.
    Helper functions of reparametrization should identify whether
    a multivariate polynomial can be reparametrized (heuristically)
    and returns the substitution. The substitution must make the equality zero.
    """
    default_configs = {
        'irrational_number': False,
        'irrational_expr': False,
    }
    finished = True
    def explore(self, configs):
        if self.state != 0:
            return

        def make_restoration(restorations: List[Callable]) -> Callable:
            restorations_ = restorations.copy()
            def restoration(x: Expr) -> Expr:
                for r in restorations_[::-1]:
                    x = r(x)
                return x
            return restoration

        # apply reparametrization repeatedly until no more equality constraints
        problem = self.problem
        restorations = []
        changed = True
        while changed:
            changed = False
            for eq, val in problem.eq_constraints.items():
                if not (val == 0):
                    continue
                for transform, inv_transform in _reparam_power(eq, configs):
                    problem, restoration = problem.transform(transform, inv_transform)
                    restorations.append(restoration)
                    changed = True
                    break
                if changed:
                    break

        if self.problem is problem:
            self.finished = True
            return

        problem, mul = problem.sqr_free(problem_sqf=True)
        restorations.append(lambda x: None if x is None else x * mul**2)
        c = _solver(problem)
        self.children.append(c)
        self.restorations[c] = make_restoration(restorations)

        self.state = -1

def _reparam_power(eq: Poly, configs):
    """
    Equality in the types of
        `C * prod(linear)^n = prod(linear)^m`

    Now we implement the simple version where the linear part is symbols.
    TODO:
    1. Implement more general version beyond monomials.
    """
    # check whether it is binomial
    monoms = eq.monoms()
    if not len(monoms) == 2:
        return []

    # -c1/c2 * [x]^m1 == [x]^m2
    m1, m2 = monoms
    gcdmonom = tuple(min(mi, mj) for mi, mj in zip(m1, m2))
    m1 = tuple(mi-mj for mi, mj in zip(m1, gcdmonom))
    m2 = tuple(mi-mj for mi, mj in zip(m2, gcdmonom))
    gcdm = int(gcd(m1 + m2))
    c1, c2 = eq.coeffs()
    c = -c1/c2
    if gcdm % 2 == 0:
        # cannot take the squareroot because sqrt(x^2) != x for x < 0
        # unless it can be proved that x >= 0
        return []

    # replace every symbol x by x = x'^(d/q), so that every symbol is a d-perfect power
    # then the equality can be rewritten as c^(1/d) * [x']^(m1/d) * [x']^(m2/d)
    # at this time, one of the variable has degree 1 and can be solved out
    m0 = [max(mi, mj) for mi, mj in zip(m1, m2)]
    d_ind = min(list(range(len(m0))), key=lambda i: m0[i] if m0[i]!=0 else float('inf'))
    d = m0[d_ind]

    # TODO: we can do more carefully to allow _perfect_power(c, gcdm) using Bezout's identity
    croot = _perfect_power(c, d)
    if not croot:
        return []

    gens = eq.gens
    nvars = len(gens)
    pows = [0] * nvars
    pows2 = [0] * nvars
    for i in range(nvars):
        di = max(m1[i], m2[i])
        if di == 0: # the symbol does not appear in the equality
            continue
        gcd_ = gcd(d, di)
        pows[i] = d // gcd_
        if pows[i] % 2 == 0:
            return []
        pows2[i] = di // gcd_ if m1[i] > m2[i] else -di // gcd_

    if any(di > 1 for di in pows) and not configs['irrational_expr']:
        return []

    need_change = lambda i: pows[i] != 0 and pows[i] != 1 and i != d_ind
    new_gens = [Dummy(gens[i].name) if need_change(i) else gens[i] for i in range(nvars)]
    transform = {gens[i]: new_gens[i]**pows[i] for i in range(nvars) if need_change(i)}
    inv_transform = {new_gens[i]: gens[i]**(Integer(1)/pows[i]) for i in range(nvars) if need_change(i)}

    # gens[d_ind] computable by other variables
    transform[gens[d_ind]] = Mul(croot[0],
        *[new_gens[i]**pows2[i] for i in range(nvars) if pows2[i] != 0 and i != d_ind])
    if m1[d_ind] > m2[d_ind]:
        transform[gens[d_ind]] = 1/transform[gens[d_ind]]

    return [(transform, inv_transform)]
