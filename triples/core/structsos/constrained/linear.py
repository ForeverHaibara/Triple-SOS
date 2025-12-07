from sympy import Poly, Dummy, Integer
from sympy.core.symbol import uniquely_named_symbol

def _eliminate_linear_ineq(poly, var, ineq_constraints, eq_constraints):
    """
    Given poly: f*var + g == expr >= 0, return (f, g, fsol, 1) if f >= 0, (f, g, fsol, -1) if f <= 0,
    and (f, g, None, 0) if the sign of f cannot be determined.
    NOTE: fsol == f, but in a sum-of-squares form.
    """
    gens = list(poly.gens)
    k = poly.diff(var).subs(var, 0)
    b = poly.subs(var, 0)

    # gens.remove(var)
    # k2 = k.as_poly(gens)
    # ineq_constraints2 = {ineq.as_poly(gens): e for ineq, e in ineq_constraints.items()}
    # eq_constraints2 = {eq.as_poly(gens): e for eq, e in eq_constraints.items()}
    k2, ineq_constraints2, eq_constraints2 = k, ineq_constraints, eq_constraints

    from ..structsos import _structural_sos
    ksol = _structural_sos(k2, ineq_constraints2, eq_constraints2)
    if ksol is not None:
        # k >= 0 and kx + b >= 0 -> x >= -b/k
        return k, b, ksol, 1
    ksol = _structural_sos(-k2, ineq_constraints2, eq_constraints2)
    if ksol is not None:
        # k <= 0 and kx + b >= 0 -> x <= -b/k
        return k, b, -ksol, -1

    return k, b, None, 0


def elimination_linear(poly, ineq_constraints, eq_constraints, restore = None):
    """
    Eliminate some linear constraints. For example, given x>=y, x<=5*y,
    we can perform a substitution x = y + 4*y*t/(t+1) where t >= 0.

    It does not solve the problem directly, but applies some transformations.
    """
    if restore is None:
        restore = lambda x: x
    gens = poly.gens
    new_problem = None
    for gen in gens:
        linear_ineqs = []
        linear_eqs = []
        remain_ineqs = []
        remain_eqs = []
        is_linear = True
        for ineq, e in ineq_constraints.items():
            d = ineq.degree(gen)
            if d > 1:
                is_linear = False
                break
            elif d == 1:
                linear_ineqs.append((ineq, e))
            else:
                remain_ineqs.append((ineq, e))
        # for eq, e in eq_constraints.items():
        #     d = eq.degree(gen)
        #     if d > 1:
        #         is_linear = False
        #         break
        #     elif d == 1:
        #         linear_eqs.append((eq, e))
        #     else:
        #         remain_eqs.append((eq, e))
        if (not is_linear) or len(linear_ineqs) + len(linear_eqs) == 0:
            continue

        l_bounds = []
        u_bounds = []
        for ineq, e in linear_ineqs:
            k, b, ksol, sign = _eliminate_linear_ineq(ineq, gen, remain_ineqs, eq_constraints)
            if sign == 0:
                continue
            elif sign == 1:
                l_bounds.append((k, b, e, ksol)) # x + b/k = e/ksol >= 0
            else: # sign == -1
                u_bounds.append((k, b, e, ksol)) # x + b/k = e/ksol <= 0

        remain_ineqs = dict(remain_ineqs)
        remain_eqs = dict(remain_eqs)
        if len(l_bounds) <= 1 and len(u_bounds) <= 1:
            l_bound = None if len(l_bounds) == 0 else l_bounds[0]
            u_bound = None if len(u_bounds) == 0 else u_bounds[0]
            new_problem = _elimination_linear_bound_lu(
                poly, remain_ineqs, remain_eqs, gen, l_bound, u_bound
            )
        if new_problem is not None:
            poly, ineq_constraints, eq_constraints, restore2 = new_problem
            return elimination_linear(poly, ineq_constraints, eq_constraints, lambda x: restore(restore2(x)))

    return poly, ineq_constraints, eq_constraints, restore

def _elimination_linear_bound_lu(poly, ineq_constraints, eq_constraints, gen, l_bound=None, u_bound=None):
    """
    ...
    """
    if l_bound is None and u_bound is None:
        return None

    ineq_constraints = ineq_constraints.copy()
    if l_bound is None or u_bound is None:
        return None # not implemented

    kl, bl, l0, klsol = l_bound
    ku, bu, u0, kusol = u_bound
    l, u = l0/klsol, u0/kusol
    # t = uniquely_named_symbol('t', poly.gens)
    gens = poly.gens
    poly = poly.as_poly(gen)
    d = poly.degree()

    # x + bl/kl = l >= 0
    # x + bu/ku = u <= 0
    # x = (-bu/ku*t - bl/kl)/(t+1)  (t >= 0)
    # t = -l/u

    # gcdklu = gcd(kl, ku) # TODO: ensure gcdklu > 0
    # ku2, kl2 = ku.div(gcdklu)[0], kl.div(gcdklu)[0]
    ku2, kl2 = ku, kl
    lcm = -ku2 * kl
    h = uniquely_named_symbol(gen.name, gens) # homogenizer
    p1, p2 = Poly(bu*kl2*gen + bl*ku2*h, gen), Poly(lcm*(gen + h), gen)
    # x, y = Dummy('x'), Dummy('y')
    # print('FACT',poly.transform(p1, p2).as_expr().xreplace({gen:-x/y})\
    #       .subs({x:gen+bl.as_expr()/kl.as_expr(), y:gen+bu.as_expr()/ku.as_expr()}).factor(),
    #       '\np1p2', p1,p2)
    poly = poly.transform(p1, p2)
    gens = gens + (h,)
    ineq_constraints = {ineq.as_poly(gens): e for ineq, e in ineq_constraints.items()}
    eq_constraints = {eq.as_poly(gens): e for eq, e in eq_constraints.items()}
    ineq_constraints[gen.as_poly(gens)] = -l/u
    ineq_constraints[h.as_poly(gens)] = Integer(1)
    restore = lambda x: x if x is None else x / (klsol*(-kusol)*(l/-u + 1))**d # x/(lcm*(gen+h))**d

    poly = poly.as_poly(gens)
    print(poly, ineq_constraints, eq_constraints, restore)
    return poly, ineq_constraints, eq_constraints, restore
