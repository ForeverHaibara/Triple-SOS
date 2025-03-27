"""
This module provides customized / optimized functions to solve
polynomial systems and provides supports for low SymPy versions.
"""
from typing import List, Tuple, Union

import sympy as sp
from sympy import Poly, Expr, Rational, Symbol
from sympy.polys.polyerrors import BasePolynomialError
from sympy.polys.polytools import resultant, groebner, PurePoly
from sympy.polys.rootoftools import ComplexRootOf as CRootOf
from sympy.utilities import postfixes
from mpmath.libmp.libhyper import NoConvergence

# Comparison of tuples of sympy Expressions, compatible with sympy <= 1.9
default_sort_key = lambda x: tuple(_.sort_key() for _ in x) if not isinstance(x, Expr) else x.sort_key()

def _filter_trivial_system(polys: List[Poly]) -> Union[List[Poly], None]:
    """
    Simplify a list of polynomial equations by removing zero or duplicate equations.
    If the system contains nonzero constants, return None.
    """
    new_polys = set()
    for poly in polys:
        if isinstance(poly, sp.Expr):
            poly = poly.expand()
            if poly is sp.S.Zero:
                continue
            if poly.is_constant(): # inconsistent system
                return None
        elif isinstance(poly, sp.Poly):
            if poly.is_zero:
                continue
            elif poly.total_degree() == 0: # inconsistent system
                return None
        new_polys.add(poly)
    return list(new_polys)

def _get_realroots_sqf(poly: Poly) -> List[Union[Rational, CRootOf]]:
    """
    Low level implementation of poly.real_roots() for square-free polynomials.
    """
    if poly.total_degree() == 0:
        return []
    if not (poly.domain.is_QQ or poly.domain.is_ZZ):
        # fallback
        return list(set(CRootOf.real_roots(poly, radicals=False)))
    if poly.total_degree() == 1:
        return [-poly.coeff_monomial((0,)) / poly.coeff_monomial((1,))]

    poly = PurePoly(poly)
    intervals = CRootOf._get_reals_sqf(poly)
    return [CRootOf._new(poly, i) for i in range(len(intervals))]

class PolyEvalf:
    """
    Class to evaluate a polynomial at a point with numerical precision.
    """
    dps = 15
    def polyeval(self, poly, point, n=dps):
        """Evaluate a polynomial at a point numerically."""
        if all(isinstance(_, Rational) for _ in point):
            return poly(*point)

        # high dps should be careful with the context domain
        domain = sp.RealField(dps=n) if n != 15 else sp.RR
        poly = poly.set_domain(domain)
        return poly(*(p.n(n) for p in point))

    def polysign(self, poly, point, max_tries=3):
        """Infer the sign of a polynomial at a point numerically."""
        if all(isinstance(_, Rational) for _ in point):
            v = poly(*point)
            return 1 if v > 0 else (0 if v == 0 else -1)
        def try_dps(dps, tries):
            for i in range(tries):
                v = self.polyeval(poly, point, n=dps) # .n(dps*2//3, chop=True)
                if -1 < v*10**(dps//3) < 1:
                    dps = dps * 2
                    continue
                return 1 if v > 0 else -1
            return 0
        return try_dps(self.dps, max_tries)

def univar_realroots(poly: Union[Poly, Expr], symbol: Symbol) -> List[Union[Rational, CRootOf]]: #, max_degree: int = 30):
    """
    Get real roots of a univariate polynomial without multiplicity.

    Parameters
    ----------
    poly : Poly or Expr
        A univariate polynomial or a sympy expression.
    symbol : Symbol
        The symbol of the polynomial.

    Returns
    -------
    roots : List[Union[Rational, CRootOf]]
        A list of real roots of the polynomial.

    Examples
    --------
    >>> from sympy.abc import x
    >>> univar_realroots(x**5 - 2*x + 1, x) # doctest: +NORMALIZE_WHITESPACE
    [1,
     CRootOf(x**4 + x**3 + x**2 + x - 1, 0),
     CRootOf(x**4 + x**3 + x**2 + x - 1, 1)]
    """
    try:
        poly = Poly(poly, symbol, extension=True)
    except BasePolynomialError: # CoercionFailed, etc.
        try:
            poly = Poly(poly, symbol)
        except BasePolynomialError:
            pass
    if not isinstance(poly, Poly):
        return []
    if poly.domain.is_Exact and poly.domain.is_Numerical:
        _, fact = poly.factor_list() # this is time-consuming
        roots = []
        for f, mul in fact:
            # if f.degree() > max_degree:
            #     continue
            roots.extend(_get_realroots_sqf(f))
        roots = sorted(roots, key=default_sort_key)
        return roots
    if poly.domain.is_Numerical:
        try:
            roots = [root for root in nroots(poly) if root.is_real]
            return sorted(roots, key=default_sort_key)
        except NoConvergence:
            pass
    return []

def nroots(poly, method = 'numpy', real = False, nonnegative = False):
    """
    Wrapper function to find the numerical roots of a sympy polynomial.
    Note that sympy nroots is not stable when the polynomial has multiplicative roots,
    so we need to factorize the polynomial sometimes.

    Parameters
    ----------
    poly : sympy.Poly
        The polynomial to be solved.
    method : str, optional
        The method to be used. 'numpy' uses numpy.roots, 'sympy' uses sympy.nroots.
    real : bool, optional
        Whether to only return real roots.
    nonnegative : bool, optional
        Whether to only return nonnegative roots.
    """
    if method == 'numpy':
        from numpy import roots as nproots
        roots = [sp.S(_) for _ in nproots(poly.all_coeffs())]
    elif method == 'sympy':
        roots = sp.nroots(poly)
    elif method == 'factor':
        roots_rational = []
        roots = []
        for part, mul in poly.factor_list()[1]:
            if part.degree() == 1:
                roots_rational.append(-part.all_coeffs()[1] / part.all_coeffs()[0])
            else:
                roots.extend(sp.polys.nroots(part))
        roots = roots_rational + roots

    if real:
        roots = [_ for _ in roots if _.is_real]
    if nonnegative:
        roots = [_ for _ in roots if _.is_nonnegative]
    
    return roots

def heuristic_groebner_order(polys: List[Poly], symbols: List[Symbol]) -> List[Symbol]:
    """
    Get a heuristic order of symbols for Groebner basis computation.
    Linear variables are sorted before nonlinear variables.
    For linear variables, less frequent or less total degree variables are sorted first.
    """
    stats = {}
    for var in symbols:
        occurs, total_degree, is_linear = 0, 0, True
        for eq in polys:
            degree = eq.degree(var)
            if degree > 0:
                occurs += 1
                total_degree += degree
                if degree > 1:
                    is_linear = False
        stats[var] = {
            'occurs': occurs,
            'total_degree': total_degree,
            'is_linear': is_linear
        }

    linear_vars = [v for v in symbols if stats[v]['is_linear']]
    nonlinear_vars = [v for v in symbols if not stats[v]['is_linear']]

    def sort_key(var):
        s = stats[var]
        return (s['occurs'], s['total_degree'])

    linear_vars_sorted = sorted(linear_vars, key=sort_key)
    nonlinear_vars_sorted = sorted(nonlinear_vars, key=sort_key)
    
    return tuple(linear_vars_sorted + nonlinear_vars_sorted)

def solve_triangulated_crt(polys: List[Poly], symbols: List[Symbol]) -> List[Tuple[CRootOf]]:
    """
    Solve a polynomial system by triangulating the Groebner basis.

    See Also
    --------
    sympy.solvers.polysys.solve_triangulated
    """
    original_symbols = tuple(symbols)
    symbols = heuristic_groebner_order(polys, symbols)
    original_inds = [symbols.index(s) for s in original_symbols]

    G = groebner(polys, symbols, polys=True)
    G = list(reversed(G))

    f, G = G[0].ltrim(-1), G[1:]

    zeros = univar_realroots(f, f.gen)
    solutions = {((zero,), sp.QQ.algebraic_field(zero)) for zero in zeros}

    var_seq = reversed(symbols[:-1])
    vars_seq = postfixes(symbols[1:])

    for var, vars in zip(var_seq, vars_seq):
        _solutions = set()

        for values, dom in solutions:
            H, mapping = [], list(zip(vars, values))

            for g in G:
                _vars = (var,) + vars

                if g.has_only_gens(*_vars) and g.degree(var) != 0:
                    g = g.set_domain(g.domain.unify(dom))
                    h = g.ltrim(var).eval(dict(mapping))

                    if g.degree(var) == h.degree():
                        H.append(h)

            if len(H):
                p = min(H, key=lambda h: h.degree())
                zeros = univar_realroots(p, p.gen)
            else:
                zeros = []

            for zero in zeros:
                if not (zero in dom):
                    dom_zero = dom.algebraic_field(zero)
                else:
                    dom_zero = dom

                _solutions.add(((zero,) + values, dom_zero))

        solutions = _solutions

    # map back to original symbols
    solutions = [tuple(s[i] for i in original_inds) for s, _ in solutions]
    return solutions



def _solve_poly_system_2vars_resultant(polys: List[Poly], symbols: List[Symbol]) -> List[Tuple[CRootOf]]:
    """
    Solve a polynomial system with two variables and at least two equations
    via the resultant method.
    Experience tells that resultant might be much faster than groebner basis
    when there are only two variables.

    Parameters
    ----------
    polys : List[Poly]
        A list of polynomials.
    symbols : List[Symbol]
        A list of symbols.

    Returns
    -------
    roots : List[Tuple[CRootOf]]
        A list of solutions.
    """
    if len(symbols) != 2 or len(polys) < 2:
        return []

    x, y = symbols    
    res0 = resultant(polys[0], polys[1], x).as_poly(y)
    for poly in polys[2:]:
        if res0.total_degree() == 0 and (not res0.is_zero):
            return []
        new_res = resultant(polys[0], poly, x).as_poly(y)
        res0 = sp.gcd(res0, new_res)

    roots1 = univar_realroots(res0, y)
    roots = []
    pevalf = PolyEvalf()
    if all(isinstance(_, Rational) for _ in roots1):
        for root1 in roots1:
            res1 = polys[1].eval(y, root1)
            roots0 = univar_realroots(res1, x)
            for root0 in roots0:
                if all(pevalf.polysign(poly, (root0, root1)) == 0 for poly in polys):
                    roots.append((root0, root1))
    else:
        res1 = resultant(polys[0], polys[1], y)
        roots0 = univar_realroots(res1, x)
        for root0 in roots0:
            for root1 in roots1:
                if all(pevalf.polysign(poly, (root0, root1)) == 0 for poly in polys):
                    roots.append((root0, root1))
    return roots


def solve_poly_system_crt(polys: List[Poly], symbols: List[Symbol]) -> List[Tuple[CRootOf]]:
    """
    Main function to solve the REAL ROOTS of a polynomial system. It tries to use the optimized
    `solve_triangulated_crt` first, but falls back to default sympy `solve` if failed.
    Nonzero dimensional variety is not supported.
    """
    if len(polys) < len(symbols):
        return []
    def default(polys, symbols):
        if len(symbols) == 0:
            filtered = _filter_trivial_system(polys)
            if filtered is not None and len(filtered) == 0:
                return [tuple()]
            return []
        sol = sp.solve(polys, symbols, dict=False)
        sol = [_ for _ in sol if all(v.is_real for v in _)]
        return sol
    if len(symbols) == 0:
        return default(polys, symbols)

    try:
        polys = [sp.Poly(_, *symbols, extension=True) for _ in polys]
    except BasePolynomialError:
        return default(polys, symbols)
    if not all(_.domain.is_Exact for _ in polys):
        return default(polys, symbols)

    polys = [_ for _ in polys if not _.is_zero]
    if len(polys) < len(symbols):
        return []
    if len(symbols) == 2:
        return _solve_poly_system_2vars_resultant(polys, symbols)
    try:
        return solve_triangulated_crt(polys, symbols)
    except (BasePolynomialError, IndexError) as e:
        # Cannot ltrim ... (due to nonzero dimensional variety)
        return []
    return default(polys, symbols)
