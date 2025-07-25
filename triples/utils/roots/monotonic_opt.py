from collections import namedtuple
from typing import List

import numpy as np
import sympy as sp

RPAResult = namedtuple('RPAResult', ['x', 'f', 'iterations', 'converged', 'a', 'b'])

_xtol = 2e-12
_rtol = 4 * np.finfo(float).eps

def _find_sup(f, a, b, xtol=_xtol, maxiter=100):
    """
    Find sup x: f(x) <= 0 over the interval [a, b] using hybrid Brent's method and bisection search.
    In the case of monotonic optimization, f must be monotonic non-decreasing over [a, b]
    and f(a) <= 0, f(b) >= 0.    
    """
    if -_rtol <= f(b) <= _rtol:
        return b

    from scipy.optimize import root_scalar
    # Step 1: Apply Brent's method (using scipy.optimize.root_scalar with brentq)
    result = root_scalar(f, bracket=(a, b), method='brentq', xtol=xtol, maxiter=maxiter)
    
    # Check if the root is found successfully and if the function at x + _xtol > 0
    x = result.root
    if f(x + _xtol) >= _rtol:
        return x
    
    # Step 2: Fallback to bisection method if the condition is not satisfied
    left, right = x + _xtol, b
    for _ in range(maxiter):
        mid = (left + right) / 2
        fmid = f(mid)
        
        if abs(right - left) < _xtol:
            return mid
        
        if fmid <= 0:
            left = mid  # Move left bound up
        else:
            right = mid  # Move right bound down
    
    # not expected to reach here because bisecting should converge
    raise ValueError("Maximum iterations exceeded")


def _red(f, g, h, a, b, cbv, root_solver=_find_sup):
    """
    Helper function for reducing the lower/upper bound of the domain [a, b] in the Reverse Polyblock Approximation Algorithm.
    Since we require f(x) < cbv and g(x) ≤ 0, h(x) ≥ 0, we can reduce the domain [a, b]
    coordinate-wise by finding the lower/upper bound of the domain that satisfies the constraints.

    See the implementation issue 1 in [1] and the so-called "valid reduction" in [2] for more details.

    Parameters
    ----------
    f : callable
        The function to be minimized, R^n -> R. It should be monotonic increasing over [a, b].
    g : callable
        Constraint function 1, R^n -> R. It should be monotonic increasing over [a, b]. The
        constraint is g(x) ≤ 0.
    h : callable
        Constraint function 2, R^n -> R. It should be monotonic decreasing over [a, b]. The
        constraint is h(x) ≥ 0.
    a : array_like
        Lower bound of the domain, R^n.
    b : array_like
        Upper bound of the domain, R^n.
    cbv : float
        Current best value.
    root_solver : callable, optional
        Function to solve sup x: f(x) <= 0 over the interval [a, b] where f is monotonic non-decreasing.
        Default is _find_sup.

    Returns
    -------
    a_new : array_like
        New lower bound of the domain, R^n.
    b_new : array_like
        New upper bound of the domain, R^n.    

    References
    ----------
    [1] H. Tuy, "Monotonic Optimization: Problems and Solution Approaches", SIAM Journal on Optimization, 2000.

    [2] H. Tuy, "Convex Analysis and Global Optimization", 1998.

    [3] H. Tuy, "Robust Solution of Nonconvex Global Optimization Problems", Journal of Global Optimization, 2005.
    """
    # Issues 1. reduce the lower/upper bound
    a, b = a.copy(), b.copy()
    n = a.size
    for i in range(n):
        # coordinate-wise reduction
        ai = a[i]
        def g_line(alpha):
            a[i] = ai + alpha * (b[i] - ai)
            return max(f(a) - cbv, g(a))
        res = root_solver(g_line, 0, 1) if g_line(1) >= 0 else 1.
        b[i] = ai + res * (b[i] - ai) # inplace update
        a[i] = ai

    if h(b) <= -_rtol: # infeasible
        return b, b

    for i in range(n):
        bi = b[i]
        def h_line(alpha):
            b[i] = bi - alpha * (bi - a[i])
            return -h(b) # make h monotonic increasing
        res = root_solver(h_line, 0, 1) if h_line(1) >= 0 else 1.
        a[i] = bi - res * (bi - a[i])
        b[i] = bi
    return a, b

def rpa_monotonic(f, g, h, a, b, tol=1e-5, max_iter=2000, cbv=np.inf, root_solver=_find_sup, verbose=False):
    """
    Reverse Polyblock Approximation Algorithm for solving the global monotonic optimization problem B:
        min { f(x) | x ∈ G ∩ H }     (B)
    where:
        G = { x ∈ [a, b] | g(x) ≤ 0 } (normal set)
        H = { x ∈ [a, b] | h(x) ≥ 0 } (reverse normal set)
    f, g, h are monotonic increasing functions over [a, b].

    This function is a modified implementation of the Algorithm 2 in the paper [1] by Hoang Tuy
    and it involves randomization to avoid infinite loops.

    For the more general optimization problem where f, g, h are not monotonic but
    difference of two monotonic functions, use `rpa_gmop`.

    Parameters
    ----------
    f : callable
        The function to be minimized, R^n -> R. It should be monotonic increasing over [a, b].
    g : callable
        Constraint function 1, R^n -> R. It should be monotonic increasing over [a, b]. The
        constraint is g(x) ≤ 0.
    h : callable
        Constraint function 2, R^n -> R. It should be monotonic decreasing over [a, b]. The
        constraint is h(x) ≥ 0.
    a : array_like
        Lower bound of the domain, R^n.
    b : array_like
        Upper bound of the domain, R^n.
    tol : float, optional
        Tolerance for stopping criterion. The algorithm stops if the current best value is no
        greater than the best value + tol. Default is 1e-5.
    max_iter : int, optional
        Maximum number of iterations. Default is 2000.
    cbv : float, optional
        Current best value. Default is np.inf. If given, it should be an upper bound of the optimal value.
        This is useful for branch cutting. However, if cbv is smaller than the true optimal value, then
        it is likely the algorithm raises an Exception.
    root_solver : callable, optional
        Function to solve sup x: f(x) <= 0 over the interval [a, b] where f is monotonic non-decreasing.
        Default is _find_sup.
    verbose : bool, optional
        If True, print the current best value at each update. Default is False.

    Returns
    ----------
    result : RPAResult
        Named tuple containing the result of the algorithm. It contains the following fields:
        x : array_like
            The best point found. If None, the algorithm did not find a feasible point.
        f : float
            The best value found. If x is None, this is np.inf.
        iterations : int
            Number of iterations performed.
        converged : bool
            True if the algorithm converged within max_iter iterations, False otherwise.
        a : array_like
            Lower bound of the domain after the algorithm.
        b : array_like
            Upper bound of the domain after the algorithm.


    References
    ----------
    [1] H. Tuy, "Monotonic Optimization: Problems and Solution Approaches", SIAM Journal on Optimization, 2000.

    [2] H. Tuy, "Convex Analysis and Global Optimization", 1998.

    [3] H. Tuy, "Robust Solution of Nonconvex Global Optimization Problems", Journal of Global Optimization, 2005.
    """
    array = lambda x: np.array(x, dtype='float') if x is not None else None
    a, b, x_best, cbv = array(a).copy(), array(b).copy(), None, cbv + tol
    n = a.size

    if g(a) > 0 or h(b) < 0 or f(a) > cbv - tol:
        x_best, cbv, k, T = None, np.inf, 0, []
    else:
        a, b = _red(f, g, h, a, b, cbv, root_solver=root_solver)
        T = [a]

    for k in range(1, 1 + max_iter):
        if not T:  # Step 2: If T is empty, terminate
            break

        # Step 3: Select z_k with maximal f(z) in T
        # The paper requires the minimal, but it seems it will fall in an infinite loop sometimes??
        # ind = min(range(len(T)), key=lambda i: f(T[i]))
        ind = np.random.choice(len(T))
        z_k = T[ind]
        T.pop(ind)  # Remove z_k from T

        # Step 4: Compute x_k = ρ_H(z_k) (last point of H on the half-line from b through z_k)
        def h_line(alpha):
            return -h(b + alpha * (z_k - b))
        alpha_k = root_solver(h_line, 0, 1) if h_line(1) >= 0 else 1.
        x_k = b + alpha_k * (z_k - b)

        if g(x_k) <= 0:  # If x_k ∈ G, update cbv and x_best
            if f(x_k) < cbv:
                cbv = f(x_k)
                x_best = x_k
                a, b = _red(f, g, h, a, b, cbv, root_solver=root_solver)
                T = [z for z in T if np.all(z >= a) and np.all(z <= b) and f(z) < cbv - tol and g(z) <= 0]
                if verbose:
                    print(k, 'New best:', cbv, 'x =', x_best, 'L =', len(T))
                    print('New bounds:', a, b)

        # Step 5: Split z_k into new vertices and add them to T
        fz_k = f(z_k)
        for i in range(n):
            if z_k[i] == x_k[i]:
                # No change in the coordinate, skip.
                # (Happens when f is constant on this line or a[i]==b[i].)
                continue
            z_new = z_k.copy()
            z_new[i] = x_k[i]
            if f(z_new) < cbv - tol and g(z_new) <= 0:
                T.append(z_new)

    if x_best is None:
        cbv = np.inf
    return RPAResult(x_best, cbv, k, len(T) == 0, a, b)


def rpa_gmop(f1_and_f2, g_and_h, a, b, tol=1e-5, max_iter=2000, cbv=np.inf, root_solver=_find_sup, verbose=False):
    """
    Reverse Polyblock Approximation Algorithm for solving the
    General Monotonic Optimization Problem (GMOP):
        min { f1(x) - f2(x) | g_i(x) - h_i(x) ≤ 0, i = 1, ..., m, x ∈ [a, b] }
    where f1, f2, g_i, h_i are monotonic increasing functions.

    This function transforms the GMOP into a monotonic optimization problem by introducing
    a slack variable then calls `rpa_monotonic` to solve it.
    Formally, it is
        min     f1(x) + u
        s.t.    g(x) + t <= 0,
                h(x) + t >= 0,
                f2(x) + u >= 0,
                a <= x <= b,
                -g(b) <= t <= -g(a),
                -f2(b) <= u <= -f2(a),
    where h(x) = sum_i h_i(x), g(x) = max_i(g_i(x) - h_i(x) + h(x))

    Note that any function with bounded variation over the interval 
    can be written as the difference of two monotonic functions.
    

    Parameters
    ----------
    f1_and_f2 : tuple of callable
        A tuple (f1, f2) where f1 and f2 are monotonic increasing functions, R^n -> R.
        The objective function is f(x) = f1(x) - f2(x).
    g_and_h : list of tuple of callable
        A list of tuples [(g1, h1), (g2, h2), ..., (gm, hm)] where gi and hi are monotonic
        increasing functions, R^n -> R. The constraints are gi(x) - hi(x) ≤ 0.
    a : array_like
        Lower bound of the domain, R^n.
    b : array_like
        Upper bound of the domain, R^n.
    tol : float, optional
        Tolerance for stopping criterion. Default is 1e-5.
    max_iter : int, optional
        Maximum number of iterations. Default is 2000.
    cbv : float, optional
        Current best value. Default is np.inf. If given, it should be an upper bound of the optimal value.
        This is useful for branch cutting. However, if cbv is smaller than the true optimal value, then
        it is likely the algorithm raises an Exception.
    root_solver : callable, optional
        Function to solve sup x: f(x) <= 0 over the interval [a, b] where f is monotonic non-decreasing.
        Default is _find_sup.
    verbose : bool, optional
        If True, print the current best value at each update. Default is False.

    Returns
    ----------
    result : RPAResult
        Named tuple containing the result of the algorithm. It contains the following fields:
        x : array_like
            The best point found. If None, the algorithm did not find a feasible point.
        f : float
            The best value found. If x is None, this is np.inf.
        iterations : int
            Number of iterations performed.
        converged : bool
            True if the algorithm converged within max_iter iterations, False otherwise.
        a : array_like
            Lower bound of the domain after the algorithm.
        b : array_like
            Upper bound of the domain after the algorithm.
    """
    f1, f2 = f1_and_f2
    a, b = np.array(a, dtype='float'), np.array(b, dtype='float')
    n = a.size

    if len(g_and_h):
        # Define h(x) = sum_i h_i(x)
        def h(x):
            return np.sum([h_i(x) for _, h_i in g_and_h])

        # Define g(x) = max_i (g_i(x) - h_i(x) + h(x))
        def g(x):
            hi = [h_i(x) for _, h_i in g_and_h]
            hx = np.sum(hi)
            return np.max([g_i(x) - h_i(x) + hx for g_i, h_i in g_and_h])
    else:
        def h(x):
            return 0

        def g(x):
            return 0

    # Define the transformed objective function
    def F(x_t_u):
        return f1(x_t_u[:n]) + x_t_u[n+1]

    # Define the transformed constraints
    def G(x_t_u):
        return g(x_t_u[:n]) + x_t_u[n]

    def H(x_t_u):
        hx_plus_t = h(x_t_u[:n]) + x_t_u[n]
        f2_plus_u = f2(x_t_u[:n]) + x_t_u[n+1]
        return min(hx_plus_t, f2_plus_u)

    # Define the new bounds
    a_new = np.concatenate([a, [-g(b), -f2(b)]])
    b_new = np.concatenate([b, [-g(a), -f2(a)]])

    # Call the rpa_monotonic function
    result = rpa_monotonic(F, G, H, a_new, b_new, tol=tol, max_iter=max_iter, cbv=cbv, root_solver=root_solver, verbose=verbose)

    # Extract the solution
    if result.x is not None:
        x, t, u = result.x[:n], result.x[n], result.x[n+1]
        f = f1(x) - f2(x) # more accurate than F(x_t_u)
    else:
        x, f = None, np.inf

    return RPAResult(x, f, result.iterations, result.converged, result.a[:n], result.b[:n])



def poly_as_dm(poly: sp.Poly, a=None):
    """
    Write a polynomial in the form of difference of two monotonic 
    increasing functions over x >= a where a is n-dimensional.
    Hint: a polynomial on R+ can be written as the difference of
    its positive and negative terms.

    Parameters
    ----------
    poly : sympy.Poly
        Polynomial to be written as difference of two monotonic functions.
    a : list, optional
        List of values for x. If None, a = [0, 0, ..., 0]. Default is None.

    Returns
    ----------
    f1, f2 : sympy.Poly
        Two polynomials such that poly = f1 - f2.
    """
    if a is not None:
        a = [min(ai, 0) for ai in a]
        poly = poly.shift_list(a)
    terms = poly.terms()
    pos = [t for t in terms if t[1] >= 0]
    neg = [t for t in terms if t[1] < 0]
    f1 = sp.Poly.from_dict(dict(pos), poly.gens, domain=poly.domain)
    f2 = -sp.Poly.from_dict(dict(neg), poly.gens, domain=poly.domain)
    if a is not None:
        f1 = f1.shift_list([-ai for ai in a])
        f2 = f2.shift_list([-ai for ai in a])
    return f1, f2


def rpa_polyopt(f: sp.Poly, ineq_constraints: List[sp.Poly]=[], eq_constraints: List[sp.Poly]=[],
    a=-10., b=10., tol=1e-5, eqtol=1e-2, max_iter=2000, cbv=np.inf, root_solver=_find_sup, verbose=False):
    """
    Globally optimize a constrained multivariate polynomial using the Reverse Polyblock Approximation Algorithm.
    The algorithm is based on the paper [1] by Hoang Tuy. This is a variant of branch-and-bound algorithm
    that is reliable for global optimization. However, it might converge very slowly. To
    find or refine a local minimum, it is more recommended to use gradient or Hessian-based methods.

    The optimization problem is:
        min f(x)
        s.t. G1(x) >= 0, G2(x) >= 0, ..., H1(x) == 0, H2(x) == 0, ...

    The equality constraints are relaxed to abs(Hi(x)) <= eqtol.

    Parameters
    ----------
    f: sympy.Poly
        The function to be minimized, which is a sympy polynomial.
    ineq_constraints: List[sympy.Expr]
        The inequality constraints, G1, G2, ... (>= 0).
    eq_constraints: List[sympy.Expr]
        The equality constraints. H1, H2, ... (== 0).
    a: float or list, optional
        Lower bound of the domain. If float, it is the same for all variables.
    b: float or list, optional
        Upper bound of the domain. If float, it is the same for all variables.
    tol: float, optional
        Tolerance for stopping criterion. Default is 1e-5.
    eqtol: float, optional
        Tolerance for equality constraints. Default is 1e-2. Strict eqtol
        may cause the algorithm to claim infeasibility within iterations.
    max_iter: int, optional
        Maximum number of iterations. Default is 2000.
    cbv: float, optional
        Current best value. Default is np.inf. If given, it should be an upper bound of the optimal value.
        This is useful for branch cutting. However, if cbv is smaller than the true optimal value, then
        it is likely the algorithm raises an Exception.
    root_solver: callable, optional
        Function to solve sup x: f(x) <= 0 over the interval [a, b] where f is monotonic non-decreasing.
        Default is _find_sup.
    verbose: bool, optional
        If True, print the current best value at each update. Default is False.

    Returns
    ----------
    result : RPAResult
        Named tuple containing the result of the algorithm. It contains the following fields:
        x : array_like
            The best point found. If None, the algorithm did not find a feasible point.
        f : float
            The best value found. If x is None, this is np.inf.
        iterations : int
            Number of iterations performed.
        converged : bool
            True if the algorithm converged within max_iter iterations, False otherwise.
        a : array_like
            Lower bound of the domain after the algorithm.
        b : array_like
            Upper bound of the domain after the algorithm.

    References
    ----------
    [1] H. Tuy, "Monotonic Optimization: Problems and Solution Approaches", SIAM Journal on Optimization, 2000.
    """
    gens = f.gens
    nvars = len(gens)
    if not hasattr(a, '__iter__'):
        a = [a] * nvars
    if not hasattr(b, '__iter__'):
        b = [b] * nvars

    from sympy.plotting.experimental_lambdify import vectorized_lambdify
    def wrap(gens, f):
        v = vectorized_lambdify(gens, f.as_expr())
        return lambda x: v(*x)

    ineq_constraints = ineq_constraints.copy()
    g_and_h = []
    for eq in eq_constraints:
        ineq_constraints.append(eq + eqtol)
        ineq_constraints.append(-eq + eqtol)
    for ineq in ineq_constraints:
        ineq = sp.Poly(ineq, gens)
        if ineq.degree() <= 0:
            if ineq.LC() < 0:
                return RPAResult(None, np.inf, 1, True, a, b)
            continue
        elif ineq.degree() == 1 and len(ineq.free_symbols) == 1: # linear
            s = ineq.free_symbols.pop()
            ind = gens.index(s)
            lc = ineq.coeff_monomial(s)
            const = ineq.coeff_monomial((0,)*nvars)
            # lc * s + const >= 0
            if lc > 0:
                a[ind] = max(a[ind], -const/lc)
            else:
                b[ind] = min(b[ind], -const/lc)
            continue
        # general case
        f1, f2 = poly_as_dm(ineq, a)
        g_and_h.append((wrap(gens, f2), wrap(gens, f1))) # f2 - f1 <= 0

    f1, f2 = poly_as_dm(f, a)
    f1, f2 = wrap(gens, f1), wrap(gens, f2)
    result = rpa_gmop((f1, f2), g_and_h, a, b, tol=tol, max_iter=max_iter, cbv=cbv, root_solver=root_solver, verbose=verbose)
    return result