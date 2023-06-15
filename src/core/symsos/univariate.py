from itertools import product

import sympy as sp

# from ...utils.roots.rationalize import rationalize
from ...utils import congruence

def _classify_roots(roots):
    """
    Given a list of sympy roots. Classify them into real roots and complex roots.
    Also, the complex roots are paired by conjugate.

    We assume there are no repeated roots.

    Parameters
    ----------
    roots : list[sp.Float]
        Roots of a polynomial.

    Returns
    -------
    real_roots : list[sp.Float]
        Real roots.
    complex_roots : list[(sp.Float, sp.Float)]
        Paired complex roots.
    """
    real_roots = [root for root in roots if root.is_real]

    # pair complex roots
    # since the poly is irreducible, we may assume there is no repeated root
    complex_roots = [root for root in roots if sp.im(root) > 0]
    complex_roots = [(root, root.conjugate()) for root in complex_roots]
    return real_roots, complex_roots


def _construct_matrix_from_roots(real_roots, complex_roots, positive = False):
    """
    Assume an irreducible polynomial f(x) is positive over R, then 
    all its roots are complex and paired by conjugate. Write 
    f(x) = \prod (x - root_i) * \prod (x - root_i.conjugate), then
    f(x) = |\prod (x - root_i)|^2 = Re^2 + Im^2 is already sum of squares.

    For each i, we can select either root_i or its conjugate to multiply.
    This generates 2^n different sum of squares expressions. We take the
    average of them to get a symmetric matrix M, then M is positive definite
    and is often full rank.

    When f(x) is positive over R+, we have f(x) = g(x) ( h^2(x) + k^2(x) )
    where all roots of g are negative numbers. Then 
    g = a_{2n}x^{2n} + a_{2n-2}x^{2n-2} + ... + a{0}
        + x(a_{2n-1}x^{2n-2} + ... + a{1})
    where all coefficients a are positive. Then we can see that 
    f = u^2 + x* v^2. Then we can construct two symmetric matrices.
    
    Parameters
    ----------
    real_roots: list[sp.Float]
        Real roots.
    complex_roots : list[(sp.Float, sp.Float)]
        Paired complex roots.
    positive:
        Whether the polynomial is positive over R or R+.

    Returns
    -------
    If positive == False:
        M : sp.Matrix
            The symmetric matrix.
    If positive == True:
        M1, M2: tuple[sp.Matrix, sp.Matrix]
            Two symmetric matrices.
    """
    x = sp.symbols('x')
    if not positive:
        M = sp.zeros(len(complex_roots) + 1)
        for comb in product(range(2), repeat=len(complex_roots)):
            vec = sp.prod((x - root[i]) for i, root in zip(comb, complex_roots))
            vec = vec.as_poly(x).all_coeffs()

            vec_re = sp.Matrix(list(map(sp.re, vec)))
            vec_im = sp.Matrix(list(map(sp.im, vec)))
            
            M += vec_re * vec_re.T + vec_im * vec_im.T

        M = M / 2**len(complex_roots)
        return M

    else:
        0

def _rationalize_off_diagonal_simultaneously(M, lcm = 144):
    """
    Rationalize off-diagonal entries of a symmetric matrix simultaneously.
    Off-diagonal entries are entries with strip > 1, i.e. not Toeplitz.

    Parameters
    ----------
    M : sp.Matrix
        The symmetric matrix.
    lcm : int, optional
        The least common multiple of denominators. The default is 144.

    Returns
    -------
    M : sp.Matrix
        The rationalized matrix.
    """
    n = M.shape[0]
    M = M.copy()
    for i in range(n):
        for j in range(i + 2, n):
            M[i,j] = round(M[i,j] * lcm) / lcm
            M[j,i] = M[i,j]
    return M


def _determine_diagonal_entries(M, poly):
    """
    Determine the Toepiltz diagonal entries of a symmetric matrix
    from the coefficient of polynomial.

    Example: x^4 + x^3 + 5x^2 + 1 <=> [[x, y, 1], [y, z, w], [1, w, u]],
    then we can solve that x = 1, y = 1/2, z = 3, w = 0, u = 1.

    Parameters
    ----------
    M : sp.Matrix
        The symmetric matrix.
    poly : sp.Poly
        The polynomial.
    
    Returns
    -------
    M : sp.Matrix
        The matrix with diagonal entries determined. The operation is 
        done in-place.
    """
    n = M.shape[0]
    all_coeffs = poly.all_coeffs()
    for i in range(len(all_coeffs)):
        m = i // 2
        if i % 2 == 0:
            l = min(m, n - m - 1)
            s = sum(M[m - j, m + j] for j in range(1, l + 1)) * 2
            M[m, m] = all_coeffs[i] - s
        else:
            l = min(m, n - m - 2)
            s = sum(M[m - j, m + j + 1] for j in range(1, l + 1))
            M[m, m + 1] = all_coeffs[i] / 2 - s
            M[m + 1, m] = M[m, m + 1] 
        
    return M


def _create_sos_from_US(U, S, x = None):
    """
    Create a sum of squares from a congruence decomposition.

    Parameters
    ----------
    U : sp.Matrix
        The upper triangular matrix.
    S : list
        The diagonal matrix. Use a list to store the diagonal entries.
    x : sp.Symbol, optional
        The symbol of polynomial. The default is None.
    """
    if x is None:
        x = sp.symbols('x')
    polys = [sp.polys.Poly.from_list(U[i,:].tolist()[0], x) for i in range(U.shape[0])]
    exprs = [Si * poly.as_expr()**2 for Si, poly in zip(S, polys)]
    return sp.Add(*exprs)


def _prove_univariate_irreducible(poly, positive = False):
    """
    Prove an irreducible univariate polynomial is positive over the real line.

    Parameters
    ----------
    poly : sp.Poly
        The polynomial.

    Returns
    -------
    sos : sp.Expr
        A sum of squares expression.
    """
    roots = sp.polys.nroots(poly)
    real_roots, complex_roots = _classify_roots(roots)
    if not positive:
        # real case does not accept any real roots
        if len(real_roots):
            return None
    else:
        # positive case requires all real roots to be negative
        if any(_ > 0 for _ in real_roots):
            return None

        if len(real_roots) == 0:
            # turn off R+ constraint if no real roots found
            positive = False

    # extract roots to construct matrix
    M = _construct_matrix_from_roots(real_roots, complex_roots, positive = positive)

    for lcm in (1, 144, 144**2, 144**6):
        # rationalize off-diagonal entries
        # and restore diagonal entries by subtraction
        M = _rationalize_off_diagonal_simultaneously(M, lcm = lcm)
        M = _determine_diagonal_entries(M, poly)

        # verify that the matrix is positive semidefinite
        U, S = congruence(M)
        if all(_ >= 0 for _ in S):
            return _create_sos_from_US(U, S, poly.gens[0])
    return