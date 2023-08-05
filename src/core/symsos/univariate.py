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


def _construct_matrix_from_roots(real_roots, complex_roots, leading_coeff = 1, positive = False):
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
    leading_coeff: sp.Rational
        The leading coefficient of the polynomial.
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
        M *= leading_coeff
        return M

    else:
        real_part = sp.prod((x - root) for root in real_roots)
        real_part_coeffs = real_part.as_poly(x).all_coeffs()

        Ms = []
        for minor in range(2):
            # minor == 0 is (f(x) + f(-x)) / 2, where there are only even-degree terms
            # minor == 1 is (f(x) - f(-x)) / (2x), where there are only odd-degree terms
            coeffs = []
            j = (len(real_part_coeffs)) % 2
            for coeff in real_part_coeffs:
                j = 1 - j
                if minor == j:
                    coeffs.append(coeff)
                else:
                    coeffs.append(0)
            if minor == 1:
                coeffs.pop()
            poly = sp.Poly(coeffs, x)
            roots = sp.polys.roots(poly)
            _, complex_roots_extra = _classify_roots(roots)
            M = _construct_matrix_from_roots([], 
                complex_roots_extra + complex_roots,
                leading_coeff = leading_coeff * poly.LC(),
                positive = False
            )
            Ms.append(M)
        return Ms


def _rationalize_matrix_simultaneously(M, lcm = 144, off_diagonal = True):
    """
    Rationalize entries of a symmetric matrix simultaneously.
    Off-diagonal entries are entries with strip > 1, i.e. not Toeplitz.

    Parameters
    ----------
    M : sp.Matrix | List[sp.Matrix]
        The symmetric matrix. If positive == True, it is a tuple of two matrices.
    lcm : int
        The least common multiple of denominators. The default is 144.
    off_diagonal : bool
        Whether only rationalize off-diagonal entries. The default is True.
        This would be slightly faster.

    Returns
    -------
    M : sp.Matrix
        The rationalized matrix.
    """
    if isinstance(M, sp.Matrix):
        n = M.shape[0]
        M = M.copy()
        for i in range(n):
            for j in range((i + 2) if off_diagonal else i, n):
                M[i,j] = round(M[i,j] * lcm) / lcm
                M[j,i] = M[i,j]
    else:
        Ms = M
        new_Ms = []
        for i, M in enumerate(Ms):
            new_Ms.append(
                _rationalize_matrix_simultaneously(M, lcm, off_diagonal = i == 0)
            )
        M = new_Ms
            
    return M


def _determine_diagonal_entries(M, poly):
    """
    Determine the Toepiltz diagonal entries of a symmetric matrix
    from the coefficient of polynomial.

    Example: x^4 + x^3 + 5x^2 + 1 <=> [[x, y, 1], [y, z, w], [1, w, u]],
    then we can solve that x = 1, y = 1/2, z = 3, w = 0, u = 1.

    Parameters
    ----------
    M : sp.Matrix | List[sp.Matrix]
        The symmetric matrix. If positive == True, it is a tuple of two matrices.
    poly : sp.Poly
        The polynomial.
    
    Returns
    -------
    M : sp.Matrix
        The matrix with diagonal entries determined. The operation is 
        done in-place.
    """
    if isinstance(M, sp.Matrix):
        n = M.shape[0]
        if isinstance(poly, sp.Poly):
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
    else:
        # positive == True
        Ms = M
        # first subtract the minor matrix from poly
        new_Ms = [None, Ms[1]]

        M = M[1]
        n = M.shape[0]
        all_coeffs = [0] * (M.shape[1] * 2 - 1)
        for i in range(len(all_coeffs)):
            m = i // 2
            if i % 2 == 0:
                l = min(m, n - m - 1)
                s = sum(M[m - j, m + j] for j in range(1, l + 1)) * 2
                s += M[m, m]
                all_coeffs[i] = s
            else:
                l = min(m, n - m - 2)
                s = sum(M[m - j, m + j + 1] for j in range(1, l + 1))
                s += M[m, m + 1]
                all_coeffs[i] = s * 2
        all_coeffs.append(0)
        poly -= sp.Poly(all_coeffs, poly.gens[0])

        # then determine the diagonal entries of the major
        M = _determine_diagonal_entries(Ms[0], poly)
        new_Ms[0] = M
        M = new_Ms

    return M


def _create_sos_from_US(U, S, x = None, return_raw = False):
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
    return_raw : bool, optional
        Whether return the raw lists.

    Returns
    -------
    Optional
        sos: sp.Expr
            When return_raw == False, return the sum of squares in sympy expression.
        S, polys: list, list[sp.polys.Poly]
            When return_raw == True, return the coeffcients and the polynomials
            such that sos = sum(S[i] * polys[i]**2).
    """
    if x is None:
        x = sp.symbols('x')
    polys = [sp.polys.Poly.from_list(U[i,:].tolist()[0], x) for i in range(U.shape[0])]
    if return_raw:
        return S, polys

    exprs = [Si * poly.as_expr()**2 for Si, poly in zip(S, polys)]
    return sp.Add(*exprs)


def _prove_univariate_irreducible(poly, return_raw = False):
    """
    Prove an irreducible univariate polynomial is positive over the real line or 
    positive over R+. Return None if the algorithm fails.

    Although it seems an SDP problem and requires an SDP solver, the univariate polynomial
    is degenerated. As guaranteed by Hilbert, a univariate polynomial positive over R
    is a sum of squares. This is quite easy to prove: if the polynomial has no real roots,
    then it is product of conjugate pairs of complex roots. Write f(x) = \prod (x - a)(x - conj(a)).
    As a consequence, f(x) = |\prod (x-a)|^2 = Re^2 + Im^2.

    This is a numerical solution. But we can reconstruct the positive definite matrix
    to make it rational and accurate.

    For polynomials positive over R+, we can use the same trick. Assume f(x) = g(x)h(x)
    where all roots of g(x) are negative and all roots of h(x) are complex. Then we see
    every coefficient of g is positive. So g(x) = g1(x) + x * g2(x) where
    g1(x) = (g(x) + g(-x))/2 is the even-order-term part while g2(x) = (g(x) - g(-x))/(2x) is the
    odd-order-term part. Both g1 and g2 are positive over R.

    Parameters
    ----------
    poly : sp.Poly
        The polynomial.
    positive : bool
        Whether to prove the polynomial is positive over R+ or positive over R. However,
        it will automatically turn off R+ constraint if no real roots are found.
    return_raw : bool, optional
        Whether return the raw lists.

    Returns
    -------
    Optional
        sos : sp.Expr
            A sum of squares expression.
        list[(multiplier, coeffs, polys)] : list
            A sum of squares expression such that sos = 
                sum(multiplier[i] * sum(coeffs[i][j] * polys[i][j]))
    """
    if poly.LC() < 0:
        return None
    if poly.degree() <= 2:
        return _prove_univariate_simple(poly, return_raw = return_raw)

    roots = sp.polys.nroots(poly)
    real_roots, complex_roots = _classify_roots(roots)

    # a mark of whether the polynomial is positive only over R+ (True) or over R (False)
    positive = True

    # we require all real roots to be negative for both cases positive == True / False
    if any(_ > 0 for _ in real_roots):
        return None

    if len(real_roots) == 0:
        # turn off R+ constraint if no real roots found
        positive = False

    if positive and all(_ >= 0 for _ in poly.all_coeffs()):
        # very trivial case if all coefficients are positive
        return poly.as_expr()

    # extract roots to construct matrix
    M0 = _construct_matrix_from_roots(real_roots, complex_roots, leading_coeff = poly.LC(), positive = positive)

    for lcm in (1, 144, 144**2, 144**6):
        # rationalize off-diagonal entries
        # and restore diagonal entries by subtraction
        M = _rationalize_matrix_simultaneously(M0, lcm = lcm)
        M = _determine_diagonal_entries(M, poly)

        # verify that the matrix is positive semidefinite
        if not positive:
            res = congruence(M)
            if res is not None and all(_ >= 0 for _ in res[1]):
                if return_raw:
                    return [(sp.S(1),) + _create_sos_from_US(*res, poly.gens[0], return_raw = True)]
                else:
                    return _create_sos_from_US(*res, poly.gens[0], return_raw = False)
        else:
            res2 = congruence(M[1])
            if res2 is not None and all(_ >= 0 for _ in res2[1]):
                res = congruence(M[0])
                if res is not None and all(_ >= 0 for _ in res[1]):
                    p1 = _create_sos_from_US(*res, poly.gens[0], return_raw = return_raw)
                    p2 = _create_sos_from_US(*res2, poly.gens[0], return_raw = return_raw)
                    if return_raw:
                        return [(sp.S(1),) + p1, (poly.gens[0],) + p2]
                    else:
                        return p1 + p2 * poly.gens[0]

    return


def _prove_univariate_simple(poly, return_raw = False):
    """
    Prove a polynomial with degree <= 2 is positive over the real line or positive over R+.

    Parameters
    ----------
    poly : sp.Poly
        The polynomial.
    return_raw : bool, optional
        Whether return the raw lists.

    Returns
    -------
    Optional
        sos : sp.Expr
            A sum of squares expression.
        list[(multiplier, coeffs, polys)] : list
            A sum of squares expression such that sos =
                sum(multiplier[i] * sum(coeffs[i][j] * polys[i][j]))
    """
    const_poly = lambda: sp.S(1).as_poly(poly.gens[0])
    if poly.degree() == 0:
        # constant polynomial
        if poly.LC() >= 0:
            if return_raw:
                return [(sp.S(1), [poly.LC()], [const_poly()])]
            return poly.LC()
        return None
    elif poly.degree() == 1:
        # linear polynomial
        if all(_ >= 0 for _ in poly.all_coeffs()):
            if return_raw:
                return [(sp.S(1), [poly.all_coeffs()[1]], [const_poly()]),
                        (poly.gens[0], [poly.all_coeffs()[0]], [const_poly()])]
            return poly.as_expr()
        return None
    else:
        # quadratic polynomial
        # first check if positive over R
        a, b, c = poly.all_coeffs()
        x = poly.gens[0]
        if a < 0:
            return
        if a > 0 and b**2 - 4 * a * c < 0:
            if return_raw:
                return [(sp.S(1), [a, c - b**2 / (4*a)], [(x +  b / (2*a)).as_poly(x), const_poly()])]
            return a * (x + b / (2 * a))**2 + (c - b**2 / (4 * a))

        # then check if positive over R+
        if a > 0 and b >= 0 and c >= 0:
            # since the minimum of the quadratic polynomial is negative
            # b must be negative so that the symmetric axis is at the left of y-axis
            if return_raw:
                return [(sp.S(1), [a, c], [x.as_poly(x), const_poly()]),
                        (x, [b], [const_poly()])]
            return a * x**2 + b * x + c
        return None


def prove_univariate(poly, return_raw = False):
    """
    Prove a polynomial is positive over the real line or positive over R+.

    Parameters
    ----------
    poly : sp.Poly
        The polynomial.
    return_raw : bool, optional
        Whether return the raw lists.
    
    Returns
    -------
    Optional
        sos : sp.Expr
            A sum of squares expression.
        list[(multiplier, coeffs, polys)] : list
            A sum of squares expression such that sos =
                sum(multiplier[i] * sum(coeffs[i][j] * polys[i][j]))
    """
    if not isinstance(poly, sp.Poly):
        if isinstance(poly, sp.Expr) and len(poly.free_symbols) == 1:
            poly = poly.as_poly(list(poly.free_symbols)[0])

    lc, factors = poly.factor_list()
    if lc < 0:
        return None
    mul = [(sp.S(1), [lc], [sp.S(1).as_poly(poly.gens[0])])] if return_raw else [lc]
    for factor, multiplicity in factors:
        if multiplicity % 2 == 0:
            if return_raw:
                mul.append([(sp.S(1), [sp.S(1)], [factor ** (multiplicity // 2)])])
            else:
                mul.append(factor.as_expr() ** multiplicity)
        else:
            ret = _prove_univariate_irreducible(factor, return_raw = return_raw)
            if ret is None:
                return None
            if return_raw:
                if multiplicity > 1:
                    mul.append([(sp.S(1), [sp.S(1)], [factor ** (multiplicity // 2)])])
                mul.append(ret)
            else:
                mul.append(ret ** multiplicity)

    if not return_raw:
        return sp.Mul(*mul)
    
    # merge the raw lists
    x = poly.gens[0]
    p1 = [mul[0][1], mul[0][2]]
    p2 = [[], []]

    def _cross_mul(p1, p2):
        coeffs1, polys1 = p1
        coeffs2, polys2 = p2
        ret = [[], []]
        for i in range(len(coeffs1)):
            for j in range(len(coeffs2)):
                ret[0].append(coeffs1[i] * coeffs2[j])
                ret[1].append(polys1[i] * polys2[j])
        return ret

    for m in mul[1:]:
        p1new = [[], []]
        p2new = [[], []]
        for multiplier, coeffs, polys in m:
            if multiplier == 1:
                padd = _cross_mul(p1, (coeffs, polys))
                p1new[0].extend(padd[0])
                p1new[1].extend(padd[1])
                padd = _cross_mul(p2, (coeffs, polys))
                p2new[0].extend(padd[0])
                p2new[1].extend(padd[1])
            else:
                padd = _cross_mul(p1, (coeffs, polys))
                p2new[0].extend(padd[0])
                p2new[1].extend(padd[1])
                padd = _cross_mul(p2, (coeffs, polys))
                p1new[0].extend(padd[0])
                p1new[1].extend(padd[1])
        p1 = p1new
        p2 = p2new

    return [(sp.S(1), p1[0], p1[1]), (x, p2[0], p2[1])]



def check_univariate(poly, positive = True):
    """
    Check whether a univariate polynomial is positive
    over the real line or positive over R+.

    Parameters
    ----------
    poly : sp.Poly
        The polynomial.
    positive : bool
        Whether to prove the polynomial is positive over R+ or positive over R. 
        When True, it checks over R+. When False, it checks over R.
    """
    if poly.LC() < 0:
        return False
    for factor, multiplicity in poly.factor_list()[1]:
        if multiplicity % 2 == 1:
            if positive:
                if sp.polys.count_roots(factor, 0, None) != 0:
                    if factor.degree() == 1 and factor(0) == 0:
                        # special case when the polynomial is x itself
                        continue
                    return False
            else:
                if sp.polys.count_roots(factor, None, None) != 0:
                    return False
    return True
