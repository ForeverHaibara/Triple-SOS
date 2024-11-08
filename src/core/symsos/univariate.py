from typing import List, Tuple, Union, Optional
from itertools import product

import sympy as sp

from ...sdp import congruence


def check_univariate(poly: sp.Poly, positive: bool = True) -> bool:
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

    Returns
    ----------
    bool
        Whether the polynomial is positive.
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


def _standardize_poly(poly: Union[sp.Expr, sp.Poly]) -> sp.Poly:
    """
    Convert an expression to a polynomial with domain == QQ.

    Parameters
    ----------
    poly : sp.Expr | sp.Poly
        The polynomial.

    Returns
    ----------
    sp.Poly
        The polynomial on QQ.
    """
    if not isinstance(poly, sp.Poly):
        if isinstance(poly, sp.Expr) and len(poly.free_symbols) == 1:
            poly = poly.as_poly(list(poly.free_symbols)[0])
        else:
            raise ValueError(f'The input should be a univariate polynomial, but received {poly}.')

    if poly.domain is sp.ZZ:
        poly = poly.to_field()

    if poly.domain is not sp.QQ:
        raise ValueError(f'The input should be a univariate polynomial over QQ, but received {poly.domain}.')

    return poly


def _classify_roots(roots: List[sp.Float]) -> Tuple[List[sp.Float], List[Tuple[sp.Float, sp.Float]]]:
    """
    Given a list of sympy roots. Classify them into real roots and complex roots.
    Also, the complex roots are paired by conjugate.

    We assume there are no repeated roots.

    Parameters
    ----------
    roots : List[sp.Float]
        Roots of a polynomial.

    Returns
    -------
    real_roots : List[sp.Float]
        Real roots.
    complex_roots : List[(sp.Float, sp.Float)]
        Paired complex roots.
    """
    real_roots = [root for root in roots if root.is_real]

    # pair complex roots
    # since the poly is irreducible, we may assume there is no repeated root
    complex_roots = [root for root in roots if sp.im(root) > 0]
    complex_roots = [(root, root.conjugate()) for root in complex_roots]
    return real_roots, complex_roots


def _construct_matrix_from_roots(
        real_roots: List[sp.Float],
        complex_roots: List[Tuple[sp.Float, sp.Float]],
        leading_coeff: sp.Rational = 1,
        positive: bool = False,
        early_stop: int = -1
    ) -> Union[sp.Matrix, Tuple[sp.Matrix, sp.Matrix]]:
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
    real_roots: List[sp.Float]
        Real roots.
    complex_roots : List[(sp.Float, sp.Float)]
        Paired complex roots.
    leading_coeff: sp.Rational
        The leading coefficient of the polynomial.
    positive:
        Whether the polynomial is positive over R or R+.
    early_stop: int
        We compute the interior matrix of by summing up multiple matrices.
        However, the time complexity is O(2^n). If early_stop == True,
        we only compute the first `early_stop` matrices.
        When `early_stop == -1`, we use 2*n + 2 matrices.

    Returns
    -------
    If positive == False:
        M : sp.Matrix
            The symmetric matrix.
    If positive == True:
        M1, M2: Tuple[sp.Matrix, sp.Matrix]
            Two symmetric matrices.
    """
    x = sp.symbols('x')
    if not positive:
        M = sp.zeros(len(complex_roots) + 1)
        cnt = 0

        if early_stop == -1:
            early_stop = 2 * len(complex_roots) + 2

        for comb in product(range(2), repeat=len(complex_roots)):
            vec = sp.prod((x - root[i]) for i, root in zip(comb, complex_roots))
            vec = vec.as_poly(x).all_coeffs()

            vec_re = sp.Matrix(list(map(sp.re, vec)))
            vec_im = sp.Matrix(list(map(sp.im, vec)))
            
            M += vec_re * vec_re.T + vec_im * vec_im.T
            cnt += 1
            if cnt >= early_stop:
                break

        M = M / cnt
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
                positive = False,
                early_stop = early_stop
            )
            Ms.append(M)
        return Ms


def _rationalize_matrix_simultaneously(
        M: Union[sp.Matrix, List[sp.Matrix]],
        lcm: int = 144,
        off_diagonal: bool = True
    ) -> Union[sp.Matrix, List[sp.Matrix]]:
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
    M : sp.Matrix | List[sp.Matrix]
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


def _determine_diagonal_entries(M: Union[sp.Matrix, List[sp.Matrix]], poly: sp.Poly) -> Union[sp.Matrix, List[sp.Matrix]]:
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
    M : sp.Matrix | List[sp.Matrix]
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
        poly -= sp.Poly(all_coeffs, poly.gen)

        # then determine the diagonal entries of the major
        M = _determine_diagonal_entries(Ms[0], poly)
        new_Ms[0] = M
        M = new_Ms

    return M


def _create_sos_from_US(
        U: sp.Matrix,
        S: sp.Matrix,
        x: Optional[sp.Symbol] = None,
        return_raw: bool = False
    ) -> Union[sp.Expr, Tuple[List[sp.Float], List[sp.Poly]]]:
    """
    Create a sum of squares from a congruence decomposition.

    Parameters
    ----------
    U : sp.Matrix
        The upper triangular matrix.
    S : sp.Matrix
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
        S, polys: List, List[sp.Poly]
            When return_raw == True, return the coeffcients and the polynomials
            such that sos = sum(S[i] * polys[i]**2).
    """
    if x is None:
        x = sp.symbols('x')
    S = list(S)
    polys = [sp.polys.Poly.from_list(U[i,:].tolist()[0], x) for i in range(U.shape[0])]
    if return_raw:
        return S, polys

    exprs = [Si * poly.as_expr()**2 for Si, poly in zip(S, polys)]
    return sp.Add(*exprs)


def _reduce_sos(
        coeff_list: List[sp.Rational],
        poly_list: List[sp.Poly],
        x: Optional[sp.Symbol] = None
    ) -> Tuple[List[sp.Rational], List[sp.Poly]]:
    """
    Sometimes after multiplying two raw lists, we have redudant terms.
    E.g. f(x) = 3 * (x - 2)^2 + 2 * (x - 1)^2. We can make it to a
    simpler form a * (x - b)^2 + c by reconstructing the Gram matrix
    and then decompose it again.

    The original polynomial should equal to
    sum(c * p**2 for c, p in zip(coeff_list, poly_list)).

    Parameters
    ----------
    coeff_list : List[sp.Rational]
        The coefficient list.
    poly_list : List[sp.Poly]
        The polynomial list.
    x : sp.Symbol, optional
        The symbol of polynomial. The default is None.

    Returns
    ----------
    coeff_list : List[sp.Rational]
        The reduced coefficient list.
    poly_list : List[sp.Poly]
        The reduced polynomial list.
    """
    if not coeff_list:
        return [], []
    n = max(_.degree() for _ in poly_list)
    M = sp.zeros(n + 1, n + 1)
    for coeff, poly in zip(coeff_list, poly_list):
        d = poly.degree()
        if coeff == 0 or d < 0:
            continue
        poly_coeffs = poly.all_coeffs()[::-1]
        for i in range(d + 1):
            for j in range(d + 1):
                M[n-i, n-j] += coeff * poly_coeffs[i] * poly_coeffs[j]
    ret = _create_sos_from_US(*congruence(M), x = x, return_raw = True)

    def _remove_zeros(p):
        if not any(_ == 0 for _ in p[0]):
            return p
        coeffs, polys = p
        new_coeffs = []
        new_polys = []
        for coeff, poly in zip(coeffs, polys):
            if coeff != 0:
                new_coeffs.append(coeff)
                new_polys.append(poly)
        return new_coeffs, new_polys

    return _remove_zeros(ret)


def _prove_univariate_irreducible(
        poly: sp.Poly,
        return_raw: bool = False,
        early_stop: int = -1,
        n: int = 15
    ) -> Union[sp.Expr, List[Tuple[sp.Rational, List[sp.Rational], List[sp.Poly]]], None]:
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
    return_raw : bool, optional
        Whether return the raw lists.
    early_stop: int
        We compute the interior matrix of by summing up multiple matrices.
        However, the time complexity is O(2^n). If early_stop == True,
        we only compute the first `early_stop` matrices.
        When `early_stop == -1`, we use 2*n + 2 matrices.
    n: int
        Working precision. The default is 15.

    Returns
    ----------
    Optional
        sos : sp.Expr
            A sum of squares expression.
        List[(multiplier, coeffs, polys)] : list
            A sum of squares expression such that sos = 
                sum(multiplier[i] * sum(coeffs[i][j] * polys[i][j]**2))
    """
    if poly.LC() < 0:
        return None
    if poly.degree() <= 2:
        return _prove_univariate_simple(poly, return_raw = return_raw)

    roots = sp.polys.nroots(poly, n = n)
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
        if return_raw:
            x = poly.gen
            result = [(sp.S(1), [], []), (x, [], [])]
            for (monom,), coeff in poly.terms():
                if coeff != 0:
                    odd = monom % 2
                    result[odd][1].append(coeff)
                    result[odd][2].append((x**(monom//2)).as_poly(x))
            return result
        return poly.as_expr()

    # extract roots to construct matrix
    M0 = _construct_matrix_from_roots(real_roots, complex_roots, leading_coeff = poly.LC(), positive = positive, early_stop = early_stop)

    lcm = 1
    while True:
        # rationalize off-diagonal entries
        # and restore diagonal entries by subtraction
        M = _rationalize_matrix_simultaneously(M0, lcm = lcm)
        M = _determine_diagonal_entries(M, poly)

        # verify that the matrix is positive semidefinite
        if not positive:
            res = congruence(M)
            if res is not None and all(_ >= 0 for _ in res[1]):
                if return_raw:
                    return [(sp.S(1),) + _create_sos_from_US(*res, poly.gen, return_raw = True)]
                else:
                    return _create_sos_from_US(*res, poly.gen, return_raw = False)
        else:
            res2 = congruence(M[1])
            if res2 is not None and all(_ >= 0 for _ in res2[1]):
                res = congruence(M[0])
                if res is not None and all(_ >= 0 for _ in res[1]):
                    p1 = _create_sos_from_US(*res, poly.gen, return_raw = return_raw)
                    p2 = _create_sos_from_US(*res2, poly.gen, return_raw = return_raw)
                    if return_raw:
                        return [(sp.S(1),) + p1, (poly.gen,) + p2]
                    else:
                        return p1 + p2 * poly.gen
        lcm *= 144
        if len(str(lcm)) > n:
            break
    return


def _prove_univariate_simple(
        poly: sp.Poly,
        return_raw: bool = False
    ) -> Union[sp.Expr, List[Tuple[sp.Rational, List[sp.Rational], List[sp.Poly]]], None]:
    """
    Prove a polynomial with degree <= 2 is positive over the real line or positive over R+.

    Parameters
    ----------
    poly : sp.Poly
        The polynomial.
    return_raw : bool, optional
        Whether return the raw lists.

    Returns
    ----------
    Optional
        sos : sp.Expr
            A sum of squares expression.
        List[(multiplier, coeffs, polys)] : list
            A sum of squares expression such that sos =
                sum(multiplier[i] * sum(coeffs[i][j] * polys[i][j]**2))
    """
    const_poly = lambda: sp.S(1).as_poly(poly.gen)
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
                        (poly.gen, [poly.all_coeffs()[0]], [const_poly()])]
            return poly.as_expr()
        return None
    else:
        # quadratic polynomial
        # first check if positive over R
        a, b, c = poly.all_coeffs()
        x = poly.gen
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


def prove_univariate(
        poly: sp.Poly,
        return_raw: bool = False,
        early_stop: int = -1,
        n: int = 15
    ) -> Union[sp.Expr, List[Tuple[sp.Rational, List[sp.Rational], List[sp.Poly]]], None]:
    """
    Prove a polynomial is positive over the real line or positive over R+.
    It automatically detects whether it is positive on R or merely on R+.
    If it is not positive, return None.

    To check whether it is not positive over R, set return_raw == True and
    check whether there exists (multiplier, coeffs, polys) such that
    multiplier == x and len(coeffs) > 0. An alternative is to use
    the function `prove_univariate_interval` with interval = (-oo, oo).

    To prove a polynomial is positive on an interval, please use `prove_univariate_interval`.

    Parameters
    ----------
    poly: sp.Poly
        The polynomial.
    return_raw : bool, optional
        Whether return the raw lists.
    early_stop: int
        We compute the interior matrix of by summing up multiple matrices.
        However, the time complexity is O(2^n). If early_stop == True,
        we only compute the first `early_stop` matrices.
        When `early_stop == -1`, we use 2*n + 2 matrices.
    n: int
        Working precision. The default is 15.

    Returns
    ----------
    Optional
        sos: sp.Expr
            A sum of squares expression.

        List[(multiplier, coeffs, polys)]: List
            A sum of squares expression such that sos =
                sum(multiplier[i] * sum(coeffs[i][j] * polys[i][j]**2))
            It is guaranteed that the returned result is a length-2 list.
            The two corresponding multipliers are given by:
                result[0][0] == sp.S(1) and result[1][0] == x where x is the symbol of the polynomial.
    """
    poly = _standardize_poly(poly)

    lc, factors = poly.factor_list()
    if lc < 0:
        return None
    mul = [(sp.S(1), [lc], [sp.S(1).as_poly(poly.gen)])] if return_raw else [lc]
    for factor, multiplicity in factors:
        if multiplicity % 2 == 0:
            if return_raw:
                mul.append([(sp.S(1), [sp.S(1)], [factor ** (multiplicity // 2)])])
            else:
                mul.append(factor.as_expr() ** multiplicity)
        else:
            ret = _prove_univariate_irreducible(factor, return_raw = return_raw, early_stop = early_stop, n = n)
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
    x = poly.gen
    p1 = [mul[0][1], mul[0][2]]
    p2 = [[], []]

    def _cross_mul(p1, p2, mul = 1):
        coeffs1, polys1 = p1
        coeffs2, polys2 = p2
        ret = [[], []]
        for i in range(len(coeffs1)):
            for j in range(len(coeffs2)):
                ret[0].append(coeffs1[i] * coeffs2[j])
                ret[1].append(mul * polys1[i] * polys2[j])
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
                padd = _cross_mul(p2, (coeffs, polys), mul = x)
                p1new[0].extend(padd[0])
                p1new[1].extend(padd[1])
        p1 = _reduce_sos(*p1new, x = x)
        p2 = _reduce_sos(*p2new, x = x)

    return [(sp.S(1), p1[0], p1[1]), (x, p2[0], p2[1])]


def prove_univariate_interval(
        poly: sp.Poly,
        interval: Tuple[sp.Expr, sp.Expr],
        return_raw: bool = False,
        early_stop: bool = -1,
        n: int = 15
    ) -> Union[sp.Expr, List[Tuple[sp.Rational, List[sp.Rational], List[sp.Poly]]], None]:
    """
    Prove a polynomial is positive over an interval. The algorithm is to
    first find a fractional linear transformation to transform the interval [a,b]
    to [0, inf), then prove the polynomial is positive over R+. If it is not positive
    on the given interval, return None.

    The function is a generalization of `prove_univariate`.

    Parameters
    ----------
    poly : sp.Poly
        The polynomial.
    interval : Tuple[sp.Expr, sp.Expr]
        The interval [a, b]. It requires that a < b. The interval also supports
        infinite endpoints, e.g. (-oo, b) or (a, oo) or (-oo, oo).
        The infinity should be represented by sp.oo or -sp.oo using sympy.
    return_raw : bool, optional
        Whether return the raw lists.
    early_stop: int
        We compute the interior matrix of by summing up multiple matrices.
        However, the time complexity is O(2^n). If early_stop == True,
        we only compute the first `early_stop` matrices.
        When `early_stop == -1`, we use 2*n + 2 matrices.
    n: int
        Working precision. The default is 15.

    Returns
    ----------
    Optional
        sos : sp.Expr
            A sum of squares expression.
        List[(multiplier, coeffs, polys)] : List
            A sum of squares expression such that sos =
                sum(multiplier[i] * sum(coeffs[i][j] * polys[i][j]**2))
            The list is guaranteed to be a length-2 list.
            When the degree of the polynomial is even, the first multiplier is 1
            and the second multiplier is (x - a) * (b - x). When the degree of the
            polynomial is odd, the first multiplier is (b - x) and the second
            multiplier is (x - a).
    """
    poly = _standardize_poly(poly)

    a, b = interval
    a, b = sp.S(a), sp.S(b)
    x = poly.gen
    kwargs = {'return_raw': True, 'early_stop': early_stop, 'n': n}
    if (a is -sp.oo) or (b is sp.oo):
        if b is sp.oo:
            if a is -sp.oo:
                poly_y = poly
            else:
                poly_y = poly.shift(a)
        else:
            poly_y = poly.transform((b - x).as_poly(x), sp.S(1).as_poly(x))
        proof = prove_univariate(poly_y, **kwargs)
        if proof is None:
            return None

        if b is sp.oo:
            if a is -sp.oo:
                # check whether it is positive over R
                for multiplier, coeff_list, poly_list in proof:
                    if multiplier != sp.S(1) and coeff_list:
                        return None
                trans = lambda p: p
                trans_mul = lambda m: m
            else:
                trans = lambda p: p.to_field().shift(-a)
                trans_mul = lambda m: m if m == 1 else (x - a)
        else:
            trans = lambda p: p.transform((b - x).as_poly(x), sp.S(1).as_poly(x))
            trans_mul = lambda m: m if m == 1 else (b - x)

        new_proof = []
        for multiplier, coeff_list, poly_list in proof:
            poly_list2 = [trans(p) for p in poly_list]
            new_proof.append((trans_mul(multiplier), coeff_list, poly_list2))
        
    else:
        d = b - a
        if d <= 0:
            raise ValueError(f'The interval [a, b] must satisfy a < b, but received a = {a}, b = {b}.')

        # y = (x - a) / (b - x)
        poly_y = poly.transform((x*b + a).as_poly(x), (x + 1).as_poly(x))
        proof = prove_univariate(poly_y, **kwargs)
        if proof is None:
            return None
        # if not return_raw: # very ugly
        #     proof = proof.xreplace({x: (x - a)/(b - x)}).together()

        new_proof = []
        n = poly.degree()
        for multiplier, coeff_list, poly_list in proof:
            if multiplier == sp.S(1):
                multiplier2 = sp.S(1) if n % 2 == 0 else (b - x)
                mul_d = 0 if n % 2 == 0 else 1
            else:
                multiplier2 = (x - a) if n % 2 == 1 else (x - a) * (b - x)
                mul_d = 1 if n % 2 == 1 else 2
            poly_list2 = []
            for poly1 in poly_list:
                # transform it back
                poly2 = poly1.transform((x - a).as_poly(x), (b - x).as_poly(x))
                poly2 = poly2 * ((b - x)**((n - mul_d)//2 - poly1.degree())).as_poly(x)
                poly_list2.append(poly2)
            coeff_list2 = [_ / d**n for _ in coeff_list]
            new_proof.append((multiplier2, coeff_list2, poly_list2))

    for i in range(len(new_proof)):
        for j in range(len(new_proof[i][2])):
            if new_proof[i][2][j].LC() < 0:
                new_proof[i][2][j] = -new_proof[i][2][j]

    if not return_raw:
        new_proof = [_ * sum(c * p.as_expr()**2 for c, p in zip(coeff_list, poly_list)) \
                        for _, coeff_list, poly_list in new_proof]
        new_proof = sum(new_proof)

    return new_proof