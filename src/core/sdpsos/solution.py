from typing import Dict, Union, Tuple

import sympy as sp

from .utils import congruence_with_perturbation, is_numer_matrix
from ...utils import (
    deg, generate_expr, 
    poly_get_factor_form,
    CyclicSum, CyclicProduct, is_cyclic_expr,
    SolutionSimple
)


class SolutionSDP(SolutionSimple):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # @property
    # def is_equal(self):
    #     return True


# For example, if f(a,b,c) = sum(a*g(a,b,c)^2 + a*h(a,b,c)^2 + ...),
# then here multiplier == a for functions g and h.
_MULTIPLIERS = {
    (0, 0): 1,
    (0, 1): sp.Symbol('a') * sp.Symbol('b'),
    (1, 0): sp.Symbol('a'),
    (1, 1): CyclicProduct(sp.Symbol('a'))
}


def _as_cyc(multiplier, x):
    """
    Beautifully convert CyclicSum(multiplier * x).
    """
    a, b, c = sp.symbols('a b c')
    if x.is_Pow:
        # Example: (p(a-b))^2 == p((a-b)^2)
        if isinstance(x.base, CyclicProduct) and x.exp > 1:
            x = CyclicProduct(x.base.args[0] ** x.exp)
    elif x.is_Mul:
        args = list(x.args)
        flg = False
        for i in range(len(args)):
            if args[i].is_Pow and isinstance(args[i].base, CyclicProduct):
                args[i] = CyclicProduct(args[i].base.args[0] ** args[i].exp)
                flg = True
        if flg: # has been updated
            x = sp.Mul(*args)

    if is_cyclic_expr(x, (a,b,c)):
        if multiplier == 1 or multiplier == CyclicProduct(a):
            return 3 * multiplier * x
        return CyclicSum(multiplier) * x
    elif multiplier == CyclicProduct(a):
        return multiplier * CyclicSum(x)
    return CyclicSum(multiplier * x)


def _matrix_as_expr(
        M: Union[sp.Matrix, Tuple[sp.Matrix, sp.Matrix]],
        multiplier: sp.Expr, 
        cyc: bool = True,
        factor_result: bool = True,
        cancel: bool = True
    ) -> sp.Expr:
    """
    Helper function to rewrite a single semipositive definite matrix 
    to sum of squares.

    Parameters
    ----------
    M : sp.Matrix | Tuple[sp.Matrix, sp.Matrix]
        The matrix to be rewritten or the decomposition of the matrix.
        If given a tuple, it should be (U, S) where U is upper triangular
        and S is a diagonal vector so that M = U.T @ diag(S) @ U.
    multiplier : sp.Expr
        The multiplier of the expression. For example, if `M` represents 
        `(a^2-b^2)^2 + (a^2-2ab+c^2)^2` while `multiplier = ab`, 
        then the result should be `ab(a^2-b^2)^2 + ab(a^2-2ab+c^2)^2`.
    cyc : bool
        Whether add a cyclic sum to the expression. Defaults to True.
    factor_result : bool
        Whether factorize the result. Defaults to True.
    cancel : bool
        Whether cancel the denominator of the expression. Defaults to True.

    Returns
    -------
    sp.Expr
        The expression of matrix as sum of squares.
    """
    is_numer = is_numer_matrix(M if isinstance(M, sp.Matrix) else M[1])
    factor_result = factor_result and (not is_numer)

    if isinstance(M, sp.Matrix):
        U, S = congruence_with_perturbation(M, perturb = is_numer)
        degree = round((2*M.shape[0] + .25)**.5 - 1.5)
    elif not isinstance(M, sp.Matrix) and len(M) == 2:
        U, S = M
        degree = round((2*U.shape[1] + .25)**.5 - 1.5)
    else:
        raise ValueError('The input should be a matrix M or a tuple (U, S) such that M = U.T @ diag(S) @ U.')


    a, b, c = sp.symbols('a b c')
    monoms = generate_expr(degree, cyc = 0)[1]

    factorizer = (lambda x: poly_get_factor_form(x.as_poly(a,b,c), return_type = 'expr'))\
                        if factor_result else (lambda x: x)
    as_cyc = (lambda m, x: m * x) if not cyc else _as_cyc

    expr = sp.S(0)
    for i, s in enumerate(S):
        if s == 0:
            continue
        val = sp.S(0)
        for j in range(min(U.shape[1], len(monoms))):
            monom = monoms[j]
            val += U[i,j] * a**monom[0] * b**monom[1] * c**monom[2]

        if cancel:
            r, val = val.together().as_coeff_Mul()
        else:
            r = 1
        val = factorizer(val)
        expr += (s * r**2) * as_cyc(multiplier, val**2)

    return expr


def _complete_M(S: Dict[str, sp.Matrix], Q: Dict[str, sp.Matrix], M: Dict[str, sp.Matrix]) -> Dict[str, sp.Matrix]:
    """
    Complete the missing matrices in dictionary M by multiplying Q @ S @ Q.T.

    Parameters
    ----------
    S : Dict[(str, sp.Matrix)]
        The symmetric matrices.
    Q : Dict[(str, sp.Matrix)]
        The low-rank transformations to the subspaces.
    M : Dict[(str, sp.Matrix)]
        The symmetric matrices to be completed.

    Returns
    -------
    M : Dict[(str, sp.Matrix)]
        The completed symmetric matrices. Each M = Q @ S @ Q.T.
    """
    if M is None: M = {}
    for key, Q in Q.items():
        if Q is None:
            continue
        if key not in M:
            M[key] = Q @ S[key] @ Q.T
    return M


def _is_numer_solution(
        S: Dict[str, sp.Matrix] = {},
        Q: Dict[str, sp.Matrix] = {},
        M: Dict[str, sp.Matrix] = {},
        decompose_method: str = 'raw',
    ) -> bool:
    """
    Check whether a solution is numerical. When it is, the solution must be inaccurate.
    """
    # Ms = _complete_M(S, Q, M)
    if decompose_method == 'raw':
        return any(is_numer_matrix(x) for x in M.values() if x is not None)
    elif decompose_method == 'reduce':
        return any(is_numer_matrix(x) for x in S.values() if x is not None)


def create_solution_from_M(
        poly: sp.Expr,
        S: Dict[str, sp.Matrix] = {},
        Q: Dict[str, sp.Matrix] = {},
        M: Dict[str, sp.Matrix] = {},
        decompose_method: str = 'raw',
        cyc: bool = True,
        factor_result: bool = True,
        cancel: bool = True,
    ) -> SolutionSDP:
    """
    Create SDP solution from symmetric matrices.

    Parameters
    ----------
    poly : sp.Expr
        The polynomial to be solved.
    S : Dict[(str, sp.Matrix)]
        The low-rank symmetric matrices. M = Q @ S @ Q.T.
    Q : Dict[(str, sp.Matrix)]
        If using decompose_method == 'reduce'. It should be the low-rank transformations to the subspaces.
    M : Dict[(str, sp.Matrix)]
        M = Q @ S @ Q.T. If given, the computation will be skipped.
    decompose_method : str
        One of 'raw' or 'reduce'. The default is 'raw'. 'raw' first computes Q.T @ S @ Q and then
        performs congruence. While 'reduce' performs congruence on S first and then multiply Q.T.
        'reduce' is useful in numerical case, which helps remove improper components.
    cyc : bool
        Whether to convert the solution to a cyclic sum.
    factor_result : bool
        Whether to factorize the result. The default is True.
    cancel : bool, optional
        Whether to cancel the denominator. The default is True.

    Returns
    -------
    SolutionSDP
        The SDP solution.
    """
    M = _complete_M(S, Q, M)
    degree = deg(poly)
    expr = sp.S(0)
    is_numer = _is_numer_solution(S, Q, M, decompose_method)

    items = []
    for key in M.keys():
        if M[key] is None:
            continue
        try:
            if decompose_method == 'raw':
                U, v = congruence_with_perturbation(M[key], perturb = is_numer)
                items.append((key, (U, v)))
            elif decompose_method == 'reduce':
                # S = U.T @ diag(v) @ U => M = QU.T @ diag(v) @ UQ.T
                U, v = congruence_with_perturbation(S[key], perturb = is_numer)
                items.append((key, (U * Q[key].T, v)))
        except TypeError:
            # TypeError: cannot unpack non-iterable NoneType object
            raise ValueError(f'Matrix M["{key}"] is not positive semidefinite.')

    for key, (U, v) in items:
        minor = key == 'minor'

        # After it gets cyclic, it will be three times as large as the original one,
        # so we require it to be divided by 3 in advance.
        originally_need_cyc = ((degree % 2) ^ minor)
        if cyc and not originally_need_cyc:
            # It is not originally a cyclic sum, but we force it to.
            # So we need to divide it by 3.
            v = v / 3

        expr += _matrix_as_expr(
            M = (U, v),
            multiplier = _MULTIPLIERS[(degree % 2, minor)],
            cyc = cyc or originally_need_cyc,
            factor_result = factor_result,
            cancel = cancel
        )

    return SolutionSDP(
        problem = poly,
        numerator = expr,
        is_equal = not is_numer,
    )