from typing import Dict, Union, Tuple

import sympy as sp

from .utils import congruence_with_perturbation, is_numer_matrix
from ...utils.polytools import deg
from ...utils.basis_generator import generate_expr
from ...utils.expression.cyclic import CyclicSum, CyclicProduct
from ...utils.expression.solution import SolutionSimple

class SolutionSDP(SolutionSimple):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # @property
    # def is_equal(self):
    #     return True

def _matrix_as_expr(
        M: Union[sp.Matrix, Tuple[sp.Matrix, sp.Matrix]],
        multiplier: sp.Expr, 
        cyc: bool = True,
        degree: int = 0,
        cancel: bool = True
    ) -> sp.Expr:
    """
    Helper function to rewrite a single semipositive definite matrix 
    to sum of squares.

    Parameters
    ----------
    M : sp.Matrix or Tuple[sp.Matrix, sp.Matrix]
        The matrix to be rewritten or the decomposition of the matrix.
    multiplier : sp.Expr
        The multiplier of the expression. For example, if `M` represents 
        `(a^2-b^2)^2 + (a^2-2ab+c^2)^2` while `multiplier = ab`, 
        then the result should be `ab(a^2-b^2)^2 + ab(a^2-2ab+c^2)^2`.
    cyc : bool
        Whether add a cyclic sum to the expression. Defaults to True.
    degree: int
        The (halved) degree of the polynomial. Defaults to 0 for auto inference.
        It should be appointed if M is cut off.
    cancel : bool
        Whether cancel the denominator of the expression. Defaults to True.

    Returns
    -------
    sp.Expr
        The expression of matrix as sum of squares.
    """
    if not isinstance(M, sp.Matrix) and len(M) == 2:
        degree = degree or round((2*M[0].shape[0] + .25)**.5 - 1.5)
        U, S = M
    else:
        degree = degree or round((2*M.shape[0] + .25)**.5 - 1.5)
        U, S = congruence_with_perturbation(M, allow_numer = is_numer_matrix(M))

    a, b, c = sp.symbols('a b c')
    monoms = generate_expr(degree, cyc = 0)[1]

    if not cyc:
        as_cyc = lambda x: multiplier * x
    else:
        if multiplier == CyclicProduct(a):
            as_cyc = lambda x: multiplier * CyclicSum(x)
        else:
            as_cyc = lambda x: CyclicSum(multiplier * x)

    expr = sp.S(0)
    for i, s in enumerate(S):
        if s == 0:
            continue
        val = sp.S(0)
        for j in range(min(U.shape[1], len(monoms))):
            monom = monoms[j]
            val += U[i,j] * a**monom[0] * b**monom[1] * c**monom[2]

        if cancel:
            val = val.together().as_coeff_Mul()
            r, val = val[0], val[1]
        else:
            r = 1
        expr += (s * r**2) * as_cyc(val**2)

    return expr
    

def create_solution_from_M(
        poly: sp.Expr,
        M: Dict[str, Union[sp.Matrix, Tuple[sp.Matrix, sp.Matrix]]] = {},
        Q: Dict[str, sp.Matrix] = {},
        decompositions: Dict[str, Tuple[sp.Matrix, sp.Matrix]] = {},
        method: str = 'raw',
        cancel: bool = True,
        **kwargs
    ) -> SolutionSDP:
    """
    Create SDP solution from symmetric matrices.

    Parameters
    ----------
    poly : sp.Expr
        The polynomial to be solved.
    M : Dict[sp.Matrix] or Dict[(sp.Matrix, sp.Matrix)]
        If using method == 'raw'. It should be the symmetric matrices or their decompositions. 
        `Ms` should have keys 'major', 'minor' and 'multiplier'.
    Q : Dict[sp.Matrix]
        If using method == 'reduce'. It should be the low-rank transformations to the subspaces.
    decompositions : Dict[sp.Matrix] or Dict[(sp.Matrix, sp.Matrix)]
        If using method == 'reduce'. It should be the decompositions of the subspace matrices.
    method : str
        One of 'raw' or 'reduce'. The default is 'raw'. 'raw' first computes Q.T @ S @ Q and then
        performs congruence. While 'reduce' performs congruence on S first and then multiply Q.T.
        'reduce' is useful in numerical case, which helps remove improper components.
    cancel : bool, optional
        Whether to cancel the denominator. The default is True.

    Returns
    -------
    SolutionSDP
        The SDP solution.
    """
    Ms = M
    degree = deg(poly)

    a, b, c = sp.symbols('a b c')
    expr = sp.S(0)
    is_equal = True

    if method == 'raw':
        items = Ms.items()
    elif method == 'reduce':
        items = [(key, (M * Q.T, v)) for (key, Q), (key_, (M, v)) in zip(Q.items(), decompositions.items())]

    for key, M in items:
        if M is None:
            continue
        minor = key == 'minor'
        # after it gets cyclic, it will be three times
        # so we require it to be divided by 3 in advance
        if not ((degree % 2) ^ minor):
            if isinstance(M, sp.Matrix):
                M = M / 3
            else:
                M = (M[0], [_ / 3 for _ in M[1]] if isinstance(M[1], list) else M[1] / 3)

        # e.g. if f(a,b,c) = sum(a*g(a,b,c)^2 + a*h(a,b,c)^2 + ...)
        # then here multiplier = a
        multiplier = {
            (0, 0): 1,
            (0, 1): a * b,
            (1, 0): a,
            (1, 1): CyclicProduct(a)
        }[(degree % 2, minor)]

        expr += _matrix_as_expr(
            M,
            multiplier,
            cyc = True,
            degree = degree // 2 - minor,
            cancel = cancel
        )

        if is_equal and is_numer_matrix(M if isinstance(M, sp.Matrix) else sp.Matrix(M[1])):
            is_equal = False

    return SolutionSDP(
        problem = poly,
        numerator = expr,
        is_equal = is_equal
    )