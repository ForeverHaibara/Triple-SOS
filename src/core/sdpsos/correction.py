import sympy as sp

from ...utils.expression.solution import congruence

def _is_positive(M):
    U, S = congruence(M)
    return all(_ >= 0 for _ in S)

def rationalize_with_positive(y, S):
    0