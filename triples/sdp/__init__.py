# sdp.__init__.py
#
# This is a symbolic SDP library intended for independent use.

# from .arithmetic import (
#     solve_column_separated_linear, solve_undetermined_linear,
#     matmul, matmul_multiple, symmetric_bilinear, symmetric_bilinear_multiple
# )

from .dual import SDPProblem
from .primal import SDPPrimal

from .arithmetic import congruence, ArithmeticTimeout
from .backends import SDPError, SDPTimeoutError
from .wedderburn import character_table, decompose_representation

__all__ = [
    # 'solve_column_separated_linear', 'solve_undetermined_linear',
    # 'matmul', 'matmul_multiple', 'symmetric_bilinear', 'symmetric_bilinear_multiple',
    'SDPProblem', 'SDPPrimal',
    'congruence', 'ArithmeticTimeout',
    'SDPError', 'SDPTimeoutError',
    'character_table', 'decompose_representation',
]
