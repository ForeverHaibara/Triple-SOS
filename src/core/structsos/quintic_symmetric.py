import sympy as sp

from .utils import CyclicSum, CyclicProduct, _sum_y_exprs

def sos_struct_quintic_symmetric(poly, coeff, recurrsion):
    if not (coeff((4,1,0)) == coeff((1,4,0)) and coeff((3,2,0)) == coeff((2,3,0))):
        return None

    if coeff((5,0,0)) == 0:
        # this case is degenerated and shall be handled in the non-symmetric case
        # TODO: add support for this case to present pretty solution
        return None

    # first determine how much abcs(a2-ab) can we subtract from the poly
    